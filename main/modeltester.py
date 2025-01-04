import cv2
import numpy as np
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA


class TensorRTYOLOv8:
    def __init__(self, engine_path):
        """Initialize the TensorRT YOLOv8 engine."""
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load TensorRT engine
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate buffers for input and output
        self.allocate_buffers()

    def allocate_buffers(self):
        """Allocate device and host buffers for input and output."""
        self.bindings = [None] * self.engine.num_io_tensors

        for idx in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(idx)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            tensor_shape = self.context.get_tensor_shape(tensor_name)
            tensor_dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

            size = trt.volume(tensor_shape) * np.dtype(tensor_dtype).itemsize
            device_mem = cuda.mem_alloc(size)
            host_mem = np.empty(size // np.dtype(tensor_dtype).itemsize, dtype=tensor_dtype)

            if tensor_mode == trt.TensorIOMode.INPUT:
                self.input_name = tensor_name
                self.input_shape = tensor_shape
                self.input_dtype = tensor_dtype
                self.input_host = host_mem
                self.input_device = device_mem
                self.bindings[idx] = self.input_device
            elif tensor_mode == trt.TensorIOMode.OUTPUT:
                self.output_name = tensor_name
                self.output_shape = tensor_shape
                self.output_dtype = tensor_dtype
                self.output_host = host_mem
                self.output_device = device_mem
                self.bindings[idx] = self.output_device

    def preprocess(self, frame):
        """Preprocess the input frame for YOLOv8."""
        height, width = self.input_shape[-2:]
        resized = cv2.resize(frame, (width, height))
        normalized = resized / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        return np.ascontiguousarray(batched, dtype=self.input_dtype)

    def infer(self, frame):
        """Run inference on a single frame."""
        preprocessed = self.preprocess(frame)
        cuda.memcpy_htod(self.input_device, preprocessed)
        self.context.execute_v2(self.bindings)
        cuda.memcpy_dtoh(self.output_host, self.output_device)
        return self.output_host

    def postprocess(self, output, input_shape, conf_threshold=0.5):
        """Postprocess the YOLO output to filter detections."""
        num_classes = 80  # Update based on your model (e.g., COCO has 80 classes)
        attributes_per_box = 5 + num_classes  # Box attributes + classes
        num_boxes = output.size // attributes_per_box  # Calculate number of boxes

        # Ensure the output size matches expectations
        if output.size % attributes_per_box != 0:
            raise ValueError(f"Unexpected output size: {output.size}. Cannot reshape into {num_boxes} boxes.")

        # Reshape the flat output array
        output = output.reshape(num_boxes, attributes_per_box)

        # Extract relevant data
        x_center = output[:, 0]
        y_center = output[:, 1]
        width = output[:, 2]
        height = output[:, 3]
        confidence = 1 / (1 + np.exp(-output[:, 4]))  # Apply sigmoid to confidence
        class_probs = np.exp(output[:, 5:]) / np.sum(np.exp(output[:, 5:]), axis=1, keepdims=True)  # Softmax

        # Calculate scores and filter by confidence
        scores = confidence * np.max(class_probs, axis=1)
        valid_indices = scores > conf_threshold

        # Keep only valid detections
        x_center, y_center, width, height, scores = (
            x_center[valid_indices], y_center[valid_indices],
            width[valid_indices], height[valid_indices],
            scores[valid_indices]
        )
        classes = np.argmax(class_probs[valid_indices], axis=1)

        # Convert [x_center, y_center, w, h] to [x_min, y_min, x_max, y_max]
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        # Combine detections into a final array
        return np.stack([x_min, y_min, x_max, y_max, scores, classes], axis=1)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="TensorRT YOLOv8 Object Detection")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLOv8 TensorRT engine file.")
    parser.add_argument("--source", type=str, required=True, help="Path to the video file or '0' for webcam.")
    parser.add_argument("--show", action="store_true", help="Display the video with detections.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detections.")
    args = parser.parse_args()

    # Initialize YOLO model
    yolo_model = TensorRTYOLOv8(args.model)

    # Open video source (file or webcam)
    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {args.source}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference and postprocess
        raw_output = yolo_model.infer(frame)
        detections = yolo_model.postprocess(raw_output, frame.shape[:2], conf_threshold=args.conf)

        # Draw detections
        for detection in detections:
            x_min, y_min, x_max, y_max, score, class_id = detection
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            label = f"Class {int(class_id)}: {score:.2f}"
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show video
        if args.show:
            cv2.imshow("YOLOv8 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
