import cv2
import time
import logging
from model1 import load_model
from utils import draw_predictions, match_tracks
import pycuda.driver as cuda
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

def setup_logger(name, log_file, level=logging.INFO):
    """
    Creates a custom logger with a specified name and log file.
    """
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def camera_process(camera_index, camera_url):
    """
    Process a single camera stream. This function will run in a separate process.
    """
    print(f"Starting process for Camera {camera_index + 1}...")

    # Create a unique logger for each camera
    log_file = f"logs_camera_{camera_index + 1}.log"
    logger = setup_logger(f"Camera{camera_index + 1}", log_file)

    # Initialize CUDA for this process
    cuda.init()
    device = cuda.Device(0)  # Use GPU 0
    context = device.make_context()

    try:
        # Load the YOLO model
        logger.info(f"Camera {camera_index + 1}: Loading YOLO model...")
        model = load_model()
        if model is None:
            raise RuntimeError("Failed to load YOLO model.")

        logger.info(f"Camera {camera_index + 1}: YOLO model loaded successfully.")

        # Open video capture for this camera
        cap = cv2.VideoCapture(camera_url)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index + 1}")

        logger.info(f"Camera {camera_index + 1}: Video stream opened successfully.")

        # Initialize variables
        prev_gray = None
        object_positions = {}
        object_tracking_count = {}
        last_speeds = {}
        frame_count = 0
        skip_frames = 2
        total_processing_time = 0

        # Camera matrix (K) and distortion coefficients (D) for fisheye camera (Camera 1)
        K = np.array([[138.04439896, 2.48299413, 206.58578153],  # f_x, skew, c_x
                [0, 138.58339369, 140.33184146],  # skew, f_y, c_y
                [0, 0, 1]])       # (0, 0, 1)

        D = np.array([0.315585252, 0.88019049, -1.51121647, 0.71726119])  # [k1, k2, p1, p2] #calluibrate

        # Precompute undistortion maps for fisheye camera (Camera 1)
        DIM = (400, 296)  # ESP32-CAM resolution
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K * 2, DIM, cv2.CV_16SC2)


        while True:
            start_time = time.perf_counter()
            ret, frame = cap.read()
            
            # Apply fisheye correction
            if camera_index in (0,1,2,3):
                frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            if not ret:
                logger.error(f"Cannot read frame from Camera {camera_index + 1}")
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # Convert to grayscale for optical flow
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            try:
                context.push()  # Push CUDA context for inference

                # Perform inference
                results = model.predict(frame, conf=0.45)
                detections = []

                # Extract detections from results
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        detections.append([x_min, y_min, x_max, y_max, confidence, class_id])

                # Match tracks for detections
                matches = match_tracks(detections, object_positions, 50)

                # Draw predictions, calculate speeds, and update tracking
                prev_gray = draw_predictions(
                    frame, detections, gray_frame, prev_gray,
                    object_positions, object_tracking_count,
                    matches, last_speeds, frame_count, camera_index
                )
		
                # Display the processed frame
                cv2.imshow(f"Camera {camera_index + 1}", frame)

            except Exception as e:
                logger.error(f"Error during frame processing: {e}", exc_info=True)

            finally:
                context.pop()  # Pop CUDA context after inference

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            total_processing_time += elapsed_time

            # Log the time taken for this frame
            logger.info(f"Frame {frame_count}: Processing time = {elapsed_time:.4f} seconds")

            # Press 'q' to quit the process
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"Stopping process for Camera {camera_index + 1}...")
                break

    except Exception as e:
        logger.error(f"Unexpected error in Camera {camera_index + 1}: {e}", exc_info=True)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        context.pop()
        context.detach()
        logger.info(f"Camera {camera_index + 1}: Total frames processed: {frame_count}")
        if frame_count > 0:
            avg_time = total_processing_time / frame_count
            logger.info(f"Camera {camera_index + 1}: Avg processing time: {avg_time:.3f} s/frame.")
