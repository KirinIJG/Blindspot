from ultralytics import YOLO

# Load the YOLOv8 model
best = YOLO("best.pt")

# Export the model to ONNX format
best.export(format="onnx", device = 0, imgsz=640, batch=1, half=True) # creates 'yolov8n.onnx'

trtexec --onnx=best.onnx --saveEngine=best.engine --fp16 --verbose=True
