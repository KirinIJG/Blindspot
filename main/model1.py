from ultralytics import YOLO

def load_model():
    """Load the YOLO model."""
    model_path = "yolov8n.engine"  
    try:
        model = YOLO(model_path)
        print("YOLO model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None
