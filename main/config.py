import os
import cv2

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Paths
VIDEO_URLS = [
    #r"../vids/esp_back_3.mp4",  # Camera 0
    #r"../vids/esp_back_3.mp4",  # Camera 1
    #r"../vids/esp_back_3.mp4",  # Camera 2
    #r"../vids/esp_back_3.mp4"  # Camera 3
    "http://192.168.0.101:81/stream", "http://192.168.0.104:81/stream"
    
]


# Frame and speed parameters"""  """
FPS = 30
FRAME_TIME = 1 / FPS
REAL_WORLD_WIDTH = 20  # (in meters) need to calibrate
PIXELS_PER_METER = 224 / REAL_WORLD_WIDTH # need to calibrate
MIN_DISTANCE = 0.1  # Minimum pixel distance to avoid noise
TRACKING_THRESHOLD = 3  # Minimum frames for tracking an object
