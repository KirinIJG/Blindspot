import numpy as np
import cv2
import collections
from config import PIXELS_PER_METER, FRAME_TIME, MIN_DISTANCE, FPS
from scipy.spatial.distance import cdist
import logging



# Speed tracking history
speed_history = collections.defaultdict(lambda: collections.deque(maxlen=5))

# Setup logger
logger = logging.getLogger("utils")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_optical_flow_cuda(prev_gray, gray_frame, points):
    """
    Calculate optical flow using CUDA-accelerated SparsePyrLK.
    """
    
    if prev_gray is None or prev_gray.size == 0:
        raise ValueError("Invalid input: prev_gray is None or empty.")
    if gray_frame is None or gray_frame.size == 0:
        raise ValueError("Invalid input: gray_frame is None or empty.")
    if points is None or len(points) == 0:
        raise ValueError("Invalid input: points array is None or empty.")

    try:
        # Create CUDA SparsePyrLK optical flow instance
        gpu_flow = cv2.cuda.SparsePyrLKOpticalFlow.create(winSize=(15, 15), maxLevel=3)

        # Upload frames to the GPU
        prev_gray_gpu = cv2.cuda_GpuMat()
        gray_frame_gpu = cv2.cuda_GpuMat()
        prev_gray_gpu.upload(prev_gray)
        gray_frame_gpu.upload(gray_frame)

        # Prepare points for tracking
        #print(f"Initial shapes: prev_gray={prev_gray.shape}, gray_frame={gray_frame.shape}, points={np.array(points).shape}")
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        #print(f"Reshaped points for optical flow: {points.shape}")
        points_gpu = cv2.cuda_GpuMat()
        points_gpu.upload(points)

        # Compute optical flow
        points_new_gpu, status_gpu, _ = gpu_flow.calc(prev_gray_gpu, gray_frame_gpu, points_gpu, None)

        # Download results to the CPU
        points_new = points_new_gpu.download()
        status = status_gpu.download()

        return points_new, status
    except Exception as e:
        logger.error(f"Optical flow calculation failed: {e}")
        return points, np.zeros_like(points, dtype=np.uint8)

def match_tracks(detections, object_positions, distance_threshold):
    # Extract detection centers from detections
    detection_centers = np.array([[d[0] + (d[2] - d[0]) / 2, d[1] + (d[3] - d[1]) / 2] for d in detections])
    #print(f"Detection centers shape: {detection_centers.shape}")
    
    # Extract previous centers from object positions, ensuring consistent shape
    prev_centers = []
    for pos in object_positions.values():
        if len(pos) > 0:
            last_position = pos[-1]
            if isinstance(last_position, (list, np.ndarray)) and len(last_position) == 2:
                prev_centers.append(last_position)
    
    # Convert to numpy array
    prev_centers = np.array(prev_centers, dtype=np.float32) if prev_centers else np.zeros((0, 2), dtype=np.float32)
    #print(f"Previous centers shape: {prev_centers.shape}")
    
    # If there are no previous centers, initialize tracks
    if prev_centers.shape[0] == 0:
        #print("No previous centers; initializing tracks.")
        return {i: f"object_{detections[i][-1]}" for i in range(len(detections))}
    
    # Compute pairwise distances between detection centers and previous centers
    distances = cdist(detection_centers, prev_centers)
    
    matches = {}
    for det_idx, _ in enumerate(detection_centers):
        closest_idx = np.argmin(distances[det_idx])
        if distances[det_idx, closest_idx] < distance_threshold:
            prev_track_id = list(object_positions.keys())[closest_idx]
            matches[det_idx] = prev_track_id
        else:
            matches[det_idx] = f"new_{det_idx}"

    return matches


def calculate_speed(pos1, pos2):
    """Calculate speed (m/s) based on pixel distance and frame time."""
    pixel_distance = np.linalg.norm(pos2 - pos1)
    if pixel_distance < MIN_DISTANCE:
        return 0.0
    distance_in_meters = pixel_distance / PIXELS_PER_METER
    return distance_in_meters / FRAME_TIME

def calculate_smoothed_speed(track_id, new_speed):
    """Calculate smoothed speed using a moving average."""
    speed_history[track_id].append(new_speed)
    return np.mean(speed_history[track_id])

def direction_from_displacement(dx, dy, camera_index):
    """
    Determine the motion direction based on displacement.
    """
    if camera_index == 0:  # for rear camera
        if abs(dy) > abs(dx):  # Vertical motion dominates
            return "Forward" if dy > 0 else "Backward"
        else:  # Horizontal motion dominates
            return "Rightward" if dx < 0 else "Leftward"

    elif camera_index == 1:  # for front camera
        if abs(dy) > abs(dx):  # Vertical motion dominates
            return "Forward" if dy < 0 else "Backward"
        else:  # Horizontal motion dominates
            return "Rightward" if dx > 0 else "Leftward"

    elif camera_index == 2:  # for left camera
        if abs(dy) > abs(dx):  # Vertical motion dominates
            return "Forward" if dy > 0 else "Backward"
        else:  # Horizontal motion dominates
            return "Rightward" if dx > 0 else "Leftward"

    elif camera_index == 3:  # for right camera
        if abs(dy) > abs(dx):  # Vertical motion dominates
            return "Forward" if dy > 0 else "Backward"
        else:  # Horizontal motion dominates
            return "Rightward" if dx > 0 else "Leftward"


def draw_predictions(frame, detections, gray_frame, prev_gray, object_positions, object_tracking_count, matches, last_speeds, frame_count, camera_index):
    """
    Draw bounding boxes, labels, speed estimates, and directions with stable track IDs.
    """
    #print(f"Frame count: {frame_count}")
    #print(f"Detections: {detections}")
    #print(f"Object positions before processing:")
    for key, pos in object_positions.items():
        print(f"Track ID: {key}, Positions: {pos}, Shape: {np.array(pos).shape if isinstance(pos, np.ndarray) else 'Not an array'}")
    for track_id, position in object_positions.items():
        print(f"Track {track_id}: {np.array(position).shape}")
    for det_idx, detection in enumerate(detections):
        x_min, y_min, x_max, y_max, confidence, class_id = map(int, detection)
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        current_position = np.array([center_x, center_y], dtype=np.float32)

        # Assign track ID from matches or create a new one
        track_id = matches.get(det_idx, f"object_{class_id}_{x_min}_{y_min}")

        try:
            # Optical flow tracking
            if prev_gray is not None and track_id in object_positions:
                prev_center = np.array(object_positions[track_id], dtype=np.float32).reshape(-1, 1, 2)
                points_new, status = calculate_optical_flow_cuda(prev_gray, gray_frame, prev_center)

                if status is not None and np.any(status == 1):
                    current_position = points_new[0].flatten()

                displacement = current_position - prev_center[0].flatten()
                dx, dy = displacement
                direction = direction_from_displacement(dx, dy, camera_index)
                cv2.putText(frame, f"Direction: {direction}", (center_x, center_y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            # Speed calculation
            if frame_count % (FPS // 3) == 0 or frame_count == 1:  # Every 1/3 second
                if track_id in object_positions:
                    speed_mps = calculate_speed(object_positions[track_id], current_position)
                    smoothed_speed_kmh = calculate_smoothed_speed(track_id, speed_mps * 3.6)
                    last_speeds[track_id] = smoothed_speed_kmh
                else:
                    last_speeds[track_id] = 0.0

            # Display speed
            if track_id in last_speeds:
                speed_text = f"Speed: {last_speeds[track_id]:.2f} km/h"
                cv2.putText(frame, speed_text, (center_x, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Update object positions
            object_positions[track_id] = current_position

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        except Exception as e:
            logger.error(f"Error drawing predictions for track ID {track_id}: {e}")

    return gray_frame
