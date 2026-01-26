import os
import cv2
import random
from ultralytics import YOLO
import numpy as np
import torch
import time
from collections import defaultdict


# Define class colors (BGR format for OpenCV)
# All objects of the same class will have the same color
CLASS_COLORS = [
    (0, 0, 50),       # person: bluelish
    (255, 165, 0),    # case: orange
    (75, 40, 0),      # case_top: yellow
    (192, 192, 192),  # battery: silver
    (140, 0, 140),    # screw: violet
    (0, 200, 0)       # tool: green
]

# Expected maximum number of objects per class
# Used for per-class track ID assignment (IDs will be 1 to max_count)
MAX_OBJECTS_PER_CLASS = {
    0: 2,   # person: up to 2 (may split when holding objects)
    1: 1,   # case: 1
    2: 1,   # case_top: 1
    3: 6,   # battery: 6
    4: 4,   # screw: 4
    5: 2,   # tool: up to 2 (may split when held by person)
}


class PerClassTrackIDManager:
    """
    Manages per-class track IDs to ensure IDs stay within expected range.
    Maps global BoT-SORT track IDs to per-class IDs (1, 2, 3, ..., max_count).
    """
    
    def __init__(self, max_objects_per_class):
        self.max_objects = max_objects_per_class
        # Maps (class_id, global_track_id) -> per_class_id
        self.global_to_local = {}
        # Maps class_id -> set of available local IDs
        self.available_ids = {
            class_id: set(range(1, max_count + 1))
            for class_id, max_count in max_objects_per_class.items()
        }
        # Maps class_id -> dict of global_track_id -> (local_id, last_seen_frame)
        self.active_tracks = defaultdict(dict)
        self.current_frame = 0
        # Number of frames before a track is considered stale
        self.stale_threshold = 30
    
    def update_frame(self):
        """Call this at the start of each frame to update frame counter and clean stale tracks."""
        self.current_frame += 1
        self._cleanup_stale_tracks()
    
    def _cleanup_stale_tracks(self):
        """Remove tracks that haven't been seen for a while and reclaim their IDs."""
        for class_id in list(self.active_tracks.keys()):
            stale_global_ids = []
            for global_id, (local_id, last_seen) in self.active_tracks[class_id].items():
                if self.current_frame - last_seen > self.stale_threshold:
                    stale_global_ids.append(global_id)
            
            for global_id in stale_global_ids:
                local_id, _ = self.active_tracks[class_id].pop(global_id)
                # Reclaim the local ID
                if class_id in self.available_ids:
                    self.available_ids[class_id].add(local_id)
                # Remove from global mapping
                key = (class_id, global_id)
                if key in self.global_to_local:
                    del self.global_to_local[key]
    
    def get_local_id(self, class_id, global_track_id):
        """
        Get or assign a per-class local ID for a global track ID.
        Returns a local ID in range [1, max_count] for the class.
        """
        if global_track_id is None:
            return None
        
        key = (class_id, global_track_id)
        
        # Check if we already have a mapping
        if key in self.global_to_local:
            local_id = self.global_to_local[key]
            # Update last seen frame
            self.active_tracks[class_id][global_track_id] = (local_id, self.current_frame)
            return local_id
        
        # Need to assign a new local ID
        max_count = self.max_objects.get(class_id, 10)
        
        # Initialize available IDs for this class if not exists
        if class_id not in self.available_ids:
            self.available_ids[class_id] = set(range(1, max_count + 1))
        
        if self.available_ids[class_id]:
            # Assign the lowest available ID
            local_id = min(self.available_ids[class_id])
            self.available_ids[class_id].remove(local_id)
        else:
            # All IDs are taken - find the oldest track and reassign its ID
            oldest_frame = self.current_frame
            oldest_global_id = None
            for gid, (lid, last_seen) in self.active_tracks[class_id].items():
                if last_seen < oldest_frame:
                    oldest_frame = last_seen
                    oldest_global_id = gid
            
            if oldest_global_id is not None:
                local_id, _ = self.active_tracks[class_id].pop(oldest_global_id)
                old_key = (class_id, oldest_global_id)
                if old_key in self.global_to_local:
                    del self.global_to_local[old_key]
            else:
                # Fallback: use modulo to wrap around
                local_id = (global_track_id % max_count) + 1
        
        # Store the mapping
        self.global_to_local[key] = local_id
        self.active_tracks[class_id][global_track_id] = (local_id, self.current_frame)
        
        return local_id
    
    def reset(self):
        """Reset all track mappings."""
        self.global_to_local.clear()
        self.available_ids = {
            class_id: set(range(1, max_count + 1))
            for class_id, max_count in self.max_objects.items()
        }
        self.active_tracks.clear()
        self.current_frame = 0


def get_class_color(class_id):
    """
    Get a consistent color for each class ID.
    All objects of the same class will have the same color.
    """
    if class_id is None or class_id < 0:
        return (128, 128, 128)  # Gray for unknown class
    
    if class_id < len(CLASS_COLORS):
        return CLASS_COLORS[class_id]
    else:
        # Generate a consistent color for unknown classes
        golden_ratio = 0.618033988749895
        hue = ((class_id * golden_ratio) % 1.0) * 180
        hsv_color = np.array([[[int(hue), 255, 255]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(map(int, bgr_color))


def get_mask_centroid(polygon_points):
    """Calculate the centroid of a polygon mask."""
    pts = np.asarray(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
    M = cv2.moments(pts)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx = polygon_points[0][0]
        cy = polygon_points[0][1]
    return (cx, cy)


def draw_rect_boxes_and_labels_with_tracking(image, boxes, classes, track_ids=None):
    """
    Draws rectangular bounding boxes, class labels, and track IDs on an image.
    """
    img_copy = image.copy()

    if boxes is not None and len(boxes) > 0:
        for i, box_data in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])
            class_id = int(box_data.cls[0])
            confidence = float(box_data.conf[0])
            
            # Get track ID if available
            track_id = track_ids[i] if track_ids is not None and i < len(track_ids) else None
            
            # Get color based on class ID (all objects of same class have same color)
            color = get_class_color(class_id)

            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            # Put label with track ID
            label = classes[class_id] if class_id < len(classes) else f"Unknown Class {class_id}"
            if track_id is not None:
                label_with_score = f"ID:{track_id} {label} {confidence:.2f}"
            else:
                label_with_score = f"{label} {confidence:.2f}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            text_size = cv2.getTextSize(label_with_score, font, font_scale, font_thickness)[0]

            # Place text above the bounding box
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10

            # Draw background for text
            cv2.rectangle(img_copy, (text_x, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, text_y + 5), color, -1)
            cv2.putText(img_copy, label_with_score, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return img_copy


def draw_masks_and_scores_with_tracking(image, masks_with_tracking, classes, track_history=None):
    """
    Draw segmentation masks, class labels, confidence scores, and track IDs with transparency.
    Also draws track trajectory lines if track_history is provided.
    
    Args:
        image: Input image
        masks_with_tracking: List of tuples (class_id, polygon_points, score, track_id)
        classes: List of class names
        track_history: Optional dict mapping track_id to list of center points for trajectory
    """
    img_copy = image.copy()
    overlay = img_copy.copy()
    alpha = 0.4  # Transparency factor

    for class_id, polygon_points, score, track_id in masks_with_tracking:
        if polygon_points is not None and len(polygon_points) > 0:
            # Get color based on class ID (all objects of same class have same color)
            color = get_class_color(class_id)

            # polygon_points can be numpy array (N,2) or list of tuples
            pts = np.asarray(polygon_points, dtype=np.int32).reshape((-1, 1, 2))

            # Fill the polygon on the overlay with semi-transparency
            cv2.fillPoly(overlay, [pts], color)

            # Draw polygon outline
            cv2.polylines(img_copy, [pts], True, color, 2)

            # Calculate centroid of the polygon for label placement
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = int(polygon_points[0][0])
                cy = int(polygon_points[0][1])

            # Put label with track ID
            label_text = classes[class_id] if class_id < len(classes) else f"Unknown Class {class_id}"
            if track_id is not None:
                label_text = f"ID:{track_id} {label_text} {score:.2f}"
            else:
                label_text = f"{label_text} {score:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.2
            font_thickness = 1
            text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]

            # Place text near the centroid
            text_x = cx - text_size[0] // 2
            text_y = cy - 10 if cy - 10 > text_size[1] else cy + text_size[1] + 10

            # Ensure text stays within image bounds
            text_x = max(0, min(text_x, image.shape[1] - text_size[0]))
            text_y = max(text_size[1], min(text_y, image.shape[0] - 5))

            # Draw background for text
            cv2.rectangle(img_copy, (text_x - 2, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, text_y + 5), color, -1)
            cv2.putText(img_copy, label_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            # Draw trajectory line if track_history is provided
            # track_history uses (class_id, local_id) as keys
            if track_history is not None and track_id is not None:
                track_key = (class_id, track_id)
                if track_key in track_history:
                    points = track_history[track_key]
                    if len(points) > 1:
                        # Draw the tracking trajectory
                        for j in range(1, len(points)):
                            pt1 = (int(points[j - 1][0]), int(points[j - 1][1]))
                            pt2 = (int(points[j][0]), int(points[j][1]))
                            # Fade effect: older points are more transparent
                            thickness = max(1, int(2 * j / len(points)))
                            cv2.line(img_copy, pt1, pt2, color, thickness)

    # Combine the original image with the overlay using the transparency factor
    img_with_masks = cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0)
    return img_with_masks


# Pick the best available device, preferring Apple MPS for macOS GPUs
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# Load the best trained model weights and move to device
model_path = os.path.join('runs', 'segment', 'yolov11n_seg_custom', 'weights', 'best.pt')
model = YOLO(model_path).to(DEVICE)

print(f"Loaded fine-tuned model from: {model_path} on {DEVICE}")


# Define CLASSES (ensure it's in scope)
CLASSES = ["person", "case", "case_top", "battery", "screw", "tool"]

# Path to the video file
video_path = os.path.join('testdata', 'rec7-89.mp4')

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video file: {video_path}")

frame_count = 0
fps = 0.0
prev_time = None

# Track history for trajectory visualization
# Maps (class_id, local_track_id) -> list of (cx, cy) center points
track_history = defaultdict(lambda: [])
MAX_TRACK_HISTORY = 50  # Maximum number of points to keep per track

# Per-class track ID manager - ensures IDs stay within expected range per class
# e.g., screws will be ID 1-4, batteries ID 1-6, person ID 1, etc.
track_id_manager = PerClassTrackIDManager(MAX_OBJECTS_PER_CLASS)

# Create a window for display
window_name = f"YOLOv11 BoT-SORT Tracking - {os.path.basename(video_path)}"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("Press 'q' to quit, 'p' to pause/resume, 'c' to clear track history, 'r' to reset track IDs")
print("Using BoT-SORT tracker for multi-object tracking")
print("Track IDs are per-class (e.g., screw 1-4, battery 1-6, person 1)")

paused = False

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video or could not read frame.")
            break

        current_time = time.perf_counter()
        if prev_time is not None:
            fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        # Run tracking with BoT-SORT
        # persist=True maintains track IDs across frames
        # tracker="botsort.yaml" specifies the BoT-SORT tracker
        results = model.track(
            frame,
            verbose=False,
            device=DEVICE,
            persist=True,
            tracker="botsort.yaml",  # Use BoT-SORT tracker
        )

        # Collect segmentation masks with tracking info
        raw_detections = []
        active_track_ids = set()
        
        for r in results:
            if r.masks is not None and len(r.masks.xy) > 0:
                # Get track IDs (may be None if tracking failed for some detections)
                track_ids = r.boxes.id
                if track_ids is not None:
                    track_ids = track_ids.cpu().numpy().astype(int)
                
                for j, poly_np in enumerate(r.masks.xy):
                    class_id = int(r.boxes.cls[j])
                    confidence_score = float(r.boxes.conf[j])
                    
                    # Skip person class (class_id = 0) - don't track persons
                    if class_id == 0:
                        continue
                    
                    # Get track ID for this detection
                    track_id = None
                    if track_ids is not None and j < len(track_ids):
                        track_id = int(track_ids[j])
                        active_track_ids.add(track_id)
                    
                    # poly_np is already a numpy array of shape (N, 2) in image coordinates
                    raw_detections.append((class_id, poly_np, confidence_score, track_id))
        
        # Update per-class track ID manager
        track_id_manager.update_frame()
        
        # Convert global track IDs to per-class local IDs
        # Also update track history with per-class keys
        predicted_masks_with_local_ids = []
        active_local_tracks = set()  # Set of (class_id, local_id) tuples
        
        for class_id, poly_np, conf, global_track_id in raw_detections:
            # Get per-class local ID (e.g., screw 1-4 instead of arbitrary BoT-SORT ID)
            local_id = track_id_manager.get_local_id(class_id, global_track_id)
            predicted_masks_with_local_ids.append((class_id, poly_np, conf, local_id))
            
            if local_id is not None:
                track_key = (class_id, local_id)
                active_local_tracks.add(track_key)
                # Calculate centroid and update track history
                cx, cy = get_mask_centroid(poly_np)
                track_history[track_key].append((cx, cy))
                # Keep only recent history
                if len(track_history[track_key]) > MAX_TRACK_HISTORY:
                    track_history[track_key] = track_history[track_key][-MAX_TRACK_HISTORY:]
        
        # Clean up old tracks that are no longer active
        stale_tracks = [key for key in track_history.keys() if key not in active_local_tracks]
        for key in stale_tracks:
            # Gradually fade out old tracks by removing oldest points
            if len(track_history[key]) > 0:
                track_history[key] = track_history[key][1:]
            if len(track_history[key]) == 0:
                del track_history[key]

        # Draw annotated frame with tracking (using per-class local IDs)
        img_display = draw_masks_and_scores_with_tracking(
            frame, predicted_masks_with_local_ids, CLASSES, track_history
        )
        
        # Add FPS and tracking info text
        cv2.putText(img_display, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        #cv2.putText(img_display, f"Detections: {len(raw_detections)}", (10, 60), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        ##cv2.putText(img_display, f"Active Tracks: {len(active_local_tracks)}", (10, 90), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        frame_count += 1
    
    # Display the frame
    cv2.imshow(window_name, img_display)
    
    # Wait for key press (1ms delay for smooth playback)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('p'):
        paused = not paused
        if not paused:
            prev_time = time.perf_counter()  # reset timer when resuming
        print("Paused" if paused else "Resumed")
    elif key == ord('c'):
        track_history.clear()
        print("Track history cleared")
    elif key == ord('r'):
        track_id_manager.reset()
        track_history.clear()
        print("Track IDs and history reset")

cap.release()
cv2.destroyAllWindows()

print(f"Processed {frame_count} frames from video '{os.path.basename(video_path)}'.")
print(f"Total unique tracks seen: {len(track_history)}")
