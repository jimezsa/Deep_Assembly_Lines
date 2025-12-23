import os
import cv2
import random
from ultralytics import YOLO
import numpy as np
import torch
import time

def draw_rect_boxes_and_labels(image, boxes, classes):
    """
    Draws rectangular bounding boxes and class labels on an image.
    """
    img_copy = image.copy()

    # Define a set of distinct colors for classes (BGR format for OpenCV)
    class_colors = [
      (0, 0, 70),   # person: bluelish
      (255, 165, 0), # case: orange
      (180, 100, 0),  # case_top:
      (0, 100, 100),  # battery: Grenn
      (128, 0, 128),   # screw: violet
      (0, 100, 0)  # tool: Green
    ]

    if boxes is not None and len(boxes) > 0:
        for box_data in boxes:
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])
            class_id = int(box_data.cls[0])
            confidence = float(box_data.conf[0]) # Extract confidence score

            # Get color for the class
            color = class_colors[class_id % len(class_colors)] if class_id < len(class_colors) else (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # Fallback random color

            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            # Put label
            label = classes[class_id] if class_id < len(classes) else f"Unknown Class {class_id}"
            label_with_score = f"{label} {confidence:.2f}" # Append confidence score
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

# Create a window for display
window_name = f"YOLOv11 Predictions - {os.path.basename(video_path)}"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("Press 'q' to quit, 'p' to pause/resume")

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

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = img_rgb.shape[:2]
       
        # Run inference
        results = model(img_rgb, verbose=False, device=DEVICE, imgsz=(frame_h, frame_w))

        # Collect all detected boxes for this frame
        all_detected_boxes = []
        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
                all_detected_boxes.extend(r.boxes)

        # Draw annotated frame
        img_with_predictions = draw_rect_boxes_and_labels(img_rgb.copy(), all_detected_boxes, CLASSES)
        
        # Convert back to BGR for cv2.imshow
        img_display = cv2.cvtColor(img_with_predictions, cv2.COLOR_RGB2BGR)
        
        # Add FPS text
        cv2.putText(img_display, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_count += 1
    
    # Display the frame
    cv2.imshow(window_name, img_display)
    
    # Wait for key press (25ms delay for ~40 fps, adjust as needed)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('p'):
        paused = not paused
        if not paused:
            prev_time = time.perf_counter()  # reset timer when resuming
        print("Paused" if paused else "Resumed")

cap.release()
cv2.destroyAllWindows()

print(f"Processed {frame_count} frames from video '{os.path.basename(video_path)}'.")
