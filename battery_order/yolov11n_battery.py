import os
import cv2
import random
from ultralytics import YOLO
import numpy as np
import torch
import time
import pickle

# =========================
# CONFIG
# =========================
CASE_ID = 1
BATTERY_ID = 3
KEEP_CLASS_IDS = [CASE_ID, BATTERY_ID]

SAVE_CACHE = True          # turn OFF if you only want visualization
DRAW_VIS = True            # turn OFF for fastest run

# =========================
# DEVICE
# =========================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# =========================
# MODEL
# =========================
model_path = os.path.join(
    r'..\yolov11_finetuned',
    'runs', 'segment', 'yolov11n_seg_custom',
    'weights', 'best.pt'
)
model = YOLO(model_path).to(DEVICE)
print(f"Loaded model from: {model_path}")

CLASSES = ["person", "case", "case_top", "battery", "screw", "tool"]

# =========================
# VIDEO
# =========================
# video_path = os.path.join(r'..\yolov11_finetuned\testdata', 'rec7-89.mp4') 
# video_path = os.path.join('data', 'wrong_top.mp4') 
video_path = os.path.join('complete_data', 'wrong5' ,'135122071615', '135122071615.mp4') # 137322071489
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

# =========================
# OUTPUT CACHE
# =========================
all_frames_data = []

# =========================
# VISUALIZATION
# =========================
def draw_masks(image, items):
    overlay = image.copy()
    alpha = 0.4

    for item in items:
        poly = item["polygon"]
        color = (255, 165, 0) if item["class_id"] == CASE_ID else (192, 192, 192)
        pts = poly.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(image, [pts], True, color, 1)

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# =========================
# MAIN LOOP
# =========================
frame_idx = 0
prev_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, device=DEVICE)

    frame_record = {
        "frame_idx": frame_idx,
        "case": None,
        "batteries": []
    }

    vis_items = []

    for r in results:
        if r.masks is None:
            continue

        for j, poly in enumerate(r.masks.xy):
            class_id = int(r.boxes.cls[j])

            if class_id not in KEEP_CLASS_IDS:
                continue  # 🔥 FILTER HERE

            poly = np.asarray(poly, dtype=np.float32)

            if class_id == CASE_ID:
                x1, y1, x2, y2 = map(int, r.boxes.xyxy[j])
                frame_record["case"] = {
                    "bbox": [x1, y1, x2, y2],
                    "polygon": poly
                }

            elif class_id == BATTERY_ID:
                cx = float(np.mean(poly[:, 0]))
                cy = float(np.mean(poly[:, 1]))
                frame_record["batteries"].append({
                    "polygon": poly,
                    "centroid": (cx, cy)
                })

            vis_items.append({
                "class_id": class_id,
                "polygon": poly
            })

    all_frames_data.append(frame_record)

    # =========================
    # VISUALIZATION
    # =========================
    if DRAW_VIS:
        frame_vis = draw_masks(frame.copy(), vis_items)

        current_time = time.perf_counter()
        if prev_time is not None:
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            cv2.putText(frame_vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        prev_time = current_time

        cv2.imshow("YOLO filtered", frame_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# =========================
# SAVE CACHE
# =========================
if SAVE_CACHE:
    # data_path = os.path.join('\testdata', 'rec7-89.mp4')
    # data_path = os.path.join('data', 'wrong_top.mp4')
    data_path = os.path.join('complete_data', 'wrong5' ,'135122071615', '135122071615.mp4') # 137322071489
    out_path = os.path.splitext(os.path.basename(data_path))[0] + "_yolo_cache.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_frames_data, f)
    print(f"Saved YOLO cache to: {out_path}")

print(f"Processed {frame_idx} frames.")

cap = cv2.VideoCapture(video_path)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(W)
print(H)
