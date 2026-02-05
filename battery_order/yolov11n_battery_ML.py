import os
import cv2
from ultralytics import YOLO
import numpy as np
import torch
import pickle

# =========================
# CONFIG
# =========================
CASE_ID = 1
BATTERY_ID = 3
KEEP_CLASS_IDS = [CASE_ID, BATTERY_ID]

FRAME_STRIDE = 15
SAVE_CACHE = True
DRAW_VIS = False

DATA_ROOT = "complete_data_missing"
OUT_DIR = "yolo_cache"
os.makedirs(OUT_DIR, exist_ok=True)

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
    r"..\yolov11_finetuned",
    "runs", "segment", "yolov11n_seg_custom",
    "weights", "best.pt"
)
model = YOLO(model_path).to(DEVICE)
print(f"Loaded model from: {model_path}")

# =========================
# GLOBAL DATA (optional)
# =========================
all_data = []

# =========================
# MAIN LOOP
# =========================
for sequence in sorted(os.listdir(DATA_ROOT)):
    sequence_path = os.path.join(DATA_ROOT, sequence)
    if not os.path.isdir(sequence_path):
        continue

    print(f"\n=== Processing sequence: {sequence} ===")

    sequence_data = []   # <-- per-sequence buffer

    for camera in sorted(os.listdir(sequence_path)):
        camera_path = os.path.join(sequence_path, camera)
        if not os.path.isdir(camera_path):
            continue

        mp4s = [f for f in os.listdir(camera_path) if f.endswith(".mp4")]
        if len(mp4s) == 0:
            print(f"  ⚠ Camera {camera}: no video found, skipping")
            continue

        video_path = os.path.join(camera_path, mp4s[0])
        print(f"  → Camera {camera}: {mp4s[0]}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"    ⚠ Could not open video")
            continue

        frame_idx = 0
        stored = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STRIDE != 0:
                frame_idx += 1
                continue

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            results = model(frame, verbose=False, device=DEVICE)

            frame_record = {
                "sequence": sequence,
                "camera": camera,
                "frame_idx": frame_idx,
                "t": timestamp,
                "case": None,
                "batteries": []
            }

            for r in results:
                if r.masks is None:
                    continue

                for j, poly in enumerate(r.masks.xy):
                    class_id = int(r.boxes.cls[j])
                    if class_id not in KEEP_CLASS_IDS:
                        continue

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
                            "centroid": (cx, cy)
                        })

            sequence_data.append(frame_record)
            all_data.append(frame_record)

            stored += 1
            frame_idx += 1

        cap.release()
        print(f"    ✔ Stored {stored} frames")

    # =========================
    # SAVE PER-SEQUENCE CACHE
    # =========================
    if SAVE_CACHE and len(sequence_data) > 0:
        seq_out = os.path.join(OUT_DIR, f"{sequence}.pkl")
        with open(seq_out, "wb") as f:
            pickle.dump(sequence_data, f)

        print(f"  💾 Saved {len(sequence_data)} frames → {seq_out}")

# =========================
# SAVE GLOBAL CACHE (optional)
# =========================
if SAVE_CACHE and len(all_data) > 0:
    out_path = os.path.join(OUT_DIR, "all_sequences_frames_yolo.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(all_data, f)

    print(f"\n✅ Saved TOTAL {len(all_data)} frames → {out_path}")
