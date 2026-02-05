# This file assumes you have the original videos available
# (render-from-cache fallback is commented out)


import os
import pickle
import cv2
import numpy as np
from collections import defaultdict

# =========================
# CONFIG - edit these
# =========================
YOLO_CACHE_DIR = "yolo_cache_missing"        # directory with per-sequence .pkl files
VIDEO_ROOT = "complete_data_missing"         # root where videos are stored (same layout as before)
OUT_IMG_DIR = "frames_for_labeling"  # where extracted frames will be saved
# OUT_RENDER_DIR = "rendered_cache"  # render mode disabled
FRAME_STRIDE = None                  # if not None, will only save frames where frame_idx % FRAME_STRIDE == 0
OVERWRITE = False                    # overwrite existing images
VERBOSE = True

# =========================
# Helpers
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def draw_render(frame_size, case_poly, centroids):
    """Return a white canvas with case polygon and centroids drawn.
    (Kept for reference but not used since render mode is disabled.)
    """
    h, w = frame_size
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    if case_poly is not None:
        try:
            pts = case_poly.astype(np.int32).reshape((-1,1,2))
            cv2.polylines(canvas, [pts], True, (0,128,255), 2)
            cv2.fillPoly(canvas, [pts], (240, 240, 255))
        except Exception:
            pass
    for (cx, cy) in centroids:
        try:
            cv2.circle(canvas, (int(cx), int(cy)), 6, (0,0,255), -1)
        except Exception:
            pass
    return canvas

def find_video_for(sequence, camera):
    """
    Attempt to locate a video file for a given sequence and camera.
    Assumes layout: VIDEO_ROOT/<sequence>/<camera>/*.mp4
    Returns full path or None.
    """
    cam_dir = os.path.join(VIDEO_ROOT, sequence, camera)
    if not os.path.isdir(cam_dir):
        return None
    mp4s = [f for f in os.listdir(cam_dir) if f.lower().endswith(".mp4")]
    if not mp4s:
        return None
    # choose first mp4 (adjust if you need a different selection)
    return os.path.join(cam_dir, mp4s[0])

# =========================
# Main
# =========================
ensure_dir(OUT_IMG_DIR)
# render mode disabled, so no OUT_RENDER_DIR creation

# iterate pickles
pkl_files = [f for f in os.listdir(YOLO_CACHE_DIR) if f.endswith(".pkl")]
if not pkl_files:
    raise SystemExit(f"No .pkl files found in {YOLO_CACHE_DIR}")

for pkl in sorted(pkl_files):
    seq_name = os.path.splitext(pkl)[0]
    pkl_path = os.path.join(YOLO_CACHE_DIR, pkl)
    if VERBOSE:
        print(f"\nProcessing sequence: {seq_name}  ({pkl_path})")

    with open(pkl_path, "rb") as fh:
        try:
            frames = pickle.load(fh)
        except Exception as e:
            print("  ERROR loading pickle:", e)
            continue

    # group frames by camera to minimize video opens
    frames_by_cam = defaultdict(list)
    for fr in frames:
        cam = fr.get("camera", "camera_unknown")
        frames_by_cam[cam].append(fr)

    for cam, cam_frames in sorted(frames_by_cam.items()):
        if VERBOSE:
            print(f"  Camera: {cam}  frames: {len(cam_frames)}")

        out_img_cam = os.path.join(OUT_IMG_DIR, seq_name, cam)
        ensure_dir(out_img_cam)
        # render mode disabled: out_render_cam = None

        # try to find video for this camera
        video_path = None
        if VIDEO_ROOT:
            video_path = find_video_for(seq_name, cam)
            if VERBOSE and video_path is None:
                print(f"    ERROR: video not found for {seq_name}/{cam} — render mode is disabled, skipping this camera")
                continue  # skip camera if no video available

        # if video exists, open it once
        cap = None
        if video_path:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                if VERBOSE:
                    print(f"    Could not open video {video_path}; skipping this camera")
                cap = None
                continue

        # sort frames by frame_idx for efficient seeking
        cam_frames_sorted = sorted(cam_frames, key=lambda x: int(x.get("frame_idx", 0)))

        for fr in cam_frames_sorted:
            idx = int(fr.get("frame_idx", 0))
            if FRAME_STRIDE is not None and (idx % FRAME_STRIDE != 0):
                continue

            out_img_path = os.path.join(out_img_cam, f"frame_{idx:06d}.jpg")

            # skip if already exists
            if not OVERWRITE and os.path.exists(out_img_path):
                if VERBOSE:
                    print(f"    Skipping existing {out_img_path}")
                continue

            saved = False
            # --- VIDEO MODE: seek and read exact frame ---
            if cap is not None:
                # seek to frame index
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, img = cap.read()
                if ret and img is not None:
                    cv2.imwrite(out_img_path, img)
                    saved = True
                    if VERBOSE:
                        print(f"    Saved frame {idx} -> {out_img_path}")
                else:
                    # read failed
                    if VERBOSE:
                        print(f"    Warning: failed to read frame {idx} from video; skipping frame")
                    saved = False

            # If saved is False here, we do NOT render from cache because render mode is disabled.
            if not saved:
                if VERBOSE:
                    print(f"    Skipped frame {idx} (no image saved)")

        # release video capture
        if cap is not None:
            cap.release()

print("\nDone. Images saved to:", OUT_IMG_DIR)
