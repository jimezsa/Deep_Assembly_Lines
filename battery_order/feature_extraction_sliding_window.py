"""
feature_extraction.py - WINDOWED VERSION

Creates fixed-length sliding windows from sequences for online inference compatibility.

Each window:
- Fixed length (WINDOW_SIZE frames)
- Overlaps with neighbors (controlled by WINDOW_STRIDE)
- Preserves frame-level labels for many-to-many LSTM training
- Tracks original sequence for proper cross-validation splitting

Output (.npz):
- feats: list of numpy arrays, each (WINDOW_SIZE, F)
- labels: list of numpy arrays, each (WINDOW_SIZE,)
- meta: list of tuples (orig_sequence, camera, window_start_idx, frame_idx_list)
"""

import os
import pickle
from collections import defaultdict
from itertools import combinations
import numpy as np

# =========================
# CONFIG
# =========================
LABELED_DIR = "labeled_cache"
OUT_PATH = "model_input/lstm_input_windowed.npz"

PREFERRED_FEAT_KEYS = [
    "feat_vector", "features", "feat", "feat_vec", "feature_vector"
]

MAX_BAT = 6  # max number of battery centroids encoded

# WINDOWING PARAMETERS
WINDOW_SIZE = 15 # 30, 15, 20   # Number of frames per window (30 frames @ stride=15 = 7.5 seconds)
WINDOW_STRIDE = 8 # 5, 8 # Slide window by this many frames (overlap = WINDOW_SIZE - WINDOW_STRIDE)

# =========================


# -------------------------------------------------
# Feature construction helpers (UNCHANGED)
# -------------------------------------------------
def pairwise_dists(centroids, max_len):
    """Flattened pairwise Euclidean distances, padded/truncated."""
    if len(centroids) < 2:
        return [0.0] * max_len

    pts = np.asarray(centroids, dtype=np.float32)
    dists = [
        float(np.linalg.norm(pts[i] - pts[j]))
        for i, j in combinations(range(len(pts)), 2)
    ]

    if len(dists) < max_len:
        dists += [0.0] * (max_len - len(dists))
    else:
        dists = dists[:max_len]

    return dists


def build_feat_from_frame(frame, max_bat=MAX_BAT):
    """
    Deterministic fixed-length feature vector (36 dims with max_bat=6):
      - number of batteries (1)
      - flattened centroids (2 * max_bat)
      - pairwise distances (max_bat choose 2)
      - case bbox center + size (4)
      - centroid mean + std (4)
    """
    bats = frame.get("batteries", [])

    centroids = [
        b.get("centroid", (0.0, 0.0)) for b in bats
    ]
    centroids = sorted(
        centroids, key=lambda c: (float(c[0]), float(c[1]))
    )[:max_bat]

    # 1) number of batteries
    n_batts = float(len(bats))

    # 2) flattened centroids
    cent_flat = []
    for cx, cy in centroids:
        cent_flat.extend([float(cx), float(cy)])
    while len(cent_flat) < max_bat * 2:
        cent_flat.append(0.0)

    # 3) pairwise distances
    max_pw = (max_bat * (max_bat - 1)) // 2
    pw = pairwise_dists(centroids, max_pw)

    # 4) case bbox center + size
    case = frame.get("case")
    if case and isinstance(case, dict) and case.get("bbox") is not None:
        try:
            x1, y1, x2, y2 = case["bbox"]
            case_cx = (x1 + x2) / 2.0
            case_cy = (y1 + y2) / 2.0
            case_w = x2 - x1
            case_h = y2 - y1
        except Exception:
            case_cx = case_cy = case_w = case_h = 0.0
    else:
        case_cx = case_cy = case_w = case_h = 0.0

    # 5) centroid mean + std
    if centroids:
        pts = np.asarray(centroids, dtype=np.float32)
        mean_cx, mean_cy = pts[:, 0].mean(), pts[:, 1].mean()
        std_cx, std_cy = pts[:, 0].std(), pts[:, 1].std()
    else:
        mean_cx = mean_cy = std_cx = std_cy = 0.0

    feat = (
        [n_batts]
        + cent_flat
        + pw
        + [case_cx, case_cy, case_w, case_h]
        + [mean_cx, mean_cy, std_cx, std_cy]
    )

    return np.asarray(feat, dtype=np.float32)


def extract_feat(frame):
    """
    Extract a 1D feature vector from a frame dict.
    1) Use existing feature keys if present
    2) Otherwise build from batteries + case geometry
    """
    for k in PREFERRED_FEAT_KEYS:
        if k in frame and frame[k] is not None:
            arr = np.asarray(frame[k], dtype=np.float32)
            return arr.reshape(-1)

    if "batteries" in frame or "case" in frame:
        return build_feat_from_frame(frame)

    raise ValueError("No usable feature found in frame")


# -------------------------------------------------
# Main processing
# -------------------------------------------------
by_seqcam = defaultdict(list)

# Load all labeled data
for fname in sorted(os.listdir(LABELED_DIR)):
    if not fname.endswith((".pkl", ".pickle")):
        continue

    path = os.path.join(LABELED_DIR, fname)
    with open(path, "rb") as f:
        data = pickle.load(f)

    fallback_seq = fname.rsplit(".", 1)[0]

    for fr in data:
        seq = fr.get("sequence", fr.get("seq", fallback_seq))
        cam = fr.get("camera", fr.get("cam", "cam0"))
        frame_idx = int(fr.get("frame_idx", -1))
        label = int(fr.get("label", 0))

        try:
            feat = extract_feat(fr)
        except Exception:
            continue

        by_seqcam[(seq, cam)].append((frame_idx, feat, label))


# -------------------------------------------------
# Create sliding windows from each sequence
# -------------------------------------------------
windowed_feats = []
windowed_labels = []
windowed_meta = []

total_windows = 0
total_sequences = 0

print(f"\nCreating sliding windows (size={WINDOW_SIZE}, stride={WINDOW_STRIDE})...")
print("=" * 80)

for (seq, cam), items in sorted(by_seqcam.items()):
    # Sort by original frame_idx
    items.sort(key=lambda x: x[0])
    
    if len(items) < WINDOW_SIZE:
        print(f"⚠️  {seq}/{cam}: Only {len(items)} frames, skipping (need {WINDOW_SIZE})")
        continue
    
    # Stack into arrays
    frame_idxs = np.array([it[0] for it in items])
    feats = np.stack([it[1] for it in items], axis=0)
    labels = np.array([it[2] for it in items], dtype=np.int64)
    
    # Create sliding windows
    seq_windows = 0
    for i in range(0, len(feats) - WINDOW_SIZE + 1, WINDOW_STRIDE):
        window_feats = feats[i:i+WINDOW_SIZE]
        window_labels = labels[i:i+WINDOW_SIZE]
        window_frame_idxs = frame_idxs[i:i+WINDOW_SIZE].tolist()
        
        windowed_feats.append(window_feats)
        windowed_labels.append(window_labels)
        
        # Metadata: (orig_sequence, camera, window_start_idx, frame_idx_list)
        # window_start_idx helps identify which window this is within the original sequence
        windowed_meta.append((seq, cam, i, window_frame_idxs))
        
        seq_windows += 1
    
    total_windows += seq_windows
    total_sequences += 1
    
    # Show error distribution in windows
    error_windows = sum(1 for i in range(len(windowed_labels) - seq_windows, len(windowed_labels)) 
                       if np.any(windowed_labels[i]))
    print(f"{seq}/{cam}: {len(items)} frames → {seq_windows} windows ({error_windows} with errors)")

# -------------------------------------------------
# Save windowed sequences
# -------------------------------------------------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

np.savez_compressed(
    OUT_PATH,
    feats=np.asarray(windowed_feats, dtype=object),
    labels=np.asarray(windowed_labels, dtype=object),
    meta=np.asarray(windowed_meta, dtype=object),
)

print("\n" + "="*80)
print(f"✅ Saved {total_windows} windows from {total_sequences} sequences → {OUT_PATH}")
print(f"   Window size: {WINDOW_SIZE} frames")
print(f"   Window stride: {WINDOW_STRIDE} frames")
print(f"   Overlap: {WINDOW_SIZE - WINDOW_STRIDE} frames")
if windowed_feats:
    print(f"   Feature dim: {windowed_feats[0].shape[1]}")
    total_error_windows = sum(1 for labels in windowed_labels if np.any(labels))
    print(f"   Windows with errors: {total_error_windows}/{total_windows} ({100*total_error_windows/total_windows:.1f}%)")

print("\n" + "="*80)
print("WINDOWED DATA STRUCTURE")
print("="*80)
print("""
Each window is now a fixed-length sequence:
  - feats[i]: shape (WINDOW_SIZE, F) - always 30 frames
  - labels[i]: shape (WINDOW_SIZE,) - frame-level labels within window
  - meta[i]: (orig_seq, camera, window_start_idx, frame_idx_list)

For cross-validation:
  - Group windows by orig_seq to prevent data leakage
  - Windows from same orig_seq must stay together in train or val

For online inference:
  - Use same WINDOW_SIZE=30 frames
  - Buffer incoming frames until you have 30
  - Run prediction on window
  - Use center frame prediction or majority vote
""")