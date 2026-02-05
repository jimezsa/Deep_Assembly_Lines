"""
Reads labeled pickles from LABELED_DIR, computes a fixed-length per-frame feature
vector, groups frames by (sequence, camera), sorts by frame_idx, and
saves variable-length sequences ready for many-to-many LSTM training.

Output (OUT_PATH .npz):
 - feats: list of numpy arrays, each array shape (T_i, F)
 - labels: list of numpy arrays, each array shape (T_i,)
 - meta: list of tuples (sequence, camera, [frame_idx list])
 
IMPORTANT: Array indices (0, 1, 2, ...) map to original frame indices via meta[i][2]
Example: meta[i][2] = [0, 15, 30, 45, ..., 270, 315]
         Array index 18 → Original frame_idx 270
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
OUT_PATH = "model_input/lstm_input.npz"

PREFERRED_FEAT_KEYS = [
    "feat_vector", "features", "feat", "feat_vec", "feature_vector"
]

MAX_BAT = 6  # max number of battery centroids encoded
# =========================


# -------------------------------------------------
# Feature construction helpers
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
        frame_idx = int(fr.get("frame_idx", -1))  # Original frame index (e.g., 0, 15, 30, ..., 270, 315)
        label = int(fr.get("label", 0))

        try:
            feat = extract_feat(fr)
        except Exception:
            continue

        by_seqcam[(seq, cam)].append((frame_idx, feat, label))


# -------------------------------------------------
# Build ordered sequences
# -------------------------------------------------
seqs_feats = []
seqs_labels = []
seqs_meta = []

for (seq, cam), items in by_seqcam.items():
    # Sort by original frame_idx (e.g., 0, 15, 30, ..., 270, 315)
    items.sort(key=lambda x: x[0])

    # Extract original frame indices (preserves the mapping!)
    frame_idxs = [it[0] for it in items]
    
    # Stack into arrays (creates implicit sequential indexing 0, 1, 2, ...)
    feats = np.stack([it[1] for it in items], axis=0)
    labels = np.asarray([it[2] for it in items], dtype=np.int64)

    # Store metadata with frame_idx mapping
    # This allows unmapping: array_index → original_frame_idx via frame_idxs[array_index]
    seqs_feats.append(feats)
    seqs_labels.append(labels)
    seqs_meta.append((seq, cam, frame_idxs))
    
    # Print mapping for verification
    print(f"{seq}/{cam}: {len(items)} frames")
    print(f"  Array indices: 0 to {len(items)-1}")
    print(f"  Original frame_idx: {frame_idxs[0]} to {frame_idxs[-1]}")
    if len(items) > 0:
        # Show example mapping for sequences with errors
        if any(labels):  # If there are any error labels
            error_positions = np.where(labels == 1)[0]
            if len(error_positions) > 0:
                print(f"  ⚠️  ERROR frames:")
                print(f"    Array indices: {error_positions[0]} to {error_positions[-1]}")
                print(f"    Original frame_idx: {frame_idxs[error_positions[0]]} to {frame_idxs[error_positions[-1]]}")


# -------------------------------------------------
# Save variable-length sequences
# -------------------------------------------------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

np.savez_compressed(
    OUT_PATH,
    feats=np.asarray(seqs_feats, dtype=object),
    labels=np.asarray(seqs_labels, dtype=object),
    meta=np.asarray(seqs_meta, dtype=object),
)

print(f"\n✅ Saved {len(seqs_feats)} sequences → {OUT_PATH}")
if seqs_feats:
    print(f"Example sequence shape: {seqs_feats[0].shape}")
    
print("\n" + "="*80)
print("FRAME INDEX MAPPING INFO")
print("="*80)
print("""
The metadata preserves the mapping between array indices and original frame indices:
  - feats[i] has shape (T, F) with array indices 0, 1, 2, ..., T-1
  - meta[i][2] is a list of original frame_idx values
  
To unmap: original_frame_idx = meta[seq_idx][2][array_idx]

Example for wrong5:
  - Array index 18 → Original frame_idx 270
  - Array index 21 → Original frame_idx 315
  
This means:
  - ERROR_RANGES in frame_labeling.py uses ORIGINAL indices (270, 315)
  - Model predictions use ARRAY indices (18, 21)
  - Use meta to convert between them!
""")



###########################################################################################################################3

# """
# Reads labeled pickles from LABELED_DIR, computes a fixed-length per-frame feature
# vector, groups frames by (sequence, camera), sorts by frame_idx, and
# saves variable-length sequences ready for many-to-many LSTM training.

# Output (OUT_PATH .npz):
#  - feats: list of numpy arrays, each array shape (T_i, F)
#  - labels: list of numpy arrays, each array shape (T_i,)
#  - meta: list of tuples (sequence, camera, [frame_idx list])
# """

# import os
# import pickle
# from collections import defaultdict
# from itertools import combinations
# import numpy as np

# # =========================
# # CONFIG
# # =========================
# LABELED_DIR = "labeled_cache"
# OUT_PATH = "model_input/lstm_input.npz"

# PREFERRED_FEAT_KEYS = [
#     "feat_vector", "features", "feat", "feat_vec", "feature_vector"
# ]

# MAX_BAT = 6  # max number of battery centroids encoded
# # =========================


# # -------------------------------------------------
# # Feature construction helpers
# # -------------------------------------------------
# def pairwise_dists(centroids, max_len):
#     """Flattened pairwise Euclidean distances, padded/truncated."""
#     if len(centroids) < 2:
#         return [0.0] * max_len

#     pts = np.asarray(centroids, dtype=np.float32)
#     dists = [
#         float(np.linalg.norm(pts[i] - pts[j]))
#         for i, j in combinations(range(len(pts)), 2)
#     ]

#     if len(dists) < max_len:
#         dists += [0.0] * (max_len - len(dists))
#     else:
#         dists = dists[:max_len]

#     return dists


# def build_feat_from_frame(frame, max_bat=MAX_BAT):
#     """
#     Deterministic fixed-length feature vector (36 dims with max_bat=6):
#       - number of batteries (1)
#       - flattened centroids (2 * max_bat)
#       - pairwise distances (max_bat choose 2)
#       - case bbox center + size (4)
#       - centroid mean + std (4)
#     """
#     bats = frame.get("batteries", [])

#     centroids = [
#         b.get("centroid", (0.0, 0.0)) for b in bats
#     ]
#     centroids = sorted(
#         centroids, key=lambda c: (float(c[0]), float(c[1]))
#     )[:max_bat]

#     # 1) number of batteries
#     n_batts = float(len(bats))

#     # 2) flattened centroids
#     cent_flat = []
#     for cx, cy in centroids:
#         cent_flat.extend([float(cx), float(cy)])
#     while len(cent_flat) < max_bat * 2:
#         cent_flat.append(0.0)

#     # 3) pairwise distances
#     max_pw = (max_bat * (max_bat - 1)) // 2
#     pw = pairwise_dists(centroids, max_pw)

#     # 4) case bbox center + size
#     case = frame.get("case")
#     if case and isinstance(case, dict) and case.get("bbox") is not None:
#         try:
#             x1, y1, x2, y2 = case["bbox"]
#             case_cx = (x1 + x2) / 2.0
#             case_cy = (y1 + y2) / 2.0
#             case_w = x2 - x1
#             case_h = y2 - y1
#         except Exception:
#             case_cx = case_cy = case_w = case_h = 0.0
#     else:
#         case_cx = case_cy = case_w = case_h = 0.0

#     # 5) centroid mean + std
#     if centroids:
#         pts = np.asarray(centroids, dtype=np.float32)
#         mean_cx, mean_cy = pts[:, 0].mean(), pts[:, 1].mean()
#         std_cx, std_cy = pts[:, 0].std(), pts[:, 1].std()
#     else:
#         mean_cx = mean_cy = std_cx = std_cy = 0.0

#     feat = (
#         [n_batts]
#         + cent_flat
#         + pw
#         + [case_cx, case_cy, case_w, case_h]
#         + [mean_cx, mean_cy, std_cx, std_cy]
#     )

#     return np.asarray(feat, dtype=np.float32)


# def extract_feat(frame):
#     """
#     Extract a 1D feature vector from a frame dict.
#     1) Use existing feature keys if present
#     2) Otherwise build from batteries + case geometry
#     """
#     for k in PREFERRED_FEAT_KEYS:
#         if k in frame and frame[k] is not None:
#             arr = np.asarray(frame[k], dtype=np.float32)
#             return arr.reshape(-1)

#     if "batteries" in frame or "case" in frame:
#         return build_feat_from_frame(frame)

#     raise ValueError("No usable feature found in frame")


# # -------------------------------------------------
# # Main processing
# # -------------------------------------------------
# by_seqcam = defaultdict(list)

# for fname in sorted(os.listdir(LABELED_DIR)):
#     if not fname.endswith((".pkl", ".pickle")):
#         continue

#     path = os.path.join(LABELED_DIR, fname)
#     with open(path, "rb") as f:
#         data = pickle.load(f)

#     fallback_seq = fname.rsplit(".", 1)[0]

#     for fr in data:
#         seq = fr.get("sequence", fr.get("seq", fallback_seq))
#         cam = fr.get("camera", fr.get("cam", "cam0"))
#         frame_idx = int(fr.get("frame_idx", -1))
#         label = int(fr.get("label", 0))

#         try:
#             feat = extract_feat(fr)
#         except Exception:
#             continue

#         by_seqcam[(seq, cam)].append((frame_idx, feat, label))


# # -------------------------------------------------
# # Build ordered sequences
# # -------------------------------------------------
# seqs_feats = []
# seqs_labels = []
# seqs_meta = []

# for (seq, cam), items in by_seqcam.items():
#     items.sort(key=lambda x: x[0])

#     frame_idxs = [it[0] for it in items]
#     feats = np.stack([it[1] for it in items], axis=0)
#     labels = np.asarray([it[2] for it in items], dtype=np.int64)

#     seqs_feats.append(feats)
#     seqs_labels.append(labels)
#     seqs_meta.append((seq, cam, frame_idxs))


# # -------------------------------------------------
# # Save variable-length sequences
# # -------------------------------------------------
# os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# np.savez_compressed(
#     OUT_PATH,
#     feats=np.asarray(seqs_feats, dtype=object),
#     labels=np.asarray(seqs_labels, dtype=object),
#     meta=np.asarray(seqs_meta, dtype=object),
# )

# print(f"Saved {len(seqs_feats)} sequences → {OUT_PATH}")
# if seqs_feats:
#     print("Example sequence shape:", seqs_feats[0].shape)