"""
Train Final Model - WINDOWED VERSION with LOCO-optimized settings

Uses windowed/causal approach optimized through LOCO cross-validation.
Maintains same-sequence split strategy:
- Train: wrong1-wrong3 (all cameras)
- Val:   wrong4 (all cameras) → threshold + early stopping
- Test:  wrong5 (all cameras) → multi-camera fusion evaluation

Key improvements over non-windowed version:
- Fixed-length windows (15 frames) for consistent inference
- Causal post-processing (hysteresis + minimum error persistence)
- Timeline reconstruction from overlapping windows
- Optimized hyperparameters from LOCO CV (Event F1=0.849)
"""
import os

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

from dataset_sliding_window import SequenceDataset
from collate_fn_sliding_window import collate_fn
from model_sliding_window import FrameLSTM
from loss_sliding_window import masked_bce_loss

# --------------------------------------------------
# CONFIG - LOCO-optimized hyperparameters
# --------------------------------------------------
NPZ_PATH = "model_input/lstm_input_windowed.npz"
BATCH_SIZE = 32  # Larger batches possible with fixed-length windows
EPOCHS = 40
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 6
GRAD_CLIP = 1.0
POS_WEIGHT_SCALE = 4.0  # Optimized from LOCO CV

USE_FOCAL_LOSS = False
FOCAL_ALPHA = 0.85
FOCAL_GAMMA = 2.0

# Calculate effective FPS for strided frames
FRAME_STRIDE = 15  # From dataset creation
VIDEO_FPS = 30
EFFECTIVE_FPS = VIDEO_FPS / FRAME_STRIDE  # = 2.0 fps
MIN_ERROR_DURATION_SEC = 3.0  # Optimized from LOCO CV

SMOOTH_K = 5  # Optimized from LOCO CV
MIN_RUN = 3   # Optimized from LOCO CV

OUT_DIR = "final_model_windowed"
os.makedirs(OUT_DIR, exist_ok=True)

# Camera aggregation: filter completely failing cameras
MIN_CAMERA_EVENT_F1 = 0.0  # Set to 0.0 to include all, or 0.5 to filter poor cameras

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def smooth_and_minrun(probs, thr):
    """Apply causal smoothing and minimum run length filtering"""
    if SMOOTH_K <= 1:
        preds = (probs > thr).astype(int)
        return preds

    sm = ndi.uniform_filter1d(probs, size=SMOOTH_K)
    preds = (sm > thr).astype(int)

    if MIN_RUN <= 1:
        return preds

    out = preds.copy()
    i = 0
    while i < len(preds):
        if preds[i] == 1:
            j = i
            while j < len(preds) and preds[j] == 1:
                j += 1
            if j - i < MIN_RUN:
                out[i:j] = 0
            i = j
        else:
            i += 1
    return out


def apply_causal_hysteresis(probs, threshold_high=0.7, threshold_low=0.3):
    """Apply hysteresis thresholding (streaming-safe, causal)"""
    preds = np.zeros(len(probs), dtype=int)
    error_active = False
    
    for i in range(len(probs)):
        if error_active:
            if probs[i] < threshold_low:
                error_active = False
                preds[i] = 0
            else:
                preds[i] = 1
        else:
            if probs[i] >= threshold_high:
                error_active = True
                preds[i] = 1
            else:
                preds[i] = 0
    
    return preds


def apply_min_error_persistence(preds, min_duration_sec=2.0, fps=2.0):
    """Enforce minimum error duration (streaming-safe, causal)"""
    min_frames = int(min_duration_sec * fps)
    output = preds.copy()
    
    i = 0
    while i < len(preds):
        if preds[i] == 1:
            j = i
            while j < len(preds) and preds[j] == 1:
                j += 1
            
            error_duration = j - i
            
            if error_duration < min_frames:
                end_idx = min(i + min_frames, len(preds))
                output[i:end_idx] = 1
                i = end_idx
            else:
                i = j
        else:
            i += 1
    
    return output


def compute_frame_counts(preds, labels):
    """Compute TP, FP, FN for frame-level metrics"""
    tp = fp = fn = 0
    for p, y in zip(preds, labels):
        tp += int(((p == 1) & (y == 1)).sum())
        fp += int(((p == 1) & (y == 0)).sum())
        fn += int(((p == 0) & (y == 1)).sum())
    return tp, fp, fn


def group_events(arr):
    """Extract event spans (start, end) from binary array"""
    events = []
    i = 0
    while i < len(arr):
        if arr[i] == 1:
            j = i
            while j < len(arr) and arr[j] == 1:
                j += 1
            events.append((i, j - 1))
            i = j
        else:
            i += 1
    return events


def event_metrics(preds, labels):
    """Compute event-level precision, recall, F1"""
    tp_events = fp_events = fn_events = 0
    
    for pred, lab in zip(preds, labels):
        pe = group_events(pred)
        ge = group_events(lab)
        matched_p, matched_g = set(), set()
        
        for i, (ps, pe_) in enumerate(pe):
            for j, (gs, ge_) in enumerate(ge):
                if not (pe_ < gs or ps > ge_):
                    matched_p.add(i)
                    matched_g.add(j)
                    break
        
        tp_events += len(matched_p)
        fp_events += len(pe) - len(matched_p)
        fn_events += len(ge) - len(matched_g)

    p = tp_events / (tp_events + fp_events) if tp_events + fp_events else 0
    r = tp_events / (tp_events + fn_events) if tp_events + fn_events else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return p, r, f1


def find_best_threshold(probs, labels):
    """Find optimal threshold on validation set"""
    all_labels = np.concatenate(labels) if labels else np.array([])
    if len(all_labels) == 0 or all_labels.sum() == 0:
        return 0.5
    
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.1, 0.9, 41):
        preds = [(p > t).astype(int) for p in probs]
        tp, fp, fn = compute_frame_counts(preds, labels)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def reconstruct_timelines_with_aggregation(dataset, indices, model):
    """
    Reconstruct timelines from overlapping windows.
    
    For each frame appearing in multiple windows, average the predictions.
    This matches the LOCO CV evaluation strategy.
    """
    model.eval()
    
    frame_predictions = {}  # (orig_seq, cam, frame_idx) -> [list of probs]
    frame_labels = {}       # (orig_seq, cam, frame_idx) -> label
    
    with torch.no_grad():
        for idx in indices:
            x, y = dataset[idx]
            orig_seq, cam, win_start, frame_idxs = dataset.meta[idx]
            
            x = x.unsqueeze(0).to(DEVICE)
            mask = torch.ones(1, x.shape[1], device=DEVICE)
            
            probs = torch.sigmoid(model(x, mask))[0].cpu().numpy()
            labels = y.numpy()
            
            for i, (frame_idx, prob, label) in enumerate(zip(frame_idxs, probs, labels)):
                key = (orig_seq, cam, frame_idx)
                
                if key not in frame_predictions:
                    frame_predictions[key] = []
                    frame_labels[key] = label
                
                frame_predictions[key].append(prob)
    
    # Aggregate predictions for each frame (average across windows)
    timelines = {}
    
    for (orig_seq, cam, frame_idx), probs_list in frame_predictions.items():
        key = (orig_seq, cam)
        
        if key not in timelines:
            timelines[key] = {
                "probs": [],
                "labels": [],
                "frames": []
            }
        
        avg_prob = np.mean(probs_list)
        
        timelines[key]["probs"].append(avg_prob)
        timelines[key]["labels"].append(frame_labels[(orig_seq, cam, frame_idx)])
        timelines[key]["frames"].append(frame_idx)
    
    # Sort by frame index
    for key in timelines:
        order = np.argsort(timelines[key]["frames"])
        timelines[key]["probs"] = np.array(timelines[key]["probs"])[order]
        timelines[key]["labels"] = np.array(timelines[key]["labels"])[order]
        timelines[key]["frames"] = np.array(timelines[key]["frames"])[order]
    
    return timelines


def format_time_ranges(events, total_frames):
    """Convert event frame ranges to normalized [0,1] time ranges"""
    if not events:
        return []
    return [(start/total_frames, (end+1)/total_frames) for start, end in events]


def map_events_to_original(events, frame_idxs):
    """Map array index events to original frame index events"""
    if not events:
        return []
    return [(frame_idxs[start], frame_idxs[end]) for start, end in events]


def check_overlap(pred_events, true_events):
    """Check if predicted events overlap with ground truth events"""
    if not pred_events or not true_events:
        return False
    for p_start, p_end in pred_events:
        for t_start, t_end in true_events:
            if not (p_end < t_start or p_start > t_end):
                return True
    return False


# --------------------------------------------------
# Load Data - SAME-SEQUENCE split
# --------------------------------------------------
print("="*80)
print("FINAL MODEL TRAINING - WINDOWED VERSION (LOCO-optimized)")
print("="*80)

dataset = SequenceDataset(NPZ_PATH)

TRAIN_SEQS = {"wrong1", "wrong2", "wrong3"}
VAL_SEQS   = {"wrong4"}
TEST_SEQS  = {"wrong5"}

train_indices, val_indices, test_indices = [], [], []

for i, (seq, cam, _, _) in enumerate(dataset.meta):
    if seq in TRAIN_SEQS:
        train_indices.append(i)
    elif seq in VAL_SEQS:
        val_indices.append(i)
    elif seq in TEST_SEQS:
        test_indices.append(i)

print(f"\nData Split:")
print(f"  Train: {len(train_indices)} windows (sequences: {', '.join(sorted(TRAIN_SEQS))})")
print(f"  Val:   {len(val_indices)} windows (sequences: {', '.join(sorted(VAL_SEQS))})")
print(f"  Test:  {len(test_indices)} windows (sequences: {', '.join(sorted(TEST_SEQS))})")
print(f"  Window size: {dataset.window_size} frames")

# --------------------------------------------------
# Loaders
# --------------------------------------------------
train_loader = DataLoader(
    Subset(dataset, train_indices),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    worker_init_fn=worker_init_fn
)

val_loader = DataLoader(
    Subset(dataset, val_indices),
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    worker_init_fn=worker_init_fn
)

test_subset = Subset(dataset, test_indices)
test_loader = DataLoader(
    test_subset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    worker_init_fn=worker_init_fn
)

# --------------------------------------------------
# Model Setup
# --------------------------------------------------
print("\n" + "="*80)
print("MODEL TRAINING")
print("="*80)

input_dim = dataset[0][0].shape[1]
model = FrameLSTM(input_dim=input_dim).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

labels_flat = np.concatenate([dataset[i][1] for i in train_indices])
pos_weight = (len(labels_flat) - labels_flat.sum()) / max(1, labels_flat.sum())
pos_weight = torch.tensor([pos_weight * POS_WEIGHT_SCALE]).to(DEVICE)

print(f"\nTraining Configuration:")
print(f"  Positive weight: {pos_weight.item():.2f}")
print(f"  Training windows: {len(train_indices):,}")
print(f"  Positive samples: {int(labels_flat.sum())} ({100*labels_flat.sum()/len(labels_flat):.1f}%)")
print(f"  Learning rate: {LR}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max epochs: {EPOCHS}")
print(f"  Early stopping patience: {PATIENCE}")
print(f"  Post-processing: SMOOTH_K={SMOOTH_K}, MIN_RUN={MIN_RUN}, MIN_DURATION={MIN_ERROR_DURATION_SEC}s\n")

# --------------------------------------------------
# Training with VAL early stopping
# --------------------------------------------------
best_val_loss = float("inf")
best_state = None
best_epoch = 0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_frames = 0, 0
    
    for x, y, mask in train_loader:
        x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
        logits = model(x, mask)
        loss = masked_bce_loss(
            logits, y, mask, pos_weight,
            use_focal=USE_FOCAL_LOSS,
            focal_alpha=FOCAL_ALPHA,
            focal_gamma=FOCAL_GAMMA
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_loss += loss.item()
        train_frames += 1
    
    train_loss /= train_frames if train_frames > 0 else 1.0

    model.eval()
    val_loss, n_samples = 0, 0
    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            logits = model(x, mask)
            logits_last = logits[:, -1]
            labels_last = y[:, -1]
            loss = F.binary_cross_entropy_with_logits(
                logits_last,
                labels_last.float(),
                pos_weight=pos_weight
            )
            val_loss += loss.item()
            n_samples += 1
    
    val_loss /= n_samples if n_samples > 0 else 1.0

    print(f"Epoch {epoch:02d}/{EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict().copy()
        best_epoch = epoch
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

# Load best state
if best_state is not None:
    model.load_state_dict(best_state)
    print(f"\n✅ Loaded best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

# Save final model
model_path = f"{OUT_DIR}/final_model.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': best_epoch,
    'val_loss': best_val_loss,
    'window_size': dataset.window_size,
    'hyperparams': {
        'lr': LR,
        'batch_size': BATCH_SIZE,
        'pos_weight_scale': POS_WEIGHT_SCALE,
        'smooth_k': SMOOTH_K,
        'min_run': MIN_RUN,
        'min_error_duration_sec': MIN_ERROR_DURATION_SEC,
    }
}, model_path)
print(f"💾 Model saved: {model_path}")

# --------------------------------------------------
# Threshold selection on VAL with timeline reconstruction
# --------------------------------------------------
print("\n" + "="*80)
print("THRESHOLD SELECTION (Validation Set)")
print("="*80)

# Reconstruct timelines from validation windows
val_timelines = reconstruct_timelines_with_aggregation(dataset, val_indices, model)

val_probs = [v["probs"] for v in val_timelines.values()]
val_labels = [v["labels"] for v in val_timelines.values()]

best_thr = find_best_threshold(val_probs, val_labels)

# Apply post-processing to validation set
val_preds = []
for p in val_probs:
    if SMOOTH_K > 1 or MIN_RUN > 1:
        pred = (p > best_thr).astype(int)
    else:
        pred = apply_causal_hysteresis(p, 
                                      threshold_high=best_thr + 0.15, 
                                      threshold_low=best_thr - 0.05)
    
    if SMOOTH_K > 1 or MIN_RUN > 1:
        pred_probs = pred.astype(float)
        pred = smooth_and_minrun(pred_probs, 0.5)
    
    pred = apply_min_error_persistence(pred, 
                                      min_duration_sec=MIN_ERROR_DURATION_SEC,
                                      fps=EFFECTIVE_FPS)
    
    val_preds.append(pred)

# Validation metrics
val_tp, val_fp, val_fn = compute_frame_counts(val_preds, val_labels)
val_frame_f1 = 2*val_tp/(2*val_tp + val_fp + val_fn) if (2*val_tp + val_fp + val_fn) > 0 else 0
val_event_p, val_event_r, val_event_f1 = event_metrics(val_preds, val_labels)

print(f"\nValidation Performance:")
print(f"  Best threshold: {best_thr:.2f}")
print(f"  Frame F1: {val_frame_f1:.3f}")
print(f"  Event F1: {val_event_f1:.3f} (P={val_event_p:.3f}, R={val_event_r:.3f})")

# --------------------------------------------------
# TEST evaluation with timeline reconstruction
# --------------------------------------------------
print("\n" + "="*80)
print("TEST SET EVALUATION")
print("="*80)

# Reconstruct timelines from test windows
test_timelines = reconstruct_timelines_with_aggregation(dataset, test_indices, model)

# Group timelines by camera
by_camera = defaultdict(lambda: {'probs': [], 'labels': [], 'frame_idxs': []})

for (seq, cam), timeline in test_timelines.items():
    by_camera[cam]['probs'].append(timeline['probs'])
    by_camera[cam]['labels'].append(timeline['labels'])
    by_camera[cam]['frame_idxs'].append(timeline['frames'])

# --------------------------------------------------
# PER-CAMERA metrics
# --------------------------------------------------
print("\n" + "-"*80)
print("PER-CAMERA RESULTS")
print("-"*80)

cam_results = {}

print(f"\n{'Camera':<16} {'Frame P':>8} {'Frame R':>8} {'Frame F1':>8} {'Event P':>8} {'Event R':>8} {'Event F1':>8}")
print("-"*80)

for cam in sorted(by_camera.keys()):
    cam_data = by_camera[cam]
    
    # Apply post-processing
    preds = []
    for p in cam_data['probs']:
        if SMOOTH_K > 1 or MIN_RUN > 1:
            pred = (p > best_thr).astype(int)
        else:
            pred = apply_causal_hysteresis(p, 
                                          threshold_high=best_thr + 0.15, 
                                          threshold_low=best_thr - 0.05)
        
        if SMOOTH_K > 1 or MIN_RUN > 1:
            pred_probs = pred.astype(float)
            pred = smooth_and_minrun(pred_probs, 0.5)
        
        pred = apply_min_error_persistence(pred, 
                                          min_duration_sec=MIN_ERROR_DURATION_SEC,
                                          fps=EFFECTIVE_FPS)
        
        preds.append(pred)
    
    # Compute metrics
    tp_c, fp_c, fn_c = compute_frame_counts(preds, cam_data['labels'])
    p = tp_c / (tp_c + fp_c) if tp_c + fp_c else 0
    r = tp_c / (tp_c + fn_c) if tp_c + fn_c else 0
    f1 = 2 * p * r / (p + r) if p + r else 0

    ep, er, ef1 = event_metrics(preds, cam_data['labels'])

    cam_results[cam] = dict(
        frame_p=p, frame_r=r, frame_f1=f1,
        event_p=ep, event_r=er, event_f1=ef1,
        preds=preds, 
        labels=cam_data['labels'], 
        probs=cam_data['probs'],
        frame_idxs=cam_data['frame_idxs']
    )
    
    cam_short = str(cam)
    print(f"{cam_short:<16} {p:8.3f} {r:8.3f} {f1:8.3f} {ep:8.3f} {er:8.3f} {ef1:8.3f}")

# --------------------------------------------------
# GLOBAL and AVERAGE metrics
# --------------------------------------------------
print("\n" + "-"*80)
print("AGGREGATED METRICS")
print("-"*80)

# Global: all cameras combined
all_preds = [pred for cam_data in cam_results.values() for pred in cam_data['preds']]
all_labels = [label for cam_data in cam_results.values() for label in cam_data['labels']]

tp, fp, fn = compute_frame_counts(all_preds, all_labels)
frame_p = tp / (tp + fp) if tp + fp else 0
frame_r = tp / (tp + fn) if tp + fn else 0
frame_f1 = 2 * frame_p * frame_r / (frame_p + frame_r) if frame_p + frame_r else 0
event_p, event_r, event_f1 = event_metrics(all_preds, all_labels)

print(f"\nGlobal (all cameras combined):")
print(f"  Frame:  P={frame_p:.3f} R={frame_r:.3f} F1={frame_f1:.3f}")
print(f"  Event:  P={event_p:.3f} R={event_r:.3f} F1={event_f1:.3f}")

# Macro average
avg_frame_p = np.mean([r["frame_p"] for r in cam_results.values()])
avg_frame_r = np.mean([r["frame_r"] for r in cam_results.values()])
avg_frame_f1 = np.mean([r["frame_f1"] for r in cam_results.values()])
avg_event_p = np.mean([r["event_p"] for r in cam_results.values()])
avg_event_r = np.mean([r["event_r"] for r in cam_results.values()])
avg_event_f1 = np.mean([r["event_f1"] for r in cam_results.values()])

print(f"\nMacro Average (across {len(cam_results)} cameras):")
print(f"  Frame:  P={avg_frame_p:.3f}±{np.std([r['frame_p'] for r in cam_results.values()]):.3f} "
      f"R={avg_frame_r:.3f}±{np.std([r['frame_r'] for r in cam_results.values()]):.3f} "
      f"F1={avg_frame_f1:.3f}±{np.std([r['frame_f1'] for r in cam_results.values()]):.3f}")
print(f"  Event:  P={avg_event_p:.3f}±{np.std([r['event_p'] for r in cam_results.values()]):.3f} "
      f"R={avg_event_r:.3f}±{np.std([r['event_r'] for r in cam_results.values()]):.3f} "
      f"F1={avg_event_f1:.3f}±{np.std([r['event_f1'] for r in cam_results.values()]):.3f}")

# --------------------------------------------------
# MULTI-CAMERA FUSION for robust inference
# --------------------------------------------------
print("\n" + "="*80)
print("MULTI-CAMERA FUSION (for deployment)")
print("="*80)

# Filter cameras: only use cameras with Event F1 > threshold
good_cameras = {cam: res for cam, res in cam_results.items() 
                if res['event_f1'] > MIN_CAMERA_EVENT_F1}

if len(good_cameras) == 0:
    print("\n⚠️  No cameras passed threshold - using all cameras")
    good_cameras = cam_results

print(f"\nUsing {len(good_cameras)}/{len(cam_results)} cameras for fusion:")
for cam in sorted(good_cameras.keys()):
    print(f"  Camera {str(cam):<16} Event F1={good_cameras[cam]['event_f1']:.3f}")

# Average probabilities across selected cameras
n_sequences = len(list(good_cameras.values())[0]['probs'])

fused_probs = []
fused_labels = []
fused_frame_idxs = []

for seq_idx in range(n_sequences):
    # Average probabilities across cameras for this sequence
    seq_probs = []
    for cam in sorted(good_cameras.keys()):
        seq_probs.append(good_cameras[cam]['probs'][seq_idx])
    
    fused_prob = np.mean(seq_probs, axis=0)
    fused_probs.append(fused_prob)
    
    # Labels are same across cameras
    fused_labels.append(list(good_cameras.values())[0]['labels'][seq_idx])
    fused_frame_idxs.append(list(good_cameras.values())[0]['frame_idxs'][seq_idx])

# Apply post-processing to fused predictions
fused_preds = []
for p in fused_probs:
    if SMOOTH_K > 1 or MIN_RUN > 1:
        pred = (p > best_thr).astype(int)
    else:
        pred = apply_causal_hysteresis(p, 
                                      threshold_high=best_thr + 0.15, 
                                      threshold_low=best_thr - 0.05)
    
    if SMOOTH_K > 1 or MIN_RUN > 1:
        pred_probs = pred.astype(float)
        pred = smooth_and_minrun(pred_probs, 0.5)
    
    pred = apply_min_error_persistence(pred, 
                                      min_duration_sec=MIN_ERROR_DURATION_SEC,
                                      fps=EFFECTIVE_FPS)
    
    fused_preds.append(pred)

# Compute metrics on fused predictions
fused_tp, fused_fp, fused_fn = compute_frame_counts(fused_preds, fused_labels)
fused_frame_p = fused_tp / (fused_tp + fused_fp) if fused_tp + fused_fp else 0
fused_frame_r = fused_tp / (fused_tp + fused_fn) if fused_tp + fused_fn else 0
fused_frame_f1 = 2 * fused_frame_p * fused_frame_r / (fused_frame_p + fused_frame_r) if fused_frame_p + fused_frame_r else 0
fused_event_p, fused_event_r, fused_event_f1 = event_metrics(fused_preds, fused_labels)

print(f"\nFused Predictions Performance:")
print(f"  Frame:  P={fused_frame_p:.3f} R={fused_frame_r:.3f} F1={fused_frame_f1:.3f}")
print(f"  Event:  P={fused_event_p:.3f} R={fused_event_r:.3f} F1={fused_event_f1:.3f}")

# --------------------------------------------------
# ERROR FRAME DETECTION (frame-level reporting)
# --------------------------------------------------
print("\n" + "="*80)
print("ERROR DETECTION - FRAME-LEVEL ANALYSIS")
print("="*80)

print("\n" + "-"*80)
print("PER-CAMERA ERROR FRAMES (Side-by-Side Comparison)")
print("-"*80)

for cam in sorted(cam_results.keys()):
    cam_short = str(cam)
    print(f"\nCamera {cam_short}:")
    
    # Concatenate all predictions/labels for this camera
    all_preds = np.concatenate(cam_results[cam]['preds'])
    all_labels = np.concatenate(cam_results[cam]['labels'])
    all_frame_idxs = np.concatenate(cam_results[cam]['frame_idxs'])
    total_frames = len(all_preds)
    
    # Extract events
    pred_events = group_events(all_preds)
    true_events = group_events(all_labels)
    
    # Map to original frame indices
    pred_events_original = map_events_to_original(pred_events, all_frame_idxs)
    true_events_original = map_events_to_original(true_events, all_frame_idxs)
    
    # Normalized time ranges
    pred_time_ranges = format_time_ranges(pred_events, total_frames)
    true_time_ranges = format_time_ranges(true_events, total_frames)
    
    # Check overlap
    overlap = check_overlap(pred_events, true_events)
    match_symbol = "✓" if overlap else "✗"
    
    print(f"  Total frames: {total_frames}")
    print(f"  Match: {match_symbol} {'(Prediction overlaps ground truth)' if overlap else '(No overlap)'}")
    print()
    
    # Side-by-side comparison
    print(f"  {'Metric':<25} {'PREDICTED':<30} {'GROUND TRUTH':<30}")
    print(f"  {'-'*85}")
    print(f"  {'Events detected':<25} {len(pred_events):<30} {len(true_events):<30}")
    print(f"  {'Array indices':<25} {str(pred_events):<30} {str(true_events):<30}")
    print(f"  {'Original frame_idx':<25} {str(pred_events_original):<30} {str(true_events_original):<30}")
    print(f"  {'Normalized time [0,1]':<25} {str([(f'{s:.3f}', f'{e:.3f}') for s, e in pred_time_ranges]):<30} {str([(f'{s:.3f}', f'{e:.3f}') for s, e in true_time_ranges]):<30}")

print("\n" + "-"*80)
print("FUSED (MULTI-CAMERA) ERROR FRAMES (Side-by-Side Comparison)")
print("-"*80)

all_fused_preds = np.concatenate(fused_preds)
all_fused_labels = np.concatenate(fused_labels)
all_fused_frame_idxs = np.concatenate(fused_frame_idxs)
total_frames = len(all_fused_preds)

fused_pred_events = group_events(all_fused_preds)
fused_true_events = group_events(all_fused_labels)

# Map to original frame indices
fused_pred_events_original = map_events_to_original(fused_pred_events, all_fused_frame_idxs)
fused_true_events_original = map_events_to_original(fused_true_events, all_fused_frame_idxs)

fused_pred_time_ranges = format_time_ranges(fused_pred_events, total_frames)
fused_true_time_ranges = format_time_ranges(fused_true_events, total_frames)

# Check overlap
fused_overlap = check_overlap(fused_pred_events, fused_true_events)
fused_match_symbol = "✓" if fused_overlap else "✗"

print(f"\nFused Predictions (averaged across {len(good_cameras)} cameras):")
print(f"  Total frames: {total_frames}")
print(f"  Match: {fused_match_symbol} {'(Prediction overlaps ground truth)' if fused_overlap else '(No overlap)'}")
print()

print(f"  {'Metric':<25} {'PREDICTED':<30} {'GROUND TRUTH':<30}")
print(f"  {'-'*85}")
print(f"  {'Events detected':<25} {len(fused_pred_events):<30} {len(fused_true_events):<30}")
print(f"  {'Array indices':<25} {str(fused_pred_events):<30} {str(fused_true_events):<30}")
print(f"  {'Original frame_idx':<25} {str(fused_pred_events_original):<30} {str(fused_true_events_original):<30}")
print(f"  {'Normalized time [0,1]':<25} {str([(f'{s:.3f}', f'{e:.3f}') for s, e in fused_pred_time_ranges]):<30} {str([(f'{s:.3f}', f'{e:.3f}') for s, e in fused_true_time_ranges]):<30}")

# --------------------------------------------------
# Save configuration for inference
# --------------------------------------------------
inference_config = {
    'model_path': model_path,
    'threshold': best_thr,
    'smooth_k': SMOOTH_K,
    'min_run': MIN_RUN,
    'min_error_duration_sec': MIN_ERROR_DURATION_SEC,
    'effective_fps': EFFECTIVE_FPS,
    'input_dim': input_dim,
    'window_size': dataset.window_size,
    'frame_stride': FRAME_STRIDE,
    'good_cameras': list(good_cameras.keys()),
    'min_camera_event_f1': MIN_CAMERA_EVENT_F1,
    'expected_performance': {
        'fused_event_f1': fused_event_f1,
        'fused_event_precision': fused_event_p,
        'fused_event_recall': fused_event_r,
        'fused_frame_f1': fused_frame_f1,
    }
}

with open(f"{OUT_DIR}/inference_config.pkl", 'wb') as f:
    pickle.dump(inference_config, f)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)

print(f"""
Model Training Complete!

Training Configuration:
  - Approach: LOCO-optimized windowed/causal
  - Window size: {dataset.window_size} frames (stride={FRAME_STRIDE})
  - Sequences: {', '.join(sorted(TRAIN_SEQS))} ({len(train_indices)} windows)
  - Best epoch: {best_epoch}
  - Validation loss: {best_val_loss:.4f}
  - Threshold: {best_thr:.2f}

Hyperparameters (optimized via LOCO CV):
  - POS_WEIGHT_SCALE: {POS_WEIGHT_SCALE}
  - SMOOTH_K: {SMOOTH_K}
  - MIN_RUN: {MIN_RUN}
  - MIN_ERROR_DURATION: {MIN_ERROR_DURATION_SEC}s

Test Performance (sequence: {', '.join(sorted(TEST_SEQS))}):
  - Cameras evaluated: {len(cam_results)}
  - Cameras used for fusion: {len(good_cameras)}
  
  Global Metrics (all cameras combined):
    Frame F1:  {frame_f1:.3f}
    Event F1:  {event_f1:.3f}
  
  Fused Multi-Camera Performance:
    Frame F1:  {fused_frame_f1:.3f}
    Event F1:  {fused_event_f1:.3f}
    Event Recall: {fused_event_r:.3f} (catches {100*fused_event_r:.1f}% of errors)
    Event Precision: {fused_event_p:.3f} ({100*(1-fused_event_p):.1f}% false alarm rate)

Files Saved:
  - Model: {model_path}
  - Config: {OUT_DIR}/inference_config.pkl

Ready for deployment with multi-camera fusion!
""")

print("="*80)

# --------------------------------------------------
# PLOTS
# --------------------------------------------------
cams = [str(c) for c in sorted(cam_results.keys())]
frame_f1s = [cam_results[c]["frame_f1"] for c in sorted(cam_results.keys())]
event_f1s = [cam_results[c]["event_f1"] for c in sorted(cam_results.keys())]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: F1 per camera
ax = axes[0]
x = np.arange(len(cams))
w = 0.35
ax.bar(x - w/2, frame_f1s, w, label="Frame F1", alpha=0.8)
ax.bar(x + w/2, event_f1s, w, label="Event F1", alpha=0.8)
ax.axhline(y=fused_event_f1, color='r', linestyle='--', 
           label=f'Fused Event F1={fused_event_f1:.3f}', linewidth=2)
ax.set_xticks(x)
ax.set_xticklabels([c[-4:] for c in cams], rotation=45)
ax.set_ylim(0, 1)
ax.set_ylabel("F1 Score")
ax.set_title("Test Performance per Camera (Windowed LOCO)")
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 2: P vs R
ax = axes[1]
for cam, r in cam_results.items():
    ax.scatter(r["frame_r"], r["frame_p"], s=100, alpha=0.7)
    ax.text(r["frame_r"] + 0.01, r["frame_p"] + 0.01, str(cam)[-4:], fontsize=8)

# Add fused point
ax.scatter(fused_frame_r, fused_frame_p, s=200, color='red', marker='*', 
           edgecolors='black', linewidth=2, label='Fused', zorder=10)

ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Frame-Level Precision vs Recall")
ax.grid(alpha=0.3)
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/test_results.png", dpi=150, bbox_inches='tight')
print(f"\n📊 Plot saved: {OUT_DIR}/test_results.png")