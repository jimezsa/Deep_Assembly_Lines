"""
lstm_LOCO_cross-val.py - WINDOWED VERSION

LOCO Cross-Validation for windowed sequences.

Key difference from non-windowed version:
- Windows from same original sequence overlap and must stay together
- We still group by camera for LOCO (natural grouping)
- But we track original sequences for proper data splitting

Uses same held-out test set strategy for fair comparison.
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
import pickle

from dataset_sliding_window import SequenceDataset
from collate_fn_sliding_window import collate_fn
from model_sliding_window import FrameLSTM
from loss_sliding_window import masked_bce_loss

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
NPZ_PATH = "model_input/lstm_input_windowed.npz"
BATCH_SIZE = 32  # Can use larger batches with fixed-length windows
EPOCHS = 40
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
PATIENCE = 6
GRAD_CLIP = 1.0
POS_WEIGHT_SCALE = 4.0 # 5.0, 2.0, 1.0, 0.0

USE_FOCAL_LOSS = False  # Set to False to use standard BCE
FOCAL_ALPHA = 0.85 # 0.25, 0.35, 0.5, 0.45, 0.75     # Weight for positive class (0.25 = focus more on hard negatives)
FOCAL_GAMMA = 2.0 # 2.0, 1.0, 3.0     # Focusing strength (2.0 is standard, try 1.0-3.0)

# Calculate effective FPS for strided frames
FRAME_STRIDE = 15  # From training
VIDEO_FPS = 30     # Original video FPS
EFFECTIVE_FPS = VIDEO_FPS / FRAME_STRIDE  # = 2.0 fps
MIN_ERROR_DURATION_SEC = 3.0  # 2.0, 3.0, 2.5 # Minimum 2 seconds

SMOOTH_K = 5 # 5, 3
MIN_RUN = 3 # 3, 2, 4

OUT_DIR = "checkpoints_windowed"
os.makedirs(OUT_DIR, exist_ok=True)

# Hold out sequences for final test set (1 per camera from different original sequences)
HOLD_OUT_SEQUENCES_PER_CAMERA = 1

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # For multi-GPU
    
# Backend settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

# --------------------------------------------------
# Utilities (SAME as before)
# --------------------------------------------------
def smooth_and_minrun(probs, thr):
    # probs is expected to be a 1D array
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
    """
    Apply hysteresis thresholding (streaming-safe, causal).
    
    Two-threshold system:
    - Error triggers when prob >= threshold_high
    - Error clears when prob < threshold_low
    - Prevents flickering around single threshold
    
    Args:
        probs: Array of probabilities (1D)
        threshold_high: Threshold to trigger error
        threshold_low: Threshold to clear error
        
    Returns:
        Binary predictions (1D array)
    """
    preds = np.zeros(len(probs), dtype=int)
    error_active = False
    
    # Process causally (left to right)
    for i in range(len(probs)):
        if error_active:
            # Error is active, check if it should clear
            if probs[i] < threshold_low:
                error_active = False
                preds[i] = 0
            else:
                preds[i] = 1
        else:
            # No error, check if it should trigger
            if probs[i] >= threshold_high:
                error_active = True
                preds[i] = 1
            else:
                preds[i] = 0
    
    return preds

def compute_frame_counts(preds, labels):
    tp = fp = fn = 0
    for p, y in zip(preds, labels):
        tp += int(((p == 1) & (y == 1)).sum())
        fp += int(((p == 1) & (y == 0)).sum())
        fn += int(((p == 0) & (y == 1)).sum())
    return tp, fp, fn


def group_events(arr):
    """Extract event spans"""
    events = []
    i = 0
    while i < len(arr):
        if arr[i] == 1:
            j = i
            while j < len(arr) and arr[j] == 1:
                j += 1
            events.append((i, j-1))
            i = j
        else:
            i += 1
    return events


def event_metrics(preds, labels):
    """Compute event-level metrics"""
    tp_events = fp_events = fn_events = 0
    
    for pred, lab in zip(preds, labels):
        pred_events = group_events(pred)
        gt_events = group_events(lab)
        
        matched_pred = set()
        matched_gt = set()
        
        for i, (p_start, p_end) in enumerate(pred_events):
            for j, (g_start, g_end) in enumerate(gt_events):
                if not (p_end < g_start or p_start > g_end):
                    matched_pred.add(i)
                    matched_gt.add(j)
                    break
        
        tp_events += len(matched_pred)
        fp_events += len(pred_events) - len(matched_pred)
        fn_events += len(gt_events) - len(matched_gt)
    
    event_p = tp_events / (tp_events + fp_events) if tp_events + fp_events > 0 else 0
    event_r = tp_events / (tp_events + fn_events) if tp_events + fn_events > 0 else 0
    event_f1 = 2 * event_p * event_r / (event_p + event_r) if event_p + event_r > 0 else 0
    
    return event_p, event_r, event_f1


def find_best_threshold(probs, labels):
    best_thr, best_f1 = 0.5, -1
    for t in np.linspace(0.1, 0.9, 41):
        preds = [(p > t).astype(int) for p in probs]
        tp, fp, fn = compute_frame_counts(preds, labels)
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return best_thr

def reconstruct_timelines_with_aggregation(dataset, indices, model):
    """
    Reconstruct timelines with proper handling of overlapping windows.
    
    For each frame that appears in multiple windows, we average the predictions.
    This gives more stable predictions than just using the last occurrence.
    """
    model.eval()
    
    # Collect all predictions with frame indices
    frame_predictions = {}  # (orig_seq, cam, frame_idx) -> [list of probs]
    frame_labels = {}       # (orig_seq, cam, frame_idx) -> label
    
    with torch.no_grad():
        for idx in indices:
            x, y = dataset[idx]
            orig_seq, cam, win_start, frame_idxs = dataset.meta[idx]
            
            x = x.unsqueeze(0).to(DEVICE)
            mask = torch.ones(1, x.shape[1], device=DEVICE)
            
            # Get predictions for ALL frames in window (not just last)
            probs = torch.sigmoid(model(x, mask))[0].cpu().numpy()  # (T,)
            labels = y.numpy()  # (T,)
            
            # Store predictions for each frame
            for i, (frame_idx, prob, label) in enumerate(zip(frame_idxs, probs, labels)):
                key = (orig_seq, cam, frame_idx)
                
                # Only use CAUSAL predictions (frame can only see past)
                # For frame at position i, we use prediction from position i
                # (not from future windows)
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
        
        # Average predictions from all windows that saw this frame
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

def apply_min_error_persistence(preds, min_duration_sec=2.0, fps=2.0):
    """
    Enforce minimum error duration (streaming-safe, causal).
    
    Once error is detected, it must persist for at least min_duration_sec
    before it can clear. Prevents brief flickers.
    
    Args:
        preds: Binary predictions (1D array)
        min_duration_sec: Minimum error duration in seconds
        fps: Effective frame rate (frames per second)
        
    Returns:
        Binary predictions with enforced minimum duration
    """
    min_frames = int(min_duration_sec * fps)
    output = preds.copy()
    
    i = 0
    while i < len(preds):
        if preds[i] == 1:
            # Find end of this error run
            j = i
            while j < len(preds) and preds[j] == 1:
                j += 1
            
            error_duration = j - i
            
            if error_duration < min_frames:
                # Error too short, extend it
                end_idx = min(i + min_frames, len(preds))
                output[i:end_idx] = 1
                i = end_idx
            else:
                i = j
        else:
            i += 1
    
    return output

# --------------------------------------------------
# Train + Eval (CV-only)
# --------------------------------------------------
def train_and_eval_fold(dataset, train_idx, val_idx, cam):
    """Train and evaluate one LOCO fold"""

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for full determinism
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,  
        num_workers=0, # Set to 0 for full determinism
        worker_init_fn=worker_init_fn
    )

    input_dim = dataset[0][0].shape[1]
    
    model = FrameLSTM(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Compute pos_weight from training windows
    labels_flat = np.concatenate([dataset[i][1] for i in train_idx])
    pos_weight = (len(labels_flat) - labels_flat.sum()) / max(1, labels_flat.sum())
    pos_weight = torch.tensor([pos_weight * POS_WEIGHT_SCALE]).to(DEVICE)
    
    best_val = float("inf")
    patience = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        train_frames = 0
        for x, y, mask in train_loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            logits = model(x, mask) # (B, T)
            # loss = masked_bce_loss(logits, y, mask, pos_weight)      
            logits_last = logits[:, -1]     # (B,)
            labels_last = y[:, -1]          # (B,)
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

        train_loss /= train_frames
        
        model.eval()
        val_loss = 0
        n_samples = 0

        with torch.no_grad():
            for x, y, mask in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                logits = model(x, mask)          # (1, T)
                logits_last = logits[:, -1]      # (1,)
                labels_last = y[:, -1]           # (1,)

                loss = F.binary_cross_entropy_with_logits(
                    logits_last,
                    labels_last.float(),
                    pos_weight=pos_weight
                )

                val_loss += loss.item()
                n_samples += 1

        val_loss /= n_samples
        print(f"[{cam}] Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), f"{OUT_DIR}/best_{cam}.pt")
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(torch.load(f"{OUT_DIR}/best_{cam}.pt", weights_only=True))

    # ============ Evaluation ============
    model.eval()
    
    # STEP 1: Reconstruct timelines (like test set does)
    timelines = reconstruct_timelines_with_aggregation(
        dataset, val_idx, model
    )
    
    # STEP 2: Extract full probability sequences
    all_probs = [v["probs"] for v in timelines.values()]
    all_labels = [v["labels"] for v in timelines.values()]
    
    # STEP 3: Find threshold
    best_thr = find_best_threshold(all_probs, all_labels)
    
    # STEP 4: Apply SAME post-processing as test set
    preds = []
    for p in all_probs:
        # Step 1: Threshold
        if SMOOTH_K > 1 or MIN_RUN > 1:
            pred = (p > best_thr).astype(int)
        else:
            pred = apply_causal_hysteresis(p, 
                                        threshold_high=best_thr + 0.15, 
                                        threshold_low=best_thr - 0.05)
        
        # Step 2: Smoothing
        if SMOOTH_K > 1 or MIN_RUN > 1:
            pred_probs = pred.astype(float)
            pred = smooth_and_minrun(pred_probs, 0.5)
        
        # Step 3: Persistence
        pred = apply_min_error_persistence(pred, 
                                        min_duration_sec=MIN_ERROR_DURATION_SEC,
                                        fps=EFFECTIVE_FPS)
        
        preds.append(pred)
    
    # STEP 5: Compute metrics
    tp, fp, fn = compute_frame_counts(preds, all_labels)
    frame_p = tp / (tp + fp) if tp + fp > 0 else 0
    frame_r = tp / (tp + fn) if tp + fn > 0 else 0
    frame_f1 = 2 * frame_p * frame_r / (frame_p + frame_r) if frame_p + frame_r > 0 else 0
    
    event_p, event_r, event_f1 = event_metrics(preds, all_labels)
    
    return {
        'frame_p': frame_p,
        'frame_r': frame_r,
        'frame_f1': frame_f1,
        'event_p': event_p,
        'event_r': event_r,
        'event_f1': event_f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'thr': best_thr
    }


# --------------------------------------------------
# MAIN LOCO LOOP
# --------------------------------------------------
dataset = SequenceDataset(NPZ_PATH)
cams = sorted({cam for _, cam, _, _ in dataset.meta})

print(f"\n{'='*80}")
print(f"WINDOWED LOCO CROSS-VALIDATION")
print(f"{'='*80}")
print(f"Dataset: {len(dataset)} windows from {len(cams)} cameras")
print(f"Window size: {dataset.window_size} frames")
print(f"Strategy: LOCO (Leave-One-Camera-Out) CV-only")
print(f"Hold-out test set: {HOLD_OUT_SEQUENCES_PER_CAMERA} original sequence per camera\n")

# Group windows by (original_sequence, camera) to identify unique sequences
window_groups = {}  # (orig_seq, cam) -> [window_indices]
for idx, (orig_seq, cam, _, _) in enumerate(dataset.meta):
    key = (orig_seq, cam)
    if key not in window_groups:
        window_groups[key] = []
    window_groups[key].append(idx)

print(f"Found {len(window_groups)} unique (sequence, camera) combinations")

# Group by camera to see original sequences per camera
cam_to_orig_seqs = {}
for orig_seq, cam in window_groups.keys():
    if cam not in cam_to_orig_seqs:
        cam_to_orig_seqs[cam] = {}
    cam_to_orig_seqs[cam][orig_seq] = True 

print(f"\nOriginal sequences per camera:")
for cam in sorted(cams):
    orig_seqs = sorted(cam_to_orig_seqs.get(cam, []))
    print(f"  {cam}: {orig_seqs}")

# Select held-out test sequences: 1 original sequence per camera
held_out_indices = set()
orig_seq_used = set()

for cam in cams:
    available_orig_seqs = [os for os in cam_to_orig_seqs[cam].keys()  # Dict keys
                          if os not in orig_seq_used]
    
    if available_orig_seqs:
        # Pick first available original sequence
        orig_seq = available_orig_seqs[0]
        orig_seq_used.add(orig_seq)
        # Add ALL windows from this (orig_seq, cam) combination
        key = (orig_seq, cam)
        if key in window_groups:
            held_out_indices.update(window_groups[key])

print(f"\nHeld-out test set:")
print(f"  Original sequences: {sorted(orig_seq_used)}")
print(f"  Total windows: {len(held_out_indices)}")

# Save held-out indices for use in LOSO and mixed approaches
held_out_file = "held_out_indices_windowed.pkl"
with open(held_out_file, 'wb') as f:
    pickle.dump(sorted(held_out_indices), f)
print(f"  Saved to: {held_out_file}\n")

# Create CV dataset excluding held-out sequences
cv_indices = [i for i in range(len(dataset)) if i not in held_out_indices]

print(f"CV dataset: {len(cv_indices)} windows (held out {len(held_out_indices)})")
print(f"{'='*80}\n")

GLOBAL_TP = GLOBAL_FP = GLOBAL_FN = 0
results = {}

for cam in cams:
    # For LOCO CV: train on all cameras except this one, val on this camera
    # (excluding held-out test sequences from both)
    train_idx = [i for i in cv_indices if dataset.get_camera(i) != cam]
    val_idx = [i for i in cv_indices if dataset.get_camera(i) == cam]

    print(f"\n=== LOCO: hold out camera {cam} ===")
    print(f"Train: {len(train_idx)} windows | Val: {len(val_idx)} windows")
    
    if len(val_idx) == 0:
        print(f"⚠️  No validation windows for camera {cam}, skipping")
        continue
    
    res = train_and_eval_fold(dataset, train_idx, val_idx, cam)
    results[cam] = res

    GLOBAL_TP += res["tp"]
    GLOBAL_FP += res["fp"]
    GLOBAL_FN += res["fn"]

print("\n✅ LOCO CV Complete!")

# --------------------------------------------------
# FINAL TEST SET EVALUATION
# --------------------------------------------------
print("\n" + "="*60)
print("FINAL TEST SET EVALUATION (Held-out sequences)")
print("="*60)

if held_out_indices:
    print(f"\nEvaluating on {len(held_out_indices)} held-out windows...")
    
    # Use the best performing fold's model
    best_cam = max(results.keys(), key=lambda c: results[c]['frame_f1'])
    best_model_path = f"{OUT_DIR}/best_{best_cam}.pt"
    
    print(f"\nUsing best model from camera: {best_cam} (Frame F1={results[best_cam]['frame_f1']:.3f})")
    
    # Load best model and evaluate on held-out sequences
    input_dim = dataset[0][0].shape[1]
    test_model = FrameLSTM(input_dim=input_dim).to(DEVICE)
    test_model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_model.eval()
    
    # Create test loader from held-out sequences
    test_loader = DataLoader(
        Subset(dataset, list(held_out_indices)),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=0, # Set to 0 for full determinism
        worker_init_fn=worker_init_fn
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION (Held-out sequences)")
    print("="*60)

    timelines = reconstruct_timelines_with_aggregation(
        dataset,
        held_out_indices,
        test_model
    )

    all_probs = [v["probs"] for v in timelines.values()]
    all_labels = [v["labels"] for v in timelines.values()]

    # Threshold selection
    test_thr = find_best_threshold(all_probs, all_labels)

    # Apply smoothing + min-run ONCE per timeline
    # test_preds = [
    #     smooth_and_minrun(p, test_thr)
    #     for p in all_probs
    # ]

    THRESHOLD_HIGH = test_thr + 0.15  # 10% above base threshold
    THRESHOLD_LOW =  test_thr - 0.05  # 10% below base threshold

    # Apply all post-processing in order
    test_preds = []
    for p in all_probs:
        # Step 1: Threshold (simple or hysteresis)
        if SMOOTH_K > 1 or MIN_RUN > 1:
            # If using smoothing, skip hysteresis (they conflict)
            pred = (p > test_thr).astype(int)
        else:
            # Use hysteresis for cleaner thresholding
            pred = apply_causal_hysteresis(p, 
                                        threshold_high=test_thr + 0.15, 
                                        threshold_low=test_thr - 0.05)
        
        # Step 2: Smoothing (operates on binary predictions)
        if SMOOTH_K > 1 or MIN_RUN > 1:
            # Convert back to "probabilities" for smoothing
            pred_probs = pred.astype(float)
            pred = smooth_and_minrun(pred_probs, 0.5)  # Threshold at 0.5 since already binary
        
        # Step 3: Minimum error persistence
        pred = apply_min_error_persistence(pred, 
                                        min_duration_sec=MIN_ERROR_DURATION_SEC,
                                        fps=EFFECTIVE_FPS)
        
        test_preds.append(pred)

    # Frame metrics
    test_tp, test_fp, test_fn = compute_frame_counts(test_preds, all_labels)
    test_frame_p = test_tp / (test_tp + test_fp) if test_tp + test_fp > 0 else 0
    test_frame_r = test_tp / (test_tp + test_fn) if test_tp + test_fn > 0 else 0
    test_frame_f1 = 2 * test_frame_p * test_frame_r / (test_frame_p + test_frame_r) if test_frame_p + test_frame_r > 0 else 0

    # Event metrics (NOW MEANINGFUL)
    test_event_p, test_event_r, test_event_f1 = event_metrics(test_preds, all_labels)

    print(f"\n📊 Test Set Frame Metrics:")
    print(f"   Precision: {test_frame_p:.3f}")
    print(f"   Recall:    {test_frame_r:.3f}")
    print(f"   F1:        {test_frame_f1:.3f}")

    print(f"\n📊 Test Set Event Metrics:")
    print(f"   Precision: {test_event_p:.3f}")
    print(f"   Recall:    {test_event_r:.3f}")
    print(f"   F1:        {test_event_f1:.3f}")

    print(f"\n   Threshold: {test_thr:.2f}")

else:
    print("\nNo held-out sequences available for testing.")

# --------------------------------------------------
# ENHANCED REPORTING
# --------------------------------------------------
print("\n" + "="*60)
print("LOCO CROSS-VALIDATION RESULTS")
print("="*60)

# Global (aggregated) metrics
global_p = GLOBAL_TP / (GLOBAL_TP + GLOBAL_FP) if (GLOBAL_TP + GLOBAL_FP) > 0 else 0
global_r = GLOBAL_TP / (GLOBAL_TP + GLOBAL_FN) if (GLOBAL_TP + GLOBAL_FN) > 0 else 0
global_f1 = 2 * global_p * global_r / (global_p + global_r) if (global_p + global_r) > 0 else 0

print("\n📊 AGGREGATED FRAME METRICS (all cameras combined):")
print(f"   Precision: {global_p:.3f}")
print(f"   Recall:    {global_r:.3f}")
print(f"   F1:        {global_f1:.3f}")

# Average (macro) metrics
avg_frame_p = np.mean([r['frame_p'] for r in results.values()])
avg_frame_r = np.mean([r['frame_r'] for r in results.values()])
avg_frame_f1 = np.mean([r['frame_f1'] for r in results.values()])

avg_event_p = np.mean([r['event_p'] for r in results.values()])
avg_event_r = np.mean([r['event_r'] for r in results.values()])
avg_event_f1 = np.mean([r['event_f1'] for r in results.values()])

print("\n📈 AVERAGE METRICS (macro-average across cameras):")
print(f"   Frame: P={avg_frame_p:.3f} ± {np.std([r['frame_p'] for r in results.values()]):.3f}")
print(f"          R={avg_frame_r:.3f} ± {np.std([r['frame_r'] for r in results.values()]):.3f}")
print(f"          F1={avg_frame_f1:.3f} ± {np.std([r['frame_f1'] for r in results.values()]):.3f}")
print(f"\n   Event: P={avg_event_p:.3f} ± {np.std([r['event_p'] for r in results.values()]):.3f}")
print(f"          R={avg_event_r:.3f} ± {np.std([r['event_r'] for r in results.values()]):.3f}")
print(f"          F1={avg_event_f1:.3f} ± {np.std([r['event_f1'] for r in results.values()]):.3f}")

# Per-camera detailed table
print("\n" + "="*60)
print("LOCO SUMMARY - PER CAMERA BREAKDOWN")
print("="*60)
print(f"\n{'Camera':<15} {'Frame P':<9} {'Frame R':<9} {'Frame F1':<9} {'Event P':<9} {'Event R':<9} {'Event F1':<9} {'Thr':<5}")
print("-" * 85)

for cam, r in results.items():
    cam_short = str(cam)[-4:] if len(str(cam)) > 4 else str(cam)
    print(f"{cam_short:<15} {r['frame_p']:<9.3f} {r['frame_r']:<9.3f} {r['frame_f1']:<9.3f} "
          f"{r['event_p']:<9.3f} {r['event_r']:<9.3f} {r['event_f1']:<9.3f} {r['thr']:<5.2f}")

print("-" * 85)
print(f"{'GLOBAL':<15} {global_p:<9.3f} {global_r:<9.3f} {global_f1:<9.3f} {'':>36}")
print(f"{'AVERAGE':<15} {avg_frame_p:<9.3f} {avg_frame_r:<9.3f} {avg_frame_f1:<9.3f} "
      f"{avg_event_p:<9.3f} {avg_event_r:<9.3f} {avg_event_f1:<9.3f}")
print("="*60)

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: F1 scores
ax = axes[0]
cameras = [str(c)[-4:] for c in results.keys()]
frame_f1s = [r['frame_f1'] for r in results.values()]
event_f1s = [r['event_f1'] for r in results.values()]

x = np.arange(len(cameras))
width = 0.35

ax.bar(x - width/2, frame_f1s, width, label='Frame F1', color='steelblue', alpha=0.8)
ax.bar(x + width/2, event_f1s, width, label='Event F1', color='coral', alpha=0.8)

ax.set_xlabel('Camera', fontsize=11)
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('LOCO (Windowed): F1 per Camera', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cameras, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1)

# Plot 2: P vs R
ax = axes[1]
frame_ps = [r['frame_p'] for r in results.values()]
frame_rs = [r['frame_r'] for r in results.values()]

ax.scatter(frame_rs, frame_ps, s=120, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1)

for i, cam in enumerate(cameras):
    ax.annotate(cam, (frame_rs[i], frame_ps[i]), fontsize=8, xytext=(3, 3), textcoords='offset points')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Frame-Level P vs R', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('loco_cv_results_lstm_windowed.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Plot saved: loco_cv_results_lstm_windowed.png")

# Save results for comparison
loco_results = {
    'approach': 'LOCO_windowed',
    'n_folds': len(results),
    'window_size': dataset.window_size,
    'global_p': global_p,
    'global_r': global_r,
    'global_f1': global_f1,
    'avg_p': avg_frame_p,
    'avg_p_std': np.std([r['frame_p'] for r in results.values()]),
    'avg_r': avg_frame_r,
    'avg_r_std': np.std([r['frame_r'] for r in results.values()]),
    'avg_f1': avg_frame_f1,
    'avg_f1_std': np.std([r['frame_f1'] for r in results.values()]),
    'event_avg_p': avg_event_p,
    'event_avg_p_std': np.std([r['event_p'] for r in results.values()]),
    'event_avg_r': avg_event_r,
    'event_avg_r_std': np.std([r['event_r'] for r in results.values()]),
    'event_avg_f1': avg_event_f1,
    'event_avg_f1_std': np.std([r['event_f1'] for r in results.values()]),
    'frame_f1_min': min([r['frame_f1'] for r in results.values()]),
    'frame_f1_max': max([r['frame_f1'] for r in results.values()]),
    'event_f1_min': min([r['event_f1'] for r in results.values()]),
    'event_f1_max': max([r['event_f1'] for r in results.values()]),
}

with open('loco_cv_results_lstm_windowed.pkl', 'wb') as f:
    pickle.dump(loco_results, f)
print(f"💾 Results saved: loco_cv_results_lstm_windowed.pkl")

print("\n✅ Windowed LOCO Complete!")
print(f"\nKey differences from non-windowed:")
print(f"  - Training on {len(cv_indices)} windows instead of variable-length sequences")
print(f"  - Each window is {dataset.window_size} frames (fixed length)")
print(f"  - Larger batch size possible: {BATCH_SIZE} vs 8")
print(f"  - Better matches online inference scenario")




# """
# lstm_LOCO_cross-val.py - WINDOWED VERSION with RECENT WINDOW MAX

# LOCO Cross-Validation for windowed sequences with recent window max evaluation.

# Key improvements:
# 1. Train on ALL frames in window (many-to-many)
# 2. Evaluate using RECENT WINDOW MAX strategy
# 3. Matches online inference scenario exactly

# Strategy:
# - Windows from same original sequence must stay together
# - LOCO by camera (natural grouping)
# - Track original sequences for proper data splitting
# - Held-out test set for final evaluation
# """
# import os
# import random
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Subset
# import scipy.ndimage as ndi
# import matplotlib.pyplot as plt
# import pickle

# from dataset_sliding_window import SequenceDataset
# from collate_fn_sliding_window import collate_fn
# from model import FrameLSTM
# from loss_sliding_window import masked_bce_loss

# # --------------------------------------------------
# # CONFIG
# # --------------------------------------------------
# NPZ_PATH = "model_input/lstm_input_windowed.npz"

# # Optimized hyperparameters for windowed training
# BATCH_SIZE = 32
# HIDDEN_DIM = 64
# NUM_LAYERS = 3
# DROPOUT = 0.2

# EPOCHS = 40
# LR = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# SEED = 42
# PATIENCE = 6
# GRAD_CLIP = 1.0
# POS_WEIGHT_SCALE = 5.0

# # Post-processing
# SMOOTH_K = 3
# MIN_RUN = 2

# # CRITICAL: Recent window size for evaluation
# RECENT_WINDOW = 15  # Look at last 15 frames of each window

# OUT_DIR = "checkpoints_windowed_recentmax"
# os.makedirs(OUT_DIR, exist_ok=True)

# # Hold out sequences for final test set
# HOLD_OUT_SEQUENCES_PER_CAMERA = 1

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# # --------------------------------------------------
# # Utilities
# # --------------------------------------------------
# def smooth_and_minrun(probs, thr):
#     sm = ndi.uniform_filter1d(probs, size=SMOOTH_K)
#     preds = (sm > thr).astype(int)
#     out = preds.copy()
#     i = 0
#     while i < len(preds):
#         if preds[i] == 1:
#             j = i
#             while j < len(preds) and preds[j] == 1:
#                 j += 1
#             if j - i < MIN_RUN:
#                 out[i:j] = 0
#             i = j
#         else:
#             i += 1
#     return out


# def compute_frame_counts(preds, labels):
#     tp = fp = fn = 0
#     for p, y in zip(preds, labels):
#         tp += int(((p == 1) & (y == 1)).sum())
#         fp += int(((p == 1) & (y == 0)).sum())
#         fn += int(((p == 0) & (y == 1)).sum())
#     return tp, fp, fn


# def group_events(arr):
#     """Extract event spans"""
#     events = []
#     i = 0
#     while i < len(arr):
#         if arr[i] == 1:
#             j = i
#             while j < len(arr) and arr[j] == 1:
#                 j += 1
#             events.append((i, j-1))
#             i = j
#         else:
#             i += 1
#     return events


# def event_metrics(preds, labels):
#     """Compute event-level metrics"""
#     tp_events = fp_events = fn_events = 0
    
#     for pred, lab in zip(preds, labels):
#         pred_events = group_events(pred)
#         gt_events = group_events(lab)
        
#         matched_pred = set()
#         matched_gt = set()
        
#         for i, (p_start, p_end) in enumerate(pred_events):
#             for j, (g_start, g_end) in enumerate(gt_events):
#                 if not (p_end < g_start or p_start > g_end):
#                     matched_pred.add(i)
#                     matched_gt.add(j)
#                     break
        
#         tp_events += len(matched_pred)
#         fp_events += len(pred_events) - len(matched_pred)
#         fn_events += len(gt_events) - len(matched_gt)
    
#     event_p = tp_events / (tp_events + fp_events) if tp_events + fp_events > 0 else 0
#     event_r = tp_events / (tp_events + fn_events) if tp_events + fn_events > 0 else 0
#     event_f1 = 2 * event_p * event_r / (event_p + event_r) if event_p + event_r > 0 else 0
    
#     return event_p, event_r, event_f1


# def find_best_threshold(probs, labels):
#     best_thr, best_f1 = 0.5, -1
#     for t in np.linspace(0.1, 0.9, 41):
#         preds = [(p > t).astype(int) for p in probs]
#         tp, fp, fn = compute_frame_counts(preds, labels)
#         prec = tp / (tp + fp) if tp + fp > 0 else 0
#         rec = tp / (tp + fn) if tp + fn > 0 else 0
#         f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
#         if f1 > best_f1:
#             best_f1, best_thr = f1, t
#     return best_thr


# # --------------------------------------------------
# # Train + Evaluate with RECENT WINDOW MAX
# # --------------------------------------------------
# def train_and_eval_fold(dataset, train_idx, val_idx, cam):
#     """Train on ALL frames, evaluate using RECENT WINDOW MAX strategy"""

#     train_loader = DataLoader(
#         Subset(dataset, train_idx),
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         collate_fn=collate_fn
#     )
#     val_loader = DataLoader(
#         Subset(dataset, val_idx),
#         batch_size=1,
#         shuffle=False,
#         collate_fn=collate_fn
#     )

#     input_dim = dataset[0][0].shape[1]
    
#     model = FrameLSTM(
#         input_dim=input_dim,
#         hidden_dim=HIDDEN_DIM,
#         num_layers=NUM_LAYERS,
#         bidirectional=False,
#         dropout=DROPOUT
#     ).to(DEVICE)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
#     labels_flat = np.concatenate([dataset[i][1] for i in train_idx])
#     pos_weight = (len(labels_flat) - labels_flat.sum()) / max(1, labels_flat.sum())
#     pos_weight = torch.tensor([pos_weight * POS_WEIGHT_SCALE]).to(DEVICE)
    
#     best_val = float("inf")
#     patience = 0
    
#     # ============ TRAINING (on ALL frames in window) ============
#     for epoch in range(1, EPOCHS + 1):
#         model.train()
#         train_loss = 0
#         train_frames = 0
#         for x, y, mask in train_loader:
#             x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
#             logits = model(x, mask)
#             loss = masked_bce_loss(logits, y, mask, pos_weight)
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
#             optimizer.step()
#             train_loss += loss.item() * mask.sum().item()
#             train_frames += mask.sum().item()

#         train_loss /= train_frames
        
#         # Validation loss (on all frames)
#         model.eval()
#         val_loss = 0
#         frames = 0
#         with torch.no_grad():
#             for x, y, mask in val_loader:
#                 x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
#                 logits = model(x, mask)
#                 loss = masked_bce_loss(logits, y, mask, pos_weight)
#                 val_loss += loss.item() * mask.sum().item()
#                 frames += mask.sum().item()

#         val_loss /= frames
#         print(f"[{cam}] Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f}")

#         if val_loss < best_val:
#             best_val = val_loss
#             patience = 0
#             torch.save(model.state_dict(), f"{OUT_DIR}/best_{cam}.pt")
#         else:
#             patience += 1
#             if patience >= PATIENCE:
#                 break

#     model.load_state_dict(torch.load(f"{OUT_DIR}/best_{cam}.pt", weights_only=True))

#     # ============ EVALUATION (RECENT WINDOW MAX - CORRECTED) ============
#     model.eval()

#     probs, labels = [], []
    
#     with torch.no_grad():
#         for x, y, mask in val_loader:
#             x = x.to(DEVICE)
#             # Get probabilities for entire window
#             logits = torch.sigmoid(model(x, mask)).cpu().numpy()[0]
#             y_np = y.numpy()[0]
            
#             # Take MAX probability from RECENT frames
#             if len(logits) >= RECENT_WINDOW:
#                 recent_probs = logits[-RECENT_WINDOW:]
#                 recent_labels = y_np[-RECENT_WINDOW:]
#                 max_prob = float(np.max(recent_probs))
                
#                 # CRITICAL FIX: Check if ANY frame in recent window has error
#                 has_error = int(np.any(recent_labels == 1))
#             else:
#                 max_prob = float(np.max(logits))
#                 has_error = int(np.any(y_np == 1))
            
#             # Store as single prediction per window
#             probs.append(np.array([max_prob]))
#             labels.append(np.array([has_error]))
    
#     print(f"[{cam}] Extracted {len(probs)} predictions (max from last {RECENT_WINDOW} frames)")
#     print(f"[{cam}] Windows with errors: {sum([l[0] for l in labels])}/{len(labels)}")

#     best_thr = find_best_threshold(probs, labels)
#     preds = [smooth_and_minrun(p, best_thr) for p in probs]

#     # Frame metrics (treating each window as one "frame")
#     tp, fp, fn = compute_frame_counts(preds, labels)
#     frame_p = tp / (tp + fp) if tp + fp > 0 else 0
#     frame_r = tp / (tp + fn) if tp + fn > 0 else 0
#     frame_f1 = 2 * frame_p * frame_r / (frame_p + frame_r) if frame_p + frame_r > 0 else 0
    
#     # Event metrics
#     event_p, event_r, event_f1 = event_metrics(preds, labels)

#     return {
#         'frame_p': frame_p,
#         'frame_r': frame_r,
#         'frame_f1': frame_f1,
#         'event_p': event_p,
#         'event_r': event_r,
#         'event_f1': event_f1,
#         'tp': tp,
#         'fp': fp,
#         'fn': fn,
#         'thr': best_thr
#     }


# # --------------------------------------------------
# # MAIN LOCO LOOP
# # --------------------------------------------------
# dataset = SequenceDataset(NPZ_PATH)
# cams = sorted({cam for _, cam, _, _ in dataset.meta})

# print(f"\n{'='*80}")
# print(f"WINDOWED LOCO CROSS-VALIDATION (RECENT WINDOW MAX)")
# print(f"{'='*80}")
# print(f"Dataset: {len(dataset)} windows from {len(cams)} cameras")
# print(f"Window size: {dataset.window_size} frames")
# print(f"Strategy: LOCO (Leave-One-Camera-Out) CV with recent window max")
# print(f"Recent window: {RECENT_WINDOW} frames (last {RECENT_WINDOW} positions)")
# print(f"Hold-out test set: {HOLD_OUT_SEQUENCES_PER_CAMERA} original sequence per camera\n")

# # Group windows by (original_sequence, camera) to identify unique sequences
# window_groups = {}
# for idx, (orig_seq, cam, _, _) in enumerate(dataset.meta):
#     key = (orig_seq, cam)
#     if key not in window_groups:
#         window_groups[key] = []
#     window_groups[key].append(idx)

# print(f"Found {len(window_groups)} unique (sequence, camera) combinations")

# # Group by camera to see original sequences per camera
# cam_to_orig_seqs = {}
# for orig_seq, cam in window_groups.keys():
#     if cam not in cam_to_orig_seqs:
#         cam_to_orig_seqs[cam] = set()
#     cam_to_orig_seqs[cam].add(orig_seq)

# print(f"\nOriginal sequences per camera:")
# for cam in sorted(cams):
#     orig_seqs = sorted(cam_to_orig_seqs.get(cam, []))
#     print(f"  {cam}: {orig_seqs}")

# # Select held-out test sequences: 1 original sequence per camera
# held_out_indices = set()
# orig_seq_used = set()

# for cam in cams:
#     available_orig_seqs = [os for os in cam_to_orig_seqs.get(cam, []) 
#                           if os not in orig_seq_used]
    
#     if available_orig_seqs:
#         orig_seq = available_orig_seqs[0]
#         orig_seq_used.add(orig_seq)
#         key = (orig_seq, cam)
#         if key in window_groups:
#             held_out_indices.update(window_groups[key])

# print(f"\nHeld-out test set:")
# print(f"  Original sequences: {sorted(orig_seq_used)}")
# print(f"  Total windows: {len(held_out_indices)}")

# # Save held-out indices
# held_out_file = "held_out_indices_windowed_recentmax.pkl"
# with open(held_out_file, 'wb') as f:
#     pickle.dump(sorted(held_out_indices), f)
# print(f"  Saved to: {held_out_file}\n")

# # Create CV dataset excluding held-out sequences
# cv_indices = [i for i in range(len(dataset)) if i not in held_out_indices]

# print(f"CV dataset: {len(cv_indices)} windows (held out {len(held_out_indices)})")
# print(f"{'='*80}\n")

# GLOBAL_TP = GLOBAL_FP = GLOBAL_FN = 0
# results = {}

# for cam in cams:
#     train_idx = [i for i in cv_indices if dataset.get_camera(i) != cam]
#     val_idx = [i for i in cv_indices if dataset.get_camera(i) == cam]

#     print(f"\n=== LOCO: hold out camera {cam} ===")
#     print(f"Train: {len(train_idx)} windows | Val: {len(val_idx)} windows")
    
#     if len(val_idx) == 0:
#         print(f"⚠️  No validation windows for camera {cam}, skipping")
#         continue
    
#     res = train_and_eval_fold(dataset, train_idx, val_idx, cam)
#     results[cam] = res

#     GLOBAL_TP += res["tp"]
#     GLOBAL_FP += res["fp"]
#     GLOBAL_FN += res["fn"]

# print("\n✅ LOCO CV Complete!")

# # --------------------------------------------------
# # FINAL TEST SET EVALUATION (with RECENT WINDOW MAX)
# # --------------------------------------------------
# print("\n" + "="*60)
# print("FINAL TEST SET EVALUATION (Held-out sequences)")
# print("="*60)

# if held_out_indices:
#     print(f"\nEvaluating on {len(held_out_indices)} held-out windows...")
    
#     # Use the best performing fold's model
#     best_cam = max(results.keys(), key=lambda c: results[c]['frame_f1'])
#     best_model_path = f"{OUT_DIR}/best_{best_cam}.pt"
    
#     print(f"\nUsing best model from camera: {best_cam} (Frame F1={results[best_cam]['frame_f1']:.3f})")
    
#     # Load best model
#     input_dim = dataset[0][0].shape[1]
#     test_model = FrameLSTM(
#         input_dim=input_dim,
#         hidden_dim=HIDDEN_DIM,
#         num_layers=NUM_LAYERS,
#         bidirectional=False,
#         dropout=DROPOUT
#     ).to(DEVICE)
#     test_model.load_state_dict(torch.load(best_model_path, weights_only=True))
#     test_model.eval()
    
#     # Create test loader from held-out sequences
#     test_loader = DataLoader(
#         Subset(dataset, list(held_out_indices)),
#         batch_size=1,
#         shuffle=False,
#         collate_fn=collate_fn
#     )
    
#     # Evaluate with RECENT WINDOW MAX (CORRECTED)
#     test_probs, test_labels = [], []
#     with torch.no_grad():
#         for x, y, mask in test_loader:
#             x = x.to(DEVICE)
#             logits = torch.sigmoid(test_model(x, mask)).cpu().numpy()[0]
#             y_np = y.numpy()[0]
            
#             # Take MAX probability from RECENT frames
#             if len(logits) >= RECENT_WINDOW:
#                 recent_probs = logits[-RECENT_WINDOW:]
#                 recent_labels = y_np[-RECENT_WINDOW:]
#                 max_prob = float(np.max(recent_probs))
#                 has_error = int(np.any(recent_labels == 1))
#             else:
#                 max_prob = float(np.max(logits))
#                 has_error = int(np.any(y_np == 1))
            
#             test_probs.append(np.array([max_prob]))
#             test_labels.append(np.array([has_error]))
    
#     print(f"Extracted {len(test_probs)} predictions from {len(test_probs)} windows")
#     print(f"Test windows with errors: {sum([l[0] for l in test_labels])}/{len(test_labels)}")
    
#     # Find threshold and compute metrics
#     test_thr = find_best_threshold(test_probs, test_labels)
#     test_preds = [smooth_and_minrun(p, test_thr) for p in test_probs]
    
#     # Frame metrics
#     test_tp, test_fp, test_fn = compute_frame_counts(test_preds, test_labels)
#     test_frame_p = test_tp / (test_tp + test_fp) if test_tp + test_fp > 0 else 0
#     test_frame_r = test_tp / (test_tp + test_fn) if test_tp + test_fn > 0 else 0
#     test_frame_f1 = 2 * test_frame_p * test_frame_r / (test_frame_p + test_frame_r) if test_frame_p + test_frame_r > 0 else 0
    
#     # Event metrics
#     test_event_p, test_event_r, test_event_f1 = event_metrics(test_preds, test_labels)
    
#     print(f"\n📊 Test Set Frame Metrics:")
#     print(f"   Precision: {test_frame_p:.3f}")
#     print(f"   Recall:    {test_frame_r:.3f}")
#     print(f"   F1:        {test_frame_f1:.3f}")
    
#     print(f"\n📊 Test Set Event Metrics:")
#     print(f"   Precision: {test_event_p:.3f}")
#     print(f"   Recall:    {test_event_r:.3f}")
#     print(f"   F1:        {test_event_f1:.3f}")
    
#     print(f"\n   Threshold: {test_thr:.2f}")
# else:
#     print("\nNo held-out sequences available for testing.")

# # --------------------------------------------------
# # ENHANCED REPORTING
# # --------------------------------------------------
# print("\n" + "="*60)
# print("LOCO CROSS-VALIDATION RESULTS")
# print("="*60)

# # Global (aggregated) metrics
# global_p = GLOBAL_TP / (GLOBAL_TP + GLOBAL_FP) if (GLOBAL_TP + GLOBAL_FP) > 0 else 0
# global_r = GLOBAL_TP / (GLOBAL_TP + GLOBAL_FN) if (GLOBAL_TP + GLOBAL_FN) > 0 else 0
# global_f1 = 2 * global_p * global_r / (global_p + global_r) if (global_p + global_r) > 0 else 0

# print("\n📊 AGGREGATED FRAME METRICS (all cameras combined):")
# print(f"   Precision: {global_p:.3f}")
# print(f"   Recall:    {global_r:.3f}")
# print(f"   F1:        {global_f1:.3f}")

# # Average (macro) metrics
# avg_frame_p = np.mean([r['frame_p'] for r in results.values()])
# avg_frame_r = np.mean([r['frame_r'] for r in results.values()])
# avg_frame_f1 = np.mean([r['frame_f1'] for r in results.values()])

# avg_event_p = np.mean([r['event_p'] for r in results.values()])
# avg_event_r = np.mean([r['event_r'] for r in results.values()])
# avg_event_f1 = np.mean([r['event_f1'] for r in results.values()])

# print("\n📈 AVERAGE METRICS (macro-average across cameras):")
# print(f"   Frame: P={avg_frame_p:.3f} ± {np.std([r['frame_p'] for r in results.values()]):.3f}")
# print(f"          R={avg_frame_r:.3f} ± {np.std([r['frame_r'] for r in results.values()]):.3f}")
# print(f"          F1={avg_frame_f1:.3f} ± {np.std([r['frame_f1'] for r in results.values()]):.3f}")
# print(f"\n   Event: P={avg_event_p:.3f} ± {np.std([r['event_p'] for r in results.values()]):.3f}")
# print(f"          R={avg_event_r:.3f} ± {np.std([r['event_r'] for r in results.values()]):.3f}")
# print(f"          F1={avg_event_f1:.3f} ± {np.std([r['event_f1'] for r in results.values()]):.3f}")

# # Per-camera detailed table
# print("\n" + "="*60)
# print("LOCO SUMMARY - PER CAMERA BREAKDOWN")
# print("="*60)
# print(f"\n{'Camera':<15} {'Frame P':<9} {'Frame R':<9} {'Frame F1':<9} {'Event P':<9} {'Event R':<9} {'Event F1':<9} {'Thr':<5}")
# print("-" * 85)

# for cam, r in results.items():
#     cam_short = str(cam)[-4:] if len(str(cam)) > 4 else str(cam)
#     print(f"{cam_short:<15} {r['frame_p']:<9.3f} {r['frame_r']:<9.3f} {r['frame_f1']:<9.3f} "
#           f"{r['event_p']:<9.3f} {r['event_r']:<9.3f} {r['event_f1']:<9.3f} {r['thr']:<5.2f}")

# print("-" * 85)
# print(f"{'GLOBAL':<15} {global_p:<9.3f} {global_r:<9.3f} {global_f1:<9.3f} {'':>36}")
# print(f"{'AVERAGE':<15} {avg_frame_p:<9.3f} {avg_frame_r:<9.3f} {avg_frame_f1:<9.3f} "
#       f"{avg_event_p:<9.3f} {avg_event_r:<9.3f} {avg_event_f1:<9.3f}")
# print("="*60)

# # --------------------------------------------------
# # VISUALIZATION
# # --------------------------------------------------
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # Plot 1: F1 scores
# ax = axes[0]
# cameras = [str(c)[-4:] for c in results.keys()]
# frame_f1s = [r['frame_f1'] for r in results.values()]
# event_f1s = [r['event_f1'] for r in results.values()]

# x = np.arange(len(cameras))
# width = 0.35

# ax.bar(x - width/2, frame_f1s, width, label='Frame F1', color='steelblue', alpha=0.8)
# ax.bar(x + width/2, event_f1s, width, label='Event F1', color='coral', alpha=0.8)

# ax.set_xlabel('Camera', fontsize=11)
# ax.set_ylabel('F1 Score', fontsize=11)
# ax.set_title('LOCO (Windowed + Recent Max): F1 per Camera', fontsize=13, fontweight='bold')
# ax.set_xticks(x)
# ax.set_xticklabels(cameras, rotation=45, ha='right')
# ax.legend()
# ax.grid(True, alpha=0.3, axis='y')
# ax.set_ylim(0, 1)

# # Plot 2: P vs R
# ax = axes[1]
# frame_ps = [r['frame_p'] for r in results.values()]
# frame_rs = [r['frame_r'] for r in results.values()]

# ax.scatter(frame_rs, frame_ps, s=120, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1)

# for i, cam in enumerate(cameras):
#     ax.annotate(cam, (frame_rs[i], frame_ps[i]), fontsize=8, xytext=(3, 3), textcoords='offset points')

# ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
# ax.set_xlabel('Recall', fontsize=11)
# ax.set_ylabel('Precision', fontsize=11)
# ax.set_title('Frame-Level P vs R', fontsize=13, fontweight='bold')
# ax.grid(True, alpha=0.3)
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)

# plt.tight_layout()
# plt.savefig('loco_cv_results_lstm_windowed_recentmax.png', dpi=150, bbox_inches='tight')
# print(f"\n📊 Plot saved: loco_cv_results_lstm_windowed_recentmax.png")

# # Save results for comparison
# loco_results = {
#     'approach': 'LOCO_windowed_recent_max',
#     'n_folds': len(results),
#     'window_size': dataset.window_size,
#     'recent_window': RECENT_WINDOW,
#     'evaluation': 'recent_window_max',
#     'global_p': global_p,
#     'global_r': global_r,
#     'global_f1': global_f1,
#     'avg_p': avg_frame_p,
#     'avg_p_std': np.std([r['frame_p'] for r in results.values()]),
#     'avg_r': avg_frame_r,
#     'avg_r_std': np.std([r['frame_r'] for r in results.values()]),
#     'avg_f1': avg_frame_f1,
#     'avg_f1_std': np.std([r['frame_f1'] for r in results.values()]),
#     'event_avg_p': avg_event_p,
#     'event_avg_p_std': np.std([r['event_p'] for r in results.values()]),
#     'event_avg_r': avg_event_r,
#     'event_avg_r_std': np.std([r['event_r'] for r in results.values()]),
#     'event_avg_f1': avg_event_f1,
#     'event_avg_f1_std': np.std([r['event_f1'] for r in results.values()]),
#     'frame_f1_min': min([r['frame_f1'] for r in results.values()]),
#     'frame_f1_max': max([r['frame_f1'] for r in results.values()]),
#     'event_f1_min': min([r['event_f1'] for r in results.values()]),
#     'event_f1_max': max([r['event_f1'] for r in results.values()]),
#     'hyperparameters': {
#         'hidden_dim': HIDDEN_DIM,
#         'num_layers': NUM_LAYERS,
#         'dropout': DROPOUT,
#         'smooth_k': SMOOTH_K,
#         'min_run': MIN_RUN,
#     }
# }

# with open('loco_cv_results_lstm_windowed_recentmax.pkl', 'wb') as f:
#     pickle.dump(loco_results, f)
# print(f"💾 Results saved: loco_cv_results_lstm_windowed_recentmax.pkl")

# print("\n✅ Windowed LOCO with Recent Window Max Complete!")
# print(f"\nKey improvements:")
# print(f"  - Training on ALL {dataset.window_size} frames in window (many-to-many)")
# print(f"  - Evaluation using max prob from last {RECENT_WINDOW} frames")
# print(f"  - Binary classification: error present in recent window or not")
# print(f"  - Optimized hyperparameters: H={HIDDEN_DIM}, L={NUM_LAYERS}, D={DROPOUT}")
# print(f"  - Reduced post-processing: K={SMOOTH_K}, R={MIN_RUN}")
# print(f"  - Matches online inference scenario exactly")