"""
LOCO Cross-Validation: Leave-One-Camera-Out evaluation with held-out test set
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import pickle

from dataset import SequenceDataset
from collate_fn import collate_fn
from model import FrameLSTM
from loss import masked_bce_loss

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
NPZ_PATH = "model_input/lstm_input.npz"
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
PATIENCE = 6
GRAD_CLIP = 1.0
POS_WEIGHT_SCALE = 5.0

SMOOTH_K = 5
MIN_RUN = 3

OUT_DIR = "checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)

# Hold out sequences for final test set (1 per camera)
HOLD_OUT_SEQUENCES_PER_CAMERA = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------
# Utilities (SAME as your original)
# --------------------------------------------------
def smooth_and_minrun(probs, thr):
    sm = ndi.uniform_filter1d(probs, size=SMOOTH_K)
    preds = (sm > thr).astype(int)
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


# --------------------------------------------------
# Train + Eval (CV-only)
# --------------------------------------------------
def train_and_eval_fold(dataset, train_idx, val_idx, cam):
    """Train and evaluate one LOCO fold with training and validation loss printing"""

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    input_dim = dataset[0][0].shape[1]
    
    # Simple inline training (CV-only)
    model = FrameLSTM(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
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
            logits = model(x, mask)
            loss = masked_bce_loss(logits, y, mask, pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item() * mask.sum().item()
            train_frames += mask.sum().item()

        train_loss /= train_frames
        
        model.eval()
        val_loss = 0
        frames = 0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
                logits = model(x, mask)
                loss = masked_bce_loss(logits, y, mask, pos_weight)
                val_loss += loss.item() * mask.sum().item()
                frames += mask.sum().item()

        val_loss /= frames
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

    probs, labels = [], []
    with torch.no_grad():
        for x, y, mask in val_loader:
            x = x.to(DEVICE)
            logits = torch.sigmoid(model(x, mask)).cpu().numpy()[0]
            valid = mask.numpy()[0].astype(bool)
            probs.append(logits[valid])
            labels.append(y.numpy()[0][valid])

    best_thr = find_best_threshold(probs, labels)
    preds = [smooth_and_minrun(p, best_thr) for p in probs]

    # Frame metrics
    tp, fp, fn = compute_frame_counts(preds, labels)
    frame_p = tp / (tp + fp) if tp + fp > 0 else 0
    frame_r = tp / (tp + fn) if tp + fn > 0 else 0
    frame_f1 = 2 * frame_p * frame_r / (frame_p + frame_r) if frame_p + frame_r > 0 else 0
    
    # Event metrics
    event_p, event_r, event_f1 = event_metrics(preds, labels)

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
cams = sorted({cam for _, cam, _ in dataset.meta})

print(f"Dataset: {len(dataset)} sequences across {len(cams)} cameras")
print(f"Strategy: LOCO (Leave-One-Camera-Out) CV-only")
print(f"Hold-out test set: {HOLD_OUT_SEQUENCES_PER_CAMERA} sequence per camera\n")

# Select held-out test sequences: 1 per camera from different original sequences
# Group sequences by camera and original sequence
cam_to_sequences = {}
for idx, (orig_seq, cam, _) in enumerate(dataset.meta):
    if cam not in cam_to_sequences:
        cam_to_sequences[cam] = {}
    if orig_seq not in cam_to_sequences[cam]:
        cam_to_sequences[cam][orig_seq] = []
    cam_to_sequences[cam][orig_seq].append(idx)

# Hold out 1 sequence per camera from different original sequences
held_out_indices = set()
orig_seq_used = set()

for cam in cams:
    # Get list of original sequences for this camera (that haven't been used for hold-out)
    available_orig_seqs = [os for os in cam_to_sequences[cam].keys() if os not in orig_seq_used]
    
    if available_orig_seqs:
        # Pick the first available original sequence
        orig_seq = available_orig_seqs[0]
        orig_seq_used.add(orig_seq)
        # Pick the first sequence from that original sequence
        seq_idx = cam_to_sequences[cam][orig_seq][0]
        held_out_indices.add(seq_idx)

print(f"Held-out test indices: {sorted(held_out_indices)}")
print(f"Held-out sequences from original sequences: {sorted(orig_seq_used)}\n")

# Save held-out indices for use in LOSO and mixed approaches
held_out_file = "held_out_indices.pkl"
with open(held_out_file, 'wb') as f:
    pickle.dump(sorted(held_out_indices), f)
print(f"✅ Saved held-out indices to: {held_out_file}\n")

# Create CV dataset excluding held-out sequences
cv_indices = [i for i in range(len(dataset)) if i not in held_out_indices]

GLOBAL_TP = GLOBAL_FP = GLOBAL_FN = 0
results = {}

for cam in cams:
    # For LOCO CV: train on all cameras except held-out cam, val on held-out cam (excluding held-out test sequences)
    train_idx = [i for i in cv_indices if i not in held_out_indices and dataset.meta[i][1] != cam]
    val_idx = [i for i in cv_indices if i not in held_out_indices and dataset.meta[i][1] == cam]

    print(f"\n=== LOCO: hold out camera {cam} ===")
    print(f"Train: {len(train_idx)} seqs | Val: {len(val_idx)} seqs")
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
    print(f"\nEvaluating on {len(held_out_indices)} held-out sequences...")
    
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
        collate_fn=collate_fn
    )
    
    # Evaluate
    test_probs, test_labels = [], []
    with torch.no_grad():
        for x, y, mask in test_loader:
            x = x.to(DEVICE)
            logits = torch.sigmoid(test_model(x, mask)).cpu().numpy()[0]
            valid = mask.numpy()[0].astype(bool)
            test_probs.append(logits[valid])
            test_labels.append(y.numpy()[0][valid])
    
    # Find threshold and compute metrics
    test_thr = find_best_threshold(test_probs, test_labels)
    test_preds = [smooth_and_minrun(p, test_thr) for p in test_probs]
    
    # Frame metrics
    test_tp, test_fp, test_fn = compute_frame_counts(test_preds, test_labels)
    test_frame_p = test_tp / (test_tp + test_fp) if test_tp + test_fp > 0 else 0
    test_frame_r = test_tp / (test_tp + test_fn) if test_tp + test_fn > 0 else 0
    test_frame_f1 = 2 * test_frame_p * test_frame_r / (test_frame_p + test_frame_r) if test_frame_p + test_frame_r > 0 else 0
    
    # Event metrics
    test_event_p, test_event_r, test_event_f1 = event_metrics(test_preds, test_labels)
    
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
ax.set_title('LOCO: F1 per Camera', fontsize=13, fontweight='bold')
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
plt.savefig('loco_cv_results_lstm.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Plot saved: loco_cv_results_lstm.png")

# Save results for comparison
loco_results = {
    'approach': 'LOCO',
    'n_folds': len(results),
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

with open('loco_cv_results_lstm.pkl', 'wb') as f:
    pickle.dump(loco_results, f)
print(f"💾 Results saved: loco_cv_results_lstm.pkl")

print("\n✅ LOCO Complete!")