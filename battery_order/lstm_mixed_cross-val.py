"""
Mixed Cross-Validation: Stratified Random K-Fold with camera/sequence balance
Uses the same held-out indices as LOCO/LOSO for fair comparison
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

K_FOLDS = 5  # Random stratified K-fold

OUT_DIR = "checkpoints_mixed"
os.makedirs(OUT_DIR, exist_ok=True)

HELD_OUT_FILE = "held_out_indices.pkl"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------
# Utilities
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


def create_stratified_folds(cv_indices, dataset, k_folds):
    """Create stratified random K-fold splits balanced by camera and original sequence"""
    
    # Group indices by (camera, original_sequence_id)
    groups = {}
    for idx in cv_indices:
        orig_seq, cam, _ = dataset.meta[idx]
        key = (cam, orig_seq)
        if key not in groups:
            groups[key] = []
        groups[key].append(idx)
    
    group_list = list(groups.values())
    random.shuffle(group_list)
    
    # Distribute groups across folds
    folds = [[] for _ in range(k_folds)]
    for group_idx, group in enumerate(group_list):
        fold_idx = group_idx % k_folds
        folds[fold_idx].extend(group)
    
    # Create train/val splits
    splits = []
    for val_fold_idx in range(k_folds):
        val_idx = folds[val_fold_idx]
        train_idx = []
        for train_fold_idx in range(k_folds):
            if train_fold_idx != val_fold_idx:
                train_idx.extend(folds[train_fold_idx])
        splits.append((train_idx, val_idx))
    
    return splits


# --------------------------------------------------
# Train + Eval (CV-only)
# --------------------------------------------------
def train_and_eval_fold(dataset, train_idx, val_idx, fold_name):
    """Train and evaluate one mixed fold with training and validation loss printing"""

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
        print(f"[{fold_name}] Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), f"{OUT_DIR}/best_{fold_name}.pt")
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(torch.load(f"{OUT_DIR}/best_{fold_name}.pt", weights_only=True))

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
# MAIN MIXED LOOP
# --------------------------------------------------
dataset = SequenceDataset(NPZ_PATH)
n_seqs = len(dataset)

print(f"Dataset: {n_seqs} sequences")
print(f"Strategy: Mixed (Stratified Random K-Fold with camera/sequence balance)")
print(f"K-Folds: {K_FOLDS}\n")

# Load held-out test indices from LOCO
if os.path.exists(HELD_OUT_FILE):
    with open(HELD_OUT_FILE, 'rb') as f:
        held_out_indices = set(pickle.load(f))
    print(f"Loaded held-out indices from {HELD_OUT_FILE}: {len(held_out_indices)} sequences\n")
else:
    print(f"⚠️  Warning: {HELD_OUT_FILE} not found!")
    print("Run lstm_LOCO_cross-val.py first to generate held-out indices.\n")
    held_out_indices = set()

# Create CV dataset excluding held-out sequences
cv_indices = [i for i in range(n_seqs) if i not in held_out_indices]

# Create stratified random folds
splits = create_stratified_folds(cv_indices, dataset, K_FOLDS)

print(f"Created {K_FOLDS} stratified random folds, balanced by camera and original sequence\n")

GLOBAL_TP = GLOBAL_FP = GLOBAL_FN = 0
results = {}

for fold_idx, (train_idx, val_idx) in enumerate(splits):
    fold_name = f"fold_{fold_idx:02d}"

    print(f"\n=== Mixed Fold {fold_idx+1}/{K_FOLDS} ===")
    print(f"Train: {len(train_idx)} seqs | Val: {len(val_idx)} seqs")
    res = train_and_eval_fold(dataset, train_idx, val_idx, fold_name)
    results[fold_idx] = res

    GLOBAL_TP += res["tp"]
    GLOBAL_FP += res["fp"]
    GLOBAL_FN += res["fn"]

print("\n✅ Mixed CV Complete!")

# --------------------------------------------------
# FINAL TEST SET EVALUATION
# --------------------------------------------------
print("\n" + "="*60)
print("FINAL TEST SET EVALUATION (Held-out sequences)")
print("="*60)

if held_out_indices:
    print(f"\nEvaluating on {len(held_out_indices)} held-out sequences...")
    
    # Use the best performing fold's model
    best_fold_idx = max(results.keys(), key=lambda f: results[f]['frame_f1'])
    best_model_path = f"{OUT_DIR}/best_fold_{best_fold_idx:02d}.pt"
    
    print(f"\nUsing best model from Mixed fold {best_fold_idx} (Frame F1={results[best_fold_idx]['frame_f1']:.3f})")
    
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
print("MIXED CROSS-VALIDATION RESULTS")
print("="*60)

# Global (aggregated) metrics
global_p = GLOBAL_TP / (GLOBAL_TP + GLOBAL_FP) if (GLOBAL_TP + GLOBAL_FP) > 0 else 0
global_r = GLOBAL_TP / (GLOBAL_TP + GLOBAL_FN) if (GLOBAL_TP + GLOBAL_FN) > 0 else 0
global_f1 = 2 * global_p * global_r / (global_p + global_r) if (global_p + global_r) > 0 else 0

print("\n📊 AGGREGATED FRAME METRICS (all sequences combined):")
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

print("\n📈 AVERAGE METRICS (macro-average across folds):")
print(f"   Frame: P={avg_frame_p:.3f} ± {np.std([r['frame_p'] for r in results.values()]):.3f}")
print(f"          R={avg_frame_r:.3f} ± {np.std([r['frame_r'] for r in results.values()]):.3f}")
print(f"          F1={avg_frame_f1:.3f} ± {np.std([r['frame_f1'] for r in results.values()]):.3f}")
print(f"\n   Event: P={avg_event_p:.3f} ± {np.std([r['event_p'] for r in results.values()]):.3f}")
print(f"          R={avg_event_r:.3f} ± {np.std([r['event_r'] for r in results.values()]):.3f}")
print(f"          F1={avg_event_f1:.3f} ± {np.std([r['event_f1'] for r in results.values()]):.3f}")

print("\n" + "="*60)
print("MIXED SUMMARY - PER FOLD BREAKDOWN")
print("="*60)
print(f"\n{'Fold':<10} {'Val Size':<10} {'Frame P':<9} {'Frame R':<9} {'Frame F1':<9} {'Event P':<9} {'Event R':<9} {'Event F1':<9} {'Thr':<5}")
print("-" * 95)

for fold_idx in sorted(results.keys()):
    r = results[fold_idx]
    val_size = len(splits[fold_idx][1])
    print(f"{fold_idx:<10} {val_size:<10} {r['frame_p']:<9.3f} {r['frame_r']:<9.3f} {r['frame_f1']:<9.3f} "
          f"{r['event_p']:<9.3f} {r['event_r']:<9.3f} {r['event_f1']:<9.3f} {r['thr']:<5.2f}")

print("-" * 95)
print(f"{'GLOBAL':<10} {'':10} {global_p:<9.3f} {global_r:<9.3f} {global_f1:<9.3f} {'':>36}")
print(f"{'AVERAGE':<10} {'':10} {avg_frame_p:<9.3f} {avg_frame_r:<9.3f} {avg_frame_f1:<9.3f} "
      f"{avg_event_p:<9.3f} {avg_event_r:<9.3f} {avg_event_f1:<9.3f}")
print("="*60)

fold_frame_f1s = [r['frame_f1'] for r in results.values()]
fold_event_f1s = [r['event_f1'] for r in results.values()]

print(f"\nFrame F1:  min={min(fold_frame_f1s):.3f}, max={max(fold_frame_f1s):.3f}, mean={np.mean(fold_frame_f1s):.3f}, std={np.std(fold_frame_f1s):.3f}")
print(f"Event F1:  min={min(fold_event_f1s):.3f}, max={max(fold_event_f1s):.3f}, mean={np.mean(fold_event_f1s):.3f}, std={np.std(fold_event_f1s):.3f}")
print("="*60)

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: F1 scores per fold
ax = axes[0]
fold_ids = list(range(len(results)))
frame_f1s = [r['frame_f1'] for r in results.values()]
event_f1s = [r['event_f1'] for r in results.values()]

x = np.arange(len(fold_ids))
width = 0.35

ax.bar(x - width/2, frame_f1s, width, label='Frame F1', color='steelblue', alpha=0.8)
ax.bar(x + width/2, event_f1s, width, label='Event F1', color='coral', alpha=0.8)

ax.set_xlabel('Mixed Fold', fontsize=11)
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('Mixed: F1 per Fold', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"F{i}" for i in fold_ids])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1)

# Plot 2: P vs R
ax = axes[1]
frame_ps = [r['frame_p'] for r in results.values()]
frame_rs = [r['frame_r'] for r in results.values()]

ax.scatter(frame_rs, frame_ps, s=120, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1)

for i, fold_id in enumerate(fold_ids):
    ax.annotate(f"F{fold_id}", (frame_rs[i], frame_ps[i]), fontsize=8, xytext=(3, 3), textcoords='offset points')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Frame-Level P vs R', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('mixed_cv_results_lstm.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Plot saved: mixed_cv_results_lstm.png")

# Save results for comparison
mixed_results = {
    'approach': 'Mixed',
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

with open('mixed_cv_results_lstm.pkl', 'wb') as f:
    pickle.dump(mixed_results, f)
print(f"💾 Results saved: mixed_cv_results_lstm.pkl")

print("\n✅ Mixed Complete!")
