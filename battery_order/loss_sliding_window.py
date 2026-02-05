"""
loss_sliding_window.py - CAUSAL VERSION with FOCAL LOSS

Focal loss for handling class imbalance by down-weighting easy examples.
"""

import torch
import torch.nn.functional as F


def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary classification.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        logits: (B,) raw model outputs
        labels: (B,) binary labels (0 or 1)
        alpha: Weight for positive class (default 0.25)
        gamma: Focusing parameter (default 2.0, higher = more focus on hard examples)
    
    Returns:
        Scalar loss
    """
    # Convert logits to probabilities
    probs = torch.sigmoid(logits)
    
    # Compute p_t: probability of true class
    # If label=1, p_t = prob; if label=0, p_t = 1-prob
    labels_float = labels.float()
    p_t = probs * labels_float + (1 - probs) * (1 - labels_float)
    
    # Compute focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma
    
    # Compute alpha weight
    alpha_t = alpha * labels_float + (1 - alpha) * (1 - labels_float)
    
    # Binary cross entropy (manual computation for control)
    bce = -labels_float * torch.log(probs + 1e-8) - (1 - labels_float) * torch.log(1 - probs + 1e-8)
    
    # Focal loss
    loss = alpha_t * focal_weight * bce
    
    return loss.mean()


def masked_bce_loss(logits, labels, mask=None, pos_weight=None, use_focal=True, 
                   focal_alpha=0.25, focal_gamma=2.0):
    """
    Causal BCE loss with optional focal loss.
    
    Args:
        logits: (B, T) raw model outputs
        labels: (B, T) binary labels
        mask: unused (kept for API compatibility)
        pos_weight: ignored when use_focal=True
        use_focal: If True, use focal loss instead of standard BCE
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss (higher = more focus on hard examples)
    
    Returns:
        Scalar loss
    """
    # Use ONLY the last timestep (causal)
    logits_last = logits[:, -1]           # (B,)
    labels_last = labels[:, -1]           # (B,)
    
    if use_focal:
        loss = focal_loss(logits_last, labels_last, alpha=focal_alpha, gamma=focal_gamma)
    else:
        # Standard BCE with pos_weight
        labels_float = labels_last.float()
        if pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits_last,
                labels_float,
                pos_weight=pos_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits_last,
                labels_float
            )
    
    return loss