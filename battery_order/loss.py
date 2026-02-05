import torch.nn.functional as F

# Padding must not contribute to loss
def masked_bce_loss(logits, targets, mask, pos_weight=None):
    """
    logits: (B, T)
    targets: (B, T)
    mask: (B, T) bool
    """
    logits = logits[mask]
    targets = targets[mask]

    return F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        pos_weight=pos_weight,
    )
