"""
collate_fn_sliding_window.py - CAUSAL WINDOW VERSION

Collate function for batching fixed-length sliding windows.
Windows are used for temporal context, but loss will be applied
ONLY to the last timestep (causal many-to-one).
"""

import torch


def collate_fn(batch):
    """Collate a batch of fixed-length windows.

    Args:
        batch: List of (x, y) tuples where:
            - x: (WINDOW_SIZE, F) tensor
            - y: (WINDOW_SIZE,) tensor

    Returns:
        x_batch: (B, WINDOW_SIZE, F)
        y_batch: (B, WINDOW_SIZE)
        mask:    (B, WINDOW_SIZE) all ones (API compatibility)
    """
    if len(batch) == 0:
        return None, None, None

    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]

    x_batch = torch.stack(xs, dim=0)  # (B, T, F)
    y_batch = torch.stack(ys, dim=0)  # (B, T)

    # Mask is kept for compatibility, but not used for causality
    mask = torch.ones_like(y_batch, dtype=torch.float32)

    return x_batch, y_batch, mask
