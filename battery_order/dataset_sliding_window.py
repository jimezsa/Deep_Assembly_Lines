"""
dataset.py - WINDOWED VERSION

Dataset for windowed sequences with fixed length.
Much simpler than variable-length version since all windows are WINDOW_SIZE frames.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Dataset for fixed-length windowed sequences.
    
    Each item is a window of WINDOW_SIZE frames with:
    - Features: (WINDOW_SIZE, F)
    - Labels: (WINDOW_SIZE,) 
    - Metadata: (orig_sequence, camera, window_start_idx, frame_idx_list)
    """
    
    def __init__(self, npz_path):
        """Load windowed data from .npz file.
        
        Args:
            npz_path: Path to .npz file created by feature_extraction.py
        """
        d = np.load(npz_path, allow_pickle=True)
        self.feats = d["feats"].tolist()    # list of (WINDOW_SIZE, F)
        self.labels = d["labels"].tolist()  # list of (WINDOW_SIZE,)
        self.meta = d["meta"].tolist()      # list of (seq, cam, start_idx, frame_idxs)
        d.close()

        assert len(self.feats) == len(self.labels) == len(self.meta), \
            "Mismatched data lengths"
        
        # Verify all windows are same size
        if len(self.feats) > 0:
            self.window_size = len(self.feats[0])
            for i, feat in enumerate(self.feats):
                assert len(feat) == self.window_size, \
                    f"Window {i} has {len(feat)} frames, expected {self.window_size}"
                assert len(self.labels[i]) == self.window_size, \
                    f"Label {i} has {len(self.labels[i])} frames, expected {self.window_size}"
        else:
            self.window_size = 0
        
        print(f"[Dataset] Loaded {len(self.feats)} windows of size {self.window_size}")
    
    def __len__(self):
        return len(self.feats)
    
    def __getitem__(self, idx):
        """Get a single window.
        
        Returns:
            x: Tensor of shape (WINDOW_SIZE, F)
            y: Tensor of shape (WINDOW_SIZE,)
        """
        x = torch.tensor(self.feats[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    
    def get_original_sequence(self, idx):
        """Get the original sequence name for a window (for CV splitting).
        
        Args:
            idx: Window index
            
        Returns:
            str: Original sequence name (e.g., "wrong1")
        """
        return self.meta[idx][0]
    
    def get_camera(self, idx):
        """Get the camera ID for a window.
        
        Args:
            idx: Window index
            
        Returns:
            str: Camera ID
        """
        return self.meta[idx][1]