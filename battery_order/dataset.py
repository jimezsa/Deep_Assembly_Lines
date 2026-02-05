import numpy as np
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.feats = d["feats"].tolist()    # list of (T_i, F)
        self.labels = d["labels"].tolist()  # list of (T_i,)
        self.meta = d["meta"].tolist()
        d.close()

        assert len(self.feats) == len(self.labels)

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        x = torch.tensor(self.feats[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
