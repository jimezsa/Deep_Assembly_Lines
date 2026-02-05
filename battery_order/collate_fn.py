import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Pads to T_max in batch and builds a boolean mask (True = valid timestep)

    batch: list of (x, y)
      x: (T_i, F)
      y: (T_i,)
    """
    xs, ys = zip(*batch)

    lengths = torch.tensor([x.shape[0] for x in xs])

    x_pad = pad_sequence(xs, batch_first=True)  # (B, T_max, F)
    y_pad = pad_sequence(ys, batch_first=True)  # (B, T_max)

    mask = torch.arange(x_pad.size(1))[None, :] < lengths[:, None]
    # mask shape: (B, T_max), dtype=bool

    return x_pad, y_pad, mask
