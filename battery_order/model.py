import torch
import torch.nn as nn

import math
import torch.nn.functional as F

# LSTM
class FrameLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim= 128, # 128, 64, 264
        num_layers=2, # 2, 3
        bidirectional=False,
        dropout=0.3, # 0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)

        self.head = nn.Linear(out_dim, 1)

    def forward(self, x, mask=None):
        """
        x: (B, T, F)
        """
        h, _ = self.lstm(x)          # (B, T, H)
        logits = self.head(h)        # (B, T, 1)
        return logits.squeeze(-1)    # (B, T)


# Transformer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        L = x.size(1)
        return x + self.pe[:, :L, :]

class FrameTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64, # 128, 64
        nhead=8, # 8, 4
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1, # 0.2
        max_len=2000,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)  # per-frame logit

    def forward(self, x, mask):
        """
        x: (B, T, F) float tensor
        mask: (B, T) bool or 0/1 tensor where 1 indicates valid frame
        returns logits: (B, T)
        """
        # project
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_enc(x)

        # Transformer expects src_key_padding_mask with True for positions to be masked (i.e., padding)
        if mask.dtype != torch.bool:
            mask_bool = mask == 0
        else:
            mask_bool = ~mask  # mask True where padding
        # encoder (batch_first=True)
        x = self.encoder(x, src_key_padding_mask=mask_bool)  # (B, T, d_model)
        logits = self.out(x).squeeze(-1)  # (B, T)
        return logits
