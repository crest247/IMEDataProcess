import torch
import torch.nn as nn


class GRULM(nn.Module):
    def __init__(self, vocab_size, n_out, d_emb=128, d_hid=256, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_out = n_out
        self.emb = nn.Embedding(vocab_size, d_emb)
        self.gru = nn.GRU(d_emb, d_hid, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_hid, d_emb)
        self.out_bias = nn.Parameter(torch.zeros(n_out))
        nn.init.normal_(self.emb.weight, std=0.05)

    def forward(self, x, h0=None):
        e = self.drop(self.emb(x))
        y, h = self.gru(e, h0)
        return self.proj(self.drop(y)), h

    def out_weight(self):
        return self.emb.weight[: self.n_out]
