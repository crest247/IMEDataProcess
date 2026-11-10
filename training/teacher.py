import torch
import torch.nn as nn


class TransformerLM(nn.Module):

    def __init__(self, vocab_size, n_out, d=512, n_layers=6, n_heads=8, dropout=0.1, max_len=128):
        super().__init__()
        self.n_out = n_out
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d)
        self.out_bias = nn.Parameter(torch.zeros(n_out))
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, x, h0=None):
        T = x.size(1)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        hdn = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        return self.norm(self.enc(hdn, mask=mask, is_causal=True)), None

    def out_weight(self):
        return self.emb.weight[: self.n_out]
