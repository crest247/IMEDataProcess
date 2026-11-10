import os

import numpy as np
import torch
import torch.utils.checkpoint

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRAINING_OUTPUT_DIR = os.path.join(ROOT, "training_output")


class Vocab:
    def __init__(self, train_tokens, n_units, min_count=2, alpha=0.5):
        cnt = np.bincount(train_tokens, minlength=n_units)
        keep = np.nonzero(cnt >= min_count)[0]
        self.K = len(keep)
        self.UNK = self.K
        self.BOS = self.K + 1
        self.size = self.K + 2
        self.n_out = self.K + 1
        self.unit2v = np.full(n_units, self.UNK, dtype=np.int64)
        self.unit2v[keep] = np.arange(self.K)
        self.v2unit = keep
        rare = cnt < min_count
        rc = np.where(rare, cnt + alpha, 0.0)
        self.unk_logp = np.where(rare, np.log(np.maximum(rc, 1e-30) / rc.sum()), 0.0).astype(np.float32)
        self.is_rare = rare

    def save(self, path):
        np.savez(path, v2unit=self.v2unit, unit2v=self.unit2v, unk_logp=self.unk_logp, is_rare=self.is_rare)


def load_vocab(name):
    z = np.load(os.path.join(TRAINING_OUTPUT_DIR, f"{name}_vocab.npz"))
    vocab = Vocab.__new__(Vocab)
    vocab.v2unit, vocab.unit2v, vocab.unk_logp, vocab.is_rare = z["v2unit"], z["unit2v"], z["unk_logp"], z["is_rare"]
    vocab.K = len(vocab.v2unit)
    vocab.UNK, vocab.BOS, vocab.size, vocab.n_out = vocab.K, vocab.K + 1, vocab.K + 2, vocab.K + 1
    return vocab


def make_batches(tokens, offsets, bos, max_len, tokens_per_batch, rng, aug=0):
    segs = []
    for i in range(len(offsets) - 1):
        a, b = int(offsets[i]), int(offsets[i + 1])
        starts = [a] + [a + int(rng.integers(1, b - a)) for _ in range(aug) if b - a > 1]
        for a2 in starts:
            for s in range(a2, b, max_len):
                segs.append((s, min(b, s + max_len)))
    segs = np.array(segs, dtype=np.int64)
    lens = segs[:, 1] - segs[:, 0]
    order = np.lexsort((rng.random(len(segs)), lens))
    batches = []
    i = 0
    while i < len(order):
        L = int(lens[order[i]])
        bs = max(1, tokens_per_batch // (L + 1))
        idx = order[i:i + bs]
        L = int(lens[idx].max())
        x = np.zeros((len(idx), L + 1), dtype=np.int32)
        y = np.full((len(idx), L + 1), -100, dtype=np.int32)
        for j, k in enumerate(idx):
            a, b = segs[k]
            seq = tokens[a:b]
            x[j, 0] = bos
            x[j, 1:1 + len(seq)] = seq
            y[j, :len(seq)] = seq
        batches.append((x[:, :L], y[:, :L]))
        i += bs
    rng.shuffle(batches)
    return batches


def _chunk_loss(o, W, b, y, selfnorm, kd_alpha, T, to, tW, tb):
    lg = (o @ W.t() + b).float()
    logz = torch.logsumexp(lg, -1)
    nll = logz - lg.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    loss = nll.sum() * (1 - kd_alpha)
    if selfnorm:
        loss = loss + selfnorm * (logz ** 2).sum()
    if to is not None:
        with torch.no_grad():
            tp = torch.softmax((to @ tW.t() + tb).float() / T, -1)
        kd = -(tp * torch.log_softmax(lg / T, -1)).sum() * T * T
        loss = loss + kd_alpha * kd
    return loss, nll.sum().detach(), logz.abs().sum().detach()


def lm_loss(model, o, y, selfnorm=0.0, teacher=None, to=None, kd_alpha=0.0, T=1.0, chunk=1024):
    mask = y != -100
    o, y = o[mask], y[mask]
    if to is not None:
        to = to[mask]
    W, b = model.out_weight(), model.out_bias
    tW = teacher.out_weight() if teacher is not None else None
    tb = teacher.out_bias if teacher is not None else None
    n = y.numel()
    total, nll_sum, absz = 0.0, 0.0, 0.0
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        args = (o[sl], W, b, y[sl], selfnorm, kd_alpha, T, None if to is None else to[sl], tW, tb)
        if torch.is_grad_enabled():
            l, nl, az = torch.utils.checkpoint.checkpoint(_chunk_loss, *args, use_reentrant=False)
        else:
            l, nl, az = _chunk_loss(*args)
        total = total + l
        nll_sum += nl.item()
        absz += az.item()
    return total / n, nll_sum, n, absz / n
