import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRAINING_OUTPUT_DIR = os.path.join(ROOT, "training_output")


def load_split(name):
    return (np.load(os.path.join(TRAINING_OUTPUT_DIR, f"{name}_tokens.npy")),
            np.load(os.path.join(TRAINING_OUTPUT_DIR, f"{name}_offsets.npy")))


class BigramKN:
    def __init__(self, V):
        self.V = V
        self.BOS = V

    def fit(self, tokens, offsets):
        V1 = self.V + 1
        prev = np.empty_like(tokens, dtype=np.int64)
        prev[1:] = tokens[:-1]
        prev[offsets[:-1]] = self.BOS
        keys = prev * V1 + tokens.astype(np.int64)
        self.keys, cnt = np.unique(keys, return_counts=True)
        self.cnt = cnt.astype(np.float64)
        n1, n2 = (cnt == 1).sum(), (cnt == 2).sum()
        self.D = n1 / (n1 + 2 * n2)
        a = self.keys // V1
        b = self.keys % V1
        self.c_a = np.bincount(a, weights=self.cnt, minlength=V1)
        self.n_a = np.bincount(a, minlength=V1).astype(np.float64)
        ncont = np.bincount(b, minlength=V1).astype(np.float64)[: self.V]
        self.pcont = (ncont + 1.0) / (ncont.sum() + self.V)
        return self

    def save(self, path):
        np.savez(path, keys=self.keys, cnt=self.cnt, D=self.D, c_a=self.c_a, n_a=self.n_a, pcont=self.pcont)

    def load(self, path):
        z = np.load(path)
        self.keys, self.cnt, self.D = z["keys"], z["cnt"], float(z["D"])
        self.c_a, self.n_a, self.pcont = z["c_a"], z["n_a"], z["pcont"]
        return self

    def init_state(self):
        return self.BOS

    def advance(self, state, uid):
        return int(uid)

    def score(self, state, uids):
        uids = np.asarray(uids, dtype=np.int64)
        ca = self.c_a[state]
        if ca == 0:
            return np.log(self.pcont[uids]).astype(np.float32)
        k = state * (self.V + 1) + uids
        idx = np.searchsorted(self.keys, k)
        idx = np.minimum(idx, len(self.keys) - 1)
        c = np.where(self.keys[idx] == k, self.cnt[idx], 0.0)
        p = np.maximum(c - self.D, 0.0) / ca + self.D * self.n_a[state] / ca * self.pcont[uids]
        return np.log(p).astype(np.float32)

    def sentence_logp(self, tokens, offsets):
        total = 0.0
        for i in range(len(offsets) - 1):
            st = self.init_state()
            for t in tokens[offsets[i]:offsets[i + 1]]:
                total += float(self.score(st, [t])[0])
                st = self.advance(st, t)
        return total, int(offsets[-1])
