import os

import numpy as np
import torch

from training.common import TRAINING_OUTPUT_DIR, Vocab
from training.student import GRULM


class NumpyGRULM:

    def __init__(self, model: GRULM, vocab: Vocab, normalize=False):
        sd = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
        self.vocab = vocab
        self.E = sd["emb.weight"]
        self.n_layers = model.gru.num_layers
        self.layers = [(sd[f"gru.weight_ih_l{l}"], sd[f"gru.weight_hh_l{l}"], sd[f"gru.bias_ih_l{l}"], sd[f"gru.bias_hh_l{l}"])
                       for l in range(self.n_layers)]
        self.P, self.pb = sd["proj.weight"], sd["proj.bias"]
        self.ob = sd["out_bias"]
        self.Eo = self.E[: vocab.n_out]
        self.H = model.gru.hidden_size
        self.normalize = normalize
        self._cache = {}

    @staticmethod
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _step(self, hs, v):
        x = self.E[v]
        new = []
        for (Wi, Wh, bi, bh), h in zip(self.layers, hs):
            gi = Wi @ x + bi
            gh = Wh @ h + bh
            H = self.H
            r = self._sig(gi[:H] + gh[:H])
            z = self._sig(gi[H:2 * H] + gh[H:2 * H])
            n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
            h2 = (1 - z) * n + z * h
            new.append(h2)
            x = h2
        return tuple(new)

    def init_state(self):
        hs = tuple(np.zeros(self.H, dtype=np.float32) for _ in range(self.n_layers))
        return self._wrap(self._step(hs, self.vocab.BOS))

    def _wrap(self, hs):
        o = self.P @ hs[-1] + self.pb
        logz = 0.0
        if self.normalize:
            lg = self.Eo @ o + self.ob
            m = lg.max()
            logz = float(m + np.log(np.exp(lg - m).sum()))
        return (hs, o, logz)

    def advance(self, state, uid):
        return self._wrap(self._step(state[0], int(self.vocab.unit2v[uid])))

    def score(self, state, uids):
        _, o, logz = state
        v = self.vocab.unit2v[uids]
        lp = self.Eo[v] @ o + self.ob[v] - logz
        return (lp + self.vocab.unk_logp[uids]).astype(np.float32)


class TorchGRULM:

    def __init__(self, model: GRULM, vocab: Vocab, normalize=False, dev="cuda"):
        self.m = model.to(dev).eval()
        self.vocab = vocab
        self.dev = dev
        self.normalize = normalize
        self.W = self.m.out_weight().detach()
        self.b = self.m.out_bias.detach()
        self.unit2v = torch.from_numpy(vocab.unit2v).to(dev)
        self.unk_logp = torch.from_numpy(vocab.unk_logp).to(dev)

    @torch.no_grad()
    def _run(self, h0, v):
        x = torch.as_tensor(v, device=self.dev).view(-1, 1)
        y, h = self.m.gru(self.m.emb(x), h0)
        o = self.m.proj(y[:, -1])
        logz = torch.logsumexp(o @ self.W.t() + self.b, -1) if self.normalize else torch.zeros(len(o), device=self.dev)
        return [(h[:, i], o[i], logz[i]) for i in range(len(o))]

    def init_state(self):
        return self._run(None, [self.vocab.BOS])[0]

    def advance(self, state, uid):
        return self.advance_batch([state], [uid])[0]

    def advance_batch(self, states, uids):
        h0 = torch.stack([s[0] for s in states], 1)
        return self._run(h0, self.unit2v[torch.as_tensor(uids, device=self.dev)])

    def score(self, state, uids):
        return self.score_batch([state], uids)[0]

    @torch.no_grad()
    def score_batch(self, states, uids):
        u = torch.as_tensor(uids, device=self.dev)
        v = self.unit2v[u]
        O = torch.stack([s[1] for s in states])
        Z = torch.stack([s[2] for s in states])
        lp = O @ self.W[v].t() + self.b[v] - Z[:, None] + self.unk_logp[u][None]
        return lp.float().cpu().numpy()


class TorchPrefixLM:

    def __init__(self, model, vocab: Vocab, dev="cuda", cache_size=2000):
        self.m = model.to(dev).eval()
        self.vocab = vocab
        self.dev = dev
        self.cache_size = cache_size
        self._cache = {}
        self.unit2v = torch.from_numpy(vocab.unit2v).to(dev)
        self.unk_logp = torch.from_numpy(vocab.unk_logp).to(dev)

    def init_state(self):
        return (self.vocab.BOS,)

    def advance(self, state, uid):
        return state + (int(self.vocab.unit2v[uid]),)

    @torch.no_grad()
    def _logp_batch(self, states):
        miss = [s for s in dict.fromkeys(states) if s not in self._cache]
        if miss:
            if len(self._cache) + len(miss) > self.cache_size:
                self._cache = {s: self._cache[s] for s in states if s in self._cache}
            L = max(len(s[-64:]) for s in miss)
            x = torch.zeros((len(miss), L), dtype=torch.long, device=self.dev)
            last = []
            for i, s in enumerate(miss):
                s = s[-64:]
                x[i, :len(s)] = torch.tensor(s, device=self.dev)
                last.append(len(s) - 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o, _ = self.m(x)
            o = o[torch.arange(len(miss), device=self.dev), torch.tensor(last, device=self.dev)].float()
            lg = o @ self.m.out_weight().float().t() + self.m.out_bias.float()
            lp = torch.log_softmax(lg, -1).half()
            for i, s in enumerate(miss):
                self._cache[s] = lp[i]
        return torch.stack([self._cache[s] for s in states])

    def score(self, state, uids):
        return self.score_batch([state], uids)[0]

    def score_batch(self, states, uids):
        u = torch.as_tensor(uids, device=self.dev)
        lp = self._logp_batch(states)[:, self.unit2v[u]].float() + self.unk_logp[u][None]
        return lp.cpu().numpy()


class QuantGRULM(NumpyGRULM):

    def __init__(self, name, vocab):
        z = np.load(os.path.join(TRAINING_OUTPUT_DIR, f"{name}_int8.npz"))
        deq = lambda k: z[k + "_q"].astype(np.float32) * z[k + "_s"][:, None]
        self.vocab = vocab
        self.E = deq("emb")
        self.n_layers = int(z["n_layers"])
        self.layers = [(deq(f"w_ih{l}"), deq(f"w_hh{l}"), z[f"b_ih{l}"], z[f"b_hh{l}"]) for l in range(self.n_layers)]
        self.P, self.pb = deq("proj"), z["proj_b"]
        self.ob = z["out_bias"]
        self.Eo = self.E[: vocab.n_out]
        self.H = int(z["d_hid"])
        self.normalize = False
        self._cache = {}
