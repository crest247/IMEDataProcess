import heapq

import numpy as np


TABLE_PENALTY = -12.0
TABLE_UNI = -20.0


class Hyp:
    __slots__ = ("score", "parent", "uid", "lm_uid", "path", "_state")

    def __init__(self, score, parent, uid, path, lm_uid=None):
        self.score = score
        self.parent = parent
        self.uid = uid
        self.lm_uid = uid if lm_uid is None else lm_uid
        self.path = path
        self._state = None

    def state(self, lm):
        if self._state is None:
            self._state = self.parent if self.uid is None else lm.advance(self.parent, self.lm_uid)
        return self._state


def unk_uid(lm, lex):
    v = getattr(lm, "vocab", None)
    if v is None:
        return 0
    return int(np.nonzero(v.unit2v == v.UNK)[0][0])


STATS = {"states": 0}


def decode(lm, lex, edges, n, uni_logp, ctx_state=None, beam=8, len_bonus=0.0, n_sent=5, margin=None,
           groups=None, target=None):
    groups = groups or []
    T = n if target is None else target
    tu = unk_uid(lm, lex) if groups else None
    vocab = getattr(lm, "vocab", None)
    t_adj = TABLE_PENALTY - (float(vocab.unk_logp[tu]) if (vocab is not None and tu is not None) else 0.0)
    if ctx_state is None:
        ctx_state = lm.init_state()
    if n == 0 or T == 0 or not edges[0]:
        return [], []

    fut = np.full(T + 1, -np.inf)
    fut[T] = 0.0
    for p in range(T - 1, -1, -1):
        for uid, e in edges[p]:
            v = (uni_logp[uid] if uid >= 0 else TABLE_UNI) + fut[e]
            if v > fut[p]:
                fut[p] = v

    hyps = [[] for _ in range(T + 1)]
    hyps[0] = [Hyp(0.0, ctx_state, None, ())]
    first_lp = None
    for p in range(T):
        if not hyps[p] or not edges[p]:
            continue
        kept = heapq.nlargest(beam, hyps[p], key=lambda h: h.score)
        if margin is not None:
            kept = [h for h in kept if h.score >= kept[0].score - margin]
        eids = [u for u, _ in edges[p]]
        uids = np.fromiter((u if u >= 0 else tu for u in eids), dtype=np.int64, count=len(eids))
        is_table = np.array([u < 0 for u in eids])
        ends = [e for _, e in edges[p]]
        need = [h for h in kept if h._state is None and h.uid is not None]
        STATS["states"] += len(need)
        if need and hasattr(lm, "advance_batch"):
            for h, st in zip(need, lm.advance_batch([h.parent for h in need], [h.lm_uid for h in need])):
                h._state = st
        states = [h.state(lm) for h in kept]
        if hasattr(lm, "score_batch"):
            lps = lm.score_batch(states, uids)
        else:
            lps = [lm.score(st, uids) for st in states]
        for h, st, lp in zip(kept, states, lps):
            lp = np.where(is_table, np.asarray(lp, dtype=np.float64) + t_adj, lp) if is_table.any() else lp
            if p == 0:
                first_lp = lp
            for i, e in enumerate(ends):
                hyps[e].append(Hyp(h.score + float(lp[i]), st, eids[i], h.path + ((eids[i], e),), int(uids[i])))
        hyps[p] = kept

    finals = heapq.nlargest(max(beam, n_sent), hyps[T], key=lambda h: h.score)
    sentences = [(h.score, list(h.path)) for h in finals[:n_sent]] if T == n else []

    exact = {}
    for h in finals:
        k = h.path[0]
        if k not in exact or exact[k] < h.score:
            exact[k] = h.score
    best = {}

    def put(sc, text, e, uid):
        k = (text, e)
        if k not in best or best[k][0] < sc:
            best[k] = (sc, text, e, uid)

    for i, (uid, e) in enumerate(edges[0]):
        sc = exact.get((uid, e), float(first_lp[i]) + float(fut[e]))
        sc += len_bonus * e
        if uid >= 0:
            put(sc, lex.units[uid][0], e, uid)
        else:
            for j, (ch, cu) in enumerate(groups[-uid - 1]):
                put(sc - 1e-6 * j, ch, e, cu)
    firsts = sorted(best.values(), key=lambda x: (-x[0], -(float(uni_logp[x[3]]) if x[3] >= 0 else TABLE_UNI), x[3]))
    return sentences, firsts


def append_homophones(firsts, homophones, limit):
    out = list(firsts[:limit])
    seen = {(t, e) for _, t, e, _ in out}
    for e, chars in homophones:
        for ch, uid in chars:
            if (ch, e) not in seen:
                seen.add((ch, e))
                out.append((float("nan"), ch, e, uid))
    return out
