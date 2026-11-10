import json
import os
import time
from collections import defaultdict

import numpy as np

from verification.decoder import decode

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRAINING_OUTPUT_DIR = os.path.join(ROOT, "training_output")


def bucket(r):
    return "縮寫≤33%" if r <= 0.34 else ("縮寫≤66%" if r <= 0.67 else "縮寫>66%")


def evaluate(lm, lex, uni_logp, split="test", limit=None, beam=8, len_bonus=0.0, verbose=0, margin=None):
    cases = [json.loads(l) for l in open(os.path.join(TRAINING_OUTPUT_DIR, f"eval_{split}.jsonl"), encoding="utf-8")]
    if limit:
        cases = cases[:limit]
    agg = defaultdict(lambda: defaultdict(float))
    t_total = 0.0
    examples = []
    for d in cases:
        s, gold, ends = d["input"], d["gold"], d["gold_ends"]
        gold_text = "".join(lex.units[u][0] for u in gold)
        t0 = time.perf_counter()
        edges = lex.edges(s)
        sents, firsts = decode(lm, lex, edges, len(s), uni_logp, beam=beam, len_bonus=len_bonus, margin=margin)
        t_total += time.perf_counter() - t0
        top_sent = "".join(lex.units[u][0] for u, _ in sents[0][1]) if sents else ""
        rank = next((i + 1 for i, f in enumerate(firsts) if f[1] == lex.units[gold[0]][0] and f[2] == ends[0]), None)
        char_ends, pos = [], 0
        for u in gold:
            for cid in lex.units[u][1]:
                i, r, t = lex.bpmf.syl[cid]
                st = d["styles"][len(char_ends)]
                pos += len(i) + (len(r) if st.startswith("完整") else 0) + (1 if st.endswith("+調") and t != 1 else 0)
                char_ends.append(pos)
        ok1 = False
        if firsts:
            _, txt, e, _ = firsts[0]
            ok1 = gold_text.startswith(txt) and char_ends[len(txt) - 1] == e
        m = {"n": 1, "ok@1": ok1, "sent_exact": top_sent == gold_text,
             "first@1": rank == 1, "first@5": rank is not None and rank <= 5, "first_found": rank is not None}
        if len(gold) > 1:
            st = lm.advance(lm.init_state(), gold[0])
            rest = s[ends[0]:]
            e2 = lex.edges(rest)
            _, f2 = decode(lm, lex, e2, len(rest), uni_logp, ctx_state=st, beam=beam, len_bonus=len_bonus, margin=margin)
            m["n_next"] = 1
            m["next@1"] = bool(f2) and f2[0][1] == lex.units[gold[1]][0] and f2[0][2] == ends[1] - ends[0]
        for key in ("全部", bucket(d["abbr_ratio"])):
            for k, v in m.items():
                agg[key][k] += float(v)
        if verbose and len(examples) < verbose:
            examples.append((s, d["text"], top_sent, [f"{f[1]}[{f[2]}]" for f in firsts[:5]], rank))
    res = {}
    for key, a in agg.items():
        n = a["n"]
        res[key] = {"n": int(n), "sent_exact": a["sent_exact"] / n, "first@1": a["first@1"] / n,
                    "first@5": a["first@5"] / n, "ok@1": a["ok@1"] / n, "first_found": a["first_found"] / n,
                    "next@1": a["next@1"] / a["n_next"] if a["n_next"] else None}
    res["ms_per_case"] = 1000 * t_total / len(cases)
    return res, examples


def print_res(name, res):
    print(f"== {name}  (解碼 {res['ms_per_case']:.1f} ms/筆，Python)")
    for key in ("全部", "縮寫≤33%", "縮寫≤66%", "縮寫>66%"):
        if key in res:
            r = res[key]
            print(f"  {key:8s} n={r['n']:5d}  整句={r['sent_exact']:.3f}  first@1={r['first@1']:.3f}  "
                  f"first@5={r['first@5']:.3f}  ok@1={r['ok@1']:.3f}  next@1={r['next@1']:.3f}  found={r['first_found']:.3f}")
