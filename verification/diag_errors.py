import json, os, sys
from collections import Counter
import numpy as np
from verification.decoder import decode
from verification.lm_ngram import TRAINING_OUTPUT_DIR
from verification.run_eval import load_teacher, load_gru
from verification.bopomofo import Lexicon, Bopomofo

name, limit = sys.argv[1], int(sys.argv[2])
lex = Lexicon(Bopomofo())
uni = np.load(os.path.join(TRAINING_OUTPUT_DIR, "unigram_logp.npy"))
lm = load_teacher(name) if name.startswith("tf_") else load_gru(name, False)
cases = [json.loads(l) for l in open(os.path.join(TRAINING_OUTPUT_DIR, "eval_test.jsonl"), encoding="utf-8")][:limit]
c = Counter(); diffs = []
for d in cases:
    s, gold = d["input"], d["gold"]
    gt = "".join(lex.units[u][0] for u in gold)
    sents, _ = decode(lm, lex, lex.edges(s), len(s), uni, beam=8)
    top = "".join(lex.units[u][0] for u, _ in sents[0][1])
    if top == gt:
        c["正確"] += 1; continue
    st = lm.init_state(); gs = 0.0
    for u in gold:
        gs += float(lm.score(st, np.array([u]))[0]); st = lm.advance(st, u)
    if gs > sents[0][0] + 1e-4:
        c["搜尋錯誤"] += 1
    else:
        c["模型錯誤"] += 1
        nd = sum(a != b for a, b in zip(top, gt)) if len(top) == len(gt) else -1
        c[f"模型錯誤-錯{nd}字" if nd >= 0 else "模型錯誤-字數不同"] += 1
        if len(diffs) < 25: diffs.append((s, gt, top, round(gs - sents[0][0], 2)))
n = len(cases)
for k, v in sorted(c.items()): print(f"{k:16s} {v:5d}  {v/n:.3f}")
for x in diffs: print("  ", x)
