import json
import os
import random
import sys

import numpy as np

from verification.bopomofo import STANDALONE, TONE_MARK, Lexicon, Bopomofo

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRAINING_OUTPUT_DIR = os.path.join(ROOT, "training_output")
MAX_SYL = 12
STYLES = ["縮寫", "縮寫+調", "完整", "完整+調"]
WEIGHTS = [4, 1, 2, 2]


def type_syllable(bpmf, syl, st):
    i, r, t = syl
    if st == "縮寫+調" and r and i in STANDALONE and (i, "", t) in bpmf.table_syl:
        st = "縮寫"
    base = i if st.startswith("縮寫") else i + r
    return base + (TONE_MARK[t] if st.endswith("+調") else ""), st


def main(split="test", n=3000, seed=1):
    bpmf = Bopomofo()
    lx = Lexicon(bpmf)
    tokens = np.load(os.path.join(TRAINING_OUTPUT_DIR, f"{split}_tokens.npy"))
    offsets = np.load(os.path.join(TRAINING_OUTPUT_DIR, f"{split}_offsets.npy"))
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        li = rng.randrange(len(offsets) - 1)
        sent = tokens[offsets[li]:offsets[li + 1]].tolist()
        st = rng.randrange(len(sent))
        k = rng.choice([1, 2, 2, 3, 3, 4, 5, 6])
        gold, parts, styles, ends = [], [], [], []
        nsyl = 0
        for uid in sent[st:st + k]:
            cids = lx.units[uid][1]
            if nsyl + len(cids) > MAX_SYL:
                break
            for cid in cids:
                txt, s2 = type_syllable(bpmf, bpmf.syl[cid], rng.choices(STYLES, WEIGHTS)[0])
                parts.append(txt)
                styles.append(s2)
            nsyl += len(cids)
            gold.append(uid)
            ends.append(sum(map(len, parts)))
        if not gold:
            continue
        s = "".join(parts)
        if bpmf.classify(s) != "混打":
            continue
        out.append({
            "input": s,
            "gold": gold,
            "gold_ends": ends,
            "text": "／".join(lx.units[u][0] for u in gold),
            "styles": styles,
            "abbr_ratio": round(sum(x.startswith("縮寫") for x in styles) / len(styles), 3),
        })
    path = os.path.join(TRAINING_OUTPUT_DIR, f"eval_{split}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(path, len(out))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["test"]), *(map(int, sys.argv[2:3]) or []))
