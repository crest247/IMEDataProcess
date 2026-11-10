import csv
import json
import lzma
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SOURCE_DATA_DIR = os.path.join(ROOT, "source_data")
TRAINING_OUTPUT_DIR = os.path.join(ROOT, "training_output")

VALID_FRAC = 0.01
TEST_FRAC = 0.01
SEED = 20261022


def unigram_logp(tokens, n_units, alpha=0.5):
    c = np.bincount(tokens, minlength=n_units).astype(np.float64) + alpha
    return np.log(c / c.sum()).astype(np.float32)


def main():
    os.makedirs(TRAINING_OUTPUT_DIR, exist_ok=True)
    syl2id = {}
    for r in csv.DictReader(open(os.path.join(SOURCE_DATA_DIR, "chars.csv"), encoding="utf-8")):
        syl2id[(r["initial"], r["remain"], int(r["tone"]))] = int(r["char_id"])
    unit2id = {}
    for r in csv.DictReader(open(os.path.join(SOURCE_DATA_DIR, "units.csv"), encoding="utf-8")):
        unit2id[(r["unit"], tuple(map(int, r["char_ids"].split())))] = int(r["id"])

    n_lines = sum(1 for _ in lzma.open(os.path.join(SOURCE_DATA_DIR, "lines.jsonl.xz"), "rt", encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    u = rng.random(n_lines)
    split_of = np.where(u < TEST_FRAC, 2, np.where(u < TEST_FRAC + VALID_FRAC, 1, 0))

    names = ["train", "valid", "test"]
    toks = [[] for _ in names]
    offs = [[0] for _ in names]
    stats = {"lines": n_lines, "units": 0, "bad_units": 0, "dropped_lines": 0, "bad_examples": []}
    with lzma.open(os.path.join(SOURCE_DATA_DIR, "lines.jsonl.xz"), "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            d = json.loads(line)
            ids = []
            ok = True
            for w, ini, rem, ton in zip(d["units"], d["initial"], d["remain"], d["tone"]):
                key = (w, tuple(syl2id.get(s, -1) for s in zip(ini, rem, ton)))
                uid = unit2id.get(key)
                stats["units"] += 1
                if uid is None:
                    stats["bad_units"] += 1
                    if len(stats["bad_examples"]) < 30:
                        stats["bad_examples"].append([w, list(zip(ini, rem, ton))])
                    ok = False
                    break
                ids.append(uid)
            if not ok or not ids:
                stats["dropped_lines"] += 1
                continue
            s = split_of[idx]
            toks[s].extend(ids)
            offs[s].append(len(toks[s]))
            if idx % 500000 == 0:
                print(idx, file=sys.stderr)

    for s, name in enumerate(names):
        np.save(os.path.join(TRAINING_OUTPUT_DIR, f"{name}_tokens.npy"), np.asarray(toks[s], dtype=np.int32))
        np.save(os.path.join(TRAINING_OUTPUT_DIR, f"{name}_offsets.npy"), np.asarray(offs[s], dtype=np.int64))
        stats[f"{name}_lines"] = len(offs[s]) - 1
        stats[f"{name}_tokens"] = len(toks[s])
    n_units = sum(1 for _ in open(os.path.join(SOURCE_DATA_DIR, "units.csv"), encoding="utf-8")) - 1
    np.save(os.path.join(TRAINING_OUTPUT_DIR, "unigram_logp.npy"), unigram_logp(np.asarray(toks[0], dtype=np.int32), n_units))
    json.dump(stats, open(os.path.join(TRAINING_OUTPUT_DIR, "prep_stats.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "bad_examples"}, ensure_ascii=False))
    print(stats["bad_examples"][:5])


if __name__ == "__main__":
    main()
