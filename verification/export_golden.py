import json
import os
import random
import sys

import numpy as np

from verification.bopomofo import Bopomofo, Lexicon
from verification.decoder import append_homophones, decode
from verification.model_wrappers import QuantGRULM
from training.common import TRAINING_OUTPUT_DIR, load_vocab


def export_golden(bpmf, lex, lm, uni, path, n_random=300):
    cases = []
    for s in ["ㄏㄍㄒㄖㄕˋㄋˇ", "ㄨㄒㄏㄨㄢㄋ", "ㄉㄥㄧㄒ", "ㄟˋㄏ", "ㄗˊㄖ", "ㄕˋㄐ", "ㄕㄐ", "ㄖˊㄕ", "ㄒㄧㄢㄕ",
              "ㄏㄨㄟˋㄧˋ", "ㄉㄧˋㄊㄧ", "ㄕ", "ㄕˋ", "ㄧˋ", "ㄍㄚ˙", "ㄦ", "ㄖ", "ㄑ˙", "ˋ", "ㄏˋˊ", "ㄐㄑㄒ", "ㄅㄚˊㄏ"]:
        cases.append((s, []))
    for rest in ["ㄕㄨ", "ㄕ", "ㄕˋ", "ㄏㄨㄟ", "ㄧˋ", "ㄒㄧㄢ"]:
        cases.append((rest, [0]))
    ev = [json.loads(l) for l in open(os.path.join(TRAINING_OUTPUT_DIR, "eval_test.jsonl"), encoding="utf-8")]
    rng = random.Random(0)
    for d in rng.sample(ev, n_random):
        cases.append((d["input"], []))
        if len(d["gold"]) > 1:
            cases.append((d["input"][d["gold_ends"][0]:], [d["gold"][0]]))
    for s in ["ㄝ", "ㄟ", "ㄙ˙", "ㄝˋ", "ㄏㄨㄟ˙", "ㄏㄝ", "ㄝㄏ", "ㄋㄧˇㄝㄏ", "ㄉㄥㄧㄒㄝ", "ㄅㄥˊㄌ", "ㄝㄟ", "ㄏㄝˇ",
              "ㄏˋˊ", "ˋㄏ", "ㄍㄚ˙ㄏ", "ㄕㄐㄝㄋ", "ㄥㄏ", "ㄅㄥˊㄝ"]:
        cases.append((s, []))
    for s in ["ㄝ", "ㄝˋ", "ㄏㄝˇ", "ㄗˊㄖ", "ㄕˉ"]:
        cases.append((s, [0]))
    for s in ["ㄓˉ", "ㄕˉ", "ㄏˉ", "ㄕˉㄐ", "ㄏㄨㄟˉ", "ㄓ", "ㄦ", "ㄏㄨㄟ", "ㄒㄧㄢ", "ㄏˉㄓ"]:
        cases.append((s, []))
    for s in ["ㄉㄚˋㄧㄚˇ", "ㄉㄚㄧㄚ", "ㄉㄚˋ", "ㄉㄚ", "ㄉ", "ㄧㄚˇㄓ", "ㄓㄨㄤˋㄎ", "ㄕˉㄐ", "ㄧˋㄙ", "ㄒㄧㄚˋ", "ㄒㄧㄢ", "ㄓㄨㄤ", "ㄧ", "ㄓ", "ㄓㄓ", "ㄧㄉ"]:
        cases.append((s, []))
    for s in ["ㄧㄚˇ", "ㄧㄚ", "ㄉㄚˋ", "ㄧ", "ㄓ", "ㄒㄧㄚˋ"]:
        cases.append((s, [29]))

    def mixed(s, ctx):
        st = lm.init_state()
        for u in ctx:
            st = lm.advance(st, u)
        e, g, T = lex.edges_ext(s)
        se, fi = decode(lm, lex, e, len(s), uni, ctx_state=st, beam=8, n_sent=3, groups=g, target=T)
        txt = lambda u: lex.units[u][0] if u >= 0 else g[-u - 1][0][0]
        hs = append_homophones(fi, lex.homophones(s), 60)[min(len(fi), 60):]
        return T, [(sc, "".join(txt(u) for u, _ in p)) for sc, p in se], fi, hs

    def table(keys):
        out = []
        for k in keys:
            out += [c for c in bpmf.char_list.get(k, []) if c not in out]
        return out

    with open(path, "w", encoding="utf-8") as f:
        f.write("# input\tcontext\tclass\tchars(前10)\tsentences(text@score|...)\tfirsts(text@end@score|... 前20)\thomophones(text@end|...)\n")
        for s, ctx in cases:
            cls = bpmf.classify(s, continuation=bool(ctx))
            chars, sents, firsts, homophones = "", "", "", ""
            if cls == "單字完整注音":
                items = table([bpmf.complete[s]])
                chars = "".join(items[:10])
            elif cls == "混打":
                T, se, fi, hs = mixed(s, ctx)
                if T == 0:
                    cls = "不合法"
                sents = "|".join(f"{t}@{sc:.4f}" for sc, t in se)
                firsts = "|".join(f"{t}@{e}@{sc:.4f}" for sc, t, e, _ in fi[:20])
                if T:
                    homophones = "|".join(f"{t}@{e}" for _, t, e, _ in hs)
            f.write(f"{s}\t{','.join(map(str, ctx))}\t{cls}\t{chars}\t{sents}\t{firsts}\t{homophones}\n")
    print(f"golden.tsv {len(cases)} 筆")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "gru_kd"
    bpmf = Bopomofo()
    lex = Lexicon(bpmf)
    vocab = load_vocab(name)
    uni = np.load(os.path.join(TRAINING_OUTPUT_DIR, "unigram_logp.npy"))
    export_golden(bpmf, lex, QuantGRULM(name, vocab), uni, os.path.join(TRAINING_OUTPUT_DIR, "golden.tsv"))


if __name__ == "__main__":
    main()
