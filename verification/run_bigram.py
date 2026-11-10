import os
import sys
import time

import numpy as np

from verification.evaluate import evaluate, print_res
from verification.lm_ngram import TRAINING_OUTPUT_DIR, BigramKN, load_split
from verification.bopomofo import Lexicon, Bopomofo


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    bpmf = Bopomofo()
    lex = Lexicon(bpmf)
    V = lex.n_units
    tr_t, tr_o = load_split("train")
    t0 = time.time()
    lm = BigramKN(V).fit(tr_t, tr_o)
    lm.save(os.path.join(TRAINING_OUTPUT_DIR, "bigram_kn.npz"))
    print(f"bigram fit {time.time() - t0:.1f}s, {len(lm.keys)} 種 bigram, D={lm.D:.3f}")
    uni = np.load(os.path.join(TRAINING_OUTPUT_DIR, "unigram_logp.npy"))

    va_t, va_o = load_split("valid")
    lp, n = lm.sentence_logp(va_t, va_o)
    print(f"valid perplexity = {np.exp(-lp / n):.1f}  ({n} tokens)")

    for lb in (0.0, 1.0, 2.0):
        res, ex = evaluate(lm, lex, uni, "test", limit=limit, beam=8, len_bonus=lb, verbose=8 if lb == 0 else 0)
        print_res(f"bigram KN, beam=8, len_bonus={lb}", res)
        for e in ex:
            print("   ", e)


if __name__ == "__main__":
    main()
