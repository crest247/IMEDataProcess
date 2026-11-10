import os, sys
import numpy as np
from verification import decoder
from verification.evaluate import evaluate
from verification.lm_ngram import TRAINING_OUTPUT_DIR
from verification.run_eval import load_gru
from verification.bopomofo import Lexicon, Bopomofo
lex = Lexicon(Bopomofo()); uni = np.load(os.path.join(TRAINING_OUTPUT_DIR, "unigram_logp.npy")); lm = load_gru("gru_kd", False)
for m in [None, 12.0, 8.0, 6.0, 4.0]:
    decoder.STATS["states"] = 0
    res, _ = evaluate(lm, lex, uni, "test", margin=m)
    r = res["全部"]
    print(f"margin={m}: 整句={r['sent_exact']:.3f} first@1={r['first@1']:.3f} first@5={r['first@5']:.3f} next@1={r['next@1']:.3f}  GRU 狀態/查詢={decoder.STATS['states'] / (r['n'] * 1.94):.1f}", flush=True)
