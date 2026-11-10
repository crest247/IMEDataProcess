import os, sys
import numpy as np
from verification.evaluate import evaluate, print_res
from verification.lm_ngram import TRAINING_OUTPUT_DIR, BigramKN
from verification.run_eval import load_gru
from verification.bopomofo import Lexicon, Bopomofo


class MixLM:
    def __init__(self, a, b, lam):
        self.a, self.b, self.lam = a, b, lam

    def init_state(self):
        return (self.a.init_state(), self.b.init_state())

    def advance(self, st, uid):
        return (self.a.advance(st[0], uid), self.b.advance(st[1], uid))

    def score(self, st, uids):
        return np.logaddexp(np.log(self.lam) + self.a.score(st[0], uids), np.log(1 - self.lam) + self.b.score(st[1], uids))


lex = Lexicon(Bopomofo())
uni = np.load(os.path.join(TRAINING_OUTPUT_DIR, "unigram_logp.npy"))
g = load_gru("gru_base", True)
b = BigramKN(lex.n_units).load(os.path.join(TRAINING_OUTPUT_DIR, "bigram_kn.npz"))
for lam in map(float, sys.argv[1:]):
    res, _ = evaluate(MixLM(g, b, lam), lex, uni, "valid", limit=600)
    print_res(f"mix λ_gru={lam} [valid 600]", res)
res, _ = evaluate(b, lex, uni, "valid", limit=600); print_res("bigram [valid 600]", res)
res, _ = evaluate(g, lex, uni, "valid", limit=600); print_res("gru [valid 600]", res)
