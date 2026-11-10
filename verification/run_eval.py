import argparse
import json
import os

import numpy as np
import torch

from verification.evaluate import evaluate, print_res
from verification.lm_ngram import TRAINING_OUTPUT_DIR, BigramKN
from verification.model_wrappers import NumpyGRULM, QuantGRULM, TorchGRULM, TorchPrefixLM
from training.common import load_vocab
from training.student import GRULM
from training.teacher import TransformerLM
from verification.bopomofo import Lexicon, Bopomofo


def load_teacher(name):
    cfg = json.load(open(os.path.join(TRAINING_OUTPUT_DIR, f"{name}_cfg.json")))
    vocab = load_vocab(name)
    m = TransformerLM(vocab.size, vocab.n_out, cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["dropout"], cfg["max_len"])
    m.load_state_dict(torch.load(os.path.join(TRAINING_OUTPUT_DIR, f"{name}.pt"), map_location="cpu"))
    return TorchPrefixLM(m, vocab)


def load_gru(name, normalize, gpu=True):
    cfg = json.load(open(os.path.join(TRAINING_OUTPUT_DIR, f"{name}_cfg.json")))
    vocab = load_vocab(name)
    m = GRULM(vocab.size, vocab.n_out, cfg["d_emb"], cfg["d_hid"], cfg["n_layers"], cfg["dropout"])
    m.load_state_dict(torch.load(os.path.join(TRAINING_OUTPUT_DIR, f"{name}.pt"), map_location="cpu"))
    return TorchGRULM(m, vocab, normalize=normalize) if gpu else NumpyGRULM(m, vocab, normalize=normalize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--len_bonus", type=float, default=0.0)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="GRU 用 numpy（與 Kotlin 版相同的計算）")
    ap.add_argument("--examples", type=int, default=0)
    args = ap.parse_args()
    lex = Lexicon(Bopomofo())
    uni = np.load(os.path.join(TRAINING_OUTPUT_DIR, "unigram_logp.npy"))
    if args.model.startswith("bigram"):
        lm = BigramKN(lex.n_units).load(os.path.join(TRAINING_OUTPUT_DIR, args.model.replace("bigram", "bigram_kn") + ".npz"))
    elif args.model.endswith("_int8"):
        lm = QuantGRULM(args.model[:-5], load_vocab(args.model[:-5]))
    elif args.model.startswith("tf_"):
        lm = load_teacher(args.model)
    else:
        lm = load_gru(args.model, args.normalize, gpu=not args.cpu)
    res, ex = evaluate(lm, lex, uni, args.split, limit=args.limit, beam=args.beam, len_bonus=args.len_bonus, verbose=args.examples)
    name = f"{args.model}{' (normalized)' if args.normalize else ''} beam={args.beam} len_bonus={args.len_bonus} [{args.split}]"
    print_res(name, res)
    for e in ex:
        print("   ", e)
    out = os.path.join(TRAINING_OUTPUT_DIR, "results.jsonl")
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"name": name, "limit": args.limit, **res}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
