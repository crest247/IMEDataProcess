import argparse
import json
import math
import os
import time

import numpy as np
import torch

from training.common import TRAINING_OUTPUT_DIR, Vocab, lm_loss, make_batches
from training.student import GRULM
from training.teacher import TransformerLM


def build_model(kind, vocab, cfg):
    if kind == "student":
        return GRULM(vocab.size, vocab.n_out, cfg["d_emb"], cfg["d_hid"], cfg["n_layers"], cfg["dropout"])
    return TransformerLM(vocab.size, vocab.n_out, cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["dropout"], cfg["max_len"])


DEFAULTS = {
    "student": dict(d_emb=128, d_hid=256, n_layers=2, dropout=0.2, lr=2e-3, epochs=10, tokens_per_batch=8192, max_len=64,
                    selfnorm=0.1, min_count=2, aug=1),
    "teacher": dict(d_model=512, n_layers=6, n_heads=8, dropout=0.15, lr=5e-4, epochs=12, tokens_per_batch=8192, max_len=64,
                    selfnorm=0.0, min_count=2, aug=1),
}


def load(name):
    return np.load(os.path.join(TRAINING_OUTPUT_DIR, f"{name}_tokens.npy")), np.load(os.path.join(TRAINING_OUTPUT_DIR, f"{name}_offsets.npy"))


@torch.no_grad()
def eval_ppl(model, batches, dev):
    model.eval()
    tot, n, az = 0.0, 0, 0.0
    for x, y in batches:
        x, y = torch.from_numpy(x).to(dev).long(), torch.from_numpy(y).to(dev).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o, _ = model(x)
            _, s, k, a = lm_loss(model, o, y)
        tot += s
        n += k
        az += a * k
    model.train()
    return math.exp(tot / n), az / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["student", "teacher"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--init", default=None, help="從既有 checkpoint 接續（例：gru_aug）")
    ap.add_argument("--kd_alpha", type=float, default=0.5)
    ap.add_argument("--kd_T", type=float, default=1.0)
    ap.add_argument("--set", nargs="*", default=[], help="覆寫設定，例 d_hid=384 epochs=5")
    args = ap.parse_args()
    cfg = dict(DEFAULTS[args.kind])
    for kv in args.set:
        k, v = kv.split("=")
        cfg[k] = type(cfg[k])(v) if k in cfg else float(v)

    dev = "cuda"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tr_t, tr_o = load("train")
    va_t, va_o = load("valid")
    n_units = sum(1 for _ in open(os.path.join(TRAINING_OUTPUT_DIR, "..", "source_data", "units.csv"), encoding="utf-8")) - 1
    vocab = Vocab(tr_t, n_units, cfg["min_count"])
    vocab.save(os.path.join(TRAINING_OUTPUT_DIR, f"{args.name}_vocab.npz"))
    tr_v = vocab.unit2v[tr_t]
    va_v = vocab.unit2v[va_t]
    print(f"vocab K={vocab.K}（≥{cfg['min_count']} 次），train tokens={len(tr_v)}，UNK 比例={np.mean(tr_v == vocab.UNK):.4f}")

    model = build_model(args.kind, vocab, cfg).to(dev)
    if args.init:
        model.load_state_dict(torch.load(os.path.join(TRAINING_OUTPUT_DIR, f"{args.init}.pt"), map_location=dev))
        print("init from", args.init)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"{args.kind} 參數量 {n_param / 1e6:.2f}M", cfg)

    teacher = None
    if args.teacher:
        tcfg = json.load(open(os.path.join(TRAINING_OUTPUT_DIR, f"{args.teacher}_cfg.json")))
        teacher = build_model("teacher", vocab, tcfg).to(dev)
        teacher.load_state_dict(torch.load(os.path.join(TRAINING_OUTPUT_DIR, f"{args.teacher}.pt"), map_location=dev))
        teacher.eval()
        print("teacher loaded", args.teacher)

    va_batches = make_batches(va_v, va_o, vocab.BOS, cfg["max_len"], cfg["tokens_per_batch"], np.random.default_rng(1))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    n_ep = int(cfg["epochs"])
    best = float("inf")
    log = []
    step = 0
    for ep in range(n_ep):
        batches = make_batches(tr_v, tr_o, vocab.BOS, cfg["max_len"], cfg["tokens_per_batch"], rng, int(cfg["aug"]))
        total_steps = n_ep * len(batches)
        t0 = time.time()
        run, runn = 0.0, 0
        for x, y in batches:
            lr = cfg["lr"] * min(1.0, (step + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps)))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = torch.from_numpy(x).to(dev).long(), torch.from_numpy(y).to(dev).long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o, _ = model(x)
                to = None
                if teacher is not None:
                    with torch.no_grad():
                        to, _ = teacher(x)
                loss, s, k, _ = lm_loss(model, o, y, cfg["selfnorm"], teacher, to, args.kd_alpha if teacher else 0.0, args.kd_T)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += s
            runn += k
            step += 1
            if step % 500 == 0:
                print(f"ep{ep} step{step} train_ppl={math.exp(run / runn):.1f} lr={lr:.2e} {time.time() - t0:.0f}s", flush=True)
                run, runn = 0.0, 0
        ppl, absz = eval_ppl(model, va_batches, dev)
        log.append({"epoch": ep, "valid_ppl": ppl, "mean_abs_logZ": absz, "sec": time.time() - t0})
        print(f"== epoch {ep}: valid ppl(UNK 視為一類)={ppl:.1f}  |logZ|={absz:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if ppl < best:
            best = ppl
            torch.save(model.state_dict(), os.path.join(TRAINING_OUTPUT_DIR, f"{args.name}.pt"))
    json.dump(cfg, open(os.path.join(TRAINING_OUTPUT_DIR, f"{args.name}_cfg.json"), "w"))
    json.dump(log, open(os.path.join(TRAINING_OUTPUT_DIR, f"{args.name}_log.json"), "w"), indent=1)
    print("best valid ppl", best)


if __name__ == "__main__":
    main()
