import json
import os
import sys

import numpy as np
import torch

from training.common import TRAINING_OUTPUT_DIR


def q_rows(m):
    m = np.asarray(m, dtype=np.float32)
    s = np.abs(m).max(axis=1) / 127.0
    s = np.where(s == 0, 1.0, s).astype(np.float32)
    q = np.clip(np.rint(m / s[:, None]), -127, 127).astype(np.int8)
    return q, s


def quantize(name):
    cfg = json.load(open(os.path.join(TRAINING_OUTPUT_DIR, f"{name}_cfg.json")))
    sd = {k: v.float().numpy() for k, v in torch.load(os.path.join(TRAINING_OUTPUT_DIR, f"{name}.pt"), map_location="cpu").items()}
    out = {"n_layers": np.int32(cfg["n_layers"]), "d_hid": np.int32(cfg["d_hid"]), "d_emb": np.int32(cfg["d_emb"])}
    mats = {"emb": sd["emb.weight"], "proj": sd["proj.weight"]}
    for l in range(cfg["n_layers"]):
        mats[f"w_ih{l}"] = sd[f"gru.weight_ih_l{l}"]
        mats[f"w_hh{l}"] = sd[f"gru.weight_hh_l{l}"]
        out[f"b_ih{l}"] = sd[f"gru.bias_ih_l{l}"]
        out[f"b_hh{l}"] = sd[f"gru.bias_hh_l{l}"]
    for k, m in mats.items():
        out[k + "_q"], out[k + "_s"] = q_rows(m)
    out["proj_b"] = sd["proj.bias"]
    out["out_bias"] = sd["out_bias"]
    path = os.path.join(TRAINING_OUTPUT_DIR, f"{name}_int8.npz")
    np.savez(path, **out)
    nbytes = sum(v.nbytes for v in out.values())
    print(f"{path}: {nbytes / 1e6:.1f} MB（未壓縮；其中 emb {out['emb_q'].nbytes / 1e6:.1f} MB）")
    return path


if __name__ == "__main__":
    quantize(sys.argv[1] if len(sys.argv) > 1 else "gru_kd")
