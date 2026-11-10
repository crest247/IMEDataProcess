import os
import shutil
import struct
import sys

import numpy as np

from training.common import TRAINING_OUTPUT_DIR, load_vocab
from training.units import UnitTable

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
APP_ASSETS_DIR = os.path.join(ROOT, "app_assets")


def u32(f, *v):
    f.write(struct.pack(f"<{len(v)}I", *v))


def arr(f, a, dt):
    f.write(np.ascontiguousarray(a, dtype=dt).tobytes())


def build_trie(lex):
    nodes = [lex.root]
    child_off, child_cid, child_node, unit_off, node_units = [0], [], [], [0], []
    i = 0
    while i < len(nodes):
        nd = nodes[i]
        for cid in sorted(k for k in nd if k is not None):
            child_cid.append(cid)
            child_node.append(len(nodes))
            nodes.append(nd[cid])
        child_off.append(len(child_cid))
        node_units.extend(uid for _, _, uid in nd.get(None, []))
        unit_off.append(len(node_units))
        i += 1
    return len(nodes), child_off, child_cid, child_node, unit_off, node_units


def export_units(lex, vocab, uni, path):
    texts = [u[0].encode("utf-8") for u in lex.units]
    text_off = np.cumsum([0] + [len(t) for t in texts])
    syl_off = np.cumsum([0] + [len(u[1]) for u in lex.units])
    syl = [c for u in lex.units for c in u[1]]
    n_nodes, child_off, child_cid, child_node, unit_off, node_units = build_trie(lex)
    with open(path, "wb") as f:
        u32(f, lex.n_units, vocab.K)
        arr(f, text_off, "<u4")
        u32(f, int(text_off[-1]))
        f.write(b"".join(texts))
        arr(f, syl_off, "<u4")
        arr(f, syl, "<u2")
        arr(f, vocab.unit2v, "<i4")
        arr(f, vocab.unk_logp, "<f4")
        arr(f, uni, "<f4")
        u32(f, n_nodes)
        arr(f, child_off, "<u4")
        arr(f, child_cid, "<u2")
        arr(f, child_node, "<u4")
        arr(f, unit_off, "<u4")
        arr(f, node_units, "<u4")
    print(f"bopomofo_units.bin {os.path.getsize(path) / 1e6:.1f} MB，trie {n_nodes} 節點")


def export_model(name, path):
    z = np.load(os.path.join(TRAINING_OUTPUT_DIR, f"{name}_int8.npz"))
    L = int(z["n_layers"])
    with open(path, "wb") as f:
        u32(f, L, int(z["d_hid"]), int(z["d_emb"]))

        def mat(k):
            q, s = z[k + "_q"], z[k + "_s"]
            u32(f, q.shape[0], q.shape[1])
            arr(f, s, "<f4")
            arr(f, q, "i1")

        def vec(k):
            u32(f, len(z[k]))
            arr(f, z[k], "<f4")

        mat("emb")
        for l in range(L):
            mat(f"w_ih{l}")
            mat(f"w_hh{l}")
            vec(f"b_ih{l}")
            vec(f"b_hh{l}")
        mat("proj")
        vec("proj_b")
        vec("out_bias")
    print(f"bopomofo_model.bin {os.path.getsize(path) / 1e6:.1f} MB")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "gru_kd"
    os.makedirs(APP_ASSETS_DIR, exist_ok=True)
    lex = UnitTable()
    vocab = load_vocab(name)
    uni = np.load(os.path.join(TRAINING_OUTPUT_DIR, "unigram_logp.npy"))
    for fn in ("chars.csv", "all_chars.csv"):
        shutil.copyfile(os.path.join(ROOT, "source_data", fn), os.path.join(APP_ASSETS_DIR, "bopomofo_" + fn))
    export_units(lex, vocab, uni, os.path.join(APP_ASSETS_DIR, "bopomofo_units.bin"))
    export_model(name, os.path.join(APP_ASSETS_DIR, "bopomofo_model.bin"))


if __name__ == "__main__":
    main()
