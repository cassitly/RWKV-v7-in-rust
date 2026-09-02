#!/usr/bin/env python3
"""Numpy reference forward pass for RWKV-7, transcribed from the upstream demos.

Follows `rwkv_v7_demo_fast.py` (RWKV_x070_TMix_one / CMix_one) and cross-checks
against `rwkv_v7_numpy.py`. Everything is f32, which is what the Rust port does.
"""
import argparse, sys
import numpy as np

LN_EPS = 1e-5
GN_EPS = 64e-5
DECAY = float(np.exp(-0.5))  # 0.60653066


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def layer_norm(x, w, b, eps=LN_EPS):
    m = x.mean()
    v = x.var()
    return (x - m) / np.sqrt(v + eps) * w + b


def group_norm(x, H, N, w, b, eps=GN_EPS):
    y = x.reshape(H, N)
    m = y.mean(axis=1, keepdims=True)
    v = y.var(axis=1, keepdims=True)
    return (((y - m) / np.sqrt(v + eps)).reshape(-1) * w + b)


def forward_one(z, cfg, token, state, v_first):
    n_layer, C, H, N = cfg["n_layer"], cfg["n_embd"], cfg["n_head"], cfg["head_size"]
    x = z["emb.weight"][token].copy()

    for i in range(n_layer):
        b, a, f = f"blocks.{i}.", f"blocks.{i}.att.", f"blocks.{i}.ffn."
        xx = layer_norm(x, z[b + "ln1.weight"], z[b + "ln1.bias"])

        x_prev = state["att_x"][i]
        d = x_prev - xx
        xr, xw, xk, xv, xa, xg = (xx + d * z[a + nm] for nm in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"))
        state["att_x"][i] = xx.copy()

        r = z[a + "receptance.weight"] @ xr
        k = z[a + "key.weight"] @ xk
        v = z[a + "value.weight"] @ xv
        w = np.tanh(xw @ z[a + "w1"]) @ z[a + "w2"]
        aa = sigmoid(z[a + "a0"] + (xa @ z[a + "a1"]) @ z[a + "a2"])
        g = sigmoid(xg @ z[a + "g1"]) @ z[a + "g2"]

        kk = (k * z[a + "k_k"]).reshape(H, N)
        kk = kk / np.maximum(np.linalg.norm(kk, axis=-1, keepdims=True), 1e-12)
        kk = kk.reshape(-1)
        k = k * (1 + (aa - 1) * z[a + "k_a"])

        if i == 0:
            v_first = v.copy()
        else:
            v = v + (v_first - v) * sigmoid(z[a + "v0"] + (xv @ z[a + "v1"]) @ z[a + "v2"])

        w = np.exp(-DECAY * sigmoid(z[a + "w0"] + w))

        # state[h] is [N, N]: row = value dim, column = key dim
        S = state["att_kv"][i]
        vk = v.reshape(H, N, 1) @ k.reshape(H, 1, N)
        ab = (-kk).reshape(H, N, 1) @ (kk * aa).reshape(H, 1, N)
        S = S * w.reshape(H, 1, N) + S @ ab + vk
        state["att_kv"][i] = S
        y = (S @ r.reshape(H, N, 1)).reshape(-1)

        y = group_norm(y, H, N, z[a + "ln_x.weight"], z[a + "ln_x.bias"])
        bonus = (r * k * z[a + "r_k"]).reshape(H, N).sum(axis=-1, keepdims=True) * v.reshape(H, N)
        y = y + bonus.reshape(-1)
        x = x + z[a + "output.weight"] @ (y * g)

        xx = layer_norm(x, z[b + "ln2.weight"], z[b + "ln2.bias"])
        fp = state["ffn_x"][i]
        kf = xx + (fp - xx) * z[f + "x_k"]
        state["ffn_x"][i] = xx.copy()
        kf = np.maximum(z[f + "key.weight"] @ kf, 0) ** 2
        x = x + z[f + "value.weight"] @ kf

    x = layer_norm(x, z["ln_out.weight"], z["ln_out.bias"])
    return z["head.weight"] @ x, state, v_first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tokens", required=True, help="comma-separated token ids")
    ap.add_argument("--out", required=True, help="raw little-endian f32 logits")
    args = ap.parse_args()

    raw = np.load(args.npz)
    z = {k: np.asarray(raw[k], dtype=np.float32).squeeze() for k in raw.files}

    vocab, C = z["emb.weight"].shape
    H, N = z["blocks.0.att.r_k"].shape
    n_layer = len([k for k in z if k.endswith("att.r_k")])
    cfg = dict(n_layer=n_layer, n_embd=C, n_head=H, head_size=N)
    z["blocks.0.att.r_k"] = z["blocks.0.att.r_k"].reshape(-1)
    for i in range(n_layer):
        z[f"blocks.{i}.att.r_k"] = z[f"blocks.{i}.att.r_k"].reshape(-1)

    # ln0 is folded into the embedding table, exactly as the demos do at load time.
    z["emb.weight"] = np.stack(
        [layer_norm(z["emb.weight"][t], z["blocks.0.ln0.weight"], z["blocks.0.ln0.bias"]) for t in range(vocab)]
    ).astype(np.float32)

    state = {
        "att_x": [np.zeros(C, np.float32) for _ in range(n_layer)],
        "att_kv": [np.zeros((H, N, N), np.float32) for _ in range(n_layer)],
        "ffn_x": [np.zeros(C, np.float32) for _ in range(n_layer)],
    }
    v_first = np.zeros(C, np.float32)
    tokens = [int(t) for t in args.tokens.split(",") if t.strip()]
    for t in tokens:
        logits, state, v_first = forward_one(z, cfg, t, state, v_first)

    logits.astype("<f4").tofile(args.out)
    print(f"reference: {len(tokens)} tokens -> {logits.size} logits, "
          f"argmax={int(np.argmax(logits))}, mean={logits.mean():.6f}, std={logits.std():.6f}")


if __name__ == "__main__":
    main()
