#!/usr/bin/env python3
"""Build a synthetic HuggingFace model directory for testing qwen-make-pth.

Plants vision, MTP and language-model-prefixed keys so every filter branch is
exercised, and spreads tensors across shards with an index.json. numpy only.
"""
import argparse, json, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ST_DTYPE = {"float32": "F32", "float16": "F16", "bfloat16": "BF16", "int64": "I64", "uint8": "U8"}


def to_bytes(arr, dtype):
    if dtype == "bfloat16":
        return (np.ascontiguousarray(arr, np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()
    if dtype == "float8":
        return np.ascontiguousarray(arr).astype(np.uint8).tobytes()
    return np.ascontiguousarray(arr, dtype=dtype).tobytes()


def write_safetensors(path, tensors):
    """tensors: {name: (ndarray, dtype_str)}"""
    hdr, blobs, off = {}, [], 0
    for name, (arr, dt) in tensors.items():
        b = to_bytes(arr, dt)
        st = "F8_E4M3" if dt == "float8" else ST_DTYPE[dt]
        hdr[name] = {"dtype": st, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def build(root, fmt="safetensors", sharded=True):
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(4242)

    def r(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.05

    # shard 1: text layers 0-1 (under the language_model wrapper) + odd dtypes
    s1 = {
        "model.language_model.embed_tokens.weight": (r(64, 32), "bfloat16"),
        "model.language_model.layers.0.self_attn.q_proj.weight": (r(32, 32), "bfloat16"),
        "model.language_model.layers.0.mlp.gate_proj.weight": (r(64, 32), "float16"),
        "model.language_model.layers.1.self_attn.k_proj.weight": (r(32, 32), "float32"),
        "model.language_model.layers.1.input_layernorm.weight": (r(32), "float32"),
        # dropped: vision
        "visual.patch_embed.proj.weight": (r(16, 16), "bfloat16"),
        "model.visual.blocks.0.attn.qkv.weight": (r(8, 8), "bfloat16"),
    }
    # shard 2: head + more filtered keys + non-float dtypes
    s2 = {
        "lm_head.weight": (r(64, 32), "bfloat16"),
        "model.language_model.norm.weight": (r(32), "float32"),
        "model.language_model.rotary_emb.inv_freq": (r(16), "float32"),
        "model.counts": ((rng.integers(0, 100, 8)).astype(np.int64), "int64"),
        "model.mask": ((rng.integers(0, 2, 8)).astype(np.uint8), "uint8"),
        "model.scales_fp8": (rng.integers(0, 255, 12).astype(np.uint8), "float8"),
        # dropped: MTP, only visible after stripping the language_model prefix
        "model.language_model.mtp.layers.0.weight": (r(8, 8), "float32"),
        "mtp.norm.weight": (r(8), "float32"),
        # dropped: multimodal projector
        "multi_modal_projector.linear_1.weight": (r(8, 8), "float32"),
    }

    shards = [s1, s2] if sharded else [{**s1, **s2}]
    names = [f"model-{i+1:05d}-of-{len(shards):05d}.safetensors" for i in range(len(shards))]
    if not sharded:
        names = ["model.safetensors"]

    if fmt == "safetensors":
        for name, sh in zip(names, shards):
            write_safetensors(os.path.join(root, name), sh)
    else:  # torch .bin shards
        from make_test_checkpoint import save_pth
        names = [n.replace(".safetensors", ".bin").replace("model-", "pytorch_model-") for n in names]
        if not sharded:
            names = ["pytorch_model.bin"]
        for name, sh in zip(names, shards):
            # save_pth writes a uniform dtype; use float32 for the .bin path
            save_pth(os.path.join(root, name), {k: v[0] for k, v in sh.items()}, dtype="float32")

    if sharded:
        wm = {}
        for name, sh in zip(names, shards):
            for k in sh:
                wm[k] = name
        idx = {"metadata": {"total_size": 0}, "weight_map": wm}
        fn = "model.safetensors.index.json" if fmt == "safetensors" else "pytorch_model.bin.index.json"
        with open(os.path.join(root, fn), "w") as f:
            json.dump(idx, f, indent=2)

    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(
            {"architectures": ["Qwen3NextForCausalLM"], "hidden_size": 32, "num_hidden_layers": 2,
             "torch_dtype": "bfloat16", "vocab_size": 64, "nested": {"a": [1, 2, 3], "b": None},
             "unicode_note": "café 🚀"},
            f, indent=2,
        )

    # what the exporter should keep, and what it should drop
    expect_keep, expect_vision, expect_mtp = {}, 0, 0
    for sh in shards:
        for k, (arr, dt) in sh.items():
            if k.startswith(("visual.", "model.visual.", "vision_tower.", "model.vision_tower.",
                             "multi_modal_projector.", "model.multi_modal_projector.")):
                expect_vision += 1
                continue
            stripped = k[len("model.language_model."):] if k.startswith("model.language_model.") else k
            if stripped.startswith("mtp."):
                expect_mtp += 1
                continue
            expect_keep[stripped] = to_bytes(arr, "float32" if fmt == "bin" else dt)
    return expect_keep, expect_vision, expect_mtp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="safetensors", choices=["safetensors", "bin"])
    ap.add_argument("--flat", action="store_true", help="single shard, no index.json")
    a = ap.parse_args()
    keep, nv, nm = build(a.out, a.format, sharded=not a.flat)
    print(json.dumps({"dir": a.out, "expect_keep": len(keep), "expect_vision": nv, "expect_mtp": nm}))
