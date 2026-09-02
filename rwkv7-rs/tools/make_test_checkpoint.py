#!/usr/bin/env python3
"""Build a small random RWKV-7 checkpoint for testing, without needing PyTorch.

Writes a genuine `torch.save` zip archive (pickle protocol 2 + stored entries)
plus the same weights as .safetensors, so both loader paths get exercised.
"""
import argparse, io, json, os, pickle, struct, sys, types, zipfile
import numpy as np

# --- fake just enough of torch for `pickle` to emit the right GLOBAL opcodes ---
_torch = types.ModuleType("torch")
_utils = types.ModuleType("torch._utils")


class FloatStorage:  # noqa: N801 - name must match torch's
    pass


class HalfStorage:
    pass


class BFloat16Storage:
    pass


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    raise NotImplementedError  # never called; only its qualified name is pickled


for cls in (FloatStorage, HalfStorage, BFloat16Storage):
    cls.__module__ = "torch"
    setattr(_torch, cls.__name__, cls)
_rebuild_tensor_v2.__module__ = "torch._utils"
_utils._rebuild_tensor_v2 = _rebuild_tensor_v2
_torch._utils = _utils
sys.modules["torch"] = _torch
sys.modules["torch._utils"] = _utils

STORAGE_OF = {"float32": FloatStorage, "float16": HalfStorage}


class _Pickler(pickle.Pickler):
    def __init__(self, f, arrays):
        super().__init__(f, protocol=2)
        self.arrays = arrays

    def persistent_id(self, obj):
        if isinstance(obj, _Storage):
            self.arrays.append(obj)
            return ("storage", STORAGE_OF[obj.dtype], str(obj.key), "cpu", obj.numel)
        return None


class _Storage:
    def __init__(self, key, arr, dtype):
        self.key, self.arr, self.dtype = key, arr, dtype
        self.numel = arr.size


def contiguous_stride(shape):
    st, acc = [], 1
    for d in reversed(shape):
        st.append(acc)
        acc *= d
    return tuple(reversed(st))


def save_pth(path, tensors, dtype="float32"):
    payload, storages = {}, []
    for i, (name, arr) in enumerate(tensors.items()):
        st = _Storage(i, np.ascontiguousarray(arr, dtype=dtype), dtype)
        payload[name] = _rebuild_tensor_v2.__reduce_placeholder__ = None  # placeholder
        payload[name] = _RebuildCall(st, arr.shape)
        storages.append(st)

    buf = io.BytesIO()
    seen = []
    p = _Pickler(buf, seen)
    p.dump(payload)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("archive/data.pkl", buf.getvalue())
        for st in storages:
            z.writestr(f"archive/data/{st.key}", st.arr.tobytes())
    return path


class _RebuildCall:
    """Pickles as `torch._utils._rebuild_tensor_v2(storage, 0, size, stride, False, OrderedDict())`."""

    def __init__(self, storage, shape):
        self.storage, self.shape = storage, tuple(int(d) for d in shape)

    def __reduce__(self):
        import collections

        return (
            _rebuild_tensor_v2,
            (self.storage, 0, self.shape, contiguous_stride(self.shape), False, collections.OrderedDict()),
        )


def save_safetensors(path, tensors, dtype="float32"):
    header, blobs, off = {}, [], 0
    st_dtype = {"float32": "F32", "float16": "F16"}[dtype]
    for name, arr in tensors.items():
        b = np.ascontiguousarray(arr, dtype=dtype).tobytes()
        header[name] = {"dtype": st_dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
    return path


def build(n_layer, n_embd, vocab, head_size, seed=1234):
    rng = np.random.default_rng(seed)
    n_head = n_embd // head_size
    dim_ffn = n_embd * 4
    # Small LoRA ranks, and different ones per projection, so the loader's rank
    # inference is genuinely exercised.
    d_decay, d_aaa, d_mv, d_gate = 24, 16, 8, 32

    def r(*shape, scale=0.1):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    z = {}
    z["emb.weight"] = r(vocab, n_embd, scale=0.5)
    z["blocks.0.ln0.weight"] = r(n_embd, scale=0.05) + 1.0
    z["blocks.0.ln0.bias"] = r(n_embd, scale=0.05)
    for i in range(n_layer):
        b, a, f = f"blocks.{i}.", f"blocks.{i}.att.", f"blocks.{i}.ffn."
        z[b + "ln1.weight"] = r(n_embd, scale=0.05) + 1.0
        z[b + "ln1.bias"] = r(n_embd, scale=0.05)
        z[b + "ln2.weight"] = r(n_embd, scale=0.05) + 1.0
        z[b + "ln2.bias"] = r(n_embd, scale=0.05)
        for nm in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            z[a + nm] = np.abs(r(1, 1, n_embd, scale=0.3))  # stored as [1,1,C], like real ckpts
        z[a + "w0"] = r(1, 1, n_embd, scale=0.5)
        z[a + "w1"] = r(n_embd, d_decay)
        z[a + "w2"] = r(d_decay, n_embd)
        z[a + "a0"] = r(1, 1, n_embd, scale=0.5)
        z[a + "a1"] = r(n_embd, d_aaa)
        z[a + "a2"] = r(d_aaa, n_embd)
        if i > 0:
            z[a + "v0"] = r(1, 1, n_embd, scale=0.5)
            z[a + "v1"] = r(n_embd, d_mv)
            z[a + "v2"] = r(d_mv, n_embd)
        z[a + "g1"] = r(n_embd, d_gate)
        z[a + "g2"] = r(d_gate, n_embd)
        z[a + "k_k"] = np.abs(r(1, 1, n_embd, scale=0.5)) + 0.2
        z[a + "k_a"] = r(1, 1, n_embd, scale=0.3) + 1.0
        z[a + "r_k"] = r(n_head, head_size, scale=0.2)
        for nm in ("receptance", "key", "value", "output"):
            z[a + nm + ".weight"] = r(n_embd, n_embd, scale=0.08)
        z[a + "ln_x.weight"] = r(n_embd, scale=0.05) + 1.0
        z[a + "ln_x.bias"] = r(n_embd, scale=0.05)
        z[f + "x_k"] = np.abs(r(1, 1, n_embd, scale=0.3))
        z[f + "key.weight"] = r(dim_ffn, n_embd, scale=0.08)
        z[f + "value.weight"] = r(n_embd, dim_ffn, scale=0.08)
    z["ln_out.weight"] = r(n_embd, scale=0.05) + 1.0
    z["ln_out.bias"] = r(n_embd, scale=0.05)
    z["head.weight"] = r(vocab, n_embd, scale=0.08)
    return z


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_model")
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=128)
    ap.add_argument("--head-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    z = build(args.n_layer, args.n_embd, args.vocab, args.head_size, args.seed)
    save_pth(args.out + ".pth", z)
    save_safetensors(args.out + ".safetensors", z)
    np.savez(args.out + ".npz", **z)
    print(f"wrote {args.out}.pth / .safetensors / .npz  "
          f"({len(z)} tensors, {sum(v.size for v in z.values()):,} params)")
