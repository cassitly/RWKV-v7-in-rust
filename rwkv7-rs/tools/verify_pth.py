#!/usr/bin/env python3
"""Verify a .pth written by qwen-make-pth is genuine torch format.

Reads it with Python's *stdlib* zipfile + pickle (torch stubbed via sys.modules),
so this checks the real serialization contract without needing PyTorch installed.
"""
import io, json, pickle, struct, sys, types, zipfile
import numpy as np

# --- stub the torch names a checkpoint's pickle refers to ---
_torch = types.ModuleType("torch"); _utils = types.ModuleType("torch._utils")
STORAGE_NP = {}
for cls_name, np_dt, esz in [
    ("DoubleStorage", "<f8", 8), ("FloatStorage", "<f4", 4), ("HalfStorage", "<f2", 2),
    ("BFloat16Storage", "<u2", 2), ("LongStorage", "<i8", 8), ("IntStorage", "<i4", 4),
    ("ShortStorage", "<i2", 2), ("CharStorage", "<i1", 1), ("ByteStorage", "<u1", 1),
    ("BoolStorage", "<u1", 1), ("Float8_e4m3fnStorage", "<u1", 1), ("Float8_e5m2Storage", "<u1", 1),
]:
    cls = type(cls_name, (), {}); cls.__module__ = "torch"
    setattr(_torch, cls_name, cls); STORAGE_NP[cls] = (np_dt, esz)

class Rebuilt:
    def __init__(self, storage, offset, size, stride, requires_grad, hooks, *rest):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)
        self.requires_grad, self.hooks = requires_grad, hooks

def _rebuild_tensor_v2(*a): return Rebuilt(*a)
_rebuild_tensor_v2.__module__ = "torch._utils"
_utils._rebuild_tensor_v2 = _rebuild_tensor_v2
_torch._utils = _utils
sys.modules["torch"], sys.modules["torch._utils"] = _torch, _utils


def load(path):
    z = zipfile.ZipFile(path)
    bad = z.testzip()
    assert bad is None, f"CRC failure in entry {bad}"

    names = z.namelist()
    pkl_name = next(n for n in names if n.endswith("data.pkl"))
    root = pkl_name[: -len("data.pkl")]

    records = {}

    class Unp(pickle.Unpickler):
        def persistent_load(self, pid):
            assert pid[0] == "storage", pid
            cls, key, loc, numel = pid[1], str(pid[2]), pid[3], pid[4]
            assert loc == "cpu", f"unexpected location {loc}"
            return (cls, key, numel)

    state = Unp(io.BytesIO(z.read(pkl_name))).load()
    assert isinstance(state, dict), f"root is {type(state)}, expected dict"

    out = {}
    for name, t in state.items():
        assert isinstance(t, Rebuilt), f"{name} is {type(t)}"
        cls, key, numel = t.storage
        np_dt, esz = STORAGE_NP[cls]
        raw = z.read(f"{root}data/{key}")
        assert len(raw) == numel * esz, f"{name}: record is {len(raw)} bytes, want {numel*esz}"
        assert t.offset == 0, f"{name}: unexpected storage_offset {t.offset}"
        exp_stride, acc = [], 1
        for d in reversed(t.size):
            exp_stride.append(acc); acc *= d
        assert t.stride == tuple(reversed(exp_stride)), f"{name}: stride {t.stride} not contiguous for {t.size}"
        assert np.prod(t.size) if t.size else 1, None
        assert int(np.prod(t.size)) == numel, f"{name}: shape {t.size} has {np.prod(t.size)} elems, storage says {numel}"
        out[name] = (raw, t.size, cls.__name__, records)
    return out, z, root


def alignment_report(path, root):
    """torch.load(mmap=True) maps records in place, so payloads must be 64-byte aligned."""
    z = zipfile.ZipFile(path)
    bad = []
    with open(path, "rb") as f:
        for info in z.infolist():
            if f"{root}data/" not in info.filename:
                continue
            f.seek(info.header_offset)
            hdr = f.read(30)
            assert struct.unpack("<I", hdr[:4])[0] == 0x04034B50
            n, e = struct.unpack("<HH", hdr[26:30])
            data_off = info.header_offset + 30 + n + e
            if data_off % 64 != 0:
                bad.append((info.filename, data_off, data_off % 64))
    return bad


if __name__ == "__main__":
    path, expect_json = sys.argv[1], sys.argv[2]
    tensors, z, root = load(path)
    expect = json.load(open(expect_json))

    assert set(tensors) == set(expect["names"]), (
        f"key mismatch\n  only in pth: {sorted(set(tensors)-set(expect['names']))}\n"
        f"  only expected: {sorted(set(expect['names'])-set(tensors))}"
    )
    for name, (raw, size, cls, _) in tensors.items():
        want_hex = expect["bytes"][name]
        assert raw.hex() == want_hex, f"{name}: tensor bytes differ from the source shard"
        assert list(size) == expect["shapes"][name], f"{name}: shape {size} != {expect['shapes'][name]}"
        assert cls == expect["storage"][name], f"{name}: storage {cls} != {expect['storage'][name]}"

    bad = alignment_report(path, root)
    assert not bad, f"records not 64-byte aligned (breaks torch.load(mmap=True)): {bad[:3]}"

    extra = [n for n in z.namelist() if n.endswith(("version", "byteorder"))]
    print(f"OK  {len(tensors)} tensors byte-identical, CRCs valid, all records 64-byte aligned")
    print(f"    archive entries present: {sorted(x.split('/')[-1] for x in extra)}")
