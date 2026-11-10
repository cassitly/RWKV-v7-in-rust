//! Emits the pickle stream that `torch.save` produces for a flat state dict.
//!
//! The output is a protocol-2 stream of exactly the shape torch's own writer
//! makes, so it loads under the restricted `weights_only=True` unpickler (which
//! allowlists `torch._utils._rebuild_tensor_v2`, `collections.OrderedDict` and
//! the legacy storage classes — the only three globals referenced here).
//!
//! Per tensor:
//!
//! ```text
//! torch._utils._rebuild_tensor_v2(
//!     persistent_id(('storage', torch.<X>Storage, key, 'cpu', numel)),
//!     0, size, stride, False, OrderedDict())
//! ```

const PROTO: u8 = 0x80;
const EMPTY_DICT: u8 = b'}';
const MARK: u8 = b'(';
const TUPLE: u8 = b't';
const EMPTY_TUPLE: u8 = b')';
const SETITEMS: u8 = b'u';
const BININT: u8 = b'J';
const BININT1: u8 = b'K';
const BININT2: u8 = b'M';
const SHORT_BINUNICODE: u8 = 0x8c;
const BINUNICODE: u8 = b'X';
const GLOBAL: u8 = b'c';
const REDUCE: u8 = b'R';
const BINPERSID: u8 = b'Q';
const NEWFALSE: u8 = 0x89;
const STOP: u8 = b'.';

/// One tensor's serialisation metadata. `storage_key` names the `data/<key>`
/// record that holds its raw bytes.
pub struct TensorRef<'a> {
    pub name: &'a str,
    /// Legacy storage class name, e.g. "BFloat16Storage".
    pub storage_class: &'a str,
    pub storage_key: &'a str,
    pub shape: &'a [usize],
    pub numel: usize,
}

fn int(out: &mut Vec<u8>, v: i64) {
    // Match CPython's pickle: narrowest opcode that fits.
    if (0..=0xFF).contains(&v) {
        out.push(BININT1);
        out.push(v as u8);
    } else if (0..=0xFFFF).contains(&v) {
        out.push(BININT2);
        out.extend_from_slice(&(v as u16).to_le_bytes());
    } else {
        out.push(BININT);
        out.extend_from_slice(&(v as i32).to_le_bytes());
    }
}

fn unicode(out: &mut Vec<u8>, s: &str) {
    let b = s.as_bytes();
    if b.len() < 256 {
        out.push(SHORT_BINUNICODE);
        out.push(b.len() as u8);
    } else {
        out.push(BINUNICODE);
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
    }
    out.extend_from_slice(b);
}

fn global(out: &mut Vec<u8>, module: &str, name: &str) {
    out.push(GLOBAL);
    out.extend_from_slice(module.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(name.as_bytes());
    out.push(b'\n');
}

fn tuple_of_ints(out: &mut Vec<u8>, vals: &[usize]) {
    out.push(MARK);
    for v in vals {
        int(out, *v as i64);
    }
    out.push(TUPLE);
}

/// Row-major (C-contiguous) strides for `shape`.
pub fn contiguous_stride(shape: &[usize]) -> Vec<usize> {
    let mut st = vec![1usize; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        st[i] = acc;
        acc *= shape[i];
    }
    st
}

pub fn build(tensors: &[TensorRef]) -> Vec<u8> {
    let mut out = Vec::with_capacity(256 + tensors.len() * 160);
    out.push(PROTO);
    out.push(2);
    out.push(EMPTY_DICT);
    out.push(MARK);

    for t in tensors {
        unicode(&mut out, t.name);

        // callable
        global(&mut out, "torch._utils", "_rebuild_tensor_v2");

        // args tuple
        out.push(MARK);

        // 1. storage, as a persistent id
        out.push(MARK);
        unicode(&mut out, "storage");
        global(&mut out, "torch", t.storage_class);
        unicode(&mut out, t.storage_key);
        unicode(&mut out, "cpu");
        int(&mut out, t.numel as i64);
        out.push(TUPLE);
        out.push(BINPERSID);

        // 2. storage_offset
        int(&mut out, 0);
        // 3. size, 4. stride
        tuple_of_ints(&mut out, t.shape);
        tuple_of_ints(&mut out, &contiguous_stride(t.shape));
        // 5. requires_grad
        out.push(NEWFALSE);
        // 6. backward_hooks
        global(&mut out, "collections", "OrderedDict");
        out.push(EMPTY_TUPLE);
        out.push(REDUCE);

        out.push(TUPLE);
        out.push(REDUCE);
    }

    out.push(SETITEMS);
    out.push(STOP);
    out
}

/// safetensors dtype string -> the legacy `torch.*Storage` class name that a
/// checkpoint's persistent id must reference.
pub fn storage_class_for(dtype: &str) -> Option<&'static str> {
    Some(match dtype {
        "F64" => "DoubleStorage",
        "F32" => "FloatStorage",
        "F16" => "HalfStorage",
        "BF16" => "BFloat16Storage",
        "I64" => "LongStorage",
        "I32" => "IntStorage",
        "I16" => "ShortStorage",
        "I8" => "CharStorage",
        "U8" => "ByteStorage",
        "BOOL" => "BoolStorage",
        "F8_E4M3" => "Float8_e4m3fnStorage",
        "F8_E5M2" => "Float8_e5m2Storage",
        _ => return None,
    })
}

/// Bytes per element for a safetensors dtype.
pub fn dtype_size(dtype: &str) -> Option<usize> {
    Some(match dtype {
        "F64" | "I64" => 8,
        "F32" | "I32" => 4,
        "F16" | "BF16" | "I16" => 2,
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" => 1,
        _ => return None,
    })
}

/// The torch dtype name that goes in the metadata sidecar (matches `str(t.dtype)`
/// with the `torch.` prefix stripped, as the Python script reports it).
pub fn torch_dtype_name(dtype: &str) -> &'static str {
    match dtype {
        "F64" => "float64",
        "F32" => "float32",
        "F16" => "float16",
        "BF16" => "bfloat16",
        "I64" => "int64",
        "I32" => "int32",
        "I16" => "int16",
        "I8" => "int8",
        "U8" => "uint8",
        "BOOL" => "bool",
        "F8_E4M3" => "float8_e4m3fn",
        "F8_E5M2" => "float8_e5m2",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pickle::{self, Value};

    #[test]
    fn round_trips_through_our_own_reader() {
        let shape = vec![2usize, 3];
        let stream = build(&[TensorRef {
            name: "model.layers.0.weight",
            storage_class: "BFloat16Storage",
            storage_key: "0",
            shape: &shape,
            numel: 6,
        }]);
        let v = pickle::load(&stream).expect("our reader should accept what we emit");
        let Value::Dict(d) = &v else {
            panic!("expected a dict")
        };
        let d = d.borrow();
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].0.as_str(), Some("model.layers.0.weight"));
        let Value::Reduce(rc) = &d[0].1 else {
            panic!("expected a reduce node")
        };
        assert_eq!(rc.1, "_rebuild_tensor_v2");
        assert_eq!(rc.2[1].as_usize(), Some(0));
        let size: Vec<usize> = rc.2[2]
            .as_seq()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_usize())
            .collect();
        let stride: Vec<usize> = rc.2[3]
            .as_seq()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_usize())
            .collect();
        assert_eq!(size, vec![2, 3]);
        assert_eq!(stride, vec![3, 1]);
    }

    #[test]
    fn strides_are_row_major() {
        assert_eq!(contiguous_stride(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(contiguous_stride(&[5]), vec![1]);
        assert_eq!(contiguous_stride(&[]), Vec::<usize>::new());
    }

    #[test]
    fn int_opcodes_narrow_correctly() {
        let mut v = Vec::new();
        int(&mut v, 5);
        assert_eq!(v, vec![BININT1, 5]);
        let mut v = Vec::new();
        int(&mut v, 300);
        assert_eq!(v[0], BININT2);
        let mut v = Vec::new();
        int(&mut v, 100_000);
        assert_eq!(v[0], BININT);
    }
}
