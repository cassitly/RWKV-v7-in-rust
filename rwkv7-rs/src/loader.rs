//! Turns a checkpoint file into `name -> f32 tensor`.
//!
//! Supports both `torch.save` zip archives (`.pth`) and `.safetensors`.
//! Every tensor is up-converted to f32 as it is read, one at a time, so peak
//! memory is the f32 model plus a single tensor rather than two full copies.

use std::collections::HashMap;
use std::io;

use crate::pickle::{self, Value};
use crate::zip::ZipArchive;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    U8,
}

impl DType {
    fn size(self) -> usize {
        match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::U8 => 1,
        }
    }

    fn from_torch_storage(name: &str) -> Option<DType> {
        Some(match name {
            "DoubleStorage" => DType::F64,
            "FloatStorage" => DType::F32,
            "HalfStorage" => DType::F16,
            "BFloat16Storage" => DType::BF16,
            "LongStorage" => DType::I64,
            "IntStorage" => DType::I32,
            "ByteStorage" | "BoolStorage" | "CharStorage" => DType::U8,
            _ => return None,
        })
    }

    fn from_safetensors(name: &str) -> Option<DType> {
        Some(match name {
            "F64" => DType::F64,
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "I64" => DType::I64,
            "I32" => DType::I32,
            "U8" | "BOOL" | "I8" => DType::U8,
            _ => return None,
        })
    }
}

#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if frac == 0 => sign << 31,
        // Subnormal half: shift the mantissa up until the implicit 1 appears at bit
        // 10, then drop it. After k shifts the value is mantissa * 2^(-14-k), so the
        // single-precision exponent field is (127-14) - k.
        0 => {
            let mut e: i32 = 0;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            let f = f & 0x3ff;
            (sign << 31) | (((127 - 14 + e) as u32) << 23) | (f << 13)
        }
        0x1f => (sign << 31) | (0xff << 23) | (frac << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

#[inline]
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Decode `raw` (little-endian, `dtype`) into f32.
fn decode(raw: &[u8], dtype: DType) -> Vec<f32> {
    let n = raw.len() / dtype.size();
    let mut out = Vec::with_capacity(n);
    match dtype {
        DType::F32 => {
            for c in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        DType::F16 => {
            for c in raw.chunks_exact(2) {
                out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
            }
        }
        DType::BF16 => {
            for c in raw.chunks_exact(2) {
                out.push(bf16_to_f32(u16::from_le_bytes([c[0], c[1]])));
            }
        }
        DType::F64 => {
            for c in raw.chunks_exact(8) {
                let mut a = [0u8; 8];
                a.copy_from_slice(c);
                out.push(f64::from_le_bytes(a) as f32);
            }
        }
        DType::I64 => {
            for c in raw.chunks_exact(8) {
                let mut a = [0u8; 8];
                a.copy_from_slice(c);
                out.push(i64::from_le_bytes(a) as f32);
            }
        }
        DType::I32 => {
            for c in raw.chunks_exact(4) {
                out.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32);
            }
        }
        DType::U8 => {
            for c in raw {
                out.push(*c as f32);
            }
        }
    }
    out
}

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    /// Drop leading/trailing unit axes, mirroring the `.squeeze()` the Python demos do.
    pub fn squeezed_shape(&self) -> Vec<usize> {
        let s: Vec<usize> = self.shape.iter().copied().filter(|d| *d != 1).collect();
        if s.is_empty() {
            vec![1]
        } else {
            s
        }
    }
}

pub type Tensors = HashMap<String, Tensor>;

fn err<T>(msg: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, msg.into()))
}

pub fn load(path: &str) -> io::Result<Tensors> {
    if path.ends_with(".safetensors") {
        load_safetensors(path)
    } else {
        load_pth(path)
    }
}

// ---------------------------------------------------------------------------
// torch.save zip archives
// ---------------------------------------------------------------------------

/// The `('storage', <XStorage>, key, location, numel)` tuple a BINPERSID carries.
struct StorageRef {
    key: String,
    dtype: DType,
}

fn parse_storage(v: &Value) -> Result<StorageRef, String> {
    let Value::PersId(inner) = v else {
        return Err("its argument is not a persistent id".into());
    };
    let Some(items) = inner.as_seq() else {
        return Err("its persistent id is not a tuple".into());
    };
    if items.len() < 3 || items[0].as_str() != Some("storage") {
        return Err("its persistent id is not a storage record".into());
    }
    let dtype = match &items[1] {
        Value::Global(g) => DType::from_torch_storage(&g.1).ok_or_else(|| {
            format!(
                "dtype {} is not supported by this loader (it decodes to f32;                  use qwen-make-pth for byte-preserving conversion)",
                g.1
            )
        })?,
        _ => return Err("its storage class is not a global".into()),
    };
    let key = match &items[2] {
        Value::Str(s) => s.as_str().to_string(),
        Value::Int(i) => i.to_string(),
        _ => return Err("its storage key is not a scalar".into()),
    };
    Ok(StorageRef { key, dtype })
}

fn load_pth(path: &str) -> io::Result<Tensors> {
    let mut zip = ZipArchive::open(path)?;

    // torch nests everything under a single top-level directory whose name varies.
    let Some(pkl_name) = zip
        .names()
        .iter()
        .find(|n| n.ends_with("data.pkl"))
        .cloned()
    else {
        return err(
            "no data.pkl in archive — this does not look like a torch.save checkpoint. \
             Legacy (pre-1.6) .pth files are not supported; re-save with a modern PyTorch \
             or convert to .safetensors.",
        );
    };
    let root = pkl_name.trim_end_matches("data.pkl").to_string();

    let pkl = zip.read(&pkl_name)?;
    let obj = pickle::load(&pkl)?;

    let Value::Dict(entries) = &obj else {
        return err("checkpoint root is not a dict (expected a state_dict)");
    };

    let mut out: Tensors = HashMap::new();
    for (k, v) in entries.borrow().iter() {
        let Some(name) = k.as_str() else { continue };

        // torch._utils._rebuild_tensor_v2(storage, offset, size, stride, ...)
        let Value::Reduce(rc) = v else { continue };
        let (_module, func, args) = rc.as_ref();
        if !func.starts_with("_rebuild_tensor") {
            continue;
        }
        if args.len() < 4 {
            return err(format!(
                "{name}: unexpected _rebuild_tensor arity {}",
                args.len()
            ));
        }

        let storage = match parse_storage(&args[0]) {
            Ok(s) => s,
            Err(why) => return err(format!("{name}: {why}")),
        };
        let offset = args[1].as_usize().unwrap_or(0);
        let shape: Vec<usize> = args[2]
            .as_seq()
            .unwrap_or_default()
            .iter()
            .filter_map(|d| d.as_usize())
            .collect();
        let stride: Vec<usize> = args[3]
            .as_seq()
            .unwrap_or_default()
            .iter()
            .filter_map(|d| d.as_usize())
            .collect();

        let numel: usize =
            shape
                .iter()
                .product::<usize>()
                .max(if shape.is_empty() { 1 } else { 0 });

        // Only contiguous views are supported; checkpoint weights always are.
        let mut expect = vec![0usize; shape.len()];
        let mut acc = 1usize;
        for i in (0..shape.len()).rev() {
            expect[i] = acc;
            acc *= shape[i];
        }
        if stride.len() == expect.len() && numel > 1 && stride != expect {
            return err(format!(
                "{name}: non-contiguous tensor (shape {shape:?}, stride {stride:?}) is not supported"
            ));
        }

        let record = format!("{root}data/{}", storage.key);
        let byte_off = (offset * storage.dtype.size()) as u64;
        let mut raw = vec![0u8; numel * storage.dtype.size()];
        zip.read_range(&record, byte_off, &mut raw)?;

        out.insert(
            name.to_string(),
            Tensor {
                shape,
                data: decode(&raw, storage.dtype),
            },
        );
    }

    if out.is_empty() {
        return err("no tensors found in checkpoint");
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// safetensors
// ---------------------------------------------------------------------------

fn load_safetensors(path: &str) -> io::Result<Tensors> {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};

    let mut f = File::open(path)?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8)?;
    let header_len = u64::from_le_bytes(len8) as usize;
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header)?;
    let header = String::from_utf8_lossy(&header).into_owned();

    let mut out: Tensors = HashMap::new();
    // The header is flat JSON: {"name": {"dtype":..,"shape":[..],"data_offsets":[a,b]}, ...}.
    // A tiny scanner beats pulling in a JSON crate for this shape.
    let bytes = header.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        let start = i + 1;
        let mut j = start;
        while j < bytes.len() && bytes[j] != b'"' {
            j += 1;
        }
        let key = &header[start..j];
        i = j + 1;
        // Only top-level tensor entries are followed by `{`.
        let mut k = i;
        while k < bytes.len() && (bytes[k] == b' ' || bytes[k] == b':') {
            k += 1;
        }
        if k >= bytes.len() || bytes[k] != b'{' || key == "__metadata__" {
            continue;
        }
        let mut depth = 0i32;
        let obj_start = k;
        while k < bytes.len() {
            if bytes[k] == b'{' {
                depth += 1;
            } else if bytes[k] == b'}' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            k += 1;
        }
        let obj = &header[obj_start..=k.min(bytes.len() - 1)];
        i = k + 1;

        let field = |name: &str| -> Option<String> {
            let p = obj.find(&format!("\"{name}\""))?;
            let rest = &obj[p + name.len() + 2..];
            let c = rest.find(':')? + 1;
            Some(rest[c..].trim_start().to_string())
        };
        let (Some(dt), Some(sh), Some(off)) =
            (field("dtype"), field("shape"), field("data_offsets"))
        else {
            continue;
        };
        let dt = dt.trim().trim_start_matches('"');
        let dt = &dt[..dt.find('"').unwrap_or(dt.len())];
        let Some(dtype) = DType::from_safetensors(dt) else {
            return err(format!("{key}: unsupported safetensors dtype {dt}"));
        };
        let nums = |s: &str| -> Vec<usize> {
            let s = &s[s.find('[').map(|p| p + 1).unwrap_or(0)..];
            let s = &s[..s.find(']').unwrap_or(s.len())];
            s.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect()
        };
        let shape = nums(&sh);
        let offs = nums(&off);
        if offs.len() != 2 {
            continue;
        }

        let mut raw = vec![0u8; offs[1] - offs[0]];
        f.seek(SeekFrom::Start(8 + header_len as u64 + offs[0] as u64))?;
        f.read_exact(&mut raw)?;
        out.insert(
            key.to_string(),
            Tensor {
                shape,
                data: decode(&raw, dtype),
            },
        );
    }

    if out.is_empty() {
        return err("no tensors found in safetensors file");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision_conversions() {
        // (bits, value) pairs covering zero, subnormals, powers of two and the max finite.
        for (bits, want) in [
            (0x0000u16, 0.0f32),
            (0x8000, -0.0),
            (0x3c00, 1.0),
            (0xbc00, -1.0),
            (0x4000, 2.0),
            (0x3555, 0.333_251_95),
            (0x0400, 6.103_515_6e-5), // smallest normal
            (0x0001, 5.960_464_5e-8), // smallest subnormal
            (0x03ff, 6.097_555_2e-5), // largest subnormal
            (0x7bff, 65504.0),        // largest finite
        ] {
            let got = f16_to_f32(bits);
            assert!(
                (got - want).abs() <= want.abs() * 1e-6,
                "f16 {bits:#06x}: got {got:e}, want {want:e}"
            );
        }
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn bfloat16_conversions() {
        // bf16 is just the top 16 bits of an f32, so truncation must round-trip.
        for v in [0.0f32, 1.0, -1.0, 2.5, 1e-30, 1e30, -0.333] {
            let bits = (v.to_bits() >> 16) as u16;
            let back = bf16_to_f32(bits);
            assert!(
                (back - v).abs() <= v.abs() * 0.01,
                "bf16 round-trip of {v}: got {back}"
            );
        }
    }
}
