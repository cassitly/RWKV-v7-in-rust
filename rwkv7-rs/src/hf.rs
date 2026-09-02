//! Reading a HuggingFace checkpoint directory: shard discovery, tensor
//! inventory, and the text-model key filters.
//!
//! Nothing here reads tensor *data* — only headers — so a full inventory of a
//! 30B checkpoint costs a few hundred KB of I/O. Conversion then streams each
//! tensor straight from its shard to the output.

use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::json;
use crate::pickle::{self, Value};
use crate::zip::ZipArchive;

/// Prefixes the upstream script drops as vision/multimodal tensors.
pub const VISION_PREFIXES: &[&str] = &[
    "visual.",
    "model.visual.",
    "vision_tower.",
    "model.vision_tower.",
    "multi_modal_projector.",
    "model.multi_modal_projector.",
];
pub const LANGUAGE_MODEL_PREFIX: &str = "model.language_model.";
pub const MTP_PREFIX: &str = "mtp.";

pub fn is_vision_key(key: &str) -> bool {
    VISION_PREFIXES.iter().any(|p| key.starts_with(p))
}

pub fn strip_language_model_prefix(key: &str) -> &str {
    key.strip_prefix(LANGUAGE_MODEL_PREFIX).unwrap_or(key)
}

pub fn is_mtp_key(key: &str) -> bool {
    key.starts_with(MTP_PREFIX)
}

/// One tensor located inside a shard, described well enough to copy its bytes
/// without ever decoding them.
#[derive(Clone, Debug)]
pub struct ShardTensor {
    pub name: String,
    /// safetensors dtype spelling ("BF16", "F32", ...), preserved verbatim.
    pub dtype: String,
    pub shape: Vec<usize>,
    pub source: PathBuf,
    /// Absolute byte offset of the payload within `source`.
    pub offset: u64,
    pub nbytes: u64,
}

impl ShardTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(if self.shape.is_empty() { 1 } else { 0 })
    }
}

fn err<T>(msg: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, msg.into()))
}

// ---------------------------------------------------------------------------
// model directory resolution
// ---------------------------------------------------------------------------

/// Resolve a local directory, or a `Org/Model` repo id via the `hf` CLI.
///
/// Downloading over HTTPS is deliberately out of scope — it would mean shipping
/// a TLS stack for one code path — so a repo id is handed to `hf download`
/// when that is on PATH, and otherwise reported with the command to run.
pub fn resolve_model_dir(name: &str, local_files_only: bool) -> io::Result<PathBuf> {
    let p = Path::new(name);
    if p.is_dir() {
        return Ok(p.to_path_buf());
    }
    if p.exists() {
        return err(format!("{name} is a file, not a model directory"));
    }
    if local_files_only {
        return err(format!(
            "{name} is not a local directory, and --local-files-only was given"
        ));
    }
    if !name.contains('/') || name.contains(' ') {
        return err(format!("no such directory: {name}"));
    }

    for cli in ["hf", "huggingface-cli"] {
        let probe = std::process::Command::new(cli)
            .arg("--help")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        if !matches!(probe, Ok(s) if s.success()) {
            continue;
        }
        eprintln!("{name} is not a local path; fetching with `{cli} download`...");
        let out = std::process::Command::new(cli)
            .args(["download", name, "--include", "config.json", "--include", "*.safetensors"])
            .args(["--include", "*.safetensors.index.json"])
            .output()?;
        if !out.status.success() {
            return err(format!(
                "`{cli} download {name}` failed:\n{}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        // The CLI prints the snapshot directory on its last line of stdout.
        let path = String::from_utf8_lossy(&out.stdout)
            .lines()
            .rev()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .unwrap_or_default()
            .to_string();
        if Path::new(&path).is_dir() {
            return Ok(PathBuf::from(path));
        }
        return err(format!("`{cli} download` succeeded but its output path was not a directory: {path:?}"));
    }

    err(format!(
        "{name} is not a local directory and no HuggingFace CLI was found.\n\
         Download it first, then pass the directory:\n\
         \n    hf download {name} --include 'config.json' --include '*.safetensors*'\n\
         \n(or `pip install huggingface_hub` to let this tool call it for you)"
    ))
}

// ---------------------------------------------------------------------------
// shard discovery
// ---------------------------------------------------------------------------

fn glob_ext(root: &Path, ext: &str) -> io::Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(root)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == ext).unwrap_or(false))
        .collect();
    out.sort();
    Ok(out)
}

/// Files named by an index's `weight_map`, deduplicated and sorted — matching
/// the `sorted(set(...))` the Python script uses.
fn files_from_index(root: &Path, index_path: &Path) -> io::Result<Vec<PathBuf>> {
    let text = std::fs::read_to_string(index_path)?;
    let j = json::parse(&text).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", index_path.display())))?;
    let Some(map) = j.get("weight_map").and_then(|m| m.as_obj()) else {
        return err(format!("{}: no weight_map object", index_path.display()));
    };
    let mut names: Vec<String> = map.iter().filter_map(|(_, v)| v.as_str().map(str::to_string)).collect();
    names.sort();
    names.dedup();
    Ok(names.into_iter().map(|n| root.join(n)).collect())
}

/// Inventory every tensor in a model directory, in the same file order the
/// Python script would load them.
pub fn scan_dir(root: &Path) -> io::Result<Vec<ShardTensor>> {
    let st_index = root.join("model.safetensors.index.json");
    let bin_index = root.join("pytorch_model.bin.index.json");

    let (files, is_safetensors) = if st_index.exists() {
        (files_from_index(root, &st_index)?, true)
    } else if bin_index.exists() {
        (files_from_index(root, &bin_index)?, false)
    } else {
        let st = glob_ext(root, "safetensors")?;
        if !st.is_empty() {
            (st, true)
        } else {
            let bins = glob_ext(root, "bin")?;
            if bins.is_empty() {
                return err(format!("no HF weight files found in {}", root.display()));
            }
            (bins, false)
        }
    };

    let mut out = Vec::new();
    for f in &files {
        if !f.exists() {
            return err(format!("shard listed in the index is missing: {}", f.display()));
        }
        let mut part = if is_safetensors { scan_safetensors(f)? } else { scan_pth(f)? };
        out.append(&mut part);
    }
    Ok(out)
}

/// Read a safetensors header and record where each tensor's bytes live.
pub fn scan_safetensors(path: &Path) -> io::Result<Vec<ShardTensor>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8)?;
    let header_len = u64::from_le_bytes(len8);
    if header_len > 512 << 20 {
        return err(format!("{}: implausible safetensors header ({header_len} bytes)", path.display()));
    }
    let mut hb = vec![0u8; header_len as usize];
    f.seek(SeekFrom::Start(8))?;
    f.read_exact(&mut hb)?;
    let text = String::from_utf8_lossy(&hb).into_owned();
    let j = json::parse(&text)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", path.display())))?;
    let Some(obj) = j.as_obj() else {
        return err(format!("{}: safetensors header is not an object", path.display()));
    };

    let data_start = 8 + header_len;
    let mut out = Vec::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let (Some(dtype), Some(shape), Some(offs)) =
            (v.get("dtype").and_then(|d| d.as_str()), v.get("shape").and_then(|s| s.as_arr()), v.get("data_offsets").and_then(|o| o.as_arr()))
        else {
            return err(format!("{}: malformed entry for '{name}'", path.display()));
        };
        let shape: Vec<usize> = shape.iter().filter_map(|d| d.as_usize()).collect();
        if offs.len() != 2 {
            return err(format!("{}: '{name}' has {} data_offsets, expected 2", path.display(), offs.len()));
        }
        let (a, b) = (offs[0].as_usize().unwrap_or(0) as u64, offs[1].as_usize().unwrap_or(0) as u64);
        out.push(ShardTensor {
            name: name.clone(),
            dtype: dtype.to_string(),
            shape,
            source: path.to_path_buf(),
            offset: data_start + a,
            nbytes: b.saturating_sub(a),
        });
    }
    // Header order is arbitrary; sort so conversion is deterministic.
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Same, for a `torch.save` shard: walk the pickle for tensor metadata, then
/// resolve each storage record to an absolute file offset.
pub fn scan_pth(path: &Path) -> io::Result<Vec<ShardTensor>> {
    let path_s = path.to_string_lossy().into_owned();
    let mut zip = ZipArchive::open(&path_s)?;
    let Some(pkl_name) = zip.names().iter().find(|n| n.ends_with("data.pkl")).cloned() else {
        return err(format!(
            "{}: no data.pkl — legacy (pre-1.6) torch files are not supported",
            path.display()
        ));
    };
    let root = pkl_name.trim_end_matches("data.pkl").to_string();
    let obj = pickle::load(&zip.read(&pkl_name)?)?;

    // Some checkpoints wrap the tensors in {"state_dict": {...}}.
    let entries = match &obj {
        Value::Dict(d) => {
            let inner = d.borrow().iter().find_map(|(k, v)| {
                matches!(k.as_str(), Some("state_dict")).then(|| v.clone())
            });
            match inner {
                Some(Value::Dict(d2)) => d2,
                _ => d.clone(),
            }
        }
        _ => return err(format!("{}: checkpoint root is not a dict", path.display())),
    };

    let mut out = Vec::new();
    for (k, v) in entries.borrow().iter() {
        let Some(name) = k.as_str() else { continue };
        let Value::Reduce(rc) = v else { continue };
        if !rc.1.starts_with("_rebuild_tensor") || rc.2.len() < 4 {
            continue;
        }
        let args = &rc.2;

        let Value::PersId(pid) = &args[0] else {
            return err(format!("{}: '{name}' has no storage record", path.display()));
        };
        let items = pid.as_seq().unwrap_or_default();
        if items.len() < 3 {
            return err(format!("{}: '{name}' has a malformed storage record", path.display()));
        }
        let Value::Global(g) = &items[1] else {
            return err(format!("{}: '{name}' storage class is not a global", path.display()));
        };
        let dtype = safetensors_dtype_for_storage(&g.1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("{}: '{name}' has unsupported storage {}", path.display(), g.1)))?;
        let key = match &items[2] {
            Value::Str(s) => s.as_str().to_string(),
            Value::Int(i) => i.to_string(),
            _ => return err(format!("{}: '{name}' has a non-scalar storage key", path.display())),
        };

        let storage_offset = args[1].as_usize().unwrap_or(0);
        let shape: Vec<usize> = args[2].as_seq().unwrap_or_default().iter().filter_map(|d| d.as_usize()).collect();
        let stride: Vec<usize> = args[3].as_seq().unwrap_or_default().iter().filter_map(|d| d.as_usize()).collect();

        let expect = crate::pickle_write::contiguous_stride(&shape);
        let numel: usize = shape.iter().product();
        if numel > 1 && stride.len() == expect.len() && stride != expect {
            return err(format!(
                "{}: '{name}' is non-contiguous (shape {shape:?}, stride {stride:?}); \
                 re-save it with .contiguous() first",
                path.display()
            ));
        }

        let esz = crate::pickle_write::dtype_size(dtype).unwrap_or(1);
        let (rec_off, rec_len) = zip.data_offset(&format!("{root}data/{key}"))?;
        let want = (numel * esz) as u64;
        let start = (storage_offset * esz) as u64;
        if start + want > rec_len {
            return err(format!("{}: '{name}' reads past the end of its storage record", path.display()));
        }
        out.push(ShardTensor {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape,
            source: path.to_path_buf(),
            offset: rec_off + start,
            nbytes: want,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn safetensors_dtype_for_storage(class: &str) -> Option<&'static str> {
    Some(match class {
        "DoubleStorage" => "F64",
        "FloatStorage" => "F32",
        "HalfStorage" => "F16",
        "BFloat16Storage" => "BF16",
        "LongStorage" => "I64",
        "IntStorage" => "I32",
        "ShortStorage" => "I16",
        "CharStorage" => "I8",
        "ByteStorage" => "U8",
        "BoolStorage" => "BOOL",
        "Float8_e4m3fnStorage" => "F8_E4M3",
        "Float8_e5m2Storage" => "F8_E5M2",
        _ => return None,
    })
}

/// The result of applying the text-model filters.
pub struct Filtered {
    pub kept: Vec<ShardTensor>,
    pub skipped_vision: usize,
    pub skipped_mtp: usize,
}

/// Drop vision tensors, strip the language-model prefix, then drop MTP tensors —
/// in that order, because the MTP check runs on the *stripped* key.
pub fn export_text_state(tensors: Vec<ShardTensor>) -> io::Result<Filtered> {
    let mut kept: Vec<ShardTensor> = Vec::with_capacity(tensors.len());
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let (mut skipped_vision, mut skipped_mtp) = (0usize, 0usize);

    for mut t in tensors {
        if is_vision_key(&t.name) {
            skipped_vision += 1;
            continue;
        }
        let stripped = strip_language_model_prefix(&t.name).to_string();
        if is_mtp_key(&stripped) {
            skipped_mtp += 1;
            continue;
        }
        if !seen.insert(stripped.clone()) {
            return err(format!("duplicate exported key after prefix stripping: {stripped}"));
        }
        t.name = stripped;
        kept.push(t);
    }
    Ok(Filtered { kept, skipped_vision, skipped_mtp })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_match_the_python_rules() {
        assert!(is_vision_key("visual.blocks.0.attn.qkv.weight"));
        assert!(is_vision_key("model.vision_tower.embeddings.weight"));
        assert!(is_vision_key("multi_modal_projector.linear_1.weight"));
        assert!(!is_vision_key("model.layers.0.self_attn.q_proj.weight"));
        // "visualise" must not be caught: the prefixes all end in a dot.
        assert!(!is_vision_key("visualiser.weight"));

        assert_eq!(strip_language_model_prefix("model.language_model.layers.0.x"), "layers.0.x");
        assert_eq!(strip_language_model_prefix("model.layers.0.x"), "model.layers.0.x");

        assert!(is_mtp_key("mtp.layers.0.weight"));
        assert!(!is_mtp_key("model.mtp.weight"));
    }

    fn t(name: &str) -> ShardTensor {
        ShardTensor {
            name: name.into(),
            dtype: "BF16".into(),
            shape: vec![2, 2],
            source: PathBuf::from("x"),
            offset: 0,
            nbytes: 8,
        }
    }

    #[test]
    fn export_applies_filters_in_order() {
        let f = export_text_state(vec![
            t("model.language_model.layers.0.w"),
            t("visual.patch_embed.w"),
            t("model.language_model.mtp.layers.0.w"), // MTP only visible after stripping
            t("lm_head.weight"),
        ])
        .unwrap();
        assert_eq!(f.skipped_vision, 1);
        assert_eq!(f.skipped_mtp, 1);
        let names: Vec<&str> = f.kept.iter().map(|k| k.name.as_str()).collect();
        assert_eq!(names, vec!["layers.0.w", "lm_head.weight"]);
    }

    #[test]
    fn export_rejects_collisions_created_by_stripping() {
        let e = export_text_state(vec![t("model.language_model.layers.0.w"), t("layers.0.w")]);
        assert!(e.is_err(), "a stripped key colliding with an existing one must be an error");
    }
}

// ---------------------------------------------------------------------------
// writing
// ---------------------------------------------------------------------------

/// Keeps one shard file open across the tensors that come from it, so a 40-shard
/// checkpoint does not reopen a file per tensor.
pub struct ShardReader {
    path: PathBuf,
    file: Option<std::fs::File>,
}

impl ShardReader {
    pub fn new() -> Self {
        ShardReader { path: PathBuf::new(), file: None }
    }

    fn for_path(&mut self, p: &Path) -> io::Result<&mut std::fs::File> {
        if self.file.is_none() || self.path != p {
            self.file = Some(std::fs::File::open(p)?);
            self.path = p.to_path_buf();
        }
        Ok(self.file.as_mut().unwrap())
    }
}

impl Default for ShardReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Write `kept` as a `torch.save` archive, streaming each tensor's bytes
/// unchanged from its source shard.
pub fn write_pth(out: &str, kept: &[ShardTensor], reader: &mut ShardReader) -> io::Result<()> {
    use crate::pickle_write::{self, TensorRef};
    use crate::zip_write::ZipWriter;

    // Header scanning already gave us every tensor's metadata, so the pickle can
    // be built up front and the payloads appended afterwards.
    let keys: Vec<String> = (0..kept.len()).map(|i| i.to_string()).collect();
    let mut refs = Vec::with_capacity(kept.len());
    for (i, t) in kept.iter().enumerate() {
        let Some(storage_class) = pickle_write::storage_class_for(&t.dtype) else {
            return err(format!("{}: dtype {} has no torch storage class", t.name, t.dtype));
        };
        refs.push(TensorRef {
            name: &t.name,
            storage_class,
            storage_key: &keys[i],
            shape: &t.shape,
            numel: t.numel(),
        });
    }
    let pkl = pickle_write::build(&refs);

    let mut z = ZipWriter::create(out)?;
    z.write_bytes("archive/data.pkl", &pkl)?;
    z.write_bytes("archive/byteorder", b"little")?;
    for (i, t) in kept.iter().enumerate() {
        let f = reader.for_path(&t.source)?;
        f.seek(SeekFrom::Start(t.offset))?;
        let mut src = f.take(t.nbytes);
        z.write_stream(&format!("archive/data/{i}"), &mut src, t.nbytes)?;
    }
    // torch writes the format version last.
    z.write_bytes("archive/version", b"3\n")?;
    z.finish()
}

/// Same, in safetensors format.
pub fn write_safetensors(out: &str, kept: &[ShardTensor], reader: &mut ShardReader) -> io::Result<()> {
    let mut header = String::from("{");
    let mut off: u64 = 0;
    for (i, t) in kept.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        crate::json::escape(&t.name, &mut header);
        header.push_str(&format!(
            ":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            t.dtype,
            t.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","),
            off,
            off + t.nbytes
        ));
        off += t.nbytes;
    }
    header.push('}');
    // The spec requires the data section to start 8-byte aligned.
    while (header.len() + 8) % 8 != 0 {
        header.push(' ');
    }

    let mut f = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(out)?);
    f.write_all(&(header.len() as u64).to_le_bytes())?;
    f.write_all(header.as_bytes())?;

    let mut buf = vec![0u8; 1 << 20];
    for t in kept {
        let src = reader.for_path(&t.source)?;
        src.seek(SeekFrom::Start(t.offset))?;
        let mut left = t.nbytes;
        while left > 0 {
            let n = left.min(buf.len() as u64) as usize;
            src.read_exact(&mut buf[..n])?;
            f.write_all(&buf[..n])?;
            left -= n as u64;
        }
    }
    f.flush()
}

/// Read a tensor's raw bytes straight out of its shard.
pub fn read_tensor_bytes(t: &ShardTensor, reader: &mut ShardReader) -> io::Result<Vec<u8>> {
    let f = reader.for_path(&t.source)?;
    f.seek(SeekFrom::Start(t.offset))?;
    let mut buf = vec![0u8; t.nbytes as usize];
    f.read_exact(&mut buf)?;
    Ok(buf)
}
