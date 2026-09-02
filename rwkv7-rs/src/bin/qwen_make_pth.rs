//! Export the text-model weights of a HuggingFace checkpoint.
//!
//! A Rust port of `run_qwen35_make_pth.py`: read an HF model directory, drop
//! vision and MTP tensors, strip the `model.language_model.` prefix, and write
//! the result as a `torch.save` `.pth` (or safetensors), plus a `.json` sidecar.
//!
//! Original dtypes are preserved — tensor bytes are copied verbatim, never
//! decoded — and each tensor is streamed from its shard to the output, so peak
//! memory is one buffer rather than the whole state dict.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;

use rwkv7::hf::{self, ShardReader, ShardTensor};
use rwkv7::json::Out;
use rwkv7::pickle_write;

const USAGE: &str = "\
qwen-make-pth — export the text-model weights of a HuggingFace checkpoint

USAGE:
    qwen-make-pth [--model <DIR|REPO_ID>] --output <FILE> [OPTIONS]

OPTIONS:
    -m, --model <DIR>        HF model directory, or a repo id to fetch via the
                             `hf` CLI                    [default: Qwen/Qwen3.5-0.8B]
    -o, --output <FILE>      Output path                                [required]
        --format <FMT>       pth | safetensors                          [pth]
        --local-files-only   Never shell out to download
        --dry-run            Report what would be written, touch nothing
    -h, --help               Show this help

Reads sharded .safetensors or .bin, via *.index.json when present. Tensor bytes
are copied without conversion, so every dtype (including fp8) round-trips.
";

struct Args {
    model: String,
    output: String,
    format: String,
    local_files_only: bool,
    dry_run: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        model: "Qwen/Qwen3.5-0.8B".into(),
        output: String::new(),
        format: "pth".into(),
        local_files_only: false,
        dry_run: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let k = argv[i].as_str();
        let mut val = |name: &str| -> Result<String, String> {
            i += 1;
            argv.get(i).cloned().ok_or_else(|| format!("{name} needs a value"))
        };
        match k {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "-m" | "--model" => a.model = val(k)?,
            "-o" | "--output" => a.output = val(k)?,
            "--format" => a.format = val(k)?,
            "--local-files-only" => a.local_files_only = true,
            "--dry-run" => a.dry_run = true,
            other => return Err(format!("unknown argument '{other}'\n\n{USAGE}")),
        }
        i += 1;
    }
    if a.output.is_empty() && !a.dry_run {
        return Err(format!("--output is required\n\n{USAGE}"));
    }
    if a.format != "pth" && a.format != "safetensors" {
        return Err(format!("--format must be 'pth' or 'safetensors', got '{}'", a.format));
    }
    Ok(a)
}

fn dtype_counts(tensors: &[ShardTensor]) -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    for t in tensors {
        *m.entry(pickle_write::torch_dtype_name(&t.dtype).to_string()).or_insert(0) += 1;
    }
    m
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let root = hf::resolve_model_dir(&args.model, args.local_files_only).map_err(|e| e.to_string())?;

    let config_path = root.join("config.json");
    if !config_path.exists() {
        return Err(format!("missing config.json in {}", root.display()));
    }
    // Passed through verbatim rather than parsed and re-emitted, so the sidecar
    // reproduces the model's own config byte-for-byte.
    let config_raw = std::fs::read_to_string(&config_path).map_err(|e| format!("{}: {e}", config_path.display()))?;
    rwkv7::json::parse(&config_raw).map_err(|e| format!("{}: {e}", config_path.display()))?;

    eprintln!("scanning {}", root.display());
    let all = hf::scan_dir(&root).map_err(|e| e.to_string())?;
    let filtered = hf::export_text_state(all).map_err(|e| e.to_string())?;
    let kept = filtered.kept;
    if kept.is_empty() {
        return Err("no tensors left after filtering".into());
    }

    let numel: usize = kept.iter().map(|t| t.numel()).sum();
    let total_bytes: u64 = kept.iter().map(|t| t.nbytes).sum();
    eprintln!(
        "  {} tensors kept, {} vision / {} mtp skipped, {numel} elements, {:.2} GiB",
        kept.len(),
        filtered.skipped_vision,
        filtered.skipped_mtp,
        total_bytes as f64 / (1u64 << 30) as f64
    );

    if args.dry_run {
        eprintln!("--dry-run: nothing written");
        return Ok(());
    }

    let out = Path::new(&args.output);
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
    }

    let mut reader = ShardReader::new();
    match args.format.as_str() {
        "pth" => hf::write_pth(&args.output, &kept, &mut reader),
        _ => hf::write_safetensors(&args.output, &kept, &mut reader),
    }
    .map_err(|e| format!("{}: {e}", args.output))?;

    // Sidecar: same fields as the Python script, sorted keys, indent 2.
    let mut meta: BTreeMap<String, Out> = BTreeMap::new();
    meta.insert("source_model".into(), Out::Str(args.model.clone()));
    meta.insert("kept_tensors".into(), Out::Num(kept.len() as i64));
    meta.insert("skipped_vision_tensors".into(), Out::Num(filtered.skipped_vision as i64));
    meta.insert("skipped_mtp_tensors".into(), Out::Num(filtered.skipped_mtp as i64));
    meta.insert("stripped_prefix".into(), Out::Str(hf::LANGUAGE_MODEL_PREFIX.into()));
    meta.insert("numel".into(), Out::Num(numel as i64));
    let mut dt: BTreeMap<String, Out> = BTreeMap::new();
    for (k, v) in dtype_counts(&kept) {
        dt.insert(k, Out::Num(v));
    }
    meta.insert("dtypes".into(), Out::Obj(dt));
    meta.insert("config".into(), Out::Raw(config_raw.trim().to_string()));

    let sidecar = format!("{}.json", args.output);
    std::fs::write(&sidecar, Out::Obj(meta).to_string_pretty() + "\n")
        .map_err(|e| format!("{sidecar}: {e}"))?;

    // Same summary the Python script prints on stdout.
    let mut summary: BTreeMap<String, Out> = BTreeMap::new();
    summary.insert("output".into(), Out::Str(args.output.clone()));
    summary.insert("kept_tensors".into(), Out::Num(kept.len() as i64));
    summary.insert("skipped_vision_tensors".into(), Out::Num(filtered.skipped_vision as i64));
    summary.insert("skipped_mtp_tensors".into(), Out::Num(filtered.skipped_mtp as i64));
    summary.insert("numel".into(), Out::Num(numel as i64));
    let mut dt2: BTreeMap<String, Out> = BTreeMap::new();
    for (k, v) in dtype_counts(&kept) {
        dt2.insert(k, Out::Num(v));
    }
    summary.insert("dtypes".into(), Out::Obj(dt2));
    println!("{}", Out::Obj(summary).to_string_pretty());
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
