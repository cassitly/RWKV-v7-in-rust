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

use rwkv7::cli::export::{self, Args, Format};
use rwkv7::hf::{self, Filtered, ShardReader, ShardTensor};
use rwkv7::json::Out;
use rwkv7::pickle_write;

fn dtype_counts(tensors: &[ShardTensor]) -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    for t in tensors {
        *m.entry(pickle_write::torch_dtype_name(&t.dtype).to_string())
            .or_insert(0) += 1;
    }
    m
}

/// Keep the sidecar and stdout report consistent as summary fields evolve.
fn tensor_summary(filtered: &Filtered) -> BTreeMap<String, Out> {
    let dtypes = dtype_counts(&filtered.kept)
        .into_iter()
        .map(|(name, count)| (name, Out::Num(count)))
        .collect();
    BTreeMap::from([
        ("kept_tensors".into(), Out::Num(filtered.kept.len() as i64)),
        (
            "skipped_vision_tensors".into(),
            Out::Num(filtered.skipped_vision as i64),
        ),
        (
            "skipped_mtp_tensors".into(),
            Out::Num(filtered.skipped_mtp as i64),
        ),
        (
            "numel".into(),
            Out::Num(filtered.kept.iter().map(|t| t.numel()).sum::<usize>() as i64),
        ),
        ("dtypes".into(), Out::Obj(dtypes)),
    ])
}

fn run(args: Args) -> Result<(), String> {
    if args.dry_run && !Path::new(&args.model).is_dir() {
        return Err(format!(
            "--dry-run requires an existing local model directory: {}",
            args.model
        ));
    }
    let root = hf::resolve_model_dir(&args.model, args.local_files_only || args.dry_run)
        .map_err(|e| e.to_string())?;

    let config_path = root.join("config.json");
    if !config_path.exists() {
        return Err(format!("missing config.json in {}", root.display()));
    }
    // Passed through verbatim rather than parsed and re-emitted, so the sidecar
    // reproduces the model's own config byte-for-byte.
    let config_raw = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("{}: {e}", config_path.display()))?;
    rwkv7::json::parse(&config_raw).map_err(|e| format!("{}: {e}", config_path.display()))?;

    eprintln!("scanning {}", root.display());
    let all = hf::scan_dir(&root).map_err(|e| e.to_string())?;
    let filtered = hf::export_text_state(all).map_err(|e| e.to_string())?;
    let kept = &filtered.kept;
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
    match args.format {
        Format::Pth => hf::write_pth(&args.output, kept, &mut reader),
        Format::Safetensors => hf::write_safetensors(&args.output, kept, &mut reader),
    }
    .map_err(|e| format!("{}: {e}", args.output))?;

    // Sidecar: same fields as the Python script, sorted keys, indent 2.
    let mut meta = tensor_summary(&filtered);
    meta.insert("source_model".into(), Out::Str(args.model.clone()));
    meta.insert(
        "stripped_prefix".into(),
        Out::Str(hf::LANGUAGE_MODEL_PREFIX.into()),
    );
    meta.insert("config".into(), Out::Raw(config_raw.trim().to_string()));

    let sidecar = format!("{}.json", args.output);
    std::fs::write(&sidecar, Out::Obj(meta).to_string_pretty() + "\n")
        .map_err(|e| format!("{sidecar}: {e}"))?;

    // Same summary the Python script prints on stdout.
    let mut summary = tensor_summary(&filtered);
    summary.insert("output".into(), Out::Str(args.output.clone()));
    println!("{}", Out::Obj(summary).to_string_pretty());
    Ok(())
}

fn main() -> ExitCode {
    let result = export::parse(std::env::args().skip(1)).and_then(|args| match args {
        Some(args) => run(args),
        None => {
            print!("{}", export::USAGE);
            Ok(())
        }
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
