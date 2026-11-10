//! Options for exporting HuggingFace text weights.

use super::value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Pth,
    Safetensors,
}

#[derive(Debug)]
pub struct Args {
    pub model: String,
    pub output: String,
    pub format: Format,
    pub local_files_only: bool,
    pub dry_run: bool,
}

pub const USAGE: &str = "\
qwen-make-pth — export the text-model weights of a HuggingFace checkpoint

USAGE:
    qwen-make-pth [--model <DIR|REPO_ID>] --output <FILE> [OPTIONS]
    qwen-make-pth --model <DIR> --dry-run

OPTIONS:
    -m, --model <DIR>        HF model directory, or a repo id to fetch via the
                             `hf` CLI                    [default: Qwen/Qwen3.5-0.8B]
    -o, --output <FILE>      Output path                                [required]
        --format <FMT>       pth | safetensors                          [pth]
        --local-files-only   Never shell out to download
        --dry-run            Inspect a local model; never download or write files
    -h, --help               Show this help

Reads sharded .safetensors or .bin, via *.index.json when present. Tensor bytes
are copied without conversion, so every dtype (including fp8) round-trips.
";

/// Parse options without accessing the filesystem or invoking the HF CLI.
pub fn parse(argv: impl IntoIterator<Item = String>) -> Result<Option<Args>, String> {
    let mut args = Args {
        model: "Qwen/Qwen3.5-0.8B".into(),
        output: String::new(),
        format: Format::Pth,
        local_files_only: false,
        dry_run: false,
    };
    let mut argv = argv.into_iter();
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "-h" | "--help" => return Ok(None),
            "-m" | "--model" => args.model = value(&mut argv, &flag)?,
            "-o" | "--output" => args.output = value(&mut argv, &flag)?,
            "--format" => {
                args.format = match value(&mut argv, &flag)?.as_str() {
                    "pth" => Format::Pth,
                    "safetensors" => Format::Safetensors,
                    other => {
                        return Err(format!(
                            "--format must be 'pth' or 'safetensors', got '{other}'"
                        ))
                    }
                };
            }
            "--local-files-only" => args.local_files_only = true,
            "--dry-run" => args.dry_run = true,
            _ => return Err(format!("unknown argument '{flag}'; use --help for usage")),
        }
    }
    if args.output.is_empty() && !args.dry_run {
        return Err("--output is required; use --help for usage".into());
    }
    Ok(Some(args))
}
