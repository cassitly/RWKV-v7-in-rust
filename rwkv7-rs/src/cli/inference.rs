//! Inference options, validated before loading a checkpoint.

use super::{number, value};
use crate::sampler::SampleOpts;

pub const DEFAULT_PROMPT: &str =
    "User: simulate SpaceX mars landing using python\n\nAssistant: <think";

/// Exactly one source conditions the model. Files are read during execution.
#[derive(Debug)]
pub enum Input {
    Text(String),
    File(String),
    Tokens(Vec<u32>),
}

#[derive(Debug)]
pub struct Args {
    pub model: String,
    pub vocab: Option<String>,
    pub input: Input,
    pub length: usize,
    pub sampling: SampleOpts,
    pub seed: u64,
    pub threads: usize,
    pub chunk: usize,
    pub show_top: usize,
    pub dump_logits: Option<String>,
    pub encode_only: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model: String::new(),
            vocab: None,
            input: Input::Text(DEFAULT_PROMPT.into()),
            length: 500,
            sampling: SampleOpts {
                temperature: 1.0,
                top_p: 0.0,
                top_k: 0,
            },
            seed: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x5eed),
            threads: 0,
            chunk: 256,
            show_top: 10,
            dump_logits: None,
            encode_only: false,
        }
    }
}

pub const USAGE: &str = "\
rwkv7 — RWKV-7 \"Goose\" x070 inference in Rust

USAGE:
    rwkv7 --model <FILE> [OPTIONS]
    rwkv7 --encode-only [--vocab <FILE>] [--prompt <TEXT> | --prompt-file <FILE>]

OPTIONS:
    -m, --model <FILE>       Checkpoint: .pth (torch.save) or .safetensors  [required]
        --vocab <FILE>       Tokenizer vocab  [default: rwkv_vocab_v20230424.txt, searched
                             next to the model, in the CWD, and one level up]
    -p, --prompt <TEXT>      Prompt to condition on
        --prompt-file <F>    Read the prompt from a file instead
    -n, --length <N>         Tokens to generate  [default: 500]
    -t, --temperature <F>    Positive, finite sampling temperature  [default: 1.0]
        --top-p <F>          Nucleus sampling in [0, 1]; 0 means greedy  [default: 0.0]
        --top-k <N>          Top-k cutoff, 0 disables  [default: 0]
        --seed <N>           RNG seed  [default: from the clock]
        --threads <N>        Worker threads; 0 selects number of cores  [default: 0]
        --chunk <N>          Positive prefill block size  [default: 256]
        --show-top <N>       Print the N most likely continuations after prefill  [default: 10]
        --tokens <IDS>       Use these comma-separated token ids as the prompt, bypassing
                             the tokenizer (for testing against a reference implementation)
        --encode-only       Print the prompt's token ids and exit
        --dump-logits <F>    Write the post-prefill logits to F as raw little-endian f32
    -h, --help               Show this help

Choose only one of --prompt, --prompt-file, or --tokens.
";

/// Parse arguments without performing I/O. Help does not require a model.
pub fn parse(argv: impl IntoIterator<Item = String>) -> Result<Option<Args>, String> {
    let mut args = Args::default();
    let mut input = None;
    let mut argv = argv.into_iter();
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "-h" | "--help" => return Ok(None),
            "-m" | "--model" => args.model = value(&mut argv, &flag)?,
            "--vocab" => args.vocab = Some(value(&mut argv, &flag)?),
            "-p" | "--prompt" | "--prompt-file" | "--tokens" => {
                if input.is_some() {
                    return Err("choose only one of --prompt, --prompt-file, or --tokens".into());
                }
                let raw = value(&mut argv, &flag)?;
                input = Some(match flag.as_str() {
                    "--prompt-file" => Input::File(raw),
                    "--tokens" => Input::Tokens(parse_tokens(&raw)?),
                    _ => Input::Text(raw),
                });
            }
            "-n" | "--length" => args.length = number(&mut argv, &flag)?,
            "-t" | "--temperature" => args.sampling.temperature = number(&mut argv, &flag)?,
            "--top-p" => args.sampling.top_p = number(&mut argv, &flag)?,
            "--top-k" => args.sampling.top_k = number(&mut argv, &flag)?,
            "--seed" => args.seed = number(&mut argv, &flag)?,
            "--threads" => args.threads = number(&mut argv, &flag)?,
            "--chunk" => args.chunk = number(&mut argv, &flag)?,
            "--show-top" => args.show_top = number(&mut argv, &flag)?,
            "--dump-logits" => args.dump_logits = Some(value(&mut argv, &flag)?),
            "--encode-only" => args.encode_only = true,
            _ => return Err(format!("unknown argument '{flag}'; use --help for usage")),
        }
    }
    if let Some(input) = input {
        args.input = input;
    }
    if args.model.is_empty() && !args.encode_only {
        return Err("--model is required; use --help for usage".into());
    }
    if args.encode_only && matches!(args.input, Input::Tokens(_)) {
        return Err("--encode-only requires text; use --prompt or --prompt-file".into());
    }
    if !args.sampling.temperature.is_finite() || args.sampling.temperature <= 0.0 {
        return Err("--temperature must be a positive, finite number".into());
    }
    if !args.sampling.top_p.is_finite() || !(0.0..=1.0).contains(&args.sampling.top_p) {
        return Err("--top-p must be a finite number between 0 and 1".into());
    }
    if args.chunk == 0 {
        return Err("--chunk must be greater than zero".into());
    }
    Ok(Some(args))
}

fn parse_tokens(raw: &str) -> Result<Vec<u32>, String> {
    if raw.trim().is_empty() {
        return Err("--tokens requires at least one token id".into());
    }
    raw.split(',')
        .map(|id| {
            id.trim()
                .parse()
                .map_err(|_| format!("invalid token id '{id}' in --tokens"))
        })
        .collect()
}
