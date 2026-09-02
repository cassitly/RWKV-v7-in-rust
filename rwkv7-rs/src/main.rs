//! RWKV-7 "Goose" x070 inference in pure Rust.
//!
//! A port of `rwkv_v7_demo_fast.py` from https://github.com/BlinkDL/RWKV-LM,
//! with the WKV-7 recurrence following the scalar form in `cuda/wkv7s.cu`.
//! No external crates: the checkpoint reader, tokenizer, kernels and thread
//! pool are all in this tree.

use rwkv7::{loader, model, pool, sampler, tokenizer};

use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use model::Model;
use sampler::{SampleOpts, Rng};
use tokenizer::Tokenizer;

const DEFAULT_PROMPT: &str = "User: simulate SpaceX mars landing using python\n\nAssistant: <think";

struct Args {
    model: String,
    vocab: Option<String>,
    prompt: String,
    length: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    seed: u64,
    threads: usize,
    chunk: usize,
    show_top: usize,
    tokens: Option<Vec<u32>>,
    dump_logits: Option<String>,
    encode_only: bool,
}

const USAGE: &str = "\
rwkv7 — RWKV-7 \"Goose\" x070 inference in Rust

USAGE:
    rwkv7 --model <FILE> [OPTIONS]

OPTIONS:
    -m, --model <FILE>       Checkpoint: .pth (torch.save) or .safetensors  [required]
        --vocab <FILE>       Tokenizer vocab  [default: rwkv_vocab_v20230424.txt, searched
                             next to the model, in the CWD, and one level up]
    -p, --prompt <TEXT>      Prompt to condition on
        --prompt-file <F>    Read the prompt from a file instead
    -n, --length <N>         Tokens to generate  [default: 500]
    -t, --temperature <F>    Sampling temperature  [default: 1.0]
        --top-p <F>          Nucleus sampling; 0 means greedy  [default: 0.0]
        --top-k <N>          Top-k cutoff, 0 disables  [default: 0]
        --seed <N>           RNG seed  [default: from the clock]
        --threads <N>        Worker threads  [default: number of cores]
        --chunk <N>          Prefill block size  [default: 256]
        --show-top <N>       Print the N most likely continuations after prefill  [default: 10]
        --tokens <IDS>       Use these comma-separated token ids as the prompt, bypassing
                             the tokenizer (for testing against a reference implementation)
        --encode-only        Print the prompt's token ids and exit
        --dump-logits <F>    Write the post-prefill logits to F as raw little-endian f32
    -h, --help               Show this help
";

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        model: String::new(),
        vocab: None,
        prompt: DEFAULT_PROMPT.to_string(),
        length: 500,
        temperature: 1.0,
        top_p: 0.0,
        top_k: 0,
        seed: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x5eed),
        threads: 0,
        chunk: 256,
        show_top: 10,
        tokens: None,
        dump_logits: None,
        encode_only: false,
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
            "--vocab" => a.vocab = Some(val(k)?),
            "-p" | "--prompt" => a.prompt = val(k)?,
            "--prompt-file" => {
                let f = val(k)?;
                a.prompt = std::fs::read_to_string(&f).map_err(|e| format!("{f}: {e}"))?;
            }
            "-n" | "--length" => a.length = val(k)?.parse().map_err(|_| "bad --length")?,
            "-t" | "--temperature" => {
                a.temperature = val(k)?.parse().map_err(|_| "bad --temperature")?
            }
            "--top-p" => a.top_p = val(k)?.parse().map_err(|_| "bad --top-p")?,
            "--top-k" => a.top_k = val(k)?.parse().map_err(|_| "bad --top-k")?,
            "--seed" => a.seed = val(k)?.parse().map_err(|_| "bad --seed")?,
            "--threads" => a.threads = val(k)?.parse().map_err(|_| "bad --threads")?,
            "--chunk" => a.chunk = val(k)?.parse::<usize>().map_err(|_| "bad --chunk")?.max(1),
            "--show-top" => a.show_top = val(k)?.parse().map_err(|_| "bad --show-top")?,
            "--tokens" => {
                let raw = val(k)?;
                let mut ids = Vec::new();
                for t in raw.split(',').filter(|t| !t.trim().is_empty()) {
                    ids.push(t.trim().parse::<u32>().map_err(|_| format!("bad token id '{t}'"))?);
                }
                a.tokens = Some(ids);
            }
            "--dump-logits" => a.dump_logits = Some(val(k)?),
            "--encode-only" => a.encode_only = true,
            other => return Err(format!("unknown argument '{other}'\n\n{USAGE}")),
        }
        i += 1;
    }

    if a.model.is_empty() && !a.encode_only {
        return Err(format!("--model is required\n\n{USAGE}"));
    }
    Ok(a)
}

/// Look for the vocab file next to the model, then in the usual relative spots.
fn find_vocab(explicit: &Option<String>, model_path: &str) -> Result<String, String> {
    if let Some(v) = explicit {
        return if std::path::Path::new(v).exists() {
            Ok(v.clone())
        } else {
            Err(format!("vocab file not found: {v}"))
        };
    }
    const NAME: &str = "rwkv_vocab_v20230424.txt";
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(dir) = std::path::Path::new(model_path).parent() {
        candidates.push(dir.join(NAME));
    }
    candidates.push(std::path::PathBuf::from(NAME));
    candidates.push(std::path::PathBuf::from("..").join(NAME));
    if let Ok(exe) = std::env::current_exe() {
        // target/release/rwkv7 -> repo root is three levels up
        for up in 1..=4 {
            let mut p = exe.clone();
            for _ in 0..up {
                p.pop();
            }
            candidates.push(p.join(NAME));
        }
    }
    for c in &candidates {
        if c.exists() {
            return Ok(c.to_string_lossy().into_owned());
        }
    }
    Err(format!(
        "could not find {NAME}. Pass --vocab <path>. Looked in:\n{}",
        candidates
            .iter()
            .map(|c| format!("  {}", c.display()))
            .collect::<Vec<_>>()
            .join("\n")
    ))
}

fn human_bytes(n: usize) -> String {
    const U: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.1} {}", U[i])
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    pool::set_threads(args.threads);
    let threads = pool::pool().threads();

    // --encode-only never touches the model, so handle it before anything is loaded.
    if args.encode_only {
        let path = find_vocab(&args.vocab, &args.model)?;
        let tok = Tokenizer::from_file(&path).map_err(|e| format!("{path}: {e}"))?;
        let ids = tok.encode(&args.prompt);
        println!("{}", ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
        return Ok(());
    }

    // With explicit token ids there is nothing to tokenize, so a missing vocab file
    // is not an error in that mode.
    let tok = match find_vocab(&args.vocab, &args.model) {
        Ok(path) => {
            eprintln!("loading tokenizer: {path}");
            Some(Tokenizer::from_file(&path).map_err(|e| format!("{path}: {e}"))?)
        }
        Err(e) => {
            if args.tokens.is_none() {
                return Err(e);
            }
            eprintln!("note: no tokenizer ({e}); running with --tokens only");
            None
        }
    };

    eprintln!("loading model:     {}", args.model);
    let t0 = Instant::now();
    let tensors = loader::load(&args.model).map_err(|e| format!("{}: {e}", args.model))?;
    let n_params: usize = tensors.values().map(|t| t.data.len()).sum();
    let model = Model::load(tensors).map_err(|e| format!("{}: {e}", args.model))?;
    let c = &model.cfg;
    eprintln!(
        "  n_layer={} n_embd={} n_head={} head_size={} dim_ffn={} vocab={}",
        c.n_layer, c.n_embd, c.n_head, c.head_size, c.dim_ffn, c.vocab_size
    );
    eprintln!(
        "  {} params, {} resident (f32), loaded in {:.2}s, {threads} thread(s)",
        n_params,
        human_bytes(n_params * 4),
        t0.elapsed().as_secs_f64()
    );
    if let Some(t) = &tok {
        if c.vocab_size < t.vocab_size() {
            eprintln!("  warning: model vocab {} < tokenizer vocab {}", c.vocab_size, t.vocab_size());
        }
    }

    let tokens = match &args.tokens {
        Some(ids) => ids.clone(),
        None => tok.as_ref().ok_or("no tokenizer available")?.encode(&args.prompt),
    };
    if tokens.is_empty() {
        return Err("prompt encoded to zero tokens".into());
    }
    if let Some(bad) = tokens.iter().find(|t| **t as usize >= c.vocab_size) {
        return Err(format!("token id {bad} is outside the model vocab ({})", c.vocab_size));
    }

    // ---- prefill ---------------------------------------------------------
    let mut state = model.new_state();
    let mut logits: Vec<f32> = Vec::new();
    let t0 = Instant::now();
    for block in tokens.chunks(args.chunk) {
        model.forward_seq(block, &mut state, &mut logits);
    }
    let prefill = t0.elapsed().as_secs_f64();
    eprintln!(
        "\nprefill: {} tokens in {:.3}s ({:.1} tok/s)\n",
        tokens.len(),
        prefill,
        tokens.len() as f64 / prefill.max(1e-9)
    );

    if let Some(path) = &args.dump_logits {
        let mut raw = Vec::with_capacity(logits.len() * 4);
        for v in &logits {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, &raw).map_err(|e| format!("{path}: {e}"))?;
        eprintln!("wrote {} logits to {path}", logits.len());
    }

    if args.show_top > 0 && tok.is_some() {
        let tok = tok.as_ref().unwrap();
        let mut probs = Vec::new();
        sampler::softmax(&logits, &mut probs);
        eprintln!("most likely continuations:");
        for id in sampler::top_indices(&probs, args.show_top) {
            let s = tok
                .decode(&[id])
                .unwrap_or_else(|| format!("{:?}", tok.decode_bytes(&[id])));
            eprintln!("  {:>7.2}%  {:?}", probs[id as usize] * 100.0, s);
        }
        eprintln!();
    }

    if args.length == 0 {
        return Ok(());
    }
    let Some(tok) = tok else {
        return Err("generation needs a tokenizer; pass --vocab or use --length 0".into());
    };

    // ---- generate --------------------------------------------------------
    print!("{}", args.prompt);
    let _ = std::io::stdout().flush();

    let opts = SampleOpts {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
    };
    let mut rng = Rng::new(args.seed);

    // Tokens are byte sequences, so a multi-byte character can straddle two of
    // them; buffer until the tail decodes as valid UTF-8.
    let mut pending: Vec<u32> = Vec::new();
    let mut generated: usize = 0;
    let t0 = Instant::now();

    for _ in 0..args.length {
        let token = sampler::sample(&logits, &opts, &mut rng);
        if token == 0 {
            break; // endoftext
        }
        pending.push(token);
        if let Some(s) = tok.decode(&pending) {
            print!("{s}");
            let _ = std::io::stdout().flush();
            pending.clear();
        }
        model.forward_one(token, &mut state, &mut logits);
        generated += 1;
    }
    if !pending.is_empty() {
        print!("{}", String::from_utf8_lossy(&tok.decode_bytes(&pending)));
    }
    println!();

    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "\n[{generated} tokens in {dt:.2}s = {:.1} tok/s]",
        generated as f64 / dt.max(1e-9)
    );
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
