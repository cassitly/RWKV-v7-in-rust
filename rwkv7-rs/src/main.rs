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
use rwkv7::cli::inference::{self, Args, Input};
use sampler::Rng;
use tokenizer::Tokenizer;

/// Look for the vocab file next to the model, then in the usual relative spots.
fn find_vocab(explicit: &Option<String>, model_path: &str) -> Result<String, String> {
    if let Some(v) = explicit {
        return if std::path::Path::new(v).is_file() {
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
        if c.is_file() {
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

fn run(args: Args) -> Result<(), String> {
    let prompt = match &args.input {
        Input::Text(text) => Some(text.clone()),
        Input::File(path) => {
            Some(std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)
        }
        Input::Tokens(_) => None,
    };
    // --encode-only never touches the model, so handle it before anything is loaded.
    if args.encode_only {
        let path = find_vocab(&args.vocab, &args.model)?;
        let tok = Tokenizer::from_file(&path).map_err(|e| format!("{path}: {e}"))?;
        let ids = tok.encode(prompt.as_deref().unwrap_or_default());
        println!(
            "{}",
            ids.iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
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
            if args.vocab.is_some() || !matches!(args.input, Input::Tokens(_)) {
                return Err(e);
            }
            eprintln!("note: no tokenizer ({e}); running with --tokens only");
            None
        }
    };

    if args.length > 0 && tok.is_none() {
        return Err("generation needs a tokenizer; pass --vocab or use --length 0".into());
    }

    pool::set_threads(args.threads);
    let threads = pool::pool().threads();

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
            eprintln!(
                "  warning: model vocab {} < tokenizer vocab {}",
                c.vocab_size,
                t.vocab_size()
            );
        }
    }

    let tokens = match &args.input {
        Input::Tokens(ids) => ids.clone(),
        _ => tok
            .as_ref()
            .ok_or("no tokenizer available")?
            .encode(prompt.as_deref().unwrap_or_default()),
    };
    if tokens.is_empty() {
        return Err("prompt encoded to zero tokens".into());
    }
    if let Some(bad) = tokens.iter().find(|t| **t as usize >= c.vocab_size) {
        return Err(format!(
            "token id {bad} is outside the model vocab ({})",
            c.vocab_size
        ));
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

    if let Some(tok) = &tok {
        show_continuations(&logits, tok, args.show_top);
    }

    if args.length == 0 {
        return Ok(());
    }
    let Some(tok) = tok else {
        return Err("generation needs a tokenizer; pass --vocab or use --length 0".into());
    };

    let prompt =
        prompt.unwrap_or_else(|| String::from_utf8_lossy(&tok.decode_bytes(&tokens)).into_owned());
    generate(
        &model,
        &mut state,
        logits,
        &tok,
        &prompt,
        &args,
        &mut std::io::stdout().lock(),
    )
}

/// Report diagnostics on stderr so stdout remains usable as generated text.
fn show_continuations(logits: &[f32], tok: &Tokenizer, count: usize) {
    if count == 0 {
        return;
    }
    let mut probs = Vec::new();
    sampler::softmax(logits, &mut probs);
    eprintln!("most likely continuations:");
    for id in sampler::top_indices(&probs, count) {
        let text = tok
            .decode(&[id])
            .unwrap_or_else(|| format!("{:?}", tok.decode_bytes(&[id])));
        eprintln!("  {:>7.2}%  {:?}", probs[id as usize] * 100.0, text);
    }
    eprintln!();
}

/// Stream decoded tokens, buffering incomplete UTF-8 sequences between steps.
fn generate(
    model: &Model,
    state: &mut model::State,
    mut logits: Vec<f32>,
    tok: &Tokenizer,
    prompt: &str,
    args: &Args,
    output: &mut impl Write,
) -> Result<(), String> {
    write!(output, "{prompt}").map_err(|e| e.to_string())?;
    output.flush().map_err(|e| e.to_string())?;
    let mut rng = Rng::new(args.seed);

    // Tokens are byte sequences, so a multi-byte character can straddle two of
    // them; buffer until the tail decodes as valid UTF-8.
    let mut pending: Vec<u32> = Vec::new();
    let mut generated: usize = 0;
    let t0 = Instant::now();

    for _ in 0..args.length {
        let token = sampler::sample(&logits, &args.sampling, &mut rng);
        if token == 0 {
            break; // endoftext
        }
        pending.push(token);
        if let Some(s) = tok.decode(&pending) {
            write!(output, "{s}").map_err(|e| e.to_string())?;
            output.flush().map_err(|e| e.to_string())?;
            pending.clear();
        }
        model.forward_one(token, state, &mut logits);
        generated += 1;
    }
    if !pending.is_empty() {
        write!(
            output,
            "{}",
            String::from_utf8_lossy(&tok.decode_bytes(&pending))
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(output).map_err(|e| e.to_string())?;
    output.flush().map_err(|e| e.to_string())?;

    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "\n[{generated} tokens in {dt:.2}s = {:.1} tok/s]",
        generated as f64 / dt.max(1e-9)
    );
    Ok(())
}

fn main() -> ExitCode {
    let result = inference::parse(std::env::args().skip(1)).and_then(|args| match args {
        Some(args) => run(args),
        None => {
            print!("{}", inference::USAGE);
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
