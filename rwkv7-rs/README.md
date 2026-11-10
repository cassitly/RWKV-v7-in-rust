# rwkv7-rs

Pure Rust ports of two things from this repo. No external crates.

| Binary | Ports | What it does |
|---|---|---|
| `rwkv7` | [`rwkv_v7_demo_fast.py`](../rwkv_v7_demo_fast.py) | RWKV-7 "Goose" x070 inference |
| `qwen-make-pth` | [`run_qwen35_make_pth.py`](../run_qwen35_make_pth.py) | Export the text weights of a HF checkpoint |

The WKV-7 recurrence follows the scalar form in [`cuda/wkv7s.cu`](../cuda/wkv7s.cu).
Checkpoint readers *and writers* (ZIP, pickle, safetensors, JSON), the tokenizer,
the kernels and the thread pool are all in this tree — `cargo build` needs
nothing but the standard library.

## Quick start — inference

Run these commands from `rwkv7-rs/` (from the repository root, use
`cd rwkv7-rs` first). A stable Rust toolchain is the only build requirement.

```sh
cargo build --release

# model: https://huggingface.co/BlinkDL/rwkv7-g1
./target/release/rwkv7 \
    --model /path/to/model.pth \
    --vocab ../rwkv_vocab_v20230424.txt \
    --prompt 'User: simulate SpaceX mars landing using python

Assistant: <think' \
    --length 500
```

The tokenizer vocab (`rwkv_vocab_v20230424.txt`) is found automatically next to
the model, in the working directory, or one level up; `--vocab` overrides it.

Accepts `.pth` (`torch.save`) and `.safetensors`. Weights may be stored as
f32 / f16 / bf16 and are up-converted to f32 on load.

## Options

```
-m, --model <FILE>       Checkpoint: .pth or .safetensors            [required]
    --vocab <FILE>       Tokenizer vocab
-p, --prompt <TEXT>      Prompt to condition on
    --prompt-file <F>    Read the prompt from a file instead
-n, --length <N>         Tokens to generate                          [500]
-t, --temperature <F>    Positive, finite sampling temperature       [1.0]
    --top-p <F>          Nucleus sampling in [0, 1]; 0 is greedy      [0.0]
    --top-k <N>          Top-k cutoff, 0 disables                    [0]
    --seed <N>           RNG seed                                    [clock]
    --threads <N>        Worker threads; 0 selects cores              [0]
    --chunk <N>          Positive prefill block size                 [256]
    --show-top <N>       Most likely continuations after prefill     [10]
    --tokens <IDS>       Use these token ids, bypassing the tokenizer
    --encode-only        Print the prompt's token ids and exit
    --dump-logits <F>    Write post-prefill logits as raw LE f32
```

Choose only one of `--prompt`, `--prompt-file`, or `--tokens`. An empty or
malformed token list is rejected. `--encode-only` accepts text or a prompt file
and needs no model. When generating from `--tokens`, stdout starts with the
decoded token input. An explicitly supplied `--vocab` must point to a file;
without it, token-only logit inspection (`--length 0`) can run without a vocab.

Use `--help` on either binary for its full usage. Numeric values are validated
before loading a checkpoint; `--chunk 0`, nonpositive temperatures, NaN, and
infinite sampling values are errors. Diagnostics go to stderr, generated text
or encoded token ids to stdout.

Shells don't expand `\n` inside a normal argument. Use `$'...'` in bash/zsh, a
literal multi-line quoted string, or `--prompt-file`.

## qwen-make-pth

Reads a HuggingFace model directory, drops vision and MTP tensors, strips the
`model.language_model.` prefix, and writes a `torch.save` `.pth` plus a `.json`
metadata sidecar.

```sh
./target/release/qwen-make-pth \
    --model /path/to/Qwen3.5-0.8B \
    --output /path/to/qwen35_0_8b_text.pth
```

```
-m, --model <DIR>        HF model directory, or a repo id  [Qwen/Qwen3.5-0.8B]
-o, --output <FILE>      Output path                                [required]
    --format <FMT>       pth | safetensors                               [pth]
    --local-files-only   Never shell out to download
    --dry-run            Inspect local weights without downloads or writes
```

Handles sharded `.safetensors` and `.bin`, via `*.index.json` when present.
`--dry-run` requires a local model directory and does not need `--output`.
This exporter changes checkpoint packaging and key names, not the architecture;
its Qwen outputs are not compatible with the RWKV-7 inference engine.

Three differences from the Python, all deliberate:

* **No HTTPS.** `snapshot_download` would mean shipping a TLS stack for one code
  path. A repo id is handed to the `hf` CLI when it is on PATH; otherwise the
  tool prints the `hf download` command to run and exits. A local directory
  always works.
* **Streaming.** The Python loads the whole state dict into RAM before saving.
  This scans only headers, then copies each tensor straight from its shard to the
  output, so tensor payload memory is bounded by one 1 MiB buffer. Tensor metadata
  remains in memory and grows with the number of tensors.
* **`config.json` is passed through verbatim** into the sidecar rather than
  parsed and re-emitted, so key order and exact numeric spellings survive. The
  Python re-serialises with `ensure_ascii=False`, so non-ASCII config values will
  differ cosmetically (`"caf\u00e9"` here vs `"café"` there); the decoded value is
  the same.

Tensor bytes are never decoded, only copied, so **every dtype round-trips** —
including fp8, which nothing else in this crate understands.

## Layout

| File | What it is |
|---|---|
| `src/cli/` | Testable argument parsers and help text for both binaries |
| `src/main.rs` | Inference orchestration, prefill diagnostics, and streamed generation |
| `src/bin/qwen_make_pth.rs` | HF export orchestration and metadata reports |
| `src/model.rs` | The architecture. `wkv7_step` is the recurrence; `forward_one` / `forward_seq` are the RNN and prefill paths |
| `src/ops.rs` | LayerNorm, GroupNorm, L2-normalise, and the four matmul shapes |
| `src/tokenizer.rs` | Byte-trie greedy longest-match tokenizer |
| `src/loader.rs` | Checkpoint → `name -> f32 tensor`, for both formats |
| `src/pickle.rs` | Pickle VM covering what `torch.save` emits |
| `src/zip.rs` | ZIP reader (stored entries + ZIP64) |
| `src/sampler.rs` | Temperature / top-p / top-k, and a xoshiro256++ RNG |
| `src/pool.rs` | Fixed thread pool for the data-parallel loops |
| `src/hf.rs` | HF shard discovery, the key filters, and the export writers |
| `src/zip_write.rs` | ZIP writer: stored entries, ZIP64, 64-byte record alignment |
| `src/pickle_write.rs` | Emits the pickle stream `torch.save` produces |
| `src/json.rs` | JSON reader/writer for index files, headers and the sidecar |

## The algorithm

Per layer, per token — `x_prev` is the previous token's input to this layer:

```
xx = LN(x, ln1)
xr,xw,xk,xv,xa,xg = xx + (x_prev - xx) * x_{r,w,k,v,a,g}    # six token-shift lerps

r = R @ xr;  k = K @ xk;  v = V @ xv
w = tanh(xw @ w1) @ w2                                       # LoRA
a = sigmoid(a0 + (xa @ a1) @ a2)                             # LoRA
g = sigmoid(xg @ g1) @ g2                                    # LoRA
kk = l2_normalize(k * k_k)  per head
k  = k * (1 + (a - 1) * k_a)
v  = v + (v_first - v) * sigmoid(v0 + (xv @ v1) @ v2)        # layer 0 sets v_first = v
w  = exp(-exp(-0.5) * sigmoid(w0 + w))                       # per-channel decay

# state S is [n_head, head_size, head_size], row = value dim, column = key dim
sa[i]   = sum_j -kk[j] * S[i][j]
S[i][j] = S[i][j] * w[j] + k[j] * v[i] + sa[i] * (kk[j] * a[j])
y[i]    = sum_j S[i][j] * r[j]

y = GroupNorm(y, groups=n_head, eps=64e-5, ln_x)
y += sum_per_head(r * k * r_k) * v                           # the "bonus" term
x += (y * g) @ O

xx = LN(x, ln2)
x += relu(xx + (x_prev - xx) * ffn_x_k @ Kf)^2 @ Vf          # channel mix, ReLU squared
```

Two details that are easy to get wrong and fail silently rather than loudly:

* **The state must be f32.** Everything else can be lower precision; the running
  `S` cannot, or output drifts within a few hundred tokens.
* **GroupNorm eps is `64e-5`**, not the usual `1e-5`, and it is applied per head
  over `head_size` channels.

State is `att_x [C]`, `att_kv [n_head, head_size, head_size]`, `ffn_x [C]` per
layer. There is no KV cache — memory is constant in context length, and prefill
is chunked only to bound scratch space.

`forward_one` and `forward_seq` compute the same thing. The seq path only
reassociates the projections into matmuls so each weight matrix is streamed once
per block instead of once per token; the recurrence itself is inherently
sequential and is scanned token by token in both.

## Tests

```sh
cargo test --release
```

* `tests/cli.rs` — argument validation, prompt sources, model-free tokenization,
  logit dumps, token-input generation, local-only dry runs, and both export
  formats with consistent metadata. All use bundled fixtures.
* `tokenizer::matches_python_reference` — 312 prompts (the repo's own prompts,
  CJK, emoji, ZWJ sequences, whitespace runs, random noise) against ids from the
  upstream Python `RWKV_TOKENIZER`.
* `tokenizer::single_tokens_round_trip` — all 65,529 vocab entries re-encode to
  themselves.
* `model::matches_reference_{pth,safetensors}` — logits after 20 tokens against
  a numpy reference, for chunk sizes 1 / 2 / 3 / 7 / 20 / 64, so the RNN and
  prefill paths are checked against each other and against the reference.
* `model::loaders_agree` — the two checkpoint readers produce identical tensors.
* `loader::{half_precision,bfloat16}_conversions` — including f16 subnormals.
* `convert::*` — conversion against `tests/hf_fixture/`, a synthetic two-shard
  model with vision / MTP / prefixed keys and bf16 / f16 / f32 / i64 / u8 / fp8
  tensors. Checks the filters, byte-for-byte round-trips through both output
  formats, and 64-byte record alignment.
* `json::*`, `pickle_write::*`, `zip_write::crc32_matches_known_vectors` — the
  writers, including a pickle round-trip through our own reader.

One test is `#[ignore]`d because it writes 4.1 GiB:

```sh
cargo test --release --test convert -- --ignored     # zip64_entry_past_4gib
```

Against the numpy reference the port agrees to **~2e-6 absolute** on the logits
after 200 tokens of recurrence, which is f32 summation-order noise.

Fixtures are regenerated with:

```sh
python3 tools/make_test_checkpoint.py --out tests/tiny --n-layer 3 --n-embd 32 \
        --vocab 96 --head-size 16 --seed 20260826
python3 tools/reference_forward.py --npz tests/tiny.npz \
        --tokens "$(cat tests/tiny_tokens.txt)" --out tests/tiny_logits.bin
python3 tools/gen_tokenizer_cases.py cases.json   # needs ../rwkv_vocab_v20230424.txt
```

`tools/` needs numpy only — not PyTorch. `make_test_checkpoint.py` writes a real
`torch.save` archive by stubbing the handful of `torch` names that pickle
resolves, so the `.pth` path is tested for real.

The same trick verifies the *output* of `qwen-make-pth`. `tools/verify_pth.py`
loads it with Python's stdlib `zipfile` + `pickle` — an implementation that
shares no code with this crate — and checks the tensor bytes, CRCs, storage
classes, contiguous strides and 64-byte alignment:

```sh
python3 tools/make_hf_fixture.py --out /tmp/hf
./target/release/qwen-make-pth --model /tmp/hf -o /tmp/out.pth
python3 tools/verify_pth.py /tmp/out.pth /tmp/expect.json
```

**Not covered:** no real Qwen3.5 checkpoint was available here, so the conversion
is verified against synthetic fixtures only. The logic and the file formats are
exercised end to end; real-checkpoint quirks are not.

## Performance

Weights are held as f32, so decoding is memory-bandwidth-bound: roughly
`4 bytes × n_params` is read per token. On an M-series laptop, a 28M-parameter
model decodes at ~160 tok/s and prefills at ~5000 tok/s.

Two things would matter most if you want it faster:

1. **Keep weights in bf16** and widen in the matmul inner loop (`(x as u32) << 16`).
   Halves the bytes moved per token, which is the actual bottleneck. Mind that
   the recurrent state stays f32.
2. **Explicit SIMD** in `ops::dot` and `wkv7_step`. The scalar loops autovectorise
   reasonably but `head_size` is 64, a natural fit for hand-written NEON/AVX.

Decode parallelises poorly (~1.4× across 10 threads) because it is bandwidth-,
not compute-bound; prefill scales better (~2.6×).

## Scope

This is RWKV-7 (x070) only. `rwkv_v7a_demo.py`, `rwkv_v7b_demo.py` and the
`rwkv_v8_rc00*` demos in the parent directory are different architectures and
will not load — the model builder rejects them rather than producing garbage.
Training is out of scope; this is inference only.
