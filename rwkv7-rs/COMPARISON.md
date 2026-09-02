# rwkv7-rs vs. the Python implementation

Date: 2026-09-02
Scope: numerical agreement, tokenizer agreement, greedy-output agreement, and CPU speed of the `rwkv7-rs` crate against the Python RWKV-7 code in this repository, on a real 0.1B checkpoint.

## Summary

The Rust crate and the Python implementation produce the same results on real weights. Logits differ by at most 7e-5 in absolute terms, which is the same magnitude as the difference between the two Python implementations themselves, so it is float32 summation-order noise rather than a logic difference. Tokenization is identical. Greedy generation for 200 tokens is identical character for character. The crate's own test suite passes. On this CPU, Rust decodes about twice as fast as numpy with all cores and prefills six to eight times faster.

| Check | Result |
|---|---|
| Tokenizer, prompt ids Rust vs Python | identical, 14 of 14 ids |
| Prefill logits, Rust vs `rwkv_v7_numpy.py` | max abs diff 6.9e-5, argmax same, top-10 same |
| Prefill logits, Python vs Python | max abs diff 3.1e-5 |
| Logits after 214 tokens, Rust vs Python | max abs diff 3.1e-5, no drift |
| Rust batched prefill vs Rust one-token path | bit-identical |
| Greedy generation, 200 tokens | identical text across all three implementations |
| `cargo test --release` | 26 passed, 0 failed, 1 ignored by design |

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| CPU | Intel Core i5-13400F, 10 cores, 16 threads |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 3070, not used (torch build is CPU-only) |
| Rust | rustc 1.90.0, cargo 1.90.0 |
| Python | 3.11.9 (Windows Store build) |
| numpy | 2.3.3 |
| torch | 2.8.0+cpu, used only for `torch.load` |
| Model | `rwkv7-g1d-0.1b-20260129-ctx8192.pth` from `BlinkDL/rwkv7-g1` on HuggingFace |

Model facts as reported by both loaders: 12 layers, embedding 768, 12 heads of size 64, FFN 3072, vocab 65536, 191,084,544 parameters, stored as bf16 in a 382 MB `torch.save` archive.

The checkpoint the demos hardcode, `rwkv7-g1a-0.1b-20250728-ctx4096.pth`, no longer exists upstream. The `g1d` file is the current x070 0.1B release. The `rwkv7a` and `rwkv7b` files in the same repo are different architectures and are rejected by the crate's model builder on purpose.

## What was compared

Three implementations were run on the same prompt and the same token ids.

| Label | Implementation | Notes |
|---|---|---|
| Rust | `rwkv7-rs`, `target/release/rwkv7.exe` | f32 weights, 16 worker threads unless stated |
| A | `rwkv_v7_numpy.py` at the repo root | The model code above its `# Verification` block, executed verbatim. Its unpacking relies on checkpoint key order, and that order matched. The block that needs the `rwkv` pip package was skipped. |
| B | `rwkv7-rs/tools/reference_forward.py` | The crate's numpy transcription of `rwkv_v7_demo_fast.py`, imported as a module, with the same preprocessing its `main()` does (ln0 folded into the embedding table). |

The tokenizer used on the Python side is the `RWKV_TOKENIZER` class taken verbatim from `rwkv_v7_demo_fast.py`.

Prompt, the one hardcoded in the demos:

```
User: simulate SpaceX mars landing using python

Assistant: <think
```

Token ids, produced by the Python tokenizer and fed to the Rust binary through `--tokens` so both sides saw exactly the same input:

```
24281,59,57236,44898,31377,52092,40085,46966,261,5585,41693,59,295,35762
```

### What could not be run

`rwkv_v7_demo.py`, `rwkv_v7_demo_rnn.py` and `rwkv_v7_demo_fast.py` hardcode `map_location='cuda'`, `.cuda()` calls, and either a JIT-compiled CUDA kernel or `torch.compile` in max-autotune mode. The torch on this machine is a CPU-only build, so those demos were not run. The comparison is therefore against the numpy implementations, which compute the same forward pass in f32.

## Tokenizer

Rust `--encode-only` on the prompt produced the same 14 ids as the Python class. The crate's unit tests additionally check 312 prompts against ids from the Python tokenizer and re-encode all 65,529 vocab entries to themselves. Both passed.

## Logits

Each row compares two raw f32 logit vectors of length 65536. `max|d|/std` is the largest absolute difference divided by the standard deviation of the logits, which is about 4 here. `top10` is how many of the ten highest-scoring tokens are shared, and in every row they are also in the same order.

| Comparison | max abs diff | mean abs diff | max abs diff / std | argmax | top-10 |
|---|---|---|---|---|---|
| After prompt: Rust (chunk 256) vs A | 6.87e-05 | 1.67e-05 | 1.70e-05 | same | 10/10 |
| After prompt: Rust (chunk 256) vs B | 6.48e-05 | 1.43e-05 | 1.60e-05 | same | 10/10 |
| After prompt: Rust (chunk 1, one-token path) vs B | 6.48e-05 | 1.43e-05 | 1.60e-05 | same | 10/10 |
| After prompt: Rust chunk 256 vs Rust chunk 1 | 0 | 0 | 0 | same | 10/10 |
| After prompt: A vs B | 3.05e-05 | 4.69e-06 | 7.54e-06 | same | 10/10 |
| After prompt + 200 tokens: Rust vs A | 3.05e-05 | 4.86e-06 | 6.41e-06 | same | 10/10 |
| After prompt + 200 tokens: Rust vs B | 3.81e-05 | 6.84e-06 | 8.01e-06 | same | 10/10 |
| After prompt + 200 tokens: A vs B | 1.91e-05 | 5.35e-06 | 4.00e-06 | same | 10/10 |

Three things stand out.

- **The Rust-to-Python gap is the same size as the Python-to-Python gap.** Two numpy programs computing the same math in a different association order differ by 3e-5. Rust differs from either by 3e-5 to 7e-5. That is the noise floor of f32 arithmetic on this problem.
- **The recurrence does not drift.** The "after 200 tokens" rows are no worse than the "after prompt" rows. The state stays in f32 in all three implementations, and the README's warning about that is borne out.
- **The two Rust paths are bit-identical.** The batched prefill (`--chunk 256`) and the one-token-at-a-time path (`--chunk 1`) produced byte-for-byte equal logits, so reassociating the projections into matmuls did not change the result at all on this input.

Most likely next tokens after the prompt, as printed by the Rust binary:

| Token | Probability |
|---|---|
| `>` | 99.42% |
| `></` | 0.51% |
| `>{` | 0.02% |
| `>\` | 0.01% |
| `>>` | 0.01% |

## Greedy generation

All three implementations generated 200 tokens greedily (`--top-p 0` on the Rust side, `argmax` on the Python side). The decoded text is identical across Rust, A, and B: 903 characters with no divergence. The token id sequences from A and B are also identical.

The continuation begins:

```
>
Okay, the user wants to simulate SpaceX's Mars landing using Python. Let me start by recalling what I know about SpaceX's Mars program. They launched a Falcon 9 rocket in 2011, and the landing was successful. But I need to think about the specific steps they took.
```

The content is the model's own confabulation and is not a property of either port.

## Speed

All figures are for f32 weights on the CPU above. The numpy figures are for one token at a time, since that is how the numpy code is written. numpy's matrix-vector products go through a multi-threaded BLAS, so the "1 thread" Rust row is not a like-for-like comparison with numpy.

| Implementation | Load | Prefill (14 tokens) | Prefill (214 tokens) | Decode |
|---|---|---|---|---|
| A, `rwkv_v7_numpy.py` | 0.3 s | 27 tok/s | | 24.0 tok/s |
| B, `reference_forward.py` | 0.3 s | 25 tok/s | | 23.4 tok/s |
| Rust, 16 threads, chunk 256 | 0.36 s | 133 to 156 tok/s | 212 tok/s | 52.9 tok/s |
| Rust, 16 threads, chunk 1 | 0.36 s | 73 tok/s | | |
| Rust, 1 thread | | 36 tok/s | | 16.7 tok/s |

Load time for Rust includes reading the ZIP and pickle stream and widening 191M bf16 values to f32, ending at 729 MiB resident. Decode on Rust scales only about 3x from 1 to 16 threads because it is bound by memory bandwidth, not compute. That matches the crate README's own account.

## Test suite

`cargo build --release` finished in 15.7 s. `cargo test --release` ran 20 library tests, 6 conversion tests, and 0 doc tests, all passing. One test, `zip64_entry_past_4gib`, is ignored by design because it writes 4.1 GiB. The library tests include the stored-fixture comparison of Rust logits against a numpy reference on a synthetic 3-layer checkpoint for chunk sizes 1, 2, 3, 7, 20 and 64, for both the `.pth` and `.safetensors` loaders.

## Method

Commands, from the repository root. `$M` is the checkpoint path and `$C` an output directory.

Python side, producing `prompt_ids.txt`, `A_prefill.bin`, `B_prefill.bin`, `A_gen_ids.txt`, `A_gen.txt`, `A_deep.bin` and the B equivalents:

```
python3.11 py_side.py . $M $C 200
```

Rust side:

```
R=rwkv7-rs/target/release/rwkv7.exe
V=rwkv_vocab_v20230424.txt
IDS=$(cat $C/prompt_ids.txt)
GEN=$(cat $C/A_gen_ids.txt)

$R --encode-only --vocab $V --prompt "$PROMPT"
$R --model $M --vocab $V --tokens $IDS --length 0 --show-top 0 --dump-logits $C/rust_prefill.bin
$R --model $M --vocab $V --tokens $IDS --length 0 --show-top 0 --chunk 1 --dump-logits $C/rust_prefill_chunk1.bin
$R --model $M --vocab $V --prompt "$PROMPT" --length 200 --top-p 0 > $C/rust_gen.txt
$R --model $M --vocab $V --tokens $IDS,$GEN --length 0 --show-top 0 --dump-logits $C/rust_deep.bin
$R --model $M --vocab $V --prompt "$PROMPT" --length 100 --top-p 0 --threads 1
```

The "deep" comparison feeds the Rust binary the prompt ids followed by the 200 ids Python generated, then dumps the logits at that position. The Python side saved its logits at the same position. This tests 214 steps of recurrence on identical input.

`py_side.py` and `compare.py`, the harness scripts, were written for this run and live outside the repository in the session scratchpad. They are about 130 and 70 lines and depend only on numpy and torch's `load`.

## Caveats

- One prompt, one model size, one machine. The crate's synthetic fixtures cover other shapes, but the real-weight check here is a single 0.1B checkpoint.
- The CUDA demos were not exercised. Agreement with them is inferred through `reference_forward.py`, which is a line-by-line transcription of the one-token path in `rwkv_v7_demo_fast.py`.
- The `rwkv` pip package reference inside `rwkv_v7_numpy.py` was not run, so this report does not independently re-verify that numpy code against the official `rwkv` package.
- Timings are single runs, not averages. Prefill on 14 tokens varied between 133 and 156 tok/s across two Rust runs.
