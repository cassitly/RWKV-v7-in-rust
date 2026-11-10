# RWKV-7 in Rust

A CPU inference engine for **RWKV-7 “Goose” x070**, implemented in Rust with
no external crates. This repository also includes Python reference demos,
CUDA kernels, training experiments, and a HuggingFace checkpoint export tool.

RWKV processes tokens using a recurrent state. The Rust engine supports both
single-token decoding and batched prompt processing (prefill); its recurrent
state uses constant memory as the context grows. Weights and state are held in
float32. There is no GPU backend or training implementation in the Rust crate.

## Start here

| Goal | Entry point |
|---|---|
| Generate text on a CPU | [`rwkv7-rs/`](rwkv7-rs/README.md), binary `rwkv7` |
| Export HuggingFace text weights | [`qwen-make-pth`](rwkv7-rs/README.md#qwen-make-pth) |
| Understand the model equations and Rust modules | [Rust architecture guide](rwkv7-rs/README.md#the-algorithm) |
| Inspect previous Rust/Python numerical and speed comparisons | [`COMPARISON.md`](rwkv7-rs/COMPARISON.md) |
| Explore the Python implementations | [`rwkv_v7_demo_fast.py`](rwkv_v7_demo_fast.py), [`rwkv_v7_numpy.py`](rwkv_v7_numpy.py) |
| Explore training | [`train_temp/`](train_temp/README.md) |

The Rust inference engine supports **x070 only**. The v7a, v7b, and v8 Python
demos use other architectures. `qwen-make-pth` exports checkpoint tensors; it
does **not** turn a Qwen model into a model that `rwkv7` can run.

## Build and run

Install a stable [Rust toolchain](https://www.rust-lang.org/tools/install), then
run these commands from the repository root:

```sh
cargo build --release --manifest-path rwkv7-rs/Cargo.toml

./rwkv7-rs/target/release/rwkv7 \
    --model /path/to/model.pth \
    --vocab ./rwkv_vocab_v20230424.txt \
    --prompt 'The future of computing is' \
    --length 100
```

Download a compatible **RWKV-7 x070** checkpoint separately, for example from
[`BlinkDL/rwkv7-g1`](https://huggingface.co/BlinkDL/rwkv7-g1). Both `.pth`
(`torch.save` ZIP archives) and `.safetensors` are supported; f32, f16, and bf16
weights are converted to f32 when loaded. Allow roughly four bytes per model
parameter for resident weights, plus state and temporary buffers.

The vocabulary is included in this repository. `--vocab` sets its path explicitly;
the tool can also discover it near the model, working directory, or executable.
Use `--prompt-file prompt.txt` for a multiline prompt.

Generation is greedy by default (`--top-p 0`). For sampling, use, for example,
`--top-p 0.9 --temperature 0.8 --seed 42`. Generated text goes to stdout; progress
and timing reports go to stderr.

```sh
./rwkv7-rs/target/release/rwkv7 --help
./rwkv7-rs/target/release/qwen-make-pth --help

# Tokenize without loading a model.
./rwkv7-rs/target/release/rwkv7 --encode-only --prompt 'Hello!'

# Exercise inference with the bundled synthetic test model; no download needed.
./rwkv7-rs/target/release/rwkv7 \
    --model rwkv7-rs/tests/tiny.pth --tokens 1,2,3 \
    --length 0 --show-top 0 --threads 1
```

The tiny checkpoint contains random weights for numerical tests. Use a trained
checkpoint for meaningful text generation.

## Export HuggingFace text weights

```sh
./rwkv7-rs/target/release/qwen-make-pth \
    --model /path/to/local/HF-model \
    --output /path/to/text-weights.pth \
    --local-files-only
```

The exporter removes vision and MTP tensors, strips the `model.language_model.`
prefix, and writes `.pth` or `.safetensors` weights plus a JSON metadata sidecar.
Tensor payloads are streamed without changing their dtype. Remote model IDs
require an installed HuggingFace CLI; `--local-files-only` disables downloads.
`--dry-run --model /path/to/local/HF-model` inspects local weights without
downloading or writing anything. See the [Rust guide](rwkv7-rs/README.md) for
the full options and supported formats.

## Development

```sh
cargo fmt --manifest-path rwkv7-rs/Cargo.toml --check
cargo test --release --manifest-path rwkv7-rs/Cargo.toml
```

Tests cover the tokenizer against Python reference cases, logits against a
NumPy reference, checkpoint round-trips, and the command-line workflows. They
use bundled fixtures and need no Python packages or model downloads. The one
ignored ZIP64 test writes about 4.1 GiB and must be run explicitly; see the
[testing guide](rwkv7-rs/README.md#tests).

Argument parsing lives in `rwkv7-rs/src/cli/`, separate from the two binaries'
execution code. The Rust guide describes the remaining modules and fixture
regeneration tools. Python and CUDA experiments have their own requirements;
building the Rust crate does not install or run them.

## References

The Rust model is a port of the Python RWKV-7 demos from
[`BlinkDL/RWKV-LM`](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7).
Its recurrence follows [`cuda/wkv7s.cu`](cuda/wkv7s.cu). For upstream training
instructions, see
[`RWKV-v7/train_temp`](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp).
