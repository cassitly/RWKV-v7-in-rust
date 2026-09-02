//! RWKV-7 inference plus the checkpoint plumbing it needs.
//!
//! The `rwkv7` binary is the inference CLI; `qwen-make-pth` reuses the zip,
//! pickle and json modules to convert HuggingFace checkpoints.

pub mod hf;
pub mod json;
pub mod loader;
pub mod model;
pub mod ops;
pub mod pickle;
pub mod pickle_write;
pub mod pool;
pub mod sampler;
pub mod tokenizer;
pub mod zip;
pub mod zip_write;
