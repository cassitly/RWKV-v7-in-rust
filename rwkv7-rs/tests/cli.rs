//! Exercise the user-facing workflows with the small bundled checkpoints.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

use rwkv7::cli::{export, inference};

fn arguments(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).into()).collect()
}

fn inference_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_rwkv7"));
    command.args(["--threads", "1"]);
    command
}

fn success(output: Output) -> Output {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

// Each test owns a unique directory, including when multiple test runs overlap.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "rwkv7-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn help_does_not_require_model_or_output() {
    for binary in [
        env!("CARGO_BIN_EXE_rwkv7"),
        env!("CARGO_BIN_EXE_qwen-make-pth"),
    ] {
        let output = success(Command::new(binary).arg("--help").output().unwrap());
        assert!(String::from_utf8_lossy(&output.stdout).contains("USAGE:"));
        assert!(output.stderr.is_empty());
    }
}

#[test]
fn sampling_and_chunk_values_are_validated_before_loading() {
    for (flag, value) in [
        ("--temperature", "0"),
        ("--temperature", "-1"),
        ("--temperature", "NaN"),
        ("--temperature", "inf"),
        ("--top-p", "-0.1"),
        ("--top-p", "1.1"),
        ("--top-p", "NaN"),
        ("--chunk", "0"),
        ("--length", "oops"),
    ] {
        let output = inference_command()
            .args(["--model", "does-not-exist.pth", flag, value])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{flag} {value} was accepted");
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains(flag), "{error}");
        assert!(!error.contains("loading model"), "{error}");
    }
    for top_p in ["0", "1"] {
        assert!(
            inference::parse(arguments(&["--model", "m.pth", "--top-p", top_p]))
                .unwrap()
                .is_some()
        );
    }
}

#[test]
fn prompt_sources_are_exclusive_and_token_ids_are_strict() {
    for flags in [
        vec!["--prompt", "text", "--tokens", "1"],
        vec!["--prompt-file", "missing.txt", "--prompt", "text"],
        vec!["--tokens", "1", "--prompt-file", "missing.txt"],
    ] {
        let mut argv = arguments(&["--model", "m.pth"]);
        argv.extend(arguments(&flags));
        assert!(inference::parse(argv)
            .unwrap_err()
            .contains("choose only one"));
    }
    for ids in ["", "1,,2", "1,", "-1", "4294967296", "abc"] {
        let argv = arguments(&["--model", "m.pth", "--tokens", ids]);
        assert!(inference::parse(argv).is_err(), "accepted {ids:?}");
    }
    assert!(
        inference::parse(arguments(&["--encode-only", "--tokens", "1"]))
            .unwrap_err()
            .contains("requires text")
    );
    assert!(
        inference::parse(arguments(&["--model", "m.pth", "--tokens", "1, 2"]))
            .unwrap()
            .is_some()
    );
}

#[test]
fn missing_values_and_unknown_options_are_actionable() {
    assert!(inference::parse(arguments(&["--model"]))
        .unwrap_err()
        .contains("--model needs a value"));
    assert!(
        inference::parse(arguments(&["--length", "--model", "m.pth"]))
            .unwrap_err()
            .contains("--length needs a value")
    );
    assert!(export::parse(arguments(&["--format", "unknown"]))
        .unwrap_err()
        .contains("--format must be"));
    assert!(export::parse(arguments(&["--output"]))
        .unwrap_err()
        .contains("--output needs a value"));
    assert!(inference::parse(arguments(&["--unknown"]))
        .unwrap_err()
        .contains("--help"));
}

#[test]
fn encode_only_reads_text_and_prompt_files_without_a_model() {
    let output = success(
        inference_command()
            .args(["--encode-only", "--prompt", "A"])
            .output()
            .unwrap(),
    );
    assert_eq!(output.stdout, b"66\n");

    let dir = TempDir::new();
    let prompt = dir.0.join("prompt.txt");
    std::fs::write(&prompt, "A").unwrap();
    let output = success(
        inference_command()
            .args(["--encode-only", "--prompt-file"])
            .arg(prompt)
            .output()
            .unwrap(),
    );
    assert_eq!(output.stdout, b"66\n");
}

#[test]
fn logit_dump_matches_reference_for_both_checkpoint_formats() {
    let dir = TempDir::new();
    let tokens = std::fs::read_to_string("tests/tiny_tokens.txt").unwrap();
    let expected = std::fs::read("tests/tiny_logits.bin").unwrap();
    for format in ["pth", "safetensors"] {
        let path = dir.0.join(format!("logits-{format}.bin"));
        let output = success(
            inference_command()
                .args([
                    "--model",
                    &format!("tests/tiny.{format}"),
                    "--tokens",
                    tokens.trim(),
                    "--length",
                    "0",
                    "--show-top",
                    "0",
                    "--chunk",
                    "3",
                    "--dump-logits",
                ])
                .arg(&path)
                .output()
                .unwrap(),
        );
        assert!(output.stdout.is_empty());
        let actual = std::fs::read(&path).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.chunks_exact(4).zip(expected.chunks_exact(4)) {
            let a = f32::from_le_bytes(a.try_into().unwrap());
            let b = f32::from_le_bytes(b.try_into().unwrap());
            assert!((a - b).abs() < 1e-4, "{format}: {a} != {b}");
        }
    }
}

#[test]
fn token_generation_echoes_the_actual_input() {
    let output = success(
        inference_command()
            .args([
                "--model",
                "tests/tiny.pth",
                "--tokens",
                "66",
                "--length",
                "2",
                "--show-top",
                "0",
                "--seed",
                "42",
            ])
            .output()
            .unwrap(),
    );
    assert!(output.stdout.starts_with(b"A"));
    assert!(!String::from_utf8_lossy(&output.stdout).contains("simulate SpaceX"));
}

#[test]
fn explicit_missing_vocab_and_out_of_range_tokens_fail_cleanly() {
    let output = inference_command()
        .args([
            "--model",
            "tests/tiny.pth",
            "--tokens",
            "1",
            "--length",
            "0",
            "--vocab",
            "missing-vocab.txt",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("vocab file not found"));

    let output = inference_command()
        .args([
            "--model",
            "tests/tiny.pth",
            "--tokens",
            "96",
            "--length",
            "0",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("outside the model vocab"));
}

#[test]
fn dry_run_writes_nothing_and_never_downloads() {
    let dir = TempDir::new();
    success(
        Command::new(env!("CARGO_BIN_EXE_qwen-make-pth"))
            .args(["--model", "tests/hf_fixture", "--dry-run", "--output"])
            .arg(dir.0.join("unused.pth"))
            .output()
            .unwrap(),
    );
    assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);

    let output = Command::new(env!("CARGO_BIN_EXE_qwen-make-pth"))
        .args(["--model", "missing-org/missing-model", "--dry-run"])
        .env("PATH", &dir.0)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("--dry-run requires an existing local model directory"));
}

#[test]
fn exporter_writes_both_formats_with_consistent_metadata() {
    let dir = TempDir::new();
    for format in ["pth", "safetensors"] {
        let path = dir.0.join(format!("nested/text.{format}"));
        let output = success(
            Command::new(env!("CARGO_BIN_EXE_qwen-make-pth"))
                .args([
                    "--model",
                    "tests/hf_fixture",
                    "--local-files-only",
                    "--format",
                    format,
                    "--output",
                ])
                .arg(&path)
                .output()
                .unwrap(),
        );
        let summary = rwkv7::json::parse(std::str::from_utf8(&output.stdout).unwrap()).unwrap();
        let sidecar = std::fs::read_to_string(format!("{}.json", path.display())).unwrap();
        let metadata = rwkv7::json::parse(&sidecar).unwrap();
        for key in [
            "kept_tensors",
            "skipped_vision_tensors",
            "skipped_mtp_tensors",
            "numel",
            "dtypes",
        ] {
            assert_eq!(summary.get(key), metadata.get(key), "{key}");
        }
        assert_eq!(metadata.get("kept_tensors").unwrap().as_usize(), Some(11));
        let tensors = if format == "pth" {
            rwkv7::hf::scan_pth(Path::new(&path)).unwrap()
        } else {
            rwkv7::hf::scan_safetensors(Path::new(&path)).unwrap()
        };
        assert_eq!(tensors.len(), 11);
    }
}
