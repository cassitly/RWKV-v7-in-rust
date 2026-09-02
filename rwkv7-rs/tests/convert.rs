//! End-to-end checks for the HF -> .pth / .safetensors conversion.
//!
//! `tests/hf_fixture/` is a synthetic two-shard model containing vision, MTP and
//! `model.language_model.`-prefixed keys, plus bf16 / f16 / f32 / i64 / u8 / fp8
//! tensors. Regenerate it with `python3 tools/make_hf_fixture.py --out tests/hf_fixture`.

use std::path::Path;

use rwkv7::hf::{self, ShardReader};

const FIXTURE: &str = "tests/hf_fixture";

fn tmp(name: &str) -> String {
    let mut p = std::env::temp_dir();
    p.push(format!("rwkv7-convert-test-{name}"));
    p.to_string_lossy().into_owned()
}

fn kept() -> Vec<hf::ShardTensor> {
    let all = hf::scan_dir(Path::new(FIXTURE)).expect("scan failed");
    let f = hf::export_text_state(all).expect("filtering failed");
    assert_eq!(f.skipped_vision, 3, "vision filter");
    assert_eq!(f.skipped_mtp, 2, "mtp filter");
    f.kept
}

#[test]
fn scan_finds_shards_via_index() {
    let all = hf::scan_dir(Path::new(FIXTURE)).unwrap();
    // 16 tensors across two shards, 5 of which the filters remove.
    assert_eq!(all.len(), 16);
    assert!(all.iter().any(|t| t.dtype == "BF16"));
    assert!(all.iter().any(|t| t.dtype == "F8_E4M3"), "fp8 must survive scanning");
    assert!(all.iter().any(|t| t.dtype == "I64"));
    // Tensors must come from more than one file.
    let mut sources: Vec<_> = all.iter().map(|t| t.source.clone()).collect();
    sources.sort();
    sources.dedup();
    assert_eq!(sources.len(), 2, "fixture should be sharded");
}

#[test]
fn keys_are_stripped_and_filtered() {
    let names: Vec<String> = kept().into_iter().map(|t| t.name).collect();
    assert!(names.contains(&"embed_tokens.weight".to_string()));
    assert!(names.contains(&"layers.0.self_attn.q_proj.weight".to_string()));
    assert!(names.contains(&"lm_head.weight".to_string()));
    assert!(!names.iter().any(|n| n.starts_with("model.language_model.")));
    assert!(!names.iter().any(|n| n.starts_with("visual.") || n.starts_with("mtp.")));
    assert!(!names.iter().any(|n| n.contains("multi_modal_projector")));
}

/// The whole point of the tool: bytes must survive unchanged, dtypes included.
#[test]
fn pth_round_trips_byte_for_byte() {
    let kept = kept();
    let out = tmp("roundtrip.pth");
    let mut r = ShardReader::new();
    hf::write_pth(&out, &kept, &mut r).expect("write failed");

    let back = hf::scan_pth(Path::new(&out)).expect("could not read back what we wrote");
    assert_eq!(back.len(), kept.len());

    let mut r2 = ShardReader::new();
    for t in &kept {
        let b = back
            .iter()
            .find(|b| b.name == t.name)
            .unwrap_or_else(|| panic!("{} missing after round-trip", t.name));
        assert_eq!(b.dtype, t.dtype, "{}: dtype changed", t.name);
        assert_eq!(b.shape, t.shape, "{}: shape changed", t.name);
        let want = hf::read_tensor_bytes(t, &mut r).unwrap();
        let got = hf::read_tensor_bytes(b, &mut r2).unwrap();
        assert_eq!(got, want, "{}: bytes changed", t.name);
    }
    let _ = std::fs::remove_file(&out);
}

#[test]
fn safetensors_round_trips_byte_for_byte() {
    let kept = kept();
    let out = tmp("roundtrip.safetensors");
    let mut r = ShardReader::new();
    hf::write_safetensors(&out, &kept, &mut r).expect("write failed");

    let back = hf::scan_safetensors(Path::new(&out)).expect("could not read back");
    assert_eq!(back.len(), kept.len());
    let mut r2 = ShardReader::new();
    for t in &kept {
        let b = back.iter().find(|b| b.name == t.name).unwrap();
        assert_eq!(b.dtype, t.dtype);
        assert_eq!(b.shape, t.shape);
        assert_eq!(
            hf::read_tensor_bytes(b, &mut r2).unwrap(),
            hf::read_tensor_bytes(t, &mut r).unwrap(),
            "{}: bytes changed",
            t.name
        );
    }
    let _ = std::fs::remove_file(&out);
}

/// PyTorch maps records in place under `torch.load(..., mmap=True)`, which the
/// consumer of this output uses, so payloads must land on 64-byte boundaries.
#[test]
fn pth_records_are_64_byte_aligned() {
    let kept = kept();
    let out = tmp("aligned.pth");
    hf::write_pth(&out, &kept, &mut ShardReader::new()).unwrap();

    let mut z = rwkv7::zip::ZipArchive::open(&out).unwrap();
    let records: Vec<String> = z
        .names()
        .iter()
        .filter(|n| n.contains("/data/"))
        .cloned()
        .collect();
    assert_eq!(records.len(), kept.len());
    for name in records {
        let (off, _) = z.data_offset(&name).unwrap();
        assert_eq!(off % 64, 0, "{name} payload starts at {off}, not 64-byte aligned");
    }
    let _ = std::fs::remove_file(&out);
}

#[test]
fn rejects_a_directory_with_no_weights() {
    let dir = tmp("empty-dir");
    let _ = std::fs::create_dir_all(&dir);
    let e = hf::scan_dir(Path::new(&dir));
    assert!(e.is_err());
    assert!(e.unwrap_err().to_string().contains("no HF weight files"));
    let _ = std::fs::remove_dir(&dir);
}

/// Entries past 4 GiB need the zip64 extended-information extra field in both the
/// local header and the central directory, and `data_offset` must still resolve.
///
/// Ignored by default: it writes ~4.1 GiB. Run with
/// `cargo test --release --test convert -- --ignored --nocapture`.
#[test]
#[ignore]
fn zip64_entry_past_4gib() {
    use rwkv7::zip_write::ZipWriter;

    let out = tmp("zip64.pth");
    let size: u64 = (1u64 << 32) + (1 << 20); // 4 GiB + 1 MiB

    let mut z = ZipWriter::create(&out).unwrap();
    z.write_bytes("archive/data.pkl", b"placeholder").unwrap();
    // io::repeat avoids needing a 4 GiB input file on disk.
    let mut src = std::io::repeat(0xAB);
    z.write_stream("archive/data/0", &mut src, size).unwrap();
    z.write_bytes("archive/version", b"3\n").unwrap();
    z.finish().unwrap();

    let mut zr = rwkv7::zip::ZipArchive::open(&out).unwrap();
    let (off, sz) = zr.data_offset("archive/data/0").unwrap();
    assert_eq!(sz, size, "size did not survive the zip64 round-trip");
    assert_eq!(off % 64, 0, "alignment must hold for zip64 entries too");

    // Spot-check the first and last bytes of the payload.
    let mut head = [0u8; 32];
    zr.read_range("archive/data/0", 0, &mut head).unwrap();
    assert!(head.iter().all(|b| *b == 0xAB));
    let mut tail = [0u8; 32];
    zr.read_range("archive/data/0", size - 32, &mut tail).unwrap();
    assert!(tail.iter().all(|b| *b == 0xAB), "payload tail is wrong");

    // An entry written after the 4 GiB one has a >4 GiB local header offset,
    // which exercises the central directory's offset sentinel.
    let (voff, vsz) = zr.data_offset("archive/version").unwrap();
    assert!(voff > ZIP64_SENTINEL, "expected a zip64-range offset, got {voff}");
    assert_eq!(vsz, 2);

    println!("zip64 ok: {} byte entry, version record at offset {voff}", size);
    let _ = std::fs::remove_file(&out);
}

const ZIP64_SENTINEL: u64 = 0xFFFF_FFFF;
