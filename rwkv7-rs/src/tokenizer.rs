//! RWKV "world" tokenizer.
//!
//! Vocabulary lines look like `<id> <python-literal> <byte-length>`, e.g.
//!     65   'A'      1
//!     129  b'\x80'  1
//!     40   "'"      1
//! Encoding is plain greedy longest-match over raw bytes, implemented with a
//! byte trie rather than the Python demo's two-level table — same result, and
//! it is the natural structure in Rust.

use std::io;

struct Node {
    /// Sorted by byte for binary search; most nodes have only a handful of children.
    children: Vec<(u8, u32)>,
    token: Option<u32>,
}

pub struct Tokenizer {
    nodes: Vec<Node>,
    id_to_bytes: Vec<Vec<u8>>,
}

fn err<T>(msg: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, msg.into()))
}

/// Decode a Python `'...'`, `"..."` or `b'...'` literal into the bytes it denotes.
///
/// The `b` prefix matters: inside a str literal `\xNN` is the *code point* U+00NN
/// and gets UTF-8 encoded (so `\x80` is two bytes), while inside a bytes literal
/// it is the raw byte `0xNN`.
fn parse_literal(src: &str) -> Option<Vec<u8>> {
    let s = src.trim();
    let (is_bytes, s) = match s.strip_prefix('b') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    let quote = s.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let body = &s[1..s.len().checked_sub(1)?];
    if !s.ends_with(quote) {
        return None;
    }

    let mut out: Vec<u8> = Vec::with_capacity(body.len());
    let mut it = body.chars();
    let push_cp = |out: &mut Vec<u8>, cp: u32| {
        if is_bytes {
            out.push(cp as u8);
        } else {
            match char::from_u32(cp) {
                Some(c) => {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
                // Lone surrogates: emit WTF-8-ish bytes so lengths still line up.
                None => out.extend_from_slice(&[0xef, 0xbf, 0xbd]),
            }
        }
    };

    while let Some(c) = it.next() {
        if c != '\\' {
            if is_bytes {
                out.push(c as u8);
            } else {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
            continue;
        }
        let e = it.next()?;
        match e {
            'n' => out.push(b'\n'),
            'r' => out.push(b'\r'),
            't' => out.push(b'\t'),
            'a' => out.push(0x07),
            'b' => out.push(0x08),
            'f' => out.push(0x0c),
            'v' => out.push(0x0b),
            '0' => out.push(0x00),
            '\\' => out.push(b'\\'),
            '\'' => out.push(b'\''),
            '"' => out.push(b'"'),
            'x' | 'u' | 'U' => {
                let n = match e {
                    'x' => 2,
                    'u' => 4,
                    _ => 8,
                };
                let mut cp = 0u32;
                for _ in 0..n {
                    cp = cp * 16 + it.next()?.to_digit(16)?;
                }
                push_cp(&mut out, cp);
            }
            other => {
                // Unknown escape: Python keeps the backslash verbatim.
                out.push(b'\\');
                let mut buf = [0u8; 4];
                out.extend_from_slice(other.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    Some(out)
}

impl Tokenizer {
    pub fn from_file(path: &str) -> io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut nodes = vec![Node {
            children: Vec::new(),
            token: None,
        }];
        let mut id_to_bytes: Vec<Vec<u8>> = Vec::new();

        for (lineno, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let Some(first) = line.find(' ') else {
                continue;
            };
            let Some(last) = line.rfind(' ') else {
                continue;
            };
            if last <= first {
                continue;
            }
            let Ok(id) = line[..first].parse::<u32>() else {
                continue;
            };
            let Ok(want_len) = line[last + 1..].trim().parse::<usize>() else {
                continue;
            };

            let Some(bytes) = parse_literal(&line[first + 1..last]) else {
                return err(format!(
                    "vocab line {}: cannot parse token literal",
                    lineno + 1
                ));
            };
            // The trailing length column is a free consistency check on the parser.
            if bytes.len() != want_len {
                return err(format!(
                    "vocab line {}: decoded {} bytes but the file says {want_len}",
                    lineno + 1,
                    bytes.len()
                ));
            }

            if id as usize >= id_to_bytes.len() {
                id_to_bytes.resize(id as usize + 1, Vec::new());
            }
            id_to_bytes[id as usize] = bytes.clone();

            let mut cur = 0u32;
            for b in bytes {
                let idx = nodes[cur as usize]
                    .children
                    .iter()
                    .position(|(c, _)| *c == b);
                cur = match idx {
                    Some(i) => nodes[cur as usize].children[i].1,
                    None => {
                        let new = nodes.len() as u32;
                        nodes.push(Node {
                            children: Vec::new(),
                            token: None,
                        });
                        nodes[cur as usize].children.push((b, new));
                        new
                    }
                };
            }
            nodes[cur as usize].token = Some(id);
        }

        for n in nodes.iter_mut() {
            n.children.sort_unstable_by_key(|(b, _)| *b);
        }
        Ok(Tokenizer { nodes, id_to_bytes })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    pub fn encode_bytes(&self, src: &[u8]) -> Vec<u32> {
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < src.len() {
            let mut cur = 0u32;
            let mut best: Option<(u32, usize)> = None;
            let mut j = i;
            while j < src.len() {
                let node = &self.nodes[cur as usize];
                let Ok(k) = node.children.binary_search_by_key(&src[j], |(b, _)| *b) else {
                    break;
                };
                cur = node.children[k].1;
                j += 1;
                if let Some(t) = self.nodes[cur as usize].token {
                    best = Some((t, j - i));
                }
            }
            match best {
                Some((t, len)) => {
                    out.push(t);
                    i += len;
                }
                // Every single byte is in the vocab, so this is unreachable for a
                // well-formed file; skip rather than loop forever if it is not.
                None => i += 1,
            }
        }
        out
    }

    pub fn encode(&self, s: &str) -> Vec<u32> {
        self.encode_bytes(s.as_bytes())
    }

    pub fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for t in tokens {
            if let Some(b) = self.id_to_bytes.get(*t as usize) {
                out.extend_from_slice(b);
            }
        }
        out
    }

    /// Decode to a string, or `None` when the tail is a partial UTF-8 sequence
    /// (the caller then buffers and retries after the next token).
    pub fn decode(&self, tokens: &[u32]) -> Option<String> {
        String::from_utf8(self.decode_bytes(tokens)).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab_path() -> Option<String> {
        for p in ["../rwkv_vocab_v20230424.txt", "rwkv_vocab_v20230424.txt"] {
            if std::path::Path::new(p).exists() {
                return Some(p.to_string());
            }
        }
        None
    }

    /// Every token id in the file must round-trip through the trie back to itself.
    #[test]
    fn single_tokens_round_trip() {
        let Some(p) = vocab_path() else { return };
        let tk = Tokenizer::from_file(&p).unwrap();
        let mut checked = 0;
        for id in 1..tk.vocab_size() as u32 {
            let bytes = tk.decode_bytes(&[id]);
            if bytes.is_empty() {
                continue;
            }
            let re = tk.encode_bytes(&bytes);
            assert_eq!(re, vec![id], "token {id} ({bytes:?}) did not round-trip");
            checked += 1;
        }
        assert!(checked > 60_000, "only checked {checked} tokens");
    }

    /// Compare against ids produced by the upstream Python RWKV_TOKENIZER.
    /// Regenerate the fixture with `python3 tools/gen_tokenizer_cases.py out.json`.
    #[test]
    fn matches_python_reference() {
        let Some(p) = vocab_path() else { return };
        let tk = Tokenizer::from_file(&p).unwrap();
        let fixture = std::fs::read_to_string("tests/tokenizer_cases.tsv").unwrap();

        let mut cases = 0usize;
        let mut tokens = 0usize;
        for line in fixture.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let (hex, ids) = line.split_once('\t').expect("malformed fixture line");
            let bytes: Vec<u8> = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect();
            let expect: Vec<u32> = ids
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.parse().unwrap())
                .collect();

            let got = tk.encode_bytes(&bytes);
            assert_eq!(
                got,
                expect,
                "mismatch on {:?}",
                String::from_utf8_lossy(&bytes)
                    .chars()
                    .take(60)
                    .collect::<String>()
            );
            assert_eq!(tk.decode_bytes(&got), bytes, "decode did not invert encode");
            cases += 1;
            tokens += got.len();
        }
        assert!(cases > 300, "fixture only had {cases} cases");
        eprintln!("checked {cases} cases / {tokens} tokens against the Python tokenizer");
    }
}
