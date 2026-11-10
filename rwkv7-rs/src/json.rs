//! A small JSON reader/writer.
//!
//! Needed for HuggingFace `*.index.json` shard maps and safetensors headers,
//! and for emitting the metadata sidecar. Deliberately complete rather than a
//! scanner: tensor names in the wild contain dots, slashes and escapes.

use std::collections::BTreeMap;
use std::fmt::Write as _;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    /// Insertion-ordered; JSON objects are ordered on the wire even if unordered in theory.
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_obj(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as usize),
            _ => None,
        }
    }
}

pub fn parse(src: &str) -> Result<Json, String> {
    let b = src.as_bytes();
    let mut p = Parser { b, i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != b.len() {
        return Err(format!("trailing data at byte {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: u8) -> Result<(), String> {
        if self.i < self.b.len() && self.b[self.i] == c {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", c as char, self.i))
        }
    }
    fn lit(&mut self, s: &str) -> bool {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        if self.i >= self.b.len() {
            return Err("unexpected end of json".into());
        }
        match self.b[self.i] {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => {
                if self.lit("true") {
                    Ok(Json::Bool(true))
                } else {
                    Err(self.bad())
                }
            }
            b'f' => {
                if self.lit("false") {
                    Ok(Json::Bool(false))
                } else {
                    Err(self.bad())
                }
            }
            b'n' => {
                if self.lit("null") {
                    Ok(Json::Null)
                } else {
                    Err(self.bad())
                }
            }
            _ => self.number(),
        }
    }

    fn bad(&self) -> String {
        format!("invalid literal at byte {}", self.i)
    }

    fn object(&mut self) -> Result<Json, String> {
        self.eat(b'{')?;
        let mut out = Vec::new();
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b'}' {
            self.i += 1;
            return Ok(Json::Obj(out));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            self.eat(b':')?;
            self.ws();
            out.push((k, self.value()?));
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(out));
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.i)),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.eat(b'[')?;
        let mut out = Vec::new();
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b']' {
            self.i += 1;
            return Ok(Json::Arr(out));
        }
        loop {
            self.ws();
            out.push(self.value()?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(out));
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.i)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let Some(&c) = self.b.get(self.i) else {
                return Err("unterminated string".into());
            };
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let Some(&e) = self.b.get(self.i) else {
                        return Err("unterminated escape".into());
                    };
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            // Surrogate pair: the low half follows as a second \u escape.
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if self.b.get(self.i) == Some(&b'\\')
                                    && self.b.get(self.i + 1) == Some(&b'u')
                                {
                                    self.i += 2;
                                    let lo = self.hex4()?;
                                    0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                                } else {
                                    0xFFFD
                                }
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                        _ => return Err(format!("bad escape '\\{}'", e as char)),
                    }
                }
                _ => {
                    // Copy the raw UTF-8 byte run; JSON strings are already UTF-8.
                    let start = self.i - 1;
                    while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\'
                    {
                        self.i += 1;
                    }
                    out.push_str(&String::from_utf8_lossy(&self.b[start..self.i]));
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() {
            return Err("truncated \\u escape".into());
        }
        let s = std::str::from_utf8(&self.b[self.i..self.i + 4]).map_err(|_| "bad \\u escape")?;
        self.i += 4;
        u32::from_str_radix(s, 16).map_err(|_| "bad \\u escape".into())
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.i < self.b.len() && (self.b[self.i] == b'-' || self.b[self.i] == b'+') {
            self.i += 1;
        }
        while self.i < self.b.len()
            && matches!(
                self.b[self.i],
                b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-'
            )
        {
            self.i += 1;
        }
        if start == self.i {
            return Err(format!("expected a value at byte {start}"));
        }
        std::str::from_utf8(&self.b[start..self.i])
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .map(Json::Num)
            .ok_or_else(|| format!("bad number at byte {start}"))
    }
}

// ---------------------------------------------------------------------------
// writing
// ---------------------------------------------------------------------------

pub fn escape(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A JSON value being built for output. `Raw` splices in already-serialised text,
/// which is how `config.json` is passed through with its key order and exact
/// values intact; only its indentation is adjusted to fit the enclosing document.
pub enum Out {
    Raw(String),
    Str(String),
    Num(i64),
    Obj(BTreeMap<String, Out>),
}

impl Out {
    pub fn write(&self, indent: usize, buf: &mut String) {
        let pad = "  ".repeat(indent);
        let pad1 = "  ".repeat(indent + 1);
        match self {
            Out::Raw(s) => {
                // Re-indent continuation lines so spliced JSON lines up with its parent.
                for (i, line) in s.lines().enumerate() {
                    if i > 0 {
                        buf.push('\n');
                        buf.push_str(&pad);
                    }
                    buf.push_str(line);
                }
            }
            Out::Str(s) => escape(s, buf),
            Out::Num(n) => {
                let _ = write!(buf, "{n}");
            }
            Out::Obj(kv) => {
                if kv.is_empty() {
                    buf.push_str("{}");
                    return;
                }
                buf.push_str("{\n");
                for (i, (k, v)) in kv.iter().enumerate() {
                    buf.push_str(&pad1);
                    escape(k, buf);
                    buf.push_str(": ");
                    v.write(indent + 1, buf);
                    if i + 1 < kv.len() {
                        buf.push(',');
                    }
                    buf.push('\n');
                }
                buf.push_str(&pad);
                buf.push('}');
            }
        }
    }

    pub fn to_string_pretty(&self) -> String {
        let mut s = String::new();
        self.write(0, &mut s);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_index_shape() {
        let src = r#"{"metadata":{"total_size":1234},
                      "weight_map":{"model.layers.0.mlp.gate_proj.weight":"model-00001-of-00002.safetensors",
                                    "lm_head.weight":"model-00002-of-00002.safetensors"}}"#;
        let j = parse(src).unwrap();
        assert_eq!(
            j.get("metadata")
                .unwrap()
                .get("total_size")
                .unwrap()
                .as_usize(),
            Some(1234)
        );
        let wm = j.get("weight_map").unwrap().as_obj().unwrap();
        assert_eq!(wm.len(), 2);
        assert_eq!(wm[0].1.as_str(), Some("model-00001-of-00002.safetensors"));
    }

    #[test]
    fn handles_escapes_and_unicode() {
        let j = parse(r#"{"a\/b":"x\ty é 🚀 \"q\""}"#).unwrap();
        assert_eq!(j.get("a/b").unwrap().as_str(), Some("x\ty é 🚀 \"q\""));
    }

    #[test]
    fn handles_nesting_and_scalars() {
        let j = parse(r#"{"a":[1,-2.5,1e3,true,false,null,{"b":[]}],"c":{}}"#).unwrap();
        let a = j.get("a").unwrap().as_arr().unwrap();
        assert_eq!(a.len(), 7);
        assert_eq!(a[0], Json::Num(1.0));
        assert_eq!(a[2], Json::Num(1000.0));
        assert_eq!(a[5], Json::Null);
        assert_eq!(j.get("c").unwrap().as_obj().unwrap().len(), 0);
    }

    #[test]
    fn rejects_malformed() {
        for bad in [
            r#"{"a":}"#,
            r#"{"a" 1}"#,
            "[1,2",
            r#""unterminated"#,
            "{} junk",
        ] {
            assert!(parse(bad).is_err(), "should have rejected {bad:?}");
        }
    }

    #[test]
    fn writes_sorted_and_indented() {
        let mut m = BTreeMap::new();
        m.insert("z".into(), Out::Num(1));
        m.insert("a".into(), Out::Str("hi\n".into()));
        m.insert("cfg".into(), Out::Raw("{\"k\": 1}".into()));
        let s = Out::Obj(m).to_string_pretty();
        assert_eq!(
            s,
            "{\n  \"a\": \"hi\\n\",\n  \"cfg\": {\"k\": 1},\n  \"z\": 1\n}"
        );
    }

    #[test]
    fn raw_multiline_is_reindented() {
        let mut m = BTreeMap::new();
        m.insert(
            "config".into(),
            Out::Raw("{\n  \"a\": 1,\n  \"b\": {\n    \"c\": 2\n  }\n}".into()),
        );
        let s = Out::Obj(m).to_string_pretty();
        // Every spliced line must sit at or below its key's indent level.
        for line in s.lines().skip(1) {
            if line.starts_with('}') {
                continue;
            }
            assert!(line.starts_with("  "), "under-indented line: {line:?}");
        }
        assert!(
            s.contains("\n      \"c\": 2"),
            "nested value not re-indented:\n{s}"
        );
    }
}
