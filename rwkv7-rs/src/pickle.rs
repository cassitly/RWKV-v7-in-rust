//! A pickle interpreter covering the subset that `torch.save` emits.
//!
//! Anything that would execute arbitrary code (plain GLOBAL/REDUCE of unknown
//! callables) is kept symbolic: `REDUCE` produces a `Value::Reduce` node that the
//! loader inspects, so nothing here can be tricked into "running" a checkpoint.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io;
use std::rc::Rc;

///  and  are never read by the torch loader, but the VM must still be
/// able to represent them or it would reject perfectly valid streams.
#[allow(dead_code)]
/// `Float` and `Bytes` are never read by the torch loader, but the VM must be able
/// to represent them or it would reject otherwise valid streams.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Rc<String>),
    Bytes(Rc<Vec<u8>>),
    Tuple(Rc<Vec<Value>>),
    List(Rc<RefCell<Vec<Value>>>),
    Dict(Rc<RefCell<Vec<(Value, Value)>>>),
    /// `module.name` referenced but not called.
    Global(Rc<(String, String)>),
    /// `persistent_load(args)` — for torch this identifies a storage record.
    PersId(Rc<Value>),
    /// `module.name(*args)`, left uninterpreted.
    Reduce(Rc<(String, String, Vec<Value>)>),
    Mark,
}

impl Value {
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(v) => Some(*v),
            Value::Bool(v) => Some(*v as i64),
            _ => None,
        }
    }
    pub fn as_usize(&self) -> Option<usize> {
        self.as_int().and_then(|v| usize::try_from(v).ok())
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn as_seq(&self) -> Option<Vec<Value>> {
        match self {
            Value::Tuple(t) => Some(t.as_ref().clone()),
            Value::List(l) => Some(l.borrow().clone()),
            _ => None,
        }
    }
}

fn err<T>(msg: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, msg.into()))
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn u8(&mut self) -> io::Result<u8> {
        if self.p >= self.b.len() {
            return err("unexpected end of pickle stream");
        }
        self.p += 1;
        Ok(self.b[self.p - 1])
    }
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            return err("unexpected end of pickle stream");
        }
        self.p += n;
        Ok(&self.b[self.p - n..self.p])
    }
    fn u16(&mut self) -> io::Result<u16> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }
    fn i32(&mut self) -> io::Result<i32> {
        let s = self.take(4)?;
        Ok(i32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(self.i32()? as u32)
    }
    fn u64(&mut self) -> io::Result<u64> {
        let s = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_le_bytes(a))
    }
    /// A newline-terminated ASCII token (GLOBAL, INT, etc.).
    fn line(&mut self) -> io::Result<&'a str> {
        let start = self.p;
        while self.p < self.b.len() && self.b[self.p] != b'\n' {
            self.p += 1;
        }
        if self.p >= self.b.len() {
            return err("unterminated line in pickle stream");
        }
        let s = &self.b[start..self.p];
        self.p += 1;
        std::str::from_utf8(s)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad utf-8 in pickle"))
    }
}

/// Interpret a pickle stream. `persistent_load` is applied to every BINPERSID
/// argument; torch uses it to point at raw storage records in the zip.
pub fn load(data: &[u8]) -> io::Result<Value> {
    let mut r = Reader { b: data, p: 0 };
    let mut stack: Vec<Value> = Vec::new();
    let mut marks: Vec<usize> = Vec::new();
    let mut memo: HashMap<u32, Value> = HashMap::new();

    macro_rules! pop {
        () => {
            match stack.pop() {
                Some(v) => v,
                None => return err("pickle stack underflow"),
            }
        };
    }
    macro_rules! last_mark {
        () => {
            match marks.pop() {
                Some(m) => m,
                None => return err("pickle mark stack underflow"),
            }
        };
    }

    loop {
        let op = r.u8()?;
        match op {
            b'\x80' => {
                r.u8()?;
            } // PROTO
            b'\x95' => {
                r.u64()?;
            } // FRAME
            b'.' => return Ok(pop!()), // STOP

            b'N' => stack.push(Value::None),
            b'\x88' => stack.push(Value::Bool(true)), // NEWTRUE
            b'\x89' => stack.push(Value::Bool(false)), // NEWFALSE

            b'K' => {
                let v = r.u8()? as i64;
                stack.push(Value::Int(v));
            } // BININT1
            b'M' => {
                let v = r.u16()? as i64;
                stack.push(Value::Int(v));
            } // BININT2
            b'J' => {
                let v = r.i32()? as i64;
                stack.push(Value::Int(v));
            } // BININT
            b'\x8a' | b'\x8b' => {
                // LONG1 / LONG4
                let n = if op == b'\x8a' {
                    r.u8()? as usize
                } else {
                    r.u32()? as usize
                };
                let raw = r.take(n)?;
                // Little-endian two's complement of arbitrary width; sign-extend into i64.
                let mut acc: i128 = 0;
                for (i, byte) in raw.iter().enumerate() {
                    acc |= (*byte as i128) << (8 * i);
                }
                if n > 0 && raw[n - 1] & 0x80 != 0 {
                    acc -= 1i128 << (8 * n);
                }
                stack.push(Value::Int(acc as i64));
            }
            b'G' => {
                // BINFLOAT (big-endian)
                let s = r.take(8)?;
                let mut a = [0u8; 8];
                a.copy_from_slice(s);
                stack.push(Value::Float(f64::from_be_bytes(a)));
            }
            b'I' | b'L' => {
                // INT / LONG (text)
                let s = r.line()?.trim_end_matches('L');
                match s {
                    "01" => stack.push(Value::Bool(true)),
                    "00" => stack.push(Value::Bool(false)),
                    _ => stack.push(Value::Int(s.parse::<i64>().unwrap_or(0))),
                }
            }

            b'X' | b'\x8c' | b'\x8d' => {
                // BINUNICODE / SHORT / 8
                let n = match op {
                    b'X' => r.u32()? as usize,
                    b'\x8c' => r.u8()? as usize,
                    _ => r.u64()? as usize,
                };
                let s = String::from_utf8_lossy(r.take(n)?).into_owned();
                stack.push(Value::Str(Rc::new(s)));
            }
            b'B' | b'C' | b'\x8e' => {
                // BINBYTES / SHORT / 8
                let n = match op {
                    b'B' => r.u32()? as usize,
                    b'C' => r.u8()? as usize,
                    _ => r.u64()? as usize,
                };
                stack.push(Value::Bytes(Rc::new(r.take(n)?.to_vec())));
            }
            b'T' | b'U' => {
                // BINSTRING / SHORT
                let n = if op == b'T' {
                    r.u32()? as usize
                } else {
                    r.u8()? as usize
                };
                let s = String::from_utf8_lossy(r.take(n)?).into_owned();
                stack.push(Value::Str(Rc::new(s)));
            }

            b'(' => {
                marks.push(stack.len());
                stack.push(Value::Mark);
            } // MARK
            b')' => stack.push(Value::Tuple(Rc::new(Vec::new()))), // EMPTY_TUPLE
            b'}' => stack.push(Value::Dict(Rc::new(RefCell::new(Vec::new())))), // EMPTY_DICT
            b']' => stack.push(Value::List(Rc::new(RefCell::new(Vec::new())))), // EMPTY_LIST
            b'\x8f' => stack.push(Value::List(Rc::new(RefCell::new(Vec::new())))), // EMPTY_SET

            b'\x85' | b'\x86' | b'\x87' => {
                // TUPLE1/2/3
                let n = (op - b'\x84') as usize;
                let at = stack.len() - n;
                let items: Vec<Value> = stack.drain(at..).collect();
                stack.push(Value::Tuple(Rc::new(items)));
            }
            b't' => {
                // TUPLE
                let m = last_mark!();
                let items: Vec<Value> = stack.drain(m + 1..).collect();
                stack.pop(); // the Mark
                stack.push(Value::Tuple(Rc::new(items)));
            }
            b'l' => {
                // LIST
                let m = last_mark!();
                let items: Vec<Value> = stack.drain(m + 1..).collect();
                stack.pop();
                stack.push(Value::List(Rc::new(RefCell::new(items))));
            }
            b'd' => {
                // DICT
                let m = last_mark!();
                let items: Vec<Value> = stack.drain(m + 1..).collect();
                stack.pop();
                let mut d = Vec::new();
                for kv in items.chunks_exact(2) {
                    d.push((kv[0].clone(), kv[1].clone()));
                }
                stack.push(Value::Dict(Rc::new(RefCell::new(d))));
            }

            b'a' => {
                // APPEND
                let v = pop!();
                match stack.last() {
                    Some(Value::List(l)) => l.borrow_mut().push(v),
                    _ => return err("APPEND on non-list"),
                }
            }
            b'e' => {
                // APPENDS
                let m = last_mark!();
                let items: Vec<Value> = stack.drain(m + 1..).collect();
                stack.pop();
                match stack.last() {
                    Some(Value::List(l)) => l.borrow_mut().extend(items),
                    // An OrderedDict rebuilt via REDUCE receives its contents here.
                    Some(Value::Dict(d)) => {
                        for kv in items.chunks_exact(2) {
                            d.borrow_mut().push((kv[0].clone(), kv[1].clone()));
                        }
                    }
                    _ => return err("APPENDS on non-list"),
                }
            }
            b's' => {
                // SETITEM
                let v = pop!();
                let k = pop!();
                match stack.last() {
                    Some(Value::Dict(d)) => d.borrow_mut().push((k, v)),
                    _ => return err("SETITEM on non-dict"),
                }
            }
            b'u' => {
                // SETITEMS
                let m = last_mark!();
                let items: Vec<Value> = stack.drain(m + 1..).collect();
                stack.pop();
                match stack.last() {
                    Some(Value::Dict(d)) => {
                        for kv in items.chunks_exact(2) {
                            d.borrow_mut().push((kv[0].clone(), kv[1].clone()));
                        }
                    }
                    _ => return err("SETITEMS on non-dict"),
                }
            }

            b'q' | b'r' => {
                // BINPUT / LONG_BINPUT
                let i = if op == b'q' { r.u8()? as u32 } else { r.u32()? };
                if let Some(v) = stack.last() {
                    memo.insert(i, v.clone());
                }
            }
            b'\x94' => {
                // MEMOIZE
                let i = memo.len() as u32;
                if let Some(v) = stack.last() {
                    memo.insert(i, v.clone());
                }
            }
            b'h' | b'j' => {
                // BINGET / LONG_BINGET
                let i = if op == b'h' { r.u8()? as u32 } else { r.u32()? };
                match memo.get(&i) {
                    Some(v) => stack.push(v.clone()),
                    None => return err(format!("pickle memo miss at {i}")),
                }
            }

            b'c' => {
                // GLOBAL
                let module = r.line()?.to_string();
                let name = r.line()?.to_string();
                stack.push(Value::Global(Rc::new((module, name))));
            }
            b'\x93' => {
                // STACK_GLOBAL
                let name = pop!();
                let module = pop!();
                let (Some(m), Some(n)) = (module.as_str(), name.as_str()) else {
                    return err("STACK_GLOBAL with non-string operands");
                };
                stack.push(Value::Global(Rc::new((m.to_string(), n.to_string()))));
            }
            b'R' | b'\x81' => {
                // REDUCE / NEWOBJ
                let args = pop!();
                let callable = pop!();
                let Value::Global(g) = callable else {
                    return err("REDUCE on a non-global callable");
                };
                let args = args.as_seq().unwrap_or_default();
                // OrderedDict()/dict() are the only constructors we materialise, so that
                // the SETITEMS/APPENDS that follow have something to write into.
                if g.1 == "OrderedDict" || g.1 == "dict" {
                    stack.push(Value::Dict(Rc::new(RefCell::new(Vec::new()))));
                } else {
                    stack.push(Value::Reduce(Rc::new((g.0.clone(), g.1.clone(), args))));
                }
            }
            b'b' => {
                // BUILD
                let state = pop!();
                let obj = pop!();
                // Merge __setstate__ payloads into dict-like objects; otherwise ignore.
                if let (Value::Dict(d), Value::Dict(s)) = (&obj, &state) {
                    d.borrow_mut().extend(s.borrow().iter().cloned());
                }
                stack.push(obj);
            }
            b'Q' => {
                // BINPERSID
                let arg = pop!();
                stack.push(Value::PersId(Rc::new(arg)));
            }
            b'P' => {
                // PERSID
                let s = r.line()?.to_string();
                stack.push(Value::PersId(Rc::new(Value::Str(Rc::new(s)))));
            }

            b'0' => {
                stack.pop();
            } // POP
            b'1' => {
                // POP_MARK
                let m = last_mark!();
                stack.truncate(m);
            }
            b'2' => {
                // DUP
                if let Some(v) = stack.last().cloned() {
                    stack.push(v);
                }
            }

            other => {
                return err(format!(
                    "unsupported pickle opcode {other:#04x} ({:?}) at byte {}",
                    other as char,
                    r.p - 1
                ))
            }
        }
    }
}
