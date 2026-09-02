////////////////////////////////////////////////////////////////////////////////
// RWKV-7 "Goose" x070 — a single-file Rust implementation
//
// A port of rwkv_v7_demo_rnn.py from https://github.com/BlinkDL/RWKV-LM
// The recurrence follows the scalar form of cuda/wkv7s.cu
//
// Standalone on purpose: no crates, no build system, nothing but libstd.
//
//     rustc -O rwkv7_demo.rs -o rwkv7_demo
//     ./rwkv7_demo --model /path/to/rwkv7-g1a-0.1b-20250728-ctx4096.pth
//
// Read it top to bottom. The model is first; the checkpoint reader and
// tokenizer are plumbing and live at the bottom.
//
// This runs in RNN mode — one token at a time, constant memory, no KV cache.
// That is the clearest form of RWKV-7 and the one worth understanding. For a
// batched prefill path and a faster CPU kernel, see ./rwkv7-rs/.
////////////////////////////////////////////////////////////////////////////////

use std::collections::HashMap;
use std::convert::TryFrom; // so a bare `rustc` (edition 2015) works too
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::OnceLock;

////////////////////////////////////////////////////////////////////////////////
// Configuration
////////////////////////////////////////////////////////////////////////////////

/// Edit these, or override them on the command line.
const MODEL: &str = "/mnt/e/RWKV-Runner/models/rwkv7-g1a-0.1b-20250728-ctx4096.pth";
const VOCAB: &str = "rwkv_vocab_v20230424.txt";
const PROMPT: &str = "User: simulate SpaceX mars landing using python\n\nAssistant: <think";
const LENGTH: usize = 500;
const TEMPERATURE: f32 = 1.0;
/// 0.0 means greedy — the same convention the Python demos use.
const TOP_P: f32 = 0.0;

/// LayerNorm epsilon.
const LN_EPS: f32 = 1e-5;
/// GroupNorm epsilon. RWKV-7 uses 64e-5, NOT the usual 1e-5. Getting this wrong
/// does not crash; it quietly degrades the output.
const GN_EPS: f32 = 64e-5;
/// exp(-0.5). The per-channel decay is exp(-DECAY * sigmoid(w0 + w)).
const DECAY: f32 = 0.606_531;

////////////////////////////////////////////////////////////////////////////////
//
// THE MODEL
//
////////////////////////////////////////////////////////////////////////////////

struct Config {
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    head_size: usize,
    dim_ffn: usize,
    vocab_size: usize,
}

/// Everything RWKV-7 carries between tokens.
///
/// There is no KV cache: memory is constant in context length. `att_kv` is the
/// actual recurrent state and must stay f32 — running it in bf16/fp16 drifts
/// visibly within a few hundred tokens, even when everything else is halved.
struct State {
    /// [n_layer][n_embd] — previous token's input to time mixing
    att_x: Vec<Vec<f32>>,
    /// [n_layer][n_head * head_size * head_size]
    att_kv: Vec<Vec<f32>>,
    /// [n_layer][n_embd] — previous token's input to channel mixing
    ffn_x: Vec<Vec<f32>>,
}

impl State {
    fn new(c: &Config) -> State {
        State {
            att_x: vec![vec![0.0; c.n_embd]; c.n_layer],
            att_kv: vec![vec![0.0; c.n_head * c.head_size * c.head_size]; c.n_layer],
            ffn_x: vec![vec![0.0; c.n_embd]; c.n_layer],
        }
    }
}

/// One block's weights. Two matrix conventions are in play, matching the
/// checkpoint layout:
///   * `nn.Linear.weight` is [out, in]  -> `matvec`   (y = W @ x)
///   * LoRA factors are    [in, out]    -> `matvec_t` (y = x @ M)
struct Layer {
    ln1_w: Vec<f32>, ln1_b: Vec<f32>,
    ln2_w: Vec<f32>, ln2_b: Vec<f32>,

    // token-shift lerp coefficients, one set per projection
    x_r: Vec<f32>, x_w: Vec<f32>, x_k: Vec<f32>,
    x_v: Vec<f32>, x_a: Vec<f32>, x_g: Vec<f32>,

    // low-rank projections: (bias, [n_embd, rank], [rank, n_embd])
    w0: Vec<f32>, w1: Vec<f32>, w2: Vec<f32>, w_rank: usize,
    a0: Vec<f32>, a1: Vec<f32>, a2: Vec<f32>, a_rank: usize,
    g1: Vec<f32>, g2: Vec<f32>, g_rank: usize,
    /// None on layer 0, which defines `v_first` instead of mixing towards it.
    v0: Option<Vec<f32>>, v1: Vec<f32>, v2: Vec<f32>, v_rank: usize,

    k_k: Vec<f32>, k_a: Vec<f32>, r_k: Vec<f32>,

    r_w: Vec<f32>, k_w: Vec<f32>, v_w: Vec<f32>, o_w: Vec<f32>,
    ln_x_w: Vec<f32>, ln_x_b: Vec<f32>,

    ffn_x_k: Vec<f32>, ffn_k: Vec<f32>, ffn_v: Vec<f32>,
}

struct Model {
    cfg: Config,
    /// ln0 is folded into the embedding table at load time, as the demos do.
    emb: Vec<f32>,
    layers: Vec<Layer>,
    ln_out_w: Vec<f32>,
    ln_out_b: Vec<f32>,
    head: Vec<f32>,
}

impl Model {
    /// One token in, logits over the whole vocabulary out.
    fn forward(&self, token: u32, s: &mut State) -> Vec<f32> {
        let c = &self.cfg;
        let mut x = self.emb[token as usize * c.n_embd..][..c.n_embd].to_vec();

        // Layer 0 publishes its value vector; every later layer mixes towards it.
        let mut v_first = vec![0.0f32; c.n_embd];

        for (i, l) in self.layers.iter().enumerate() {
            let xn = layer_norm(&x, &l.ln1_w, &l.ln1_b);
            let dx = self.time_mixing(i, l, &xn, s, &mut v_first);
            for k in 0..c.n_embd {
                x[k] += dx[k];
            }

            let xn = layer_norm(&x, &l.ln2_w, &l.ln2_b);
            let dx = self.channel_mixing(i, l, &xn, s);
            for k in 0..c.n_embd {
                x[k] += dx[k];
            }
        }

        let xn = layer_norm(&x, &self.ln_out_w, &self.ln_out_b);
        matvec(&self.head, c.vocab_size, c.n_embd, &xn)
    }

    /// Time mixing — RWKV-7's replacement for attention.
    fn time_mixing(
        &self,
        layer_id: usize,
        l: &Layer,
        x: &[f32],
        s: &mut State,
        v_first: &mut Vec<f32>,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (h, n, e) = (c.n_head, c.head_size, c.n_embd);

        // Token shift: interpolate between this token and the previous one, with a
        // separately learned coefficient per projection. Swapping the state out
        // here both reads the old value and stores the new one.
        let x_prev = std::mem::replace(&mut s.att_x[layer_id], x.to_vec());
        let lerp = |m: &[f32]| -> Vec<f32> {
            (0..e).map(|k| x[k] + (x_prev[k] - x[k]) * m[k]).collect()
        };
        let (xr, xw, xk, xv, xa, xg) = (
            lerp(&l.x_r), lerp(&l.x_w), lerp(&l.x_k),
            lerp(&l.x_v), lerp(&l.x_a), lerp(&l.x_g),
        );

        let r = matvec(&l.r_w, e, e, &xr);
        let k = matvec(&l.k_w, e, e, &xk);
        let mut v = matvec(&l.v_w, e, e, &xv);

        // Three LoRA-style low-rank projections.
        //   w: the decay,  a: the in-context learning rate,  g: the output gate
        let mut w = {
            let mut t = matvec_t(&l.w1, e, l.w_rank, &xw);
            for u in t.iter_mut() {
                *u = u.tanh();
            }
            matvec_t(&l.w2, l.w_rank, e, &t)
        };
        let a = {
            let t = matvec_t(&l.a1, e, l.a_rank, &xa);
            let mut t = matvec_t(&l.a2, l.a_rank, e, &t);
            for k in 0..e {
                t[k] = sigmoid(l.a0[k] + t[k]);
            }
            t
        };
        let g = {
            // Note the nonlinearity sits *between* the two factors here.
            let mut t = matvec_t(&l.g1, e, l.g_rank, &xg);
            for u in t.iter_mut() {
                *u = sigmoid(*u);
            }
            matvec_t(&l.g2, l.g_rank, e, &t)
        };

        // Removal key: unit-norm per head, so it defines a direction to subtract.
        let mut kk: Vec<f32> = (0..e).map(|i| k[i] * l.k_k[i]).collect();
        l2_normalize_heads(&mut kk, h, n);

        // Scale the key by the learning rate.
        let k: Vec<f32> = (0..e).map(|i| k[i] * (1.0 + (a[i] - 1.0) * l.k_a[i])).collect();

        // Value residual: every layer can pull back towards layer 0's value.
        if layer_id == 0 {
            v_first.copy_from_slice(&v);
        } else {
            let t = matvec_t(&l.v1, e, l.v_rank, &xv);
            let t = matvec_t(&l.v2, l.v_rank, e, &t);
            let v0 = l.v0.as_ref().expect("layers past 0 must have a v-mix LoRA");
            for i in 0..e {
                v[i] += (v_first[i] - v[i]) * sigmoid(v0[i] + t[i]);
            }
        }

        // Decay, in (0, 1) per channel. The naive spelling is
        //   exp(-exp(-softplus(-(w0+w)) - 0.5))
        // which simplifies to exactly this.
        for i in 0..e {
            w[i] = (-DECAY * sigmoid(l.w0[i] + w[i])).exp();
        }

        // ---- the recurrence ----
        let mut y = wkv7(&mut s.att_kv[layer_id], h, n, &r, &w, &k, &v, &kk, &a);

        group_norm_heads(&mut y, h, n, &l.ln_x_w, &l.ln_x_b);

        // Bonus: a per-head rank-1 read-out added on top of the state's output.
        for head in 0..h {
            let b = head * n;
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += r[b + j] * k[b + j] * l.r_k[b + j];
            }
            for j in 0..n {
                y[b + j] += acc * v[b + j];
            }
        }

        for i in 0..e {
            y[i] *= g[i];
        }
        matvec(&l.o_w, e, e, &y)
    }

    /// Channel mixing — a two-layer MLP with a squared ReLU and a token shift.
    fn channel_mixing(&self, layer_id: usize, l: &Layer, x: &[f32], s: &mut State) -> Vec<f32> {
        let c = &self.cfg;
        let x_prev = std::mem::replace(&mut s.ffn_x[layer_id], x.to_vec());
        let xk: Vec<f32> = (0..c.n_embd)
            .map(|i| x[i] + (x_prev[i] - x[i]) * l.ffn_x_k[i])
            .collect();

        let mut k = matvec(&l.ffn_k, c.dim_ffn, c.n_embd, &xk);
        for u in k.iter_mut() {
            let t = u.max(0.0);
            *u = t * t; // ReLU squared
        }
        matvec(&l.ffn_v, c.n_embd, c.dim_ffn, &k)
    }
}

/// The RWKV-7 recurrence, for one token, all heads.
///
/// Each head owns an `n x n` state matrix `S`, indexed `S[i][j]` with
/// `i` the value dimension and `j` the key dimension. Per head:
///
///     sa[i]   = sum_j -kk[j] * S[i][j]        // how much of the removal
///                                             // direction the state holds
///     S[i][j] = S[i][j] * w[j]                // decay, per key channel
///             + k[j] * v[i]                   // write the new pair in
///             + sa[i] * (kk[j] * a[j])        // erase along kk, scaled by a
///     y[i]    = sum_j S[i][j] * r[j]          // read it back out with r
///
/// So `a` is a per-channel learning rate: 0 keeps what is stored, 1 fully
/// replaces it along the `kk` direction. That "delete then write" step is what
/// distinguishes RWKV-7 from a plain decaying sum.
///
/// This is the scalar form of cuda/wkv7s.cu. The demos in Python spell it as
/// `S * w + S @ (-kk (x) kk*a) + v (x) k`, which is the same thing with the two
/// rank-1 outer products materialised.
fn wkv7(
    state: &mut [f32],
    n_head: usize,
    n: usize,
    r: &[f32],
    w: &[f32],
    k: &[f32],
    v: &[f32],
    kk: &[f32],
    a: &[f32],
) -> Vec<f32> {
    let mut y = vec![0.0f32; n_head * n];
    for head in 0..n_head {
        let b = head * n;
        let s = &mut state[head * n * n..(head + 1) * n * n];

        for i in 0..n {
            let row = &mut s[i * n..(i + 1) * n];

            let mut sa = 0.0f32;
            for j in 0..n {
                sa -= kk[b + j] * row[j];
            }

            let vi = v[b + i];
            let mut acc = 0.0f32;
            for j in 0..n {
                let sv = row[j] * w[b + j] + k[b + j] * vi + sa * (kk[b + j] * a[b + j]);
                row[j] = sv;
                acc += sv * r[b + j];
            }
            y[b + i] = acc;
        }
    }
    y
}

////////////////////////////////////////////////////////////////////////////////
//
// MATH
//
////////////////////////////////////////////////////////////////////////////////

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Four accumulators: breaks the dependency chain on FMA latency and lets the
/// autovectoriser fill a full-width register.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let m = n & !3;
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut i = 0;
    while i < m {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut s = (s0 + s1) + (s2 + s3);
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

fn threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("RWKV_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .or_else(|| std::thread::available_parallelism().ok().map(|v| v.get()))
            .unwrap_or(1)
            .max(1)
    })
}

/// `out[o] = dot(w[o], x)` for `w: [n_out, n_in]`.
///
/// Split across threads by output row. `chunks`/`chunks_mut` hand each thread a
/// disjoint slice, so this needs no unsafe and no shared-state reasoning.
fn matvec(w: &[f32], n_out: usize, n_in: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(w.len(), n_out * n_in);
    let mut out = vec![0.0f32; n_out];

    // Waking threads costs more than the work itself for the small projections.
    const PARALLEL_THRESHOLD: usize = 1 << 17;
    let nt = threads();
    if nt == 1 || n_out * n_in < PARALLEL_THRESHOLD {
        for (o, dst) in out.iter_mut().enumerate() {
            *dst = dot(&w[o * n_in..][..n_in], x);
        }
        return out;
    }

    let rows = n_out.div_ceil(nt);
    std::thread::scope(|scope| {
        for (wc, oc) in w.chunks(rows * n_in).zip(out.chunks_mut(rows)) {
            scope.spawn(move || {
                for (o, dst) in oc.iter_mut().enumerate() {
                    *dst = dot(&wc[o * n_in..][..n_in], x);
                }
            });
        }
    });
    out
}

/// `out = x @ m` for `m: [n_in, n_out]` — the LoRA layout. Always small.
fn matvec_t(m: &[f32], n_in: usize, n_out: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(m.len(), n_in * n_out);
    let mut out = vec![0.0f32; n_out];
    for i in 0..n_in {
        let xi = x[i];
        let row = &m[i * n_out..][..n_out];
        for o in 0..n_out {
            out[o] += xi * row[o];
        }
    }
    out
}

/// LayerNorm over the whole vector, with the biased variance PyTorch uses.
fn layer_norm(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let inv_n = 1.0 / x.len() as f32;
    let mean = x.iter().sum::<f32>() * inv_n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() * inv_n;
    let scale = 1.0 / (var + LN_EPS).sqrt();
    (0..x.len()).map(|i| (x[i] - mean) * scale * w[i] + b[i]).collect()
}

/// GroupNorm with one group per head: normalise each run of `head_size` values
/// independently. Note GN_EPS, which is 64x the usual epsilon.
fn group_norm_heads(x: &mut [f32], n_head: usize, n: usize, w: &[f32], b: &[f32]) {
    let inv_n = 1.0 / n as f32;
    for head in 0..n_head {
        let s = head * n;
        let seg = &mut x[s..s + n];
        let mean = seg.iter().sum::<f32>() * inv_n;
        let var = seg.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() * inv_n;
        let scale = 1.0 / (var + GN_EPS).sqrt();
        for i in 0..n {
            seg[i] = (seg[i] - mean) * scale * w[s + i] + b[s + i];
        }
    }
}

/// L2-normalise each head slice in place. PyTorch clamps the denominator rather
/// than skipping the division, so a zero vector stays zero instead of becoming NaN.
fn l2_normalize_heads(x: &mut [f32], n_head: usize, n: usize) {
    for head in 0..n_head {
        let seg = &mut x[head * n..(head + 1) * n];
        let inv = 1.0 / seg.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for v in seg.iter_mut() {
            *v *= inv;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// SAMPLING
//
////////////////////////////////////////////////////////////////////////////////

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = p.iter().map(|v| *v as f64).sum();
    let inv = (1.0 / sum) as f32;
    for v in p.iter_mut() {
        *v *= inv;
    }
    p
}

/// xoshiro256++, seeded by SplitMix64.
struct Rng([u64; 4]);

impl Rng {
    fn new(seed: u64) -> Rng {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng([next(), next(), next(), next()])
    }
    fn next_f32(&mut self) -> f32 {
        let s = &mut self.0;
        let r = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        (r >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Matches sample_logits() in the Python demos, quirks included: temperature is
/// applied *after* the top-p/top-k cut, and top_p == 0 collapses to greedy.
fn sample(logits: &[f32], temperature: f32, top_p: f32, top_k: usize, rng: &mut Rng) -> u32 {
    let mut probs = softmax(logits);
    let mut order: Vec<u32> = (0..probs.len() as u32).collect();
    order.sort_unstable_by(|a, b| {
        probs[*b as usize]
            .partial_cmp(&probs[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });

    if top_k > 0 && top_k < order.len() {
        for &i in &order[top_k..] {
            probs[i as usize] = 0.0;
        }
    }

    if top_p < 1.0 {
        // The smallest probability we keep: the first point where the running
        // total reaches top_p. With top_p == 0 that is the argmax.
        let mut cum = 0.0f32;
        let mut cutoff = probs[*order.last().unwrap() as usize];
        for &i in &order {
            cum += probs[i as usize];
            if cum >= top_p {
                cutoff = probs[i as usize];
                break;
            }
        }
        for v in probs.iter_mut() {
            if *v < cutoff {
                *v = 0.0;
            }
        }
    }

    if temperature != 1.0 {
        let inv = 1.0 / temperature;
        for v in probs.iter_mut() {
            if *v > 0.0 {
                *v = v.powf(inv);
            }
        }
    }

    let total: f32 = probs.iter().sum();
    if !(total > 0.0) {
        return order[0];
    }
    let mut target = rng.next_f32() * total;
    for (i, p) in probs.iter().enumerate() {
        target -= p;
        if target <= 0.0 {
            return i as u32;
        }
    }
    order[0]
}

////////////////////////////////////////////////////////////////////////////////
//
// TOKENIZER
//
// The RWKV "world" vocabulary: greedy longest-match over raw bytes. Lines look
// like `<id> <python-literal> <byte-length>`, e.g.
//     65   'A'      1
//     129  b'\x80'  1
//     40   "'"      1
// The `b` prefix matters: in a str literal \xNN is the code point U+00NN and is
// UTF-8 encoded (so \x80 is two bytes); in a bytes literal it is the raw byte.
//
////////////////////////////////////////////////////////////////////////////////

struct Tokenizer {
    /// Byte trie. Each node: children sorted by byte, plus the token ending here.
    nodes: Vec<(Vec<(u8, u32)>, Option<u32>)>,
    id_to_bytes: Vec<Vec<u8>>,
}

impl Tokenizer {
    fn load(path: &str) -> Result<Tokenizer, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let mut nodes: Vec<(Vec<(u8, u32)>, Option<u32>)> = vec![(Vec::new(), None)];
        let mut id_to_bytes: Vec<Vec<u8>> = Vec::new();

        for (ln, line) in text.lines().enumerate() {
            let (Some(first), Some(last)) = (line.find(' '), line.rfind(' ')) else { continue };
            if last <= first {
                continue;
            }
            let Ok(id) = line[..first].parse::<u32>() else { continue };
            let want: usize = line[last + 1..].trim().parse().unwrap_or(usize::MAX);
            let bytes = parse_py_literal(&line[first + 1..last])
                .ok_or_else(|| format!("{path}:{}: cannot parse token literal", ln + 1))?;
            // The trailing length column is a free check on the parser.
            if bytes.len() != want {
                return Err(format!(
                    "{path}:{}: decoded {} bytes but the file says {want}",
                    ln + 1,
                    bytes.len()
                ));
            }

            if id as usize >= id_to_bytes.len() {
                id_to_bytes.resize(id as usize + 1, Vec::new());
            }
            id_to_bytes[id as usize] = bytes.clone();

            let mut cur = 0u32;
            for b in bytes {
                cur = match nodes[cur as usize].0.iter().find(|(c, _)| *c == b) {
                    Some((_, next)) => *next,
                    None => {
                        let next = nodes.len() as u32;
                        nodes.push((Vec::new(), None));
                        nodes[cur as usize].0.push((b, next));
                        next
                    }
                };
            }
            nodes[cur as usize].1 = Some(id);
        }
        for nd in nodes.iter_mut() {
            nd.0.sort_unstable_by_key(|(b, _)| *b);
        }
        Ok(Tokenizer { nodes, id_to_bytes })
    }

    fn encode(&self, s: &str) -> Vec<u32> {
        let src = s.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < src.len() {
            // Walk as deep as the trie allows, remembering the last real token.
            let (mut cur, mut best, mut j) = (0u32, None, i);
            while j < src.len() {
                let Ok(k) = self.nodes[cur as usize].0.binary_search_by_key(&src[j], |(b, _)| *b)
                else {
                    break;
                };
                cur = self.nodes[cur as usize].0[k].1;
                j += 1;
                if let Some(t) = self.nodes[cur as usize].1 {
                    best = Some((t, j - i));
                }
            }
            match best {
                Some((t, len)) => {
                    out.push(t);
                    i += len;
                }
                None => i += 1, // unreachable for a well-formed vocab
            }
        }
        out
    }

    fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for t in tokens {
            if let Some(b) = self.id_to_bytes.get(*t as usize) {
                out.extend_from_slice(b);
            }
        }
        out
    }
}

/// Decode a Python `'...'`, `"..."` or `b'...'` literal into the bytes it denotes.
fn parse_py_literal(src: &str) -> Option<Vec<u8>> {
    let s = src.trim();
    let (is_bytes, s) = match s.strip_prefix('b') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    let quote = s.chars().next()?;
    if (quote != '\'' && quote != '"') || !s.ends_with(quote) || s.len() < 2 {
        return None;
    }
    let body = &s[1..s.len() - 1];

    let mut out: Vec<u8> = Vec::with_capacity(body.len());
    let push = |out: &mut Vec<u8>, cp: u32| {
        if is_bytes {
            out.push(cp as u8);
        } else {
            let c = char::from_u32(cp).unwrap_or('\u{FFFD}');
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        }
    };

    let mut it = body.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            push(&mut out, c as u32);
            continue;
        }
        match it.next()? {
            'n' => out.push(b'\n'),
            'r' => out.push(b'\r'),
            't' => out.push(b'\t'),
            '\\' => out.push(b'\\'),
            '\'' => out.push(b'\''),
            '"' => out.push(b'"'),
            '0' => out.push(0),
            e @ ('x' | 'u' | 'U') => {
                let n = match e { 'x' => 2, 'u' => 4, _ => 8 };
                let mut cp = 0u32;
                for _ in 0..n {
                    cp = cp * 16 + it.next()?.to_digit(16)?;
                }
                push(&mut out, cp);
            }
            other => {
                out.push(b'\\');
                push(&mut out, other as u32);
            }
        }
    }
    Some(out)
}

////////////////////////////////////////////////////////////////////////////////
//
// CHECKPOINT READER
//
// A .pth is a ZIP archive holding a pickled index (`data.pkl`) plus one raw
// record per tensor. Nothing here is about RWKV; skip it on a first read.
//
// Two things make this tractable without a zip or pickle library:
//   * torch.save stores entries uncompressed, so no DEFLATE decoder is needed
//   * only a small, fixed slice of the pickle opcode set ever appears
//
////////////////////////////////////////////////////////////////////////////////

struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

type Tensors = HashMap<String, Tensor>;

fn u16le(b: &[u8], o: usize) -> u16 { u16::from_le_bytes([b[o], b[o + 1]]) }
fn u32le(b: &[u8], o: usize) -> u32 { u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) }
fn u64le(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}

/// Half-precision to f32, subnormals included.
fn f16_to_f32(h: u16) -> f32 {
    let (sign, exp, frac) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    f32::from_bits(match exp {
        0 if frac == 0 => sign << 31,
        0 => {
            // Shift the mantissa up until the implicit 1 appears at bit 10, then
            // drop it. After k shifts the value is mantissa * 2^(-14-k).
            let (mut e, mut f) = (0i32, frac);
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | (((127 - 14 + e) as u32) << 23) | ((f & 0x3ff) << 13)
        }
        0x1f => (sign << 31) | (0xff << 23) | (frac << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    })
}

/// bfloat16 is just the top 16 bits of an f32.
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Pickle values, kept symbolic: `Reduce` records a call without performing it,
/// so nothing here can be talked into executing a checkpoint.
#[derive(Clone)]
enum Val {
    None,
    Int(i64),
    Str(String),
    Seq(Vec<Val>),
    Dict(std::rc::Rc<std::cell::RefCell<Vec<(Val, Val)>>>),
    Global(String),
    PersId(Box<Val>),
    Reduce(String, Vec<Val>),
    Mark,
}

impl Val {
    fn int(&self) -> Option<usize> {
        match self {
            Val::Int(v) => usize::try_from(*v).ok(),
            _ => None,
        }
    }
    fn str(&self) -> Option<&str> {
        match self {
            Val::Str(s) => Some(s),
            _ => None,
        }
    }
    fn seq(&self) -> Vec<Val> {
        match self {
            Val::Seq(v) => v.clone(),
            _ => Vec::new(),
        }
    }
}

/// Interpret the subset of protocol-2 pickle that torch.save emits.
fn unpickle(b: &[u8]) -> Result<Val, String> {
    let (mut p, mut stack, mut marks) = (0usize, Vec::<Val>::new(), Vec::<usize>::new());
    let mut memo: HashMap<u32, Val> = HashMap::new();

    macro_rules! take {
        ($n:expr) => {{
            let n = $n;
            if p + n > b.len() {
                return Err("truncated pickle".into());
            }
            p += n;
            &b[p - n..p]
        }};
    }
    macro_rules! pop {
        () => { stack.pop().ok_or("pickle stack underflow")? };
    }
    /// Everything between the last MARK and the top of the stack.
    macro_rules! since_mark {
        () => {{
            let m = marks.pop().ok_or("pickle mark underflow")?;
            let items: Vec<Val> = stack.drain(m + 1..).collect();
            stack.pop();
            items
        }};
    }

    loop {
        if p >= b.len() {
            return Err("pickle ended without STOP".into());
        }
        let op = b[p];
        p += 1;
        match op {
            0x80 => { let _ = take!(1); }                            // PROTO
            0x95 => { let _ = take!(8); }                            // FRAME
            b'.' => return Ok(pop!()),                               // STOP
            b'N' => stack.push(Val::None),
            0x88 => stack.push(Val::Int(1)),                         // NEWTRUE
            0x89 => stack.push(Val::Int(0)),                         // NEWFALSE
            b'K' => { let v = take!(1)[0] as i64; stack.push(Val::Int(v)); }
            b'M' => { let v = u16le(take!(2), 0) as i64; stack.push(Val::Int(v)); }
            b'J' => { let v = u32le(take!(4), 0) as i32 as i64; stack.push(Val::Int(v)); }
            0x8a | 0x8b => {                                         // LONG1 / LONG4
                let n = if op == 0x8a { take!(1)[0] as usize } else { u32le(take!(4), 0) as usize };
                let raw = take!(n);
                let mut acc: i128 = 0;
                for (i, byte) in raw.iter().enumerate() {
                    acc |= (*byte as i128) << (8 * i);
                }
                if n > 0 && raw[n - 1] & 0x80 != 0 {
                    acc -= 1i128 << (8 * n);
                }
                stack.push(Val::Int(acc as i64));
            }
            b'G' => { let _ = take!(8); stack.push(Val::None); }     // BINFLOAT (unused)
            b'X' | 0x8c | 0x8d | b'C' | b'B' | 0x8e => {             // strings / bytes
                let n = match op {
                    0x8c | b'C' => take!(1)[0] as usize,
                    b'X' | b'B' => u32le(take!(4), 0) as usize,
                    _ => u64le(take!(8), 0) as usize,
                };
                stack.push(Val::Str(String::from_utf8_lossy(take!(n)).into_owned()));
            }
            b'(' => { marks.push(stack.len()); stack.push(Val::Mark); }
            b')' => stack.push(Val::Seq(Vec::new())),
            b'}' => stack.push(Val::Dict(Default::default())),
            b']' | 0x8f => stack.push(Val::Seq(Vec::new())),
            0x85 | 0x86 | 0x87 => {                                  // TUPLE1 / 2 / 3
                let at = stack.len() - (op - 0x84) as usize;
                let items: Vec<Val> = stack.drain(at..).collect();
                stack.push(Val::Seq(items));
            }
            b't' | b'l' => { let items = since_mark!(); stack.push(Val::Seq(items)); }
            b'e' | b'u' => {                                         // APPENDS / SETITEMS
                let items = since_mark!();
                if let Some(Val::Dict(d)) = stack.last() {
                    for kv in items.chunks_exact(2) {
                        d.borrow_mut().push((kv[0].clone(), kv[1].clone()));
                    }
                }
            }
            b's' => {                                                // SETITEM
                let (v, k) = (pop!(), pop!());
                if let Some(Val::Dict(d)) = stack.last() {
                    d.borrow_mut().push((k, v));
                }
            }
            b'q' | b'r' | 0x94 => {                                  // BINPUT / LONG_BINPUT / MEMOIZE
                let i = match op {
                    b'q' => take!(1)[0] as u32,
                    b'r' => u32le(take!(4), 0),
                    _ => memo.len() as u32,
                };
                if let Some(v) = stack.last() {
                    memo.insert(i, v.clone());
                }
            }
            b'h' | b'j' => {                                         // BINGET / LONG_BINGET
                let i = if op == b'h' { take!(1)[0] as u32 } else { u32le(take!(4), 0) };
                stack.push(memo.get(&i).ok_or("pickle memo miss")?.clone());
            }
            b'c' => {                                                // GLOBAL
                let mut parts = Vec::new();
                for _ in 0..2 {
                    let start = p;
                    while p < b.len() && b[p] != b'\n' {
                        p += 1;
                    }
                    parts.push(String::from_utf8_lossy(&b[start..p]).into_owned());
                    p += 1;
                }
                stack.push(Val::Global(parts.remove(1)));
            }
            0x93 => {                                                // STACK_GLOBAL
                let name = pop!();
                let _module = pop!();
                stack.push(Val::Global(name.str().unwrap_or_default().to_string()));
            }
            b'R' | 0x81 => {                                         // REDUCE / NEWOBJ
                let args = pop!();
                let Val::Global(name) = pop!() else {
                    return Err("REDUCE on a non-global".into());
                };
                // OrderedDict/dict are materialised so the SETITEMS that follow
                // have something to write into; everything else stays symbolic.
                if name == "OrderedDict" || name == "dict" {
                    stack.push(Val::Dict(Default::default()));
                } else {
                    stack.push(Val::Reduce(name, args.seq()));
                }
            }
            b'b' => { let _state = pop!(); }                         // BUILD
            b'Q' => { let a = pop!(); stack.push(Val::PersId(Box::new(a))); }
            b'0' => { stack.pop(); }                                 // POP
            _ => return Err(format!("unsupported pickle opcode {op:#04x} at byte {}", p - 1)),
        }
    }
}

/// Read a torch.save archive into `name -> f32 tensor`.
///
/// Tensors are read one at a time and up-converted as they go, so peak memory is
/// the f32 model plus a single tensor rather than two full copies.
fn load_checkpoint(path: &str) -> Result<Tensors, String> {
    let e = |m: String| -> String { format!("{path}: {m}") };
    let mut f = File::open(path).map_err(|x| e(x.to_string()))?;
    let flen = f.metadata().map_err(|x| e(x.to_string()))?.len();

    // --- locate the central directory ---
    let tail_len = flen.min(66 * 1024) as usize;
    let mut tail = vec![0u8; tail_len];
    f.seek(SeekFrom::End(-(tail_len as i64))).map_err(|x| e(x.to_string()))?;
    f.read_exact(&mut tail).map_err(|x| e(x.to_string()))?;
    let tail_start = flen - tail_len as u64;

    let eocd = (0..tail_len.saturating_sub(21))
        .rev()
        .find(|&i| u32le(&tail, i) == 0x0605_4b50)
        .ok_or_else(|| e("not a zip archive; is this really a torch.save .pth?".into()))?;

    let mut cd_off = u32le(&tail, eocd + 16) as u64;
    let mut n_entries = u16le(&tail, eocd + 10) as u64;
    if cd_off == 0xFFFF_FFFF || n_entries == 0xFFFF {
        // Zip64, for checkpoints past 4 GiB.
        let loc = (0..eocd)
            .rev()
            .find(|&i| u32le(&tail, i) == 0x0706_4b50)
            .ok_or_else(|| e("zip64 locator not found".into()))?;
        let mut hdr = [0u8; 56];
        f.seek(SeekFrom::Start(u64le(&tail, loc + 8))).map_err(|x| e(x.to_string()))?;
        f.read_exact(&mut hdr).map_err(|x| e(x.to_string()))?;
        n_entries = u64le(&hdr, 32);
        cd_off = u64le(&hdr, 48);
    }

    let mut cd = vec![0u8; (tail_start + eocd as u64).saturating_sub(cd_off) as usize];
    f.seek(SeekFrom::Start(cd_off)).map_err(|x| e(x.to_string()))?;
    f.read_exact(&mut cd).map_err(|x| e(x.to_string()))?;

    // --- index the entries ---
    let mut entries: HashMap<String, u64> = HashMap::new();
    let mut p = 0usize;
    for _ in 0..n_entries {
        if p + 46 > cd.len() || u32le(&cd, p) != 0x0201_4b50 {
            break;
        }
        if u16le(&cd, p + 10) != 0 {
            return Err(e("entries are compressed; only stored (0) is supported".into()));
        }
        let (nl, el, cl) = (u16le(&cd, p + 28) as usize, u16le(&cd, p + 30) as usize, u16le(&cd, p + 32) as usize);
        let name = String::from_utf8_lossy(&cd[p + 46..p + 46 + nl]).into_owned();

        let mut local = u32le(&cd, p + 42) as u64;
        if local == 0xFFFF_FFFF {
            // Pull the real offset out of the zip64 extra field.
            let extra = &cd[p + 46 + nl..p + 46 + nl + el];
            let mut q = 0usize;
            while q + 4 <= extra.len() {
                let (tag, sz) = (u16le(extra, q), u16le(extra, q + 2) as usize);
                if tag == 0x0001 {
                    let mut r = 0usize;
                    // Fields appear only if their fixed-header value was sentinelled.
                    if u32le(&cd, p + 24) == 0xFFFF_FFFF { r += 8; }
                    if u32le(&cd, p + 20) == 0xFFFF_FFFF { r += 8; }
                    if r + 8 <= sz {
                        local = u64le(&extra[q + 4..], r);
                    }
                }
                q += 4 + sz;
            }
        }
        entries.insert(name, local);
        p += 46 + nl + el + cl;
    }

    // Resolve an entry's payload offset via its local header, whose extra field
    // length can differ from the central directory's.
    let data_offset = |f: &mut File, name: &str| -> Result<u64, String> {
        let off = *entries.get(name).ok_or_else(|| e(format!("missing zip entry '{name}'")))?;
        let mut hdr = [0u8; 30];
        f.seek(SeekFrom::Start(off)).map_err(|x| e(x.to_string()))?;
        f.read_exact(&mut hdr).map_err(|x| e(x.to_string()))?;
        Ok(off + 30 + u16le(&hdr, 26) as u64 + u16le(&hdr, 28) as u64)
    };

    // --- the pickled index ---
    let pkl_name = entries
        .keys()
        .find(|n| n.ends_with("data.pkl"))
        .cloned()
        .ok_or_else(|| e("no data.pkl; legacy (pre-1.6) .pth files are not supported".into()))?;
    let root = pkl_name.trim_end_matches("data.pkl").to_string();

    let pkl_at = data_offset(&mut f, &pkl_name)?;
    let mut pkl = vec![0u8; 1 << 22];
    f.seek(SeekFrom::Start(pkl_at)).map_err(|x| e(x.to_string()))?;
    let n = f.read(&mut pkl).map_err(|x| e(x.to_string()))?;
    pkl.truncate(n);
    let root_val = unpickle(&pkl).map_err(|m| e(m))?;

    let Val::Dict(items) = root_val else {
        return Err(e("checkpoint root is not a state dict".into()));
    };

    // --- the tensors ---
    let mut out: Tensors = HashMap::new();
    for (k, v) in items.borrow().iter() {
        let Some(name) = k.str() else { continue };
        let Val::Reduce(func, args) = v else { continue };
        if !func.starts_with("_rebuild_tensor") || args.len() < 4 {
            continue;
        }

        // args = (storage, storage_offset, size, stride, requires_grad, hooks)
        // storage = ('storage', torch.<X>Storage, key, location, numel)
        let Val::PersId(pid) = &args[0] else {
            return Err(e(format!("{name}: no storage record")));
        };
        let sid = pid.seq();
        let Some(Val::Global(cls)) = sid.get(1) else {
            return Err(e(format!("{name}: malformed storage record")));
        };
        let (esz, kind) = match cls.as_str() {
            "FloatStorage" => (4, 'f'),
            "HalfStorage" => (2, 'h'),
            "BFloat16Storage" => (2, 'b'),
            "DoubleStorage" => (8, 'd'),
            other => return Err(e(format!("{name}: unsupported storage type {other}"))),
        };
        let key = match sid.get(2) {
            Some(Val::Str(s)) => s.clone(),
            Some(Val::Int(i)) => i.to_string(),
            _ => return Err(e(format!("{name}: bad storage key"))),
        };

        let shape: Vec<usize> = args[2].seq().iter().filter_map(|d| d.int()).collect();
        let numel: usize = shape.iter().product();
        let start = (args[1].int().unwrap_or(0) * esz) as u64;

        let at = data_offset(&mut f, &format!("{root}data/{key}"))?;
        let mut raw = vec![0u8; numel * esz];
        f.seek(SeekFrom::Start(at + start)).map_err(|x| e(x.to_string()))?;
        f.read_exact(&mut raw).map_err(|x| e(x.to_string()))?;

        let data: Vec<f32> = match kind {
            'f' => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            'h' => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            'b' => raw.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            _ => raw.chunks_exact(8).map(|c| {
                let mut a = [0u8; 8];
                a.copy_from_slice(c);
                f64::from_le_bytes(a) as f32
            }).collect(),
        };
        out.insert(name.to_string(), Tensor { shape, data });
    }

    if out.is_empty() {
        return Err(e("no tensors found".into()));
    }
    Ok(out)
}

////////////////////////////////////////////////////////////////////////////////
//
// BUILDING THE MODEL FROM A CHECKPOINT
//
////////////////////////////////////////////////////////////////////////////////

impl Model {
    fn build(mut z: Tensors) -> Result<Model, String> {
        // Every dimension is discovered from the weights, so one binary handles
        // 0.1B through 7B without a config file.
        let emb = z.get("emb.weight").ok_or("checkpoint has no emb.weight")?;
        if emb.shape.len() != 2 {
            return Err(format!("emb.weight should be 2-D, found {:?}", emb.shape));
        }
        let (vocab_size, n_embd) = (emb.shape[0], emb.shape[1]);

        let r_k = z.get("blocks.0.att.r_k").ok_or(
            "checkpoint has no blocks.0.att.r_k — is this RWKV-7? \
             (the v7a/v7b/v8 demos are different architectures)",
        )?;
        let sq: Vec<usize> = r_k.shape.iter().copied().filter(|d| *d != 1).collect();
        if sq.len() != 2 || sq[0] * sq[1] != n_embd {
            return Err(format!("blocks.0.att.r_k has shape {:?}, expected [n_head, head_size]", r_k.shape));
        }
        let (n_head, head_size) = (sq[0], sq[1]);
        let n_layer = (0..).take_while(|i| z.contains_key(&format!("blocks.{i}.att.r_k"))).count();
        let dim_ffn = z.get("blocks.0.ffn.key.weight").ok_or("no blocks.0.ffn.key.weight")?.data.len() / n_embd;

        let cfg = Config { n_layer, n_embd, n_head, head_size, dim_ffn, vocab_size };

        // Pull a tensor out by name, checking its element count.
        let get = |z: &mut Tensors, name: &str, want: usize| -> Result<Vec<f32>, String> {
            let t = z.remove(name).ok_or(format!("checkpoint is missing '{name}'"))?;
            if want != 0 && t.data.len() != want {
                return Err(format!("{name}: expected {want} elements, found {}", t.data.len()));
            }
            Ok(t.data)
        };

        // Fold ln0 into the embedding table so the per-token path skips it.
        let ln0_w = get(&mut z, "blocks.0.ln0.weight", n_embd)?;
        let ln0_b = get(&mut z, "blocks.0.ln0.bias", n_embd)?;
        let mut emb = get(&mut z, "emb.weight", vocab_size * n_embd)?;
        for v in 0..vocab_size {
            let row = layer_norm(&emb[v * n_embd..][..n_embd], &ln0_w, &ln0_b);
            emb[v * n_embd..][..n_embd].copy_from_slice(&row);
        }

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let (b, a, f) = (format!("blocks.{i}."), format!("blocks.{i}.att."), format!("blocks.{i}.ffn."));

            // LoRA ranks vary by model size, so read each from its first factor
            // and hold the second one to it.
            let lora = |z: &mut Tensors, lo: &str, hi: &str| -> Result<(Vec<f32>, Vec<f32>, usize), String> {
                let first = get(z, &format!("{a}{lo}"), 0)?;
                if first.len() % n_embd != 0 {
                    return Err(format!("{a}{lo}: {} is not a multiple of n_embd", first.len()));
                }
                let rank = first.len() / n_embd;
                let second = get(z, &format!("{a}{hi}"), rank * n_embd)?;
                Ok((first, second, rank))
            };
            let (w1, w2, w_rank) = lora(&mut z, "w1", "w2")?;
            let (a1, a2, a_rank) = lora(&mut z, "a1", "a2")?;
            let (g1, g2, g_rank) = lora(&mut z, "g1", "g2")?;
            // Layer 0 has no v-mix LoRA: it defines v_first rather than mixing.
            let (v0, v1, v2, v_rank) = if i == 0 {
                (None, Vec::new(), Vec::new(), 0)
            } else {
                let (v1, v2, r) = lora(&mut z, "v1", "v2")?;
                (Some(get(&mut z, &format!("{a}v0"), n_embd)?), v1, v2, r)
            };

            layers.push(Layer {
                ln1_w: get(&mut z, &format!("{b}ln1.weight"), n_embd)?,
                ln1_b: get(&mut z, &format!("{b}ln1.bias"), n_embd)?,
                ln2_w: get(&mut z, &format!("{b}ln2.weight"), n_embd)?,
                ln2_b: get(&mut z, &format!("{b}ln2.bias"), n_embd)?,
                x_r: get(&mut z, &format!("{a}x_r"), n_embd)?,
                x_w: get(&mut z, &format!("{a}x_w"), n_embd)?,
                x_k: get(&mut z, &format!("{a}x_k"), n_embd)?,
                x_v: get(&mut z, &format!("{a}x_v"), n_embd)?,
                x_a: get(&mut z, &format!("{a}x_a"), n_embd)?,
                x_g: get(&mut z, &format!("{a}x_g"), n_embd)?,
                w0: get(&mut z, &format!("{a}w0"), n_embd)?, w1, w2, w_rank,
                a0: get(&mut z, &format!("{a}a0"), n_embd)?, a1, a2, a_rank,
                g1, g2, g_rank,
                v0, v1, v2, v_rank,
                k_k: get(&mut z, &format!("{a}k_k"), n_embd)?,
                k_a: get(&mut z, &format!("{a}k_a"), n_embd)?,
                r_k: get(&mut z, &format!("{a}r_k"), n_embd)?,
                r_w: get(&mut z, &format!("{a}receptance.weight"), n_embd * n_embd)?,
                k_w: get(&mut z, &format!("{a}key.weight"), n_embd * n_embd)?,
                v_w: get(&mut z, &format!("{a}value.weight"), n_embd * n_embd)?,
                o_w: get(&mut z, &format!("{a}output.weight"), n_embd * n_embd)?,
                ln_x_w: get(&mut z, &format!("{a}ln_x.weight"), n_embd)?,
                ln_x_b: get(&mut z, &format!("{a}ln_x.bias"), n_embd)?,
                ffn_x_k: get(&mut z, &format!("{f}x_k"), n_embd)?,
                ffn_k: get(&mut z, &format!("{f}key.weight"), dim_ffn * n_embd)?,
                ffn_v: get(&mut z, &format!("{f}value.weight"), n_embd * dim_ffn)?,
            });
        }

        Ok(Model {
            ln_out_w: get(&mut z, "ln_out.weight", n_embd)?,
            ln_out_b: get(&mut z, "ln_out.bias", n_embd)?,
            head: get(&mut z, "head.weight", vocab_size * n_embd)?,
            cfg,
            emb,
            layers,
        })
    }
}

////////////////////////////////////////////////////////////////////////////////
//
// MAIN
//
////////////////////////////////////////////////////////////////////////////////

fn run() -> Result<(), String> {
    let (mut model_path, mut vocab_path) = (MODEL.to_string(), VOCAB.to_string());
    let (mut prompt, mut length) = (PROMPT.to_string(), LENGTH);
    let (mut temperature, mut top_p, mut top_k) = (TEMPERATURE, TOP_P, 0usize);
    let mut seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x5eed);

    /// Consume the value that follows a flag.
    fn next(argv: &[String], i: &mut usize, k: &str) -> Result<String, String> {
        *i += 1;
        argv.get(*i).cloned().ok_or(format!("{k} needs a value"))
    }

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => {
                println!(
                    "rwkv7_demo — RWKV-7 \"Goose\" x070 in one file\n\n\
                       --model <FILE>        .pth checkpoint\n\
                       --vocab <FILE>        tokenizer vocab [{VOCAB}]\n\
                       --prompt <TEXT>       prompt\n\
                       --length <N>          tokens to generate [{LENGTH}]\n\
                       --temperature <F>     [{TEMPERATURE}]\n\
                       --top-p <F>           0 means greedy [{TOP_P}]\n\
                       --top-k <N>           0 disables [0]\n\
                       --seed <N>            RNG seed\n\n\
                     Set RWKV_THREADS to control worker threads."
                );
                return Ok(());
            }
            "--model" | "-m" => model_path = next(&argv, &mut i, "--model")?,
            "--vocab" => vocab_path = next(&argv, &mut i, "--vocab")?,
            "--prompt" | "-p" => prompt = next(&argv, &mut i, "--prompt")?,
            "--length" | "-n" => length = next(&argv, &mut i, "--length")?.parse().map_err(|_| "bad --length")?,
            "--temperature" | "-t" => temperature = next(&argv, &mut i, "-t")?.parse().map_err(|_| "bad --temperature")?,
            "--top-p" => top_p = next(&argv, &mut i, "--top-p")?.parse().map_err(|_| "bad --top-p")?,
            "--top-k" => top_k = next(&argv, &mut i, "--top-k")?.parse().map_err(|_| "bad --top-k")?,
            "--seed" => seed = next(&argv, &mut i, "--seed")?.parse().map_err(|_| "bad --seed")?,
            other => return Err(format!("unknown argument '{other}' (try --help)")),
        }
        i += 1;
    }

    eprintln!("loading tokenizer {vocab_path}");
    let tok = Tokenizer::load(&vocab_path)?;

    eprintln!("loading model {model_path}");
    let t0 = std::time::Instant::now();
    let weights = load_checkpoint(&model_path)?;
    let params: usize = weights.values().map(|t| t.data.len()).sum();
    let model = Model::build(weights)?;
    let c = &model.cfg;
    eprintln!(
        "  n_layer={} n_embd={} n_head={} head_size={} dim_ffn={} vocab={}\n  \
         {:.1}M params, {:.1}s, {} thread(s)",
        c.n_layer, c.n_embd, c.n_head, c.head_size, c.dim_ffn, c.vocab_size,
        params as f64 / 1e6, t0.elapsed().as_secs_f64(), threads()
    );

    let tokens = tok.encode(&prompt);
    if tokens.is_empty() {
        return Err("prompt encoded to zero tokens".into());
    }
    if let Some(bad) = tokens.iter().find(|t| **t as usize >= c.vocab_size) {
        return Err(format!(
            "token id {bad} is outside the model's vocab ({}); \
             the tokenizer and the checkpoint do not match",
            c.vocab_size
        ));
    }

    // --- prefill ---
    let mut state = State::new(&model.cfg);
    let mut logits = Vec::new();
    let t0 = std::time::Instant::now();
    for t in &tokens {
        logits = model.forward(*t, &mut state);
    }
    eprintln!(
        "\nprefill: {} tokens in {:.2}s\n",
        tokens.len(),
        t0.elapsed().as_secs_f64()
    );

    // Show what the model thinks comes next, like the Python demos do.
    let probs = softmax(&logits);
    let mut order: Vec<u32> = (0..probs.len() as u32).collect();
    order.sort_unstable_by(|a, b| probs[*b as usize].total_cmp(&probs[*a as usize]));
    eprintln!("most likely continuations:");
    for &id in order.iter().take(10) {
        let s = String::from_utf8_lossy(&tok.decode_bytes(&[id])).into_owned();
        eprintln!("  {:>6.2}%  {s:?}", probs[id as usize] * 100.0);
    }
    eprintln!();

    // --- generate ---
    print!("{prompt}");
    let _ = std::io::stdout().flush();

    let mut rng = Rng::new(seed);
    // A character can straddle two tokens, so hold bytes back until they decode.
    let mut pending: Vec<u32> = Vec::new();
    let t0 = std::time::Instant::now();
    let mut n = 0usize;

    for _ in 0..length {
        let token = sample(&logits, temperature, top_p, top_k, &mut rng);
        if token == 0 {
            break; // endoftext
        }
        pending.push(token);
        if let Ok(s) = String::from_utf8(tok.decode_bytes(&pending)) {
            print!("{s}");
            let _ = std::io::stdout().flush();
            pending.clear();
        }
        logits = model.forward(token, &mut state);
        n += 1;
    }
    if !pending.is_empty() {
        print!("{}", String::from_utf8_lossy(&tok.decode_bytes(&pending)));
    }
    println!();

    eprintln!(
        "\n[{n} tokens in {:.2}s = {:.1} tok/s]",
        t0.elapsed().as_secs_f64(),
        n as f64 / t0.elapsed().as_secs_f64().max(1e-9)
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
