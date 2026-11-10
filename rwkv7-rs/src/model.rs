//! RWKV-7 "Goose" x070 — a direct port of `rwkv_v7_demo_fast.py`.
//!
//! Both paths from the Python demo are here:
//!   * `forward_one`  — the RNN step (one token, O(1) memory, used for decoding)
//!   * `forward_seq`  — the batched prefill (matmuls over T tokens, sequential WKV scan)
//! They are numerically the same model; `forward_seq` only reassociates the
//! projections so the weight matrices are fetched once per block instead of per token.

use std::io;

use crate::loader::{Tensor, Tensors};
use crate::ops::*;

/// exp(-0.5); the decay is `exp(-0.606531 * sigmoid(w0 + w))`.
const DECAY_SCALE: f32 = 0.606_531;
const LN_EPS: f32 = 1e-5;
/// RWKV-7's GroupNorm uses 64e-5, not the usual 1e-5. Getting this wrong degrades
/// output quality silently instead of failing.
const GN_EPS: f32 = 64e-5;

pub struct Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub vocab_size: usize,
    pub head_size: usize,
    pub n_head: usize,
    pub dim_ffn: usize,
}

/// A `[rows, cols]` row-major matrix.
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

struct Layer {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,

    // token-shift lerp coefficients
    x_r: Vec<f32>,
    x_w: Vec<f32>,
    x_k: Vec<f32>,
    x_v: Vec<f32>,
    x_a: Vec<f32>,
    x_g: Vec<f32>,

    // low-rank projections, stored [in, out]
    w0: Vec<f32>,
    w1: Mat,
    w2: Mat,
    a0: Vec<f32>,
    a1: Mat,
    a2: Mat,
    /// None on layer 0, which defines `v_first` instead of mixing towards it.
    v0: Option<Vec<f32>>,
    v1: Option<Mat>,
    v2: Option<Mat>,
    g1: Mat,
    g2: Mat,

    k_k: Vec<f32>,
    k_a: Vec<f32>,
    r_k: Vec<f32>,

    // nn.Linear weights, stored [out, in]
    r_w: Mat,
    k_w: Mat,
    v_w: Mat,
    o_w: Mat,

    ln_x_w: Vec<f32>,
    ln_x_b: Vec<f32>,

    ffn_x_k: Vec<f32>,
    ffn_k: Mat,
    ffn_v: Mat,
}

pub struct Model {
    pub cfg: Config,
    /// ln0 is folded into the embedding table at load time, as the demo does.
    emb: Mat,
    layers: Vec<Layer>,
    ln_out_w: Vec<f32>,
    ln_out_b: Vec<f32>,
    head: Mat,
}

/// Per-layer recurrent state. `att_kv` is the only part that must stay f32:
/// running it in lower precision diverges within a few hundred tokens.
#[derive(Clone)]
pub struct State {
    pub att_x: Vec<Vec<f32>>,
    pub att_kv: Vec<Vec<f32>>,
    pub ffn_x: Vec<Vec<f32>>,
}

impl State {
    pub fn new(cfg: &Config) -> Self {
        State {
            att_x: vec![vec![0.0; cfg.n_embd]; cfg.n_layer],
            att_kv: vec![vec![0.0; cfg.n_head * cfg.head_size * cfg.head_size]; cfg.n_layer],
            ffn_x: vec![vec![0.0; cfg.n_embd]; cfg.n_layer],
        }
    }
}

fn err<T>(msg: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, msg.into()))
}

fn take(t: &mut Tensors, name: &str) -> io::Result<Tensor> {
    match t.remove(name) {
        Some(v) => Ok(v),
        None => err(format!("checkpoint is missing '{name}'")),
    }
}

/// Fetch a 1-D parameter, dropping unit axes (`[1,1,C]` -> `[C]`).
fn vec1(t: &mut Tensors, name: &str, expect: usize) -> io::Result<Vec<f32>> {
    let x = take(t, name)?;
    if x.data.len() != expect {
        return err(format!(
            "{name}: expected {expect} elements, found {} (shape {:?})",
            x.data.len(),
            x.shape
        ));
    }
    Ok(x.data)
}

/// Fetch a 2-D parameter and assert its shape, so a mismatched checkpoint fails
/// at load with a useful message rather than producing garbage.
fn mat2(t: &mut Tensors, name: &str, rows: usize, cols: usize) -> io::Result<Mat> {
    let x = take(t, name)?;
    let s = x.squeezed_shape();
    if x.data.len() != rows * cols {
        return err(format!(
            "{name}: expected {rows}x{cols}, found {} elements (shape {:?})",
            x.data.len(),
            x.shape
        ));
    }
    if s.len() == 2 && (s[0] != rows || s[1] != cols) {
        return err(format!(
            "{name}: expected shape [{rows}, {cols}], found {s:?}"
        ));
    }
    Ok(Mat {
        rows,
        cols,
        data: x.data,
    })
}

/// Load a LoRA factor pair `[n_embd, rank]` / `[rank, n_embd]`. The rank varies by
/// model size, so it is derived from the first factor and then checked against the
/// second rather than hard-coded.
fn lora_pair(
    t: &mut Tensors,
    prefix: &str,
    lo: &str,
    hi: &str,
    n_embd: usize,
) -> io::Result<(Mat, Mat)> {
    let first = take(t, &format!("{prefix}{lo}"))?;
    if first.data.len() % n_embd != 0 {
        return err(format!(
            "{prefix}{lo}: {} elements is not a multiple of n_embd ({n_embd})",
            first.data.len()
        ));
    }
    let rank = first.data.len() / n_embd;
    if rank == 0 || rank > MAX_LORA {
        return err(format!(
            "{prefix}{lo}: implausible LoRA rank {rank} (max {MAX_LORA})"
        ));
    }
    let second = take(t, &format!("{prefix}{hi}"))?;
    if second.data.len() != rank * n_embd {
        return err(format!(
            "{prefix}{hi}: expected {rank}x{n_embd} to match {lo}, found {} elements",
            second.data.len()
        ));
    }
    Ok((
        Mat {
            rows: n_embd,
            cols: rank,
            data: first.data,
        },
        Mat {
            rows: rank,
            cols: n_embd,
            data: second.data,
        },
    ))
}

impl Model {
    pub fn load(mut z: Tensors) -> io::Result<Model> {
        // ---- shape discovery -------------------------------------------------
        let emb = z.get("emb.weight").ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "checkpoint is missing 'emb.weight'",
            )
        })?;
        if emb.shape.len() != 2 {
            return err(format!("emb.weight should be 2-D, found {:?}", emb.shape));
        }
        let vocab_size = emb.shape[0];
        let n_embd = emb.shape[1];

        let r_k = z.get("blocks.0.att.r_k").ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "checkpoint is missing 'blocks.0.att.r_k' — is this an RWKV-7 model? \
                 (v7a/v7b/v8 use a different architecture)",
            )
        })?;
        let rs = r_k.squeezed_shape();
        if rs.len() != 2 {
            return err(format!(
                "blocks.0.att.r_k should be [n_head, head_size], found {rs:?}"
            ));
        }
        let (n_head, head_size) = (rs[0], rs[1]);
        if n_head * head_size != n_embd {
            return err(format!(
                "n_head*head_size ({n_head}*{head_size}) != n_embd ({n_embd})"
            ));
        }

        let n_layer = (0..)
            .take_while(|i| z.contains_key(&format!("blocks.{i}.att.r_k")))
            .count();
        if n_layer == 0 {
            return err("no blocks found in checkpoint");
        }

        let ffn = z.get("blocks.0.ffn.key.weight").ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "checkpoint is missing 'blocks.0.ffn.key.weight'",
            )
        })?;
        let dim_ffn = ffn.data.len() / n_embd;

        let cfg = Config {
            n_layer,
            n_embd,
            vocab_size,
            head_size,
            n_head,
            dim_ffn,
        };

        // ---- fold ln0 into the embedding table -------------------------------
        let ln0_w = vec1(&mut z, "blocks.0.ln0.weight", n_embd)?;
        let ln0_b = vec1(&mut z, "blocks.0.ln0.bias", n_embd)?;
        let mut emb = take(&mut z, "emb.weight")?.data;
        {
            let mut row = vec![0.0f32; n_embd];
            for v in 0..vocab_size {
                let src = &emb[v * n_embd..(v + 1) * n_embd];
                layer_norm(src, &ln0_w, &ln0_b, LN_EPS, &mut row);
                emb[v * n_embd..(v + 1) * n_embd].copy_from_slice(&row);
            }
        }
        let emb = Mat {
            rows: vocab_size,
            cols: n_embd,
            data: emb,
        };

        // ---- per-layer weights -----------------------------------------------
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let b = format!("blocks.{i}.");
            let a = format!("blocks.{i}.att.");
            let f = format!("blocks.{i}.ffn.");

            // The LoRA rank is whatever the checkpoint uses, so read it from the first
            // factor and hold the second one to it. `[in, rank]` paired with `[rank, in]`.
            let (w1, w2) = lora_pair(&mut z, &a, "w1", "w2", n_embd)?;
            let (a1, a2) = lora_pair(&mut z, &a, "a1", "a2", n_embd)?;
            let (g1, g2) = lora_pair(&mut z, &a, "g1", "g2", n_embd)?;

            let layer = Layer {
                ln1_w: vec1(&mut z, &format!("{b}ln1.weight"), n_embd)?,
                ln1_b: vec1(&mut z, &format!("{b}ln1.bias"), n_embd)?,
                ln2_w: vec1(&mut z, &format!("{b}ln2.weight"), n_embd)?,
                ln2_b: vec1(&mut z, &format!("{b}ln2.bias"), n_embd)?,

                x_r: vec1(&mut z, &format!("{a}x_r"), n_embd)?,
                x_w: vec1(&mut z, &format!("{a}x_w"), n_embd)?,
                x_k: vec1(&mut z, &format!("{a}x_k"), n_embd)?,
                x_v: vec1(&mut z, &format!("{a}x_v"), n_embd)?,
                x_a: vec1(&mut z, &format!("{a}x_a"), n_embd)?,
                x_g: vec1(&mut z, &format!("{a}x_g"), n_embd)?,

                w0: vec1(&mut z, &format!("{a}w0"), n_embd)?,
                w1: w1,
                w2: w2,
                a0: vec1(&mut z, &format!("{a}a0"), n_embd)?,
                a1: a1,
                a2: a2,
                v0: None,
                v1: None,
                v2: None,
                g1: g1,
                g2: g2,

                k_k: vec1(&mut z, &format!("{a}k_k"), n_embd)?,
                k_a: vec1(&mut z, &format!("{a}k_a"), n_embd)?,
                r_k: vec1(&mut z, &format!("{a}r_k"), n_embd)?,

                r_w: mat2(&mut z, &format!("{a}receptance.weight"), n_embd, n_embd)?,
                k_w: mat2(&mut z, &format!("{a}key.weight"), n_embd, n_embd)?,
                v_w: mat2(&mut z, &format!("{a}value.weight"), n_embd, n_embd)?,
                o_w: mat2(&mut z, &format!("{a}output.weight"), n_embd, n_embd)?,

                ln_x_w: vec1(&mut z, &format!("{a}ln_x.weight"), n_embd)?,
                ln_x_b: vec1(&mut z, &format!("{a}ln_x.bias"), n_embd)?,

                ffn_x_k: vec1(&mut z, &format!("{f}x_k"), n_embd)?,
                ffn_k: mat2(&mut z, &format!("{f}key.weight"), dim_ffn, n_embd)?,
                ffn_v: mat2(&mut z, &format!("{f}value.weight"), n_embd, dim_ffn)?,
            };
            let mut layer = layer;

            // Layer 0 has no v-mix LoRA (the demo aliases a0/a1/a2 there and ignores them).
            if i > 0 {
                let (v1, v2) = lora_pair(&mut z, &a, "v1", "v2", n_embd)?;
                layer.v0 = Some(vec1(&mut z, &format!("{a}v0"), n_embd)?);
                layer.v1 = Some(v1);
                layer.v2 = Some(v2);
            }

            layers.push(layer);
        }

        let ln_out_w = vec1(&mut z, "ln_out.weight", n_embd)?;
        let ln_out_b = vec1(&mut z, "ln_out.bias", n_embd)?;
        let head = mat2(&mut z, "head.weight", vocab_size, n_embd)?;

        Ok(Model {
            cfg,
            emb,
            layers,
            ln_out_w,
            ln_out_b,
            head,
        })
    }

    pub fn new_state(&self) -> State {
        State::new(&self.cfg)
    }

    // -----------------------------------------------------------------------
    // RNN step: one token
    // -----------------------------------------------------------------------

    pub fn forward_one(&self, token: u32, state: &mut State, logits: &mut Vec<f32>) {
        let c = &self.cfg;

        let mut x =
            self.emb.data[token as usize * c.n_embd..(token as usize + 1) * c.n_embd].to_vec();
        let mut xx = vec![0.0f32; c.n_embd];
        let mut v_first = vec![0.0f32; c.n_embd];

        // Scratch reused across layers to keep this allocation-free in the hot loop.
        let mut sc = Scratch::new(c);

        for (i, l) in self.layers.iter().enumerate() {
            layer_norm(&x, &l.ln1_w, &l.ln1_b, LN_EPS, &mut xx);
            self.time_mix_one(i, l, &xx, state, &mut v_first, &mut sc);
            for k in 0..c.n_embd {
                x[k] += sc.out[k];
            }

            layer_norm(&x, &l.ln2_w, &l.ln2_b, LN_EPS, &mut xx);
            self.channel_mix_one(i, l, &xx, state, &mut sc);
            for k in 0..c.n_embd {
                x[k] += sc.out[k];
            }
        }

        layer_norm(&x, &self.ln_out_w, &self.ln_out_b, LN_EPS, &mut xx);
        logits.resize(c.vocab_size, 0.0);
        matvec(&self.head.data, c.vocab_size, c.n_embd, &xx, logits);
    }

    fn time_mix_one(
        &self,
        layer_id: usize,
        l: &Layer,
        x: &[f32],
        state: &mut State,
        v_first: &mut [f32],
        s: &mut Scratch,
    ) {
        let c = &self.cfg;
        let (h, n) = (c.n_head, c.head_size);
        let x_prev = &state.att_x[layer_id];

        // Token shift: six independent lerps between this token and the previous one.
        for k in 0..c.n_embd {
            let d = x_prev[k] - x[k];
            s.xr[k] = x[k] + d * l.x_r[k];
            s.xw[k] = x[k] + d * l.x_w[k];
            s.xk[k] = x[k] + d * l.x_k[k];
            s.xv[k] = x[k] + d * l.x_v[k];
            s.xa[k] = x[k] + d * l.x_a[k];
            s.xg[k] = x[k] + d * l.x_g[k];
        }
        state.att_x[layer_id].copy_from_slice(x);

        matvec(&l.r_w.data, c.n_embd, c.n_embd, &s.xr, &mut s.r);
        matvec(&l.k_w.data, c.n_embd, c.n_embd, &s.xk, &mut s.k);
        matvec(&l.v_w.data, c.n_embd, c.n_embd, &s.xv, &mut s.v);

        // w = exp(-exp(-0.5) * sigmoid(w0 + tanh(xw @ w1) @ w2))
        matvec_t(&l.w1.data, c.n_embd, l.w1.cols, &s.xw, &mut s.lora_w);
        for v in s.lora_w.iter_mut() {
            *v = v.tanh();
        }
        matvec_t(&l.w2.data, l.w2.rows, c.n_embd, &s.lora_w, &mut s.w);
        for k in 0..c.n_embd {
            s.w[k] = (-DECAY_SCALE * sigmoid(l.w0[k] + s.w[k])).exp();
        }

        // a = sigmoid(a0 + (xa @ a1) @ a2)
        matvec_t(&l.a1.data, c.n_embd, l.a1.cols, &s.xa, &mut s.lora_a);
        matvec_t(&l.a2.data, l.a2.rows, c.n_embd, &s.lora_a, &mut s.a);
        for k in 0..c.n_embd {
            s.a[k] = sigmoid(l.a0[k] + s.a[k]);
        }

        // g = sigmoid(xg @ g1) @ g2   (note: the nonlinearity sits between the factors)
        matvec_t(&l.g1.data, c.n_embd, l.g1.cols, &s.xg, &mut s.lora_g);
        for v in s.lora_g.iter_mut() {
            *v = sigmoid(*v);
        }
        matvec_t(&l.g2.data, l.g2.rows, c.n_embd, &s.lora_g, &mut s.g);

        // kk = normalize(k * k_k) per head;  k = k * (1 + (a-1) * k_a)
        for k in 0..c.n_embd {
            s.kk[k] = s.k[k] * l.k_k[k];
        }
        l2_normalize_heads(&mut s.kk, h, n);
        for k in 0..c.n_embd {
            s.k[k] *= 1.0 + (s.a[k] - 1.0) * l.k_a[k];
        }

        // Value residual: layer 0 publishes v_first, later layers mix towards it.
        if layer_id == 0 {
            v_first.copy_from_slice(&s.v);
        } else {
            let (v0, v1, v2) = (
                l.v0.as_ref().unwrap(),
                l.v1.as_ref().unwrap(),
                l.v2.as_ref().unwrap(),
            );
            matvec_t(&v1.data, c.n_embd, v1.cols, &s.xv, &mut s.lora_v);
            matvec_t(&v2.data, v2.rows, c.n_embd, &s.lora_v, &mut s.vmix);
            for k in 0..c.n_embd {
                let m = sigmoid(v0[k] + s.vmix[k]);
                s.v[k] += (v_first[k] - s.v[k]) * m;
            }
        }

        wkv7_step(
            &mut state.att_kv[layer_id],
            h,
            n,
            &s.r,
            &s.w,
            &s.k,
            &s.v,
            &s.kk,
            &s.a,
            &mut s.y,
        );

        group_norm_heads(&mut s.y, h, n, &l.ln_x_w, &l.ln_x_b, GN_EPS);

        // Bonus term: a per-head rank-1 read-out added on top of the state output.
        for hh in 0..h {
            let base = hh * n;
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += s.r[base + j] * s.k[base + j] * l.r_k[base + j];
            }
            for j in 0..n {
                s.y[base + j] += acc * s.v[base + j];
            }
        }

        for k in 0..c.n_embd {
            s.y[k] *= s.g[k];
        }
        matvec(&l.o_w.data, c.n_embd, c.n_embd, &s.y, &mut s.out);
    }

    fn channel_mix_one(
        &self,
        layer_id: usize,
        l: &Layer,
        x: &[f32],
        state: &mut State,
        s: &mut Scratch,
    ) {
        let c = &self.cfg;
        let x_prev = &state.ffn_x[layer_id];
        for k in 0..c.n_embd {
            s.xk[k] = x[k] + (x_prev[k] - x[k]) * l.ffn_x_k[k];
        }
        state.ffn_x[layer_id].copy_from_slice(x);

        matvec(&l.ffn_k.data, c.dim_ffn, c.n_embd, &s.xk, &mut s.ffn);
        for v in s.ffn.iter_mut() {
            let r = v.max(0.0);
            *v = r * r; // ReLU squared
        }
        matvec(&l.ffn_v.data, c.n_embd, c.dim_ffn, &s.ffn, &mut s.out);
    }

    // -----------------------------------------------------------------------
    // Batched prefill
    // -----------------------------------------------------------------------

    /// Run `tokens` through the model, returning the logits for the final token.
    /// Equivalent to calling `forward_one` in a loop, but the projections are
    /// done as matmuls so each weight matrix is streamed once per block.
    pub fn forward_seq(&self, tokens: &[u32], state: &mut State, logits: &mut Vec<f32>) {
        let c = &self.cfg;
        let t = tokens.len();
        if t == 0 {
            return;
        }
        if t == 1 {
            return self.forward_one(tokens[0], state, logits);
        }

        let mut x = vec![0.0f32; t * c.n_embd];
        for (i, tok) in tokens.iter().enumerate() {
            let src = *tok as usize * c.n_embd;
            x[i * c.n_embd..(i + 1) * c.n_embd]
                .copy_from_slice(&self.emb.data[src..src + c.n_embd]);
        }

        let mut sq = SeqScratch::new(c, t);
        let mut v_first = vec![0.0f32; t * c.n_embd];

        for (i, l) in self.layers.iter().enumerate() {
            for k in 0..t {
                layer_norm(
                    &x[k * c.n_embd..(k + 1) * c.n_embd],
                    &l.ln1_w,
                    &l.ln1_b,
                    LN_EPS,
                    &mut sq.xx[k * c.n_embd..(k + 1) * c.n_embd],
                );
            }
            self.time_mix_seq(i, l, t, state, &mut v_first, &mut sq);
            for k in 0..t * c.n_embd {
                x[k] += sq.out[k];
            }

            for k in 0..t {
                layer_norm(
                    &x[k * c.n_embd..(k + 1) * c.n_embd],
                    &l.ln2_w,
                    &l.ln2_b,
                    LN_EPS,
                    &mut sq.xx[k * c.n_embd..(k + 1) * c.n_embd],
                );
            }
            self.channel_mix_seq(i, l, t, state, &mut sq);
            for k in 0..t * c.n_embd {
                x[k] += sq.out[k];
            }
        }

        // Only the last position is needed, so the big head matmul stays a matvec.
        let last = &x[(t - 1) * c.n_embd..];
        let mut xn = vec![0.0f32; c.n_embd];
        layer_norm(last, &self.ln_out_w, &self.ln_out_b, LN_EPS, &mut xn);
        logits.resize(c.vocab_size, 0.0);
        matvec(&self.head.data, c.vocab_size, c.n_embd, &xn, logits);
    }

    fn time_mix_seq(
        &self,
        layer_id: usize,
        l: &Layer,
        t: usize,
        state: &mut State,
        v_first: &mut [f32],
        s: &mut SeqScratch,
    ) {
        let c = &self.cfg;
        let (h, n) = (c.n_head, c.head_size);
        let e = c.n_embd;

        // Token shift across the block: position 0 reaches back into the carried state.
        for i in 0..t {
            let cur = &s.xx[i * e..(i + 1) * e];
            for k in 0..e {
                let prev = if i == 0 {
                    state.att_x[layer_id][k]
                } else {
                    s.xx[(i - 1) * e + k]
                };
                let d = prev - cur[k];
                s.xr[i * e + k] = cur[k] + d * l.x_r[k];
                s.xw[i * e + k] = cur[k] + d * l.x_w[k];
                s.xk[i * e + k] = cur[k] + d * l.x_k[k];
                s.xv[i * e + k] = cur[k] + d * l.x_v[k];
                s.xa[i * e + k] = cur[k] + d * l.x_a[k];
                s.xg[i * e + k] = cur[k] + d * l.x_g[k];
            }
        }
        state.att_x[layer_id].copy_from_slice(&s.xx[(t - 1) * e..t * e]);

        matmat(&l.r_w.data, e, e, &s.xr, t, &mut s.r);
        matmat(&l.k_w.data, e, e, &s.xk, t, &mut s.k);
        matmat(&l.v_w.data, e, e, &s.xv, t, &mut s.v);

        matmat_t(&l.w1.data, e, l.w1.cols, &s.xw, t, &mut s.lora_w);
        for v in s.lora_w[..t * l.w1.cols].iter_mut() {
            *v = v.tanh();
        }
        matmat_t(&l.w2.data, l.w2.rows, e, &s.lora_w, t, &mut s.w);
        for i in 0..t {
            for k in 0..e {
                let idx = i * e + k;
                s.w[idx] = (-DECAY_SCALE * sigmoid(l.w0[k] + s.w[idx])).exp();
            }
        }

        matmat_t(&l.a1.data, e, l.a1.cols, &s.xa, t, &mut s.lora_a);
        matmat_t(&l.a2.data, l.a2.rows, e, &s.lora_a, t, &mut s.a);
        for i in 0..t {
            for k in 0..e {
                let idx = i * e + k;
                s.a[idx] = sigmoid(l.a0[k] + s.a[idx]);
            }
        }

        matmat_t(&l.g1.data, e, l.g1.cols, &s.xg, t, &mut s.lora_g);
        for v in s.lora_g[..t * l.g1.cols].iter_mut() {
            *v = sigmoid(*v);
        }
        matmat_t(&l.g2.data, l.g2.rows, e, &s.lora_g, t, &mut s.g);

        for i in 0..t {
            for k in 0..e {
                s.kk[i * e + k] = s.k[i * e + k] * l.k_k[k];
            }
            l2_normalize_heads(&mut s.kk[i * e..(i + 1) * e], h, n);
            for k in 0..e {
                s.k[i * e + k] *= 1.0 + (s.a[i * e + k] - 1.0) * l.k_a[k];
            }
        }

        if layer_id == 0 {
            v_first[..t * e].copy_from_slice(&s.v[..t * e]);
        } else {
            let (v0, v1, v2) = (
                l.v0.as_ref().unwrap(),
                l.v1.as_ref().unwrap(),
                l.v2.as_ref().unwrap(),
            );
            matmat_t(&v1.data, e, v1.cols, &s.xv, t, &mut s.lora_v);
            matmat_t(&v2.data, v2.rows, e, &s.lora_v, t, &mut s.vmix);
            for i in 0..t {
                for k in 0..e {
                    let idx = i * e + k;
                    let m = sigmoid(v0[k] + s.vmix[idx]);
                    s.v[idx] += (v_first[idx] - s.v[idx]) * m;
                }
            }
        }

        // The recurrence itself cannot be batched: scan token by token.
        let kv = &mut state.att_kv[layer_id];
        for i in 0..t {
            let sl = i * e..(i + 1) * e;
            wkv7_step(
                kv,
                h,
                n,
                &s.r[sl.clone()],
                &s.w[sl.clone()],
                &s.k[sl.clone()],
                &s.v[sl.clone()],
                &s.kk[sl.clone()],
                &s.a[sl.clone()],
                &mut s.y[sl],
            );
        }

        for i in 0..t {
            let base = i * e;
            group_norm_heads(&mut s.y[base..base + e], h, n, &l.ln_x_w, &l.ln_x_b, GN_EPS);
            for hh in 0..h {
                let hb = base + hh * n;
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += s.r[hb + j] * s.k[hb + j] * l.r_k[hh * n + j];
                }
                for j in 0..n {
                    s.y[hb + j] += acc * s.v[hb + j];
                }
            }
            for k in 0..e {
                s.y[base + k] *= s.g[base + k];
            }
        }

        matmat(&l.o_w.data, e, e, &s.y, t, &mut s.out);
    }

    fn channel_mix_seq(
        &self,
        layer_id: usize,
        l: &Layer,
        t: usize,
        state: &mut State,
        s: &mut SeqScratch,
    ) {
        let c = &self.cfg;
        let e = c.n_embd;
        for i in 0..t {
            let cur = &s.xx[i * e..(i + 1) * e];
            for k in 0..e {
                let prev = if i == 0 {
                    state.ffn_x[layer_id][k]
                } else {
                    s.xx[(i - 1) * e + k]
                };
                s.xk[i * e + k] = cur[k] + (prev - cur[k]) * l.ffn_x_k[k];
            }
        }
        state.ffn_x[layer_id].copy_from_slice(&s.xx[(t - 1) * e..t * e]);

        matmat(&l.ffn_k.data, c.dim_ffn, e, &s.xk, t, &mut s.ffn);
        for v in s.ffn[..t * c.dim_ffn].iter_mut() {
            let r = v.max(0.0);
            *v = r * r;
        }
        matmat(&l.ffn_v.data, e, c.dim_ffn, &s.ffn, t, &mut s.out);
    }
}

/// The RWKV-7 recurrence, one token, all heads.
///
/// For each head, with `S` the `[head_size, head_size]` state (row = value dim,
/// column = key dim):
///
/// ```text
/// sa[i]   = sum_j (-kk[j]) * S[i][j]
/// S[i][j] = S[i][j] * w[j] + k[j] * v[i] + sa[i] * (kk[j] * a[j])
/// y[i]    = sum_j S[i][j] * r[j]
/// ```
///
/// This is the scalar form of `cuda/wkv7s.cu`, which is equivalent to (and much
/// cheaper than) materialising the two rank-1 outer products from the demo's
/// `state * w + state @ ab + vk`.
#[inline]
fn wkv7_step(
    state: &mut [f32],
    n_head: usize,
    n: usize,
    r: &[f32],
    w: &[f32],
    k: &[f32],
    v: &[f32],
    kk: &[f32],
    a: &[f32],
    y: &mut [f32],
) {
    for h in 0..n_head {
        let base = h * n;
        let s = &mut state[h * n * n..(h + 1) * n * n];
        let (rh, wh, kh, vh) = (
            &r[base..base + n],
            &w[base..base + n],
            &k[base..base + n],
            &v[base..base + n],
        );
        let kkh = &kk[base..base + n];
        let ah = &a[base..base + n];

        for i in 0..n {
            let row = &mut s[i * n..(i + 1) * n];

            // sa = <-kk, S[i]>
            let mut sa = 0.0f32;
            for j in 0..n {
                sa -= kkh[j] * row[j];
            }

            let vi = vh[i];
            let mut acc = 0.0f32;
            for j in 0..n {
                let sv = row[j] * wh[j] + kh[j] * vi + sa * (kkh[j] * ah[j]);
                row[j] = sv;
                acc += sv * rh[j];
            }
            y[base + i] = acc;
        }
    }
}

/// Per-token scratch buffers.
struct Scratch {
    xr: Vec<f32>,
    xw: Vec<f32>,
    xk: Vec<f32>,
    xv: Vec<f32>,
    xa: Vec<f32>,
    xg: Vec<f32>,
    r: Vec<f32>,
    w: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    a: Vec<f32>,
    g: Vec<f32>,
    kk: Vec<f32>,
    y: Vec<f32>,
    out: Vec<f32>,
    ffn: Vec<f32>,
    vmix: Vec<f32>,
    lora_w: Vec<f32>,
    lora_a: Vec<f32>,
    lora_g: Vec<f32>,
    lora_v: Vec<f32>,
}

/// LoRA ranks differ per tensor; one buffer sized to the largest is simpler than
/// tracking each, and they are tiny relative to everything else.
const MAX_LORA: usize = 1024;

impl Scratch {
    fn new(c: &Config) -> Self {
        let e = c.n_embd;
        let z = || vec![0.0f32; e];
        Scratch {
            xr: z(),
            xw: z(),
            xk: z(),
            xv: z(),
            xa: z(),
            xg: z(),
            r: z(),
            w: z(),
            k: z(),
            v: z(),
            a: z(),
            g: z(),
            kk: z(),
            y: z(),
            out: z(),
            ffn: vec![0.0; c.dim_ffn],
            vmix: z(),
            lora_w: vec![0.0; MAX_LORA],
            lora_a: vec![0.0; MAX_LORA],
            lora_g: vec![0.0; MAX_LORA],
            lora_v: vec![0.0; MAX_LORA],
        }
    }
}

/// Same, sized for a T-token block.
struct SeqScratch {
    xx: Vec<f32>,
    xr: Vec<f32>,
    xw: Vec<f32>,
    xk: Vec<f32>,
    xv: Vec<f32>,
    xa: Vec<f32>,
    xg: Vec<f32>,
    r: Vec<f32>,
    w: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    a: Vec<f32>,
    g: Vec<f32>,
    kk: Vec<f32>,
    y: Vec<f32>,
    out: Vec<f32>,
    ffn: Vec<f32>,
    vmix: Vec<f32>,
    lora_w: Vec<f32>,
    lora_a: Vec<f32>,
    lora_g: Vec<f32>,
    lora_v: Vec<f32>,
}

impl SeqScratch {
    fn new(c: &Config, t: usize) -> Self {
        let e = c.n_embd;
        let z = || vec![0.0f32; t * e];
        SeqScratch {
            xx: z(),
            xr: z(),
            xw: z(),
            xk: z(),
            xv: z(),
            xa: z(),
            xg: z(),
            r: z(),
            w: z(),
            k: z(),
            v: z(),
            a: z(),
            g: z(),
            kk: z(),
            y: z(),
            out: z(),
            ffn: vec![0.0; t * c.dim_ffn],
            vmix: z(),
            lora_w: vec![0.0; t * MAX_LORA],
            lora_a: vec![0.0; t * MAX_LORA],
            lora_g: vec![0.0; t * MAX_LORA],
            lora_v: vec![0.0; t * MAX_LORA],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both checkpoint formats, both forward paths, against logits produced by the
    /// numpy reference in `tools/reference_forward.py` (itself a transcription of
    /// `rwkv_v7_demo_fast.py`). Regenerate the fixtures with `tools/make_test_checkpoint.py`.
    fn check(path: &str) {
        let expect: Vec<f32> = std::fs::read("tests/tiny_logits.bin")
            .expect("missing tests/tiny_logits.bin")
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let tokens: Vec<u32> = std::fs::read_to_string("tests/tiny_tokens.txt")
            .unwrap()
            .trim()
            .split(',')
            .map(|s| s.trim().parse().unwrap())
            .collect();

        let model =
            Model::load(crate::loader::load(path).expect("load failed")).expect("build failed");
        assert_eq!(model.cfg.n_layer, 3);
        assert_eq!(model.cfg.n_embd, 32);
        assert_eq!(model.cfg.n_head, 2);
        assert_eq!(model.cfg.head_size, 16);

        // Every chunk size must give the same answer: chunk=1 is the pure RNN path,
        // anything larger goes through the batched prefill.
        for chunk in [1usize, 2, 3, 7, 20, 64] {
            let mut state = model.new_state();
            let mut logits = Vec::new();
            for block in tokens.chunks(chunk) {
                model.forward_seq(block, &mut state, &mut logits);
            }
            assert_eq!(logits.len(), expect.len());
            let max_diff = logits
                .iter()
                .zip(&expect)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-4,
                "{path} chunk={chunk}: max|diff| = {max_diff:e} vs the numpy reference"
            );
        }
    }

    #[test]
    fn matches_reference_pth() {
        check("tests/tiny.pth");
    }

    #[test]
    fn matches_reference_safetensors() {
        check("tests/tiny.safetensors");
    }

    /// The two loaders must agree bit-for-bit on the same weights.
    #[test]
    fn loaders_agree() {
        let a = crate::loader::load("tests/tiny.pth").unwrap();
        let b = crate::loader::load("tests/tiny.safetensors").unwrap();
        assert_eq!(a.len(), b.len());
        for (k, v) in &a {
            let w = b
                .get(k)
                .unwrap_or_else(|| panic!("{k} missing from safetensors"));
            assert_eq!(v.shape, w.shape, "{k}: shape mismatch between loaders");
            assert_eq!(v.data, w.data, "{k}: data mismatch between loaders");
        }
    }
}
