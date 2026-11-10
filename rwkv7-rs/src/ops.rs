//! Dense kernels. Everything is f32, row-major, and deliberately plain so the
//! mapping back to `rwkv_v7_demo_fast.py` stays obvious.
//!
//! Two matrix conventions are in play, matching the checkpoint layout:
//!   * `Linear.weight` is `[out, in]`  -> `matvec` / `matmat`      (y = W @ x)
//!   * LoRA factors are `[in, out]`    -> `matvec_t` / `matmat_t`  (y = x @ M)

use crate::pool::{chunk, pool};

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Four independent accumulators: breaks the serial dependency on the FMA
    // latency and lets the autovectoriser use full-width registers.
    debug_assert_eq!(a.len(), b.len());
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

/// Work below this many multiply-adds runs inline; the sync cost of waking the
/// pool is not worth it for the small LoRA projections.
const PARALLEL_THRESHOLD: usize = 1 << 17;

/// `out[o] = dot(w[o], x)` for `w: [n_out, n_in]`.
pub fn matvec(w: &[f32], n_out: usize, n_in: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(w.len(), n_out * n_in);
    debug_assert_eq!(x.len(), n_in);
    debug_assert_eq!(out.len(), n_out);

    if n_out * n_in < PARALLEL_THRESHOLD {
        for (o, dst) in out.iter_mut().enumerate() {
            *dst = dot(&w[o * n_in..o * n_in + n_in], x);
        }
        return;
    }

    let p = pool();
    let out_ptr = SendPtr(out.as_mut_ptr());
    p.parallel(|t| {
        let (lo, hi) = chunk(n_out, p.threads(), t);
        for o in lo..hi {
            let v = dot(&w[o * n_in..o * n_in + n_in], x);
            // SAFETY: worker `t` owns rows [lo, hi) exclusively.
            unsafe { *out_ptr.get().add(o) = v };
        }
    });
}

/// `out[t][o] = dot(w[o], x[t])` for `w: [n_out, n_in]`, `x: [n_tok, n_in]`.
/// Parallelised over output rows so each `w` row is fetched once and reused
/// across all tokens in the block.
pub fn matmat(w: &[f32], n_out: usize, n_in: usize, x: &[f32], n_tok: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), n_tok * n_out);

    let body = |lo: usize, hi: usize, out_ptr: *mut f32| {
        for o in lo..hi {
            let row = &w[o * n_in..o * n_in + n_in];
            for t in 0..n_tok {
                let v = dot(row, &x[t * n_in..t * n_in + n_in]);
                unsafe { *out_ptr.add(t * n_out + o) = v };
            }
        }
    };

    if n_out * n_in * n_tok < PARALLEL_THRESHOLD {
        body(0, n_out, out.as_mut_ptr());
        return;
    }
    let p = pool();
    let out_ptr = SendPtr(out.as_mut_ptr());
    p.parallel(|t| {
        let (lo, hi) = chunk(n_out, p.threads(), t);
        body(lo, hi, out_ptr.get());
    });
}

/// `out = x @ m` for `m: [n_in, n_out]` (the LoRA layout).
pub fn matvec_t(m: &[f32], n_in: usize, n_out: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(m.len(), n_in * n_out);
    out.fill(0.0);
    for i in 0..n_in {
        let xi = x[i];
        let row = &m[i * n_out..i * n_out + n_out];
        for o in 0..n_out {
            out[o] += xi * row[o];
        }
    }
}

/// `out[t] = x[t] @ m` for `m: [n_in, n_out]`, `x: [n_tok, n_in]`.
pub fn matmat_t(m: &[f32], n_in: usize, n_out: usize, x: &[f32], n_tok: usize, out: &mut [f32]) {
    let body = |lo: usize, hi: usize, out_ptr: *mut f32| {
        for t in lo..hi {
            // SAFETY: worker owns token rows [lo, hi) exclusively.
            let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr.add(t * n_out), n_out) };
            matvec_t(m, n_in, n_out, &x[t * n_in..t * n_in + n_in], dst);
        }
    };
    if n_tok * n_in * n_out < PARALLEL_THRESHOLD {
        body(0, n_tok, out.as_mut_ptr());
        return;
    }
    let p = pool();
    let out_ptr = SendPtr(out.as_mut_ptr());
    p.parallel(|t| {
        let (lo, hi) = chunk(n_tok, p.threads(), t);
        body(lo, hi, out_ptr.get());
    });
}

/// LayerNorm over the whole vector. Uses the biased variance, as PyTorch does.
pub fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let inv_n = 1.0 / n as f32;
    let mean = x.iter().sum::<f32>() * inv_n;
    let var = x
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        * inv_n;
    let scale = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        out[i] = (x[i] - mean) * scale * w[i] + b[i];
    }
}

/// GroupNorm with one group per head, i.e. normalise each contiguous run of
/// `head_size` elements independently. RWKV-7 uses eps = 64e-5, not the usual 1e-5.
pub fn group_norm_heads(
    x: &mut [f32],
    n_head: usize,
    head_size: usize,
    w: &[f32],
    b: &[f32],
    eps: f32,
) {
    let inv_n = 1.0 / head_size as f32;
    for h in 0..n_head {
        let s = h * head_size;
        let seg = &mut x[s..s + head_size];
        let mean = seg.iter().sum::<f32>() * inv_n;
        let var = seg
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            * inv_n;
        let scale = 1.0 / (var + eps).sqrt();
        for i in 0..head_size {
            seg[i] = (seg[i] - mean) * scale * w[s + i] + b[s + i];
        }
    }
}

/// L2-normalise each head slice in place (`F.normalize(..., dim=-1, p=2)`).
pub fn l2_normalize_heads(x: &mut [f32], n_head: usize, head_size: usize) {
    for h in 0..n_head {
        let seg = &mut x[h * head_size..(h + 1) * head_size];
        let norm = seg.iter().map(|v| v * v).sum::<f32>().sqrt();
        // PyTorch clamps the denominator rather than skipping the division.
        let inv = 1.0 / norm.max(1e-12);
        for v in seg.iter_mut() {
            *v *= inv;
        }
    }
}

/// Wrapper that lets a raw output pointer cross into worker closures.
/// Every use below partitions the destination so no two workers touch the same index.
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    /// Going through a method (rather than touching `.0`) makes the closure capture
    /// the whole wrapper; edition-2021 disjoint capture would otherwise grab the bare
    /// `*mut f32` field and lose the `Send`/`Sync` impls.
    #[inline(always)]
    fn get(self) -> *mut f32 {
        self.0
    }
}
