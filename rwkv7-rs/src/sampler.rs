//! Logit sampling, matching `sample_logits` in the Python demos exactly —
//! including the quirk that temperature is applied *after* top-p/top-k, and that
//! `top_p == 0.0` collapses to greedy decoding.

/// xoshiro256++ — small, fast, and good enough for token sampling.
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to spread a single seed across the full state.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    fn next_u64(&mut self) -> u64 {
        let r = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

pub struct SampleOpts {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

/// Softmax in f32 with the usual max-subtraction for stability.
pub fn softmax(logits: &[f32], out: &mut Vec<f32>) {
    out.clear();
    out.reserve(logits.len());
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &l in logits {
        let e = (l - max).exp();
        out.push(e);
        sum += e as f64;
    }
    let inv = (1.0 / sum) as f32;
    for v in out.iter_mut() {
        *v *= inv;
    }
}

pub fn sample(logits: &[f32], opts: &SampleOpts, rng: &mut Rng) -> u32 {
    let mut probs = Vec::new();
    softmax(logits, &mut probs);

    // Descending order by probability; ties broken by index for reproducibility.
    let mut order: Vec<u32> = (0..probs.len() as u32).collect();
    order.sort_unstable_by(|a, b| {
        probs[*b as usize]
            .partial_cmp(&probs[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });

    if opts.top_k > 0 && opts.top_k < order.len() {
        for &i in &order[opts.top_k..] {
            probs[i as usize] = 0.0;
        }
    }

    if opts.top_p < 1.0 {
        // Smallest kept probability: the first point where the cumulative mass
        // reaches top_p. With top_p == 0 that is the argmax, i.e. greedy.
        let mut cum = 0.0f32;
        // Fallback if rounding keeps the running sum just under top_p: keep everything.
        // (NumPy's searchsorted would return an out-of-range index here and throw.)
        let mut cutoff = probs[order[order.len() - 1] as usize];
        for &i in &order {
            cum += probs[i as usize];
            if cum >= opts.top_p {
                cutoff = probs[i as usize];
                break;
            }
        }
        for v in probs.iter_mut() {
            if *v < cutoff {
                *v = 0.0;
            }
        }
        if opts.top_p > 0.0 {
            // Redistribute so the surviving mass is exactly top_p, as the demo does.
            let idx: Vec<usize> = (0..probs.len()).filter(|&i| probs[i] == cutoff).collect();
            if !idx.is_empty() {
                let total: f32 = probs.iter().sum();
                let share = cutoff + (opts.top_p - total) / idx.len() as f32;
                for i in idx {
                    probs[i] = share;
                }
            }
        }
    }

    if opts.temperature != 1.0 {
        let inv = 1.0 / opts.temperature;
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

/// Indices of the `k` largest values, most likely first.
pub fn top_indices(probs: &[f32], k: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..probs.len() as u32).collect();
    let k = k.min(order.len());
    if k == 0 {
        return Vec::new();
    }
    order.sort_unstable_by(|a, b| {
        probs[*b as usize]
            .partial_cmp(&probs[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(k);
    order
}
