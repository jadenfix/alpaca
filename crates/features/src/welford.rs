//! Welford's online mean/variance — single-pass, numerically stable.
//!
//! This is the streaming form. For batch (per-bar across the universe) the
//! `ndarray` BLAS path in `zscore.rs` is preferred.

#[derive(Clone, Debug, Default)]
pub struct Welford {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Welford {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn n(&self) -> u64 {
        self.n
    }
    pub fn mean(&self) -> f64 {
        self.mean
    }
    /// Sample variance (Bessel-corrected). NaN if n < 2.
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            f64::NAN
        } else {
            self.m2 / (self.n - 1) as f64
        }
    }
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_batch_within_eps() {
        let xs: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let mut w = Welford::new();
        for &x in &xs {
            w.push(x);
        }
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
        assert!((w.mean() - mean).abs() < 1e-9);
        assert!((w.variance() - var).abs() < 1e-9);
    }

    #[test]
    fn n_lt_2_is_nan() {
        let mut w = Welford::new();
        w.push(1.0);
        assert!(w.variance().is_nan());
    }
}
