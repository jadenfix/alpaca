//! Vectorized returns. These run per-bar on the universe; vectorized is the
//! discipline rule.

use ndarray::{Array1, ArrayView1};

/// Simple returns from a price vector: `(p_t - p_{t-1}) / p_{t-1}`.
/// Returns a vector of length `prices.len() - 1`.
pub fn simple_returns_vec(prices: ArrayView1<f64>) -> Array1<f64> {
    if prices.len() < 2 {
        return Array1::zeros(0);
    }
    let prev = prices.slice(ndarray::s![..-1]);
    let curr = prices.slice(ndarray::s![1..]);
    // Vectorized: (curr - prev) / prev
    let mut out = &curr - &prev;
    out /= &prev;
    out
}

/// Log returns. Vectorized via ndarray's `mapv`.
pub fn log_returns_vec(prices: ArrayView1<f64>) -> Array1<f64> {
    if prices.len() < 2 {
        return Array1::zeros(0);
    }
    let prev = prices.slice(ndarray::s![..-1]);
    let curr = prices.slice(ndarray::s![1..]);
    let ratio = &curr / &prev;
    ratio.mapv(f64::ln)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn simple_returns_known() {
        let p = array![100.0, 101.0, 99.99];
        let r = simple_returns_vec(p.view());
        assert!((r[0] - 0.01).abs() < 1e-12);
        // (99.99 - 101) / 101 = -1.01 / 101 = -0.01
        assert!((r[1] - (-1.01 / 101.0)).abs() < 1e-12);
    }

    #[test]
    fn log_returns_match_log_ratio() {
        let p = array![100.0, 110.0];
        let r = log_returns_vec(p.view());
        assert!((r[0] - (110f64 / 100.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn empty_when_too_short() {
        let p = array![1.0];
        assert_eq!(simple_returns_vec(p.view()).len(), 0);
        assert_eq!(log_returns_vec(p.view()).len(), 0);
    }
}
