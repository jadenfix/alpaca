//! Transfer Entropy — Schreiber (2000) "Measuring Information Transfer."
//!
//! For two time series X, Y, the transfer entropy from X to Y is the
//! reduction in uncertainty of Y_{t+1} given Y's own past, attributable
//! to also knowing X's past:
//!
//! ```text
//! T_{X->Y} = H(Y_{t+1} | Y_t^k) - H(Y_{t+1} | Y_t^k, X_t^l)
//!          = sum p(y_{t+1}, y_t^k, x_t^l) * log[ p(y_{t+1} | y_t^k, x_t^l)
//!                                                / p(y_{t+1} | y_t^k) ]
//! ```
//!
//! Unlike Pearson, MI, or Granger F-tests, transfer entropy is:
//!   - **directed**: T_{X→Y} ≠ T_{Y→X} in general.
//!   - **model-free**: no linearity / Gaussianity assumption.
//!   - **nonlinear**: catches dependencies invisible to Pearson.
//!
//! Implementation: binned plug-in estimator. We discretize each series into
//! `bins` quantile-based bins and tally a joint histogram. Practical use
//! caps at k+l ≤ 3 (curse of dimensionality).
//!
//! Returned units: nats (natural log).

use std::collections::HashMap;

/// Compute transfer entropy T_{X→Y} in nats, where `src = X`, `dst = Y`.
/// `k` is the history depth of Y, `l` is the history depth of X (typically
/// 1, 1). `bins` is the number of quantile bins per variable (4-8 sweet spot).
pub fn transfer_entropy(src: &[f64], dst: &[f64], k: usize, l: usize, bins: usize) -> f64 {
    assert_eq!(src.len(), dst.len(), "transfer_entropy: lengths differ");
    assert!(k >= 1 && l >= 1, "k and l must be ≥ 1");
    assert!(bins >= 2, "bins must be ≥ 2");
    let n = src.len();
    let history = k.max(l);
    if n < history + 2 {
        return 0.0;
    }

    let dst_bins = quantile_bin(dst, bins);
    let src_bins = quantile_bin(src, bins);

    let mut joint: HashMap<u64, f64> = HashMap::new();
    let mut p_yk_xl: HashMap<u64, f64> = HashMap::new();
    let mut p_y1_yk: HashMap<u64, f64> = HashMap::new();
    let mut p_yk: HashMap<u64, f64> = HashMap::new();

    let stride = (bins as u64).pow(history as u32 + 1);
    let mut total = 0.0;
    for t in history..(n - 1) {
        let y_next = dst_bins[t + 1] as u64;
        let y_hist = encode_history(&dst_bins, t, k, bins);
        let x_hist = encode_history(&src_bins, t, l, bins);
        let joint_key = y_next * stride * stride + y_hist * stride + x_hist;
        let yk_xl_key = y_hist * stride + x_hist;
        let y1_yk_key = y_next * stride + y_hist;
        let yk_key = y_hist;
        *joint.entry(joint_key).or_insert(0.0) += 1.0;
        *p_yk_xl.entry(yk_xl_key).or_insert(0.0) += 1.0;
        *p_y1_yk.entry(y1_yk_key).or_insert(0.0) += 1.0;
        *p_yk.entry(yk_key).or_insert(0.0) += 1.0;
        total += 1.0;
    }
    if total <= 0.0 {
        return 0.0;
    }

    let mut te = 0.0;
    for (&jk, &cnt) in &joint {
        if cnt <= 0.0 {
            continue;
        }
        let p_joint = cnt / total;
        let y_next = jk / (stride * stride);
        let y_hist = (jk % (stride * stride)) / stride;
        let x_hist = jk % stride;
        let p_yk_xl_v = *p_yk_xl.get(&(y_hist * stride + x_hist)).unwrap_or(&0.0) / total;
        let p_y1_yk_v = *p_y1_yk.get(&(y_next * stride + y_hist)).unwrap_or(&0.0) / total;
        let p_yk_v = *p_yk.get(&y_hist).unwrap_or(&0.0) / total;
        if p_yk_xl_v <= 0.0 || p_y1_yk_v <= 0.0 || p_yk_v <= 0.0 {
            continue;
        }
        let ratio = (p_joint * p_yk_v) / (p_yk_xl_v * p_y1_yk_v);
        if ratio > 0.0 {
            te += p_joint * ratio.ln();
        }
    }
    te.max(0.0)
}

fn quantile_bin(xs: &[f64], bins: usize) -> Vec<u32> {
    let n = xs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| xs[i].partial_cmp(&xs[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0u32; n];
    for (rank, &i) in idx.iter().enumerate() {
        let b = ((rank * bins) / n).min(bins - 1) as u32;
        out[i] = b;
    }
    out
}

fn encode_history(bins_arr: &[u32], t: usize, k: usize, bins: usize) -> u64 {
    let base = bins as u64;
    let mut acc = 0u64;
    for off in 0..k {
        acc = acc * base + bins_arr[t - off] as u64;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_normal(seed: u64) -> impl FnMut() -> f64 {
        let mut s = if seed == 0 { 1u64 } else { seed };
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u1 = ((s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u2 = ((s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    #[test]
    fn te_independent_series_is_small() {
        let mut g = rng_normal(7);
        let n = 500;
        let x: Vec<f64> = (0..n).map(|_| g()).collect();
        let mut h = rng_normal(99);
        let y: Vec<f64> = (0..n).map(|_| h()).collect();
        let te = transfer_entropy(&x, &y, 1, 1, 4);
        assert!(te < 0.1, "independent TE too large: {te}");
    }

    #[test]
    fn te_x_drives_y_is_strictly_positive_and_asymmetric() {
        // y_{t+1} depends nonlinearly on x_t.
        let mut g = rng_normal(11);
        let n = 1500;
        let mut x = vec![g()];
        let mut y = vec![g()];
        for t in 0..(n - 1) {
            x.push(0.3 * x[t] + g());
            let next_y = 0.5 * y[t].abs().powf(0.9) * x[t].signum() + 0.1 * g();
            y.push(next_y);
        }
        let te_xy = transfer_entropy(&x, &y, 1, 1, 4);
        let te_yx = transfer_entropy(&y, &x, 1, 1, 4);
        assert!(te_xy > 0.02, "TE(X→Y) should be positive, got {te_xy}");
        assert!(
            te_xy > te_yx,
            "TE asymmetry: expected TE(X→Y)={te_xy:.4} > TE(Y→X)={te_yx:.4}"
        );
    }

    #[test]
    fn te_self_is_near_zero() {
        // Conditioning on Y_t^k = X_t^k makes X_t^l = same info redundant → TE ≈ 0.
        let mut g = rng_normal(13);
        let n = 600;
        let mut x = vec![g()];
        for t in 0..(n - 1) {
            x.push(0.6 * x[t] + g());
        }
        let te = transfer_entropy(&x, &x, 1, 1, 4);
        assert!(te < 0.05, "self-TE should be ≈ 0, got {te}");
    }

    #[test]
    fn quantile_bin_assigns_balanced_buckets() {
        let xs: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let b = quantile_bin(&xs, 4);
        for bucket in 0..4 {
            let count = b.iter().filter(|&&v| v == bucket).count();
            assert!(
                (count as i64 - 25).abs() <= 1,
                "bucket {bucket} has {count} elements, expected ~25"
            );
        }
    }
}
