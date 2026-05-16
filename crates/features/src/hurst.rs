//! Hurst exponent via rescaled-range (R/S) analysis.
//!
//! The Hurst exponent H ∈ (0, 1) classifies a series' fractal behavior:
//!   - H > 0.5  → trending (persistent; trend-following works)
//!   - H = 0.5  → random walk (efficient market)
//!   - H < 0.5  → mean-reverting (anti-persistent; pairs / MR works)
//!
//! Used as a regime filter: enable trend-following only when H > 0.55, enable
//! mean-reversion only when H < 0.45.
//!
//! Reference: Hurst (1951); Mandelbrot & Wallis (1968).

/// Estimate H via R/S analysis over a price series.
/// Returns `None` if `series.len() < 32`.
pub fn hurst_rs(series: &[f64]) -> Option<f64> {
    let n = series.len();
    if n < 32 {
        return None;
    }
    // Use log-returns
    let mut returns: Vec<f64> = Vec::with_capacity(n - 1);
    for i in 1..n {
        let r = (series[i] / series[i - 1].max(1e-12)).ln();
        if r.is_finite() {
            returns.push(r);
        } else {
            return None;
        }
    }
    let m = returns.len();
    if m < 32 {
        return None;
    }
    // Subseries lengths from ~8 to m/2, log-spaced.
    let mut ks = Vec::new();
    let mut k = 8usize;
    while k <= m / 2 {
        ks.push(k);
        k = (k as f64 * 1.5).ceil() as usize;
    }
    if ks.len() < 4 {
        return None;
    }
    let mut log_k = Vec::with_capacity(ks.len());
    let mut log_rs = Vec::with_capacity(ks.len());
    for &k in &ks {
        let n_chunks = m / k;
        let mut sum_rs = 0.0;
        let mut count = 0;
        for c in 0..n_chunks {
            let chunk = &returns[c * k..(c + 1) * k];
            let mean: f64 = chunk.iter().sum::<f64>() / k as f64;
            // Cumulative deviation
            let mut dev = 0.0;
            let mut min_dev = 0.0;
            let mut max_dev = 0.0;
            for &x in chunk {
                dev += x - mean;
                if dev < min_dev { min_dev = dev; }
                if dev > max_dev { max_dev = dev; }
            }
            let r = max_dev - min_dev;
            let s: f64 = (chunk.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / k as f64).sqrt();
            if s > 1e-12 && r > 0.0 {
                sum_rs += r / s;
                count += 1;
            }
        }
        if count > 0 {
            let avg_rs = sum_rs / count as f64;
            if avg_rs > 0.0 {
                log_k.push((k as f64).ln());
                log_rs.push(avg_rs.ln());
            }
        }
    }
    if log_k.len() < 4 {
        return None;
    }
    // OLS slope of log_rs ~ a + H * log_k
    let n = log_k.len() as f64;
    let mean_x = log_k.iter().sum::<f64>() / n;
    let mean_y = log_rs.iter().sum::<f64>() / n;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..log_k.len() {
        let dx = log_k[i] - mean_x;
        sxx += dx * dx;
        sxy += dx * (log_rs[i] - mean_y);
    }
    if sxx <= 0.0 {
        return None;
    }
    let h = sxy / sxx;
    Some(h.clamp(0.0, 1.0))
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
    fn random_walk_hurst_near_half() {
        let mut g = rng_normal(42);
        let mut p = vec![100.0_f64];
        for _ in 0..2000 {
            let last = *p.last().unwrap();
            p.push(last * (1.0 + 0.01 * g()));
        }
        let h = hurst_rs(&p).unwrap();
        // R/S has positive bias for finite samples; accept a generous band.
        assert!(
            h > 0.40 && h < 0.75,
            "expected H near 0.5 for random walk, got {h}"
        );
    }

    #[test]
    fn trending_series_has_h_above_half() {
        // Strong positive drift = trending → H > 0.5
        let mut g = rng_normal(7);
        let mut p = vec![100.0_f64];
        for _ in 0..2000 {
            let last = *p.last().unwrap();
            p.push(last * (1.0 + 0.001 + 0.005 * g()));
        }
        let h = hurst_rs(&p).unwrap();
        assert!(h > 0.5, "expected H > 0.5 for trending series, got {h}");
    }

    #[test]
    fn mean_reverting_series_has_h_below_half() {
        // OU process around constant mean → mean-reverting → H < 0.5
        let mut g = rng_normal(11);
        let mut x = 100.0_f64;
        let mut p = vec![x];
        for _ in 0..2000 {
            // Strong mean reversion: theta = 0.3
            x += -0.3 * (x - 100.0) + 0.5 * g();
            p.push(x);
        }
        let h = hurst_rs(&p).unwrap();
        assert!(h < 0.5, "expected H < 0.5 for mean-reverting series, got {h}");
    }
}
