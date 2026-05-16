//! Lead-lag detection between two return series via lagged cross-correlation.
//!
//! For two return series r_x and r_y of equal length, compute
//!   ρ(τ) = corr(r_y_t, r_x_{t-τ})   for τ ∈ [-max_lag, +max_lag]
//! and return the τ* maximizing |ρ|, along with the value.
//!
//! Interpretation:
//!   - τ* > 0  → X leads Y by τ* bars (Y reacts to X)
//!   - τ* < 0  → Y leads X
//!   - τ* = 0  → contemporaneous correlation
//!
//! Use case: trade lagged response to a known leader (e.g. SPY → individual
//! large-cap names; index ETF → constituent reversion).
//!
//! References:
//!   - Hayashi & Yoshida (2005), "On covariance estimation of non-synchronously
//!     observed diffusion processes" (for the tick-data version)
//!   - de Jong & Nijman (1997), "High-frequency lead-lag relationships"

#[derive(Copy, Clone, Debug)]
pub struct LeadLagResult {
    pub best_lag: i32,
    pub best_corr: f64,
    pub contemporaneous: f64,
}

/// Compute lead-lag using lagged Pearson correlation.
/// `max_lag` is the maximum absolute lag in bars to consider.
/// Returns `None` if the series are too short or have zero variance.
pub fn lead_lag(rx: &[f64], ry: &[f64], max_lag: usize) -> Option<LeadLagResult> {
    let n = rx.len();
    if n != ry.len() || n < 2 * max_lag + 10 {
        return None;
    }
    let mean_x = rx.iter().sum::<f64>() / n as f64;
    let mean_y = ry.iter().sum::<f64>() / n as f64;
    let var_x: f64 = rx.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>() / n as f64;
    let var_y: f64 = ry.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>() / n as f64;
    if var_x <= 0.0 || var_y <= 0.0 {
        return None;
    }
    let denom = (var_x * var_y).sqrt();

    let mut best_lag = 0i32;
    let mut best_corr = 0.0_f64;
    let mut contemporaneous = 0.0_f64;

    for lag in -(max_lag as i32)..=(max_lag as i32) {
        let mut cov = 0.0;
        let mut cnt = 0i32;
        // corr(y_t, x_{t-lag}): both indices must be in [0, n)
        for t in 0..n {
            let ix = t as i32 - lag;
            if ix < 0 || (ix as usize) >= n {
                continue;
            }
            cov += (ry[t] - mean_y) * (rx[ix as usize] - mean_x);
            cnt += 1;
        }
        if cnt == 0 {
            continue;
        }
        let c = cov / cnt as f64 / denom;
        if lag == 0 {
            contemporaneous = c;
        }
        if c.abs() > best_corr.abs() {
            best_corr = c;
            best_lag = lag;
        }
    }

    Some(LeadLagResult {
        best_lag,
        best_corr,
        contemporaneous,
    })
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
    fn detects_x_leads_y_by_one() {
        // Construct: y_t = x_{t-1} + small_noise; X leads Y by 1.
        let mut g = rng_normal(7);
        let n = 1_000;
        let x: Vec<f64> = (0..n).map(|_| g()).collect();
        let mut y = vec![0.0; n];
        for t in 1..n {
            y[t] = x[t - 1] + 0.1 * g();
        }
        let r = lead_lag(&x, &y, 5).unwrap();
        assert_eq!(r.best_lag, 1, "expected X to lead Y by 1, got lag={}", r.best_lag);
        assert!(r.best_corr.abs() > 0.5, "lagged corr should be strong, got {}", r.best_corr);
    }

    #[test]
    fn no_lead_when_independent() {
        let mut g = rng_normal(11);
        let n = 1_000;
        let x: Vec<f64> = (0..n).map(|_| g()).collect();
        let y: Vec<f64> = (0..n).map(|_| g()).collect();
        let r = lead_lag(&x, &y, 5).unwrap();
        // Independent series: |best_corr| should be small (~ 1/√n)
        assert!(r.best_corr.abs() < 0.15, "independent series, got |corr|={}", r.best_corr.abs());
    }

    #[test]
    fn contemporaneous_when_y_equals_x() {
        let mut g = rng_normal(13);
        let n = 500;
        let x: Vec<f64> = (0..n).map(|_| g()).collect();
        let y = x.clone();
        let r = lead_lag(&x, &y, 5).unwrap();
        assert_eq!(r.best_lag, 0);
        assert!((r.contemporaneous - 1.0).abs() < 1e-9);
    }
}
