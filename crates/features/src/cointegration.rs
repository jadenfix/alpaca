//! Cointegration: Engle-Granger two-step + Augmented Dickey-Fuller test.
//!
//! Algorithm (Engle-Granger):
//!   1. OLS regress `y_t = α + β x_t + ε_t` to get hedge ratio β.
//!   2. Compute residuals `ε_t = y_t - α - β x_t`.
//!   3. Test residuals for stationarity via ADF.
//!   4. If stationary (p < threshold), the pair is cointegrated; trade ε.
//!
//! References:
//!   - Engle & Granger (1987), "Co-integration and Error Correction"
//!   - Avellaneda & Lee (2010), "Statistical Arbitrage in the US Equities Market"
//!
//! Implementation notes:
//!   - Uses normal-equation OLS (X'X β = X'y) for the regression; fine for the
//!     2-column design matrix of Engle-Granger.
//!   - ADF test statistic is the t-stat on the lagged level coefficient in
//!     `Δε_t = ρ ε_{t-1} + Σ φ_i Δε_{t-i} + u_t`. Critical values from
//!     MacKinnon (1996), interpolated; reject H0 (non-stationary) if t < critical.

use ndarray::{Array1, Array2, ArrayView1};

#[derive(Clone, Debug)]
pub struct EngleGrangerResult {
    pub alpha: f64,
    pub beta: f64,
    /// Test statistic on the residual ADF; lower = more stationary.
    pub adf_t: f64,
    /// Approximate one-sided p-value (small = reject H0 of unit root → stationary).
    pub p_value: f64,
    /// Last residual observed (= z_t up to scaling by σ_ε).
    pub last_residual: f64,
    pub residual_mean: f64,
    pub residual_std: f64,
}

impl EngleGrangerResult {
    pub fn is_cointegrated(&self, alpha_level: f64) -> bool {
        self.p_value <= alpha_level
    }

    /// Latest spread z-score: (ε_T - mean) / std.
    pub fn last_z_score(&self) -> f64 {
        if self.residual_std > 0.0 {
            (self.last_residual - self.residual_mean) / self.residual_std
        } else {
            0.0
        }
    }
}

/// Engle-Granger cointegration test on `y` against `x` (same length, ≥ 30).
/// Returns hedge ratio β, ADF p-value, and residual statistics.
pub fn engle_granger(
    y: ArrayView1<f64>,
    x: ArrayView1<f64>,
    adf_lags: usize,
) -> Option<EngleGrangerResult> {
    let n = y.len();
    if n != x.len() || n < 30 {
        return None;
    }
    // OLS via normal equations on [1, x] design matrix.
    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..n {
        let dx = x[i] - mean_x;
        sxx += dx * dx;
        sxy += dx * (y[i] - mean_y);
    }
    if sxx <= 0.0 {
        return None;
    }
    let beta = sxy / sxx;
    let alpha = mean_y - beta * mean_x;

    // Residuals
    let mut resid = Array1::<f64>::zeros(n);
    let mut sum_r = 0.0;
    for i in 0..n {
        let r = y[i] - alpha - beta * x[i];
        resid[i] = r;
        sum_r += r;
    }
    let mean_r = sum_r / n as f64;
    let mut var_r = 0.0;
    for i in 0..n {
        var_r += (resid[i] - mean_r).powi(2);
    }
    let std_r = (var_r / (n as f64 - 1.0)).max(0.0).sqrt();

    let adf_t = adf_test_statistic(resid.view(), adf_lags)?;
    let p_value = adf_pvalue(adf_t, n);

    Some(EngleGrangerResult {
        alpha,
        beta,
        adf_t,
        p_value,
        last_residual: resid[n - 1],
        residual_mean: mean_r,
        residual_std: std_r,
    })
}

/// Augmented Dickey-Fuller test statistic on series `y`.
///
/// Regresses Δy_t = α + ρ y_{t-1} + Σ_{i=1..lags} φ_i Δy_{t-i} + u_t
/// and returns the t-statistic on `ρ`. Returns `None` on degenerate input.
///
/// A more negative t indicates stronger rejection of the unit-root null.
pub fn adf_test_statistic(y: ArrayView1<f64>, lags: usize) -> Option<f64> {
    let n = y.len();
    if n < lags + 4 {
        return None;
    }
    // Build first differences
    let mut diff = Vec::with_capacity(n - 1);
    for i in 1..n {
        diff.push(y[i] - y[i - 1]);
    }
    // Rows: t ∈ [lags+1 .. n-1] in original index terms (since we need Δy_{t-lags})
    // Predictor columns: 1 (intercept), y_{t-1}, Δy_{t-1}, …, Δy_{t-lags}
    let n_eff = diff.len() - lags;
    if n_eff < 5 {
        return None;
    }
    let k = 2 + lags; // intercept + y_{t-1} + lags
    let mut xmat = Array2::<f64>::zeros((n_eff, k));
    let mut yvec = Array1::<f64>::zeros(n_eff);

    for row in 0..n_eff {
        // diff index for current row's Δy_t
        let t = row + lags; // diff[t] is Δy_t
        yvec[row] = diff[t];
        xmat[(row, 0)] = 1.0;
        // y_{t-1} in original series: original index of diff[t] is t+1, so y_{t-1} = y[t]
        xmat[(row, 1)] = y[t];
        for j in 0..lags {
            // Δy_{t-(j+1)} = diff[t - (j+1)]
            xmat[(row, 2 + j)] = diff[t - (j + 1)];
        }
    }

    // OLS β̂ = (X'X)^-1 X'y, plus se(β̂_1) = σ_e * sqrt((X'X)^-1_{1,1})
    let xt = xmat.t();
    let xtx = xt.dot(&xmat);
    let xty = xt.dot(&yvec);
    let xtx_inv = invert_psd(&xtx)?;
    let beta_hat = xtx_inv.dot(&xty);
    // residuals
    let yhat = xmat.dot(&beta_hat);
    let mut sse = 0.0;
    for i in 0..n_eff {
        let r = yvec[i] - yhat[i];
        sse += r * r;
    }
    let dof = (n_eff as i64 - k as i64).max(1) as f64;
    let sigma2 = sse / dof;
    let var_beta1 = sigma2 * xtx_inv[(1, 1)];
    if var_beta1 <= 0.0 {
        return None;
    }
    Some(beta_hat[1] / var_beta1.sqrt())
}

/// Approximate one-sided p-value for the ADF t-statistic.
/// Uses interpolated MacKinnon (1996) critical values for a model with constant.
/// Returns probability that we'd see a t-stat this extreme under the null
/// (unit root). Low p ⇒ reject unit root ⇒ stationary.
pub fn adf_pvalue(t_stat: f64, n: usize) -> f64 {
    // MacKinnon critical values (n=large) for ADF with constant:
    // 1%: -3.43, 5%: -2.86, 10%: -2.57
    // Apply small-sample adjustment for n.
    let adj = if n < 50 {
        0.20
    } else if n < 100 {
        0.10
    } else if n < 250 {
        0.05
    } else {
        0.0
    };
    let cv_1 = -3.43 - adj;
    let cv_5 = -2.86 - adj;
    let cv_10 = -2.57 - adj;
    // Piecewise linear interpolation
    if t_stat <= cv_1 {
        // Beyond 1%; clamp at 0.005
        0.005
    } else if t_stat <= cv_5 {
        // Between 1% and 5%
        interp(t_stat, cv_1, cv_5, 0.01, 0.05)
    } else if t_stat <= cv_10 {
        interp(t_stat, cv_5, cv_10, 0.05, 0.10)
    } else if t_stat <= 0.0 {
        // 10% < p < 50%
        interp(t_stat, cv_10, 0.0, 0.10, 0.50)
    } else {
        // Positive t (no evidence for stationarity)
        let v = interp(t_stat, 0.0, 5.0, 0.50, 0.99);
        v.clamp(0.50, 0.99)
    }
}

fn interp(x: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    if (x1 - x0).abs() < 1e-12 {
        y0
    } else {
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }
}

/// Invert a symmetric positive-definite matrix via Gauss-Jordan.
/// Adequate for the small (2..6 column) design matrices used here.
fn invert_psd(m: &Array2<f64>) -> Option<Array2<f64>> {
    let n = m.nrows();
    if m.ncols() != n {
        return None;
    }
    let mut aug = Array2::<f64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = m[(i, j)];
        }
        aug[(i, n + i)] = 1.0;
    }
    // Forward elimination with partial pivoting
    for i in 0..n {
        // Pivot
        let mut max_row = i;
        let mut max_val = aug[(i, i)].abs();
        for r in i + 1..n {
            let v = aug[(r, i)].abs();
            if v > max_val {
                max_val = v;
                max_row = r;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        if max_row != i {
            for c in 0..2 * n {
                let tmp = aug[(i, c)];
                aug[(i, c)] = aug[(max_row, c)];
                aug[(max_row, c)] = tmp;
            }
        }
        // Normalize pivot row
        let pivot = aug[(i, i)];
        for c in 0..2 * n {
            aug[(i, c)] /= pivot;
        }
        // Eliminate other rows
        for r in 0..n {
            if r == i {
                continue;
            }
            let factor = aug[(r, i)];
            if factor.abs() < 1e-15 {
                continue;
            }
            for c in 0..2 * n {
                let v = aug[(i, c)] * factor;
                aug[(r, c)] -= v;
            }
        }
    }
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[(i, j)] = aug[(i, n + j)];
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_normal(seed: u64) -> impl FnMut() -> f64 {
        let mut s = if seed == 0 { 1u64 } else { seed };
        move || {
            // xorshift + Box-Muller
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
    fn psd_inverse_identity() {
        let m = Array2::from_shape_vec((3, 3), vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0]).unwrap();
        let inv = invert_psd(&m).unwrap();
        let id = m.dot(&inv);
        for i in 0..3 {
            for j in 0..3 {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((id[(i, j)] - target).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn engle_granger_finds_known_relationship() {
        // y = 0.5 + 1.5 * x + ε, ε ~ N(0, 0.1)
        let mut g = rng_normal(7);
        let x: Vec<f64> = (0..200).map(|i| i as f64 * 0.1 + g() * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|xi| 0.5 + 1.5 * xi + g() * 0.1).collect();
        let xa = Array1::from(x);
        let ya = Array1::from(y);
        let res = engle_granger(ya.view(), xa.view(), 1).unwrap();
        assert!((res.beta - 1.5).abs() < 0.1, "expected β≈1.5, got {}", res.beta);
        assert!((res.alpha - 0.5).abs() < 0.5);
        // residuals should be highly stationary
        assert!(res.p_value < 0.05, "expected stationary residuals, p={}", res.p_value);
    }

    #[test]
    fn adf_rejects_unit_root_for_stationary_series() {
        // AR(1) with ρ=0.3: very stationary
        let mut g = rng_normal(11);
        let mut y = vec![0.0_f64];
        for _ in 0..300 {
            let last = *y.last().unwrap();
            y.push(0.3 * last + g());
        }
        let arr = Array1::from(y);
        let t = adf_test_statistic(arr.view(), 1).unwrap();
        assert!(t < -2.0, "expected strongly negative t-stat, got {}", t);
        let p = adf_pvalue(t, 300);
        assert!(p < 0.10, "expected p<0.10 for stationary series, got {}", p);
    }

    #[test]
    fn adf_does_not_reject_random_walk() {
        // ρ=1 (random walk): NOT stationary
        let mut g = rng_normal(13);
        let mut y = vec![0.0_f64];
        for _ in 0..300 {
            let last = *y.last().unwrap();
            y.push(last + g());
        }
        let arr = Array1::from(y);
        let t = adf_test_statistic(arr.view(), 1).unwrap();
        let p = adf_pvalue(t, 300);
        assert!(p > 0.05, "expected p>0.05 for random walk, got p={} t={}", p, t);
    }
}
