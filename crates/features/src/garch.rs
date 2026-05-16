//! GARCH(1,1) volatility model with closed-form one-step MLE-ish updates.
//!
//! σ²_t = ω + α · r²_{t-1} + β · σ²_{t-1}
//!
//! Constraints:
//!   ω > 0,  α ≥ 0,  β ≥ 0,  α + β < 1   (stationarity)
//!
//! We expose:
//!   - `Garch11::new(ω, α, β)` — for known parameters.
//!   - `Garch11::fit_simple(returns, …)` — coarse grid search over (α, β) with
//!     ω determined by the long-run variance constraint. Full MLE would
//!     require numerical optimization; this is enough for vol-scaled
//!     position sizing where directional accuracy of σ matters more than
//!     parameter recovery.
//!
//! Reference: Bollerslev (1986); Hansen & Lunde (2005) compared GARCH(1,1) to
//! 330 competitors and found it hard to beat for daily return volatility.

#[derive(Copy, Clone, Debug)]
pub struct Garch11 {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
    sigma2: f64,
    last_return: Option<f64>,
}

impl Garch11 {
    pub fn new(omega: f64, alpha: f64, beta: f64) -> Self {
        // Initialize σ²_0 to the unconditional variance: ω / (1 - α - β).
        let sigma2 = if alpha + beta < 1.0 {
            omega / (1.0 - alpha - beta).max(1e-9)
        } else {
            omega.max(1e-9)
        };
        Self {
            omega,
            alpha,
            beta,
            sigma2,
            last_return: None,
        }
    }

    /// Push a new return; returns the updated conditional variance σ²_t.
    pub fn update(&mut self, r: f64) -> f64 {
        let r2 = if let Some(last) = self.last_return {
            last * last
        } else {
            r * r
        };
        self.sigma2 = self.omega + self.alpha * r2 + self.beta * self.sigma2;
        self.last_return = Some(r);
        self.sigma2
    }

    pub fn variance(&self) -> f64 {
        self.sigma2
    }
    pub fn vol(&self) -> f64 {
        self.sigma2.max(0.0).sqrt()
    }

    /// Coarse fit on a returns series. Grid-searches (α, β) ∈ [0.02, 0.20] ×
    /// [0.7, 0.97] and picks the (α, β, ω) minimizing the negative log-
    /// likelihood `Σ [ ln σ²_t + r²_t / σ²_t ]`.
    ///
    /// ω is pinned to the unconditional variance: ω = σ̄² · (1 - α - β),
    /// where σ̄² is the sample variance of returns.
    pub fn fit_simple(returns: &[f64]) -> Option<Self> {
        if returns.len() < 50 {
            return None;
        }
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let sample_var = returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        if sample_var <= 0.0 || !sample_var.is_finite() {
            return None;
        }

        let alphas = [0.02_f64, 0.05, 0.08, 0.12, 0.15, 0.20];
        let betas = [0.70_f64, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97];

        let mut best: Option<(f64, Garch11)> = None;
        for &a in &alphas {
            for &b in &betas {
                if a + b >= 0.999 {
                    continue;
                }
                let omega = sample_var * (1.0 - a - b);
                if omega <= 0.0 {
                    continue;
                }
                let mut g = Garch11::new(omega, a, b);
                let mut nll = 0.0_f64;
                for &r in returns {
                    g.update(r);
                    let s2 = g.sigma2.max(1e-12);
                    nll += s2.ln() + r * r / s2;
                }
                if nll.is_finite() {
                    if best.as_ref().map_or(true, |(b_nll, _)| nll < *b_nll) {
                        best = Some((nll, Garch11::new(omega, a, b)));
                    }
                }
            }
        }
        best.map(|(_, g)| g)
    }

    /// Long-run unconditional variance σ̄² = ω / (1 - α - β). NaN if non-stationary.
    pub fn long_run_variance(&self) -> f64 {
        if self.alpha + self.beta < 1.0 {
            self.omega / (1.0 - self.alpha - self.beta)
        } else {
            f64::NAN
        }
    }
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
    fn responds_to_volatility_shock() {
        // Calm returns → sustained vol shock → vol must rise.
        let mut g = Garch11::new(0.00001, 0.15, 0.80);
        let mut gen = rng_normal(7);
        for _ in 0..300 {
            g.update(0.001 * gen());
        }
        let calm_vol = g.vol();
        // Sustained 100× shocks for long enough to dominate β·σ²_{t-1}
        for _ in 0..50 {
            g.update(0.10 * gen());
        }
        let shocked_vol = g.vol();
        assert!(
            shocked_vol > calm_vol,
            "expected vol to rise after shock: {} → {}",
            calm_vol,
            shocked_vol
        );
        // Multiplicative jump expected to be material
        assert!(
            shocked_vol / calm_vol > 1.5,
            "vol shock too small: {}× ({} → {})",
            shocked_vol / calm_vol,
            calm_vol,
            shocked_vol
        );
    }

    #[test]
    fn long_run_variance_is_sensible() {
        let g = Garch11::new(0.0001, 0.10, 0.85);
        let lrv = g.long_run_variance();
        // ω / (1 - α - β) = 0.0001 / 0.05 = 0.002
        assert!((lrv - 0.002).abs() < 1e-9);
    }

    #[test]
    fn fit_picks_reasonable_params() {
        let mut g = rng_normal(99);
        // 500 i.i.d. returns ~ N(0, 0.01²); GARCH should fit low α, low β
        // (since there's no actual GARCH structure)
        let rets: Vec<f64> = (0..500).map(|_| 0.01 * g()).collect();
        let fit = Garch11::fit_simple(&rets).unwrap();
        // With pure noise, alpha and beta are not well-identified but the
        // unconditional variance should match sample variance roughly.
        let lrv = fit.long_run_variance();
        assert!(lrv > 0.0 && lrv < 0.001, "lrv={}", lrv);
    }
}
