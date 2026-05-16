//! Ornstein-Uhlenbeck calibration.
//!
//! For a series x_t assumed to follow the discrete OU process
//!     x_{t+1} = a + b·x_t + ε_t,    ε_t ~ N(0, σ²)
//! we have the continuous-time mapping
//!     θ (mean reversion speed) = -ln(b) / Δt
//!     μ (long-run mean)        = a / (1 - b)
//!     σ_OU (instantaneous vol) = σ · sqrt(-2 ln(b) / (Δt · (1 - b²)))
//!     half-life T_½            = ln(2) / θ
//!
//! Reference: Avellaneda & Lee 2010 §3, "Statistical Arbitrage in the US
//! Equities Market".

use ndarray::ArrayView1;

#[derive(Copy, Clone, Debug)]
pub struct OuParams {
    pub theta: f64,    // mean reversion speed (per `dt` unit)
    pub mu: f64,       // long-run mean
    pub sigma: f64,    // instantaneous vol
    pub half_life: f64, // bars
}

/// Fit OU parameters by OLS on x_{t+1} = a + b·x_t + ε_t.
/// `dt` is the per-bar time step in the same units you want θ measured in
/// (e.g., dt=1.0 → θ per bar; dt=1/252 → θ per year for daily bars).
/// Returns `None` if the series is too short, b is non-mean-reverting (b ≥ 1
/// or b ≤ 0), or the OLS is degenerate.
pub fn fit_ou(x: ArrayView1<f64>, dt: f64) -> Option<OuParams> {
    let n = x.len();
    if n < 10 || dt <= 0.0 {
        return None;
    }
    // OLS on (lag, current)
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let m = (n - 1) as f64;
    for i in 0..n - 1 {
        let xi = x[i];
        let yi = x[i + 1];
        sx += xi;
        sy += yi;
        sxx += xi * xi;
        sxy += xi * yi;
    }
    let mean_x = sx / m;
    let mean_y = sy / m;
    let denom = sxx - m * mean_x * mean_x;
    if denom.abs() < 1e-12 {
        return None;
    }
    let b = (sxy - m * mean_x * mean_y) / denom;
    let a = mean_y - b * mean_x;
    // For OU, need 0 < b < 1.
    if !(b > 0.0 && b < 1.0) {
        return None;
    }
    // Residual variance
    let mut sse = 0.0;
    for i in 0..n - 1 {
        let pred = a + b * x[i];
        let r = x[i + 1] - pred;
        sse += r * r;
    }
    let dof = (m - 2.0).max(1.0);
    let s2 = sse / dof;
    let theta = -b.ln() / dt;
    let mu = a / (1.0 - b);
    let sigma = (s2 * (-2.0 * b.ln() / (dt * (1.0 - b * b)))).max(0.0).sqrt();
    let half_life = std::f64::consts::LN_2 / theta;
    Some(OuParams {
        theta,
        mu,
        sigma,
        half_life,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

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
    fn recovers_known_ou_parameters() {
        // Simulate OU with θ=2, μ=10, σ=0.5, dt=1/100
        let theta_true: f64 = 2.0;
        let mu_true: f64 = 10.0;
        let sigma_true: f64 = 0.5;
        let dt: f64 = 1.0 / 100.0;
        let b_true: f64 = (-theta_true * dt).exp();
        // discrete OU std for innovations:
        //   σ_ε = σ · sqrt((1 - b²) / (2 θ))
        let sigma_eps: f64 = sigma_true * ((1.0 - b_true * b_true) / (2.0 * theta_true)).sqrt();
        let mut g = rng_normal(42);
        let n = 2_000;
        let mut x = vec![mu_true];
        for _ in 1..n {
            let last = *x.last().unwrap();
            let next = mu_true + b_true * (last - mu_true) + sigma_eps * g();
            x.push(next);
        }
        let arr = Array1::from(x);
        let p = fit_ou(arr.view(), dt).unwrap();
        // Theta recovery within 30% with 2000 samples is reasonable
        assert!(
            (p.theta - theta_true).abs() / theta_true < 0.3,
            "theta off: {} vs {}",
            p.theta,
            theta_true
        );
        assert!((p.mu - mu_true).abs() < 0.5);
        assert!(p.half_life > 0.0);
    }

    #[test]
    fn fitted_random_walk_has_extreme_half_life() {
        // Random walk has b ≈ 1, so even if fit_ou returns Some, the half-life
        // must be extreme (>> reasonable trading horizons). Strategies should
        // use this as a filter: skip pairs with half_life > N bars.
        let mut g = rng_normal(99);
        let n = 1_000;
        let mut x = vec![0.0_f64];
        for _ in 1..n {
            let last = *x.last().unwrap();
            x.push(last + g());
        }
        let arr = Array1::from(x);
        match fit_ou(arr.view(), 1.0) {
            None => {} // fine — b drifted outside (0, 1)
            Some(p) => {
                // For a random walk, theta should be near zero ⇒ half-life huge
                assert!(
                    p.half_life > 100.0 || p.theta < 0.05,
                    "random walk produced unreasonably small half-life {:.2} (theta={})",
                    p.half_life,
                    p.theta
                );
            }
        }
    }
}
