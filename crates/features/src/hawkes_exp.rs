//! Hawkes exponential-kernel MLE — Ogata (1981) recursion.
//!
//! A Hawkes process with exponential kernel has intensity
//!     λ(t) = μ + α · Σ_{t_i < t} exp(−β · (t − t_i))
//!
//! Each event raises future intensity by α; the bump decays at rate β.
//! The "branching ratio" n = α/β is the expected number of descendants
//! per ancestor event; for stationarity we require n < 1.
//!
//! Ogata's 1981 recursion computes the log-likelihood in O(n) by storing
//!     R(t_i) = Σ_{j<i} exp(−β·(t_i − t_j))
//! which satisfies R(t_i) = exp(−β·(t_i − t_{i−1})) · (1 + R(t_{i−1})).
//!
//! Public functions:
//!   - `hawkes_exp_loglik(times, μ, α, β) -> f64`
//!   - `fit_hawkes_exp(times) -> Option<HawkesParams>` via Nelder-Mead on
//!     log-parameters (positivity is automatic; stationarity is enforced
//!     by a soft penalty).

#[derive(Clone, Copy, Debug)]
pub struct HawkesParams {
    pub mu: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl HawkesParams {
    pub fn branching_ratio(&self) -> f64 {
        if self.beta <= 0.0 { f64::INFINITY } else { self.alpha / self.beta }
    }
    pub fn half_life(&self) -> f64 {
        if self.beta <= 0.0 { f64::INFINITY } else { 2.0_f64.ln() / self.beta }
    }
}

/// Hawkes exponential-kernel log-likelihood via Ogata's recursion. `times`
/// must be strictly non-decreasing event times; we treat T = times[n−1].
pub fn hawkes_exp_loglik(times: &[f64], mu: f64, alpha: f64, beta: f64) -> f64 {
    if times.is_empty() || mu <= 0.0 || alpha < 0.0 || beta <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let n = times.len();
    // First event: R(t_0) = 0, λ(t_0) = μ, ∫_0^{t_0} μ ds = μ·t_0.
    let mut r = 0.0_f64;
    let mut log_sum = mu.ln();
    let mut integral = mu * times[0];
    for i in 1..n {
        let dt = times[i] - times[i - 1];
        if dt < 0.0 {
            return f64::NEG_INFINITY;
        }
        let exp_term = (-beta * dt).exp();
        // Compensator increment over (t_{i−1}, t_i]:
        //   μ·dt + (α/β) · (1 − e^{−β·dt}) · (1 + R(t_{i−1}))
        // (1 + R_prev) appears because R updates to exp_term·(1+R_prev) AFTER
        // this step; here we still have access to the OLD R = R_prev.
        let r_prev_plus_one = 1.0 + r;
        integral += mu * dt + (alpha / beta) * (1.0 - exp_term) * r_prev_plus_one;
        // Now update R(t_i) and accumulate the log-intensity contribution.
        r = exp_term * r_prev_plus_one;
        let lambda_i = mu + alpha * r;
        if lambda_i <= 0.0 {
            return f64::NEG_INFINITY;
        }
        log_sum += lambda_i.ln();
    }
    log_sum - integral
}

/// Fit (μ, α, β) by Nelder-Mead on log-parameters with a soft stationarity
/// penalty (α/β < 1). Returns `None` if the optimizer fails to find a
/// stationary fit.
pub fn fit_hawkes_exp(times: &[f64]) -> Option<HawkesParams> {
    if times.len() < 5 {
        return None;
    }
    let n = times.len() as f64;
    let span = times[times.len() - 1] - times[0];
    if span <= 0.0 {
        return None;
    }
    let mean_iat = span / n.max(1.0);
    let mu0 = (n / span) * 0.5;
    let beta0 = 2.0 / mean_iat.max(1e-9);
    let alpha0 = 0.5 * beta0;
    let theta0 = [mu0.ln(), alpha0.ln(), beta0.ln()];

    let nll = |theta: [f64; 3]| -> f64 {
        let mu = theta[0].exp();
        let alpha = theta[1].exp();
        let beta = theta[2].exp();
        let n_ratio = alpha / beta;
        if n_ratio >= 1.0 {
            // Soft barrier so the optimizer is pushed back inside the stationary region.
            return 1e8 + 1e6 * (n_ratio - 1.0);
        }
        -hawkes_exp_loglik(times, mu, alpha, beta)
    };

    let opt = nelder_mead(theta0, nll, 0.10, 1e-5, 800);
    let mu = opt[0].exp();
    let alpha = opt[1].exp();
    let beta = opt[2].exp();
    if !(mu.is_finite() && alpha.is_finite() && beta.is_finite()) {
        return None;
    }
    if alpha / beta >= 1.0 {
        return None;
    }
    Some(HawkesParams { mu, alpha, beta })
}

/// Minimal Nelder-Mead simplex optimizer for a 3-D scalar function.
fn nelder_mead<F>(start: [f64; 3], f: F, step: f64, tol: f64, max_iter: usize) -> [f64; 3]
where
    F: Fn([f64; 3]) -> f64,
{
    let mut simplex: Vec<[f64; 3]> = Vec::with_capacity(4);
    simplex.push(start);
    for i in 0..3 {
        let mut v = start;
        v[i] += step;
        simplex.push(v);
    }
    let mut fvals: Vec<f64> = simplex.iter().map(|&x| f(x)).collect();

    for _it in 0..max_iter {
        let mut idx: Vec<usize> = (0..4).collect();
        idx.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap_or(std::cmp::Ordering::Equal));
        let simplex_sorted: Vec<[f64; 3]> = idx.iter().map(|&i| simplex[i]).collect();
        let fvals_sorted: Vec<f64> = idx.iter().map(|&i| fvals[i]).collect();
        simplex = simplex_sorted;
        fvals = fvals_sorted;

        let mut max_diff = 0.0_f64;
        for i in 0..3 {
            let span = simplex.iter().map(|v| v[i]).fold(f64::NEG_INFINITY, f64::max)
                - simplex.iter().map(|v| v[i]).fold(f64::INFINITY, f64::min);
            max_diff = max_diff.max(span.abs());
        }
        if max_diff < tol {
            break;
        }

        let mut centroid = [0.0_f64; 3];
        for v in simplex.iter().take(3) {
            for j in 0..3 {
                centroid[j] += v[j];
            }
        }
        for c in centroid.iter_mut() {
            *c /= 3.0;
        }

        let worst = simplex[3];
        let mut reflected = [0.0_f64; 3];
        for j in 0..3 {
            reflected[j] = centroid[j] + (centroid[j] - worst[j]);
        }
        let f_ref = f(reflected);

        if f_ref < fvals[0] {
            let mut expanded = [0.0_f64; 3];
            for j in 0..3 {
                expanded[j] = centroid[j] + 2.0 * (centroid[j] - worst[j]);
            }
            let f_exp = f(expanded);
            if f_exp < f_ref {
                simplex[3] = expanded;
                fvals[3] = f_exp;
            } else {
                simplex[3] = reflected;
                fvals[3] = f_ref;
            }
        } else if f_ref < fvals[2] {
            simplex[3] = reflected;
            fvals[3] = f_ref;
        } else {
            let mut contracted = [0.0_f64; 3];
            for j in 0..3 {
                contracted[j] = centroid[j] - 0.5 * (centroid[j] - worst[j]);
            }
            let f_con = f(contracted);
            if f_con < fvals[3] {
                simplex[3] = contracted;
                fvals[3] = f_con;
            } else {
                for i in 1..4 {
                    for j in 0..3 {
                        simplex[i][j] = simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]);
                    }
                    fvals[i] = f(simplex[i]);
                }
            }
        }
    }
    simplex[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulate a Hawkes process by Ogata's thinning algorithm.
    fn simulate_hawkes(mu: f64, alpha: f64, beta: f64, t_max: f64, seed: u64) -> Vec<f64> {
        let mut state = if seed == 0 { 1u64 } else { seed };
        let mut next_u01 = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 1.0)
                / ((1u64 << 53) as f64 + 1.0)
        };
        let mut events: Vec<f64> = Vec::new();
        let mut t = 0.0_f64;
        loop {
            let mut lambda_t = mu;
            for &t_i in &events {
                lambda_t += alpha * (-beta * (t - t_i)).exp();
            }
            let u = next_u01();
            let dt = -u.ln() / lambda_t;
            t += dt;
            if t > t_max { break; }
            let mut lambda_new = mu;
            for &t_i in &events {
                lambda_new += alpha * (-beta * (t - t_i)).exp();
            }
            let v = next_u01();
            if v <= lambda_new / lambda_t {
                events.push(t);
            }
        }
        events
    }

    #[test]
    fn loglik_finite_on_valid_input() {
        let times: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let ll = hawkes_exp_loglik(&times, 1.0, 0.3, 1.5);
        assert!(ll.is_finite(), "log-lik not finite: {ll}");
    }

    #[test]
    fn loglik_rejects_invalid_params() {
        let times: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        assert_eq!(hawkes_exp_loglik(&times, -1.0, 0.3, 1.5), f64::NEG_INFINITY);
        assert_eq!(hawkes_exp_loglik(&times, 1.0, -0.1, 1.5), f64::NEG_INFINITY);
        assert_eq!(hawkes_exp_loglik(&times, 1.0, 0.3, -1.0), f64::NEG_INFINITY);
    }

    #[test]
    fn loglik_increases_when_params_match_data() {
        // Simulate with known params, then verify log-lik at TRUE params >
        // log-lik at far-off params.
        let times = simulate_hawkes(0.5, 0.6, 1.5, 2000.0, 7);
        assert!(times.len() > 50);
        let ll_true = hawkes_exp_loglik(&times, 0.5, 0.6, 1.5);
        let ll_far = hawkes_exp_loglik(&times, 5.0, 0.05, 10.0);
        assert!(
            ll_true > ll_far,
            "log-lik should be higher at true params: true={ll_true}, far={ll_far}"
        );
    }

    #[test]
    fn fit_returns_stationary_params() {
        // Simulate a clearly stationary process (n=0.4) and verify the fit
        // returns a stationary parameter set.
        let times = simulate_hawkes(0.5, 0.6, 1.5, 3000.0, 11);
        assert!(times.len() > 50, "too few events: {}", times.len());
        let fit = fit_hawkes_exp(&times).expect("fit should succeed");
        assert!(
            fit.branching_ratio() < 1.0,
            "fit must be stationary, got n={}",
            fit.branching_ratio()
        );
        // Half-life and intensity sanity.
        assert!(fit.half_life() > 0.0 && fit.half_life().is_finite());
        assert!(fit.mu > 0.0);
        assert!(fit.alpha >= 0.0);
        assert!(fit.beta > 0.0);
    }
}
