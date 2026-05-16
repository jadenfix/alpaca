//! 2-state Gaussian Hidden Markov Model for market regime detection.
//!
//! States: { low-vol calm / high-vol turbulent }. Each emits Gaussian returns
//! with state-specific mean and variance.
//!
//! Inference: online Viterbi-style "filtered" posterior — at each step we
//! update the posterior P(state_t | r_1..r_t) using the forward (alpha)
//! recursion. This is exact for a 2-state model and O(1) per observation.
//!
//! Training: we fit by EM (Baum-Welch) on a batch of returns. The
//! implementation is small (≤ 30 iterations); fine for offline calibration
//! on a few thousand bars.
//!
//! Reference: Rabiner (1989), "A Tutorial on Hidden Markov Models".

use std::f64::consts::PI;

#[derive(Copy, Clone, Debug)]
pub struct GaussianState {
    pub mean: f64,
    pub var: f64,
}

impl GaussianState {
    pub fn pdf(&self, x: f64) -> f64 {
        let std = self.var.max(1e-12).sqrt();
        let z = (x - self.mean) / std;
        (-0.5 * z * z).exp() / ((2.0 * PI).sqrt() * std)
    }
}

#[derive(Clone, Debug)]
pub struct TwoStateGaussianHmm {
    /// Initial distribution π[0..1].
    pub pi: [f64; 2],
    /// Transition matrix A[i][j] = P(s_t=j | s_{t-1}=i).
    pub a: [[f64; 2]; 2],
    /// Emission parameters for each state.
    pub states: [GaussianState; 2],
    /// Filtered posterior P(s_t | r_1..r_t).
    pub alpha: [f64; 2],
    initialized: bool,
}

impl TwoStateGaussianHmm {
    /// Sensible default for equities: state 0 = calm (low vol), state 1 = turbulent.
    pub fn default_equity() -> Self {
        Self {
            pi: [0.7, 0.3],
            a: [
                [0.98, 0.02], // calm tends to stay calm
                [0.05, 0.95], // turbulent has slightly higher exit rate
            ],
            states: [
                GaussianState { mean: 0.0001, var: 0.0001 },  // 1% daily vol-ish (low)
                GaussianState { mean: -0.0005, var: 0.0009 }, // 3% daily vol-ish (high), slight neg drift
            ],
            alpha: [0.7, 0.3],
            initialized: false,
        }
    }

    /// One-step forward filtering update. Returns posterior P(state_t).
    pub fn update(&mut self, r: f64) -> [f64; 2] {
        if !self.initialized {
            // Initialize alpha from prior
            let p0 = self.pi[0] * self.states[0].pdf(r);
            let p1 = self.pi[1] * self.states[1].pdf(r);
            let s = (p0 + p1).max(1e-300);
            self.alpha = [p0 / s, p1 / s];
            self.initialized = true;
            return self.alpha;
        }
        // Predict
        let pred0 = self.alpha[0] * self.a[0][0] + self.alpha[1] * self.a[1][0];
        let pred1 = self.alpha[0] * self.a[0][1] + self.alpha[1] * self.a[1][1];
        // Update
        let lik0 = self.states[0].pdf(r);
        let lik1 = self.states[1].pdf(r);
        let p0 = pred0 * lik0;
        let p1 = pred1 * lik1;
        let s = (p0 + p1).max(1e-300);
        self.alpha = [p0 / s, p1 / s];
        self.alpha
    }

    /// Probability of being in the "turbulent" state (state index 1).
    pub fn p_turbulent(&self) -> f64 {
        self.alpha[1]
    }

    /// Most likely current state.
    pub fn argmax_state(&self) -> usize {
        if self.alpha[1] > self.alpha[0] { 1 } else { 0 }
    }

    /// Fit by Baum-Welch (EM) on a batch of returns. Re-initializes from
    /// reasonable priors and runs until log-likelihood improvement < tol or
    /// max_iter iterations elapsed.
    pub fn fit_baum_welch(&mut self, returns: &[f64], max_iter: usize, tol: f64) {
        let n = returns.len();
        if n < 30 {
            return;
        }
        let mut last_ll = f64::NEG_INFINITY;
        for _iter in 0..max_iter {
            // Forward
            let mut alpha = vec![[0.0_f64; 2]; n];
            let mut scale = vec![0.0_f64; n];
            // t=0
            let l0_0 = self.states[0].pdf(returns[0]);
            let l0_1 = self.states[1].pdf(returns[0]);
            alpha[0][0] = self.pi[0] * l0_0;
            alpha[0][1] = self.pi[1] * l0_1;
            scale[0] = (alpha[0][0] + alpha[0][1]).max(1e-300);
            alpha[0][0] /= scale[0];
            alpha[0][1] /= scale[0];
            for t in 1..n {
                for j in 0..2 {
                    let mut s = 0.0;
                    for i in 0..2 {
                        s += alpha[t - 1][i] * self.a[i][j];
                    }
                    alpha[t][j] = s * self.states[j].pdf(returns[t]);
                }
                scale[t] = (alpha[t][0] + alpha[t][1]).max(1e-300);
                alpha[t][0] /= scale[t];
                alpha[t][1] /= scale[t];
            }
            let ll: f64 = scale.iter().map(|s| s.ln()).sum();
            let converged = (ll - last_ll).abs() < tol;
            last_ll = ll;
            if converged {
                self.alpha = alpha[n - 1];
                break;
            }
            // Backward
            let mut beta = vec![[0.0_f64; 2]; n];
            beta[n - 1] = [1.0, 1.0];
            for t in (0..n - 1).rev() {
                for i in 0..2 {
                    let mut s = 0.0;
                    for j in 0..2 {
                        s += self.a[i][j] * self.states[j].pdf(returns[t + 1]) * beta[t + 1][j];
                    }
                    beta[t][i] = s / scale[t + 1];
                }
            }
            // Gamma (posterior) and xi (joint)
            let mut gamma = vec![[0.0_f64; 2]; n];
            let mut xi = vec![[[0.0_f64; 2]; 2]; n - 1];
            for t in 0..n {
                let denom = (alpha[t][0] * beta[t][0] + alpha[t][1] * beta[t][1]).max(1e-300);
                for i in 0..2 {
                    gamma[t][i] = alpha[t][i] * beta[t][i] / denom;
                }
            }
            for t in 0..n - 1 {
                let mut denom = 0.0;
                for i in 0..2 {
                    for j in 0..2 {
                        denom += alpha[t][i] * self.a[i][j] * self.states[j].pdf(returns[t + 1])
                            * beta[t + 1][j];
                    }
                }
                let denom = denom.max(1e-300);
                for i in 0..2 {
                    for j in 0..2 {
                        xi[t][i][j] = alpha[t][i] * self.a[i][j] * self.states[j].pdf(returns[t + 1])
                            * beta[t + 1][j] / denom;
                    }
                }
            }
            // Re-estimate
            self.pi = gamma[0];
            for i in 0..2 {
                let denom: f64 = (0..n - 1).map(|t| gamma[t][i]).sum::<f64>().max(1e-300);
                for j in 0..2 {
                    let num: f64 = (0..n - 1).map(|t| xi[t][i][j]).sum();
                    self.a[i][j] = num / denom;
                }
            }
            for j in 0..2 {
                let denom: f64 = (0..n).map(|t| gamma[t][j]).sum::<f64>().max(1e-300);
                let num_mean: f64 = (0..n).map(|t| gamma[t][j] * returns[t]).sum();
                let mu = num_mean / denom;
                let num_var: f64 = (0..n).map(|t| gamma[t][j] * (returns[t] - mu).powi(2)).sum();
                let var = (num_var / denom).max(1e-12);
                self.states[j].mean = mu;
                self.states[j].var = var;
            }
            self.alpha = alpha[n - 1];
        }
        // Enforce ordering: state 0 = lower-variance (calm), state 1 = higher
        if self.states[0].var > self.states[1].var {
            self.states.swap(0, 1);
            // Swap rows AND columns of A
            self.a = [
                [self.a[1][1], self.a[1][0]],
                [self.a[0][1], self.a[0][0]],
            ];
            self.pi.swap(0, 1);
            self.alpha.swap(0, 1);
        }
        self.initialized = true;
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
    fn online_filter_assigns_high_prob_to_calm_under_quiet_returns() {
        let mut h = TwoStateGaussianHmm::default_equity();
        let mut g = rng_normal(7);
        for _ in 0..200 {
            h.update(0.0001 * g()); // very quiet
        }
        // Under low vol returns, should mostly be in calm state
        assert!(
            h.p_turbulent() < 0.5,
            "expected p(turbulent) < 0.5, got {}",
            h.p_turbulent()
        );
    }

    #[test]
    fn online_filter_responds_to_vol_burst() {
        let mut h = TwoStateGaussianHmm::default_equity();
        let mut g = rng_normal(11);
        for _ in 0..200 {
            h.update(0.0001 * g());
        }
        let p_calm_before = h.alpha[0];
        for _ in 0..50 {
            h.update(0.05 * g()); // turbulent: 5% bars
        }
        let p_calm_after = h.alpha[0];
        assert!(
            p_calm_after < p_calm_before,
            "calm prob should drop after vol burst: {} -> {}",
            p_calm_before,
            p_calm_after
        );
    }

    #[test]
    fn baum_welch_recovers_two_regimes() {
        // Synthesize alternating regimes: 200 quiet, 200 noisy, 200 quiet, 200 noisy.
        let mut g = rng_normal(13);
        let mut rets = Vec::with_capacity(800);
        for _ in 0..200 { rets.push(0.0005 * g()); }
        for _ in 0..200 { rets.push(0.05 * g()); }
        for _ in 0..200 { rets.push(0.0005 * g()); }
        for _ in 0..200 { rets.push(0.05 * g()); }
        let mut h = TwoStateGaussianHmm::default_equity();
        h.fit_baum_welch(&rets, 30, 1e-4);
        // State 0 should be the quiet regime; var should be much smaller than state 1.
        assert!(
            h.states[1].var > h.states[0].var * 10.0,
            "state vars not separated: {:?}",
            h.states
        );
    }
}
