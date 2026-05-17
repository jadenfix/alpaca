//! Monte Carlo path simulator under a Markov-switching regime model.
//!
//! Given:
//!   - a fitted regime model (transition matrix A, emission means μ, vols σ)
//!   - a current starting regime s_0
//!   - N paths × T steps
//! Simulate forward log-return paths, returning the per-step return matrix
//! and the regime path matrix. Used by the strategy recommender to estimate
//! P&L distributions and tail risk under each regime.
//!
//! Reference: Hamilton (1989); standard Markov-chain Monte Carlo.

use crate::markov_switching::{Regime, ThreeStateMarkov};

pub struct McConfig {
    pub n_paths: usize,
    pub n_steps: usize,
    pub seed: u64,
    pub start_regime_idx: usize,
}

#[derive(Clone, Debug)]
pub struct McResult {
    /// (path × step) cumulative log-return.
    pub cumulative_log_ret: Vec<Vec<f64>>,
    /// (path × step) integer regime index.
    pub regime_path: Vec<Vec<u8>>,
    /// Terminal return per path.
    pub terminals: Vec<f64>,
}

impl McResult {
    /// Sorted terminal log-returns; useful for percentile queries.
    pub fn percentile(&self, q: f64) -> f64 {
        let mut t = self.terminals.clone();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if t.is_empty() {
            return 0.0;
        }
        let idx = ((q.clamp(0.0, 1.0) * (t.len() - 1) as f64).round() as usize).min(t.len() - 1);
        t[idx]
    }

    pub fn mean(&self) -> f64 {
        if self.terminals.is_empty() {
            0.0
        } else {
            self.terminals.iter().sum::<f64>() / self.terminals.len() as f64
        }
    }

    pub fn fraction_of_paths(&self, predicate: impl Fn(f64) -> bool) -> f64 {
        if self.terminals.is_empty() {
            return 0.0;
        }
        let n = self.terminals.iter().filter(|&&t| predicate(t)).count();
        n as f64 / self.terminals.len() as f64
    }
}

/// Run `cfg.n_paths` Monte Carlo paths over `cfg.n_steps` steps, starting in
/// `cfg.start_regime_idx`.
pub fn simulate(model: &ThreeStateMarkov, cfg: &McConfig) -> McResult {
    let mut rng = XorShift64::new(cfg.seed);
    let mut cumulative_log_ret = Vec::with_capacity(cfg.n_paths);
    let mut regime_path = Vec::with_capacity(cfg.n_paths);
    let mut terminals = Vec::with_capacity(cfg.n_paths);
    for _ in 0..cfg.n_paths {
        let mut path_ret = Vec::with_capacity(cfg.n_steps);
        let mut path_reg = Vec::with_capacity(cfg.n_steps);
        let mut cum = 0.0_f64;
        let mut s = cfg.start_regime_idx % 3;
        for _ in 0..cfg.n_steps {
            // Sample next state
            s = sample_next_state(model, s, &mut rng);
            // Sample return ~ N(μ_s, σ_s²)
            let mu = model.states[s].mean;
            let sigma = model.states[s].var.max(0.0).sqrt();
            let r = mu + sigma * rng.next_normal();
            cum += r;
            path_ret.push(cum);
            path_reg.push(s as u8);
        }
        terminals.push(cum);
        cumulative_log_ret.push(path_ret);
        regime_path.push(path_reg);
    }
    McResult {
        cumulative_log_ret,
        regime_path,
        terminals,
    }
}

/// Fraction of simulated time spent in each labeled regime.
pub fn regime_occupancy(model: &ThreeStateMarkov, result: &McResult) -> [(Regime, f64); 3] {
    let mut counts = [0usize; 3];
    let mut total = 0usize;
    for path in &result.regime_path {
        for &r in path {
            counts[r as usize] += 1;
            total += 1;
        }
    }
    [
        (model.regime_labels[0], counts[0] as f64 / total.max(1) as f64),
        (model.regime_labels[1], counts[1] as f64 / total.max(1) as f64),
        (model.regime_labels[2], counts[2] as f64 / total.max(1) as f64),
    ]
}

fn sample_next_state(model: &ThreeStateMarkov, from: usize, rng: &mut XorShift64) -> usize {
    let u = rng.next_uniform();
    let mut cum = 0.0;
    for j in 0..3 {
        cum += model.a[from][j];
        if u < cum {
            return j;
        }
    }
    2
}

pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEADBEEF } else { seed },
        }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn next_uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulate_produces_correct_shape() {
        let m = ThreeStateMarkov::default_equity();
        let r = simulate(
            &m,
            &McConfig {
                n_paths: 100,
                n_steps: 50,
                seed: 7,
                start_regime_idx: 0,
            },
        );
        assert_eq!(r.cumulative_log_ret.len(), 100);
        assert_eq!(r.regime_path.len(), 100);
        assert_eq!(r.terminals.len(), 100);
        for path in &r.cumulative_log_ret {
            assert_eq!(path.len(), 50);
        }
    }

    #[test]
    fn percentile_bounds_make_sense() {
        let m = ThreeStateMarkov::default_equity();
        let r = simulate(
            &m,
            &McConfig {
                n_paths: 500,
                n_steps: 100,
                seed: 13,
                start_regime_idx: 0,
            },
        );
        let p10 = r.percentile(0.10);
        let p50 = r.percentile(0.50);
        let p90 = r.percentile(0.90);
        assert!(p10 <= p50 && p50 <= p90, "percentile ordering: {} {} {}", p10, p50, p90);
    }

    #[test]
    fn regime_occupancy_sums_to_one() {
        let m = ThreeStateMarkov::default_equity();
        let r = simulate(
            &m,
            &McConfig {
                n_paths: 200,
                n_steps: 50,
                seed: 7,
                start_regime_idx: 0,
            },
        );
        let occ = regime_occupancy(&m, &r);
        let s: f64 = occ.iter().map(|(_, p)| p).sum();
        assert!((s - 1.0).abs() < 1e-9);
    }

    // ── convergence tests below would fail on a "fixed N(0,1)" simulator ──

    /// Construct a model with KNOWN emission means/variances and verify
    /// the empirical mean and variance of all per-step returns matches
    /// (within MC tolerance) the regime-occupancy-weighted theoretical
    /// values. A simulator that ignored the regime model entirely (e.g.
    /// emitted fixed N(0, 1)) would fail this test.
    #[test]
    fn simulator_honors_known_emission_distributions() {
        // Build a model with three clearly-separated emission distributions.
        let mut m = ThreeStateMarkov::default_equity();
        m.states[0] = crate::markov_switching::GaussianState { mean:  0.0010, var: 1.0e-5 };
        m.states[1] = crate::markov_switching::GaussianState { mean: -0.0020, var: 6.0e-4 };
        m.states[2] = crate::markov_switching::GaussianState { mean:  0.0000, var: 1.5e-4 };
        let n_paths = 1000usize;
        let n_steps = 200usize;
        let r = simulate(&m, &McConfig {
            n_paths, n_steps, seed: 1234, start_regime_idx: 0,
        });
        // Empirical per-step mean and variance (all paths × steps).
        let mut all_steps: Vec<f64> = Vec::with_capacity(n_paths * n_steps);
        for (path_idx, path) in r.cumulative_log_ret.iter().enumerate() {
            let _ = path_idx;
            let mut prev = 0.0;
            for &cum in path {
                all_steps.push(cum - prev);
                prev = cum;
            }
        }
        let n_total = all_steps.len() as f64;
        let mean_emp = all_steps.iter().sum::<f64>() / n_total;
        let var_emp = all_steps.iter().map(|x| (x - mean_emp).powi(2)).sum::<f64>() / n_total;
        // Theoretical expectation = Σ_i occupancy_i * μ_i.
        let occ = regime_occupancy(&m, &r);
        let mut mean_theo = 0.0;
        let mut var_theo = 0.0;
        for (i, (_, p)) in occ.iter().enumerate() {
            // Map regime label back to underlying state index.
            let state_idx = (0..3).find(|&j| m.regime_labels[j] == occ[i].0).unwrap();
            mean_theo += p * m.states[state_idx].mean;
            // Mixture variance = Σ p_i (σ_i² + μ_i²) - (Σ p_i μ_i)².
            var_theo += p * (m.states[state_idx].var + m.states[state_idx].mean.powi(2));
        }
        var_theo -= mean_theo.powi(2);
        // MC tolerance: 1σ_MC on the mean is sqrt(var_theo / n_total).
        let mc_sigma_mean = (var_theo / n_total).sqrt();
        assert!(
            (mean_emp - mean_theo).abs() < 3.0 * mc_sigma_mean,
            "empirical mean {:.6} too far from theoretical {:.6} (3σ_MC = {:.6})",
            mean_emp, mean_theo, 3.0 * mc_sigma_mean,
        );
        // Variance within 15% (very generous; finite-sample bias).
        assert!(
            (var_emp - var_theo).abs() / var_theo < 0.15,
            "empirical var {:.6} too far from theoretical {:.6}",
            var_emp, var_theo,
        );
    }

    /// Force the model's transition matrix to a degenerate form
    /// (state 0 → state 0 with probability 1.0) and verify that every
    /// simulated path stays in state 0. A simulator that ignored A would
    /// produce non-zero transitions.
    #[test]
    fn simulator_honors_transition_matrix() {
        let mut m = ThreeStateMarkov::default_equity();
        // Pin everything to state 0.
        m.a = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let r = simulate(&m, &McConfig {
            n_paths: 100, n_steps: 200, seed: 99, start_regime_idx: 0,
        });
        for (i, path) in r.regime_path.iter().enumerate() {
            for (t, &s) in path.iter().enumerate() {
                assert_eq!(s, 0, "path {} step {} drifted to state {}", i, t, s);
            }
        }
    }
}
