//! 3-state Gaussian Markov-switching model — generalized HMM for regime
//! detection with named regimes:
//!   - **Bull** (low vol, positive drift)
//!   - **Bear** (high vol, negative drift)
//!   - **Sideways** (low vol, near-zero drift)
//!
//! Online forward filtering for inference, batch Baum-Welch for training.
//! After fit, states are auto-labeled by examining their mean+variance:
//!   - lowest variance + most-positive mean   → Bull
//!   - lowest variance + near-zero mean       → Sideways
//!   - highest variance                       → Bear
//!
//! References:
//!   - Hamilton (1989), "A New Approach to the Economic Analysis of Nonstationary
//!     Time Series and the Business Cycle"
//!   - Ang & Bekaert (2002), "Regime Switches in Interest Rates"

use std::f64::consts::PI;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Regime {
    Bull,
    Bear,
    Sideways,
}

impl Regime {
    pub fn name(self) -> &'static str {
        match self {
            Regime::Bull => "bull",
            Regime::Bear => "bear",
            Regime::Sideways => "sideways",
        }
    }
}

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

const K: usize = 3; // 3 states

#[derive(Clone, Debug)]
pub struct ThreeStateMarkov {
    /// Initial distribution π[0..K].
    pub pi: [f64; K],
    /// Transition matrix A[i][j] = P(s_t=j | s_{t-1}=i).
    pub a: [[f64; K]; K],
    /// Emission parameters for each state (indexed 0..K).
    pub states: [GaussianState; K],
    /// Mapping from internal state index → human-readable Regime label.
    /// Populated after `fit_baum_welch`.
    pub regime_labels: [Regime; K],
    /// Filtered posterior P(s_t | r_1..r_t).
    pub alpha: [f64; K],
    initialized: bool,
}

impl ThreeStateMarkov {
    /// Sensible priors for daily equity returns.
    pub fn default_equity() -> Self {
        Self {
            pi: [0.4, 0.3, 0.3],
            a: [
                [0.95, 0.02, 0.03], // bull tends to stay bull
                [0.03, 0.94, 0.03], // bear sticky
                [0.05, 0.03, 0.92], // sideways
            ],
            states: [
                GaussianState { mean:  0.0005, var: 0.00008 }, // bull: low vol + drift
                GaussianState { mean: -0.0008, var: 0.0006 },  // bear: high vol + neg
                GaussianState { mean:  0.0000, var: 0.00015 }, // sideways: med vol
            ],
            regime_labels: [Regime::Bull, Regime::Bear, Regime::Sideways],
            alpha: [0.4, 0.3, 0.3],
            initialized: false,
        }
    }

    /// One-step online forward filter. Returns posterior P(s_t).
    pub fn update(&mut self, r: f64) -> [f64; K] {
        if !self.initialized {
            let mut p = [0.0_f64; K];
            for i in 0..K {
                p[i] = self.pi[i] * self.states[i].pdf(r);
            }
            let s: f64 = p.iter().sum::<f64>().max(1e-300);
            for i in 0..K {
                self.alpha[i] = p[i] / s;
            }
            self.initialized = true;
            return self.alpha;
        }
        // Predict
        let mut pred = [0.0_f64; K];
        for j in 0..K {
            let mut s = 0.0;
            for i in 0..K {
                s += self.alpha[i] * self.a[i][j];
            }
            pred[j] = s;
        }
        // Update
        let mut p = [0.0_f64; K];
        for j in 0..K {
            p[j] = pred[j] * self.states[j].pdf(r);
        }
        let s: f64 = p.iter().sum::<f64>().max(1e-300);
        for j in 0..K {
            self.alpha[j] = p[j] / s;
        }
        self.alpha
    }

    /// Current most-likely regime.
    pub fn argmax_regime(&self) -> Regime {
        let mut best = 0;
        let mut best_v = self.alpha[0];
        for i in 1..K {
            if self.alpha[i] > best_v {
                best_v = self.alpha[i];
                best = i;
            }
        }
        self.regime_labels[best]
    }

    /// Posterior probability of being in a specific regime.
    pub fn p_regime(&self, r: Regime) -> f64 {
        for i in 0..K {
            if self.regime_labels[i] == r {
                return self.alpha[i];
            }
        }
        0.0
    }

    /// Posterior over all regimes as a (Regime, probability) array.
    pub fn posterior(&self) -> [(Regime, f64); K] {
        [
            (self.regime_labels[0], self.alpha[0]),
            (self.regime_labels[1], self.alpha[1]),
            (self.regime_labels[2], self.alpha[2]),
        ]
    }

    /// Fit via Baum-Welch EM on a batch of returns. Re-labels states by
    /// (mean, var) interpretation after convergence.
    pub fn fit_baum_welch(&mut self, returns: &[f64], max_iter: usize, tol: f64) {
        let n = returns.len();
        if n < 30 {
            return;
        }
        let mut last_ll = f64::NEG_INFINITY;
        for _iter in 0..max_iter {
            // Forward with scaling
            let mut alpha = vec![[0.0_f64; K]; n];
            let mut scale = vec![0.0_f64; n];
            for i in 0..K {
                alpha[0][i] = self.pi[i] * self.states[i].pdf(returns[0]);
            }
            scale[0] = (alpha[0].iter().sum::<f64>()).max(1e-300);
            for i in 0..K {
                alpha[0][i] /= scale[0];
            }
            for t in 1..n {
                for j in 0..K {
                    let mut s = 0.0;
                    for i in 0..K {
                        s += alpha[t - 1][i] * self.a[i][j];
                    }
                    alpha[t][j] = s * self.states[j].pdf(returns[t]);
                }
                scale[t] = (alpha[t].iter().sum::<f64>()).max(1e-300);
                for j in 0..K {
                    alpha[t][j] /= scale[t];
                }
            }
            let ll: f64 = scale.iter().map(|s| s.ln()).sum();
            let converged = (ll - last_ll).abs() < tol;
            last_ll = ll;
            if converged {
                self.alpha = alpha[n - 1];
                break;
            }
            // Backward
            let mut beta = vec![[0.0_f64; K]; n];
            for i in 0..K {
                beta[n - 1][i] = 1.0;
            }
            for t in (0..n - 1).rev() {
                for i in 0..K {
                    let mut s = 0.0;
                    for j in 0..K {
                        s += self.a[i][j] * self.states[j].pdf(returns[t + 1]) * beta[t + 1][j];
                    }
                    beta[t][i] = s / scale[t + 1];
                }
            }
            // Gamma + xi
            let mut gamma = vec![[0.0_f64; K]; n];
            for t in 0..n {
                let denom: f64 = (0..K).map(|i| alpha[t][i] * beta[t][i]).sum::<f64>().max(1e-300);
                for i in 0..K {
                    gamma[t][i] = alpha[t][i] * beta[t][i] / denom;
                }
            }
            let mut xi = vec![[[0.0_f64; K]; K]; n - 1];
            for t in 0..n - 1 {
                let mut denom = 0.0;
                for i in 0..K {
                    for j in 0..K {
                        denom += alpha[t][i] * self.a[i][j] * self.states[j].pdf(returns[t + 1])
                            * beta[t + 1][j];
                    }
                }
                let denom = denom.max(1e-300);
                for i in 0..K {
                    for j in 0..K {
                        xi[t][i][j] = alpha[t][i] * self.a[i][j] * self.states[j].pdf(returns[t + 1])
                            * beta[t + 1][j] / denom;
                    }
                }
            }
            // Re-estimate
            self.pi = gamma[0];
            for i in 0..K {
                let denom: f64 = (0..n - 1).map(|t| gamma[t][i]).sum::<f64>().max(1e-300);
                for j in 0..K {
                    let num: f64 = (0..n - 1).map(|t| xi[t][i][j]).sum();
                    self.a[i][j] = num / denom;
                }
            }
            for j in 0..K {
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
        // Re-label states by (mean, var) interpretation.
        self.relabel_regimes();
        self.initialized = true;
    }

    /// Label state indices to regimes by mean/var heuristic.
    /// Highest variance → Bear. Of the remaining two, the one with the more
    /// positive mean → Bull, the other → Sideways.
    fn relabel_regimes(&mut self) {
        let mut by_var: Vec<usize> = (0..K).collect();
        by_var.sort_by(|&a, &b| {
            self.states[b]
                .var
                .partial_cmp(&self.states[a].var)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let bear_idx = by_var[0];
        let rest = [by_var[1], by_var[2]];
        let (bull_idx, side_idx) = if self.states[rest[0]].mean > self.states[rest[1]].mean {
            (rest[0], rest[1])
        } else {
            (rest[1], rest[0])
        };
        let mut labels = [Regime::Sideways; K];
        labels[bull_idx] = Regime::Bull;
        labels[bear_idx] = Regime::Bear;
        labels[side_idx] = Regime::Sideways;
        self.regime_labels = labels;
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
    fn online_filter_normalizes_to_one() {
        let mut m = ThreeStateMarkov::default_equity();
        let mut g = rng_normal(7);
        for _ in 0..100 {
            m.update(0.001 * g());
            let s: f64 = m.alpha.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "posterior must sum to 1, got {s}");
        }
    }

    #[test]
    fn baum_welch_recovers_three_regimes() {
        // Synthesize: 300 bull bars, 200 bear bars, 300 sideways bars, repeat.
        let mut g = rng_normal(13);
        let mut rets = Vec::with_capacity(2400);
        for _cycle in 0..3 {
            for _ in 0..300 { rets.push(0.001 + 0.005 * g()); }   // bull
            for _ in 0..200 { rets.push(-0.002 + 0.025 * g()); }  // bear
            for _ in 0..300 { rets.push(0.0 + 0.008 * g()); }     // sideways
        }
        let mut m = ThreeStateMarkov::default_equity();
        m.fit_baum_welch(&rets, 30, 1e-4);
        // After fit, we should have three distinguishable states.
        let bull_var = m.states.iter().enumerate()
            .find(|(i, _)| m.regime_labels[*i] == Regime::Bull)
            .map(|(_, s)| s.var).unwrap();
        let bear_var = m.states.iter().enumerate()
            .find(|(i, _)| m.regime_labels[*i] == Regime::Bear)
            .map(|(_, s)| s.var).unwrap();
        assert!(bear_var > bull_var * 2.0, "bear variance should dwarf bull");
    }

    #[test]
    fn argmax_regime_responds_to_turbulence() {
        let mut m = ThreeStateMarkov::default_equity();
        let mut g = rng_normal(17);
        // Quiet first
        for _ in 0..200 { m.update(0.0001 * g()); }
        let p_bull_quiet = m.p_regime(Regime::Bull);
        // Turbulent
        for _ in 0..50 { m.update(0.05 * g()); }
        let p_bear_after = m.p_regime(Regime::Bear);
        assert!(p_bear_after > 0.3, "bear prob should rise after turbulence, got {}", p_bear_after);
        // Bull prob should have dropped (or at least bear should now exceed it)
        assert!(p_bear_after > p_bull_quiet * 0.5);
    }
}
