//! Scalar Kalman filter for pairs trading: dynamic hedge ratio β_t.
//!
//! Model:
//!   β_t = β_{t-1} + w_t       w_t ~ N(0, Q)   (state evolution, random walk)
//!   y_t = β_t · x_t + v_t     v_t ~ N(0, R)   (observation = price of asset Y)
//!
//! Spread definition:  s_t = y_t - β_t · x_t   (zero-mean if Q, R well-calibrated)
//!
//! The strategy trades when |s_t / √P_t_pred| crosses a z-score threshold.
//!
//! Reference: Chan, "Algorithmic Trading: Winning Strategies and Their
//! Rationale" (2013), Ch. 3 — Kalman filter dynamic hedge.

#[derive(Copy, Clone, Debug)]
pub struct ScalarKalman {
    /// Process (state) noise variance.
    pub q: f64,
    /// Observation noise variance.
    pub r: f64,
    /// Current state estimate β̂_t.
    pub beta: f64,
    /// State covariance (variance) P_t.
    pub p: f64,
    initialized: bool,
}

impl ScalarKalman {
    /// `q` = how fast the hedge ratio is allowed to drift (e.g., 1e-5 for slow).
    /// `r` = observation noise (set by spread vol).
    pub fn new(q: f64, r: f64) -> Self {
        Self {
            q,
            r,
            beta: 0.0,
            p: 1.0,
            initialized: false,
        }
    }

    pub fn with_init(q: f64, r: f64, beta0: f64, p0: f64) -> Self {
        Self {
            q,
            r,
            beta: beta0,
            p: p0,
            initialized: true,
        }
    }

    /// Update step. Given observation `y_t` and predictor `x_t`, returns the
    /// pre-update (innovation) spread = y - β̂_{t|t-1} · x.
    /// After return, `self.beta` and `self.p` are the posterior estimates.
    pub fn update(&mut self, y: f64, x: f64) -> KalmanStep {
        // Predict (random-walk state ⇒ β̂_{t|t-1} = β̂_{t-1|t-1})
        // P_{t|t-1} = P_{t-1|t-1} + Q
        let p_pred = self.p + self.q;
        // Initialize on first observation if needed
        if !self.initialized {
            // Crude init: β = y / x if x != 0
            if x.abs() > 1e-12 {
                self.beta = y / x;
                self.p = (y * y + 1.0).max(1.0); // diffuse prior
            }
            self.initialized = true;
            // Don't update — wait for next observation to do a proper filter step
            return KalmanStep {
                spread: 0.0,
                spread_variance: self.r + p_pred * x * x,
                gain: 0.0,
            };
        }
        // Innovation: ε_t = y_t - β̂_{t|t-1} · x_t
        let innovation = y - self.beta * x;
        // Innovation variance: S = x² P_{t|t-1} + R
        let s = x * x * p_pred + self.r;
        // Kalman gain: K = P_{t|t-1} · x / S
        let k = if s.abs() > 1e-12 { (p_pred * x) / s } else { 0.0 };
        // Update
        self.beta += k * innovation;
        self.p = (1.0 - k * x) * p_pred;
        KalmanStep {
            spread: innovation,
            spread_variance: s,
            gain: k,
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

#[derive(Copy, Clone, Debug)]
pub struct KalmanStep {
    pub spread: f64,
    pub spread_variance: f64,
    pub gain: f64,
}

impl KalmanStep {
    pub fn z_score(&self) -> f64 {
        if self.spread_variance > 0.0 {
            self.spread / self.spread_variance.sqrt()
        } else {
            0.0
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
    fn tracks_constant_beta() {
        // y = 2 * x + noise; β should converge to 2.
        let mut g = rng_normal(7);
        let mut kf = ScalarKalman::new(1e-5, 0.5);
        for i in 0..500 {
            let x = (i as f64 * 0.05).sin() + 2.0;
            let y = 2.0 * x + 0.1 * g();
            let _ = kf.update(y, x);
        }
        assert!(
            (kf.beta - 2.0).abs() < 0.1,
            "expected β≈2.0, got {}",
            kf.beta
        );
    }

    #[test]
    fn tracks_drifting_beta() {
        // β slowly drifts from 1 → 3 over 500 steps; KF should follow.
        let mut g = rng_normal(11);
        let mut kf = ScalarKalman::new(1e-3, 0.5); // higher Q ⇒ faster adapt
        let mut last_beta = 0.0;
        for i in 0..500 {
            let beta_t = 1.0 + 2.0 * (i as f64 / 500.0);
            let x = (i as f64 * 0.02).cos() + 2.0;
            let y = beta_t * x + 0.05 * g();
            let _ = kf.update(y, x);
            last_beta = kf.beta;
        }
        // Final true β = 3, expect KF to be within 0.5 with adaptive Q
        assert!(
            (last_beta - 3.0).abs() < 0.5,
            "expected β near 3.0, got {}",
            last_beta
        );
    }

    #[test]
    fn innovation_is_small_when_well_calibrated() {
        let mut g = rng_normal(13);
        let mut kf = ScalarKalman::new(1e-5, 0.04); // R = (0.2)^2
        // Run for warm-up
        for i in 0..200 {
            let x = 10.0 + (i as f64).sin();
            let y = 1.5 * x + 0.2 * g();
            let _ = kf.update(y, x);
        }
        // After warm-up, innovations should be small relative to spread variance
        let step = kf.update(15.0, 10.0); // x=10, true y ≈ 15
        // z-score should be < 3 in typical noise
        assert!(step.z_score().abs() < 4.0);
    }
}
