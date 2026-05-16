//! CUSUM (cumulative sum) change-point detector.
//!
//! Page (1954): for each new observation x_t, maintain two cumulative sums
//!   S+_t = max(0, S+_{t-1} + (x_t - μ - k))
//!   S-_t = min(0, S-_{t-1} + (x_t - μ + k))
//!
//! Trigger when |S| > h. `μ` is the expected mean (set to running EWMA), `k`
//! is the slack (typically 0.5σ), `h` is the threshold (typically 5σ).
//!
//! Use cases:
//!   - Adaptive stop-trading signal (regime break detected → flatten + pause).
//!   - News reaction filter (large surprise → temporarily mute strategy).
//!
//! Reference: Page (1954); Roberts (1959); used by Lopez de Prado (2018).

#[derive(Copy, Clone, Debug)]
pub struct CusumDetector {
    pub k: f64,
    pub h: f64,
    pub mean: f64,
    pub s_pos: f64,
    pub s_neg: f64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CusumSignal {
    None,
    UpwardBreak,
    DownwardBreak,
}

impl CusumDetector {
    pub fn new(k: f64, h: f64) -> Self {
        Self {
            k,
            h,
            mean: 0.0,
            s_pos: 0.0,
            s_neg: 0.0,
        }
    }

    /// Update with a new observation. Returns the signal. If a break fires,
    /// the cumulative sums are reset to zero (so the next break can be detected).
    pub fn update(&mut self, x: f64) -> CusumSignal {
        let dev = x - self.mean;
        self.s_pos = (self.s_pos + dev - self.k).max(0.0);
        self.s_neg = (self.s_neg + dev + self.k).min(0.0);
        if self.s_pos > self.h {
            self.s_pos = 0.0;
            self.s_neg = 0.0;
            return CusumSignal::UpwardBreak;
        }
        if self.s_neg < -self.h {
            self.s_pos = 0.0;
            self.s_neg = 0.0;
            return CusumSignal::DownwardBreak;
        }
        CusumSignal::None
    }

    /// Adapt the mean towards `x` via EWMA with parameter `alpha`.
    pub fn track_mean(&mut self, x: f64, alpha: f64) {
        self.mean = (1.0 - alpha) * self.mean + alpha * x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_on_persistent_upward_shift() {
        let mut d = CusumDetector::new(0.5, 5.0);
        // 100 zero-mean obs, then 20 obs at +1
        let mut signal = CusumSignal::None;
        for _ in 0..100 {
            d.update(0.0);
        }
        for _ in 0..20 {
            let s = d.update(1.0);
            if s != CusumSignal::None {
                signal = s;
                break;
            }
        }
        assert_eq!(signal, CusumSignal::UpwardBreak);
    }

    #[test]
    fn does_not_fire_on_iid_noise() {
        let mut d = CusumDetector::new(0.5, 5.0);
        // Symmetric +1 / -1 oscillation: should mostly cancel
        let mut fired = false;
        for i in 0..200 {
            let x = if i % 2 == 0 { 1.0 } else { -1.0 };
            if d.update(x) != CusumSignal::None {
                fired = true;
                break;
            }
        }
        assert!(!fired, "should not fire on symmetric oscillation");
    }
}
