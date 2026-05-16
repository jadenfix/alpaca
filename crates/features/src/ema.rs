//! Exponentially weighted moving average + variance.

#[derive(Clone, Debug)]
pub struct Ema {
    alpha: f64,
    mean: Option<f64>,
    var: f64,
}

impl Ema {
    /// `half_life` in samples; alpha = 1 - 2^(-1/half_life).
    pub fn from_half_life(half_life: f64) -> Self {
        assert!(half_life > 0.0);
        let alpha = 1.0 - 2f64.powf(-1.0 / half_life);
        Self {
            alpha,
            mean: None,
            var: 0.0,
        }
    }

    pub fn from_alpha(alpha: f64) -> Self {
        assert!((0.0..=1.0).contains(&alpha));
        Self {
            alpha,
            mean: None,
            var: 0.0,
        }
    }

    pub fn update(&mut self, x: f64) -> f64 {
        match self.mean {
            None => {
                self.mean = Some(x);
                x
            }
            Some(m) => {
                let new_m = m + self.alpha * (x - m);
                let diff = x - m;
                self.var = (1.0 - self.alpha) * (self.var + self.alpha * diff * diff);
                self.mean = Some(new_m);
                new_m
            }
        }
    }

    pub fn mean(&self) -> Option<f64> {
        self.mean
    }
    pub fn variance(&self) -> f64 {
        self.var
    }
    pub fn stddev(&self) -> f64 {
        self.var.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_to_constant() {
        let mut e = Ema::from_half_life(5.0);
        for _ in 0..200 {
            e.update(3.0);
        }
        assert!((e.mean().unwrap() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn responds_to_shift() {
        let mut e = Ema::from_half_life(10.0);
        for _ in 0..100 {
            e.update(0.0);
        }
        for _ in 0..100 {
            e.update(1.0);
        }
        let m = e.mean().unwrap();
        assert!(m > 0.9 && m < 1.0);
    }
}
