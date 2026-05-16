//! Clock abstraction. Every wall-clock read in the engine must go through this.
//!
//! `clippy.toml` disallows `std::time::Instant::now`, `SystemTime::now`, and
//! `chrono::Utc::now` everywhere except this module so backtest is deterministic.

use crate::time::Ts;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Source of "now" for the engine. Both backtest and live go through this.
pub trait Clock: Send + Sync {
    fn now(&self) -> Ts;
}

/// Real wall-clock. Use only at process boundaries (the live binary).
#[derive(Default, Debug)]
pub struct RealClock;

impl Clock for RealClock {
    fn now(&self) -> Ts {
        // Allowed within `core::clock` only.
        #[allow(clippy::disallowed_methods)]
        let d = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch");
        Ts::from_nanos(d.as_nanos() as i64)
    }
}

/// Manually-advanced clock for backtests and tests.
#[derive(Clone, Debug)]
pub struct SimClock {
    inner: Arc<Mutex<Ts>>,
}

impl SimClock {
    pub fn new(start: Ts) -> Self {
        Self {
            inner: Arc::new(Mutex::new(start)),
        }
    }

    /// Advance to a specific timestamp; panics if it would go backwards.
    pub fn set(&self, ts: Ts) {
        let mut g = self.inner.lock();
        assert!(ts >= *g, "SimClock cannot move backwards");
        *g = ts;
    }

    pub fn advance_nanos(&self, n: i64) {
        let mut g = self.inner.lock();
        *g = g.checked_add_nanos(n).expect("clock overflow");
    }
}

impl Clock for SimClock {
    fn now(&self) -> Ts {
        *self.inner.lock()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sim_clock_monotonic_advance() {
        let c = SimClock::new(Ts::from_nanos(100));
        assert_eq!(c.now(), Ts::from_nanos(100));
        c.advance_nanos(50);
        assert_eq!(c.now(), Ts::from_nanos(150));
    }

    #[test]
    #[should_panic(expected = "cannot move backwards")]
    fn sim_clock_rejects_backwards() {
        let c = SimClock::new(Ts::from_nanos(100));
        c.set(Ts::from_nanos(50));
    }

    #[test]
    fn real_clock_returns_recent_time() {
        let c = RealClock;
        let t = c.now();
        // sanity: after year 2020
        assert!(t.nanos > 1_577_836_800_000_000_000);
    }
}
