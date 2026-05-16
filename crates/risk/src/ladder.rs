//! Multi-level loss-limit ladder.
//!
//! The ladder is monotone: once a level trips, deeper levels can still trip;
//! shallower levels do not auto-reset. Operators reset via the ops binary.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LossLimits {
    pub per_strategy_intraday: f64, // L2
    pub portfolio_intraday: f64,    // L3
    pub portfolio_hard_stop: f64,   // L4
    pub rolling_5day: f64,          // L5
    pub trailing_20day: f64,        // L6
    pub all_time_peak: f64,         // L7
}

impl Default for LossLimits {
    fn default() -> Self {
        Self {
            per_strategy_intraday: 0.010,
            portfolio_intraday: 0.020,
            portfolio_hard_stop: 0.035,
            rolling_5day: 0.050,
            trailing_20day: 0.080,
            all_time_peak: 0.120,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Default)]
pub enum LossLevel {
    #[default]
    None,
    L2StrategyHalt(StrategyId),
    L3NoNewEntries,
    L4FlattenAll,
    L5RollingHalt,
    L6TrailingHalt,
    L7PermanentKill,
}

/// Strategy is identified by static name; we hash to a small id for storage.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct StrategyId(pub u64);

#[derive(Clone, Debug, Default)]
pub struct LossLadder {
    limits: LossLimits,
    /// All-time NAV peak (only ratchets up).
    peak_nav: f64,
    /// Day-open NAV (resets daily).
    day_open_nav: f64,
    /// Per-strategy day-open allocated capital.
    strategy_day_open: HashMap<StrategyId, f64>,
    /// Rolling 5-day NAV peak.
    rolling_5d_peak: f64,
    /// Trailing 20-day NAV peak.
    trailing_20d_peak: f64,
    state: LossLevel,
}

impl LossLadder {
    pub fn new(limits: LossLimits, initial_nav: f64) -> Self {
        Self {
            limits,
            peak_nav: initial_nav,
            day_open_nav: initial_nav,
            strategy_day_open: HashMap::new(),
            rolling_5d_peak: initial_nav,
            trailing_20d_peak: initial_nav,
            state: LossLevel::None,
        }
    }

    pub fn on_session_open(&mut self, nav: f64) {
        self.day_open_nav = nav;
        self.strategy_day_open.clear();
        // L2/L3/L4 are intraday; lower the state to L5/L6/L7 if those were active.
        self.state = match self.state {
            LossLevel::L2StrategyHalt(_) | LossLevel::L3NoNewEntries | LossLevel::L4FlattenAll => {
                LossLevel::None
            }
            other => other,
        };
    }

    pub fn observe_strategy_open(&mut self, strategy: StrategyId, allocated: f64) {
        self.strategy_day_open.insert(strategy, allocated);
    }

    pub fn observe_5d_peak(&mut self, nav: f64) {
        if nav > self.rolling_5d_peak {
            self.rolling_5d_peak = nav;
        }
    }

    pub fn observe_20d_peak(&mut self, nav: f64) {
        if nav > self.trailing_20d_peak {
            self.trailing_20d_peak = nav;
        }
    }

    /// Update with latest NAV and per-strategy PnL. Returns the strictest level reached.
    pub fn update(&mut self, nav: f64, strategy_pnls: &HashMap<StrategyId, f64>) -> LossLevel {
        if nav > self.peak_nav {
            self.peak_nav = nav;
        }
        // L7: permanent kill — never recovers in-process.
        if self.peak_nav > 0.0
            && (self.peak_nav - nav) / self.peak_nav >= self.limits.all_time_peak
        {
            self.state = LossLevel::L7PermanentKill;
            return self.state;
        }
        // L6
        if self.trailing_20d_peak > 0.0
            && (self.trailing_20d_peak - nav) / self.trailing_20d_peak >= self.limits.trailing_20day
        {
            self.state = self.escalate(LossLevel::L6TrailingHalt);
            return self.state;
        }
        // L5
        if self.rolling_5d_peak > 0.0
            && (self.rolling_5d_peak - nav) / self.rolling_5d_peak >= self.limits.rolling_5day
        {
            self.state = self.escalate(LossLevel::L5RollingHalt);
            return self.state;
        }
        // L4 hard stop
        if self.day_open_nav > 0.0
            && (self.day_open_nav - nav) / self.day_open_nav >= self.limits.portfolio_hard_stop
        {
            self.state = self.escalate(LossLevel::L4FlattenAll);
            return self.state;
        }
        // L3 no new entries
        if self.day_open_nav > 0.0
            && (self.day_open_nav - nav) / self.day_open_nav >= self.limits.portfolio_intraday
        {
            self.state = self.escalate(LossLevel::L3NoNewEntries);
            return self.state;
        }
        // L2 strategy halt
        for (sid, pnl) in strategy_pnls {
            if let Some(alloc) = self.strategy_day_open.get(sid) {
                if *alloc > 0.0 && (-*pnl) / *alloc >= self.limits.per_strategy_intraday {
                    self.state = self.escalate(LossLevel::L2StrategyHalt(*sid));
                    return self.state;
                }
            }
        }
        self.state
    }

    fn escalate(&self, candidate: LossLevel) -> LossLevel {
        // Permanent kill never downgrades.
        if matches!(self.state, LossLevel::L7PermanentKill) {
            return LossLevel::L7PermanentKill;
        }
        // Otherwise take the deeper of (current, candidate) by ordering depth.
        if depth(candidate) > depth(self.state) {
            candidate
        } else {
            self.state
        }
    }

    pub fn state(&self) -> LossLevel {
        self.state
    }

    pub fn manual_reset(&mut self) {
        if !matches!(self.state, LossLevel::L7PermanentKill) {
            self.state = LossLevel::None;
        }
    }
}

fn depth(l: LossLevel) -> u8 {
    match l {
        LossLevel::None => 0,
        LossLevel::L2StrategyHalt(_) => 2,
        LossLevel::L3NoNewEntries => 3,
        LossLevel::L4FlattenAll => 4,
        LossLevel::L5RollingHalt => 5,
        LossLevel::L6TrailingHalt => 6,
        LossLevel::L7PermanentKill => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l3_then_l4_then_no_downgrade() {
        let mut ld = LossLadder::new(LossLimits::default(), 100_000.0);
        ld.on_session_open(100_000.0);
        let pnls = HashMap::new();
        // -2.0% → L3
        assert_eq!(ld.update(98_000.0, &pnls), LossLevel::L3NoNewEntries);
        // -3.5% → L4
        assert_eq!(ld.update(96_500.0, &pnls), LossLevel::L4FlattenAll);
        // Improve back to -2.5%; still L4 (no downgrade intraday)
        assert_eq!(ld.update(97_500.0, &pnls), LossLevel::L4FlattenAll);
    }

    #[test]
    fn l7_is_permanent() {
        let mut ld = LossLadder::new(LossLimits::default(), 100_000.0);
        ld.on_session_open(100_000.0);
        let pnls = HashMap::new();
        // -13% from peak → L7
        assert_eq!(ld.update(87_000.0, &pnls), LossLevel::L7PermanentKill);
        // Reset: still permanent
        ld.manual_reset();
        assert_eq!(ld.state(), LossLevel::L7PermanentKill);
    }

    #[test]
    fn per_strategy_halt() {
        let mut ld = LossLadder::new(LossLimits::default(), 100_000.0);
        ld.on_session_open(100_000.0);
        let sid = StrategyId(42);
        ld.observe_strategy_open(sid, 20_000.0);
        let mut pnls = HashMap::new();
        // -1.1% of $20k = -$220 → L2
        pnls.insert(sid, -220.0);
        assert!(matches!(ld.update(99_780.0, &pnls), LossLevel::L2StrategyHalt(_)));
    }

    #[test]
    fn intraday_state_resets_on_new_session() {
        let mut ld = LossLadder::new(LossLimits::default(), 100_000.0);
        ld.on_session_open(100_000.0);
        let pnls = HashMap::new();
        ld.update(98_000.0, &pnls); // L3
        assert_eq!(ld.state(), LossLevel::L3NoNewEntries);
        ld.on_session_open(98_000.0);
        assert_eq!(ld.state(), LossLevel::None);
    }
}
