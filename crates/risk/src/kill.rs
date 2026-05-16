//! Kill switch — orthogonal triggers that produce a ladder-equivalent action.

use crate::ladder::LossLevel;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KillTrigger {
    /// Path to a kill file; existence trips the switch.
    pub kill_file: Option<PathBuf>,
    pub max_ws_gap_ms: u64,
    pub max_order_rejects_per_min: u32,
    pub max_clock_skew_ms: u64,
    pub min_buying_power: f64,
    pub max_heartbeat_misses: u32,
}

impl Default for KillTrigger {
    fn default() -> Self {
        Self {
            kill_file: Some(PathBuf::from("./KILL")),
            max_ws_gap_ms: 500,
            max_order_rejects_per_min: 5,
            max_clock_skew_ms: 250,
            min_buying_power: 1_000.0,
            max_heartbeat_misses: 3,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Tripped {
    KillFilePresent,
    WsGapExceeded,
    RejectRateExceeded,
    ClockSkewExceeded,
    BuyingPowerLow,
    HeartbeatMissed,
    PositionDrift,
    StrategyRunaway,
    DayTradeLimitClose,
    SymbolHalt,
}

impl Tripped {
    /// Equivalent ladder action: at minimum L3 (no new entries).
    /// Position-drift and kill-file → L4 flatten-all.
    pub fn ladder_action(self) -> LossLevel {
        match self {
            Tripped::KillFilePresent | Tripped::PositionDrift => LossLevel::L4FlattenAll,
            _ => LossLevel::L3NoNewEntries,
        }
    }
}

#[derive(Default)]
pub struct SwitchState {
    pub tripped: Vec<Tripped>,
}

impl SwitchState {
    pub fn add(&mut self, t: Tripped) {
        if !self.tripped.contains(&t) {
            self.tripped.push(t);
        }
    }
    pub fn is_tripped(&self) -> bool {
        !self.tripped.is_empty()
    }
    pub fn strictest(&self) -> Option<LossLevel> {
        self.tripped
            .iter()
            .map(|t| t.ladder_action())
            .max_by_key(|l| match l {
                LossLevel::L4FlattenAll => 4,
                LossLevel::L3NoNewEntries => 3,
                _ => 0,
            })
    }
    pub fn clear(&mut self) {
        self.tripped.clear();
    }
}

pub struct KillSwitch {
    cfg: KillTrigger,
}

impl KillSwitch {
    pub fn new(cfg: KillTrigger) -> Self {
        Self { cfg }
    }

    pub fn evaluate(&self, inputs: &SwitchInputs) -> SwitchState {
        let mut s = SwitchState::default();
        if let Some(p) = &self.cfg.kill_file {
            if p.exists() {
                s.add(Tripped::KillFilePresent);
            }
        }
        if inputs.ws_gap_ms > self.cfg.max_ws_gap_ms {
            s.add(Tripped::WsGapExceeded);
        }
        if inputs.order_rejects_per_min > self.cfg.max_order_rejects_per_min {
            s.add(Tripped::RejectRateExceeded);
        }
        if inputs.clock_skew_ms > self.cfg.max_clock_skew_ms {
            s.add(Tripped::ClockSkewExceeded);
        }
        if inputs.buying_power < self.cfg.min_buying_power {
            s.add(Tripped::BuyingPowerLow);
        }
        if inputs.heartbeat_misses > self.cfg.max_heartbeat_misses {
            s.add(Tripped::HeartbeatMissed);
        }
        if inputs.position_drift {
            s.add(Tripped::PositionDrift);
        }
        if inputs.runaway_strategy {
            s.add(Tripped::StrategyRunaway);
        }
        if inputs.day_trades_remaining <= 1 {
            s.add(Tripped::DayTradeLimitClose);
        }
        if inputs.any_symbol_halted {
            s.add(Tripped::SymbolHalt);
        }
        s
    }
}

#[derive(Default, Debug)]
pub struct SwitchInputs {
    pub ws_gap_ms: u64,
    pub order_rejects_per_min: u32,
    pub clock_skew_ms: u64,
    pub buying_power: f64,
    pub heartbeat_misses: u32,
    pub position_drift: bool,
    pub runaway_strategy: bool,
    pub day_trades_remaining: u32,
    pub any_symbol_halted: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ws_gap_trips_l3() {
        let k = KillSwitch::new(KillTrigger::default());
        let mut inp = SwitchInputs::default();
        inp.ws_gap_ms = 1_000;
        inp.buying_power = 100_000.0;
        inp.day_trades_remaining = 100;
        let s = k.evaluate(&inp);
        assert!(s.tripped.contains(&Tripped::WsGapExceeded));
        assert_eq!(s.strictest(), Some(LossLevel::L3NoNewEntries));
    }

    #[test]
    fn position_drift_trips_l4() {
        let k = KillSwitch::new(KillTrigger::default());
        let mut inp = SwitchInputs::default();
        inp.buying_power = 100_000.0;
        inp.day_trades_remaining = 100;
        inp.position_drift = true;
        let s = k.evaluate(&inp);
        assert_eq!(s.strictest(), Some(LossLevel::L4FlattenAll));
    }
}
