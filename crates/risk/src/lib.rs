//! Pre-trade risk + multi-level loss-limit ladder + kill switch.

pub mod kill;
pub mod ladder;
pub mod pretrade;

pub use kill::{KillSwitch, KillTrigger, SwitchInputs, SwitchState, Tripped};
pub use ladder::{LossLadder, LossLevel, LossLimits};
pub use pretrade::{PretradeChecks, RiskDecision, RiskLimits};
