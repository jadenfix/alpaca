//! Strategy implementations. Each module is a standalone, disciplined,
//! literature-backed strategy.
//!
//! ## Real implementations
//! - `xs_momentum`           — cross-sectional momentum + vol scaling
//! - `pairs_mean_reversion`  — Engle-Granger cointegration + OU half-life
//! - `kalman_pairs`          — Kalman filter dynamic hedge ratio
//! - `garch_vol_target`      — GARCH(1,1) vol-targeted momentum
//! - `regime_hmm`            — 2-state HMM regime-gated momentum
//! - `markov_router`         — 3-regime Markov-switching strategy router
//! - `hurst_regime`          — Hurst-exponent-gated trend / mean-reversion
//! - `leadlag_pairs`         — lead-lag cross-correlation between two assets
//!
//! ## Stubs (planned)
//! - `overnight_drift`       — close→open factor
//! - `last30min_momentum`    — close-auction drift
//! - `pead`                  — post-earnings announcement drift

pub mod garch_vol_target;
pub mod hurst_regime;
pub mod kalman_pairs;
pub mod last30min_momentum;
pub mod leadlag_pairs;
pub mod markov_router;
pub mod overnight_drift;
pub mod pairs_mean_reversion;
pub mod pead;
pub mod regime_hmm;
pub mod xs_momentum;

pub use garch_vol_target::{GarchVolTarget, GarchVolTargetConfig};
pub use hurst_regime::{HurstRegime, HurstRegimeConfig};
pub use kalman_pairs::{KalmanPairs, KalmanPairsConfig};
pub use last30min_momentum::Last30MinMomentum;
pub use leadlag_pairs::{LeadLagPairs, LeadLagPairsConfig};
pub use markov_router::{MarkovRouter, MarkovRouterConfig};
pub use overnight_drift::OvernightDrift;
pub use pairs_mean_reversion::{PairsConfig, PairsMeanReversion};
pub use pead::Pead;
pub use regime_hmm::{RegimeHmm, RegimeHmmConfig};
pub use xs_momentum::XsMomentum;
