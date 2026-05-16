//! Strategy implementations. Each module is a standalone, disciplined,
//! literature-backed strategy.

pub mod last30min_momentum;
pub mod overnight_drift;
pub mod pairs_mean_reversion;
pub mod pead;
pub mod xs_momentum;

pub use last30min_momentum::Last30MinMomentum;
pub use overnight_drift::OvernightDrift;
pub use pairs_mean_reversion::PairsMeanReversion;
pub use pead::Pead;
pub use xs_momentum::XsMomentum;
