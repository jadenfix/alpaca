//! Feature engineering: look-ahead-safe windows + vectorized rolling stats.
//!
//! Design rules:
//! - Never expose raw `&[T]` of historical data. Only `Window<T>` with
//!   `.last_n()` and `.as_of(ts)` so strategies cannot peek forward.
//! - Streaming updates use Welford (incremental moments) — literally cannot
//!   see the future.
//! - Batch updates (per-bar across the whole universe) use vectorized
//!   `ndarray` ops with BLAS, not scalar loops.

pub mod ema;
pub mod returns;
pub mod welford;
pub mod window;
pub mod zscore;

// Deep statistics — built from scratch, used by strategies and tests.
pub mod cointegration;
pub mod cusum;
pub mod garch;
pub mod hmm;
pub mod hurst;
pub mod kalman;
pub mod leadlag;
pub mod markov_switching;
pub mod monte_carlo;
pub mod ou;

pub use cointegration::{adf_pvalue, adf_test_statistic, engle_granger, EngleGrangerResult};
pub use cusum::CusumDetector;
pub use ema::Ema;
pub use garch::Garch11;
pub use hmm::TwoStateGaussianHmm;
pub use hurst::hurst_rs;
pub use kalman::ScalarKalman;
pub use leadlag::{lead_lag, LeadLagResult};
pub use markov_switching::{Regime, ThreeStateMarkov};
pub use monte_carlo::{regime_occupancy, simulate as mc_simulate, McConfig, McResult};
pub use ou::{fit_ou, OuParams};
pub use returns::{log_returns_vec, simple_returns_vec};
pub use welford::Welford;
pub use window::Window;
pub use zscore::{zscore_vec, zscore_vec_into};
