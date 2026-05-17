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
pub mod correlation;
pub mod cusum;
pub mod garch;
pub mod hmm;
pub mod hurst;
pub mod kalman;
pub mod leadlag;
pub mod markov_switching;
pub mod monte_carlo;
pub mod ou;

// Econophysics deep stack — Random Matrix Theory, information theory, point
// processes. Each module is a publishable-grade method with known-truth tests.
pub mod distance_correlation;
pub mod hawkes_exp;
pub mod pca_rmt;
pub mod transfer_entropy;

pub use cointegration::{adf_pvalue, adf_test_statistic, engle_granger, EngleGrangerResult};
pub use correlation::{
    correlation_matrix, kendall_tau, pearson_corr, spearman_corr, RollingCorr,
};
pub use distance_correlation::distance_correlation;
pub use hawkes_exp::{fit_hawkes_exp, hawkes_exp_loglik, HawkesParams};
pub use pca_rmt::{
    marchenko_pastur_density, marchenko_pastur_bounds, pca, rie_shrinkage,
    tracy_widom_pvalue, PcaResult,
};
pub use transfer_entropy::transfer_entropy;
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
