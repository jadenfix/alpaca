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

pub use ema::Ema;
pub use returns::{log_returns_vec, simple_returns_vec};
pub use welford::Welford;
pub use window::Window;
pub use zscore::{zscore_vec, zscore_vec_into};
