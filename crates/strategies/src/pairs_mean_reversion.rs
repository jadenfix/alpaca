//! Pairs / cointegration mean-reversion.
//!
//! Stub for now. Real implementation:
//!   - Johansen test (or Engle-Granger for 2-leg) on the trailing window.
//!   - Estimate Ornstein-Uhlenbeck half-life; reject if τ ∉ [hours, days].
//!   - Trade z-score of spread vs hedge ratio; exit on z-cross-zero or 2×τ.

use algo_core::{MarketEvent, OrderIntent, Ts};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use smallvec::SmallVec;

#[derive(Clone, Debug, Default)]
pub struct PairsMeanReversion;

impl Strategy for PairsMeanReversion {
    fn name(&self) -> &'static str {
        "pairs_mean_reversion"
    }
    fn on_event(
        &mut self,
        _: &MarketEvent,
        _: &PortfolioView,
        _: &[PositionView],
    ) -> SmallVec<[OrderIntent; 4]> {
        SmallVec::new()
    }
    fn on_session_close(&mut self, _ts: Ts) -> SmallVec<[OrderIntent; 4]> {
        SmallVec::new()
    }
}
