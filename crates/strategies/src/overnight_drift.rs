//! Overnight drift (close→open).
//! Long the universe at close, flat at open. Sized as a small allocation
//! (risk-premium style) — high hit rate, low per-trade edge.

use algo_core::{MarketEvent, OrderIntent, Ts};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use smallvec::SmallVec;

#[derive(Clone, Debug, Default)]
pub struct OvernightDrift;

impl Strategy for OvernightDrift {
    fn name(&self) -> &'static str {
        "overnight_drift"
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
