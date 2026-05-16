//! Post-earnings announcement drift (PEAD).
//! Decades of academic backing. Trade signed earnings-surprise direction
//! for 1–5 sessions after the announcement.

use algo_core::{MarketEvent, OrderIntent, Ts};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use smallvec::SmallVec;

#[derive(Clone, Debug, Default)]
pub struct Pead;

impl Strategy for Pead {
    fn name(&self) -> &'static str {
        "pead"
    }
    fn requires_gpu(&self) -> bool {
        // Optional sentiment classifier upgrade path.
        false
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
