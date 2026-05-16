//! Last-30-min close-auction momentum.
//! Documented close-auction-drift effect: signed last-30-min return tends
//! to extend into the close. Trade marketable limit ~15:30 ET, exit at close.

use algo_core::{MarketEvent, OrderIntent, Ts};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use smallvec::SmallVec;

#[derive(Clone, Debug, Default)]
pub struct Last30MinMomentum;

impl Strategy for Last30MinMomentum {
    fn name(&self) -> &'static str {
        "last30min_momentum"
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
