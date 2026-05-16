//! Strategy trait. Every strategy implements this and is fed events one at a
//! time. Output: zero or more `OrderIntent`s.

use algo_core::{MarketEvent, OrderIntent, Symbol, Ts};
use smallvec::SmallVec;

/// What the strategy needs to know about the portfolio at decision time.
#[derive(Copy, Clone, Debug)]
pub struct PortfolioView {
    pub nav: f64,
    pub buying_power: f64,
    pub now: Ts,
}

/// Information about a current open position the strategy holds.
#[derive(Copy, Clone, Debug)]
pub struct PositionView {
    pub symbol: Symbol,
    pub qty: f64,    // signed
    pub avg_px: f64, // entry-weighted
    pub mark_px: f64,
}

pub trait Strategy: Send {
    fn name(&self) -> &'static str;

    /// Whether this strategy needs the `gpu` feature compiled in.
    fn requires_gpu(&self) -> bool {
        false
    }

    /// Called for every market event. Return zero or more intents.
    fn on_event(
        &mut self,
        event: &MarketEvent,
        portfolio: &PortfolioView,
        positions: &[PositionView],
    ) -> SmallVec<[OrderIntent; 4]>;

    /// Called at the end of each session — final exits, telemetry reset.
    fn on_session_close(&mut self, _ts: Ts) -> SmallVec<[OrderIntent; 4]> {
        SmallVec::new()
    }
}
