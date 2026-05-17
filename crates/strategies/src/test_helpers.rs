//! Test-only fixtures shared across strategy unit tests.

#![cfg(test)]

use algo_core::{Bar, MarketEvent, Price, Qty, Symbol, Ts};
use algo_strategy::PortfolioView;

pub fn mk_bar(sym: Symbol, ts_ns: i64, close: f64) -> MarketEvent {
    MarketEvent::Bar(Bar {
        ts: Ts::from_nanos(ts_ns),
        symbol: sym,
        open: Price::from_f64(close).unwrap(),
        high: Price::from_f64(close * 1.0001).unwrap(),
        low: Price::from_f64(close * 0.9999).unwrap(),
        close: Price::from_f64(close).unwrap(),
        volume: Qty::from_i64(10_000),
        span_secs: 60,
    })
}

pub fn pv(now_ns: i64) -> PortfolioView {
    PortfolioView { nav: 1_000_000.0, buying_power: 1_000_000.0, now: Ts::from_nanos(now_ns) }
}

pub fn pv_with_nav(now_ns: i64, nav: f64) -> PortfolioView {
    PortfolioView { nav, buying_power: nav, now: Ts::from_nanos(now_ns) }
}
