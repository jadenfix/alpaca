//! Market events. Live and backtest both produce a stream of these.

use crate::money::{Price, Qty};
use crate::symbol::Symbol;
use crate::time::Ts;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Trade {
    pub ts: Ts,
    pub symbol: Symbol,
    pub price: Price,
    pub size: Qty,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Quote {
    pub ts: Ts,
    pub symbol: Symbol,
    pub bid: Price,
    pub ask: Price,
    pub bid_size: Qty,
    pub ask_size: Qty,
}

impl Quote {
    pub fn mid_f64(&self) -> f64 {
        0.5 * (self.bid.to_f64() + self.ask.to_f64())
    }
    pub fn half_spread_f64(&self) -> f64 {
        0.5 * (self.ask.to_f64() - self.bid.to_f64()).max(0.0)
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Bar {
    pub ts: Ts, // bar close time
    pub symbol: Symbol,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: Qty,
    /// Span of the bar in seconds (60, 300, 86400, …).
    pub span_secs: u32,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Gap {
    pub from: Ts,
    pub to: Ts,
    pub reason: GapReason,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GapReason {
    WsDisconnect,
    StaleData,
    SessionBoundary,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum MarketEvent {
    Trade(Trade),
    Quote(Quote),
    Bar(Bar),
    Gap(Gap),
    SessionOpen { ts: Ts },
    SessionClose { ts: Ts },
}

impl MarketEvent {
    pub fn ts(&self) -> Ts {
        match self {
            MarketEvent::Trade(t) => t.ts,
            MarketEvent::Quote(q) => q.ts,
            MarketEvent::Bar(b) => b.ts,
            MarketEvent::Gap(g) => g.to,
            MarketEvent::SessionOpen { ts } | MarketEvent::SessionClose { ts } => *ts,
        }
    }

    pub fn symbol(&self) -> Option<Symbol> {
        match self {
            MarketEvent::Trade(t) => Some(t.symbol),
            MarketEvent::Quote(q) => Some(q.symbol),
            MarketEvent::Bar(b) => Some(b.symbol),
            MarketEvent::Gap(_) | MarketEvent::SessionOpen { .. } | MarketEvent::SessionClose { .. } => None,
        }
    }
}
