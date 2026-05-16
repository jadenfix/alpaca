//! Order intent, lifecycle, and fill types.

use crate::money::{Price, Qty};
use crate::symbol::Symbol;
use crate::time::Ts;
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    pub fn signum_i8(self) -> i8 {
        match self {
            Side::Buy => 1,
            Side::Sell => -1,
        }
    }
    pub fn opposite(self) -> Self {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum OrderType {
    /// Market order. Filled at best available price; only credible backtest type.
    Market,
    /// Marketable limit: a limit set inside the spread; nearly always fills, but bounded.
    MarketableLimit { limit: Price },
    /// Resting limit. Paper/live only — no credible backtest fill simulation.
    Limit { limit: Price },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TimeInForce {
    Day,
    Ioc, // immediate-or-cancel
    Gtc,
    Opg, // at open
    Cls, // at close
}

/// What backtest fidelity an order type supports.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BacktestableOrderType {
    MarketOnly,
    Marketable,
    RestingLimit,
}

impl OrderType {
    pub fn backtestable(&self) -> BacktestableOrderType {
        match self {
            OrderType::Market => BacktestableOrderType::MarketOnly,
            OrderType::MarketableLimit { .. } => BacktestableOrderType::Marketable,
            OrderType::Limit { .. } => BacktestableOrderType::RestingLimit,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct OrderId(pub Uuid);

impl OrderId {
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for OrderId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Strategy intent. Risk layer turns this into an `Order` sent to the broker.
///
/// `strategy` is a short owned string (typically a `&'static str` from a
/// strategy module). Owned for serde round-tripping in the journal.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OrderIntent {
    pub id: OrderId,
    pub ts: Ts,
    pub symbol: Symbol,
    pub side: Side,
    pub qty: Qty,
    pub order_type: OrderType,
    pub tif: TimeInForce,
    pub strategy: String,
    pub tag: Option<String>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum OrderStatus {
    New,
    Accepted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Fill {
    pub ts: Ts,
    pub order_id: OrderId,
    pub symbol: Symbol,
    pub side: Side,
    pub qty: Qty,
    pub price: Price,
    /// Fees the broker charged (Alpaca = $0 commission for equities, but
    /// SEC/TAF fees apply on sells).
    pub fees: crate::money::Notional,
}
