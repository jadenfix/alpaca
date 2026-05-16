//! Foundational types for the trading engine.
//!
//! Every other crate depends on this one. Keep it dependency-light and
//! `no_std`-friendly where reasonable.

pub mod clock;
pub mod event;
pub mod money;
pub mod order;
pub mod symbol;
pub mod time;

pub use clock::{Clock, RealClock, SimClock};
pub use event::{Bar, Gap, MarketEvent, Quote, Trade};
pub use money::{Notional, Price, Qty};
pub use order::{Fill, OrderId, OrderIntent, OrderStatus, OrderType, Side, TimeInForce};
pub use symbol::Symbol;
pub use time::Ts;
