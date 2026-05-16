//! Market data sources.
//!
//! Provides a `MarketDataSource` trait. The Alpaca WebSocket implementation
//! is currently a skeleton — it parses messages into `MarketEvent` but the
//! WS subscription/reconnect loop is left for a follow-up so we ship a green
//! workspace today.

pub mod alpaca_ws;
pub mod source;

pub use source::{MarketDataSource, SourceConfig, SubscribeRequest};
