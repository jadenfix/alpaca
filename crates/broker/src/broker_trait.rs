//! `Broker` trait — abstracts paper / live / shadow.

use algo_core::{Fill, OrderId, OrderIntent};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BrokerError {
    #[error("transport: {0}")]
    Transport(String),
    #[error("auth: {0}")]
    Auth(String),
    #[error("rate limited")]
    RateLimited,
    #[error("rejected: {0}")]
    Rejected(String),
    #[error("not implemented")]
    NotImplemented,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderAck {
    pub id: OrderId,
    pub broker_id: Option<String>,
    pub accepted: bool,
    pub reason: Option<String>,
}

#[async_trait]
pub trait Broker: Send + Sync {
    async fn submit(&self, intent: &OrderIntent) -> Result<OrderAck, BrokerError>;
    async fn cancel(&self, id: OrderId) -> Result<(), BrokerError>;
    async fn fills_since(&self, ts_nanos: i64) -> Result<Vec<Fill>, BrokerError>;
    /// Buying power + portfolio NAV snapshot (for reconciliation).
    async fn account_snapshot(&self) -> Result<AccountSnapshot, BrokerError>;
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AccountSnapshot {
    pub cash: f64,
    pub buying_power: f64,
    pub equity: f64,
    pub day_trades_remaining: u32,
    pub pdt_restricted: bool,
}
