//! `MarketDataSource` trait + common config.

use algo_core::{MarketEvent, Symbol};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceConfig {
    /// e.g. "wss://stream.data.alpaca.markets/v2/iex" for IEX feed.
    pub url: String,
    pub api_key_env: String,
    pub api_secret_env: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubscribeRequest {
    pub trades: Vec<Symbol>,
    pub quotes: Vec<Symbol>,
    pub bars: Vec<Symbol>,
}

#[async_trait]
pub trait MarketDataSource: Send {
    /// Connect, authenticate, and subscribe. Pushes `MarketEvent`s into `out`.
    async fn run(
        &mut self,
        subs: SubscribeRequest,
        out: mpsc::Sender<MarketEvent>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}
