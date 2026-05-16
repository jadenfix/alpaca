//! Alpaca WebSocket market data — skeleton.
//!
//! Wire-protocol contract (Alpaca v2):
//! - Connect to wss://stream.data.alpaca.markets/v2/{feed}  (feed = iex|sip)
//! - Receive `{T:"success",msg:"connected"}` → send `{action:"auth",key:..,secret:..}`
//! - Receive `{T:"success",msg:"authenticated"}` → send `{action:"subscribe",trades:[..],quotes:[..],bars:[..]}`
//! - Messages then arrive as arrays of `{T, S, t, ...}` objects.
//!
//! Full implementation (reconnect, resubscribe, gap detection) is left to
//! the live-trading milestone. This file ships the message parser + a stub
//! `run` so the workspace compiles cleanly.

use crate::source::{MarketDataSource, SourceConfig, SubscribeRequest};
use algo_core::{Bar, MarketEvent, Price, Qty, Quote, Symbol, Trade, Ts};
use async_trait::async_trait;
use serde::Deserialize;
use thiserror::Error;
use tokio::sync::mpsc;

#[derive(Error, Debug)]
pub enum AlpacaWsError {
    #[error("not yet implemented (skeleton)")]
    NotImplemented,
    #[error("missing env var: {0}")]
    MissingEnv(String),
}

pub struct AlpacaWs {
    cfg: SourceConfig,
}

impl AlpacaWs {
    pub fn new(cfg: SourceConfig) -> Self {
        Self { cfg }
    }
}

#[async_trait]
impl MarketDataSource for AlpacaWs {
    async fn run(
        &mut self,
        _subs: SubscribeRequest,
        _out: mpsc::Sender<MarketEvent>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Validate creds present so the live binary fails loudly at startup
        // even before the WS code is wired.
        let _ = std::env::var(&self.cfg.api_key_env)
            .map_err(|_| AlpacaWsError::MissingEnv(self.cfg.api_key_env.clone()))?;
        let _ = std::env::var(&self.cfg.api_secret_env)
            .map_err(|_| AlpacaWsError::MissingEnv(self.cfg.api_secret_env.clone()))?;
        tracing::warn!(
            "AlpacaWs::run is a skeleton; WS connect not yet wired. \
             See plan §Fallback & resilience infrastructure."
        );
        Err(Box::new(AlpacaWsError::NotImplemented))
    }
}

// ----- parsers -----

#[derive(Deserialize, Debug)]
#[serde(tag = "T")]
pub enum AlpacaWsMsg {
    #[serde(rename = "t")]
    Trade {
        #[serde(rename = "S")]
        symbol: String,
        #[serde(rename = "p")]
        price: f64,
        #[serde(rename = "s")]
        size: u64,
        #[serde(rename = "t")]
        ts: String, // ISO 8601
    },
    #[serde(rename = "q")]
    Quote {
        #[serde(rename = "S")]
        symbol: String,
        #[serde(rename = "bp")]
        bid_price: f64,
        #[serde(rename = "ap")]
        ask_price: f64,
        #[serde(rename = "bs")]
        bid_size: u64,
        #[serde(rename = "as")]
        ask_size: u64,
        #[serde(rename = "t")]
        ts: String,
    },
    #[serde(rename = "b")]
    Bar {
        #[serde(rename = "S")]
        symbol: String,
        #[serde(rename = "o")]
        open: f64,
        #[serde(rename = "h")]
        high: f64,
        #[serde(rename = "l")]
        low: f64,
        #[serde(rename = "c")]
        close: f64,
        #[serde(rename = "v")]
        volume: u64,
        #[serde(rename = "t")]
        ts: String,
    },
    #[serde(rename = "success")]
    Success { msg: String },
    #[serde(rename = "error")]
    Error { code: i64, msg: String },
    #[serde(other)]
    Other,
}

fn iso_to_ts(s: &str) -> Option<Ts> {
    // Alpaca emits RFC3339 with nanos; hifitime can parse.
    hifitime::Epoch::from_gregorian_str(s)
        .ok()
        .map(|e| Ts::from_nanos(e.to_unix_seconds() as i64 * 1_000_000_000))
}

pub fn convert(msg: AlpacaWsMsg) -> Option<MarketEvent> {
    match msg {
        AlpacaWsMsg::Trade { symbol, price, size, ts } => Some(MarketEvent::Trade(Trade {
            ts: iso_to_ts(&ts)?,
            symbol: Symbol::new(&symbol)?,
            price: Price::from_f64(price)?,
            size: Qty::from_i64(size as i64),
        })),
        AlpacaWsMsg::Quote {
            symbol,
            bid_price,
            ask_price,
            bid_size,
            ask_size,
            ts,
        } => Some(MarketEvent::Quote(Quote {
            ts: iso_to_ts(&ts)?,
            symbol: Symbol::new(&symbol)?,
            bid: Price::from_f64(bid_price)?,
            ask: Price::from_f64(ask_price)?,
            bid_size: Qty::from_i64(bid_size as i64),
            ask_size: Qty::from_i64(ask_size as i64),
        })),
        AlpacaWsMsg::Bar {
            symbol,
            open,
            high,
            low,
            close,
            volume,
            ts,
        } => Some(MarketEvent::Bar(Bar {
            ts: iso_to_ts(&ts)?,
            symbol: Symbol::new(&symbol)?,
            open: Price::from_f64(open)?,
            high: Price::from_f64(high)?,
            low: Price::from_f64(low)?,
            close: Price::from_f64(close)?,
            volume: Qty::from_i64(volume as i64),
            span_secs: 60,
        })),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_trade_message() {
        let s = r#"{"T":"t","S":"AAPL","p":190.10,"s":100,"t":"2025-01-02T15:30:00.000Z"}"#;
        let msg: AlpacaWsMsg = serde_json::from_str(s).unwrap();
        let ev = convert(msg).unwrap();
        assert!(matches!(ev, MarketEvent::Trade(_)));
    }

    #[test]
    fn parses_success_handshake() {
        let s = r#"{"T":"success","msg":"connected"}"#;
        let msg: AlpacaWsMsg = serde_json::from_str(s).unwrap();
        assert!(matches!(msg, AlpacaWsMsg::Success { .. }));
    }
}
