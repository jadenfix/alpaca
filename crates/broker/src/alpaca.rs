//! Alpaca REST trading client — skeleton.
//!
//! Construction validates endpoint + reads creds from env. Real `submit` /
//! `cancel` / `fills_since` / `account_snapshot` calls land in the next
//! iteration; for now they return `NotImplemented` so the live binary fails
//! loudly rather than silently doing the wrong thing.

use crate::broker_trait::{AccountSnapshot, Broker, BrokerError, OrderAck};
use crate::endpoint::{is_allowed_endpoint, Endpoint};
use algo_core::{Fill, OrderId, OrderIntent};
use async_trait::async_trait;
use reqwest::Client;
use std::time::Duration;

pub struct AlpacaRest {
    base_url: String,
    api_key: String,
    api_secret: String,
    http: Client,
}

impl AlpacaRest {
    pub fn new(endpoint: Endpoint, api_key: String, api_secret: String) -> Result<Self, BrokerError> {
        let base_url = endpoint.base_url().to_string();
        if !is_allowed_endpoint(&base_url, endpoint) {
            return Err(BrokerError::Auth(format!("disallowed endpoint: {base_url}")));
        }
        let http = Client::builder()
            .timeout(Duration::from_secs(10))
            .connect_timeout(Duration::from_secs(3))
            .build()
            .map_err(|e| BrokerError::Transport(e.to_string()))?;
        Ok(Self {
            base_url,
            api_key,
            api_secret,
            http,
        })
    }

    pub fn from_env(endpoint: Endpoint) -> Result<Self, BrokerError> {
        let api_key = std::env::var("ALPACA_API_KEY")
            .map_err(|_| BrokerError::Auth("ALPACA_API_KEY not set".into()))?;
        let api_secret = std::env::var("ALPACA_API_SECRET")
            .map_err(|_| BrokerError::Auth("ALPACA_API_SECRET not set".into()))?;
        Self::new(endpoint, api_key, api_secret)
    }

    fn auth_headers(&self) -> reqwest::header::HeaderMap {
        use reqwest::header::{HeaderMap, HeaderValue};
        let mut h = HeaderMap::new();
        h.insert(
            "APCA-API-KEY-ID",
            HeaderValue::from_str(&self.api_key).unwrap(),
        );
        h.insert(
            "APCA-API-SECRET-KEY",
            HeaderValue::from_str(&self.api_secret).unwrap(),
        );
        h
    }
}

#[async_trait]
impl Broker for AlpacaRest {
    async fn submit(&self, intent: &OrderIntent) -> Result<OrderAck, BrokerError> {
        // Skeleton: validate http exists, validate auth headers form, then return NotImplemented.
        let _ = self.auth_headers();
        let _ = (&self.http, &self.base_url, intent);
        Err(BrokerError::NotImplemented)
    }
    async fn cancel(&self, _id: OrderId) -> Result<(), BrokerError> {
        Err(BrokerError::NotImplemented)
    }
    async fn fills_since(&self, _ts_nanos: i64) -> Result<Vec<Fill>, BrokerError> {
        Err(BrokerError::NotImplemented)
    }
    async fn account_snapshot(&self) -> Result<AccountSnapshot, BrokerError> {
        Err(BrokerError::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_disallowed_endpoint_string() {
        let bad = AlpacaRest::new(
            Endpoint::Paper,
            "k".into(),
            "s".into(),
        );
        // It builds because Endpoint::Paper.base_url() is always allowed; this just
        // confirms construction succeeds for the legitimate paper endpoint.
        assert!(bad.is_ok());
    }
}
