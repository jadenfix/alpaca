//! In-process paper broker. Used by shadow-live and tests.
//!
//! Accepts every order and simulates an immediate fill at the intent ts
//! using a flat assumed half-spread. Real shadow-live uses live quotes to
//! mark fills; this is the dev convenience.

use crate::broker_trait::{AccountSnapshot, Broker, BrokerError, OrderAck};
use algo_core::{Fill, Notional, OrderId, OrderIntent};
use async_trait::async_trait;
use parking_lot::Mutex;

pub struct PaperBroker {
    fills: Mutex<Vec<Fill>>,
}

impl Default for PaperBroker {
    fn default() -> Self {
        Self {
            fills: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Broker for PaperBroker {
    async fn submit(&self, intent: &OrderIntent) -> Result<OrderAck, BrokerError> {
        // Simulate immediate fill at order.intent price assumption (set by caller's risk layer).
        // Here we synthesize a fill at $0 — callers using PaperBroker should
        // route through the simulator's CostModel instead for realistic fills.
        let f = Fill {
            ts: intent.ts,
            order_id: intent.id,
            symbol: intent.symbol,
            side: intent.side,
            qty: intent.qty,
            price: algo_core::Price::ZERO,
            fees: Notional::ZERO,
        };
        self.fills.lock().push(f);
        Ok(OrderAck {
            id: intent.id,
            broker_id: Some(format!("paper-{}", intent.id)),
            accepted: true,
            reason: None,
        })
    }
    async fn cancel(&self, _id: OrderId) -> Result<(), BrokerError> {
        Ok(())
    }
    async fn fills_since(&self, ts_nanos: i64) -> Result<Vec<Fill>, BrokerError> {
        Ok(self
            .fills
            .lock()
            .iter()
            .filter(|f| f.ts.nanos >= ts_nanos)
            .cloned()
            .collect())
    }
    async fn account_snapshot(&self) -> Result<AccountSnapshot, BrokerError> {
        Ok(AccountSnapshot {
            cash: 100_000.0,
            buying_power: 100_000.0,
            equity: 100_000.0,
            day_trades_remaining: 99,
            pdt_restricted: false,
        })
    }
}
