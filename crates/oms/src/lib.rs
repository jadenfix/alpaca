//! Order Management System — open-order state + position cache.
//!
//! Uses `papaya` for the concurrent positions map so multiple tokio tasks
//! (broker ack handler, reconciliation task, hot-thread query path) can
//! safely access shared state without contending on a single mutex.

use algo_core::{Fill, OrderId, OrderIntent, OrderStatus, Symbol};
use papaya::HashMap as ConcurrentMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenOrder {
    pub intent: OrderIntent,
    pub status: OrderStatus,
    pub broker_id: Option<String>,
    pub filled_qty: f64,
    pub avg_fill_px: f64,
}

#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct Position {
    pub qty: f64,
    pub avg_px: f64,
    pub realized_pnl: f64,
}

pub struct Oms {
    open_orders: Mutex<std::collections::HashMap<OrderId, OpenOrder>>,
    positions: ConcurrentMap<Symbol, Position>,
}

impl Default for Oms {
    fn default() -> Self {
        Self {
            open_orders: Mutex::new(Default::default()),
            positions: ConcurrentMap::new(),
        }
    }
}

impl Oms {
    pub fn on_submit(&self, intent: OrderIntent, broker_id: Option<String>) {
        let id = intent.id;
        self.open_orders.lock().insert(
            id,
            OpenOrder {
                intent,
                status: OrderStatus::Accepted,
                broker_id,
                filled_qty: 0.0,
                avg_fill_px: 0.0,
            },
        );
    }

    pub fn on_fill(&self, fill: &Fill) {
        // Update open order
        let mut og = self.open_orders.lock();
        if let Some(o) = og.get_mut(&fill.order_id) {
            let f = fill.qty.to_f64();
            let new_filled = o.filled_qty + f;
            o.avg_fill_px = if new_filled > 0.0 {
                (o.avg_fill_px * o.filled_qty + fill.price.to_f64() * f) / new_filled
            } else {
                0.0
            };
            o.filled_qty = new_filled;
            if o.filled_qty >= o.intent.qty.to_f64() {
                o.status = OrderStatus::Filled;
            } else {
                o.status = OrderStatus::PartiallyFilled;
            }
        }
        // Update position
        let map = self.positions.pin();
        let signed = fill.qty.to_f64() * fill.side.signum_i8() as f64;
        let prev = map.get(&fill.symbol).copied().unwrap_or_default();
        let mut next = prev;
        let was_zero = prev.qty == 0.0;
        // Same-direction add or open fresh
        if was_zero || prev.qty.signum() == signed.signum() {
            let total = prev.qty.abs() + signed.abs();
            if total > 0.0 {
                next.avg_px = (prev.avg_px * prev.qty.abs() + fill.price.to_f64() * signed.abs())
                    / total;
            }
            next.qty = prev.qty + signed;
        } else {
            // closing or flipping
            let closing = signed.abs().min(prev.qty.abs());
            let pnl = (fill.price.to_f64() - prev.avg_px) * closing * prev.qty.signum();
            next.realized_pnl += pnl;
            next.qty = prev.qty + signed;
            if next.qty.signum() != prev.qty.signum() && next.qty != 0.0 {
                next.avg_px = fill.price.to_f64();
            } else if next.qty == 0.0 {
                next.avg_px = 0.0;
            }
        }
        map.insert(fill.symbol, next);
    }

    pub fn position(&self, sym: Symbol) -> Position {
        self.positions.pin().get(&sym).copied().unwrap_or_default()
    }

    pub fn open_orders_count(&self) -> usize {
        self.open_orders.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algo_core::{Notional, OrderType, Price, Qty, Side, TimeInForce, Ts};

    fn mk_intent(qty: i64) -> OrderIntent {
        OrderIntent {
            id: OrderId::new(),
            ts: Ts::from_nanos(0),
            symbol: Symbol::new("AAPL").unwrap(),
            side: Side::Buy,
            qty: Qty::from_i64(qty),
            order_type: OrderType::Market,
            tif: TimeInForce::Day,
            strategy: "t".into(),
            tag: None,
        }
    }

    #[test]
    fn fill_updates_position_and_order() {
        let oms = Oms::default();
        let i = mk_intent(10);
        let id = i.id;
        oms.on_submit(i.clone(), Some("bid".into()));
        let f = Fill {
            ts: Ts::from_nanos(1),
            order_id: id,
            symbol: i.symbol,
            side: Side::Buy,
            qty: Qty::from_i64(10),
            price: Price::from_f64(100.0).unwrap(),
            fees: Notional::ZERO,
        };
        oms.on_fill(&f);
        let pos = oms.position(i.symbol);
        assert_eq!(pos.qty, 10.0);
        assert_eq!(pos.avg_px, 100.0);
    }
}
