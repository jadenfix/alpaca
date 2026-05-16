//! Pre-trade checks: per-name notional, fat-finger guard, halt list, buying power.

use ahash::AHashSet;
use algo_core::{Notional, OrderIntent, Price, Symbol};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RiskLimits {
    /// Max single-name notional as fraction of NAV at entry.
    pub max_name_pct_nav: f64,
    /// Max absolute notional per order (fat-finger guard).
    pub max_order_notional: f64,
    /// Max ratio of (order qty) / (recent ADV bars) — guards against accidentally
    /// trading huge volume in a low-liquidity name.
    pub max_qty_vs_adv: f64,
    /// Buying power cushion: never use more than this fraction of available cash.
    pub buying_power_cushion: f64,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_name_pct_nav: 0.05,
            max_order_notional: 25_000.0,
            max_qty_vs_adv: 0.01,
            buying_power_cushion: 0.95,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RiskDecision {
    Accept,
    Reject(String),
}

pub struct PretradeChecks {
    limits: RiskLimits,
    halt_list: AHashSet<Symbol>,
    /// Symbol → most recent reference price (mid or last trade), for sizing.
    last_price: HashMap<Symbol, f64>,
    /// Symbol → recent ADV estimate in shares.
    adv_shares: HashMap<Symbol, f64>,
}

impl PretradeChecks {
    pub fn new(limits: RiskLimits) -> Self {
        Self {
            limits,
            halt_list: AHashSet::new(),
            last_price: HashMap::new(),
            adv_shares: HashMap::new(),
        }
    }

    pub fn set_halted(&mut self, symbol: Symbol, halted: bool) {
        if halted {
            self.halt_list.insert(symbol);
        } else {
            self.halt_list.remove(&symbol);
        }
    }

    pub fn observe_price(&mut self, symbol: Symbol, price: Price) {
        self.last_price.insert(symbol, price.to_f64());
    }

    pub fn observe_adv(&mut self, symbol: Symbol, adv_shares: f64) {
        self.adv_shares.insert(symbol, adv_shares);
    }

    pub fn check(&self, intent: &OrderIntent, nav: f64, buying_power: f64) -> RiskDecision {
        if self.halt_list.contains(&intent.symbol) {
            return RiskDecision::Reject(format!("halt list: {}", intent.symbol));
        }
        let px = self.last_price.get(&intent.symbol).copied();
        let Some(px) = px else {
            return RiskDecision::Reject(format!("no reference price for {}", intent.symbol));
        };
        if px <= 0.0 || !px.is_finite() {
            return RiskDecision::Reject(format!("bad reference price {} for {}", px, intent.symbol));
        }
        let qty = intent.qty.to_f64().abs();
        if qty <= 0.0 {
            return RiskDecision::Reject("non-positive qty".into());
        }
        let notional = px * qty;
        if notional > self.limits.max_order_notional {
            return RiskDecision::Reject(format!(
                "fat-finger guard: notional {} > {}",
                notional, self.limits.max_order_notional
            ));
        }
        if nav > 0.0 && notional / nav > self.limits.max_name_pct_nav {
            return RiskDecision::Reject(format!(
                "name cap: {} > {}% of NAV",
                notional,
                self.limits.max_name_pct_nav * 100.0
            ));
        }
        if buying_power > 0.0 && notional > buying_power * self.limits.buying_power_cushion {
            return RiskDecision::Reject(format!(
                "buying power: {} > {} of {}",
                notional,
                self.limits.buying_power_cushion,
                buying_power
            ));
        }
        if let Some(&adv) = self.adv_shares.get(&intent.symbol) {
            if adv > 0.0 && qty / adv > self.limits.max_qty_vs_adv {
                return RiskDecision::Reject(format!(
                    "qty/ADV {:.4} > {:.4}",
                    qty / adv,
                    self.limits.max_qty_vs_adv
                ));
            }
        }
        RiskDecision::Accept
    }
}

/// Convenience: convert a Notional to f64 for fast comparisons.
pub fn nominal_f64(n: Notional) -> f64 {
    n.to_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use algo_core::{OrderId, OrderType, Qty, Side, TimeInForce};

    fn mk_intent(sym: &str, qty: f64) -> OrderIntent {
        OrderIntent {
            id: OrderId::new(),
            ts: algo_core::Ts::from_nanos(0),
            symbol: Symbol::new(sym).unwrap(),
            side: Side::Buy,
            qty: Qty::from_i64(qty as i64),
            order_type: OrderType::Market,
            tif: TimeInForce::Day,
            strategy: "test".into(),
            tag: None,
        }
    }

    #[test]
    fn rejects_without_reference_price() {
        let p = PretradeChecks::new(RiskLimits::default());
        let d = p.check(&mk_intent("AAPL", 10.0), 100_000.0, 100_000.0);
        assert!(matches!(d, RiskDecision::Reject(_)));
    }

    #[test]
    fn rejects_oversize_name() {
        let mut p = PretradeChecks::new(RiskLimits::default());
        p.observe_price(Symbol::new("AAPL").unwrap(), Price::from_f64(200.0).unwrap());
        // 100 shares * $200 = $20k; NAV = $100k → 20% > 5% cap
        let d = p.check(&mk_intent("AAPL", 100.0), 100_000.0, 100_000.0);
        assert!(matches!(d, RiskDecision::Reject(msg) if msg.contains("name cap")));
    }

    #[test]
    fn accepts_normal_order() {
        let mut p = PretradeChecks::new(RiskLimits::default());
        p.observe_price(Symbol::new("AAPL").unwrap(), Price::from_f64(200.0).unwrap());
        // 20 shares * $200 = $4k; 4% of $100k NAV → ok
        let d = p.check(&mk_intent("AAPL", 20.0), 100_000.0, 100_000.0);
        assert_eq!(d, RiskDecision::Accept);
    }

    #[test]
    fn rejects_halted_symbol() {
        let mut p = PretradeChecks::new(RiskLimits::default());
        let sym = Symbol::new("AAPL").unwrap();
        p.observe_price(sym, Price::from_f64(200.0).unwrap());
        p.set_halted(sym, true);
        let d = p.check(&mk_intent("AAPL", 5.0), 100_000.0, 100_000.0);
        assert!(matches!(d, RiskDecision::Reject(msg) if msg.contains("halt")));
    }
}
