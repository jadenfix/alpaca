//! In-memory portfolio state for the simulator. Tracks cash, positions,
//! realized + unrealized PnL.

use ahash::AHashMap;
use algo_core::{Fill, Side, Symbol};

#[derive(Copy, Clone, Debug, Default)]
pub struct Position {
    pub qty: f64,
    pub avg_px: f64,
    pub realized_pnl: f64,
}

impl Position {
    pub fn apply_fill(&mut self, side: Side, qty: f64, price: f64) {
        let signed = qty * side.signum_i8() as f64;
        let new_qty = self.qty + signed;
        // Closing or reducing
        if self.qty != 0.0 && (self.qty.signum() != signed.signum() && signed != 0.0) {
            let closing = signed.abs().min(self.qty.abs());
            let pnl = (price - self.avg_px) * closing * self.qty.signum();
            self.realized_pnl += pnl;
        }
        // Update avg_px when adding to position in same direction or opening fresh.
        if self.qty == 0.0 || self.qty.signum() == signed.signum() {
            let total = self.qty.abs() + signed.abs();
            if total > 0.0 {
                self.avg_px = (self.avg_px * self.qty.abs() + price * signed.abs()) / total;
            }
        } else if new_qty.signum() != self.qty.signum() && new_qty != 0.0 {
            // Flipped direction; reset avg to the fill price for the remainder.
            self.avg_px = price;
        }
        self.qty = new_qty;
        if self.qty == 0.0 {
            self.avg_px = 0.0;
        }
    }
}

#[derive(Clone, Debug)]
pub struct PortfolioState {
    pub cash: f64,
    pub positions: AHashMap<Symbol, Position>,
    pub fee_paid: f64,
    pub last_mark: AHashMap<Symbol, f64>,
}

impl PortfolioState {
    pub fn new(initial_cash: f64) -> Self {
        Self {
            cash: initial_cash,
            positions: AHashMap::new(),
            fee_paid: 0.0,
            last_mark: AHashMap::new(),
        }
    }

    pub fn apply_fill(&mut self, fill: &Fill) {
        let qty = fill.qty.to_f64();
        let price = fill.price.to_f64();
        let fees = fill.fees.to_f64();
        let signed_notional = qty * price * fill.side.signum_i8() as f64;
        // Buying spends cash; selling receives cash. Fees always reduce cash.
        self.cash -= signed_notional;
        self.cash -= fees;
        self.fee_paid += fees;
        let pos = self.positions.entry(fill.symbol).or_default();
        pos.apply_fill(fill.side, qty, price);
    }

    pub fn mark(&mut self, symbol: Symbol, price: f64) {
        self.last_mark.insert(symbol, price);
    }

    pub fn nav(&self) -> f64 {
        let mut n = self.cash;
        for (sym, pos) in &self.positions {
            if let Some(px) = self.last_mark.get(sym) {
                n += pos.qty * px;
            } else {
                n += pos.qty * pos.avg_px;
            }
        }
        n
    }

    pub fn buying_power(&self) -> f64 {
        // Cash-account approximation: cash minus current long exposure already booked.
        self.cash.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algo_core::{Notional, Price, Qty, Side, Ts};

    fn mk_fill(side: Side, qty: f64, price: f64) -> Fill {
        Fill {
            ts: Ts::from_nanos(0),
            order_id: algo_core::OrderId::new(),
            symbol: Symbol::new("AAPL").unwrap(),
            side,
            qty: Qty::from_i64(qty as i64),
            price: Price::from_f64(price).unwrap(),
            fees: Notional::from_f64(0.0).unwrap(),
        }
    }

    #[test]
    fn buy_then_sell_realizes_pnl() {
        let mut p = PortfolioState::new(10_000.0);
        p.apply_fill(&mk_fill(Side::Buy, 10.0, 100.0));
        assert_eq!(p.cash, 9_000.0);
        p.apply_fill(&mk_fill(Side::Sell, 10.0, 110.0));
        assert_eq!(p.cash, 10_100.0);
        assert_eq!(p.positions[&Symbol::new("AAPL").unwrap()].realized_pnl, 100.0);
    }

    #[test]
    fn nav_reflects_marks() {
        let mut p = PortfolioState::new(10_000.0);
        p.apply_fill(&mk_fill(Side::Buy, 10.0, 100.0));
        p.mark(Symbol::new("AAPL").unwrap(), 110.0);
        assert_eq!(p.nav(), 9_000.0 + 1_100.0);
    }
}
