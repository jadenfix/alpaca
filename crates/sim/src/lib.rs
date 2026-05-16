//! Cost + slippage model used by backtest fills AND live PnL attribution.
//!
//! Model: `cost = commission + half_spread * qty + impact_coef * sqrt(qty/adv) * price * qty`
//!
//! Backtest fills marketable / market orders at:
//!   buy:  mid + half_spread + impact
//!   sell: mid - half_spread - impact

use algo_core::{Notional, Price, Qty, Side};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CostModel {
    /// Per-share commission (Alpaca = 0 for equities; SEC/TAF apply on sells).
    pub commission_per_share: f64,
    /// SEC fee on sells, as fraction of notional (e.g. 0.0000278 = $27.80 per $1MM in 2024).
    pub sec_fee_sell_pct: f64,
    /// FINRA TAF per share on sells.
    pub finra_taf_per_share_sell: f64,
    /// Impact coefficient on √(qty/ADV).
    pub impact_coef: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            commission_per_share: 0.0,
            sec_fee_sell_pct: 0.0000278,
            finra_taf_per_share_sell: 0.000166,
            impact_coef: 0.10,
        }
    }
}

pub struct FillContext {
    pub mid: f64,
    pub half_spread: f64,
    pub adv_shares: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct FillResult {
    pub fill_price: Price,
    pub fees: Notional,
}

impl CostModel {
    /// Compute fill price + fees for a market order under the model.
    pub fn fill(&self, side: Side, qty: Qty, ctx: &FillContext) -> FillResult {
        let q = qty.to_f64().abs();
        let participation = if ctx.adv_shares > 0.0 {
            (q / ctx.adv_shares).max(0.0)
        } else {
            0.0
        };
        let impact_per_share = self.impact_coef * participation.sqrt() * ctx.mid;
        let slip_per_share = ctx.half_spread + impact_per_share;
        let signed = match side {
            Side::Buy => ctx.mid + slip_per_share,
            Side::Sell => ctx.mid - slip_per_share,
        };
        let fill_price = Price::from_f64(signed.max(0.0)).unwrap_or(Price::ZERO);
        let commission = self.commission_per_share * q;
        let sells_only = match side {
            Side::Sell => {
                let notional = signed * q;
                self.sec_fee_sell_pct * notional + self.finra_taf_per_share_sell * q
            }
            Side::Buy => 0.0,
        };
        let total_fees = commission + sells_only;
        let fees = Notional::from_f64(total_fees).unwrap_or(Notional::ZERO);
        FillResult { fill_price, fees }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algo_core::Qty;

    #[test]
    fn buy_pays_more_than_mid() {
        let m = CostModel::default();
        let ctx = FillContext {
            mid: 100.0,
            half_spread: 0.01,
            adv_shares: 1_000_000.0,
        };
        let f = m.fill(Side::Buy, Qty::from_i64(100), &ctx);
        assert!(f.fill_price.to_f64() > 100.0);
    }

    #[test]
    fn sell_pays_less_than_mid_and_has_fees() {
        let m = CostModel::default();
        let ctx = FillContext {
            mid: 100.0,
            half_spread: 0.01,
            adv_shares: 1_000_000.0,
        };
        let f = m.fill(Side::Sell, Qty::from_i64(100), &ctx);
        assert!(f.fill_price.to_f64() < 100.0);
        assert!(f.fees.to_f64() > 0.0);
    }

    #[test]
    fn impact_scales_with_size() {
        let m = CostModel::default();
        let ctx = FillContext {
            mid: 100.0,
            half_spread: 0.0,
            adv_shares: 1_000_000.0,
        };
        let small = m.fill(Side::Buy, Qty::from_i64(100), &ctx).fill_price.to_f64();
        let large = m.fill(Side::Buy, Qty::from_i64(100_000), &ctx).fill_price.to_f64();
        assert!(large > small);
    }
}
