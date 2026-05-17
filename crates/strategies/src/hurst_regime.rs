//! Hurst-exponent-gated strategy.
//!
//! Per symbol, track an EWMA momentum signal AND compute the Hurst exponent
//! from a rolling price window. The trading action depends on the Hurst value:
//!   - H > `trend_threshold` → trend-follow (long if momo > 0, short if < 0)
//!   - H < `mr_threshold`    → mean-revert (long if momo < 0, short if > 0)
//!   - else                  → flat
//!
//! This is a regime-adaptive single-asset strategy: it self-classifies each
//! symbol's microstructure and trades accordingly. Useful complement to
//! cross-sectional momentum because it judges each name on its own merits.
//!
//! Reference: Hurst (1951); Lo (1991), "Long-Term Memory in Stock Prices".

use ahash::AHashMap;
use algo_core::{
    MarketEvent, OrderId, OrderIntent, OrderType, Qty, Side, Symbol, TimeInForce,
};
use algo_features::{hurst_rs, Ema, Window};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HurstRegimeConfig {
    pub hurst_window: usize,
    pub momentum_half_life: f64,
    pub trend_threshold: f64, // H > this → trending
    pub mr_threshold: f64,    // H < this → mean-reverting
    pub gross_per_name: f64,
    pub rebalance_every_bars: u32,
}

impl Default for HurstRegimeConfig {
    fn default() -> Self {
        Self {
            hurst_window: 200,
            momentum_half_life: 30.0,
            trend_threshold: 0.55,
            mr_threshold: 0.45,
            gross_per_name: 0.02,
            rebalance_every_bars: 20,
        }
    }
}

struct PerSymbol {
    prices: Window<f64>,
    momo: Ema,
}

impl PerSymbol {
    fn new(hl: f64, cap: usize) -> Self {
        Self {
            prices: Window::new(cap.max(64)),
            momo: Ema::from_half_life(hl),
        }
    }
}

pub struct HurstRegime {
    cfg: HurstRegimeConfig,
    per_sym: AHashMap<Symbol, PerSymbol>,
    bars_since_rebal: u32,
}

impl HurstRegime {
    pub fn new(cfg: HurstRegimeConfig) -> Self {
        Self {
            cfg,
            per_sym: AHashMap::new(),
            bars_since_rebal: 0,
        }
    }
}

impl Strategy for HurstRegime {
    fn name(&self) -> &'static str {
        "hurst_regime"
    }

    fn validate_config(&self) -> Result<(), String> {
        if self.cfg.hurst_window < 32 {
            return Err("hurst_window must be ≥ 32".into());
        }
        if !(self.cfg.mr_threshold < self.cfg.trend_threshold) {
            return Err("mr_threshold must be less than trend_threshold".into());
        }
        if !(0.0 < self.cfg.trend_threshold && self.cfg.trend_threshold < 1.0) {
            return Err("trend_threshold must be in (0, 1)".into());
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.per_sym.clear();
        self.bars_since_rebal = 0;
    }

    fn on_event(
        &mut self,
        event: &MarketEvent,
        portfolio: &PortfolioView,
        positions: &[PositionView],
    ) -> SmallVec<[OrderIntent; 4]> {
        let MarketEvent::Bar(bar) = event else { return SmallVec::new(); };
        let px = bar.close.to_f64();
        if px <= 0.0 || !px.is_finite() {
            return SmallVec::new();
        }
        let hl = self.cfg.momentum_half_life;
        let cap = self.cfg.hurst_window + 16;
        let entry = self.per_sym
            .entry(bar.symbol)
            .or_insert_with(|| PerSymbol::new(hl, cap));
        // Need previous price for return / momo update
        let last_px = entry.prices.last().map(|(_, p)| *p);
        entry.prices.push(bar.ts, px);
        if let Some(prev) = last_px {
            if prev > 0.0 {
                let r = (px / prev).ln();
                if r.is_finite() {
                    entry.momo.update(r);
                }
            }
        }

        self.bars_since_rebal += 1;
        if self.bars_since_rebal < self.cfg.rebalance_every_bars {
            return SmallVec::new();
        }
        self.bars_since_rebal = 0;

        let mut intents: SmallVec<[OrderIntent; 4]> = SmallVec::new();
        for (sym, state) in &self.per_sym {
            let Some(prices) = state.prices.last_n(self.cfg.hurst_window) else { continue };
            let Some(h) = hurst_rs(&prices) else { continue };
            let Some(momo) = state.momo.mean() else { continue };
            let Some(last_px) = state.prices.last().map(|(_, p)| *p) else { continue };
            if last_px <= 0.0 { continue }

            // Decide direction by regime
            let dir: i8 = if h > self.cfg.trend_threshold {
                if momo > 0.0 { 1 } else if momo < 0.0 { -1 } else { 0 }
            } else if h < self.cfg.mr_threshold {
                if momo > 0.0 { -1 } else if momo < 0.0 { 1 } else { 0 }
            } else {
                0
            };

            let have = position_qty(positions, *sym);
            let target = if dir == 0 {
                0.0
            } else {
                let notional = portfolio.nav * self.cfg.gross_per_name * dir as f64;
                (notional / last_px).round()
            };
            let delta = target - have;
            if delta.abs() < 1.0 { continue }
            let side = if delta > 0.0 { Side::Buy } else { Side::Sell };
            let tag = if dir == 0 {
                Some("hurst_flat".to_string())
            } else if h > self.cfg.trend_threshold {
                Some(format!("trend_H={:.3}", h))
            } else {
                Some(format!("mr_H={:.3}", h))
            };
            intents.push(OrderIntent {
                id: OrderId::new(),
                ts: portfolio.now,
                symbol: *sym,
                side,
                qty: Qty::from_i64(delta.abs().ceil() as i64),
                order_type: OrderType::Market,
                tif: TimeInForce::Day,
                strategy: "hurst_regime".to_string(),
                tag,
            });
        }
        intents
    }
}

fn position_qty(positions: &[PositionView], sym: Symbol) -> f64 {
    positions.iter().find(|p| p.symbol == sym).map(|p| p.qty).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{mk_bar, pv_with_nav};

    fn run_series(returns: Vec<f64>) -> Vec<OrderIntent> {
        let mut s = HurstRegime::new(HurstRegimeConfig {
            hurst_window: 64,
            momentum_half_life: 10.0,
            trend_threshold: 0.55,
            mr_threshold: 0.45,
            gross_per_name: 0.05,
            rebalance_every_bars: 1,
        });
        let sym = Symbol::new("TESTSYM").unwrap();
        let mut px = 100.0_f64;
        let mut all = Vec::new();
        for (i, r) in returns.iter().enumerate() {
            px = (px * (1.0 + r)).max(0.01);
            let ts = (i as i64 + 1) * 60_000_000_000;
            let intents = s.on_event(&mk_bar(sym, ts, px), &pv_with_nav(ts, 1_000_000.0), &[]);
            all.extend(intents);
        }
        all
    }

    #[test]
    fn trending_series_produces_orders() {
        // Positive trend with small noise (R/S needs return variance).
        // After hurst_window=64 + warmup, the strategy should fire orders.
        let trending: Vec<f64> = (0..300)
            .map(|i| 0.003 + 0.002 * ((i as f64 * 0.27).sin()))
            .collect();
        let intents = run_series(trending);
        assert!(!intents.is_empty(), "trending series should generate at least one order");
    }

    #[test]
    fn validate_config_rejects_misordered_thresholds() {
        let s = HurstRegime::new(HurstRegimeConfig {
            trend_threshold: 0.40,
            mr_threshold: 0.60,  // mr > trend — invalid
            ..HurstRegimeConfig::default()
        });
        assert!(s.validate_config().is_err());
    }

    #[test]
    fn validate_config_rejects_small_hurst_window() {
        let s = HurstRegime::new(HurstRegimeConfig {
            hurst_window: 16,  // < 32 minimum
            ..HurstRegimeConfig::default()
        });
        assert!(s.validate_config().is_err());
    }
}
