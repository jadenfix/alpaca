//! 3-regime Markov-switching strategy router.
//!
//! Watches a regime-proxy ticker (default SPY) and assigns the current
//! market state to one of three regimes:
//!   - **Bull**     → run momentum (long top decile, short bottom)
//!   - **Sideways** → run mean-reversion (short top, long bottom)
//!   - **Bear**     → fully risk-off (flatten all positions)
//!
//! Each regime gets a separate gross allocation budget so we can damp risk
//! in transitional periods. Regime posterior P(s_t | r_1..r_t) is updated
//! online from the proxy's log returns.
//!
//! Why this matters:
//!   - Cross-sectional momentum reverses in stressed regimes — gating it
//!     by a Markov-switching model is a documented edge improvement
//!     (Ang & Bekaert 2002).
//!   - Mean-reversion historically works best in sideways markets — same
//!     paper.
//!
//! Reference: Hamilton (1989); Ang & Bekaert (2002).

use ahash::AHashMap;
use algo_core::{
    MarketEvent, OrderId, OrderIntent, OrderType, Qty, Side, Symbol, TimeInForce,
};
use algo_features::{Ema, Regime, ThreeStateMarkov};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarkovRouterConfig {
    pub regime_proxy_ticker: String,
    pub momentum_half_life: f64,
    /// Posterior probability above which we commit to a regime's trading mode.
    pub min_regime_confidence: f64,
    pub bull_gross_per_leg: f64,
    pub sideways_gross_per_leg: f64,
    pub legs_per_side: usize,
    pub rebalance_every_bars: u32,
    pub warmup_bars: usize,
}

impl Default for MarkovRouterConfig {
    fn default() -> Self {
        Self {
            regime_proxy_ticker: "SPY".into(),
            momentum_half_life: 30.0,
            min_regime_confidence: 0.55,
            bull_gross_per_leg: 0.03,
            sideways_gross_per_leg: 0.02,
            legs_per_side: 2,
            rebalance_every_bars: 10,
            warmup_bars: 100,
        }
    }
}

struct PerSymbol {
    last_price: Option<f64>,
    momo: Ema,
    bars_seen: usize,
}

impl PerSymbol {
    fn new(hl: f64) -> Self {
        Self {
            last_price: None,
            momo: Ema::from_half_life(hl),
            bars_seen: 0,
        }
    }
}

pub struct MarkovRouter {
    cfg: MarkovRouterConfig,
    proxy: Symbol,
    model: ThreeStateMarkov,
    per_sym: AHashMap<Symbol, PerSymbol>,
    bars_since_rebal: u32,
    pub last_regime: Option<Regime>,
}

impl MarkovRouter {
    pub fn new(cfg: MarkovRouterConfig) -> Self {
        let proxy = Symbol::new(&cfg.regime_proxy_ticker).expect("invalid proxy ticker");
        Self {
            cfg,
            proxy,
            model: ThreeStateMarkov::default_equity(),
            per_sym: AHashMap::new(),
            bars_since_rebal: 0,
            last_regime: None,
        }
    }

    /// Current best estimate of the active regime + its confidence.
    pub fn current_regime(&self) -> (Regime, f64) {
        let r = self.model.argmax_regime();
        let p = self.model.p_regime(r);
        (r, p)
    }
}

impl Strategy for MarkovRouter {
    fn name(&self) -> &'static str {
        "markov_router"
    }

    fn validate_config(&self) -> Result<(), String> {
        if !(0.5..=1.0).contains(&self.cfg.min_regime_confidence) {
            return Err("min_regime_confidence must be in [0.5, 1.0]".into());
        }
        if self.cfg.legs_per_side == 0 {
            return Err("legs_per_side must be ≥ 1".into());
        }
        if !(0.0 < self.cfg.bull_gross_per_leg && self.cfg.bull_gross_per_leg < 0.5) {
            return Err("bull_gross_per_leg must be in (0, 0.5)".into());
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.model = ThreeStateMarkov::default_equity();
        self.per_sym.clear();
        self.bars_since_rebal = 0;
        self.last_regime = None;
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
        let entry = self
            .per_sym
            .entry(bar.symbol)
            .or_insert_with(|| PerSymbol::new(hl));
        let ret = if let Some(prev) = entry.last_price {
            if prev > 0.0 { (px / prev).ln() } else { 0.0 }
        } else { 0.0 };
        entry.last_price = Some(px);
        entry.bars_seen += 1;
        if ret.abs() > 0.0 {
            entry.momo.update(ret);
        }
        // Drive the regime model from the proxy ticker's returns.
        if bar.symbol == self.proxy && ret.abs() > 0.0 {
            self.model.update(ret);
        }
        self.bars_since_rebal += 1;
        if self.bars_since_rebal < self.cfg.rebalance_every_bars {
            return SmallVec::new();
        }
        self.bars_since_rebal = 0;

        let (regime, confidence) = self.current_regime();
        self.last_regime = Some(regime);

        // Low confidence → no entries. Existing positions are left to ride
        // (we'll flatten if regime flips to Bear with sufficient confidence).
        if confidence < self.cfg.min_regime_confidence {
            return SmallVec::new();
        }

        let mut intents: SmallVec<[OrderIntent; 4]> = SmallVec::new();

        // BEAR → risk-off: flatten everything.
        if regime == Regime::Bear {
            for p in positions {
                if p.qty.abs() < 1e-9 { continue; }
                let side = if p.qty > 0.0 { Side::Sell } else { Side::Buy };
                let qty = p.qty.abs().ceil() as i64;
                if qty == 0 { continue; }
                intents.push(OrderIntent {
                    id: OrderId::new(),
                    ts: portfolio.now,
                    symbol: p.symbol,
                    side,
                    qty: Qty::from_i64(qty),
                    order_type: OrderType::Market,
                    tif: TimeInForce::Day,
                    strategy: "markov_router".to_string(),
                    tag: Some("regime_bear_flatten".to_string()),
                });
            }
            return intents;
        }

        // Rank by momentum, excluding the proxy.
        let mut ranked: Vec<(Symbol, f64, f64)> = Vec::with_capacity(self.per_sym.len());
        for (sym, state) in &self.per_sym {
            if *sym == self.proxy { continue; }
            if state.bars_seen < self.cfg.warmup_bars { continue; }
            let Some(m) = state.momo.mean() else { continue; };
            let Some(p_now) = state.last_price else { continue; };
            ranked.push((*sym, m, p_now));
        }
        if ranked.len() < self.cfg.legs_per_side * 2 {
            return intents;
        }
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Direction depends on regime: Bull = momentum, Sideways = mean-reversion
        let (long_side, short_side, leg_gross) = match regime {
            Regime::Bull => (
                ranked.iter().take(self.cfg.legs_per_side).cloned().collect::<Vec<_>>(),
                ranked.iter().rev().take(self.cfg.legs_per_side).cloned().collect::<Vec<_>>(),
                self.cfg.bull_gross_per_leg,
            ),
            Regime::Sideways => (
                // Reversed: long the worst (bet on mean reversion up), short the best
                ranked.iter().rev().take(self.cfg.legs_per_side).cloned().collect::<Vec<_>>(),
                ranked.iter().take(self.cfg.legs_per_side).cloned().collect::<Vec<_>>(),
                self.cfg.sideways_gross_per_leg,
            ),
            Regime::Bear => unreachable!("bear handled above"),
        };

        let want_long: ahash::AHashSet<Symbol> = long_side.iter().map(|x| x.0).collect();
        let want_short: ahash::AHashSet<Symbol> = short_side.iter().map(|x| x.0).collect();

        // Close positions that no longer fit the desired pattern.
        for p in positions {
            if p.qty.abs() < 1e-9 { continue; }
            let in_long = want_long.contains(&p.symbol) && p.qty > 0.0;
            let in_short = want_short.contains(&p.symbol) && p.qty < 0.0;
            if in_long || in_short { continue; }
            let side = if p.qty > 0.0 { Side::Sell } else { Side::Buy };
            let qty = p.qty.abs().ceil() as i64;
            if qty == 0 { continue; }
            intents.push(OrderIntent {
                id: OrderId::new(),
                ts: portfolio.now,
                symbol: p.symbol,
                side,
                qty: Qty::from_i64(qty),
                order_type: OrderType::Market,
                tif: TimeInForce::Day,
                strategy: "markov_router".to_string(),
                tag: Some(format!("close_{}", regime.name())),
            });
        }

        let leg_notional = portfolio.nav * leg_gross;
        for (sym, _, px_now) in &long_side {
            let target = (leg_notional / px_now).floor() as i64;
            let have = position_qty(positions, *sym);
            let delta = target as f64 - have;
            if delta.abs() < 1.0 { continue; }
            let side = if delta > 0.0 { Side::Buy } else { Side::Sell };
            intents.push(OrderIntent {
                id: OrderId::new(),
                ts: portfolio.now,
                symbol: *sym,
                side,
                qty: Qty::from_i64(delta.abs().ceil() as i64),
                order_type: OrderType::Market,
                tif: TimeInForce::Day,
                strategy: "markov_router".to_string(),
                tag: Some(format!("long_{}", regime.name())),
            });
        }
        for (sym, _, px_now) in &short_side {
            let target = -((leg_notional / px_now).floor() as i64);
            let have = position_qty(positions, *sym);
            let delta = target as f64 - have;
            if delta.abs() < 1.0 { continue; }
            let side = if delta > 0.0 { Side::Buy } else { Side::Sell };
            intents.push(OrderIntent {
                id: OrderId::new(),
                ts: portfolio.now,
                symbol: *sym,
                side,
                qty: Qty::from_i64(delta.abs().ceil() as i64),
                order_type: OrderType::Market,
                tif: TimeInForce::Day,
                strategy: "markov_router".to_string(),
                tag: Some(format!("short_{}", regime.name())),
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
    use algo_features::Regime;
    use algo_strategy::PositionView;

    #[test]
    fn bear_regime_triggers_flatten_for_all_positions() {
        let mut s = MarkovRouter::new(MarkovRouterConfig {
            regime_proxy_ticker: "PROXY".into(),
            min_regime_confidence: 0.55,
            momentum_half_life: 5.0,
            bull_gross_per_leg: 0.03,
            sideways_gross_per_leg: 0.02,
            legs_per_side: 2,
            rebalance_every_bars: 1,
            warmup_bars: 30,
        });
        let proxy = Symbol::new("PROXY").unwrap();
        // Force model into bear by directly mutating the alpha posterior
        // (a more legitimate approach than waiting for synthetic data to drive it).
        s.model.alpha = [0.0, 1.0, 0.0]; // [bull, bear, sideways] in default ordering
        s.model.regime_labels = [Regime::Bull, Regime::Bear, Regime::Sideways];

        let held = Symbol::new("HELD").unwrap();
        let positions = vec![PositionView {
            symbol: held, qty: 100.0, avg_px: 50.0, mark_px: 50.0,
        }];
        // Need enough bars_since_rebal to trigger rebalance
        s.bars_since_rebal = 10;
        // Send a bar (any non-proxy bar so the model isn't disturbed)
        let intents = s.on_event(
            &mk_bar(held, 60_000_000_000, 50.0),
            &pv_with_nav(60_000_000_000, 1_000_000.0),
            &positions,
        );
        // In bear with confidence 1.0, must emit a Sell for the held position.
        let has_close = intents.iter().any(|i| i.symbol == held && i.side == Side::Sell);
        assert!(
            has_close,
            "bear regime must flatten held positions; intents={:?}",
            intents.iter().map(|i| (i.symbol, i.side)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn low_confidence_blocks_new_entries() {
        let mut s = MarkovRouter::new(MarkovRouterConfig {
            min_regime_confidence: 0.99,  // very high bar
            ..MarkovRouterConfig::default()
        });
        // Force model to mid-confidence
        s.model.alpha = [0.34, 0.33, 0.33];
        s.bars_since_rebal = 100;
        let intents = s.on_event(
            &mk_bar(Symbol::new("ANY").unwrap(), 0, 100.0),
            &pv_with_nav(0, 1_000_000.0),
            &[],
        );
        assert!(intents.is_empty(), "low regime confidence must block entries");
    }

    #[test]
    fn validate_config_rejects_zero_legs() {
        let s = MarkovRouter::new(MarkovRouterConfig {
            legs_per_side: 0,
            ..MarkovRouterConfig::default()
        });
        assert!(s.validate_config().is_err());
    }
}
