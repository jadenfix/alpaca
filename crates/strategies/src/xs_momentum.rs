//! Cross-sectional momentum + volatility scaling.
//!
//! On each bar close: compute trailing N-bar return per symbol → divide by
//! trailing realized vol → z-score across the universe → long top decile,
//! short bottom decile (or top-K / bottom-K for small universes).
//!
//! Exit: signal flips (z-score sign change) or `max_hold` bars.

use ahash::AHashMap;
use algo_core::{
    Bar, MarketEvent, OrderId, OrderIntent, OrderType, Qty, Side, Symbol, TimeInForce,
};
use algo_features::{log_returns_vec, zscore_vec_into, Window};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XsMomentumConfig {
    pub lookback_bars: usize,
    pub vol_lookback_bars: usize,
    /// Per-leg gross weight as fraction of NAV.
    pub gross_per_leg: f64,
    /// How many top / bottom names to trade (≥1).
    pub legs_per_side: usize,
    /// Re-evaluate signal every N bars (default 1: every close).
    pub rebalance_every_bars: u32,
    /// Hard max-hold; force flatten after this many bars.
    pub max_hold_bars: u32,
}

impl Default for XsMomentumConfig {
    fn default() -> Self {
        Self {
            lookback_bars: 20,
            vol_lookback_bars: 20,
            gross_per_leg: 0.02,
            legs_per_side: 2,
            rebalance_every_bars: 1,
            max_hold_bars: 60,
        }
    }
}

pub struct XsMomentum {
    cfg: XsMomentumConfig,
    /// Per-symbol price history (close).
    price_hist: AHashMap<Symbol, Window<f64>>,
    /// Bars since last rebalance.
    bars_since_rebal: u32,
    /// Bars-held per current open name (signed direction).
    bars_held: AHashMap<Symbol, (i8, u32)>,
}

impl XsMomentum {
    pub fn new(cfg: XsMomentumConfig) -> Self {
        Self {
            cfg,
            price_hist: AHashMap::new(),
            bars_since_rebal: 0,
            bars_held: AHashMap::new(),
        }
    }

    fn ingest(&mut self, bar: &Bar) {
        let cap = self.cfg.lookback_bars.max(self.cfg.vol_lookback_bars) + 2;
        let w = self
            .price_hist
            .entry(bar.symbol)
            .or_insert_with(|| Window::new(cap));
        w.push(bar.ts, bar.close.to_f64());
    }

    /// Compute cross-sectional momentum scores for every symbol with enough history.
    /// Score = (lookback log-return) / (lookback realized stddev). Then z-score across syms.
    fn scores(&self) -> Vec<(Symbol, f64)> {
        let lookback = self.cfg.lookback_bars;
        let mut syms = Vec::new();
        let mut raw = Vec::new();
        for (&sym, w) in &self.price_hist {
            let Some(prices) = w.last_n(lookback + 1) else {
                continue;
            };
            if prices.iter().any(|p| !p.is_finite() || *p <= 0.0) {
                continue;
            }
            let pv = Array1::from(prices);
            let rets = log_returns_vec(pv.view());
            let momo = rets.sum();
            // Bessel-corrected vol on the same returns vector.
            let n = rets.len() as f64;
            if n < 2.0 {
                continue;
            }
            let mean = rets.mean().unwrap_or(0.0);
            let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
            let vol = var.sqrt().max(1e-9);
            let score = momo / vol;
            syms.push(sym);
            raw.push(score);
        }
        if syms.is_empty() {
            return Vec::new();
        }
        let mut arr = Array1::from(raw);
        zscore_vec_into(&mut arr);
        syms.into_iter().zip(arr.into_iter()).collect()
    }
}

impl Strategy for XsMomentum {
    fn name(&self) -> &'static str {
        "xs_momentum"
    }

    fn on_event(
        &mut self,
        event: &MarketEvent,
        portfolio: &PortfolioView,
        positions: &[PositionView],
    ) -> SmallVec<[OrderIntent; 4]> {
        let MarketEvent::Bar(bar) = event else {
            return SmallVec::new();
        };
        self.ingest(bar);
        self.bars_since_rebal += 1;

        // Reconcile bars_held against the broker's actual positions. This is the
        // single source of truth — strategy's belief follows reality. Prevents
        // phantom positions (e.g. after a process restart, walk-forward fold
        // reset, or any time broker state diverges from strategy state).
        let live_pos: ahash::AHashSet<Symbol> = positions
            .iter()
            .filter(|p| p.qty.abs() > 1e-9)
            .map(|p| p.symbol)
            .collect();
        self.bars_held.retain(|sym, _| live_pos.contains(sym));
        // Tick the held counter for currently-open names only.
        for (_, (_, n)) in self.bars_held.iter_mut() {
            *n += 1;
        }
        if self.bars_since_rebal < self.cfg.rebalance_every_bars {
            return SmallVec::new();
        }
        self.bars_since_rebal = 0;

        let scores = self.scores();
        if scores.len() < (self.cfg.legs_per_side * 2).max(2) {
            return SmallVec::new();
        }

        let mut sorted = scores;
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let longs: Vec<Symbol> = sorted
            .iter()
            .take(self.cfg.legs_per_side)
            .map(|(s, _)| *s)
            .collect();
        let shorts: Vec<Symbol> = sorted
            .iter()
            .rev()
            .take(self.cfg.legs_per_side)
            .map(|(s, _)| *s)
            .collect();

        let mut intents: SmallVec<[OrderIntent; 4]> = SmallVec::new();
        let leg_notional = portfolio.nav * self.cfg.gross_per_leg;

        // Build desired positions
        let mut desired: AHashMap<Symbol, i8> = AHashMap::new();
        for s in &longs {
            desired.insert(*s, 1);
        }
        for s in &shorts {
            desired.insert(*s, -1);
        }

        // Time-based exit: if held longer than max_hold, force exit (override of desired).
        let mut force_flat: Vec<Symbol> = Vec::new();
        for (sym, (dir, held)) in &self.bars_held {
            if *held >= self.cfg.max_hold_bars {
                force_flat.push(*sym);
                let _ = dir;
            }
        }
        for s in &force_flat {
            desired.remove(s);
        }

        // Compute order intents to move from current positions to desired ones.
        let pos_map: AHashMap<Symbol, f64> = positions
            .iter()
            .map(|p| (p.symbol, p.qty))
            .collect();
        let all_symbols: ahash::AHashSet<Symbol> = desired
            .keys()
            .copied()
            .chain(pos_map.keys().copied())
            .collect();
        for sym in all_symbols {
            let want_dir = desired.get(&sym).copied().unwrap_or(0);
            let have_qty = pos_map.get(&sym).copied().unwrap_or(0.0);
            let want_qty = if want_dir != 0 {
                // approximate by latest close
                let px = self
                    .price_hist
                    .get(&sym)
                    .and_then(|w| w.last().map(|(_, p)| *p))
                    .unwrap_or(0.0);
                if px <= 0.0 {
                    continue;
                }
                (leg_notional / px) * want_dir as f64
            } else {
                0.0
            };
            let delta = want_qty - have_qty;
            if delta.abs() < 1.0 {
                continue;
            }
            let side = if delta > 0.0 { Side::Buy } else { Side::Sell };
            let qty = (delta.abs().floor()).max(1.0) as i64;
            intents.push(OrderIntent {
                id: OrderId::new(),
                ts: portfolio.now,
                symbol: sym,
                side,
                qty: Qty::from_i64(qty),
                order_type: OrderType::Market,
                tif: TimeInForce::Day,
                strategy: self.name().to_string(),
                tag: None,
            });
            // Track new position direction & reset held counter.
            if want_dir != 0 {
                self.bars_held.insert(sym, (want_dir as i8, 0));
            } else {
                self.bars_held.remove(&sym);
            }
        }
        intents
    }
}

impl XsMomentum {
    fn _kelly_silence_warning(_p: &PositionView) -> SmallVec<[OrderIntent; 4]> {
        smallvec![]
    }
}
