//! Cross-sectional momentum + volatility scaling.
//!
//! Optimized hot path:
//!   - All per-event scratch buffers live on `self` and are cleared (not
//!     reallocated) between events: `sym_buf`, `score_buf`, `price_scratch`,
//!     `desired_buf`, `live_pos_buf`.
//!   - `live_pos` reconciliation walks the broker `positions` slice directly
//!     instead of building an `AHashSet`.
//!   - Score computation uses a single Welford-style streaming pass over the
//!     last-N price window via `Window::last_n_into`, then a single Welford
//!     reduction over the resulting returns. No intermediate `Array1` clones.
//!   - Z-scoring runs in-place on the score vector.

use ahash::AHashMap;
use algo_core::{
    Bar, MarketEvent, OrderId, OrderIntent, OrderType, Qty, Side, Symbol, TimeInForce,
};
use algo_features::{zscore_vec_into, Window};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XsMomentumConfig {
    pub lookback_bars: usize,
    pub vol_lookback_bars: usize,
    pub gross_per_leg: f64,
    pub legs_per_side: usize,
    pub rebalance_every_bars: u32,
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
    name: &'static str,
    price_hist: AHashMap<Symbol, Window<f64>>,
    bars_since_rebal: u32,
    /// Map of currently-open positions the strategy has emitted orders for.
    /// Value: (signed_dir, bars_held).
    bars_held: AHashMap<Symbol, (i8, u32)>,

    // Hot-path scratch buffers. Cleared, not freed.
    sym_buf: Vec<Symbol>,
    score_buf: Vec<f64>,
    price_scratch: Vec<f64>,
    /// `desired[sym] = ±1`. Cleared per rebalance.
    desired_buf: AHashMap<Symbol, i8>,
    /// Indexes into `sym_buf` ordered by descending score.
    sort_idx: Vec<usize>,
}

impl XsMomentum {
    pub fn new(cfg: XsMomentumConfig) -> Self {
        Self {
            cfg,
            name: "xs_momentum",
            price_hist: AHashMap::new(),
            bars_since_rebal: 0,
            bars_held: AHashMap::new(),
            sym_buf: Vec::with_capacity(64),
            score_buf: Vec::with_capacity(64),
            price_scratch: Vec::with_capacity(64),
            desired_buf: AHashMap::with_capacity(8),
            sort_idx: Vec::with_capacity(64),
        }
    }

    #[inline]
    fn ingest(&mut self, bar: &Bar) {
        let cap = self.cfg.lookback_bars.max(self.cfg.vol_lookback_bars) + 2;
        let w = self
            .price_hist
            .entry(bar.symbol)
            .or_insert_with(|| Window::new(cap));
        w.push(bar.ts, bar.close.to_f64());
    }

    /// Compute cross-sectional momentum scores into `self.sym_buf` /
    /// `self.score_buf`. Returns the number of scored symbols.
    fn compute_scores(&mut self) -> usize {
        let lookback = self.cfg.lookback_bars;
        self.sym_buf.clear();
        self.score_buf.clear();

        for (&sym, w) in &self.price_hist {
            if !w.last_n_into(lookback + 1, &mut self.price_scratch) {
                continue;
            }
            let prices = &self.price_scratch;
            // Reject any non-positive / non-finite price as bad data.
            if prices.iter().any(|p| !p.is_finite() || *p <= 0.0) {
                continue;
            }
            // Streaming pass: compute log-returns and Welford mean/variance in one walk.
            let mut prev_log = prices[0].ln();
            let mut momo = 0.0_f64;
            // Welford on returns
            let mut n = 0u64;
            let mut mean = 0.0_f64;
            let mut m2 = 0.0_f64;
            for &p in &prices[1..] {
                let l = p.ln();
                let r = l - prev_log;
                momo += r;
                n += 1;
                let delta = r - mean;
                mean += delta / n as f64;
                let delta2 = r - mean;
                m2 += delta * delta2;
                prev_log = l;
            }
            if n < 2 {
                continue;
            }
            let var = m2 / (n - 1) as f64;
            let vol = var.sqrt().max(1e-9);
            let score = momo / vol;
            self.sym_buf.push(sym);
            self.score_buf.push(score);
        }
        if self.sym_buf.is_empty() {
            return 0;
        }
        // Z-score in place
        let mut arr = Array1::from_vec(std::mem::take(&mut self.score_buf));
        zscore_vec_into(&mut arr);
        self.score_buf = arr.into_raw_vec_and_offset().0;
        self.sym_buf.len()
    }
}

impl Strategy for XsMomentum {
    fn name(&self) -> &'static str {
        self.name
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

        // Reconcile bars_held against actual broker positions. O(open_positions × pos.len)
        // which is tiny in practice (≤ legs_per_side*2 entries).
        self.bars_held.retain(|sym, _| {
            positions
                .iter()
                .any(|p| p.symbol == *sym && p.qty.abs() > 1e-9)
        });
        for (_, (_, n)) in self.bars_held.iter_mut() {
            *n += 1;
        }

        if self.bars_since_rebal < self.cfg.rebalance_every_bars {
            return SmallVec::new();
        }
        self.bars_since_rebal = 0;

        let n_scored = self.compute_scores();
        let need = (self.cfg.legs_per_side * 2).max(2);
        if n_scored < need {
            return SmallVec::new();
        }

        // Argsort score descending using a scratch index vector.
        self.sort_idx.clear();
        self.sort_idx.extend(0..n_scored);
        let scores = &self.score_buf;
        self.sort_idx.sort_unstable_by(|&i, &j| {
            scores[j]
                .partial_cmp(&scores[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build desired{sym -> ±1} via reused buffer.
        self.desired_buf.clear();
        for &i in self.sort_idx.iter().take(self.cfg.legs_per_side) {
            self.desired_buf.insert(self.sym_buf[i], 1);
        }
        for &i in self.sort_idx.iter().rev().take(self.cfg.legs_per_side) {
            self.desired_buf.insert(self.sym_buf[i], -1);
        }

        // Force-flat any held position that exceeded max_hold_bars.
        let max_hold = self.cfg.max_hold_bars;
        for (sym, (_, held)) in self.bars_held.iter() {
            if *held >= max_hold {
                self.desired_buf.remove(sym);
            }
        }

        let leg_notional = portfolio.nav * self.cfg.gross_per_leg;
        let mut intents: SmallVec<[OrderIntent; 4]> = SmallVec::new();

        // Walk the union (desired ∪ current_open_positions). Both sets are
        // tiny so a small linear pass is fastest.
        // First: positions that the strategy wants opened/resized/closed via desired.
        let mut seen_in_desired = SmallVec::<[Symbol; 16]>::new();
        for (&sym, &want_dir) in self.desired_buf.iter() {
            seen_in_desired.push(sym);
            let have_qty = position_qty(positions, sym);
            let last_px = self
                .price_hist
                .get(&sym)
                .and_then(|w| w.last().map(|(_, p)| *p))
                .unwrap_or(0.0);
            if last_px <= 0.0 {
                continue;
            }
            let want_qty = (leg_notional / last_px) * want_dir as f64;
            let delta = want_qty - have_qty;
            if delta.abs() < 1.0 {
                continue;
            }
            push_intent(&mut intents, sym, delta, portfolio.now, self.name);
            self.bars_held.insert(sym, (want_dir, 0));
        }
        // Second: existing positions NOT in desired → flatten them.
        for p in positions.iter() {
            if p.qty.abs() < 1e-9 || seen_in_desired.contains(&p.symbol) {
                continue;
            }
            let delta = -p.qty;
            if delta.abs() < 1.0 {
                continue;
            }
            push_intent(&mut intents, p.symbol, delta, portfolio.now, self.name);
            self.bars_held.remove(&p.symbol);
        }
        intents
    }
}

#[inline]
fn position_qty(positions: &[PositionView], sym: Symbol) -> f64 {
    positions
        .iter()
        .find(|p| p.symbol == sym)
        .map(|p| p.qty)
        .unwrap_or(0.0)
}

#[inline]
fn push_intent(
    out: &mut SmallVec<[OrderIntent; 4]>,
    sym: Symbol,
    delta_qty: f64,
    now: algo_core::Ts,
    strategy: &'static str,
) {
    let side = if delta_qty > 0.0 { Side::Buy } else { Side::Sell };
    let qty = (delta_qty.abs().floor()).max(1.0) as i64;
    out.push(OrderIntent {
        id: OrderId::new(),
        ts: now,
        symbol: sym,
        side,
        qty: Qty::from_i64(qty),
        order_type: OrderType::Market,
        tif: TimeInForce::Day,
        strategy: strategy.to_string(),
        tag: None,
    });
}
