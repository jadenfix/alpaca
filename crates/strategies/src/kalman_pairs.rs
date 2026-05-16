//! Kalman-filter pairs trading: hedge ratio β_t evolves as a random walk,
//! reducing the brittleness of a static OLS estimate when the relationship
//! between assets shifts over time.
//!
//! Spread = y_t - β̂_t · x_t (running Kalman estimate). We z-score the
//! filter's innovation against its predicted variance and trade when the
//! score exceeds an entry threshold.
//!
//! Reference: Chan (2013) Ch. 3 "Algorithmic Trading".

use algo_core::{
    MarketEvent, OrderId, OrderIntent, OrderType, Qty, Side, Symbol, TimeInForce, Ts,
};
use algo_features::{Ema, ScalarKalman};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KalmanPairsConfig {
    pub symbol_a: String, // y
    pub symbol_b: String, // x
    pub process_noise_q: f64,
    pub observation_noise_r: f64,
    pub entry_z: f64,
    pub exit_z: f64,
    pub warmup_bars: usize,
    pub gross_per_leg: f64,
    pub max_hold_bars: u32,
}

impl Default for KalmanPairsConfig {
    fn default() -> Self {
        Self {
            symbol_a: "PAIRA".into(),
            symbol_b: "PAIRB".into(),
            // Q VERY small: we want a near-static β so the OU spread signal
            // shows up as residuals, not absorbed into β drift. Real pairs
            // re-estimate β much more slowly than they drift.
            process_noise_q: 1e-9,
            observation_noise_r: 1e-5,
            entry_z: 1.5,
            exit_z: 0.3,
            warmup_bars: 100,
            gross_per_leg: 0.02,
            max_hold_bars: 100,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Pos {
    Flat,
    Long,
    Short,
}

pub struct KalmanPairs {
    cfg: KalmanPairsConfig,
    sym_a: Symbol,
    sym_b: Symbol,
    kf: ScalarKalman,
    /// Empirical EWMA mean of innovations (should be ~0 after warmup).
    innov_mean: Ema,
    /// EWMA of innovation squared, for empirical variance.
    innov_sq_mean: Ema,
    last_a: Option<(Ts, f64)>,
    last_b: Option<(Ts, f64)>,
    bars_seen: usize,
    pos: Pos,
    bars_in_trade: u32,
}

impl KalmanPairs {
    pub fn new(cfg: KalmanPairsConfig) -> Self {
        let sym_a = Symbol::new(&cfg.symbol_a).expect("invalid sym_a");
        let sym_b = Symbol::new(&cfg.symbol_b).expect("invalid sym_b");
        let kf = ScalarKalman::new(cfg.process_noise_q, cfg.observation_noise_r);
        // EMA half-life of ~50 bars — longer than typical mean-reversion cycle
        // so we measure the spread's natural variability, not the trade-window noise.
        let hl = cfg.warmup_bars as f64 * 0.5;
        Self {
            cfg,
            sym_a,
            sym_b,
            kf,
            innov_mean: Ema::from_half_life(hl.max(20.0)),
            innov_sq_mean: Ema::from_half_life(hl.max(20.0)),
            last_a: None,
            last_b: None,
            bars_seen: 0,
            pos: Pos::Flat,
            bars_in_trade: 0,
        }
    }
}

impl Strategy for KalmanPairs {
    fn name(&self) -> &'static str {
        "kalman_pairs"
    }

    fn validate_config(&self) -> Result<(), String> {
        if self.cfg.process_noise_q <= 0.0 || self.cfg.observation_noise_r <= 0.0 {
            return Err("Q and R must be > 0".into());
        }
        if self.cfg.entry_z <= self.cfg.exit_z {
            return Err("entry_z must exceed exit_z".into());
        }
        if self.cfg.warmup_bars < 20 {
            return Err("warmup_bars must be ≥ 20".into());
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.kf = ScalarKalman::new(self.cfg.process_noise_q, self.cfg.observation_noise_r);
        self.last_a = None;
        self.last_b = None;
        self.bars_seen = 0;
        self.pos = Pos::Flat;
        self.bars_in_trade = 0;
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
        // Use log prices — scale-free hedge ratio.
        let log_px = px.ln();
        if bar.symbol == self.sym_a {
            self.last_a = Some((bar.ts, log_px));
        } else if bar.symbol == self.sym_b {
            self.last_b = Some((bar.ts, log_px));
        } else {
            return SmallVec::new();
        }
        // Only update the KF when both legs have observations at the SAME timestamp
        // (or within one bar of each other for slightly mis-aligned feeds).
        let (Some((ta, a)), Some((tb, b))) = (self.last_a, self.last_b) else {
            return SmallVec::new();
        };
        if ta != tb {
            return SmallVec::new();
        }
        let step = self.kf.update(a, b);
        self.bars_seen += 1;
        // Track empirical mean and variance of innovations for a robust
        // data-driven z-score (KF's predicted variance is the *measurement*
        // uncertainty, not the spread's natural distribution).
        self.innov_mean.update(step.spread);
        self.innov_sq_mean.update(step.spread * step.spread);
        if self.pos != Pos::Flat {
            self.bars_in_trade += 1;
        }
        if self.bars_seen < self.cfg.warmup_bars {
            return SmallVec::new();
        }
        // Empirical z-score: (innovation - rolling_mean) / rolling_std.
        let (Some(mu), Some(sq_mu)) = (self.innov_mean.mean(), self.innov_sq_mean.mean()) else {
            return SmallVec::new();
        };
        let var = (sq_mu - mu * mu).max(1e-18);
        let std = var.sqrt();
        let z = (step.spread - mu) / std;
        let beta = self.kf.beta;
        let pos_a = position_qty(positions, self.sym_a);
        let pos_b = position_qty(positions, self.sym_b);

        let mut intents: SmallVec<[OrderIntent; 4]> = SmallVec::new();
        let max_hold = self.cfg.max_hold_bars;
        let should_exit = self.pos != Pos::Flat
            && (z.abs() < self.cfg.exit_z || self.bars_in_trade >= max_hold);
        if should_exit {
            if pos_a.abs() > 0.0 {
                let side = if pos_a > 0.0 { Side::Sell } else { Side::Buy };
                push_intent(&mut intents, self.sym_a, side, pos_a.abs(), portfolio.now);
            }
            if pos_b.abs() > 0.0 {
                let side = if pos_b > 0.0 { Side::Sell } else { Side::Buy };
                push_intent(&mut intents, self.sym_b, side, pos_b.abs(), portfolio.now);
            }
            self.pos = Pos::Flat;
            self.bars_in_trade = 0;
            return intents;
        }
        if self.pos != Pos::Flat {
            return intents;
        }
        let want = if z > self.cfg.entry_z {
            Some(Pos::Short)
        } else if z < -self.cfg.entry_z {
            Some(Pos::Long)
        } else {
            None
        };
        let Some(want) = want else { return intents; };
        // Use linear prices (exp of log) for share sizing.
        let leg = portfolio.nav * self.cfg.gross_per_leg;
        let px_a = a.exp();
        let px_b = b.exp();
        let qty_a = (leg / px_a).max(1.0).floor() as i64;
        let qty_b = ((leg * beta.abs()) / px_b).max(1.0).floor() as i64;
        let (side_a, side_b) = match want {
            Pos::Long => (Side::Buy, Side::Sell),
            Pos::Short => (Side::Sell, Side::Buy),
            _ => unreachable!(),
        };
        push_intent(&mut intents, self.sym_a, side_a, qty_a as f64, portfolio.now);
        push_intent(&mut intents, self.sym_b, side_b, qty_b as f64, portfolio.now);
        self.pos = want;
        self.bars_in_trade = 0;
        intents
    }
}

fn position_qty(positions: &[PositionView], sym: Symbol) -> f64 {
    positions
        .iter()
        .find(|p| p.symbol == sym)
        .map(|p| p.qty)
        .unwrap_or(0.0)
}

fn push_intent(
    out: &mut SmallVec<[OrderIntent; 4]>,
    sym: Symbol,
    side: Side,
    qty: f64,
    ts: Ts,
) {
    let q = qty.abs().ceil() as i64;
    if q == 0 {
        return;
    }
    out.push(OrderIntent {
        id: OrderId::new(),
        ts,
        symbol: sym,
        side,
        qty: Qty::from_i64(q),
        order_type: OrderType::Market,
        tif: TimeInForce::Day,
        strategy: "kalman_pairs".to_string(),
        tag: None,
    });
}
