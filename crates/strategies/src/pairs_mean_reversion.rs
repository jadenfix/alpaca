//! Pairs / cointegration mean-reversion (Avellaneda-Lee 2010 style).
//!
//! Algorithm:
//!   1. Collect bars for two symbols. Once we have `min_history` bars each,
//!      run Engle-Granger cointegration: y = α + β x + ε.
//!   2. If ADF p-value on ε < `entry_alpha_level`, the pair is cointegrated.
//!      Estimate the OU half-life of ε; require half_life ∈ [min_hl, max_hl].
//!   3. Trade on z-score of the spread:
//!        z > entry_z       → short spread (sell y, buy β·x)
//!        z < -entry_z      → long spread  (buy y, sell β·x)
//!        |z| < exit_z      → close
//!      Force exit if held longer than `2 × half_life`.
//!
//! Holds at most one position per pair at a time. Trade sizing splits
//! `gross_per_leg` evenly between the two legs.

use algo_core::{
    Bar, MarketEvent, OrderId, OrderIntent, OrderType, Qty, Side, Symbol, TimeInForce, Ts,
};
use algo_features::{engle_granger, fit_ou, Window};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairsConfig {
    pub symbol_a: String,
    pub symbol_b: String,
    pub min_history: usize,
    pub recalibrate_every_bars: u32,
    pub entry_alpha_level: f64, // ADF p-value threshold for cointegration
    pub entry_z: f64,
    pub exit_z: f64,
    pub min_half_life_bars: f64,
    pub max_half_life_bars: f64,
    pub gross_per_leg: f64,
    pub adf_lags: usize,
}

impl Default for PairsConfig {
    fn default() -> Self {
        Self {
            symbol_a: "PAIRA".into(),
            symbol_b: "PAIRB".into(),
            min_history: 200,
            recalibrate_every_bars: 50,
            entry_alpha_level: 0.05,
            entry_z: 2.0,
            exit_z: 0.5,
            min_half_life_bars: 5.0,
            max_half_life_bars: 200.0,
            gross_per_leg: 0.02,
            adf_lags: 1,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum PairPos {
    Flat,
    Long,  // long spread (long Y, short β·X)
    Short, // short spread (short Y, long β·X)
}

pub struct PairsMeanReversion {
    cfg: PairsConfig,
    sym_a: Symbol,
    sym_b: Symbol,
    hist_a: Window<f64>,
    hist_b: Window<f64>,
    bars_since_recal: u32,
    bars_in_trade: u32,
    // Last calibration cache
    last_beta: Option<f64>,
    last_alpha: Option<f64>,
    last_spread_mean: f64,
    last_spread_std: f64,
    last_half_life: f64,
    is_cointegrated: bool,
    pos: PairPos,
}

impl PairsMeanReversion {
    pub fn new(cfg: PairsConfig) -> Self {
        let sym_a = Symbol::new(&cfg.symbol_a).expect("invalid sym_a");
        let sym_b = Symbol::new(&cfg.symbol_b).expect("invalid sym_b");
        let cap = (cfg.min_history + cfg.recalibrate_every_bars as usize + 50).max(256);
        Self {
            cfg,
            sym_a,
            sym_b,
            hist_a: Window::new(cap),
            hist_b: Window::new(cap),
            bars_since_recal: 0,
            bars_in_trade: 0,
            last_beta: None,
            last_alpha: None,
            last_spread_mean: 0.0,
            last_spread_std: 0.0,
            last_half_life: 0.0,
            is_cointegrated: false,
            pos: PairPos::Flat,
        }
    }

    fn ingest(&mut self, bar: &Bar) {
        let px = bar.close.to_f64();
        if bar.symbol == self.sym_a {
            self.hist_a.push(bar.ts, px);
        } else if bar.symbol == self.sym_b {
            self.hist_b.push(bar.ts, px);
        }
    }

    fn recalibrate(&mut self) {
        let needed = self.cfg.min_history;
        let Some(ya) = self.hist_a.last_n(needed) else { return; };
        let Some(yb) = self.hist_b.last_n(needed) else { return; };
        if ya.len() != yb.len() {
            return;
        }
        // Use log prices to make hedge ratio scale-free.
        let log_a: Vec<f64> = ya.iter().map(|p| p.max(1e-12).ln()).collect();
        let log_b: Vec<f64> = yb.iter().map(|p| p.max(1e-12).ln()).collect();
        let arr_a = Array1::from(log_a);
        let arr_b = Array1::from(log_b);
        // Engle-Granger: y = α + β x + ε, so y = log_a, x = log_b
        let Some(eg) = engle_granger(arr_a.view(), arr_b.view(), self.cfg.adf_lags) else {
            self.is_cointegrated = false;
            return;
        };
        self.last_alpha = Some(eg.alpha);
        self.last_beta = Some(eg.beta);
        self.last_spread_mean = eg.residual_mean;
        self.last_spread_std = eg.residual_std;
        self.is_cointegrated = eg.is_cointegrated(self.cfg.entry_alpha_level);
        // Estimate OU half-life on the residuals
        let resid: Vec<f64> = arr_a
            .iter()
            .zip(arr_b.iter())
            .map(|(a, b)| a - eg.alpha - eg.beta * b)
            .collect();
        let arr_r = Array1::from(resid);
        if let Some(ou) = fit_ou(arr_r.view(), 1.0) {
            self.last_half_life = ou.half_life;
        } else {
            self.last_half_life = f64::INFINITY;
        }
    }

    fn current_z(&self, log_a_now: f64, log_b_now: f64) -> Option<f64> {
        let alpha = self.last_alpha?;
        let beta = self.last_beta?;
        if self.last_spread_std <= 0.0 {
            return None;
        }
        let spread = log_a_now - alpha - beta * log_b_now;
        Some((spread - self.last_spread_mean) / self.last_spread_std)
    }
}

impl Strategy for PairsMeanReversion {
    fn name(&self) -> &'static str {
        "pairs_mean_reversion"
    }

    fn validate_config(&self) -> Result<(), String> {
        if self.cfg.min_history < 50 {
            return Err("min_history must be ≥ 50".into());
        }
        if self.cfg.entry_alpha_level <= 0.0 || self.cfg.entry_alpha_level >= 1.0 {
            return Err("entry_alpha_level must be in (0, 1)".into());
        }
        if self.cfg.entry_z <= self.cfg.exit_z {
            return Err("entry_z must exceed exit_z".into());
        }
        if !(self.cfg.gross_per_leg > 0.0 && self.cfg.gross_per_leg < 1.0) {
            return Err("gross_per_leg must be in (0, 1)".into());
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.hist_a = Window::new(self.hist_a.len().max(256));
        self.hist_b = Window::new(self.hist_b.len().max(256));
        self.bars_since_recal = 0;
        self.bars_in_trade = 0;
        self.last_beta = None;
        self.last_alpha = None;
        self.is_cointegrated = false;
        self.pos = PairPos::Flat;
    }

    fn on_event(
        &mut self,
        event: &MarketEvent,
        portfolio: &PortfolioView,
        positions: &[PositionView],
    ) -> SmallVec<[OrderIntent; 4]> {
        let MarketEvent::Bar(bar) = event else { return SmallVec::new(); };
        if bar.symbol != self.sym_a && bar.symbol != self.sym_b {
            return SmallVec::new();
        }
        self.ingest(bar);
        self.bars_since_recal += 1;
        if self.pos != PairPos::Flat {
            self.bars_in_trade += 1;
        }

        // Recalibrate periodically (need both legs to have enough history)
        let both_have_history = self.hist_a.len() >= self.cfg.min_history
            && self.hist_b.len() >= self.cfg.min_history;
        if both_have_history
            && (self.bars_since_recal >= self.cfg.recalibrate_every_bars
                || self.last_beta.is_none())
        {
            self.recalibrate();
            self.bars_since_recal = 0;
        }

        let (Some(px_a), Some(px_b)) = (
            self.hist_a.last().map(|(_, p)| *p),
            self.hist_b.last().map(|(_, p)| *p),
        ) else {
            return SmallVec::new();
        };
        let log_a = px_a.max(1e-12).ln();
        let log_b = px_b.max(1e-12).ln();
        let Some(z) = self.current_z(log_a, log_b) else {
            return SmallVec::new();
        };
        let Some(beta) = self.last_beta else { return SmallVec::new(); };

        let in_band =
            self.last_half_life >= self.cfg.min_half_life_bars
                && self.last_half_life <= self.cfg.max_half_life_bars;
        let pos_a = position_qty(positions, self.sym_a);
        let pos_b = position_qty(positions, self.sym_b);

        let mut intents: SmallVec<[OrderIntent; 4]> = SmallVec::new();

        // ---- Exit logic ----
        let max_hold = (2.0 * self.last_half_life).ceil() as u32;
        let should_exit = match self.pos {
            PairPos::Flat => false,
            PairPos::Long => z.abs() < self.cfg.exit_z || self.bars_in_trade >= max_hold,
            PairPos::Short => z.abs() < self.cfg.exit_z || self.bars_in_trade >= max_hold,
        };
        if should_exit {
            if pos_a.abs() > 0.0 {
                let side = if pos_a > 0.0 { Side::Sell } else { Side::Buy };
                push_intent(&mut intents, self.sym_a, side, pos_a.abs(), portfolio.now);
            }
            if pos_b.abs() > 0.0 {
                let side = if pos_b > 0.0 { Side::Sell } else { Side::Buy };
                push_intent(&mut intents, self.sym_b, side, pos_b.abs(), portfolio.now);
            }
            self.pos = PairPos::Flat;
            self.bars_in_trade = 0;
            return intents;
        }

        // ---- Entry logic ----
        if self.pos != PairPos::Flat || !self.is_cointegrated || !in_band {
            return intents;
        }
        let want = if z > self.cfg.entry_z {
            Some(PairPos::Short) // short spread
        } else if z < -self.cfg.entry_z {
            Some(PairPos::Long)
        } else {
            None
        };
        let Some(want) = want else { return intents; };

        // Sizing — leg notional in dollars
        let leg = portfolio.nav * self.cfg.gross_per_leg;
        let qty_a = (leg / px_a).max(1.0).floor() as i64;
        // Hedge ratio β applied to leg-B notional, sized to maintain dollar neutrality.
        let qty_b = ((leg * beta.abs()) / px_b).max(1.0).floor() as i64;
        let (side_a, side_b) = match want {
            PairPos::Long => (Side::Buy, Side::Sell),  // long Y, short β·X
            PairPos::Short => (Side::Sell, Side::Buy), // short Y, long β·X
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
        strategy: "pairs_mean_reversion".to_string(),
        tag: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{mk_bar, pv};

    #[test]
    fn calibrates_hedge_ratio_and_opens_position_on_spread() {
        // Build two perfectly cointegrated log series:
        //   log_A_t = α + β * log_B_t + ε_t   with α=0, β=1.5, ε ~ small noise.
        // After recalibration, last_beta should be ≈ 1.5.
        let mut s = PairsMeanReversion::new(PairsConfig {
            symbol_a: "PAIRA".into(),
            symbol_b: "PAIRB".into(),
            min_history: 60,
            recalibrate_every_bars: 30,
            entry_alpha_level: 0.20,  // generous
            entry_z: 2.0,
            exit_z: 0.5,
            min_half_life_bars: 1.0,
            max_half_life_bars: 1e6,
            gross_per_leg: 0.02,
            adf_lags: 1,
        });
        let a = Symbol::new("PAIRA").unwrap();
        let b = Symbol::new("PAIRB").unwrap();
        // Generate cointegrated series: log_B random walk, log_A = 1.5*log_B + noise.
        let mut log_b = 4.6_f64; // log(100)
        for i in 0i64..200 {
            // Deterministic micro-shock so β recovery is stable.
            let shock = ((i as f64 * 0.0735).sin()) * 0.02;
            log_b += shock;
            let eps = ((i as f64 * 0.314).cos()) * 0.005;
            let log_a = 1.5 * log_b + eps;
            let px_a = log_a.exp();
            let px_b = log_b.exp();
            let ts = (i + 1) * 60_000_000_000;
            s.on_event(&mk_bar(a, ts, px_a), &pv(ts), &[]);
            s.on_event(&mk_bar(b, ts + 1, px_b), &pv(ts + 1), &[]);
        }
        let beta = s.last_beta.expect("β must be set after calibration");
        assert!(
            (beta - 1.5).abs() < 0.10,
            "β recovery failed: got {} expected ≈1.5",
            beta
        );
    }

    #[test]
    fn no_orders_before_min_history() {
        let mut s = PairsMeanReversion::new(PairsConfig {
            symbol_a: "PAIRA".into(),
            symbol_b: "PAIRB".into(),
            min_history: 100,
            ..PairsConfig::default()
        });
        let a = Symbol::new("PAIRA").unwrap();
        let b = Symbol::new("PAIRB").unwrap();
        let mut total = 0;
        for i in 0i64..30 {
            let ts = (i + 1) * 60_000_000_000;
            total += s.on_event(&mk_bar(a, ts, 100.0), &pv(ts), &[]).len();
            total += s.on_event(&mk_bar(b, ts + 1, 100.0), &pv(ts + 1), &[]).len();
        }
        assert_eq!(total, 0, "no orders allowed before min_history");
    }

    #[test]
    fn validate_config_rejects_garbage() {
        let s = PairsMeanReversion::new(PairsConfig {
            entry_z: 0.5,
            exit_z: 1.0, // exit > entry — invalid
            ..PairsConfig::default()
        });
        assert!(s.validate_config().is_err());
    }
}
