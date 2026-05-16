//! Lead-lag pairs strategy.
//!
//! Detects when a "leader" symbol (e.g., SPY, QQQ) leads a "follower" by a
//! short lag, then trades the follower in the direction of the leader's
//! recent return. The lead-lag relationship is estimated from a rolling
//! returns window; trades only fire when the absolute cross-correlation
//! exceeds a minimum threshold AND the lag is positive (leader truly leads).
//!
//! Reference: de Jong & Nijman (1997); Hayashi & Yoshida (2005) for the
//! asynchronous tick-data refinement (not needed here for bar data).

use algo_core::{
    MarketEvent, OrderId, OrderIntent, OrderType, Qty, Side, Symbol, TimeInForce,
};
use algo_features::{lead_lag, Window};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeadLagPairsConfig {
    pub leader_ticker: String,
    pub follower_ticker: String,
    pub window_bars: usize,
    pub max_lag_bars: usize,
    pub min_abs_corr: f64,
    pub gross_per_leg: f64,
    pub max_hold_bars: u32,
    pub recalibrate_every_bars: u32,
}

impl Default for LeadLagPairsConfig {
    fn default() -> Self {
        Self {
            leader_ticker: "SPY".into(),
            follower_ticker: "QQQ".into(),
            window_bars: 300,
            max_lag_bars: 5,
            min_abs_corr: 0.3,
            gross_per_leg: 0.02,
            max_hold_bars: 10,
            recalibrate_every_bars: 50,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Pos {
    Flat,
    Long,
    Short,
}

pub struct LeadLagPairs {
    cfg: LeadLagPairsConfig,
    leader: Symbol,
    follower: Symbol,
    leader_rets: Window<f64>,
    follower_rets: Window<f64>,
    last_leader_px: Option<f64>,
    last_follower_px: Option<f64>,
    bars_since_recal: u32,
    bars_in_trade: u32,
    /// Cached best lag and correlation from last recalibration.
    best_lag: i32,
    best_corr: f64,
    pos: Pos,
}

impl LeadLagPairs {
    pub fn new(cfg: LeadLagPairsConfig) -> Self {
        let leader = Symbol::new(&cfg.leader_ticker).expect("invalid leader");
        let follower = Symbol::new(&cfg.follower_ticker).expect("invalid follower");
        let cap = cfg.window_bars + cfg.max_lag_bars + 32;
        Self {
            cfg,
            leader,
            follower,
            leader_rets: Window::new(cap),
            follower_rets: Window::new(cap),
            last_leader_px: None,
            last_follower_px: None,
            bars_since_recal: 0,
            bars_in_trade: 0,
            best_lag: 0,
            best_corr: 0.0,
            pos: Pos::Flat,
        }
    }

    fn recalibrate(&mut self) {
        let want = self.cfg.window_bars;
        let Some(rx) = self.leader_rets.last_n(want) else { return; };
        let Some(ry) = self.follower_rets.last_n(want) else { return; };
        let Some(res) = lead_lag(&rx, &ry, self.cfg.max_lag_bars) else { return; };
        self.best_lag = res.best_lag;
        self.best_corr = res.best_corr;
    }
}

impl Strategy for LeadLagPairs {
    fn name(&self) -> &'static str {
        "leadlag_pairs"
    }

    fn validate_config(&self) -> Result<(), String> {
        if self.cfg.window_bars < 60 {
            return Err("window_bars must be ≥ 60".into());
        }
        if self.cfg.leader_ticker == self.cfg.follower_ticker {
            return Err("leader and follower must differ".into());
        }
        if !(0.0 < self.cfg.min_abs_corr && self.cfg.min_abs_corr <= 1.0) {
            return Err("min_abs_corr must be in (0, 1]".into());
        }
        Ok(())
    }

    fn reset(&mut self) {
        let cap = self.cfg.window_bars + self.cfg.max_lag_bars + 32;
        self.leader_rets = Window::new(cap);
        self.follower_rets = Window::new(cap);
        self.last_leader_px = None;
        self.last_follower_px = None;
        self.bars_since_recal = 0;
        self.bars_in_trade = 0;
        self.best_lag = 0;
        self.best_corr = 0.0;
        self.pos = Pos::Flat;
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
        // Update returns history for whichever symbol this bar is for.
        if bar.symbol == self.leader {
            if let Some(prev) = self.last_leader_px {
                if prev > 0.0 {
                    let r = (px / prev).ln();
                    if r.is_finite() {
                        self.leader_rets.push(bar.ts, r);
                    }
                }
            }
            self.last_leader_px = Some(px);
        } else if bar.symbol == self.follower {
            if let Some(prev) = self.last_follower_px {
                if prev > 0.0 {
                    let r = (px / prev).ln();
                    if r.is_finite() {
                        self.follower_rets.push(bar.ts, r);
                    }
                }
            }
            self.last_follower_px = Some(px);
        } else {
            return SmallVec::new();
        }
        self.bars_since_recal += 1;
        if self.pos != Pos::Flat {
            self.bars_in_trade += 1;
        }
        let have_history = self.leader_rets.len() >= self.cfg.window_bars
            && self.follower_rets.len() >= self.cfg.window_bars;
        if have_history && self.bars_since_recal >= self.cfg.recalibrate_every_bars {
            self.recalibrate();
            self.bars_since_recal = 0;
        }

        let mut intents: SmallVec<[OrderIntent; 4]> = SmallVec::new();
        let Some(px_now) = self.last_follower_px else { return intents; };
        let pos_qty = position_qty(positions, self.follower);

        // Exit on time
        if self.pos != Pos::Flat && self.bars_in_trade >= self.cfg.max_hold_bars {
            if pos_qty.abs() > 0.0 {
                let side = if pos_qty > 0.0 { Side::Sell } else { Side::Buy };
                intents.push(OrderIntent {
                    id: OrderId::new(),
                    ts: portfolio.now,
                    symbol: self.follower,
                    side,
                    qty: Qty::from_i64(pos_qty.abs().ceil() as i64),
                    order_type: OrderType::Market,
                    tif: TimeInForce::Day,
                    strategy: "leadlag_pairs".to_string(),
                    tag: Some("time_exit".to_string()),
                });
            }
            self.pos = Pos::Flat;
            self.bars_in_trade = 0;
            return intents;
        }

        // Only enter when our correlation estimate is strong AND the leader truly leads.
        if self.pos != Pos::Flat
            || self.best_corr.abs() < self.cfg.min_abs_corr
            || self.best_lag <= 0
        {
            return intents;
        }

        // Get the most recent leader returns at the best_lag horizon.
        let need = self.best_lag as usize;
        let Some(leader_recent) = self.leader_rets.last_n(need) else { return intents; };
        if leader_recent.is_empty() {
            return intents;
        }
        // Sum of leader's last `lag` returns is our forecast signal.
        let signal: f64 = leader_recent.iter().sum();
        // Direction: if corr is positive, follow the leader's sign; if negative, oppose.
        let dir = if self.best_corr >= 0.0 {
            if signal > 0.0 { 1 } else if signal < 0.0 { -1 } else { 0 }
        } else if signal > 0.0 { -1 } else if signal < 0.0 { 1 } else { 0 };

        if dir == 0 {
            return intents;
        }
        let leg_notional = portfolio.nav * self.cfg.gross_per_leg;
        let target = ((leg_notional / px_now).floor() as i64) * dir as i64;
        let delta = target as f64 - pos_qty;
        if delta.abs() < 1.0 {
            return intents;
        }
        let side = if delta > 0.0 { Side::Buy } else { Side::Sell };
        intents.push(OrderIntent {
            id: OrderId::new(),
            ts: portfolio.now,
            symbol: self.follower,
            side,
            qty: Qty::from_i64(delta.abs().ceil() as i64),
            order_type: OrderType::Market,
            tif: TimeInForce::Day,
            strategy: "leadlag_pairs".to_string(),
            tag: Some(format!("lag={} corr={:.3}", self.best_lag, self.best_corr)),
        });
        self.pos = if dir > 0 { Pos::Long } else { Pos::Short };
        self.bars_in_trade = 0;
        intents
    }
}

fn position_qty(positions: &[PositionView], sym: Symbol) -> f64 {
    positions.iter().find(|p| p.symbol == sym).map(|p| p.qty).unwrap_or(0.0)
}
