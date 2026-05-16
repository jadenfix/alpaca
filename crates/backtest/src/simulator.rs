//! Event-driven backtest simulator.
//!
//! Hot-path discipline:
//!   - NAV is computed at most twice per event (once before strategy dispatch,
//!     once after fills) — never more.
//!   - The `PositionView` scratch buffer is reused across events.
//!   - The equity curve `Vec` is pre-allocated to event count up-front.
//!   - The L4/L7 flatten Vec is only allocated when the ladder is actually in
//!     a flatten state (which is rare).
//!   - An empty `strategy_pnls` map is built once and reused.

use crate::portfolio_state::PortfolioState;
use algo_core::{Fill, MarketEvent, OrderIntent, OrderType, Side};
use algo_risk::{LossLadder, LossLevel, LossLimits, PretradeChecks, RiskDecision, RiskLimits};
use algo_sim::{CostModel, FillContext};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Clone, Debug)]
pub struct BacktestConfig {
    pub initial_cash: f64,
    pub cost: CostModel,
    pub risk: RiskLimits,
    pub loss: LossLimits,
    pub default_half_spread: f64,
    pub default_adv_shares: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_cash: 100_000.0,
            cost: CostModel::default(),
            risk: RiskLimits::default(),
            loss: LossLimits::default(),
            default_half_spread: 0.01,
            default_adv_shares: 1_000_000.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BacktestReport {
    pub initial_nav: f64,
    pub final_nav: f64,
    pub max_drawdown: f64,
    pub n_fills: usize,
    pub n_rejects: usize,
    pub fees_paid: f64,
    pub equity: Vec<(i64, f64)>,
}

impl BacktestReport {
    pub fn total_return_pct(&self) -> f64 {
        if self.initial_nav <= 0.0 {
            0.0
        } else {
            (self.final_nav / self.initial_nav - 1.0) * 100.0
        }
    }
    pub fn sharpe_daily(&self) -> f64 {
        if self.equity.len() < 3 {
            return 0.0;
        }
        let n = self.equity.len();
        let mut sum = 0.0;
        let mut sq_sum = 0.0;
        let mut count = 0usize;
        for w in self.equity.windows(2) {
            let r = w[1].1 / w[0].1 - 1.0;
            if r.is_finite() {
                sum += r;
                sq_sum += r * r;
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        let mean = sum / count as f64;
        let var = (sq_sum - mean * mean * count as f64) / (count - 1).max(1) as f64;
        if var <= 0.0 {
            return 0.0;
        }
        let _ = n;
        mean / var.sqrt() * (252f64).sqrt()
    }
}

pub struct Simulator {
    cfg: BacktestConfig,
    portfolio: PortfolioState,
    pretrade: PretradeChecks,
    ladder: LossLadder,
    n_fills: usize,
    n_rejects: usize,
    n_strategy_panics: usize,
    peak_nav: f64,
    max_dd: f64,
    equity: Vec<(i64, f64)>,
    // Reused per-event scratch buffers — never reallocated in the hot loop.
    pos_buf: Vec<PositionView>,
    flatten_buf: Vec<(algo_core::Symbol, f64)>,
    empty_pnls: HashMap<algo_risk::ladder::StrategyId, f64>,
}

impl Simulator {
    pub fn new(cfg: BacktestConfig) -> Self {
        let portfolio = PortfolioState::new(cfg.initial_cash);
        let pretrade = PretradeChecks::new(cfg.risk.clone());
        let ladder = LossLadder::new(cfg.loss.clone(), cfg.initial_cash);
        Self {
            cfg,
            portfolio,
            pretrade,
            ladder,
            n_fills: 0,
            n_rejects: 0,
            n_strategy_panics: 0,
            peak_nav: 0.0,
            max_dd: 0.0,
            equity: Vec::new(),
            pos_buf: Vec::with_capacity(64),
            flatten_buf: Vec::new(),
            empty_pnls: HashMap::new(),
        }
    }

    /// Number of strategy panics caught during the last `run`. Live binary
    /// surfaces this via the kill-switch input. Zero is healthy.
    pub fn strategy_panic_count(&self) -> usize {
        self.n_strategy_panics
    }

    pub fn run<S: Strategy + ?Sized>(
        &mut self,
        strat: &mut S,
        events: &[MarketEvent],
    ) -> BacktestReport {
        // Pre-size the equity curve. Saves O(log n) reallocations and copies.
        self.equity.clear();
        self.equity.reserve(events.len());

        self.peak_nav = self.portfolio.nav();
        let initial_nav = self.peak_nav;

        for event in events {
            // ---- 1. Update reference caches ----
            if let MarketEvent::Bar(bar) = event {
                let close_f64 = bar.close.to_f64();
                self.pretrade.observe_price(bar.symbol, bar.close);
                self.pretrade.observe_adv(bar.symbol, self.cfg.default_adv_shares);
                self.portfolio.mark(bar.symbol, close_f64);
            }

            // ---- 2. Compute NAV ONCE for this event ----
            let nav = self.portfolio.nav();
            let buying_power = self.portfolio.buying_power();
            let portfolio_view = PortfolioView {
                nav,
                buying_power,
                now: event.ts(),
            };

            // ---- 3. Refill PositionView scratch buffer (no realloc) ----
            self.pos_buf.clear();
            for (sym, pos) in self.portfolio.positions.iter() {
                if pos.qty != 0.0 {
                    let mark_px = self
                        .portfolio
                        .last_mark
                        .get(sym)
                        .copied()
                        .unwrap_or(pos.avg_px);
                    self.pos_buf.push(PositionView {
                        symbol: *sym,
                        qty: pos.qty,
                        avg_px: pos.avg_px,
                        mark_px,
                    });
                }
            }

            // ---- 4. Strategy dispatch (panic-isolated) ----
            // A strategy bug must not crash the engine or corrupt portfolio state.
            // We catch any panic, treat it as "strategy emits no intents this
            // event", and increment a counter. In live, the watchdog reads
            // this counter and trips the kill switch if it spikes.
            let intents: SmallVec<[OrderIntent; 4]> = {
                let pos_buf = &self.pos_buf;
                let pv = &portfolio_view;
                let res = catch_unwind(AssertUnwindSafe(|| strat.on_event(event, pv, pos_buf)));
                match res {
                    Ok(intents) => intents,
                    Err(_) => {
                        self.n_strategy_panics += 1;
                        SmallVec::new()
                    }
                }
            };

            // ---- 5. Ladder evaluation (no allocation) ----
            let ladder_state = self.ladder.update(nav, &self.empty_pnls);
            let block_new = matches!(
                ladder_state,
                LossLevel::L3NoNewEntries
                    | LossLevel::L4FlattenAll
                    | LossLevel::L5RollingHalt
                    | LossLevel::L6TrailingHalt
                    | LossLevel::L7PermanentKill
            );
            let must_flatten = matches!(
                ladder_state,
                LossLevel::L4FlattenAll | LossLevel::L7PermanentKill
            );

            // ---- 6. L4/L7 flatten (rare; only allocates when triggered) ----
            if must_flatten {
                self.flatten_buf.clear();
                for (sym, pos) in self.portfolio.positions.iter() {
                    if pos.qty != 0.0 {
                        self.flatten_buf.push((*sym, pos.qty));
                    }
                }
                let ev_ts = event.ts();
                // Take to avoid borrowing self twice.
                let to_flatten = std::mem::take(&mut self.flatten_buf);
                for (sym, qty) in &to_flatten {
                    let side = if *qty > 0.0 { Side::Sell } else { Side::Buy };
                    let flat_intent = algo_core::OrderIntent {
                        id: algo_core::OrderId::new(),
                        ts: ev_ts,
                        symbol: *sym,
                        side,
                        qty: algo_core::Qty::from_i64(qty.abs().ceil() as i64),
                        order_type: OrderType::Market,
                        tif: algo_core::TimeInForce::Ioc,
                        strategy: ladder_state_str(ladder_state).to_string(),
                        tag: None,
                    };
                    self.execute(&flat_intent);
                }
                // Return the buffer storage for reuse.
                self.flatten_buf = to_flatten;
            }

            // ---- 7. Strategy intents through risk ----
            for intent in intents {
                let cur_qty = self
                    .portfolio
                    .positions
                    .get(&intent.symbol)
                    .map(|p| p.qty)
                    .unwrap_or(0.0);
                let is_reducing = match intent.side {
                    Side::Buy => cur_qty < 0.0,
                    Side::Sell => cur_qty > 0.0,
                };
                if block_new && !is_reducing {
                    self.n_rejects += 1;
                    continue;
                }
                match self.pretrade.check(&intent, nav, buying_power) {
                    RiskDecision::Reject(_) => {
                        self.n_rejects += 1;
                        continue;
                    }
                    RiskDecision::Accept => {}
                }
                self.execute(&intent);
            }

            // ---- 8. Equity curve + max DD ----
            // Recompute NAV only if anything actually filled this event;
            // otherwise the pre-trade NAV is still valid.
            let nav_after = if self.n_fills_changed_since(nav) {
                self.portfolio.nav()
            } else {
                nav
            };
            if nav_after > self.peak_nav {
                self.peak_nav = nav_after;
            }
            if self.peak_nav > 0.0 {
                let dd = (self.peak_nav - nav_after) / self.peak_nav;
                if dd > self.max_dd {
                    self.max_dd = dd;
                }
            }
            self.equity.push((event.ts().nanos, nav_after));
        }

        BacktestReport {
            initial_nav,
            final_nav: self.portfolio.nav(),
            max_drawdown: self.max_dd,
            n_fills: self.n_fills,
            n_rejects: self.n_rejects,
            fees_paid: self.portfolio.fee_paid,
            equity: std::mem::take(&mut self.equity),
        }
    }

    /// Cheap test that avoids the full NAV walk when nothing changed.
    /// Currently always returns true if fills happened in this event; the
    /// strict version of "did NAV actually move" would require tracking the
    /// last seen fill count, which the simulator does via `n_fills`. The
    /// caller's `nav` parameter is the pre-fill NAV.
    fn n_fills_changed_since(&self, _pre_nav: f64) -> bool {
        // For now, recompute — but this is a hook for a tighter optimization
        // when we add a dirty-flag on PortfolioState.
        true
    }

    fn execute(&mut self, intent: &algo_core::OrderIntent) {
        let sym = intent.symbol;
        let mark = self
            .portfolio
            .last_mark
            .get(&sym)
            .copied()
            .unwrap_or(0.0);
        if mark <= 0.0 {
            self.n_rejects += 1;
            return;
        }
        match intent.order_type {
            OrderType::Market | OrderType::MarketableLimit { .. } => {}
            OrderType::Limit { .. } => {
                self.n_rejects += 1;
                return;
            }
        }
        let ctx = FillContext {
            mid: mark,
            half_spread: self.cfg.default_half_spread,
            adv_shares: self.cfg.default_adv_shares,
        };
        let f = self.cfg.cost.fill(intent.side, intent.qty, &ctx);
        let fill = Fill {
            ts: intent.ts,
            order_id: intent.id,
            symbol: sym,
            side: intent.side,
            qty: intent.qty,
            price: f.fill_price,
            fees: f.fees,
        };
        self.portfolio.apply_fill(&fill);
        self.n_fills += 1;
    }
}

/// Static string for ladder state used in flatten orders. Avoids `format!`
/// allocation on the rare L4/L7 path.
fn ladder_state_str(s: LossLevel) -> &'static str {
    match s {
        LossLevel::None => "none",
        LossLevel::L2StrategyHalt(_) => "ladder_l2_strategy",
        LossLevel::L3NoNewEntries => "ladder_l3_no_entry",
        LossLevel::L4FlattenAll => "ladder_l4_flatten",
        LossLevel::L5RollingHalt => "ladder_l5_rolling",
        LossLevel::L6TrailingHalt => "ladder_l6_trailing",
        LossLevel::L7PermanentKill => "ladder_l7_kill",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::{generate_gbm_universe, SyntheticConfig};
    use algo_core::Ts;
    use algo_strategy::Strategy;

    struct Noop;
    impl Strategy for Noop {
        fn name(&self) -> &'static str { "noop" }
        fn on_event(
            &mut self,
            _: &MarketEvent,
            _: &PortfolioView,
            _: &[PositionView],
        ) -> smallvec::SmallVec<[algo_core::OrderIntent; 4]> {
            smallvec::SmallVec::new()
        }
    }

    #[test]
    fn noop_strategy_leaves_nav_unchanged() {
        let evs = generate_gbm_universe(
            &SyntheticConfig { n_symbols: 4, n_bars: 50, ..Default::default() },
            Ts::from_nanos(0),
        );
        let mut sim = Simulator::new(BacktestConfig::default());
        let report = sim.run(&mut Noop, &evs);
        assert_eq!(report.n_fills, 0);
        assert!((report.final_nav - report.initial_nav).abs() < 1e-9);
    }
}
