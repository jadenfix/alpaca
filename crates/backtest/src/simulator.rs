//! Event-driven backtest simulator.
//!
//! Loops over a chronological `Vec<MarketEvent>`, dispatches each event to
//! every strategy, simulates fills via the shared `CostModel`, applies the
//! same `PretradeChecks` and `LossLadder` used live, and accumulates PnL.

use crate::portfolio_state::PortfolioState;
use algo_core::{Fill, MarketEvent, OrderType, Side};
use algo_risk::{LossLadder, LossLevel, LossLimits, PretradeChecks, RiskDecision, RiskLimits};
use algo_sim::{CostModel, FillContext};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct BacktestConfig {
    pub initial_cash: f64,
    pub cost: CostModel,
    pub risk: RiskLimits,
    pub loss: LossLimits,
    /// Assumed half-spread in dollars when no quote is present (e.g. for synthetic).
    pub default_half_spread: f64,
    /// Assumed ADV in shares when not provided.
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
    /// Equity curve: (ts_nanos, nav).
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
        // Crude: from the equity curve, compute daily-equivalent returns.
        if self.equity.len() < 3 {
            return 0.0;
        }
        let mut rets = Vec::with_capacity(self.equity.len() - 1);
        for w in self.equity.windows(2) {
            let r = w[1].1 / w[0].1 - 1.0;
            if r.is_finite() {
                rets.push(r);
            }
        }
        if rets.is_empty() {
            return 0.0;
        }
        let mean = rets.iter().sum::<f64>() / rets.len() as f64;
        let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (rets.len() - 1).max(1) as f64;
        if var <= 0.0 {
            return 0.0;
        }
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
    peak_nav: f64,
    max_dd: f64,
    equity: Vec<(i64, f64)>,
    next_intent_idx: u64,
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
            peak_nav: 0.0,
            max_dd: 0.0,
            equity: Vec::new(),
            next_intent_idx: 0,
        }
    }

    pub fn run<S: Strategy + ?Sized>(
        &mut self,
        strat: &mut S,
        events: &[MarketEvent],
    ) -> BacktestReport {
        self.peak_nav = self.portfolio.nav();
        let initial_nav = self.peak_nav;

        for event in events {
            // Update reference prices and marks.
            if let MarketEvent::Bar(bar) = event {
                self.pretrade.observe_price(bar.symbol, bar.close);
                self.pretrade.observe_adv(bar.symbol, self.cfg.default_adv_shares);
                self.portfolio.mark(bar.symbol, bar.close.to_f64());
            }

            // Dispatch to strategy. Build PortfolioView + PositionView.
            let nav = self.portfolio.nav();
            let buying_power = self.portfolio.buying_power();
            let portfolio_view = PortfolioView {
                nav,
                buying_power,
                now: event.ts(),
            };
            let pos_views: Vec<PositionView> = self
                .portfolio
                .positions
                .iter()
                .filter(|(_, p)| p.qty != 0.0)
                .map(|(sym, p)| PositionView {
                    symbol: *sym,
                    qty: p.qty,
                    avg_px: p.avg_px,
                    mark_px: self
                        .portfolio
                        .last_mark
                        .get(sym)
                        .copied()
                        .unwrap_or(p.avg_px),
                })
                .collect();

            let intents = strat.on_event(event, &portfolio_view, &pos_views);

            // Loss ladder evaluation (after the event update, before order send).
            let strategy_pnls: HashMap<algo_risk::ladder::StrategyId, f64> = HashMap::new();
            let ladder_state = self.ladder.update(nav, &strategy_pnls);
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

            // L4 / L7: issue close orders for every open position immediately.
            // These orders bypass strategy intents and go directly to the cost model.
            if must_flatten {
                let to_flatten: Vec<(algo_core::Symbol, f64)> = self
                    .portfolio
                    .positions
                    .iter()
                    .filter(|(_, p)| p.qty != 0.0)
                    .map(|(sym, p)| (*sym, p.qty))
                    .collect();
                for (sym, qty) in to_flatten {
                    let side = if qty > 0.0 { Side::Sell } else { Side::Buy };
                    let flat_intent = algo_core::OrderIntent {
                        id: algo_core::OrderId::new(),
                        ts: event.ts(),
                        symbol: sym,
                        side,
                        qty: algo_core::Qty::from_i64(qty.abs().ceil() as i64),
                        order_type: OrderType::Market,
                        tif: algo_core::TimeInForce::Ioc,
                        strategy: "ladder_flatten".to_string(),
                        tag: Some(format!("{ladder_state:?}")),
                    };
                    // Skip pre-trade for flatten: this is a safety-mandated unwind.
                    self.execute(&flat_intent);
                }
            }

            for intent in intents {
                // Determine if this is an exit vs entry.
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

            // Track equity curve + max DD.
            let nav_after = self.portfolio.nav();
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
        // Honor backtestable order types: market & marketable-limit only.
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
        self.next_intent_idx += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::{generate_gbm_universe, SyntheticConfig};
    use algo_core::Ts;
    use algo_strategy::Strategy;

    /// A no-op strategy for testing the loop infrastructure.
    struct Noop;
    impl Strategy for Noop {
        fn name(&self) -> &'static str {
            "noop"
        }
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
            &SyntheticConfig {
                n_symbols: 4,
                n_bars: 50,
                ..Default::default()
            },
            Ts::from_nanos(0),
        );
        let mut sim = Simulator::new(BacktestConfig::default());
        let report = sim.run(&mut Noop, &evs);
        assert_eq!(report.n_fills, 0);
        assert!((report.final_nav - report.initial_nav).abs() < 1e-9);
    }
}
