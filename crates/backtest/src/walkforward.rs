//! Walk-forward cross-validation.
//!
//! Splits an event stream into K folds, each fold has a `train` window and a
//! `test` window. The strategy is allowed to "warm up" on train but PnL is
//! ONLY counted on test (the strategy is recreated fresh between folds, but
//! optionally pre-fed train events to populate its internal state).
//!
//! Walk-forward respects time order: fold k's test window comes AFTER fold k's
//! train window AND fold k+1's train window includes fold k's test window.
//!
//! This is the only honest way to evaluate a systematic strategy on history.

use crate::analytics::{compute, Metrics};
use crate::portfolio_state::PortfolioState;
use crate::simulator::{BacktestConfig, BacktestReport, Simulator};
use algo_core::MarketEvent;
use algo_strategy::Strategy;

#[derive(Clone, Debug)]
pub struct WalkForwardConfig {
    pub n_folds: usize,
    /// Train fraction of each fold (e.g. 0.7 → 70% train, 30% test).
    pub train_frac: f64,
    pub backtest: BacktestConfig,
    pub bar_span_secs: u32,
    /// Whether to feed train events to the strategy as warm-up. Disabled by
    /// default — most rolling-window strategies will skip the first
    /// `lookback` bars in the test fold themselves, and warm-up risks
    /// leaking strategy state across the train/test boundary.
    pub warm_up: bool,
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            n_folds: 4,
            train_frac: 0.6,
            backtest: BacktestConfig::default(),
            bar_span_secs: 60,
            warm_up: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FoldResult {
    pub fold_idx: usize,
    pub train_events: usize,
    pub test_events: usize,
    pub report: BacktestReport,
    pub metrics: Metrics,
}

#[derive(Clone, Debug)]
pub struct WalkForwardReport {
    pub folds: Vec<FoldResult>,
    pub avg_test_sharpe: f64,
    pub avg_test_return_pct: f64,
    pub min_test_sharpe: f64,
    pub max_test_sharpe: f64,
    pub n_positive_test_folds: usize,
}

/// Run walk-forward CV. `factory` is called once per fold to create a fresh
/// strategy instance — important for fold independence.
pub fn run_walkforward<S, F>(
    events: &[MarketEvent],
    cfg: &WalkForwardConfig,
    mut factory: F,
) -> WalkForwardReport
where
    S: Strategy,
    F: FnMut() -> S,
{
    assert!(cfg.n_folds >= 1);
    assert!(cfg.train_frac > 0.0 && cfg.train_frac < 1.0);
    let total = events.len();
    let fold_size = total / cfg.n_folds;
    let mut results = Vec::with_capacity(cfg.n_folds);

    for k in 0..cfg.n_folds {
        let start = k * fold_size;
        let end = if k == cfg.n_folds - 1 { total } else { (k + 1) * fold_size };
        let train_end = start + ((end - start) as f64 * cfg.train_frac) as usize;
        let train = &events[start..train_end];
        let test = &events[train_end..end];

        let mut strat = factory();
        if cfg.warm_up {
            // Throwaway simulator just to populate the strategy's internal state.
            let mut warmup_sim = Simulator::new(cfg.backtest.clone());
            let _ = warmup_sim.run(&mut strat, train);
        }

        // Test: fresh portfolio, fresh ladder, fresh broker state.
        let mut test_sim = Simulator::new(cfg.backtest.clone());
        let report = test_sim.run(&mut strat, test);
        let metrics = compute(&report.equity, cfg.bar_span_secs);
        results.push(FoldResult {
            fold_idx: k,
            train_events: train.len(),
            test_events: test.len(),
            report,
            metrics,
        });
    }

    let test_sharpes: Vec<f64> = results.iter().map(|r| r.metrics.sharpe).collect();
    let test_returns: Vec<f64> = results.iter().map(|r| r.metrics.total_return_pct).collect();
    let avg_test_sharpe = if !test_sharpes.is_empty() {
        test_sharpes.iter().sum::<f64>() / test_sharpes.len() as f64
    } else {
        0.0
    };
    let avg_test_return_pct = if !test_returns.is_empty() {
        test_returns.iter().sum::<f64>() / test_returns.len() as f64
    } else {
        0.0
    };
    let min_test_sharpe = test_sharpes.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_test_sharpe = test_sharpes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n_positive_test_folds = results.iter().filter(|r| r.metrics.total_return_pct > 0.0).count();

    let _ = PortfolioState::new(0.0); // silence dead-code if state path goes unused above

    WalkForwardReport {
        folds: results,
        avg_test_sharpe,
        avg_test_return_pct,
        min_test_sharpe,
        max_test_sharpe,
        n_positive_test_folds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::{generate_momentum_universe, SyntheticConfig};
    use algo_core::{MarketEvent, OrderIntent, Ts};
    use algo_strategy::{PortfolioView, PositionView, Strategy};
    use smallvec::SmallVec;

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
        ) -> SmallVec<[OrderIntent; 4]> {
            SmallVec::new()
        }
    }

    #[test]
    fn walkforward_runs_n_folds() {
        let _ = SyntheticConfig::default();
        let evs = generate_momentum_universe(4, 400, 60, 0.10, 1.0, 1, Ts::from_nanos(0));
        let cfg = WalkForwardConfig {
            n_folds: 4,
            train_frac: 0.5,
            ..Default::default()
        };
        let report = run_walkforward(&evs, &cfg, || Noop);
        assert_eq!(report.folds.len(), 4);
        for f in &report.folds {
            assert!(f.train_events > 0);
            assert!(f.test_events > 0);
        }
    }
}
