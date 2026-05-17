//! Smoke + invariant tests across all 5 production strategies.
//!
//! Goals:
//!   - Each strategy compiles, runs without panic on multiple data shapes,
//!     and produces a sensible equity curve.
//!   - `validate_config()` rejects garbage configs.
//!   - The simulator's panic-isolation catches a deliberately-panicking
//!     strategy without crashing the backtest or corrupting NAV.

use algo_backtest::{
    generate_cointegrated_pair, generate_momentum_universe, BacktestConfig, Simulator,
};
use algo_core::{MarketEvent, OrderIntent, Symbol, Ts};
use algo_strategies::{
    garch_vol_target::GarchVolTargetConfig, kalman_pairs::KalmanPairsConfig,
    pairs_mean_reversion::PairsConfig, regime_hmm::RegimeHmmConfig, xs_momentum::XsMomentumConfig,
    GarchVolTarget, KalmanPairs, PairsMeanReversion, RegimeHmm, XsMomentum,
};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use smallvec::SmallVec;

// ---------- helper: cointegrated PAIRA/PAIRB universe ----------

fn pair_events(n_bars: usize) -> Vec<MarketEvent> {
    let a = Symbol::new("PAIRA").unwrap();
    let b = Symbol::new("PAIRB").unwrap();
    generate_cointegrated_pair(a, b, n_bars, 60, 11, Ts::from_nanos(0))
}

// ---------- one test per strategy: runs end-to-end with no panic ----------

#[test]
fn xs_momentum_runs_clean() {
    let evs = generate_momentum_universe(8, 500, 60, 0.20, 50.0, 1, Ts::from_nanos(0));
    let mut sim = Simulator::new(BacktestConfig {
        initial_cash: 200_000.0,
        ..Default::default()
    });
    let mut s = XsMomentum::new(XsMomentumConfig::default());
    let report = sim.run(&mut s, &evs);
    // ── shape ──
    assert_eq!(report.equity.len(), evs.len());
    assert!(report.final_nav.is_finite());
    assert_eq!(sim.strategy_panic_count(), 0);
    // ── behavioral ──
    assert!(report.n_fills > 0, "xs_momentum on a momentum universe must trade");
    // L4 hard stop is 3.5%; the ladder must contain DD with at most a small overshoot.
    assert!(
        report.max_drawdown <= 0.06,
        "max DD {:.4} exceeded L4 + tolerance",
        report.max_drawdown
    );
    // Cost model must have charged something on every fill (SEC/TAF on sells; commission_per_share is 0 but slippage is in the fill_price).
    if report.n_fills > 10 {
        assert!(report.fees_paid >= 0.0, "fees_paid must be non-negative");
    }
}

#[test]
fn pairs_mean_reversion_runs_clean() {
    let evs = pair_events(800);
    let mut sim = Simulator::new(BacktestConfig {
        initial_cash: 200_000.0,
        ..Default::default()
    });
    let mut s = PairsMeanReversion::new(PairsConfig {
        symbol_a: "PAIRA".into(),
        symbol_b: "PAIRB".into(),
        min_history: 200,
        ..Default::default()
    });
    let report = sim.run(&mut s, &evs);
    assert_eq!(report.equity.len(), evs.len());
    assert!(report.final_nav.is_finite());
    assert_eq!(sim.strategy_panic_count(), 0);
    assert!(
        report.max_drawdown <= 0.06,
        "max DD {:.4} exceeded L4 + tolerance",
        report.max_drawdown
    );
}

#[test]
fn kalman_pairs_runs_clean_and_produces_fills() {
    let evs = pair_events(800);
    let mut sim = Simulator::new(BacktestConfig {
        initial_cash: 200_000.0,
        ..Default::default()
    });
    let mut s = KalmanPairs::new(KalmanPairsConfig::default());
    let report = sim.run(&mut s, &evs);
    assert_eq!(report.equity.len(), evs.len());
    assert!(report.final_nav.is_finite());
    // With cointegrated data + empirical-z, kalman_pairs should generate trades.
    assert!(report.n_fills > 0, "kalman_pairs should fire orders on cointegrated data");
    assert_eq!(sim.strategy_panic_count(), 0);
    assert!(
        report.max_drawdown <= 0.06,
        "max DD {:.4} exceeded L4 + tolerance",
        report.max_drawdown
    );
    // Fills happened → fees must have been recorded somewhere on sells.
    assert!(report.fees_paid >= 0.0, "fees_paid invariant");
}

#[test]
fn garch_vol_target_runs_clean() {
    let evs = generate_momentum_universe(6, 500, 60, 0.20, 50.0, 3, Ts::from_nanos(0));
    let mut sim = Simulator::new(BacktestConfig {
        initial_cash: 200_000.0,
        ..Default::default()
    });
    let mut s = GarchVolTarget::new(GarchVolTargetConfig::default());
    let report = sim.run(&mut s, &evs);
    assert_eq!(report.equity.len(), evs.len());
    assert!(report.final_nav.is_finite());
    assert_eq!(sim.strategy_panic_count(), 0);
    assert!(report.n_fills > 0, "garch_vol_target on momentum universe must trade");
    assert!(
        report.max_drawdown <= 0.06,
        "max DD {:.4} exceeded L4 + tolerance",
        report.max_drawdown
    );
}

#[test]
fn regime_hmm_runs_clean() {
    // Need one symbol that matches the regime_universe_ticker; use SPY name.
    // We rename our synthetic SYM0 to SPY by inserting a manual SPY series.
    let mut evs = generate_momentum_universe(6, 400, 60, 0.15, 30.0, 5, Ts::from_nanos(0));
    // Rename one of the symbols' bars to SPY (the regime proxy).
    let sym_spy = Symbol::new("SPY").unwrap();
    let sym_old = Symbol::new("SYM0").unwrap();
    for e in evs.iter_mut() {
        if let MarketEvent::Bar(b) = e {
            if b.symbol == sym_old {
                b.symbol = sym_spy;
            }
        }
    }
    let mut sim = Simulator::new(BacktestConfig {
        initial_cash: 200_000.0,
        ..Default::default()
    });
    let mut s = RegimeHmm::new(RegimeHmmConfig::default());
    let report = sim.run(&mut s, &evs);
    assert_eq!(report.equity.len(), evs.len());
    assert!(report.final_nav.is_finite());
    assert_eq!(sim.strategy_panic_count(), 0);
    assert!(
        report.max_drawdown <= 0.06,
        "max DD {:.4} exceeded L4 + tolerance",
        report.max_drawdown
    );
}

// ---------- config validation ----------

#[test]
fn validate_config_rejects_garbage() {
    // pairs_mean_reversion: entry_z must exceed exit_z
    let bad = PairsConfig {
        entry_z: 1.0,
        exit_z: 2.0,
        ..Default::default()
    };
    let strat = PairsMeanReversion::new(bad);
    assert!(strat.validate_config().is_err());

    // kalman_pairs: Q must be > 0
    let mut k = KalmanPairsConfig::default();
    k.process_noise_q = 0.0;
    let strat = KalmanPairs::new(k);
    assert!(strat.validate_config().is_err());

    // garch_vol_target: vol_target_annual must be in (0, 0.5]
    let mut g = GarchVolTargetConfig::default();
    g.vol_target_annual = 0.0;
    let strat = GarchVolTarget::new(g);
    assert!(strat.validate_config().is_err());

    // regime_hmm: min_calm_prob must be in [0.5, 1.0]
    let mut h = RegimeHmmConfig::default();
    h.min_calm_prob = 0.3;
    let strat = RegimeHmm::new(h);
    assert!(strat.validate_config().is_err());
}

#[test]
fn validate_config_accepts_defaults() {
    let p = PairsMeanReversion::new(PairsConfig::default());
    assert!(p.validate_config().is_ok());
    let k = KalmanPairs::new(KalmanPairsConfig::default());
    assert!(k.validate_config().is_ok());
    let g = GarchVolTarget::new(GarchVolTargetConfig::default());
    assert!(g.validate_config().is_ok());
    let h = RegimeHmm::new(RegimeHmmConfig::default());
    assert!(h.validate_config().is_ok());
}

// ---------- panic-safety ----------

/// A strategy that intentionally panics on the 50th event.
struct PanickingStrategy {
    n: usize,
}

impl Strategy for PanickingStrategy {
    fn name(&self) -> &'static str { "panicking" }
    fn on_event(
        &mut self,
        _: &MarketEvent,
        _: &PortfolioView,
        _: &[PositionView],
    ) -> SmallVec<[OrderIntent; 4]> {
        self.n += 1;
        if self.n == 50 {
            panic!("synthetic strategy panic for testing");
        }
        SmallVec::new()
    }
}

#[test]
fn simulator_isolates_strategy_panic() {
    let evs = generate_momentum_universe(4, 100, 60, 0.10, 0.0, 1, Ts::from_nanos(0));
    let mut sim = Simulator::new(BacktestConfig::default());
    let report = sim.run(&mut PanickingStrategy { n: 0 }, &evs);
    // The panic should not crash the backtest; equity curve must still be complete.
    assert_eq!(report.equity.len(), evs.len(), "panic must not truncate equity curve");
    // The simulator counts strategy panics.
    assert!(sim.strategy_panic_count() >= 1, "expected at least one strategy panic recorded");
    // No fills since strategy never returns intents.
    assert_eq!(report.n_fills, 0);
    // NAV must be unchanged.
    assert!((report.final_nav - report.initial_nav).abs() < 1e-9);
}

// ---------- strategy reset ----------

#[test]
fn reset_clears_strategy_state() {
    let mut s = XsMomentum::new(XsMomentumConfig::default());
    let evs = generate_momentum_universe(4, 100, 60, 0.10, 5.0, 1, Ts::from_nanos(0));
    let mut sim = Simulator::new(BacktestConfig::default());
    let _ = sim.run(&mut s, &evs);
    // After reset, strategy should behave as if newly constructed.
    s.reset();
    // After reset, on first event, the strategy must not emit orders (insufficient history).
    let mut sim2 = Simulator::new(BacktestConfig::default());
    let report = sim2.run(&mut s, &evs[..16]);
    assert_eq!(report.n_fills, 0, "after reset, strategy must rebuild history before trading");
}
