//! Property-based tests for engine invariants.
//!
//! These are randomized tests that check engine-level properties hold across
//! a wide range of inputs. Run with: `cargo test -p algo-backtest --test properties`.

use algo_backtest::{generate_gbm_universe, BacktestConfig, Simulator, SyntheticConfig};
use algo_core::{MarketEvent, OrderIntent, Ts};
use algo_features::{Welford, Window};
use algo_risk::{LossLadder, LossLevel, LossLimits};
use algo_strategies::{xs_momentum::XsMomentumConfig, XsMomentum};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use proptest::prelude::*;
use smallvec::SmallVec;
use std::collections::HashMap;

// ---------- Welford properties ----------

proptest! {
    /// Welford's online algorithm must agree with the textbook batch
    /// formula within numerical precision for any input sequence.
    #[test]
    fn welford_matches_batch(xs in prop::collection::vec(-1000.0_f64..1000.0, 2..1000)) {
        let mut w = Welford::new();
        for x in &xs {
            w.push(*x);
        }
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
        prop_assert!((w.mean() - mean).abs() < 1e-6 * (1.0 + mean.abs()));
        prop_assert!((w.variance() - var).abs() < 1e-6 * (1.0 + var.abs()));
    }

    /// Welford variance must be non-negative for any input.
    #[test]
    fn welford_variance_nonneg(xs in prop::collection::vec(-1e9_f64..1e9, 2..500)) {
        let mut w = Welford::new();
        for x in &xs {
            w.push(*x);
        }
        prop_assert!(w.variance() >= 0.0);
    }
}

// ---------- Window properties ----------

proptest! {
    /// last_n_slice and last_n must agree on the same window state.
    #[test]
    fn window_slice_matches_clone(values in prop::collection::vec(any::<i32>(), 1..100), n in 1usize..50) {
        let mut w: Window<i32> = Window::new(values.len().max(n));
        for (i, &v) in values.iter().enumerate() {
            w.push(Ts::from_nanos(i as i64), v);
        }
        if values.len() >= n {
            let owned = w.last_n(n).unwrap();
            let (front, back) = w.last_n_slice(n).unwrap();
            let zero_copy: Vec<i32> = front.iter().chain(back).map(|(_, v)| *v).collect();
            prop_assert_eq!(owned, zero_copy);
        }
    }

    /// Window must never return data beyond its capacity.
    #[test]
    fn window_respects_capacity(cap in 1usize..50, n_push in 0usize..200) {
        let mut w: Window<u64> = Window::new(cap);
        for i in 0..n_push {
            w.push(Ts::from_nanos(i as i64), i as u64);
        }
        prop_assert!(w.len() <= cap);
    }
}

// ---------- Loss ladder properties ----------

proptest! {
    /// The ladder is monotone: once L7 is reached, it stays L7 across any
    /// number of subsequent updates.
    #[test]
    fn ladder_l7_is_absorbing(navs in prop::collection::vec(10_000.0_f64..200_000.0, 1..50)) {
        let mut ladder = LossLadder::new(LossLimits::default(), 100_000.0);
        ladder.on_session_open(100_000.0);
        let empty = HashMap::new();
        // Trip L7 with -15% drawdown
        let _ = ladder.update(85_000.0, &empty);
        prop_assert!(matches!(ladder.state(), LossLevel::L7PermanentKill));
        for v in navs {
            let _ = ladder.update(v, &empty);
            prop_assert!(matches!(ladder.state(), LossLevel::L7PermanentKill));
        }
    }

    /// L3 should fire iff DD from day-open ≥ portfolio_intraday limit.
    #[test]
    fn ladder_l3_fires_at_threshold(nav_drop_bps in 0u64..600) {
        let limits = LossLimits::default();
        let mut ladder = LossLadder::new(limits.clone(), 100_000.0);
        ladder.on_session_open(100_000.0);
        let empty = HashMap::new();
        let drop = nav_drop_bps as f64 / 10_000.0;
        let new_nav = 100_000.0 * (1.0 - drop);
        let state = ladder.update(new_nav, &empty);
        if drop >= limits.portfolio_hard_stop {
            // L4 or deeper
            prop_assert!(matches!(state, LossLevel::L4FlattenAll | LossLevel::L5RollingHalt | LossLevel::L6TrailingHalt | LossLevel::L7PermanentKill));
        } else if drop >= limits.portfolio_intraday {
            prop_assert!(matches!(state, LossLevel::L3NoNewEntries | LossLevel::L4FlattenAll));
        } else {
            prop_assert!(matches!(state, LossLevel::None | LossLevel::L2StrategyHalt(_)));
        }
    }
}

// ---------- Backtest invariants ----------

/// A strategy that emits no orders. Used in invariant tests.
struct Noop;
impl Strategy for Noop {
    fn name(&self) -> &'static str { "noop" }
    fn on_event(&mut self, _: &MarketEvent, _: &PortfolioView, _: &[PositionView]) -> SmallVec<[OrderIntent; 4]> {
        SmallVec::new()
    }
}

proptest! {
    /// Backtest determinism: running the same Simulator twice on the same
    /// events with the same seed must produce identical equity curves.
    #[test]
    fn backtest_deterministic(seed in 1u64..1000, n_bars in 50usize..500) {
        let events = generate_gbm_universe(
            &SyntheticConfig { n_symbols: 4, n_bars, seed, ..Default::default() },
            Ts::from_nanos(0),
        );
        let mut sim_a = Simulator::new(BacktestConfig::default());
        let report_a = sim_a.run(&mut XsMomentum::new(XsMomentumConfig::default()), &events);
        let mut sim_b = Simulator::new(BacktestConfig::default());
        let report_b = sim_b.run(&mut XsMomentum::new(XsMomentumConfig::default()), &events);
        prop_assert_eq!(report_a.equity.len(), report_b.equity.len());
        for (a, b) in report_a.equity.iter().zip(report_b.equity.iter()) {
            prop_assert_eq!(a.0, b.0);
            prop_assert!((a.1 - b.1).abs() < 1e-6);
        }
    }

    /// Equity curve length must equal events length for any synthetic input.
    #[test]
    fn equity_len_equals_events_len(n_syms in 1usize..16, n_bars in 5usize..200) {
        let events = generate_gbm_universe(
            &SyntheticConfig { n_symbols: n_syms, n_bars, seed: 42, ..Default::default() },
            Ts::from_nanos(0),
        );
        let mut sim = Simulator::new(BacktestConfig::default());
        let report = sim.run(&mut Noop, &events);
        prop_assert_eq!(report.equity.len(), events.len());
    }

    /// With Noop strategy, no fills happen and final NAV == initial NAV exactly.
    #[test]
    fn noop_preserves_nav(seed in 1u64..1000) {
        let events = generate_gbm_universe(
            &SyntheticConfig { n_symbols: 4, n_bars: 100, seed, ..Default::default() },
            Ts::from_nanos(0),
        );
        let mut sim = Simulator::new(BacktestConfig {
            initial_cash: 100_000.0,
            ..Default::default()
        });
        let report = sim.run(&mut Noop, &events);
        prop_assert_eq!(report.n_fills, 0);
        prop_assert!((report.final_nav - report.initial_nav).abs() < 1e-9);
        prop_assert_eq!(report.max_drawdown, 0.0);
    }
}

// ---------- Money math properties ----------

proptest! {
    /// `Price::from_f64` + `to_f64` must round-trip within a small tolerance.
    #[test]
    fn price_roundtrip(p in 0.01_f64..1_000_000.0) {
        let price = algo_core::Price::from_f64(p).unwrap();
        let back = price.to_f64();
        prop_assert!((back - p).abs() / p < 1e-9, "price roundtrip drifted: {} vs {}", p, back);
    }

    /// Notional addition is commutative.
    #[test]
    fn notional_addition_commutative(a in -1_000_000.0_f64..1_000_000.0, b in -1_000_000.0_f64..1_000_000.0) {
        let na = algo_core::Notional::from_f64(a).unwrap();
        let nb = algo_core::Notional::from_f64(b).unwrap();
        prop_assert_eq!((na + nb).to_f64(), (nb + na).to_f64());
    }
}
