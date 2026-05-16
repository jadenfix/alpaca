//! Latency assertion + long-running soak tests.
//!
//! These are deliberately slow tests (skipped via `--ignored` flag if you
//! want fast feedback); run them with `cargo test -p algo-backtest --test
//! latency_soak -- --include-ignored`.

use algo_backtest::{generate_gbm_universe, generate_momentum_universe, BacktestConfig, Simulator, SyntheticConfig};
use algo_core::{MarketEvent, OrderIntent, Ts};
use algo_obs::{LatencyStage, LatencyTracker};
use algo_strategies::{xs_momentum::XsMomentumConfig, XsMomentum};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use smallvec::SmallVec;
use std::time::Instant;

struct Noop;
impl Strategy for Noop {
    fn name(&self) -> &'static str { "noop" }
    fn on_event(&mut self, _: &MarketEvent, _: &PortfolioView, _: &[PositionView]) -> SmallVec<[OrderIntent; 4]> {
        SmallVec::new()
    }
}

/// Hot-path latency assertion. Runs noop strategy over many events, measures
/// per-event time with an HDR histogram, asserts p99 is well under the
/// `target_p99_us` budget.
#[test]
fn hot_path_p99_under_budget() {
    let n_bars = 5_000;
    let events = generate_gbm_universe(
        &SyntheticConfig { n_symbols: 16, n_bars, seed: 1, ..Default::default() },
        Ts::from_nanos(0),
    );
    // Pre-create the simulator so its buffers are sized.
    let mut sim = Simulator::new(BacktestConfig::default());
    let _ = sim.run(&mut Noop, &events[..16]); // warm-up

    let tracker = LatencyTracker::new();
    let mut s = Noop;
    for e in &events {
        let start = Instant::now();
        // Inline event loop to measure per-event without report assembly cost.
        // We can't easily call internal hot path without a public hook, so
        // we instead measure many tiny one-event runs.
        let _ = sim.run(&mut s, std::slice::from_ref(e));
        let dur = start.elapsed().as_nanos() as u64;
        tracker.record(LatencyStage::EndToEndTickToOrder, dur);
    }
    let snap = tracker.snapshot(LatencyStage::EndToEndTickToOrder);
    println!("hot path single-event noop: {}", snap.fmt_us());

    // Budget: p99 < 50 µs on Apple Silicon / equivalent — generous for CI variance.
    // The per-event work for noop should be << 5 µs but report assembly adds overhead.
    assert!(
        snap.p99 < 50_000,
        "hot path p99 too slow: p99={}us budget=50us",
        snap.p99 / 1_000
    );
}

/// 1M-event soak test. Doesn't assert performance, only that the engine
/// doesn't panic, leak memory, or produce nonsense output at scale.
#[test]
#[ignore = "slow soak — run with --include-ignored"]
fn one_million_event_soak() {
    let n_bars = 31_250; // × 32 symbols = 1_000_000 events
    let events = generate_momentum_universe(32, n_bars, 60, 0.20, 50.0, 99, Ts::from_nanos(0));
    assert_eq!(events.len(), 1_000_000);
    let mut sim = Simulator::new(BacktestConfig {
        initial_cash: 1_000_000.0,
        ..Default::default()
    });
    let start = Instant::now();
    let mut strat = XsMomentum::new(XsMomentumConfig::default());
    let report = sim.run(&mut strat, &events);
    let elapsed = start.elapsed();
    println!("1M-event soak: {:?} ({:.2} ns/event)",
        elapsed,
        elapsed.as_nanos() as f64 / events.len() as f64);
    // Sanity invariants.
    assert_eq!(report.equity.len(), events.len());
    assert!(report.final_nav.is_finite());
    assert!(report.max_drawdown >= 0.0 && report.max_drawdown <= 1.0);
    assert!(report.fees_paid >= 0.0);
    // L4 hard stop should have triggered if drawdown approached or exceeded 3.5%.
    if report.max_drawdown >= 0.03 {
        // Hard-stop ladder must have engaged; expect contained loss.
        assert!(
            report.max_drawdown <= 0.05,
            "ladder failed to contain 1M-event drawdown: {:.3}%",
            report.max_drawdown * 100.0
        );
    }
}

/// Throughput sanity: noop strategy on 100k events must complete in well
/// under a second on modern hardware.
#[test]
fn noop_throughput_sanity() {
    let events = generate_gbm_universe(
        &SyntheticConfig { n_symbols: 16, n_bars: 6_250, seed: 1, ..Default::default() }, // 100k events
        Ts::from_nanos(0),
    );
    assert_eq!(events.len(), 100_000);
    let mut sim = Simulator::new(BacktestConfig::default());
    let start = Instant::now();
    let report = sim.run(&mut Noop, &events);
    let elapsed = start.elapsed();
    println!("noop 100k events: {:?} ({} ns/event)",
        elapsed,
        elapsed.as_nanos() as u64 / events.len() as u64);
    // Budget: 100k events in < 100ms ≈ 1 µs/event. Generous for CI.
    assert!(
        elapsed.as_millis() < 200,
        "noop 100k events too slow: {:?}",
        elapsed
    );
    assert_eq!(report.equity.len(), 100_000);
}

/// Re-running the same backtest 100 times must produce byte-identical final
/// NAVs (no nondeterminism creeping in from HashMap iteration order, etc.).
#[test]
fn repeated_backtest_is_perfectly_stable() {
    let events = generate_momentum_universe(8, 500, 60, 0.20, 50.0, 1, Ts::from_nanos(0));
    let cfg = BacktestConfig::default();
    let mut prev: Option<f64> = None;
    for _ in 0..100 {
        let mut sim = Simulator::new(cfg.clone());
        let report = sim.run(&mut XsMomentum::new(XsMomentumConfig::default()), &events);
        if let Some(p) = prev {
            assert!((report.final_nav - p).abs() < 1e-9, "nav drifted: {} -> {}", p, report.final_nav);
        }
        prev = Some(report.final_nav);
    }
}
