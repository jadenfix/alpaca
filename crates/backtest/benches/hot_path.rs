//! Benchmarks for the per-event hot path.
//!
//! Run with `cargo bench -p algo-backtest`.

use algo_backtest::{
    generate_gbm_universe, generate_momentum_universe, BacktestConfig, Simulator, SyntheticConfig,
};
use algo_core::{MarketEvent, Ts};
use algo_features::{log_returns_vec, Welford, Window};
use algo_strategies::{xs_momentum::XsMomentumConfig, XsMomentum};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array1;
use smallvec::SmallVec;

// ---------- helper strategies ----------

struct Noop;
impl Strategy for Noop {
    fn name(&self) -> &'static str { "noop" }
    fn on_event(
        &mut self,
        _: &MarketEvent,
        _: &PortfolioView,
        _: &[PositionView],
    ) -> SmallVec<[algo_core::OrderIntent; 4]> {
        SmallVec::new()
    }
}

// ---------- benches ----------

fn bench_backtest_noop(c: &mut Criterion) {
    let mut g = c.benchmark_group("backtest/noop");
    for &n_bars in &[1_000usize, 10_000, 100_000] {
        let events = generate_gbm_universe(
            &SyntheticConfig {
                n_symbols: 16,
                n_bars,
                seed: 7,
                ..Default::default()
            },
            Ts::from_nanos(0),
        );
        g.throughput(Throughput::Elements(events.len() as u64));
        g.bench_with_input(BenchmarkId::from_parameter(events.len()), &events, |b, events| {
            b.iter(|| {
                let mut sim = Simulator::new(BacktestConfig::default());
                let report = sim.run(&mut Noop, black_box(events));
                black_box(report)
            });
        });
    }
    g.finish();
}

fn bench_backtest_xs_momentum(c: &mut Criterion) {
    let mut g = c.benchmark_group("backtest/xs_momentum");
    for &(syms, bars) in &[(8usize, 1_000usize), (16, 5_000), (32, 5_000)] {
        let events = generate_momentum_universe(syms, bars, 60, 0.20, 50.0, 11, Ts::from_nanos(0));
        g.throughput(Throughput::Elements(events.len() as u64));
        g.bench_with_input(
            BenchmarkId::new("syms_bars", format!("{syms}x{bars}")),
            &events,
            |b, events| {
                b.iter(|| {
                    let mut sim = Simulator::new(BacktestConfig::default());
                    let mut s = XsMomentum::new(XsMomentumConfig::default());
                    let report = sim.run(&mut s, black_box(events));
                    black_box(report)
                });
            },
        );
    }
    g.finish();
}

fn bench_window_last_n(c: &mut Criterion) {
    let mut g = c.benchmark_group("window/last_n");
    let mut w: Window<f64> = Window::new(100);
    for i in 0..100 {
        w.push(Ts::from_nanos(i), i as f64);
    }
    g.bench_function("last_n_clone_20", |b| {
        b.iter(|| {
            let v = w.last_n(20).unwrap();
            black_box(v)
        });
    });
    g.bench_function("last_n_slice_20", |b| {
        b.iter(|| {
            let s: (&[(_, f64)], &[(_, f64)]) = w.last_n_slice(20).unwrap();
            black_box(s);
        });
    });
    g.finish();
}

fn bench_welford(c: &mut Criterion) {
    let mut g = c.benchmark_group("features/welford");
    let xs: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.001).sin()).collect();
    g.throughput(Throughput::Elements(xs.len() as u64));
    g.bench_function("push_10k", |b| {
        b.iter(|| {
            let mut w = Welford::new();
            for &x in &xs {
                w.push(black_box(x));
            }
            black_box(w.variance())
        });
    });
    g.finish();
}

fn bench_log_returns_vec(c: &mut Criterion) {
    let mut g = c.benchmark_group("features/log_returns_vec");
    for &n in &[64usize, 1_024, 16_384] {
        let prices: Array1<f64> = Array1::from_iter((0..n).map(|i| 100.0 + i as f64 * 0.01));
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::from_parameter(n), &prices, |b, p| {
            b.iter(|| {
                let r = log_returns_vec(p.view());
                black_box(r)
            });
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_backtest_noop,
    bench_backtest_xs_momentum,
    bench_window_last_n,
    bench_welford,
    bench_log_returns_vec,
);
criterion_main!(benches);
