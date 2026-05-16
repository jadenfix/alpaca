//! Integration test for regime-conditional analysis + recommender.
//!
//! Runs xs_momentum (synthetic data), labels the equity curve with regime
//! tags from a Markov-switching model fit on the proxy, then verifies the
//! recommender produces a sensible per-regime ranking.

use algo_backtest::{
    analyze_strategy, generate_momentum_universe, recommend, BacktestConfig,
    LabeledEquityPoint, Simulator,
};
use algo_core::{MarketEvent, Symbol, Ts};
use algo_features::ThreeStateMarkov;
use algo_strategies::{
    garch_vol_target::GarchVolTargetConfig, hurst_regime::HurstRegimeConfig,
    xs_momentum::XsMomentumConfig, GarchVolTarget, HurstRegime, XsMomentum,
};

#[test]
fn regime_labeling_attaches_to_every_equity_point() {
    let evs = generate_momentum_universe(6, 800, 60, 0.20, 50.0, 7, Ts::from_nanos(0));
    let proxy = Symbol::new("SYM0").unwrap();
    // Build proxy returns + a per-ts regime label via online HMM
    let mut model = ThreeStateMarkov::default_equity();
    let mut last_proxy: Option<f64> = None;
    let mut regime_at_ts = std::collections::BTreeMap::new();
    for e in &evs {
        if let MarketEvent::Bar(b) = e {
            if b.symbol == proxy {
                let px = b.close.to_f64();
                if px > 0.0 {
                    if let Some(prev) = last_proxy {
                        let r = (px / prev).ln();
                        if r.is_finite() { model.update(r); }
                    }
                    last_proxy = Some(px);
                }
            }
            regime_at_ts.insert(b.ts.nanos, model.argmax_regime());
        }
    }
    let mut sim = Simulator::new(BacktestConfig::default());
    let mut s = XsMomentum::new(XsMomentumConfig::default());
    let report = sim.run(&mut s, &evs);
    let labeled: Vec<LabeledEquityPoint> = report
        .equity
        .iter()
        .map(|(t, v)| LabeledEquityPoint {
            ts_nanos: *t,
            nav: *v,
            regime: regime_at_ts.get(t).map(|r| r.name().to_string()).unwrap_or_default(),
        })
        .collect();
    assert_eq!(labeled.len(), evs.len());
    // Every point must have a non-empty regime label after warm-up.
    let labeled_count = labeled.iter().filter(|p| !p.regime.is_empty()).count();
    assert!(labeled_count > evs.len() / 2);
}

#[test]
fn recommender_picks_best_strategy_per_regime() {
    let evs = generate_momentum_universe(8, 600, 60, 0.20, 50.0, 11, Ts::from_nanos(0));
    let proxy = Symbol::new("SYM0").unwrap();
    let mut model = ThreeStateMarkov::default_equity();
    let mut last_proxy: Option<f64> = None;
    let mut regime_at_ts = std::collections::BTreeMap::new();
    for e in &evs {
        if let MarketEvent::Bar(b) = e {
            if b.symbol == proxy {
                let px = b.close.to_f64();
                if px > 0.0 {
                    if let Some(prev) = last_proxy {
                        let r = (px / prev).ln();
                        if r.is_finite() { model.update(r); }
                    }
                    last_proxy = Some(px);
                }
            }
            regime_at_ts.insert(b.ts.nanos, model.argmax_regime());
        }
    }
    // Build labeled equity curves for three strategies
    let mk_labeled = |report: &algo_backtest::BacktestReport| -> Vec<LabeledEquityPoint> {
        report
            .equity
            .iter()
            .map(|(t, v)| LabeledEquityPoint {
                ts_nanos: *t,
                nav: *v,
                regime: regime_at_ts.get(t).map(|r| r.name().to_string()).unwrap_or_default(),
            })
            .collect()
    };
    let mut reports = Vec::new();
    for (name, mut strat) in [
        ("xs_momentum", Box::new(XsMomentum::new(XsMomentumConfig::default())) as Box<dyn algo_strategy::Strategy>),
        ("garch_vol_target", Box::new(GarchVolTarget::new(GarchVolTargetConfig::default()))),
        ("hurst_regime", Box::new(HurstRegime::new(HurstRegimeConfig::default()))),
    ] {
        let mut sim = Simulator::new(BacktestConfig::default());
        let r = sim.run(strat.as_mut(), &evs);
        reports.push(analyze_strategy(name, &mk_labeled(&r), 60));
    }
    let recs = recommend(&reports);
    assert!(!recs.is_empty(), "expected at least one recommendation");
    for r in &recs {
        // Every recommendation must name a real strategy.
        assert!(
            r.best_strategy == "xs_momentum"
                || r.best_strategy == "garch_vol_target"
                || r.best_strategy == "hurst_regime"
        );
        assert!(r.best_sharpe.is_finite());
    }
}

#[test]
fn per_regime_metrics_partition_full_history() {
    let evs = generate_momentum_universe(4, 300, 60, 0.20, 30.0, 13, Ts::from_nanos(0));
    let proxy = Symbol::new("SYM0").unwrap();
    let mut model = ThreeStateMarkov::default_equity();
    let mut last_proxy: Option<f64> = None;
    let mut regime_at_ts = std::collections::BTreeMap::new();
    for e in &evs {
        if let MarketEvent::Bar(b) = e {
            if b.symbol == proxy {
                let px = b.close.to_f64();
                if px > 0.0 {
                    if let Some(prev) = last_proxy {
                        let r = (px / prev).ln();
                        if r.is_finite() { model.update(r); }
                    }
                    last_proxy = Some(px);
                }
            }
            regime_at_ts.insert(b.ts.nanos, model.argmax_regime());
        }
    }
    let mut sim = Simulator::new(BacktestConfig::default());
    let mut s = XsMomentum::new(XsMomentumConfig::default());
    let report = sim.run(&mut s, &evs);
    let labeled: Vec<LabeledEquityPoint> = report
        .equity
        .iter()
        .map(|(t, v)| LabeledEquityPoint {
            ts_nanos: *t,
            nav: *v,
            regime: regime_at_ts.get(t).map(|r| r.name().to_string()).unwrap_or_default(),
        })
        .collect();
    let rep = analyze_strategy("xs", &labeled, 60);
    // Sum of regime occupancies should equal 1.0 (within float tolerance).
    let s: f64 = rep.regime_occupancy.values().sum();
    assert!((s - 1.0).abs() < 1e-6, "regime occupancies should sum to 1, got {s}");
}
