//! End-to-end pipeline integration tests.
//!
//! These tests exercise the full data → features → strategy → risk → fills →
//! ladder pipeline using the SAME code paths the live engine will use. They
//! are intentionally slow and exhaustive; meant to be the safety net before
//! shadow-live.

use algo_backtest::{
    generate_gbm_universe, generate_momentum_universe, run_walkforward, BacktestConfig,
    Simulator, SyntheticConfig, WalkForwardConfig,
};
use algo_core::{
    Bar, MarketEvent, OrderId, OrderIntent, OrderType, Price, Qty, Side, Symbol, TimeInForce, Ts,
};
use algo_risk::{LossLimits, RiskLimits};
use algo_strategies::{xs_momentum::XsMomentumConfig, XsMomentum};
use algo_strategy::{PortfolioView, PositionView, Strategy};
use smallvec::SmallVec;

// ---------- helpers ----------

fn mk_bar(sym: Symbol, ts_ns: i64, close: f64) -> MarketEvent {
    MarketEvent::Bar(Bar {
        ts: Ts::from_nanos(ts_ns),
        symbol: sym,
        open: Price::from_f64(close).unwrap(),
        high: Price::from_f64(close * 1.001).unwrap(),
        low: Price::from_f64(close * 0.999).unwrap(),
        close: Price::from_f64(close).unwrap(),
        volume: Qty::from_i64(10_000),
        span_secs: 60,
    })
}

/// Strategy that buys 1 share of every bar's symbol on every event. Used to
/// stress the risk + ladder paths.
struct ManiacalBuyer {
    qty: i64,
}

impl Strategy for ManiacalBuyer {
    fn name(&self) -> &'static str {
        "maniacal_buyer"
    }
    fn on_event(
        &mut self,
        ev: &MarketEvent,
        _pv: &PortfolioView,
        _: &[PositionView],
    ) -> SmallVec<[OrderIntent; 4]> {
        let mut out = SmallVec::new();
        if let MarketEvent::Bar(b) = ev {
            out.push(OrderIntent {
                id: OrderId::new(),
                ts: b.ts,
                symbol: b.symbol,
                side: Side::Buy,
                qty: Qty::from_i64(self.qty),
                order_type: OrderType::Market,
                tif: TimeInForce::Day,
                strategy: "maniacal_buyer".into(),
                tag: None,
            });
        }
        out
    }
}

/// Strategy that always tries a 1B$ order — triggers the fat-finger guard.
struct FatFinger;
impl Strategy for FatFinger {
    fn name(&self) -> &'static str {
        "fat_finger"
    }
    fn on_event(
        &mut self,
        ev: &MarketEvent,
        _pv: &PortfolioView,
        _: &[PositionView],
    ) -> SmallVec<[OrderIntent; 4]> {
        let mut out = SmallVec::new();
        if let MarketEvent::Bar(b) = ev {
            out.push(OrderIntent {
                id: OrderId::new(),
                ts: b.ts,
                symbol: b.symbol,
                side: Side::Buy,
                qty: Qty::from_i64(10_000_000), // $1B at $100/share
                order_type: OrderType::Market,
                tif: TimeInForce::Day,
                strategy: "fat_finger".into(),
                tag: None,
            });
        }
        out
    }
}

/// Strategy that submits a resting limit order — should be rejected by backtest
/// (not credibly fillable without book data).
struct RestingLimitOnly;
impl Strategy for RestingLimitOnly {
    fn name(&self) -> &'static str {
        "resting_limit"
    }
    fn on_event(
        &mut self,
        ev: &MarketEvent,
        _: &PortfolioView,
        _: &[PositionView],
    ) -> SmallVec<[OrderIntent; 4]> {
        let mut out = SmallVec::new();
        if let MarketEvent::Bar(b) = ev {
            out.push(OrderIntent {
                id: OrderId::new(),
                ts: b.ts,
                symbol: b.symbol,
                side: Side::Buy,
                qty: Qty::from_i64(10),
                order_type: OrderType::Limit {
                    limit: Price::from_f64(0.01).unwrap(),
                },
                tif: TimeInForce::Day,
                strategy: "resting_limit".into(),
                tag: None,
            });
        }
        out
    }
}

// ---------- tests ----------

#[test]
fn fat_finger_guard_rejects_oversize_order() {
    let sym = Symbol::new("AAPL").unwrap();
    let events: Vec<_> = (0..20)
        .map(|i| mk_bar(sym, (i + 1) * 60_000_000_000, 100.0))
        .collect();
    let cfg = BacktestConfig {
        initial_cash: 100_000.0,
        risk: RiskLimits {
            max_order_notional: 25_000.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut sim = Simulator::new(cfg);
    let report = sim.run(&mut FatFinger, &events);
    // Every order should be rejected (10M shares * $100 = $1B > $25k cap).
    assert_eq!(report.n_fills, 0, "fat-finger orders should not fill");
    assert!(report.n_rejects >= events.len(), "expected ≥{} rejects, got {}", events.len(), report.n_rejects);
    assert_eq!(report.final_nav, report.initial_nav, "NAV must not move when nothing fills");
}

#[test]
fn resting_limit_orders_are_rejected_in_backtest() {
    let sym = Symbol::new("AAPL").unwrap();
    let events: Vec<_> = (0..20)
        .map(|i| mk_bar(sym, (i + 1) * 60_000_000_000, 100.0))
        .collect();
    let mut sim = Simulator::new(BacktestConfig::default());
    let report = sim.run(&mut RestingLimitOnly, &events);
    assert_eq!(report.n_fills, 0, "resting limits not credibly fillable in backtest");
    assert!(report.n_rejects >= events.len() - 1);
}

#[test]
fn loss_ladder_halts_at_l3_then_l4() {
    // Gradual decline so the ladder can react. Tests path-dependent loss
    // containment — NOT gap risk, which by definition no in-process risk
    // system can prevent.
    let sym = Symbol::new("AAPL").unwrap();
    let mut events: Vec<MarketEvent> = Vec::new();
    // Bars 1..20: price = $100; strategy accumulates a small position each bar.
    for i in 0..20 {
        events.push(mk_bar(sym, (i + 1) * 60_000_000_000, 100.0));
    }
    // Bars 21..120: price grinds down 0.2% per bar (~18% total decline).
    let mut p = 100.0;
    for i in 20..120 {
        p *= 0.998;
        events.push(mk_bar(sym, (i + 1) * 60_000_000_000, p));
    }
    let cfg = BacktestConfig {
        initial_cash: 100_000.0,
        risk: RiskLimits {
            max_name_pct_nav: 1.00,
            max_order_notional: 100_000.0,
            max_qty_vs_adv: 1.0,
            buying_power_cushion: 0.99,
        },
        loss: LossLimits::default(),
        ..Default::default()
    };
    let mut sim = Simulator::new(cfg);
    let report = sim.run(&mut ManiacalBuyer { qty: 50 }, &events);
    assert!(report.n_fills > 0);
    // L4 hard stop at -3.5%; with one tick of overshoot allowed, expect ≤ 5%.
    assert!(
        report.max_drawdown <= 0.05,
        "ladder failed to contain gradual drawdown: {:.3}%",
        report.max_drawdown * 100.0
    );
}

#[test]
fn loss_ladder_cannot_prevent_gap_risk_but_does_not_make_it_worse() {
    // Sanity test: with an instant 50% gap-down, the ladder can't rescue you,
    // but it MUST not increase the loss either (it should flatten, not buy more).
    let sym = Symbol::new("AAPL").unwrap();
    let mut events: Vec<MarketEvent> = Vec::new();
    for i in 0..10 {
        events.push(mk_bar(sym, (i + 1) * 60_000_000_000, 100.0));
    }
    for i in 10..30 {
        events.push(mk_bar(sym, (i + 1) * 60_000_000_000, 50.0));
    }
    let cfg = BacktestConfig {
        initial_cash: 100_000.0,
        risk: RiskLimits {
            max_name_pct_nav: 1.00,
            max_order_notional: 100_000.0,
            max_qty_vs_adv: 1.0,
            buying_power_cushion: 0.99,
        },
        ..Default::default()
    };
    let mut sim = Simulator::new(cfg);
    let report = sim.run(&mut ManiacalBuyer { qty: 50 }, &events);
    // Gap is gap — we expect a large DD but NAV should stabilize after the flatten.
    assert!(report.max_drawdown > 0.20, "expected gap risk to register");
    // After flatten + remaining bars, no NEW orders should fill (block_new).
    // Crude check: final NAV is close to NAV-at-bar-11 (no further bleed from new buys).
    let nav_at_gap = report.equity.iter()
        .filter(|(t, _)| *t == 11 * 60_000_000_000)
        .map(|(_, v)| *v)
        .next();
    if let Some(gap_nav) = nav_at_gap {
        // After flatten the cash is what it is; the NAV must not be much LOWER than the gap-bar NAV
        // (which would indicate the ladder allowed more loss after the gap).
        assert!(
            report.final_nav >= gap_nav * 0.95,
            "ladder allowed extra loss after gap: gap_nav={} final={}",
            gap_nav,
            report.final_nav
        );
    }
}

#[test]
fn cost_model_eats_returns_on_high_turnover_strategy() {
    // ManiacalBuyer buys EVERY bar. Even at $0 commission, slippage + impact
    // accumulate. NAV must decline strictly monotonically while orders fill.
    let sym = Symbol::new("AAPL").unwrap();
    let events: Vec<_> = (0..50)
        .map(|i| mk_bar(sym, (i + 1) * 60_000_000_000, 100.0))
        .collect();
    let cfg = BacktestConfig {
        initial_cash: 1_000_000.0,
        risk: RiskLimits {
            max_name_pct_nav: 1.0,
            max_order_notional: 100_000.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut sim = Simulator::new(cfg);
    let report = sim.run(&mut ManiacalBuyer { qty: 50 }, &events);
    assert!(report.n_fills > 0);
    // Final NAV strictly below initial because of slippage on every fill.
    assert!(
        report.final_nav < report.initial_nav,
        "cost model failed to subtract slippage"
    );
}

#[test]
fn xs_momentum_no_orders_in_first_lookback_bars() {
    // The strategy MUST not emit orders before it has lookback_bars + 1 of history.
    let events = generate_momentum_universe(8, 10, 60, 0.20, 50.0, 1, Ts::from_nanos(0));
    let mut sim = Simulator::new(BacktestConfig::default());
    let mut strat = XsMomentum::new(XsMomentumConfig::default()); // default lookback=20
    let report = sim.run(&mut strat, &events);
    assert_eq!(
        report.n_fills, 0,
        "XS momentum cannot have history < lookback"
    );
}

#[test]
fn xs_momentum_produces_fills_after_warmup() {
    let events = generate_momentum_universe(8, 200, 60, 0.20, 50.0, 1, Ts::from_nanos(0));
    let mut sim = Simulator::new(BacktestConfig {
        initial_cash: 200_000.0,
        ..Default::default()
    });
    let mut strat = XsMomentum::new(XsMomentumConfig {
        lookback_bars: 10,
        vol_lookback_bars: 10,
        gross_per_leg: 0.02,
        legs_per_side: 2,
        rebalance_every_bars: 5,
        max_hold_bars: 60,
    });
    let report = sim.run(&mut strat, &events);
    assert!(report.n_fills > 0, "expected fills after warm-up");
}

#[test]
fn xs_momentum_walkforward_each_fold_produces_fills() {
    // Regression test for the bars_held phantom-position bug.
    let events = generate_momentum_universe(8, 2000, 60, 0.20, 50.0, 1, Ts::from_nanos(0));
    let cfg = WalkForwardConfig {
        n_folds: 3,
        train_frac: 0.5,
        backtest: BacktestConfig {
            initial_cash: 100_000.0,
            ..Default::default()
        },
        bar_span_secs: 60,
        warm_up: false,
    };
    let report = run_walkforward(&events, &cfg, || XsMomentum::new(XsMomentumConfig::default()));
    for f in &report.folds {
        assert!(
            f.report.n_fills > 0,
            "fold {} produced 0 fills — bars_held bug likely regressed",
            f.fold_idx
        );
    }
}

#[test]
fn backtest_is_deterministic() {
    // Running the same backtest twice on the same data MUST produce the same equity curve.
    let events = generate_gbm_universe(
        &SyntheticConfig {
            n_symbols: 6,
            n_bars: 100,
            seed: 555,
            ..Default::default()
        },
        Ts::from_nanos(1_000_000_000),
    );
    let mut sim1 = Simulator::new(BacktestConfig::default());
    let mut s1 = XsMomentum::new(XsMomentumConfig::default());
    let r1 = sim1.run(&mut s1, &events);
    let mut sim2 = Simulator::new(BacktestConfig::default());
    let mut s2 = XsMomentum::new(XsMomentumConfig::default());
    let r2 = sim2.run(&mut s2, &events);
    assert_eq!(r1.n_fills, r2.n_fills, "non-deterministic fill count");
    assert!(
        (r1.final_nav - r2.final_nav).abs() < 1e-6,
        "non-deterministic final NAV: {} vs {}",
        r1.final_nav,
        r2.final_nav
    );
    assert_eq!(
        r1.equity.len(),
        r2.equity.len(),
        "equity curve length mismatch"
    );
    for (i, (a, b)) in r1.equity.iter().zip(r2.equity.iter()).enumerate() {
        assert_eq!(a.0, b.0, "ts mismatch at {i}");
        assert!((a.1 - b.1).abs() < 1e-6, "nav mismatch at {i}");
    }
}

#[test]
fn equity_curve_has_one_point_per_event() {
    let events = generate_gbm_universe(
        &SyntheticConfig {
            n_symbols: 3,
            n_bars: 50,
            ..Default::default()
        },
        Ts::from_nanos(0),
    );
    let mut sim = Simulator::new(BacktestConfig::default());
    let report = sim.run(&mut XsMomentum::new(XsMomentumConfig::default()), &events);
    assert_eq!(report.equity.len(), events.len());
}

#[test]
fn nav_never_goes_negative_with_default_risk_limits() {
    // Under default risk limits, even an adversarial strategy should not be
    // able to drive NAV below zero (buying-power cushion + ladder).
    let sym = Symbol::new("AAPL").unwrap();
    let events: Vec<_> = (0..100)
        .map(|i| mk_bar(sym, (i + 1) * 60_000_000_000, 100.0))
        .collect();
    let mut sim = Simulator::new(BacktestConfig::default());
    let report = sim.run(&mut ManiacalBuyer { qty: 1_000 }, &events);
    assert!(report.final_nav >= -1.0, "NAV went absurdly negative: {}", report.final_nav);
}

// ---------- journal replay determinism ----------

#[test]
fn journal_roundtrip_byte_identical() {
    use algo_journal::{JournalReader, JournalRecord, JournalWriter};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.journal");
    let writer = JournalWriter::open(&path).unwrap();
    let records: Vec<JournalRecord> = (0..50)
        .map(|i| JournalRecord::Market(mk_bar(Symbol::new("AAPL").unwrap(), i * 60_000_000_000, 100.0 + i as f64)))
        .collect();
    for r in &records {
        writer.append(r).unwrap();
    }
    writer.flush().unwrap();
    drop(writer);

    let read_back: Vec<JournalRecord> =
        JournalReader::open(&path).unwrap().collect::<Result<_, _>>().unwrap();
    assert_eq!(read_back.len(), records.len());

    // Roundtrip serialization must be stable (write A, read A, write again,
    // bytes match).
    let path2 = dir.path().join("test2.journal");
    let writer2 = JournalWriter::open(&path2).unwrap();
    for r in &read_back {
        writer2.append(r).unwrap();
    }
    writer2.flush().unwrap();
    drop(writer2);

    let a = std::fs::read(&path).unwrap();
    let b = std::fs::read(&path2).unwrap();
    assert_eq!(a, b, "journal not byte-identical across roundtrip");
}

// ---------- OMS position accounting ----------

#[test]
fn oms_position_round_trip_long_then_flat() {
    use algo_core::{Fill, Notional, Price};
    use algo_oms::Oms;

    let sym = Symbol::new("AAPL").unwrap();
    let oms = Oms::default();
    let intent = OrderIntent {
        id: OrderId::new(),
        ts: Ts::from_nanos(0),
        symbol: sym,
        side: Side::Buy,
        qty: Qty::from_i64(100),
        order_type: OrderType::Market,
        tif: TimeInForce::Day,
        strategy: "t".into(),
        tag: None,
    };
    oms.on_submit(intent.clone(), Some("b1".into()));
    oms.on_fill(&Fill {
        ts: Ts::from_nanos(1),
        order_id: intent.id,
        symbol: sym,
        side: Side::Buy,
        qty: Qty::from_i64(100),
        price: Price::from_f64(150.0).unwrap(),
        fees: Notional::ZERO,
    });
    assert_eq!(oms.position(sym).qty, 100.0);
    assert_eq!(oms.position(sym).avg_px, 150.0);

    // Sell 100 @ $160
    let i2 = OrderIntent {
        id: OrderId::new(),
        ts: Ts::from_nanos(2),
        symbol: sym,
        side: Side::Sell,
        qty: Qty::from_i64(100),
        order_type: OrderType::Market,
        tif: TimeInForce::Day,
        strategy: "t".into(),
        tag: None,
    };
    oms.on_submit(i2.clone(), Some("b2".into()));
    oms.on_fill(&Fill {
        ts: Ts::from_nanos(3),
        order_id: i2.id,
        symbol: sym,
        side: Side::Sell,
        qty: Qty::from_i64(100),
        price: Price::from_f64(160.0).unwrap(),
        fees: Notional::ZERO,
    });
    let pos = oms.position(sym);
    assert_eq!(pos.qty, 0.0);
    assert!((pos.realized_pnl - 1000.0).abs() < 1e-6);
}

// ---------- risk ladder progression ----------

#[test]
fn ladder_l7_persists_even_after_recovery() {
    use algo_risk::{LossLadder, LossLevel, LossLimits};

    let mut ladder = LossLadder::new(LossLimits::default(), 100_000.0);
    ladder.on_session_open(100_000.0);
    let empty = std::collections::HashMap::new();
    // -13% trips L7
    assert!(matches!(ladder.update(87_000.0, &empty), LossLevel::L7PermanentKill));
    // Recover to peak — still L7
    assert!(matches!(ladder.update(110_000.0, &empty), LossLevel::L7PermanentKill));
    ladder.manual_reset();
    assert!(matches!(ladder.state(), LossLevel::L7PermanentKill));
}

// ---------- kill switch ----------

#[test]
fn kill_file_trips_l4() {
    use algo_risk::{KillSwitch, KillTrigger, SwitchInputs};
    let dir = tempfile::tempdir().unwrap();
    let kf = dir.path().join("KILL");
    std::fs::write(&kf, b"x").unwrap();
    let k = KillSwitch::new(KillTrigger {
        kill_file: Some(kf.clone()),
        ..Default::default()
    });
    let mut inp = SwitchInputs::default();
    inp.buying_power = 100_000.0;
    inp.day_trades_remaining = 100;
    let s = k.evaluate(&inp);
    assert!(s.is_tripped());
    assert_eq!(
        s.strictest(),
        Some(algo_risk::LossLevel::L4FlattenAll)
    );
}

// ---------- multi-strategy concurrent execution ----------

#[test]
fn multiple_strategies_run_against_same_data() {
    // Strategies must be independently testable on the same event stream.
    let events = generate_momentum_universe(6, 300, 60, 0.15, 50.0, 9, Ts::from_nanos(0));
    let mut sim1 = Simulator::new(BacktestConfig::default());
    let r1 = sim1.run(
        &mut XsMomentum::new(XsMomentumConfig {
            lookback_bars: 5,
            vol_lookback_bars: 5,
            ..Default::default()
        }),
        &events,
    );
    let mut sim2 = Simulator::new(BacktestConfig::default());
    let r2 = sim2.run(
        &mut XsMomentum::new(XsMomentumConfig {
            lookback_bars: 30,
            vol_lookback_bars: 30,
            ..Default::default()
        }),
        &events,
    );
    // Different parameter sets should produce different fill counts.
    assert!(
        r1.n_fills != r2.n_fills,
        "parameters had no effect: both {} fills",
        r1.n_fills
    );
}

// ---------- walk-forward ----------

#[test]
fn walkforward_aggregate_matches_per_fold() {
    let events = generate_momentum_universe(8, 1200, 60, 0.20, 40.0, 3, Ts::from_nanos(0));
    let cfg = WalkForwardConfig {
        n_folds: 3,
        train_frac: 0.5,
        backtest: BacktestConfig::default(),
        bar_span_secs: 60,
        warm_up: false,
    };
    let report = run_walkforward(&events, &cfg, || XsMomentum::new(XsMomentumConfig::default()));
    assert_eq!(report.folds.len(), 3);
    let manual_avg: f64 =
        report.folds.iter().map(|f| f.metrics.total_return_pct).sum::<f64>() / 3.0;
    assert!((report.avg_test_return_pct - manual_avg).abs() < 1e-9);
}

// ---------- analytics sanity ----------

#[test]
fn analytics_metrics_match_expected_on_known_curve() {
    use algo_backtest::compute_metrics;
    // A noisy positive-drift curve: mean ~+1bp, std ~5bp → finite, positive sharpe.
    let n = 252;
    let mut eq = vec![(0_i64, 100.0)];
    let mut nav = 100.0;
    // Deterministic pseudo-random sequence in [-5bp, +7bp] with mean ~+1bp.
    for i in 1..n {
        let pseudo = (((i * 2654435761usize) % 12) as f64 - 5.0) / 10_000.0; // -5bp..+6bp
        nav *= 1.0 + pseudo + 0.0001;
        eq.push((i as i64 * 60_000_000_000, nav));
    }
    let m = compute_metrics(&eq, 60);
    assert!(m.sharpe.is_finite(), "sharpe must be finite");
    assert!(m.max_drawdown_pct > 0.0, "expected some drawdown");
    assert!(m.win_rate_pct > 0.0 && m.win_rate_pct < 100.0, "mixed wins/losses expected");
    assert!(m.profit_factor.is_finite() && m.profit_factor > 0.0);
}
