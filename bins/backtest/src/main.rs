//! Backtest CLI.
//!
//! Subcommands:
//!   - `gen-data`   : generate sample bars and write to a CSV file
//!   - `synthetic`  : run a strategy on freshly-generated synthetic data
//!   - `csv`        : run a strategy on bars loaded from a CSV
//!   - `walk-forward`: run K-fold walk-forward CV and print fold metrics

use algo_backtest::{
    analyze_strategy, compute_metrics, format_report, generate_cointegrated_pair,
    generate_gbm_universe, generate_momentum_universe, recommend, run_walkforward,
    BacktestConfig, LabeledEquityPoint, Simulator, SyntheticConfig, WalkForwardConfig,
};
use algo_features::{Regime, ThreeStateMarkov};
use algo_core::{MarketEvent, Symbol, Ts};
use algo_obs::init_tracing;
use algo_storage::{read_bars_csv, write_bars_csv};
use algo_strategies::{
    garch_vol_target::GarchVolTargetConfig, hurst_regime::HurstRegimeConfig,
    kalman_pairs::KalmanPairsConfig, leadlag_pairs::LeadLagPairsConfig,
    markov_router::MarkovRouterConfig, pairs_mean_reversion::PairsConfig,
    regime_hmm::RegimeHmmConfig, xs_momentum::XsMomentumConfig, GarchVolTarget, HurstRegime,
    KalmanPairs, LeadLagPairs, MarkovRouter, PairsMeanReversion, RegimeHmm, XsMomentum,
};
use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "algo-backtest", version)]
struct Cli {
    #[arg(long, default_value_t = false)]
    json_logs: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum SyntheticKind {
    Gbm,
    Momentum,
    Cointegrated,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Generate sample data and write to CSV (for offline backtesting).
    GenData {
        #[arg(long, default_value_t = String::from("data/sample/momentum_universe.csv"))]
        out: String,
        #[arg(long, value_enum, default_value_t = SyntheticKind::Momentum)]
        kind: SyntheticKind,
        #[arg(long, default_value_t = 16)]
        symbols: usize,
        #[arg(long, default_value_t = 5000)]
        bars: usize,
        #[arg(long, default_value_t = 60)]
        bar_secs: u32,
        #[arg(long, default_value_t = 0.20)]
        sigma: f64,
        #[arg(long, default_value_t = 0.05)]
        mu: f64,
        #[arg(long, default_value_t = 0.10)]
        dispersion: f64,
        #[arg(long, default_value_t = 50.0)]
        factor_strength: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Run a strategy on freshly-generated synthetic data.
    Synthetic {
        #[arg(long, default_value_t = String::from("xs_momentum"))]
        strategy: String,
        #[arg(long, value_enum, default_value_t = SyntheticKind::Momentum)]
        kind: SyntheticKind,
        #[arg(long, default_value_t = 16)]
        symbols: usize,
        #[arg(long, default_value_t = 2_500)]
        bars: usize,
        #[arg(long, default_value_t = 60)]
        bar_secs: u32,
        #[arg(long, default_value_t = 0.05)]
        mu: f64,
        #[arg(long, default_value_t = 0.20)]
        sigma: f64,
        #[arg(long, default_value_t = 0.10)]
        dispersion: f64,
        #[arg(long, default_value_t = 50.0)]
        factor_strength: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 100_000.0)]
        initial_cash: f64,
        #[arg(long)]
        equity_csv: Option<PathBuf>,
    },
    /// Run a strategy on bars loaded from CSV.
    Csv {
        #[arg(long)]
        path: PathBuf,
        #[arg(long, default_value_t = String::from("xs_momentum"))]
        strategy: String,
        #[arg(long, default_value_t = 100_000.0)]
        initial_cash: f64,
        #[arg(long)]
        equity_csv: Option<PathBuf>,
    },
    /// Run every strategy on the same data, label each timestamp's regime
    /// from a fitted 3-state Markov-switching model on the regime-proxy
    /// symbol's returns, and emit a per-regime performance report +
    /// best-strategy-per-regime recommendation.
    BenchStrategies {
        #[arg(long)]
        path: PathBuf,
        #[arg(long, default_value_t = String::from("SPY"))]
        regime_proxy: String,
        #[arg(long, default_value_t = 100_000.0)]
        initial_cash: f64,
        #[arg(long, default_value_t = 60)]
        bar_secs: u32,
        /// Strategies to include (comma-separated). Omit for all.
        #[arg(long)]
        strategies: Option<String>,
    },
    /// K-fold walk-forward CV on a CSV bar dataset.
    WalkForward {
        #[arg(long)]
        path: PathBuf,
        #[arg(long, default_value_t = String::from("xs_momentum"))]
        strategy: String,
        #[arg(long, default_value_t = 4)]
        folds: usize,
        #[arg(long, default_value_t = 0.6)]
        train_frac: f64,
        #[arg(long, default_value_t = 100_000.0)]
        initial_cash: f64,
        #[arg(long, default_value_t = 60)]
        bar_secs: u32,
    },
}

fn build_events(
    kind: SyntheticKind,
    symbols: usize,
    bars: usize,
    bar_secs: u32,
    mu: f64,
    sigma: f64,
    dispersion: f64,
    factor_strength: f64,
    seed: u64,
) -> Vec<MarketEvent> {
    let start = Ts::from_nanos(1_700_000_000_000_000_000);
    match kind {
        SyntheticKind::Gbm => {
            let cfg = SyntheticConfig {
                n_symbols: symbols,
                n_bars: bars,
                bar_secs,
                mu_annual: mu,
                sigma_annual: sigma,
                dispersion,
                seed,
                start_price: 100.0,
            };
            generate_gbm_universe(&cfg, start)
        }
        SyntheticKind::Momentum => {
            generate_momentum_universe(symbols, bars, bar_secs, sigma, factor_strength, seed, start)
        }
        SyntheticKind::Cointegrated => {
            let a = Symbol::new("PAIRA").unwrap();
            let b = Symbol::new("PAIRB").unwrap();
            generate_cointegrated_pair(a, b, bars, bar_secs, seed, start)
        }
    }
}

fn make_strategy(name: &str) -> Result<Box<dyn algo_strategy::Strategy>> {
    match name {
        "xs_momentum" => Ok(Box::new(XsMomentum::new(XsMomentumConfig::default()))),
        "pairs_mean_reversion" => Ok(Box::new(PairsMeanReversion::new(PairsConfig::default()))),
        "kalman_pairs" => Ok(Box::new(KalmanPairs::new(KalmanPairsConfig::default()))),
        "garch_vol_target" => Ok(Box::new(GarchVolTarget::new(GarchVolTargetConfig::default()))),
        "regime_hmm" => Ok(Box::new(RegimeHmm::new(RegimeHmmConfig::default()))),
        "markov_router" => Ok(Box::new(MarkovRouter::new(MarkovRouterConfig::default()))),
        "hurst_regime" => Ok(Box::new(HurstRegime::new(HurstRegimeConfig::default()))),
        "leadlag_pairs" => Ok(Box::new(LeadLagPairs::new(LeadLagPairsConfig::default()))),
        other => Err(anyhow!(
            "unknown strategy: {other}. known: xs_momentum, pairs_mean_reversion, \
             kalman_pairs, garch_vol_target, regime_hmm, markov_router, hurst_regime, \
             leadlag_pairs"
        )),
    }
}

fn write_equity_csv(path: &PathBuf, equity: &[(i64, f64)]) -> Result<()> {
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(f);
    writeln!(w, "ts_nanos,nav")?;
    for (t, v) in equity {
        writeln!(w, "{t},{v}")?;
    }
    Ok(())
}

fn events_to_bars(events: &[MarketEvent]) -> Vec<algo_core::Bar> {
    events
        .iter()
        .filter_map(|e| match e {
            MarketEvent::Bar(b) => Some(*b),
            _ => None,
        })
        .collect()
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.json_logs);

    match cli.cmd {
        Cmd::GenData {
            out,
            kind,
            symbols,
            bars,
            bar_secs,
            sigma,
            mu,
            dispersion,
            factor_strength,
            seed,
        } => {
            let events = build_events(kind, symbols, bars, bar_secs, mu, sigma, dispersion, factor_strength, seed);
            let bars_vec = events_to_bars(&events);
            if let Some(parent) = PathBuf::from(&out).parent() {
                std::fs::create_dir_all(parent)?;
            }
            write_bars_csv(&out, &bars_vec)?;
            tracing::info!(path = %out, bars = bars_vec.len(), "wrote sample data");
            println!("wrote {} bars to {}", bars_vec.len(), out);
        }
        Cmd::Synthetic {
            strategy,
            kind,
            symbols,
            bars,
            bar_secs,
            mu,
            sigma,
            dispersion,
            factor_strength,
            seed,
            initial_cash,
            equity_csv,
        } => {
            let events = build_events(kind, symbols, bars, bar_secs, mu, sigma, dispersion, factor_strength, seed);
            tracing::info!(symbols, bars, events = events.len(), "generated synthetic data");
            let bt = BacktestConfig {
                initial_cash,
                ..Default::default()
            };
            let mut sim = Simulator::new(bt);
            let mut s = make_strategy(&strategy)?;
            let report = sim.run(s.as_mut(), &events);
            let metrics = compute_metrics(&report.equity, bar_secs);
            print_report(&strategy, &report, &metrics);
            if let Some(p) = equity_csv {
                write_equity_csv(&p, &report.equity)?;
                println!("equity curve → {}", p.display());
            }
        }
        Cmd::Csv {
            path,
            strategy,
            initial_cash,
            equity_csv,
        } => {
            let bars = read_bars_csv(&path)?;
            tracing::info!(path = %path.display(), bars = bars.len(), "loaded CSV bars");
            let events: Vec<MarketEvent> = bars.into_iter().map(MarketEvent::Bar).collect();
            let span_secs = events
                .first()
                .and_then(|e| match e {
                    MarketEvent::Bar(b) => Some(b.span_secs),
                    _ => None,
                })
                .unwrap_or(60);
            let bt = BacktestConfig {
                initial_cash,
                ..Default::default()
            };
            let mut sim = Simulator::new(bt);
            let mut s = make_strategy(&strategy)?;
            let report = sim.run(s.as_mut(), &events);
            let metrics = compute_metrics(&report.equity, span_secs);
            print_report(&strategy, &report, &metrics);
            if let Some(p) = equity_csv {
                write_equity_csv(&p, &report.equity)?;
                println!("equity curve → {}", p.display());
            }
        }
        Cmd::BenchStrategies {
            path,
            regime_proxy,
            initial_cash,
            bar_secs,
            strategies,
        } => {
            let bars = read_bars_csv(&path)?;
            let events: Vec<MarketEvent> = bars.into_iter().map(MarketEvent::Bar).collect();
            tracing::info!(
                path = %path.display(),
                events = events.len(),
                proxy = %regime_proxy,
                "bench-strategies start"
            );
            let proxy_sym = Symbol::new(&regime_proxy)
                .ok_or_else(|| anyhow!("invalid regime proxy ticker: {regime_proxy}"))?;
            // Step 1: fit a 3-state Markov model on the proxy's log returns.
            let mut proxy_returns: Vec<f64> = Vec::new();
            let mut last_proxy_px: Option<f64> = None;
            for e in &events {
                if let MarketEvent::Bar(b) = e {
                    if b.symbol == proxy_sym {
                        let px = b.close.to_f64();
                        if px > 0.0 {
                            if let Some(prev) = last_proxy_px {
                                let r = (px / prev).ln();
                                if r.is_finite() { proxy_returns.push(r); }
                            }
                            last_proxy_px = Some(px);
                        }
                    }
                }
            }
            let mut model = ThreeStateMarkov::default_equity();
            if proxy_returns.len() >= 60 {
                model.fit_baum_welch(&proxy_returns, 25, 1e-4);
            }
            // Step 2: build a per-timestamp regime label map by replaying the
            // proxy returns through the model's online filter.
            let mut model_online = model.clone();
            let mut regime_at_ts: std::collections::BTreeMap<i64, Regime> = Default::default();
            let mut last_proxy_px: Option<f64> = None;
            for e in &events {
                if let MarketEvent::Bar(b) = e {
                    if b.symbol == proxy_sym {
                        let px = b.close.to_f64();
                        if px > 0.0 {
                            if let Some(prev) = last_proxy_px {
                                let r = (px / prev).ln();
                                if r.is_finite() {
                                    model_online.update(r);
                                }
                            }
                            last_proxy_px = Some(px);
                        }
                    }
                    regime_at_ts.insert(b.ts.nanos, model_online.argmax_regime());
                }
            }
            // Step 3: run each strategy, label its equity curve by the regime
            // active at each timestamp, and analyze.
            let known_strategies = [
                "xs_momentum",
                "pairs_mean_reversion",
                "kalman_pairs",
                "garch_vol_target",
                "regime_hmm",
                "markov_router",
                "hurst_regime",
                "leadlag_pairs",
            ];
            let selected: Vec<&str> = match strategies {
                Some(ref s) => s.split(',').map(|x| x.trim()).collect(),
                None => known_strategies.to_vec(),
            };
            let bt = BacktestConfig {
                initial_cash,
                ..Default::default()
            };
            let mut reports: Vec<algo_backtest::StrategyReport> = Vec::new();
            for name in &selected {
                let mut strat = match make_strategy(name) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(strategy = name, "skipping ({e})");
                        continue;
                    }
                };
                let mut sim = Simulator::new(bt.clone());
                let report = sim.run(strat.as_mut(), &events);
                // Attach regime labels to each equity point.
                let labeled: Vec<LabeledEquityPoint> = report
                    .equity
                    .iter()
                    .map(|(t, v)| LabeledEquityPoint {
                        ts_nanos: *t,
                        nav: *v,
                        regime: regime_at_ts
                            .get(t)
                            .map(|r| r.name().to_string())
                            .unwrap_or_else(|| "unknown".to_string()),
                    })
                    .collect();
                let r = analyze_strategy(name, &labeled, bar_secs);
                reports.push(r);
            }
            let recs = recommend(&reports);
            println!("{}", format_report(&reports, &recs));
        }
        Cmd::WalkForward {
            path,
            strategy,
            folds,
            train_frac,
            initial_cash,
            bar_secs,
        } => {
            let bars = read_bars_csv(&path)?;
            let events: Vec<MarketEvent> = bars.into_iter().map(MarketEvent::Bar).collect();
            tracing::info!(path = %path.display(), events = events.len(), folds, "starting walk-forward");
            let cfg = WalkForwardConfig {
                n_folds: folds,
                train_frac,
                backtest: BacktestConfig {
                    initial_cash,
                    ..Default::default()
                },
                bar_span_secs: bar_secs,
                warm_up: false,
            };
            // Walk-forward needs Sync factories. We dispatch by strategy
            // name and call `run_walkforward` with a closure that constructs
            // a fresh strategy per fold.
            let report = match strategy.as_str() {
                "xs_momentum" => run_walkforward(&events, &cfg, || {
                    XsMomentum::new(XsMomentumConfig::default())
                }),
                "pairs_mean_reversion" => run_walkforward(&events, &cfg, || {
                    PairsMeanReversion::new(PairsConfig::default())
                }),
                "kalman_pairs" => run_walkforward(&events, &cfg, || {
                    KalmanPairs::new(KalmanPairsConfig::default())
                }),
                "garch_vol_target" => run_walkforward(&events, &cfg, || {
                    GarchVolTarget::new(GarchVolTargetConfig::default())
                }),
                "regime_hmm" => run_walkforward(&events, &cfg, || {
                    RegimeHmm::new(RegimeHmmConfig::default())
                }),
                "markov_router" => run_walkforward(&events, &cfg, || {
                    MarkovRouter::new(MarkovRouterConfig::default())
                }),
                "hurst_regime" => run_walkforward(&events, &cfg, || {
                    HurstRegime::new(HurstRegimeConfig::default())
                }),
                "leadlag_pairs" => run_walkforward(&events, &cfg, || {
                    LeadLagPairs::new(LeadLagPairsConfig::default())
                }),
                other => anyhow::bail!("unknown strategy: {other}"),
            };
            println!("=== Walk-Forward Report (strategy: {strategy}) ===");
            for f in &report.folds {
                println!(
                    "fold {}: train={} test={} return={:.3}% sharpe={:.3} max_dd={:.3}% fills={}",
                    f.fold_idx,
                    f.train_events,
                    f.test_events,
                    f.metrics.total_return_pct,
                    f.metrics.sharpe,
                    f.metrics.max_drawdown_pct,
                    f.report.n_fills,
                );
            }
            println!("--- aggregate ---");
            println!("avg test return : {:.3}%", report.avg_test_return_pct);
            println!("avg test sharpe : {:.3}", report.avg_test_sharpe);
            println!("min test sharpe : {:.3}", report.min_test_sharpe);
            println!("max test sharpe : {:.3}", report.max_test_sharpe);
            println!("positive folds  : {} / {}", report.n_positive_test_folds, report.folds.len());
        }
    }
    Ok(())
}

fn print_report(strategy: &str, report: &algo_backtest::BacktestReport, m: &algo_backtest::Metrics) {
    println!("=== Backtest Report ({strategy}) ===");
    println!("initial NAV     : ${:.2}", report.initial_nav);
    println!("final NAV       : ${:.2}", report.final_nav);
    println!("fills           : {}", report.n_fills);
    println!("rejects         : {}", report.n_rejects);
    println!("fees paid       : ${:.2}", report.fees_paid);
    println!("--- analytics ---");
    println!("{}", m.pretty());
}
