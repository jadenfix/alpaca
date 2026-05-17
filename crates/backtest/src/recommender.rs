//! Strategy recommender: per-regime quantitative analysis.
//!
//! Given a set of strategy backtest results (equity curves with regime labels
//! attached), compute:
//!   - per-regime Sharpe / Sortino / hit rate / max drawdown
//!   - regime occupancy (% of time spent in each regime)
//!   - the best strategy for each regime by risk-adjusted return
//!
//! Output is intended for a human operator deciding which strategy to enable
//! conditional on the current regime, OR for a meta-strategy that does the
//! switching automatically.

use crate::analytics::compute as compute_metrics;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One observation in the per-strategy time series.
#[derive(Clone, Debug)]
pub struct LabeledEquityPoint {
    pub ts_nanos: i64,
    pub nav: f64,
    pub regime: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PerRegimeMetrics {
    pub regime: String,
    pub n_periods: usize,
    pub total_return_pct: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub max_drawdown_pct: f64,
    pub win_rate_pct: f64,
    pub profit_factor: f64,
}

#[derive(Clone, Debug)]
pub struct StrategyReport {
    pub strategy: String,
    pub per_regime: BTreeMap<String, PerRegimeMetrics>,
    pub overall: PerRegimeMetrics,
    /// Fraction of total time spent in each regime.
    pub regime_occupancy: BTreeMap<String, f64>,
}

#[derive(Clone, Debug)]
pub struct Recommendation {
    pub regime: String,
    pub best_strategy: String,
    pub best_sharpe: f64,
    /// Sharpe of the next-best strategy, for diff display.
    pub runner_up_sharpe: Option<f64>,
}

/// Compute per-regime + overall metrics from a strategy's labeled equity curve.
pub fn analyze_strategy(
    strategy_name: &str,
    series: &[LabeledEquityPoint],
    bar_span_secs: u32,
) -> StrategyReport {
    // Group consecutive points by regime label into sub-equity-curves.
    // For Sharpe within a regime we slice the period returns by their regime label.
    let mut per_regime_returns: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut occupancy: BTreeMap<String, usize> = BTreeMap::new();
    for w in series.windows(2) {
        let r = if w[0].nav > 0.0 { w[1].nav / w[0].nav - 1.0 } else { 0.0 };
        if r.is_finite() {
            per_regime_returns.entry(w[1].regime.clone()).or_default().push(r);
        }
        *occupancy.entry(w[1].regime.clone()).or_default() += 1;
    }

    let total: usize = occupancy.values().sum();
    let regime_occupancy: BTreeMap<String, f64> = occupancy
        .iter()
        .map(|(k, v)| (k.clone(), *v as f64 / total.max(1) as f64))
        .collect();

    let mut per_regime: BTreeMap<String, PerRegimeMetrics> = BTreeMap::new();
    for (regime, rets) in &per_regime_returns {
        per_regime.insert(regime.clone(), metrics_from_returns(regime, rets, bar_span_secs));
    }

    // Overall metrics: run full analytics on the equity curve.
    let equity: Vec<(i64, f64)> = series.iter().map(|p| (p.ts_nanos, p.nav)).collect();
    let m = compute_metrics(&equity, bar_span_secs);
    let overall = PerRegimeMetrics {
        regime: "overall".to_string(),
        n_periods: m.n_periods,
        total_return_pct: m.total_return_pct,
        sharpe: m.sharpe,
        sortino: m.sortino,
        max_drawdown_pct: m.max_drawdown_pct,
        win_rate_pct: m.win_rate_pct,
        profit_factor: m.profit_factor,
    };

    StrategyReport {
        strategy: strategy_name.to_string(),
        per_regime,
        overall,
        regime_occupancy,
    }
}

fn metrics_from_returns(regime: &str, rets: &[f64], bar_span_secs: u32) -> PerRegimeMetrics {
    let n = rets.len();
    if n < 2 {
        return PerRegimeMetrics {
            regime: regime.to_string(),
            n_periods: n,
            ..Default::default()
        };
    }
    let mean = rets.iter().sum::<f64>() / n as f64;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    let std = var.max(0.0).sqrt();
    let ppy = crate::analytics::periods_per_year_for_bar(bar_span_secs);
    let sharpe = if std > 0.0 { mean / std * ppy.sqrt() } else { 0.0 };
    let dd: Vec<f64> = rets.iter().filter(|r| **r < 0.0).copied().collect();
    let sortino = if !dd.is_empty() {
        let dvar = dd.iter().map(|r| r * r).sum::<f64>() / dd.len() as f64;
        let dstd = dvar.sqrt();
        if dstd > 0.0 { mean / dstd * ppy.sqrt() } else { 0.0 }
    } else {
        0.0
    };
    // Reconstruct a running NAV inside this regime to compute its own max DD.
    let mut peak = 1.0_f64;
    let mut nav = 1.0_f64;
    let mut max_dd = 0.0_f64;
    for r in rets {
        nav *= 1.0 + r;
        if nav > peak { peak = nav; }
        if peak > 0.0 {
            let d = (peak - nav) / peak;
            if d > max_dd { max_dd = d; }
        }
    }
    let wins = rets.iter().filter(|r| **r > 0.0).count() as f64;
    let win_rate = wins / n as f64 * 100.0;
    let gross_win: f64 = rets.iter().filter(|r| **r > 0.0).sum();
    let gross_loss: f64 = rets.iter().filter(|r| **r < 0.0).map(|r| -r).sum();
    let pf = if gross_loss > 0.0 {
        gross_win / gross_loss
    } else if gross_win > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };
    PerRegimeMetrics {
        regime: regime.to_string(),
        n_periods: n,
        total_return_pct: (nav - 1.0) * 100.0,
        sharpe,
        sortino,
        max_drawdown_pct: max_dd * 100.0,
        win_rate_pct: win_rate,
        profit_factor: pf,
    }
}

/// Pick the best strategy per regime by Sharpe ratio.
pub fn recommend(reports: &[StrategyReport]) -> Vec<Recommendation> {
    // Collect all regimes mentioned across all strategies.
    let mut regimes: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for r in reports {
        for k in r.per_regime.keys() {
            regimes.insert(k.clone());
        }
    }
    let mut recs = Vec::with_capacity(regimes.len());
    for regime in regimes {
        let mut by_sharpe: Vec<(&str, f64)> = reports
            .iter()
            .filter_map(|r| r.per_regime.get(&regime).map(|m| (r.strategy.as_str(), m.sharpe)))
            .collect();
        by_sharpe.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if by_sharpe.is_empty() {
            continue;
        }
        let best = by_sharpe[0];
        let runner_up = by_sharpe.get(1).map(|x| x.1);
        recs.push(Recommendation {
            regime,
            best_strategy: best.0.to_string(),
            best_sharpe: best.1,
            runner_up_sharpe: runner_up,
        });
    }
    recs
}

/// Format a multi-strategy report as a plain-text table.
pub fn format_report(reports: &[StrategyReport], recs: &[Recommendation]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(out, "═══════════════════════════════════════════════════════════════").unwrap();
    writeln!(out, "         REGIME-CONDITIONAL STRATEGY ANALYSIS").unwrap();
    writeln!(out, "═══════════════════════════════════════════════════════════════").unwrap();
    for rep in reports {
        writeln!(out, "\n▸ strategy: {}", rep.strategy).unwrap();
        writeln!(
            out,
            "  overall: ret={:.2}% sharpe={:.3} sortino={:.3} maxDD={:.2}% win%={:.1}",
            rep.overall.total_return_pct,
            rep.overall.sharpe,
            rep.overall.sortino,
            rep.overall.max_drawdown_pct,
            rep.overall.win_rate_pct,
        )
        .unwrap();
        for (regime, occ) in &rep.regime_occupancy {
            let m = rep.per_regime.get(regime);
            if let Some(m) = m {
                writeln!(
                    out,
                    "    [{:>10}] occ={:.1}% n={:>5}  ret={:>7.2}%  sharpe={:>6.3}  maxDD={:>5.2}%  win%={:>4.1}  pf={:>5.2}",
                    regime,
                    occ * 100.0,
                    m.n_periods,
                    m.total_return_pct,
                    m.sharpe,
                    m.max_drawdown_pct,
                    m.win_rate_pct,
                    m.profit_factor,
                )
                .unwrap();
            }
        }
    }
    writeln!(out, "\n═══════════════════════════════════════════════════════════════").unwrap();
    writeln!(out, "         RECOMMENDED STRATEGY PER REGIME (by Sharpe)").unwrap();
    writeln!(out, "═══════════════════════════════════════════════════════════════").unwrap();
    for r in recs {
        match r.runner_up_sharpe {
            Some(s) => writeln!(
                out,
                "  [{:>10}] → {:<22}  sharpe={:>6.3}  (next best: {:>6.3})",
                r.regime, r.best_strategy, r.best_sharpe, s
            )
            .unwrap(),
            None => writeln!(
                out,
                "  [{:>10}] → {:<22}  sharpe={:>6.3}",
                r.regime, r.best_strategy, r.best_sharpe
            )
            .unwrap(),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_equity_for_regime(seed: u64, n: usize, drift: f64, vol: f64, regime: &str) -> Vec<LabeledEquityPoint> {
        let mut s = if seed == 0 { 1u64 } else { seed };
        let mut g = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u1 = ((s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u2 = ((s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let mut out = Vec::with_capacity(n);
        let mut nav = 100.0;
        for i in 0..n {
            out.push(LabeledEquityPoint {
                ts_nanos: i as i64 * 60_000_000_000,
                nav,
                regime: regime.to_string(),
            });
            nav *= 1.0 + drift + vol * g();
        }
        out
    }

    #[test]
    fn analyze_handles_single_regime() {
        let series = synth_equity_for_regime(7, 200, 0.0005, 0.005, "bull");
        let rep = analyze_strategy("xs_momentum", &series, 60);
        assert!(rep.per_regime.contains_key("bull"));
        assert_eq!(rep.regime_occupancy["bull"], 1.0);
    }

    /// Build an equity curve with one regime per half: first half labeled
    /// `regime_a`, second half labeled `regime_b`, with the strategy
    /// performing differently in each half.
    fn synth_two_regime_curve(
        a_drift: f64, a_vol: f64,
        b_drift: f64, b_vol: f64,
        seed: u64,
    ) -> Vec<LabeledEquityPoint> {
        let mut a = synth_equity_for_regime(seed, 200, a_drift, a_vol, "bull");
        // Continue from a's terminal nav into the second regime.
        let mut s = if seed == 0 { 1u64 } else { seed.wrapping_add(99) };
        let mut g = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u1 = ((s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u2 = ((s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let mut nav = a.last().unwrap().nav;
        let last_ts = a.last().unwrap().ts_nanos;
        for i in 1..=200 {
            nav *= 1.0 + b_drift + b_vol * g();
            a.push(LabeledEquityPoint {
                ts_nanos: last_ts + (i as i64) * 60_000_000_000,
                nav,
                regime: "bear".to_string(),
            });
        }
        a
    }

    #[test]
    fn recommend_picks_different_winners_per_regime() {
        // Build a non-trivial scenario:
        //   Strategy A: very profitable in BULL (+drift), terrible in BEAR (-drift)
        //   Strategy B: bad in BULL (small +drift), excellent in BEAR (+drift)
        // The recommender should pick A for bull and B for bear.
        let a_curve = synth_two_regime_curve(0.003, 0.003, -0.002, 0.005, 7);
        let b_curve = synth_two_regime_curve(0.0001, 0.003, 0.003, 0.003, 11);
        let rep_a = analyze_strategy("A", &a_curve, 60);
        let rep_b = analyze_strategy("B", &b_curve, 60);
        let recs = recommend(&[rep_a, rep_b]);
        // We get one recommendation per regime — should be two.
        assert_eq!(recs.len(), 2);
        let bull = recs.iter().find(|r| r.regime == "bull").expect("bull rec");
        let bear = recs.iter().find(|r| r.regime == "bear").expect("bear rec");
        assert_eq!(bull.best_strategy, "A", "A should win in bull: {:?}", recs);
        assert_eq!(bear.best_strategy, "B", "B should win in bear: {:?}", recs);
    }

    #[test]
    fn recommend_is_input_order_invariant() {
        // The recommendation must not depend on the order reports are passed in.
        let a_curve = synth_two_regime_curve(0.003, 0.003, -0.002, 0.005, 7);
        let b_curve = synth_two_regime_curve(0.0001, 0.003, 0.003, 0.003, 11);
        let c_curve = synth_two_regime_curve(0.0002, 0.005, 0.0002, 0.005, 13);
        let rep_a = analyze_strategy("A", &a_curve, 60);
        let rep_b = analyze_strategy("B", &b_curve, 60);
        let rep_c = analyze_strategy("C", &c_curve, 60);
        let recs_abc = recommend(&[rep_a.clone(), rep_b.clone(), rep_c.clone()]);
        let recs_cab = recommend(&[rep_c.clone(), rep_a.clone(), rep_b.clone()]);
        let recs_bca = recommend(&[rep_b, rep_c, rep_a]);
        let pick = |rs: &[Recommendation]| -> Vec<(String, String)> {
            let mut v: Vec<_> = rs.iter()
                .map(|r| (r.regime.clone(), r.best_strategy.clone()))
                .collect();
            v.sort();
            v
        };
        assert_eq!(pick(&recs_abc), pick(&recs_cab));
        assert_eq!(pick(&recs_abc), pick(&recs_bca));
    }

    #[test]
    fn analyze_sharpe_matches_hand_calculation() {
        // 11 equity points → 10 step-returns of 1% each (deterministic).
        // mean = 0.01, std = 0 → Sharpe formula returns 0 (division-by-zero guard).
        // Use a varied series instead: alternate +1% and -0.5%.
        let mut nav = 100.0_f64;
        let mut series = vec![LabeledEquityPoint {
            ts_nanos: 0, nav, regime: "bull".to_string(),
        }];
        let returns = [0.01_f64, -0.005, 0.012, -0.003, 0.008, -0.002, 0.015, -0.004, 0.006, -0.001];
        for (i, r) in returns.iter().enumerate() {
            nav *= 1.0 + r;
            series.push(LabeledEquityPoint {
                ts_nanos: (i as i64 + 1) * 60_000_000_000,
                nav,
                regime: "bull".to_string(),
            });
        }
        // Hand-calculate Sharpe for this returns series at bar_secs=60:
        //   mean = sum/N, var = sum((r-mean)^2)/(N-1), std = sqrt(var)
        //   ppy = (252*6.5*3600)/60 = 98_280 (minute-bar periods per year)
        //   sharpe = (mean/std) * sqrt(ppy)
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = var.sqrt();
        let ppy: f64 = (252.0 * 6.5 * 3600.0) / 60.0;
        let expected_sharpe = mean / std * ppy.sqrt();
        let rep = analyze_strategy("X", &series, 60);
        let actual_sharpe = rep.per_regime["bull"].sharpe;
        assert!(
            (actual_sharpe - expected_sharpe).abs() < 1e-6,
            "sharpe mismatch: expected {} got {}",
            expected_sharpe, actual_sharpe
        );
    }
}
