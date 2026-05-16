//! Performance analytics.
//!
//! All metrics work on an equity curve `&[(i64 ts_nanos, f64 nav)]`. The
//! engine emits this from a single backtest run; walk-forward stitches
//! multiple equity segments together.

use serde::{Deserialize, Serialize};

/// Trading periods per year. Used to annualize Sharpe / Sortino.
/// For minute bars × ~390 minutes × 252 days ≈ 98,280. For daily bars: 252.
/// Pass the appropriate value based on the bar span.
pub fn periods_per_year_for_bar(span_secs: u32) -> f64 {
    let secs_per_year = 252.0 * 6.5 * 60.0 * 60.0;
    secs_per_year / span_secs.max(1) as f64
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub n_periods: usize,
    pub total_return_pct: f64,
    pub cagr_pct: f64,
    pub annualized_vol_pct: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub max_drawdown_pct: f64,
    pub calmar: f64,
    pub win_rate_pct: f64,
    pub profit_factor: f64,
    pub avg_return_bp: f64,
    pub best_period_pct: f64,
    pub worst_period_pct: f64,
    pub ulcer_index: f64,
}

/// Compute all metrics from an equity curve and bar span (in seconds).
pub fn compute(equity: &[(i64, f64)], bar_span_secs: u32) -> Metrics {
    let mut m = Metrics::default();
    if equity.len() < 2 {
        return m;
    }
    let n = equity.len();
    m.n_periods = n;

    let v0 = equity[0].1;
    let vn = equity[n - 1].1;
    m.total_return_pct = (vn / v0 - 1.0) * 100.0;

    // Period returns
    let mut rets = Vec::with_capacity(n - 1);
    for w in equity.windows(2) {
        let r = if w[0].1 > 0.0 { w[1].1 / w[0].1 - 1.0 } else { 0.0 };
        if r.is_finite() {
            rets.push(r);
        }
    }
    if rets.is_empty() {
        return m;
    }

    let mean = rets.iter().sum::<f64>() / rets.len() as f64;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (rets.len() - 1).max(1) as f64;
    let std = var.max(0.0).sqrt();

    let ppy = periods_per_year_for_bar(bar_span_secs);

    m.annualized_vol_pct = std * ppy.sqrt() * 100.0;
    m.sharpe = if std > 0.0 { mean / std * ppy.sqrt() } else { 0.0 };

    // Sortino: downside deviation only
    let downside: Vec<f64> = rets.iter().filter(|r| **r < 0.0).copied().collect();
    if !downside.is_empty() {
        let dd_var = downside.iter().map(|r| r * r).sum::<f64>() / downside.len() as f64;
        let dd_std = dd_var.sqrt();
        m.sortino = if dd_std > 0.0 { mean / dd_std * ppy.sqrt() } else { 0.0 };
    }

    // Drawdown sweep
    let mut peak = equity[0].1;
    let mut max_dd = 0.0;
    let mut ulcer_sq_sum = 0.0;
    for &(_, v) in equity {
        if v > peak {
            peak = v;
        }
        if peak > 0.0 {
            let dd = (peak - v) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
            ulcer_sq_sum += dd * dd;
        }
    }
    m.max_drawdown_pct = max_dd * 100.0;
    m.ulcer_index = (ulcer_sq_sum / n as f64).sqrt() * 100.0;

    // CAGR — approximate using the elapsed time of the equity curve
    let elapsed_secs = (equity[n - 1].0 - equity[0].0).max(1) as f64 / 1e9;
    let years = elapsed_secs / (252.0 * 6.5 * 3600.0); // active years
    if years > 0.0 && v0 > 0.0 {
        m.cagr_pct = ((vn / v0).powf(1.0 / years) - 1.0) * 100.0;
    }
    if m.max_drawdown_pct > 0.0 {
        m.calmar = m.cagr_pct / m.max_drawdown_pct;
    }

    let wins = rets.iter().filter(|r| **r > 0.0).count();
    m.win_rate_pct = wins as f64 / rets.len() as f64 * 100.0;
    let gross_win: f64 = rets.iter().filter(|r| **r > 0.0).sum();
    let gross_loss: f64 = rets.iter().filter(|r| **r < 0.0).map(|r| -r).sum();
    m.profit_factor = if gross_loss > 0.0 {
        gross_win / gross_loss
    } else if gross_win > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };
    m.avg_return_bp = mean * 10_000.0;
    m.best_period_pct = rets.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 100.0;
    m.worst_period_pct = rets.iter().cloned().fold(f64::INFINITY, f64::min) * 100.0;

    m
}

impl Metrics {
    pub fn pretty(&self) -> String {
        format!(
            "n_periods       : {}\n\
             total_return    : {:.3}%\n\
             cagr            : {:.3}%\n\
             ann. vol        : {:.3}%\n\
             sharpe (annl)   : {:.3}\n\
             sortino (annl)  : {:.3}\n\
             max drawdown    : {:.3}%\n\
             calmar          : {:.3}\n\
             win rate        : {:.2}%\n\
             profit factor   : {:.3}\n\
             avg return / bar: {:.3} bp\n\
             best period     : {:.3}%\n\
             worst period    : {:.3}%\n\
             ulcer index     : {:.3}",
            self.n_periods,
            self.total_return_pct,
            self.cagr_pct,
            self.annualized_vol_pct,
            self.sharpe,
            self.sortino,
            self.max_drawdown_pct,
            self.calmar,
            self.win_rate_pct,
            self.profit_factor,
            self.avg_return_bp,
            self.best_period_pct,
            self.worst_period_pct,
            self.ulcer_index
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_curve_has_zero_metrics() {
        let eq: Vec<(i64, f64)> = (0..100).map(|i| (i * 60_000_000_000, 100.0)).collect();
        let m = compute(&eq, 60);
        assert_eq!(m.total_return_pct, 0.0);
        assert_eq!(m.max_drawdown_pct, 0.0);
        assert_eq!(m.sharpe, 0.0);
    }

    #[test]
    fn straight_up_has_no_drawdown() {
        let eq: Vec<(i64, f64)> = (0..100).map(|i| (i * 60_000_000_000, 100.0 * (1.001f64).powi(i as i32))).collect();
        let m = compute(&eq, 60);
        assert!(m.total_return_pct > 0.0);
        assert!(m.max_drawdown_pct < 1e-9);
        assert!(m.sharpe > 0.0);
        assert!(m.profit_factor.is_infinite() || m.profit_factor > 100.0);
    }

    #[test]
    fn drawdown_detected() {
        // Goes up, then dips, then recovers
        let eq = vec![
            (0_i64, 100.0),
            (60_000_000_000, 110.0),
            (120_000_000_000, 90.0),
            (180_000_000_000, 105.0),
        ];
        let m = compute(&eq, 60);
        // Peak 110 → trough 90 → DD = 18.18%
        assert!((m.max_drawdown_pct - 18.181818).abs() < 0.01);
    }

    #[test]
    fn sortino_is_finite_and_reasonable() {
        // Mixed: many small positives, a few small negatives.
        let mut eq = vec![(0_i64, 100.0)];
        let mut nav = 100.0;
        for i in 1..100 {
            // alternating: +0.5%, then small -0.2%; expectancy positive
            let factor = if i % 3 == 0 { 0.998 } else { 1.005 };
            nav *= factor;
            eq.push((i * 60_000_000_000, nav));
        }
        let m = compute(&eq, 60);
        assert!(m.sortino.is_finite());
        assert!(m.sharpe.is_finite());
        // Both should be positive in this expectancy-positive series.
        assert!(m.sharpe > 0.0);
        assert!(m.sortino > 0.0);
    }

    #[test]
    fn metrics_handle_empty_curve_gracefully() {
        let m = compute(&[], 60);
        assert_eq!(m.n_periods, 0);
        assert_eq!(m.sharpe, 0.0);
    }

    #[test]
    fn profit_factor_is_infinity_when_no_losses() {
        let mut eq = vec![(0_i64, 100.0)];
        let mut nav = 100.0;
        for i in 1..50 {
            nav *= 1.001;
            eq.push((i * 60_000_000_000, nav));
        }
        let m = compute(&eq, 60);
        assert!(m.profit_factor.is_infinite() || m.profit_factor > 1e6);
    }
}
