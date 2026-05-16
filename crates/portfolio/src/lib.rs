//! Portfolio allocator: vol targeting + half-Kelly per strategy.
//!
//! The allocator takes per-strategy notional weights (gross, ungated) and
//! scales them so:
//!  1. Each strategy's allocation ≤ half-Kelly fraction inferred from its
//!     rolling Sharpe (capped at 50%).
//!  2. Aggregate gross ≤ `max_gross_leverage`.
//!  3. Aggregate annualized portfolio vol ≈ `vol_target` (EWMA-estimated).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortfolioConfig {
    /// Target annualized vol, e.g. 0.10 = 10%.
    pub vol_target: f64,
    /// Half-Kelly cap on any single strategy's NAV fraction.
    pub kelly_cap: f64,
    pub max_gross_leverage: f64,
    pub max_net_leverage: f64,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            vol_target: 0.10,
            kelly_cap: 0.50,
            max_gross_leverage: 2.0,
            max_net_leverage: 1.0,
        }
    }
}

/// Per-strategy stats the allocator uses.
#[derive(Copy, Clone, Debug, Default)]
pub struct StrategyStats {
    /// Annualized Sharpe ratio (rolling, e.g. 60-day).
    pub sharpe: f64,
    /// Annualized vol estimate.
    pub vol: f64,
}

impl StrategyStats {
    /// Half-Kelly fraction: f* = mean/var; we approximate as Sharpe/vol.
    /// Capped at `cap` and floored at 0 (no shorting the allocation).
    pub fn half_kelly_fraction(&self, cap: f64) -> f64 {
        if self.vol <= 0.0 || !self.sharpe.is_finite() {
            return 0.0;
        }
        let k = (self.sharpe / self.vol) * 0.5;
        k.clamp(0.0, cap)
    }
}

pub struct Allocator {
    cfg: PortfolioConfig,
}

impl Allocator {
    pub fn new(cfg: PortfolioConfig) -> Self {
        Self { cfg }
    }

    /// Given per-strategy *desired* gross weights (sum may exceed 1.0), per-strategy
    /// stats, and a portfolio vol estimate, return per-strategy final weights.
    ///
    /// Weights are NAV fractions (positive for long, negative for short).
    pub fn allocate(
        &self,
        desired: &HashMap<String, f64>,
        stats: &HashMap<String, StrategyStats>,
        portfolio_vol_est: f64,
    ) -> HashMap<String, f64> {
        // 1) Cap each strategy at half-Kelly.
        let mut capped: HashMap<String, f64> = desired
            .iter()
            .map(|(k, &w)| {
                let s = stats.get(k).copied().unwrap_or_default();
                let cap = s.half_kelly_fraction(self.cfg.kelly_cap);
                let sign = if w >= 0.0 { 1.0 } else { -1.0 };
                let mag = w.abs().min(cap);
                (k.clone(), sign * mag)
            })
            .collect();
        // 2) Enforce gross/net leverage.
        let gross: f64 = capped.values().map(|v| v.abs()).sum();
        if gross > self.cfg.max_gross_leverage && gross > 0.0 {
            let scale = self.cfg.max_gross_leverage / gross;
            for v in capped.values_mut() {
                *v *= scale;
            }
        }
        let net: f64 = capped.values().sum();
        if net.abs() > self.cfg.max_net_leverage && net.abs() > 0.0 {
            // Shift uniformly to bring net into range.
            let excess = net.signum() * (net.abs() - self.cfg.max_net_leverage);
            let n = capped.len().max(1) as f64;
            let shift = excess / n;
            for v in capped.values_mut() {
                *v -= shift;
            }
        }
        // 3) Vol target: scale all weights to bring portfolio vol toward target.
        if portfolio_vol_est > 0.0 && portfolio_vol_est.is_finite() {
            let scale = (self.cfg.vol_target / portfolio_vol_est).min(1.0); // never lever UP just to chase target
            for v in capped.values_mut() {
                *v *= scale;
            }
        }
        capped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kelly_caps_a_too_eager_strategy() {
        let a = Allocator::new(PortfolioConfig::default());
        let mut desired = HashMap::new();
        desired.insert("xs_momo".to_string(), 1.0); // wants 100% of NAV
        let mut stats = HashMap::new();
        stats.insert(
            "xs_momo".to_string(),
            StrategyStats { sharpe: 0.5, vol: 0.20 },
        );
        // half-Kelly = (0.5/0.20)/2 = 1.25, capped at 0.50
        let out = a.allocate(&desired, &stats, 0.10);
        let w = out["xs_momo"];
        assert!(w <= 0.50 + 1e-9 && w > 0.0);
    }

    #[test]
    fn vol_target_scales_down_when_realized_high() {
        let a = Allocator::new(PortfolioConfig::default());
        let mut desired = HashMap::new();
        desired.insert("xs_momo".to_string(), 0.30);
        let mut stats = HashMap::new();
        stats.insert(
            "xs_momo".to_string(),
            StrategyStats { sharpe: 1.0, vol: 0.15 },
        );
        // portfolio_vol_est = 0.20, target 0.10 → scale 0.5
        let out = a.allocate(&desired, &stats, 0.20);
        let w = out["xs_momo"];
        assert!(w > 0.0 && w < 0.30);
    }

    #[test]
    fn never_levers_up_to_chase_vol_target() {
        let a = Allocator::new(PortfolioConfig::default());
        let mut desired = HashMap::new();
        desired.insert("xs_momo".to_string(), 0.10);
        let mut stats = HashMap::new();
        stats.insert(
            "xs_momo".to_string(),
            StrategyStats { sharpe: 1.0, vol: 0.10 },
        );
        // realized vol 0.05, target 0.10 → would lever 2x; should NOT.
        let out = a.allocate(&desired, &stats, 0.05);
        assert!(out["xs_momo"] <= 0.10 + 1e-9);
    }
}
