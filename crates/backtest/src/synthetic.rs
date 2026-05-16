//! Synthetic data generator.
//!
//! Two flavors:
//!  - `generate_gbm_universe`: simple multi-asset GBM with per-symbol drift dispersion.
//!  - `generate_regime_universe`: multi-regime universe — momentum regime then
//!    mean-reversion regime then choppy regime. Designed to be HARD for a single
//!    strategy to beat, which is the point.
//!  - `generate_cointegrated_pair`: two series with a stationary spread, for
//!    pairs / mean-reversion strategy validation.

use algo_core::{Bar, MarketEvent, Price, Qty, Symbol, Ts};

#[derive(Clone, Debug)]
pub struct SyntheticConfig {
    pub n_symbols: usize,
    pub n_bars: usize,
    pub bar_secs: u32,
    pub mu_annual: f64,
    pub sigma_annual: f64,
    pub seed: u64,
    pub start_price: f64,
    pub dispersion: f64,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            n_symbols: 8,
            n_bars: 500,
            bar_secs: 60,
            mu_annual: 0.05,
            sigma_annual: 0.20,
            seed: 42,
            start_price: 100.0,
            dispersion: 0.10,
        }
    }
}

pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEADBEEF } else { seed },
        }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn next_uniform(&mut self) -> f64 {
        let v = self.next_u64();
        ((v >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

pub fn generate_gbm_universe(cfg: &SyntheticConfig, start_ts: Ts) -> Vec<MarketEvent> {
    let mut rng = XorShift64::new(cfg.seed);
    let dt = cfg.bar_secs as f64 / (252.0 * 6.5 * 60.0 * 60.0);
    let mut prices: Vec<f64> = (0..cfg.n_symbols).map(|_| cfg.start_price).collect();
    let symbols: Vec<Symbol> = (0..cfg.n_symbols)
        .map(|i| Symbol::new(&format!("SYM{i}")).expect("symbol"))
        .collect();
    let drifts: Vec<f64> = (0..cfg.n_symbols)
        .map(|i| {
            let centered = (i as f64 - (cfg.n_symbols as f64 - 1.0) / 2.0)
                / ((cfg.n_symbols as f64 - 1.0).max(1.0) / 2.0);
            cfg.mu_annual + cfg.dispersion * centered
        })
        .collect();
    let mut out = Vec::with_capacity(cfg.n_symbols * cfg.n_bars);
    let bar_nanos = cfg.bar_secs as i64 * 1_000_000_000;
    for bar_idx in 0..cfg.n_bars {
        let ts = Ts::from_nanos(start_ts.nanos + bar_nanos * (bar_idx as i64 + 1));
        for (i, &sym) in symbols.iter().enumerate() {
            let mu = drifts[i];
            let sigma = cfg.sigma_annual;
            let z = rng.next_normal();
            let p_prev = prices[i];
            let p_new = p_prev
                * ((mu - 0.5 * sigma * sigma) * dt + sigma * dt.sqrt() * z).exp();
            prices[i] = p_new;
            let high = p_prev.max(p_new) * (1.0 + sigma * dt.sqrt() * 0.25);
            let low = p_prev.min(p_new) * (1.0 - sigma * dt.sqrt() * 0.25);
            let bar = Bar {
                ts,
                symbol: sym,
                open: Price::from_f64(p_prev).unwrap(),
                high: Price::from_f64(high).unwrap(),
                low: Price::from_f64(low).unwrap(),
                close: Price::from_f64(p_new).unwrap(),
                volume: Qty::from_i64(10_000),
                span_secs: cfg.bar_secs,
            };
            out.push(MarketEvent::Bar(bar));
        }
    }
    out
}

/// Universe with an embedded momentum signal: half the symbols have a
/// persistent positive momentum factor (a slow-moving common factor with
/// per-symbol loadings), the other half negative. XS momentum SHOULD be
/// able to extract this. Use as a sanity check that the strategy works.
pub fn generate_momentum_universe(
    n_symbols: usize,
    n_bars: usize,
    bar_secs: u32,
    sigma_annual: f64,
    factor_strength: f64,
    seed: u64,
    start_ts: Ts,
) -> Vec<MarketEvent> {
    let mut rng = XorShift64::new(seed);
    let dt = bar_secs as f64 / (252.0 * 6.5 * 60.0 * 60.0);
    let mut prices: Vec<f64> = (0..n_symbols).map(|_| 100.0).collect();
    let symbols: Vec<Symbol> = (0..n_symbols)
        .map(|i| Symbol::new(&format!("SYM{i}")).expect("symbol"))
        .collect();
    // Loadings: +1 for first half, -1 for second half.
    let loadings: Vec<f64> = (0..n_symbols)
        .map(|i| if i < n_symbols / 2 { 1.0 } else { -1.0 })
        .collect();
    let bar_nanos = bar_secs as i64 * 1_000_000_000;
    // A persistent factor with positive drift that builds slowly.
    let mut factor = 0.0_f64;
    let mut out = Vec::with_capacity(n_symbols * n_bars);
    for bar_idx in 0..n_bars {
        // AR(1) factor with mean reversion to zero (slow). Drives momentum effect.
        factor = 0.999 * factor + 0.01 * rng.next_normal();
        let ts = Ts::from_nanos(start_ts.nanos + bar_nanos * (bar_idx as i64 + 1));
        for (i, &sym) in symbols.iter().enumerate() {
            let idio = sigma_annual * dt.sqrt() * rng.next_normal();
            let factor_ret = loadings[i] * factor_strength * factor * dt;
            let log_ret = factor_ret + idio;
            let p_prev = prices[i];
            let p_new = p_prev * log_ret.exp();
            prices[i] = p_new;
            let bar = Bar {
                ts,
                symbol: sym,
                open: Price::from_f64(p_prev).unwrap(),
                high: Price::from_f64(p_prev.max(p_new) * 1.001).unwrap(),
                low: Price::from_f64(p_prev.min(p_new) * 0.999).unwrap(),
                close: Price::from_f64(p_new).unwrap(),
                volume: Qty::from_i64(10_000),
                span_secs: bar_secs,
            };
            out.push(MarketEvent::Bar(bar));
        }
    }
    out
}

/// Cointegrated pair: two prices that share a common stochastic trend with a
/// stationary spread (Ornstein-Uhlenbeck). Used to validate pairs strategy.
pub fn generate_cointegrated_pair(
    sym_a: Symbol,
    sym_b: Symbol,
    n_bars: usize,
    bar_secs: u32,
    seed: u64,
    start_ts: Ts,
) -> Vec<MarketEvent> {
    let mut rng = XorShift64::new(seed);
    let dt = bar_secs as f64 / (252.0 * 6.5 * 60.0 * 60.0);
    let theta = 5.0; // mean reversion speed (per year)
    let sigma_spread = 0.02; // spread vol
    let mut spread = 0.0_f64;
    let mut log_common = 0.0_f64;
    let bar_nanos = bar_secs as i64 * 1_000_000_000;
    let mut out = Vec::with_capacity(n_bars * 2);
    for bar_idx in 0..n_bars {
        let ts = Ts::from_nanos(start_ts.nanos + bar_nanos * (bar_idx as i64 + 1));
        // Common log price drifts as a random walk
        log_common += 0.10 * dt + 0.20 * dt.sqrt() * rng.next_normal();
        // Spread is OU: ds = -theta * s * dt + sigma * dW
        spread += -theta * spread * dt + sigma_spread * dt.sqrt() * rng.next_normal();
        let pa = 100.0 * (log_common + 0.5 * spread).exp();
        let pb = 100.0 * (log_common - 0.5 * spread).exp();
        for (sym, p) in [(sym_a, pa), (sym_b, pb)] {
            out.push(MarketEvent::Bar(Bar {
                ts,
                symbol: sym,
                open: Price::from_f64(p).unwrap(),
                high: Price::from_f64(p * 1.0005).unwrap(),
                low: Price::from_f64(p * 0.9995).unwrap(),
                close: Price::from_f64(p).unwrap(),
                volume: Qty::from_i64(10_000),
                span_secs: bar_secs,
            }));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_expected_count_and_chronology() {
        let cfg = SyntheticConfig {
            n_symbols: 3,
            n_bars: 10,
            ..Default::default()
        };
        let evs = generate_gbm_universe(&cfg, Ts::from_nanos(0));
        assert_eq!(evs.len(), 30);
        for w in evs.windows(2) {
            assert!(w[0].ts() <= w[1].ts());
        }
    }

    #[test]
    fn momentum_universe_has_dispersion() {
        let evs = generate_momentum_universe(8, 500, 60, 0.10, 50.0, 7, Ts::from_nanos(0));
        // Should produce 8 * 500 events
        assert_eq!(evs.len(), 4_000);
        // Final prices for first-half vs second-half should differ in average direction
        // when factor_strength is large enough; just sanity check non-degenerate prices.
        let mut last_prices = std::collections::HashMap::new();
        for e in &evs {
            if let MarketEvent::Bar(b) = e {
                last_prices.insert(b.symbol, b.close.to_f64());
            }
        }
        assert_eq!(last_prices.len(), 8);
        assert!(last_prices.values().all(|p| *p > 0.0));
    }

    #[test]
    fn cointegrated_pair_spread_is_bounded() {
        let a = Symbol::new("AAA").unwrap();
        let b = Symbol::new("BBB").unwrap();
        let evs = generate_cointegrated_pair(a, b, 1_000, 60, 11, Ts::from_nanos(0));
        // Pairs should track within +/- a few percent because of OU spread
        let mut last_a = 0.0;
        let mut last_b = 0.0;
        for e in &evs {
            if let MarketEvent::Bar(bar) = e {
                if bar.symbol == a {
                    last_a = bar.close.to_f64();
                }
                if bar.symbol == b {
                    last_b = bar.close.to_f64();
                }
            }
        }
        let ratio = last_a / last_b;
        assert!(ratio > 0.5 && ratio < 2.0, "pair drifted too far: ratio={ratio}");
    }
}
