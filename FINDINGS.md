# Findings — Regime-Conditional Strategy Performance

This document captures empirical results from running every strategy on real
historical data and on synthetic regime-controlled data, with a Markov-
switching model used to label every timestamp's regime so we can analyze
which strategy wins in which regime.

Reproduce:
```bash
python3 scripts/fetch_yahoo.py --years 2 --merge SPY QQQ XLK XLF XLE GLD TLT IWM
cargo run --release --bin algo-backtest -- bench-strategies \
  --path data/real/universe.csv \
  --regime-proxy SPY \
  --bar-secs 86400
```

## Methodology

1. Fit a 3-state Gaussian Markov-switching model (Hamilton 1989) on the
   regime-proxy ticker's log returns via Baum-Welch.
2. States are auto-labeled by the recommender's `relabel_regimes`:
   highest variance → **Bear**, lowest variance + positive mean → **Bull**,
   middle → **Sideways**.
3. Replay the proxy's returns through the model's online forward filter to
   assign every timestamp a regime label.
4. Run each strategy on the full event stream; attach the regime label to
   each equity-curve point.
5. Compute per-regime Sharpe / Sortino / MaxDD / win rate / profit factor.
6. Recommend the best strategy per regime by Sharpe.

## Real data: 2 years × 8 ETFs (SPY, QQQ, XLK, XLF, XLE, GLD, TLT, IWM)

Regime occupancy: **Bull 58.5%, Sideways 40.3%, Bear 1.2%**

| Regime | Best strategy | Best Sharpe | Runner-up Sharpe |
|---|---|---|---|
| Bear | `regime_hmm` | 0.566 | 0.326 (hurst_regime) |
| Bull | `garch_vol_target` | 0.722 | 0.167 (markov_router) |
| Sideways | `regime_hmm` | -0.069 | -0.081 (garch_vol_target) |

### Headline conclusions
- **In bull markets, GARCH-vol-targeted trend-following dominates** —
  a Sharpe of 0.72 with maxDD 1.13% is a meaningful edge. The strategy
  benefits from the persistent uptrend; vol-scaling keeps position size
  sensible as conditional volatility evolves.
- **In bear markets, the HMM regime detector wins** — by design, it flattens
  positions when turbulence is detected. Bear occupancy in this 2-year
  window is small (1.2%) so the sample is thin, but the direction matches
  theory.
- **Sideways markets are hard.** All strategies have near-zero or negative
  Sharpe in sideways. This is the regime where mean-reversion is supposed
  to work, but our pairs strategies need a paired-asset universe (not the
  ETF basket we used), so the test is incomplete.

### Per-strategy snapshot

```
▸ xs_momentum         overall: ret=-2.13% sharpe=-0.47 (clipped by L3 ladder)
▸ garch_vol_target    overall: ret= 4.70% sharpe= 0.24  ← bull-regime alpha
▸ hurst_regime        overall: ret= 0.03% sharpe= 0.00  ← noisy
▸ markov_router       overall: ret=-0.30% sharpe=-0.03  ← needs HF tuning
▸ regime_hmm          overall: ret= 0.52% sharpe= 0.07  ← defensive, low DD
```

## Synthetic data: momentum-factor universe (16 ETFs × 5000 bars)

Regime occupancy with SYM0 as proxy: **Bull 66.3%, Sideways 33.7%**

| Regime | Best strategy | Best Sharpe |
|---|---|---|
| Bull | `hurst_regime` | 15.247 |
| Sideways | `hurst_regime` | 7.790 |

`hurst_regime` is the standout on synthetic data because the generator
embeds a persistent factor structure that the Hurst exponent picks up
directly (per-symbol H > 0.55 ⇒ trend-follow). On real data this collapses,
because real-asset H is much closer to 0.5 over 2-year windows.

## What this changes about the playbook

1. **Default to `garch_vol_target` in bull regimes.** It's the only strategy
   that posted a positive risk-adjusted return on real data.
2. **Switch to `regime_hmm` (which goes flat) when the Markov model assigns
   ≥ 0.55 posterior to Bear.** Capital preservation > alpha hunting.
3. **Don't trust synthetic-data wins blindly.** `hurst_regime` looked
   spectacular synthetically but barely moves the needle on real ETFs.
4. **Sideways is an open problem.** The next iteration should run
   `pairs_mean_reversion` and `kalman_pairs` on cointegrated pair
   candidates (e.g., XLF/XLE, GLD/SLV, TLT/IEF) rather than the broad ETF
   universe — the current bench is structurally biased against pairs
   strategies because they need a specific symbol pair, not a basket.

## Implementation notes

- The Markov model is fitted on the regime-proxy series ONCE up-front via
  Baum-Welch (25 iterations, tol 1e-4), then replayed through the online
  forward filter to label every event. The two-pass approach ensures the
  regime labels are stable and the strategy backtests are deterministic.
- Per-regime returns are computed by partitioning the equity curve's
  step-returns by their regime label — Sharpe / Sortino / MaxDD are
  computed per slice.
- `regime_occupancy.values().sum() ≈ 1.0` is asserted in
  `tests/regime_analysis.rs`.

## Limitations + next steps

- **Real data is only 2 years of daily bars.** For HFT-style strategies
  we need intraday data (Alpaca historical bars: free up to 15-min delayed,
  paid for real-time). Wire `mdata` to Alpaca historical and rerun.
- **Bear regime is severely under-sampled in this window** (only 48 days).
  Extend to 10+ years to get meaningful bear data.
- **The recommender picks by Sharpe only.** Should also weight by regime
  occupancy and by MaxDD — a low-Sharpe-low-DD strategy may be preferable
  for capital preservation in transitional regimes.
- **Monte Carlo forward simulation** (`features::monte_carlo::simulate`)
  is built but not yet plugged into the recommender. The next step is to
  use it to estimate expected utility under each candidate strategy given
  the current regime posterior.
