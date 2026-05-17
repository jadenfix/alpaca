# Research Data Platform

A deep, free, multi-source time-series research library for the engine.
Every fetcher is a single self-contained Python script under `scripts/data/`.
Everything funnels into a unified directory layout, a SQLite catalog, and
a regime-conditional analytics path through the Rust backtester.

> Everything below is free (no API key) unless explicitly noted.

## Quick start

```bash
# One-shot: fetch everything, validate, aggregate, catalog
scripts/data/pipeline.sh

# Or selectively
YEARS=5 SKIP=binance,gdelt scripts/data/pipeline.sh

# Browse the catalog
sqlite3 data/catalog.sqlite '
  SELECT source, asset_class, COUNT(*) AS n, SUM(n_rows) AS rows
  FROM datasets GROUP BY source, asset_class ORDER BY rows DESC'

# Compute 25-feature dataframe for any series
python3 scripts/data/signal_extract.py data/macro/VIXCLS.csv

# Regime-conditional strategy bench on real data
cargo run --release --bin algo-backtest -- bench-strategies \
  --path data/real/universe.csv --regime-proxy SPY --bar-secs 86400
```

## Data sources (all free, no API key unless noted)

| Source | Asset class | Frequency | Coverage | Script |
|---|---|---|---|---|
| **Yahoo Finance** | equity ETFs / stocks | daily | full history | `fetch_yahoo.py` |
| **FRED** | macro (12 series default) | daily–quarterly | back to 1950s | `fetch_fred.py` |
| **Fama-French (Dartmouth)** | factor returns (3+mom + 5-factor) | daily | back to 1926 | `fetch_famafrench.py` |
| **CoinGecko** | crypto OHLCV | daily | ~365 days | `fetch_coingecko.py` |
| **Binance public** | crypto OHLCV | 1m / 5m / 15m / 1h / 1d | ~4000 bars/req | `fetch_binance.py` |
| **Wikipedia** | constituent lists (SP500/NDX/DJI) | snapshot | live page | `fetch_wikipedia.py` |
| **SEC EDGAR** | 10-K/10-Q filing metadata | as-filed | full company history | `fetch_sec_edgar.py` |
| **CFTC COT** | futures positioning | weekly | back to 1995 | `fetch_cftc_cot.py` |
| **EIA** | energy (WTI, Brent, nat gas, crude stocks) | daily–weekly | back to 1986 | `fetch_eia.py` |
| **US Treasury** | full yield curve (13 tenors) | daily | back to 1990 | `fetch_treasury_yields.py` |
| **NOAA GSOD** | weather (temp, precip, wind) | daily | back to 1929 | `fetch_noaa.py` |
| **World Bank** | country macro (GDP, CPI, unemp, etc.) | annual | back to 1960 | `fetch_worldbank.py` |
| **Wikipedia pageviews** | search-interest proxies | daily | back to 2015 | `fetch_wiki_pageviews.py` |
| **AAII Investor Sentiment** | bull/bear/neutral spread | weekly | back to 1987 | `fetch_aaii.py` (best-effort) |
| **GDELT** | global news event tone | daily (15-min) | 2015+ | `fetch_gdelt.py` |

### What this covers

- **6 asset classes**: equity, crypto, rates, FX (via FRED), commodities (via EIA / FF), macro
- **4 data types**: prices/OHLCV, indicators (single-value time series), positioning, alternative data (search, news, weather, sentiment, filings)
- **Time horizons**: minutes (Binance) → years (World Bank, Damodaran)
- **Geographic**: US, Europe, China, Japan, Germany, UK (extendable)

### What we deliberately skipped

- **News articles full-text** (legal/licensing minefield; GDELT gives us tone aggregates instead)
- **Real-time tick data** (none of the truly-free sources offer this; use Alpaca live for that)
- **Earnings transcripts** (paywalled or scraped from Seeking Alpha — legal grey area)
- **Anything that requires a paid API key**

## Directory layout

```
data/
├── real/        # equity ETF bars (Yahoo)
├── crypto/      # daily crypto bars (CoinGecko)
├── binance/     # high-res crypto bars (1m / 5m / 1h)
├── macro/       # FRED single-value series
├── factors/     # Fama-French + momentum factor returns
├── yields/      # US Treasury full yield curve
├── energy/      # EIA crude / nat-gas / stocks
├── weather/     # NOAA GSOD daily climate
├── worldbank/   # country / indicator combos
├── universe/    # constituent lists (SP500, NDX, DJI)
├── filings/     # SEC 10-K/10-Q metadata
├── cot/         # CFTC weekly positioning
├── pageviews/   # Wikipedia daily pageviews
├── sentiment/   # AAII weekly survey
├── gdelt/       # global news event tone
├── sample/      # synthetic data (GBM, momentum, cointegrated pair)
├── features/    # signal_extract.py output
├── manifests/   # per-fetcher run manifests + inventory snapshots
└── catalog.sqlite  # unified searchable index of every dataset
```

## Schemas

All bar CSVs use the canonical schema the Rust engine expects:

```
ts_nanos, symbol, open, high, low, close, volume, span_secs
```

Single-value series (`macro/`, `energy/`, `pageviews/`, `factors/`):

```
ts_nanos, series, value          # or `factor`, `article`, etc.
```

Yield curve (special case — wide-format tenors):

```
ts_nanos, tenor, yield_pct
```

Positioning (CFTC COT):

```
ts_nanos, contract, mm_long, mm_short, prod_long, prod_short, mm_net, prod_net
```

Constituent lists:

```
symbol, name, sector, weight
```

## The agentic validator

`scripts/data/validate.py` walks every bar CSV and applies eight quantitative
checks:

| Check | Action |
|---|---|
| Schema | Reject row if column count / header mismatch |
| Monotonic timestamps | Count violations |
| Continuity (gap detection) | Flag gaps > `median_gap × gap_threshold` |
| Positivity (price > 0, volume ≥ 0) | Drop row |
| OHLC sanity (low ≤ {open, close} ≤ high) | Drop row |
| Outlier returns (\|z\| > `outlier_z` over rolling window) | Flag |
| Candidate splits (close-to-close ratio > `split_ratio`) | Flag |
| Duplicate (symbol, ts) | Drop |

Output: a JSON manifest at `data/manifests/validate_<unix-ts>.json`. With
`--autofix`, cleaned data is written to a sibling `.cleaned.csv` (the
source is never modified).

**Empirically validated**: when run on real Yahoo Finance ETF data, the
validator caught real 5σ+ moves in QQQ (+11.3% in one day), HYG (-7.0σ),
GLD (-6.1σ). Those weren't bad data — they were real macro events worth
flagging for follow-up.

## The catalog

`scripts/data/catalog.py` builds `data/catalog.sqlite` with a single
`datasets` table indexed by source and asset class:

```sql
SELECT source, asset_class, COUNT(*) AS n, SUM(n_rows) AS rows
FROM datasets GROUP BY source, asset_class ORDER BY rows DESC;
```

Every entry tracks: `dataset_id`, `source`, `asset_class`, `symbol`, `path`,
`n_rows`, `first_ts_nanos`, `last_ts_nanos`, `frequency`, `license_url`,
`fetched_at`. The catalog is regenerated deterministically from the on-disk
CSVs (no source of truth in the DB).

### Sample inventory (after a minimal pipeline run)

```
SOURCE        ASSET_CLASS        N_DATASETS   TOTAL_ROWS
--------------------------------------------------------
famafrench    factor                      3       225,711
fred          macro                      12        73,642
wikimedia     alt_search                 12         8,760
worldbank     macro                      36         1,875
yahoo         equity_etf                 12         6,012
synthetic     synthetic                   7       350,000
```

That's already ~665k observations across ~80 datasets, from 5 minutes of
pipeline run time. Add `binance`, `noaa`, `eia` (with key), and `gdelt`
for several more million rows.

## The signal extractor

`scripts/data/signal_extract.py` turns any time series into a tidy long-format
feature dataframe:

```
ts_nanos, series, feature, value
```

For each series, **25 features** are emitted across 4 rolling windows
(5, 21, 63, 252):

- `level`, `log`, `diff`, `log_return`, `drawdown`
- `ma_W`, `std_W`, `zscore_W`, `momentum_W`, `rv_ann_W`, `skew_W`, `kurt_W`
  for W ∈ {5, 21, 63, 252}

Example: running on `data/macro/VIXCLS.csv` (12k daily obs since 1990)
produced **300,773 feature rows** of size-25 dataframe — enough for any
downstream regression / regime model.

## Research workflows

### A) Macro regime classification

```bash
# Build a Markov-switching model over VIX / yield-curve slope / unemployment
python3 scripts/data/signal_extract.py data/macro/VIXCLS.csv data/macro/T10Y2Y.csv data/macro/UNRATE.csv

# Then in Rust:
cargo run --release --bin algo-backtest -- bench-strategies \
  --path data/real/universe.csv --regime-proxy SPY --bar-secs 86400
```

### B) Cross-asset correlation

```bash
# Joint a macro series with equity returns and look for lead-lag
python3 scripts/data/signal_extract.py data/factors/F-F_Research_Data_Factors_daily.csv

# Use the engine's lead-lag estimator (algo_features::lead_lag) to mine
# leader/follower pairs across the corpus.
```

### C) Sentiment-adjusted strategies

Wikipedia pageviews + GDELT tone + AAII bull-bear spread give three
independent crowd-sentiment signals. Run `signal_extract.py` on each
and join in pandas / polars:

```python
import polars as pl
vix = pl.read_csv("data/features/VIXCLS_features.csv")
spy = pl.read_csv("data/features/SPY_features.csv")
aaii = pl.read_csv("data/sentiment/aaii.csv")
# join + study contemporaneous and lead-lag relationships
```

### D) Commodity ↔ weather

```bash
# Heating-degree-days vs natural gas prices
python3 scripts/data/fetch_noaa.py 2023 2024
python3 scripts/data/fetch_eia.py
# Then compute correlations against EIA's RNGWHHD.D
```

### E) Cointegration mining

The engine has `algo_features::engle_granger` + `fit_ou`. Walk a basket of
ETFs and test every pair for cointegration; rank by ADF p-value × half-life.
Wire results into `pairs_mean_reversion` configs.

## Pipeline guarantees

- **Idempotent**: re-running overwrites downloads, appends manifests
- **Best-effort**: any single fetcher failing logs a warning, doesn't abort
- **Source-attributed**: every dataset in the catalog carries its license URL
- **Reproducible**: deterministic from upstream → catalog rebuild yields the same SQLite
- **No silent edits**: validator's autofix writes alongside; source stays pristine

## Adding a new source

1. Copy any `fetch_*.py` to `fetch_<newsrc>.py`
2. Adopt the canonical schema for the asset class (`BAR_HEADER` for OHLCV, `MACRO_HEADER` for indicators)
3. Add an entry to `DIR_META` in `catalog.py` with the license URL
4. Add a `run_step` line to `pipeline.sh`

That's it. The validator + aggregator + catalog pick it up automatically.

## Limitations

- Stooq now requires API keys for bulk download (we use Yahoo as the
  primary equity source).
- AAII's modern site gates the survey behind login; our scraper is
  best-effort and may return zero rows.
- NOAA GSOD URLs occasionally change format mid-year.
- The Treasury Direct CSV endpoint has rate limited some IPs since 2023;
  retry from a different network if it 403s.

## License attributions

The `catalog.py` `DIR_META` table records each source's license/TOS URL.
**You are responsible for complying with each upstream's terms.** Most are
permissive for research; some (Yahoo, Wikipedia) require attribution if you
redistribute the raw data. We never redistribute — `data/` is gitignored.
