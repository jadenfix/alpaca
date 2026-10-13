# algo — Rust low-latency trading engine on Alpaca

A Cargo workspace for systematic trading on Alpaca: CPU-first hot path,
optional GPU inference, single code path for backtest and live, multi-level
loss containment, and discipline-first strategy design.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design rationale.

The offline backtest and research tooling are the supported demonstration surface.
The live market-data transport and Alpaca REST broker are currently skeletons
that return explicit errors; live and paper trading are not operational yet.

## Quick start

```bash
# build everything
cargo build --release

# run a backtest on synthetic data
cargo run --release --bin algo-backtest -- synthetic --strategy xs_momentum --seed 42 --equity-csv equity.csv

```

## Layout

| Path | Purpose |
|---|---|
| `crates/core` | shared types: events, orders, money, Clock trait |
| `crates/features` | vectorized rolling stats, cointegration |
| `crates/strategy` | Strategy trait + scheduler |
| `crates/strategies` | individual strategy implementations |
| `crates/backtest` | event-driven simulator |
| `crates/risk` | pre-trade checks, kill switch, DD ladder |
| `crates/portfolio` | vol targeting, Kelly cap, allocator |
| `crates/mdata` | Alpaca WebSocket market data |
| `crates/broker` | Alpaca REST trading client |
| `crates/oms` | order lifecycle, position cache |
| `crates/sim` | cost / slippage model |
| `crates/obs` | HDR histograms, tracing, Prometheus |
| `crates/journal` | append-only event log + replay |
| `crates/storage` | Parquet bar storage |
| `crates/calendar` | NYSE session helpers |
| `crates/universe` | symbol selection + filters |
| `crates/gpu` | candle inference (feature = "gpu") |
| `bins/*` | live, backtest, shadow, replay, ops |

## Safety posture

- **No per-trade stop-losses by default** — replaced by signal-reversal exits,
  time-based exits, and a 7-level portfolio loss-limit ladder.
- **Multi-trigger kill switch** (DD, WS gap, reject rate, position drift,
  clock skew, runaway loop guard, file-based, dead-man's switch).
- **Live binary refuses to start** without explicit env var + flag, and rejects
  unknown endpoints.
- **Shadow-live phase** (live data, no order send) before any capital.

## Offline research smoke run

The Python research harness can run without broker credentials or market-data
network access. Install NumPy and run:

```bash
python scripts/backtest/smoke.py --out artifacts/research-smoke.json
python -m unittest discover -s scripts/data/tests -p test_walkforward.py
```

The smoke run generates seeded synthetic returns, fits strategies only on
prior bars, charges turnover costs, and exports actual equity curves and
metrics for four baselines. Synthetic results are engineering checks, not
investment-performance evidence.
