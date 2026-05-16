# algo — Rust low-latency trading engine on Alpaca

A Cargo workspace for systematic trading on Alpaca: CPU-first hot path,
optional GPU inference, single code path for backtest and live, multi-level
loss containment, and discipline-first strategy design.

See [plan file](../.claude/plans/please-use-these-ethereal-kite.md) for the
full design rationale.

## Quick start

```bash
# build everything
cargo build --release

# run a backtest on synthetic data
cargo run --release --bin algo-backtest -- --strategy xs_momentum --synthetic

# paper trading (requires ALPACA_API_KEY and ALPACA_API_SECRET env vars)
ALPACA_API_KEY=... ALPACA_API_SECRET=... \
  cargo run --release --bin algo-live -- --config configs/base.toml --paper
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
