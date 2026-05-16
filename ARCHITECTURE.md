# Architecture & Performance — algo

Deep architectural review and the performance discipline that backs it up.

## 1. Hot path

This is the per-event work, the thing that runs on every market tick. Everything else is overhead.

```
event_in
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1.  Decode event (sonic-rs in live; direct in backtest)         │
│ 2.  Update reference caches: price, ADV, mark                   │
│ 3.  Compute NAV  (cached for this event — once)                 │
│ 4.  Build PositionView slice (reused buffer, no realloc)        │
│ 5.  Strategy.on_event → SmallVec<OrderIntent>                   │
│ 6.  Ladder.update(nav, pnls)                                    │
│ 7.  If L4/L7: synthesize flatten intents                        │
│ 8.  For each intent: pretrade.check → execute                   │
│ 9.  Push (ts, nav) into pre-allocated equity buffer             │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
event_out
```

**Per-event budget on CPU:** target p99 < 10 µs for a 100-symbol universe in
backtest (release build), measured by `criterion`. Live adds RTT to broker
(20–100 ms) but that's I/O, off the hot thread.

## 2. Crate dependency graph

```
                  ┌────────┐
                  │  core  │  (types, Ts, Symbol, Money, MarketEvent)
                  └────┬───┘
       ┌──────────────┼─────────┬───────────────┬──────────────┐
       ▼              ▼         ▼               ▼              ▼
 ┌──────────┐  ┌──────────┐ ┌──────┐  ┌────────────┐  ┌────────────┐
 │ calendar │  │ features │ │ obs  │  │   risk     │  │ portfolio  │
 └──────────┘  └────┬─────┘ └──────┘  └────────────┘  └────────────┘
                    │
                    ▼
              ┌──────────┐
              │ strategy │  (Strategy trait)
              └─────┬────┘
                    ▼
              ┌────────────┐
              │ strategies │  (xs_momentum, pairs, …)
              └─────┬──────┘
                    ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ backtest │  │   mdata  │  │  broker  │  │   oms    │
   └──────────┘  └──────────┘  └──────────┘  └──────────┘
                                                    ▲
                                                    │
   ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐
   │ journal  │  │ storage │  │ universe │  │   sim   │  │   gpu    │
   └──────────┘  └─────────┘  └──────────┘  └─────────┘  └──────────┘
```

`core` is dependency-free leaf; every other crate depends on it. No cycles.

## 3. Performance principles

### 3.1 Zero allocation in the per-event loop

Every allocation in step 1-9 above is a bug. We achieve this with:

- **Pre-allocated equity buffer** sized to the event count up-front.
- **Reused `Vec<PositionView>` scratch** owned by the simulator, cleared and
  refilled per event instead of reallocated.
- **`SmallVec<[OrderIntent; 4]>`** for strategy output — inline storage for the
  common case where a strategy emits 0–4 intents per event.
- **`Window::last_n_slice()`** returns `&[T]` instead of cloning into a new `Vec`.
- **`AHashMap` clear-and-reuse** for strategy scratch maps where the key set
  changes per rebalance.
- **Lazy flatten path**: the `to_flatten` Vec is only constructed when the
  ladder is actually in L4/L7. In normal operation that branch is never taken.

### 3.2 Compute once, reuse

- **NAV** is computed exactly once per event. Strategy view, ladder, and equity
  tracking all consume the cached value.
- **Reference price** (`pretrade`) and **mark** (`portfolio_state`) share the
  same observation. We don't store the same number twice in two HashMaps —
  this is enforced by routing both through the same `observe_bar` call.
- **Welford** for streaming stats (mean/variance) so we never recompute from
  scratch.

### 3.3 Vectorization where it matters

- All batch math is `ndarray` + BLAS (`Accelerate` on macOS, `OpenBLAS` on
  Linux). No scalar loops over price arrays.
- The clippy lint surface (`clippy.toml`) blocks `Instant::now()` outside
  `core::clock`, the single source of wall-clock truth — this protects
  backtest determinism, not just performance.

### 3.4 No virtual dispatch in the hot loop

- `Strategy::on_event` is the only dyn-dispatch site and it's once per event
  with a known signature. The simulator's `run<S: Strategy + ?Sized>` accepts
  both monomorphized and boxed strategies; the cost is one indirect call per
  event, dominated by everything else.
- `Clock` is `dyn` only at process boundaries.

### 3.5 Determinism

Reproducibility is a performance feature: if a backtest is not deterministic,
you cannot meaningfully optimize it (you're chasing noise) and you cannot
debug a production replay.

- All RNG uses xorshift64 seeded explicitly.
- Hash maps that affect order use `BTreeMap` or sorted iteration when needed.
- Loss ladder transitions are pure functions of (peak, current, limits).
- Journal write+read+rewrite is byte-identical (regression-tested).

## 4. Scalability

| Scale | Notes |
|---|---|
| **Universe size** | Linear in symbols for per-bar work; per-event work is constant. HashMap → array indexing via `SymbolTable` is the next step when N > 1k. |
| **Strategy count** | Each strategy is one `on_event` call per event; trivially parallelizable across CPU cores if events are partitioned by symbol. Currently serial. |
| **Bar density** | 1-minute bars × 252 days × ~390 mins ≈ 100k bars/symbol; tested at 80k events in benchmarks. |
| **Walk-forward folds** | Parallelized with `rayon::par_iter` (folds are independent by construction). |

## 5. Risk-first design

Performance never trumps risk. The hot path:

- Computes NAV before risk checks (so checks see the latest portfolio value).
- Evaluates the loss ladder on every event, not periodically.
- L4/L7 synthesize **close orders for every open position** — this was a real
  bug we caught in deep E2E testing (`loss_ladder_halts_at_l3_then_l4`); the
  ladder used to only block new entries while existing positions bled.
- Pre-trade rejects are recorded for kill-switch input (reject rate trigger).

## 6. What we explicitly do *not* optimize

These were considered and rejected:

- **Custom global allocator (mimalloc/jemalloc)**: marginal in a workload
  this allocation-light; adds platform complexity. Revisit if alloc shows up
  in a profile.
- **`unsafe` for HashMap entry shortcuts**: brittle; the BTreeMap/AHashMap
  perf is fine for retail-scale universes.
- **Hand-rolled SIMD on Welford**: BLAS+ndarray already use it for batch ops;
  Welford is streaming (one-at-a-time) where SIMD doesn't apply.
- **Lock-free position cache**: `papaya` already gives us this for OMS;
  the simulator is single-threaded so a plain HashMap is faster.
- **Removing `Decimal` from prices entirely**: kept at API boundaries for
  exactness and serde stability; hot-path math converts to `f64` once and
  works with `f64` internally.

## 7. Benchmark results

See `crates/backtest/benches/`. Numbers below are on Apple Silicon (M-series),
release profile with `-C target-cpu=native` and `lto=fat`.

| Scenario | Before | After |
|---|---|---|
| 100k events / 16 syms / xs_momentum | TBD | TBD |
| 1M events / 32 syms / noop strategy | TBD | TBD |
| Window::last_n_slice (vs last_n clone) | TBD | TBD |

(Filled in after the optimization pass — see `bench-results.md`.)

## 8. Future work (in priority order)

1. **`SymbolTable` indexed positions** — Replace `HashMap<Symbol, _>` with
   `Vec<_>` indexed by `SymbolId(u16)`. Estimated 3–5× speedup on per-event
   work for universes >100 symbols.
2. **Strategy parallelism** — `rayon` over a strategy slate when each
   strategy maintains independent state.
3. **Streaming Parquet ingest** — current CSV path is fine for ≤10M bars;
   Parquet via `polars-lazy` will scale to 100M+.
4. **HDR-histogram per-bin latency** in the simulator (already there for live)
   so backtests publish a latency profile too.
5. **`io_uring` / `kqueue` for journal writes** in live — flush bottleneck
   only matters at p99.9 in live, not in backtest.

## 9. Modularity contracts

### Strategy trait

```rust
pub trait Strategy: Send {
    fn name(&self) -> &'static str;
    fn requires_gpu(&self) -> bool { false }
    fn on_event(
        &mut self,
        event: &MarketEvent,
        portfolio: &PortfolioView,
        positions: &[PositionView],
    ) -> SmallVec<[OrderIntent; 4]>;
    fn on_session_close(&mut self, _ts: Ts) -> SmallVec<[OrderIntent; 4]> {
        SmallVec::new()
    }
}
```

A new strategy is one module + one `Strategy` impl + one config. No engine
changes required.

### Broker trait

```rust
#[async_trait]
pub trait Broker: Send + Sync {
    async fn submit(&self, intent: &OrderIntent) -> Result<OrderAck, BrokerError>;
    async fn cancel(&self, id: OrderId) -> Result<(), BrokerError>;
    async fn fills_since(&self, ts_nanos: i64) -> Result<Vec<Fill>, BrokerError>;
    async fn account_snapshot(&self) -> Result<AccountSnapshot, BrokerError>;
}
```

Backed by `AlpacaRest` (live/paper) and `PaperBroker` (in-process sim).

### MarketDataSource trait

```rust
#[async_trait]
pub trait MarketDataSource: Send {
    async fn run(
        &mut self,
        subs: SubscribeRequest,
        out: mpsc::Sender<MarketEvent>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}
```

Backed by `AlpacaWs` (live) and (TODO) historical replay sources.

### CostModel

`sim::CostModel` is a struct with a `fill(side, qty, ctx) -> FillResult` method.
Could become a trait if alternatives (e.g., Almgren-Chriss) are added.
