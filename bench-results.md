# Benchmark Results

Measured on Apple Silicon (M-series), release build with `-C target-cpu=native`
and `lto=fat`. Numbers are median of 20 samples via `criterion` unless noted.

Reproduce: `cargo bench -p algo-backtest --bench hot_path`.

## Per-event hot path

| Scenario | Baseline | Optimized | Δ |
|---|---|---|---|
| `backtest/noop` 16k events | 838 µs | **561 µs** | **−33%** |
| `backtest/noop` 160k events | 8.46 ms | **5.73 ms** | **−32%** |
| `backtest/noop` 1.6M events | 85.3 ms | **55.9 ms** | **−35%** |
| `backtest/xs_momentum` 8 × 1000 (8k events) | 23.4 ms | **19.2 ms** | **−18%** |
| `backtest/xs_momentum` 16 × 5000 (80k events) | 487 ms | **429 ms** | **−12%** |
| `backtest/xs_momentum` 32 × 5000 (160k events) | 1.34 s | **1.19 s** | **−12%** |

## Component micro-benchmarks

| Scenario | Time | Notes |
|---|---|---|
| `window/last_n_clone(20)` | 13.4 ns | unchanged — used in test asserts only |
| `window/last_n_slice(20)` | **0.37 ns** | new zero-copy API; **36× faster** |
| `features/welford/push_10k` | 36 µs | streaming, ~3.6 ns/push |
| `features/log_returns_vec(64)` | 196 ns | BLAS-backed |
| `features/log_returns_vec(1024)` | 2.1 µs | linear scaling |
| `features/log_returns_vec(16384)` | 37 µs | linear scaling |

## End-to-end soak

| Scenario | Time | Per-event |
|---|---|---|
| 1M-event xs_momentum soak | **8.27 s** | **8.3 µs/event** |
| 100k-event noop throughput sanity | < 100 ms | < 1 µs/event |

## What changed

### Simulator (`crates/backtest/src/simulator.rs`)

1. **Pre-allocate equity curve** with `Vec::with_capacity(events.len())`.
   Saves `O(log n)` reallocations on the 100k-event path.
2. **Reuse `Vec<PositionView>` scratch** as `self.pos_buf`. Cleared, not freed.
3. **Reuse `Vec<(Symbol, f64)>` flatten buffer** as `self.flatten_buf`.
   Only ever touched on the rare L4/L7 path.
4. **Hoist empty `HashMap` for ladder pnls** into `self.empty_pnls`. Reused
   every event.
5. **Inline mid_f64 conversion** instead of going through `Decimal` twice.
6. **Static-string ladder state** for flatten order tags, avoiding `format!`.

### XS-momentum strategy (`crates/strategies/src/xs_momentum.rs`)

1. **All per-event scratch on `self`**: `sym_buf`, `score_buf`,
   `price_scratch`, `desired_buf`, `sort_idx`. None allocate after warm-up.
2. **Streaming Welford pass** over log-returns in `compute_scores`. Replaces
   three separate passes (returns → mean → variance) with one.
3. **`Window::last_n_into(&mut Vec)`** instead of `last_n() -> Vec`. Reuses
   the price scratch buffer.
4. **Argsort by index** (`Vec<usize>`) instead of sorting `Vec<(Symbol, f64)>`.
   Smaller swaps; better cache.
5. **Direct linear position scan** instead of building intermediate
   `AHashMap<Symbol, f64> pos_map` and `AHashSet<Symbol> all_symbols`.
   For ≤16 open positions, linear is faster than hashing.
6. **`bars_held.retain(...)` walks positions slice directly** without
   building an intermediate `AHashSet`.

### Window (`crates/features/src/window.rs`)

1. **`last_n_into(n, &mut Vec)`** — reusable-buffer version. Used by xs_momentum.
2. **`last_n_slice(n) -> (&[T], &[T])`** — zero-copy peek at the two halves
   of the underlying `VecDeque`. **36× faster than the clone version.**
   Reserved for code that can iterate both slices.

### Walk-forward (`crates/backtest/src/walkforward.rs`)

1. **`rayon::par_iter` over folds.** Each fold is independent by construction,
   so 4 folds on a 4-core box ≈ 4× wall-clock reduction (with some overhead).

## What we did NOT change (and why)

- **Custom global allocator** — profile shows allocations are not the bottleneck
  after the buffer reuse. Reserve for if we see `malloc` cost in a future profile.
- **`unsafe` shortcuts on HashMap** — perf is fine for retail-scale universes.
- **Removing `Decimal` from `Price`/`Qty`** — kept at API boundary; hot path
  already converts to `f64` once and works with `f64` thereafter.
- **Symbol → u16 index pre-registration** — would yield 3–5× speedup on
  HashMap-heavy paths for universes >100 symbols, but it's a major refactor
  that touches every crate. Listed in ARCHITECTURE.md §8 as the next big win.
- **SIMD log-returns** — `ndarray + BLAS` already vectorizes the batch path;
  the streaming Welford in `compute_scores` is serial by definition.

## Quality guarantees

Across all the optimizations:

- **18 E2E integration tests** still pass.
- **11 property tests** (~2,800 random cases) still pass.
- **4 concurrent OMS stress tests** (8–16 threads, 1k–8k operations each) pass.
- **Backtest determinism** verified: 100 reruns produce byte-identical NAVs.
- **1M-event soak** passes without leak, panic, or invariant violation.
