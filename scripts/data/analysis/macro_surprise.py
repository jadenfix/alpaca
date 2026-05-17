"""Macro-release surprise scoring.

FRED-style macro series (CPI, PCE, unemployment, retail sales, ...)
move markets via the SURPRISE component — the deviation of the new
release from the market's prior expectation — not the level itself.
We approximate the prior expectation with a causal EWMA forecast over
the trailing series, then z-score the forecast residual against the
rolling distribution of prior residuals to obtain a unit-free
"surprise" score per release.

Schema in, schema out:
    input  data/macro/<series>.csv         (ts_nanos,series,value)
    output data/macro_surprise/<series>_surprise.csv
                                           (ts_nanos,series,value)

The output `value` column is a z-score whose magnitude tells you how
unexpected the latest print was relative to recent surprises. Pair
with `event_indicator` to extract the discrete, tradeable shock
timestamps (|z| > threshold).

Pure numpy + stdlib.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


# ──────────────────────────── I/O helpers ──────────────────────────────


def load_macro_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a macro CSV, return (ts_nanos, values) as numpy arrays
    sorted by ts_nanos. Drop rows with non-finite values.

    The schema is (ts_nanos, series, value); the `series` column is
    ignored because we already know which file we are reading.
    """
    ts_list: list[int] = []
    val_list: list[float] = []
    with Path(path).open("r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        # Tolerate either a known schema or a header-less file.
        try:
            ts_idx = header.index("ts_nanos")
            val_idx = header.index("value")
        except ValueError:
            ts_idx, val_idx = 0, 2
            # The "header" was really a data row.
            try:
                ts_list.append(int(header[ts_idx]))
                val_list.append(float(header[val_idx]))
            except (ValueError, IndexError):
                pass
        for row in reader:
            if not row:
                continue
            try:
                ts = int(row[ts_idx])
                val = float(row[val_idx])
            except (ValueError, IndexError):
                continue
            if not math.isfinite(val):
                continue
            ts_list.append(ts)
            val_list.append(val)

    if not ts_list:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    ts_arr = np.asarray(ts_list, dtype=np.int64)
    val_arr = np.asarray(val_list, dtype=np.float64)
    order = np.argsort(ts_arr, kind="stable")
    return ts_arr[order], val_arr[order]


# ─────────────────────────── EWMA forecast ─────────────────────────────


def ewma_forecast(values: np.ndarray, halflife: int = 6) -> np.ndarray:
    """Causal EWMA: forecast[t] uses only data[:t].

    The forecast at time t is the exponentially-weighted average of
    values[0..t-1] (strictly causal). forecast[0] is undefined and is
    returned as NaN.

    The smoothing factor follows the standard halflife convention:
        α = 1 - exp(-ln 2 / halflife)
    so the weight of an observation k steps in the past decays to 1/2
    after exactly `halflife` steps.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or halflife <= 0:
        return out

    alpha = 1.0 - math.exp(-math.log(2.0) / float(halflife))
    # Running EWMA state, fed with values[0..t-1].
    state = float("nan")
    for t in range(n):
        # Publish the forecast that uses only values[:t].
        out[t] = state
        v = values[t]
        if not math.isfinite(v):
            # Skip non-finite samples — state is unchanged.
            continue
        if math.isnan(state):
            state = v
        else:
            state = state + alpha * (v - state)
    return out


# ────────────────────── surprise (z-score) signal ──────────────────────


def macro_surprises(values: np.ndarray, halflife: int = 6,
                    z_window: int = 36) -> np.ndarray:
    """For each release t, compute
        z[t] = (values[t] - forecast[t]) / std(residuals[max(0,t-z_window):t])
    where `forecast` is the causal EWMA and the std uses ONLY prior
    residuals (sample std, ddof=1).

    Returns a same-length array of z-scores; entries with insufficient
    history (fewer than 2 prior finite residuals, or zero std) are NaN.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    forecast = ewma_forecast(values, halflife=halflife)
    residuals = values - forecast  # NaN where forecast is NaN

    for t in range(n):
        r_t = residuals[t]
        if not math.isfinite(r_t):
            continue
        lo = max(0, t - int(z_window))
        prior = residuals[lo:t]
        prior = prior[np.isfinite(prior)]
        if prior.size < 2:
            continue
        sd = float(np.std(prior, ddof=1))
        if not math.isfinite(sd) or sd <= 0.0:
            continue
        out[t] = r_t / sd
    return out


# ──────────────────────── batch feature builder ────────────────────────


def surprise_features(macro_dir: Path, out_dir: Path) -> dict[str, np.ndarray]:
    """For each .csv in `macro_dir`, compute macro_surprises and write
    `<out_dir>/<series>_surprise.csv` with schema
    ts_nanos,series,value  (series column = "<name>_surprise_z").

    Series with fewer than 50 finite observations are skipped.

    Returns a dict mapping series name -> z-score array (only for
    series that produced output).
    """
    macro_dir = Path(macro_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, np.ndarray] = {}
    for csv_path in sorted(macro_dir.glob("*.csv")):
        name = csv_path.stem
        ts, values = load_macro_series(csv_path)
        if values.size < 50:
            continue
        z = macro_surprises(values)
        results[name] = z

        out_path = out_dir / f"{name}_surprise.csv"
        series_label = f"{name}_surprise_z"
        with out_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["ts_nanos", "series", "value"])
            for ts_i, z_i in zip(ts.tolist(), z.tolist()):
                if not math.isfinite(z_i):
                    continue
                writer.writerow([int(ts_i), series_label, f"{z_i:.10f}"])

    return results


# ─────────────────────────── event extraction ──────────────────────────


def event_indicator(surprise: np.ndarray, ts: np.ndarray,
                    z_threshold: float = 1.5) -> list[dict]:
    """Return a list of event rows ``{ts_nanos, value, sign}`` for every
    index where |surprise| > z_threshold (these are the tradeable macro
    shocks).
    """
    surprise = np.asarray(surprise, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.int64)
    if surprise.shape[0] != ts.shape[0]:
        raise ValueError("surprise and ts must have the same length")
    events: list[dict] = []
    for i in range(surprise.shape[0]):
        z = surprise[i]
        if not math.isfinite(z):
            continue
        if abs(z) > float(z_threshold):
            events.append({
                "ts_nanos": int(ts[i]),
                "value": float(z),
                "sign": 1 if z > 0 else -1,
            })
    return events
