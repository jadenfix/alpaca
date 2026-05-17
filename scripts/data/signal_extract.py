#!/usr/bin/env python3
"""
Turn any time series (bar CSV, macro CSV, factor CSV, …) into a research
feature dataframe with standard statistical transforms:

  - level
  - log
  - first-difference
  - log-return
  - rolling mean / std (windows: 5, 21, 63, 252)
  - z-score (level vs rolling window)
  - momentum (cumulative log-return over rolling window)
  - rolling realized volatility (daily-equivalent annualized)
  - rolling skew / kurtosis
  - Hurst exponent (per rolling window, optional & slow)
  - drawdown from running peak

The output is a tidy long-format CSV: (ts_nanos, series, feature, value)
suitable for cross-source merge / joins in pandas / polars / SQL.

Usage:
    python3 scripts/data/signal_extract.py data/macro/VIXCLS.csv
    python3 scripts/data/signal_extract.py data/real/SPY.csv --out features/spy.csv
    python3 scripts/data/signal_extract.py data/macro/*.csv --out-dir features/
"""

from __future__ import annotations
import argparse
import csv
import glob
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable

from _common import DATA_ROOT, ensure_dir, log

FEATURES_ROOT = DATA_ROOT / "features"
ROLLING_WINDOWS = [5, 21, 63, 252]


def detect_schema(header: list[str]) -> tuple[str, int, int]:
    """Returns (kind, ts_col_idx, value_col_idx). kind ∈ {bar, macro, factor, wide}."""
    if header[:8] == ["ts_nanos", "symbol", "open", "high", "low", "close", "volume", "span_secs"]:
        return "bar", 0, 5  # close column
    if header == ["ts_nanos", "series", "value"]:
        return "macro", 0, 2
    if header == ["ts_nanos", "factor", "value"]:
        return "factor", 0, 2
    if header[:2] == ["ts_nanos", "tenor"]:
        return "yield", 0, 2
    if header[:2] == ["ts_nanos", "article"]:
        return "pageviews", 0, 2
    return "unknown", 0, len(header) - 1


def load_series(path: Path) -> dict[str, list[tuple[int, float]]]:
    """Returns {series_name → [(ts, value), …]} grouped by the 'symbol/series/factor' col."""
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None) or []
        kind, ts_idx, val_idx = detect_schema(header)
        if kind == "bar":
            name_idx = 1
        elif kind in ("macro", "factor", "yield", "pageviews"):
            name_idx = 1
        else:
            name_idx = -1  # one anonymous series
        grouped: dict[str, list[tuple[int, float]]] = {}
        for row in reader:
            if not row:
                continue
            try:
                ts = int(row[ts_idx])
                v = float(row[val_idx])
            except (ValueError, IndexError):
                continue
            name = row[name_idx] if name_idx >= 0 else path.stem
            grouped.setdefault(name, []).append((ts, v))
    for k in grouped:
        grouped[k].sort(key=lambda x: x[0])
    return grouped


def rolling(values: list[float], window: int, fn):
    out = [float("nan")] * len(values)
    for i in range(window - 1, len(values)):
        chunk = values[i - window + 1: i + 1]
        try:
            out[i] = fn(chunk)
        except Exception:
            out[i] = float("nan")
    return out


def mean_(xs):  return sum(xs) / len(xs)
def std_(xs):
    if len(xs) < 2: return float("nan")
    return statistics.pstdev(xs)
def skew_(xs):
    if len(xs) < 3: return float("nan")
    m = mean_(xs); s = std_(xs)
    if s == 0: return 0.0
    return sum((x - m) ** 3 for x in xs) / (len(xs) * s ** 3)
def kurt_(xs):
    if len(xs) < 4: return float("nan")
    m = mean_(xs); s = std_(xs)
    if s == 0: return 0.0
    return sum((x - m) ** 4 for x in xs) / (len(xs) * s ** 4) - 3.0


def extract_features(name: str, series: list[tuple[int, float]]) -> Iterable[list]:
    """Yield rows (ts_nanos, series, feature, value)."""
    ts_list = [t for t, _ in series]
    vals = [v for _, v in series]
    n = len(vals)

    def emit(feature: str, sequence: list[float]):
        for t, v in zip(ts_list, sequence):
            if v == v:  # not NaN
                yield [t, name, feature, f"{v:.10f}"]

    yield from emit("level", vals)
    # log
    log_vals = [math.log(v) if v > 0 else float("nan") for v in vals]
    yield from emit("log", log_vals)
    # first diff
    diff = [float("nan")] + [vals[i] - vals[i - 1] for i in range(1, n)]
    yield from emit("diff", diff)
    # log-return
    ret = [float("nan")]
    for i in range(1, n):
        if vals[i - 1] > 0 and vals[i] > 0:
            ret.append(math.log(vals[i] / vals[i - 1]))
        else:
            ret.append(float("nan"))
    yield from emit("log_return", ret)

    # Drawdown from running peak
    peak = float("-inf")
    dd = []
    for v in vals:
        if v > peak:
            peak = v
        dd.append((v - peak) / peak if peak > 0 else float("nan"))
    yield from emit("drawdown", dd)

    for w in ROLLING_WINDOWS:
        if n < w + 1:
            continue
        ma = rolling(vals, w, mean_)
        sd = rolling(vals, w, std_)
        yield from emit(f"ma_{w}", ma)
        yield from emit(f"std_{w}", sd)
        # z-score: (v - ma) / sd
        z = [
            (vals[i] - ma[i]) / sd[i] if (sd[i] and sd[i] == sd[i] and sd[i] > 0) else float("nan")
            for i in range(n)
        ]
        yield from emit(f"zscore_{w}", z)
        # Momentum: cumulative log-return over window
        momo = [float("nan")] * n
        for i in range(w, n):
            if vals[i - w] > 0 and vals[i] > 0:
                momo[i] = math.log(vals[i] / vals[i - w])
        yield from emit(f"momentum_{w}", momo)
        # Rolling realized vol (annualized assuming daily bars)
        rv = rolling([r if r == r else 0.0 for r in ret], w, std_)
        rv_ann = [v * math.sqrt(252) if v == v else float("nan") for v in rv]
        yield from emit(f"rv_ann_{w}", rv_ann)
        # Skew / Kurt of returns
        skew = rolling([r if r == r else 0.0 for r in ret], w, skew_)
        kurt = rolling([r if r == r else 0.0 for r in ret], w, kurt_)
        yield from emit(f"skew_{w}", skew)
        yield from emit(f"kurt_{w}", kurt)


def process_file(in_path: Path, out_path: Path) -> tuple[int, int]:
    """Returns (n_series, n_feature_rows)."""
    grouped = load_series(in_path)
    ensure_dir(out_path.parent)
    n_series = 0
    n_rows = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_nanos", "series", "feature", "value"])
        for name, series in grouped.items():
            n_series += 1
            for row in extract_features(name, series):
                w.writerow(row)
                n_rows += 1
    return n_series, n_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="CSV files (or glob patterns) to process")
    ap.add_argument("--out", help="Output path (single input only)")
    ap.add_argument("--out-dir", default=str(FEATURES_ROOT), help="Output directory (multi-input)")
    args = ap.parse_args()
    paths: list[Path] = []
    for pat in args.inputs:
        matched = [Path(p) for p in glob.glob(pat)] or [Path(pat)]
        paths.extend(p for p in matched if p.exists())
    if not paths:
        log("no input files found"); sys.exit(1)
    if args.out and len(paths) > 1:
        log("--out is incompatible with multiple inputs"); sys.exit(1)
    total_series = 0
    total_rows = 0
    for p in paths:
        out = Path(args.out) if args.out else Path(args.out_dir) / f"{p.stem}_features.csv"
        n_s, n_r = process_file(p, out)
        log(f"  ✓ {p} → {out}  ({n_s} series, {n_r} feature rows)")
        total_series += n_s
        total_rows += n_r
    log(f"\nDone: {len(paths)} files, {total_series} series, {total_rows} feature rows.")


if __name__ == "__main__":
    main()
