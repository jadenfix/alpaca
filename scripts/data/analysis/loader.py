"""Multi-CSV time-series alignment for econophysics analysis.

Each CSV is read in one of three supported schemas (bar, macro, factor),
the canonical value column is extracted, and all series are inner-joined
on `ts_nanos` so the output is a single dense `(T, N)` numpy array plus
the column names.

Optional forward-fill within a max-gap budget: useful for monthly macro
series joined against daily equity bars, where the macro value is held
constant across the daily timestamps within its month.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# Make `_common` importable when run from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def detect_value_col(header: list[str]) -> tuple[str, int, int]:
    """Map a CSV header to (series_kind, ts_col_idx, value_col_idx).

    Kinds:
      - bar:    [ts_nanos, symbol, open, high, low, close, volume, span_secs]
                -> we use `close` (index 5) as the canonical value
      - macro:  [ts_nanos, series, value]
      - factor: [ts_nanos, factor, value]
      - yield:  [ts_nanos, tenor, yield_pct]
      - pageviews: [ts_nanos, article, views]
    """
    if header[:8] == ["ts_nanos", "symbol", "open", "high", "low", "close", "volume", "span_secs"]:
        return "bar", 0, 5
    if header == ["ts_nanos", "series", "value"]:
        return "macro", 0, 2
    if header == ["ts_nanos", "factor", "value"]:
        return "factor", 0, 2
    if header[:2] == ["ts_nanos", "tenor"]:
        return "yield", 0, 2
    if header[:2] == ["ts_nanos", "article"]:
        return "pageviews", 0, 2
    raise ValueError(f"unknown CSV schema: header={header}")


def load_one(path: Path) -> dict[int, float]:
    """Load a single CSV into a `{ts_nanos -> value}` dict.

    For bar files we keep the `close` value.
    For multi-series files (factor with multiple factor names) we collapse
    to the first encountered series name to keep this loader simple.
    Callers wanting per-factor decomposition should pre-split.
    """
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        kind, ts_idx, val_idx = detect_value_col(header)
        out: dict[int, float] = {}
        for row in reader:
            if len(row) <= max(ts_idx, val_idx):
                continue
            try:
                ts = int(row[ts_idx])
                v = float(row[val_idx])
            except (ValueError, TypeError):
                continue
            if v != v:  # NaN
                continue
            # For multi-series files in long format (multiple factor names
            # at the same timestamp), keep the FIRST one we see per ts.
            if ts not in out:
                out[ts] = v
        return out


def align(paths: list[Path], max_gap_bars: int = 0) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Inner-join multiple CSVs on `ts_nanos`.

    Returns:
        data: ndarray of shape (T, N) where N = len(paths) after pruning.
        names: list of column names (file stems for missing-data series).
        ts: ndarray of ts_nanos of length T.

    If `max_gap_bars > 0`, each series is forward-filled up to that many
    bars (in the union timestamp grid) before the inner join is computed.
    This is critical when joining monthly macro against daily bars.
    """
    if not paths:
        raise ValueError("loader.align: no input paths")
    per_file: list[dict[int, float]] = [load_one(p) for p in paths]
    names = [p.stem for p in paths]

    if max_gap_bars > 0:
        # Build the union of timestamps and forward-fill each series across it.
        union_ts = sorted(set().union(*[d.keys() for d in per_file]))
        for i, d in enumerate(per_file):
            last_v: float | None = None
            gap = 0
            for ts in union_ts:
                if ts in d:
                    last_v = d[ts]
                    gap = 0
                elif last_v is not None and gap < max_gap_bars:
                    d[ts] = last_v
                    gap += 1
                else:
                    gap += 1

    # Inner-join: keep timestamps where every series has a value.
    common = set(per_file[0].keys())
    for d in per_file[1:]:
        common &= set(d.keys())
    ts_sorted = np.array(sorted(common), dtype=np.int64)
    if ts_sorted.size == 0:
        return np.zeros((0, len(paths))), names, ts_sorted
    data = np.zeros((ts_sorted.size, len(paths)), dtype=np.float64)
    for j, d in enumerate(per_file):
        for i, ts in enumerate(ts_sorted):
            data[i, j] = d[int(ts)]
    return data, names, ts_sorted


def returns(data: np.ndarray) -> np.ndarray:
    """Per-column "returns": log-diff for strictly positive series, plain
    first-diff for series that can take non-positive values (rates,
    spreads). Output shape is (T-1, N). NaNs from the transform are
    filled with 0.0 so downstream covariance/PCA stays finite.
    """
    if data.shape[0] < 2:
        return np.zeros((0, data.shape[1]))
    out = np.zeros((data.shape[0] - 1, data.shape[1]), dtype=np.float64)
    for j in range(data.shape[1]):
        col = data[:, j]
        if np.all(col > 0):
            out[:, j] = np.diff(np.log(col))
        else:
            out[:, j] = np.diff(col)
    # Replace any residual non-finite values with 0 to keep downstream
    # spectral methods well-defined.
    out = np.where(np.isfinite(out), out, 0.0)
    return out
