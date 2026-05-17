#!/usr/bin/env python3
"""
Fetch fundamentals snapshots from Yahoo Finance (via yfinance) and compute
classic factor scores (value / quality / growth / profitability / leverage /
size) per ticker.  Per-symbol features are written to
    data/fundamentals/<SYM>_features.csv
in the macro schema (ts_nanos, series, value).

The module also exposes helpers to build a cross-sectional factor table
(median-and-MAD robust z-scores) and a composite quality+value+growth score
suitable for driving a factor-tilt strategy.

Usage:
    pip install yfinance==0.2.66
    python3 scripts/data/fetch_fundamentals.py --symbols AAPL MSFT GOOG \
        --out data/fundamentals/

Pitfalls:
  - yfinance.Ticker(sym).info often raises or returns None for ETFs (SPY, QQQ,
    XLK, ...) — wrap calls in try/except and return {} cleanly.
  - Many fields can be missing or non-finite — skip silently.
  - debtToEquity is reported in percent by yfinance; we convert to decimal.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Iterable

# Make local helpers importable both when run as a script and as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import MACRO_HEADER, ensure_dir  # noqa: E402

import yfinance as yf  # noqa: E402


# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------

# (feature_name, info_key, transform) — transform takes the raw value and
# returns the stored feature value (or raises / returns non-finite to skip).
def _identity(x: float) -> float:
    return float(x)


def _earnings_yield(pe: float) -> float:
    pe = float(pe)
    if pe == 0:
        return float("nan")
    return 1.0 / pe


def _div_by_100(x: float) -> float:
    return float(x) / 100.0


def _log(x: float) -> float:
    x = float(x)
    if x <= 0:
        return float("nan")
    return math.log(x)


# Features that derive from a *single* info field can be expressed this way.
# More complex features (those needing two fields) are handled inline below.
SINGLE_FIELD_FEATURES: list[tuple[str, str, callable]] = [
    # quality
    ("roe", "returnOnEquity", _identity),
    ("roa", "returnOnAssets", _identity),
    ("gross_margin", "grossMargins", _identity),
    ("current_ratio", "currentRatio", _identity),
    # growth
    ("revenue_growth_yoy", "revenueGrowth", _identity),
    ("earnings_growth_yoy", "earningsGrowth", _identity),
    # profitability
    ("operating_margin", "operatingMargins", _identity),
    ("profit_margin", "profitMargins", _identity),
    # leverage (yfinance reports debtToEquity as a percent)
    ("debt_to_equity", "debtToEquity", _div_by_100),
    # size
    ("log_market_cap", "marketCap", _log),
]

# Buckets used by the composite score and any downstream tilt logic.
VALUE_FEATURES = ("earnings_yield", "ev_to_ebitda", "book_to_market")
QUALITY_FEATURES = ("roe", "roa", "gross_margin", "current_ratio")
GROWTH_FEATURES = ("revenue_growth_yoy", "earnings_growth_yoy")

# For the composite, lower ev_to_ebitda is "better" / cheaper, so we negate
# its z-score when aggregating into value_z.  All other features are oriented
# so that higher = better.
VALUE_SIGNS = {"earnings_yield": +1.0, "ev_to_ebitda": -1.0, "book_to_market": +1.0}


# ---------------------------------------------------------------------------
# Tiny numeric helpers (no numpy dep so this stays test-friendly)
# ---------------------------------------------------------------------------

def _is_finite_number(x) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _mad(xs: list[float], med: float) -> float:
    if not xs:
        return float("nan")
    return _median([abs(x - med) for x in xs])


def _robust_z(x: float, med: float, mad: float) -> float:
    """Median-and-MAD z-score.  Returns 0.0 when MAD = 0 (degenerate cross-
    section: every symbol has the same value)."""
    if not math.isfinite(mad) or mad == 0.0:
        return 0.0
    return 0.6745 * (x - med) / mad


# ---------------------------------------------------------------------------
# Core fetcher
# ---------------------------------------------------------------------------

def _compute_features(info: dict) -> dict[str, float]:
    """Translate a raw yfinance .info dict into our feature dict, silently
    dropping fields whose underlying inputs are missing or non-finite."""
    if not isinstance(info, dict):
        return {}
    feats: dict[str, float] = {}

    # --- composite value features (two-field derivations) ------------------
    pe = info.get("trailingPE")
    if _is_finite_number(pe) and float(pe) != 0.0:
        v = _earnings_yield(pe)
        if _is_finite_number(v):
            feats["earnings_yield"] = float(v)

    ev = info.get("enterpriseValue")
    ebitda = info.get("ebitda")
    if (_is_finite_number(ev) and _is_finite_number(ebitda)
            and float(ebitda) != 0.0):
        v = float(ev) / float(ebitda)
        if _is_finite_number(v):
            feats["ev_to_ebitda"] = v

    book = info.get("bookValue")
    mcap = info.get("marketCap")
    if (_is_finite_number(book) and _is_finite_number(mcap)
            and float(mcap) != 0.0):
        v = float(book) / float(mcap)
        if _is_finite_number(v):
            feats["book_to_market"] = v

    # --- single-field features --------------------------------------------
    for name, key, transform in SINGLE_FIELD_FEATURES:
        raw = info.get(key)
        if not _is_finite_number(raw):
            continue
        try:
            v = transform(raw)
        except (ValueError, ZeroDivisionError, TypeError):
            continue
        if _is_finite_number(v):
            feats[name] = float(v)

    return feats


def _write_features_csv(path: Path, feats: dict[str, float]) -> None:
    ensure_dir(path.parent)
    ts = int(time.time() * 1_000_000_000)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MACRO_HEADER)
        for name in sorted(feats):
            w.writerow([ts, name, f"{feats[name]:.10g}"])


def fetch_fundamentals(symbol: str, out_dir: Path) -> dict[str, float]:
    """Return computed feature dict for `symbol`. Skip features where the
    underlying field is missing or non-finite. Write the result to
    `out_dir/<SYMBOL>_features.csv` in macro schema. If no features could be
    computed (e.g. ETF / network error), return {} and write nothing.
    """
    out_dir = Path(out_dir)
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
    except Exception:
        return {}
    if not info:
        return {}
    feats = _compute_features(info)
    if not feats:
        return {}
    _write_features_csv(out_dir / f"{symbol}_features.csv", feats)
    return feats


# ---------------------------------------------------------------------------
# Cross-sectional aggregation
# ---------------------------------------------------------------------------

def _read_features_csv(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with open(path) as f:
        r = csv.reader(f)
        header = next(r, None)
        if header != MACRO_HEADER:
            return out
        for row in r:
            if len(row) < 3:
                continue
            _ts, name, raw = row[0], row[1], row[2]
            try:
                v = float(raw)
            except ValueError:
                continue
            if math.isfinite(v):
                out[name] = v
    return out


def universe_factor_table(
    symbols: list[str], in_dir: Path
) -> dict[str, dict[str, float]]:
    """Aggregate previously-fetched per-symbol feature CSVs into
    `{symbol: {feature: z}}` where each feature is cross-sectionally z-scored
    across `symbols` using a robust median-and-MAD scaler:
        z = 0.6745 * (x - median) / mad
    Symbols missing a particular feature are simply absent from that feature's
    z-score column (the returned per-symbol dict only contains features that
    were present for that symbol)."""
    in_dir = Path(in_dir)
    per_symbol: dict[str, dict[str, float]] = {}
    for sym in symbols:
        per_symbol[sym] = _read_features_csv(in_dir / f"{sym}_features.csv")

    # Collect all feature names that appear anywhere.
    feature_names: set[str] = set()
    for d in per_symbol.values():
        feature_names.update(d.keys())

    # Compute median / mad per feature using only the symbols that have it.
    stats: dict[str, tuple[float, float]] = {}
    for fname in feature_names:
        xs = [per_symbol[s][fname] for s in symbols if fname in per_symbol[s]]
        if not xs:
            continue
        med = _median(xs)
        mad = _mad(xs, med)
        stats[fname] = (med, mad)

    result: dict[str, dict[str, float]] = {}
    for sym in symbols:
        row: dict[str, float] = {}
        for fname, v in per_symbol[sym].items():
            med, mad = stats[fname]
            row[fname] = _robust_z(v, med, mad)
        result[sym] = row
    return result


def composite_score(symbols: list[str], in_dir: Path) -> dict[str, float]:
    """Compute a composite quality+value+growth score per symbol:
        score = 0.4 * value_z + 0.4 * quality_z + 0.2 * growth_z
    where each bucket z is the mean of its constituent feature z-scores
    (with `ev_to_ebitda` negated so lower-is-better is rewarded). Buckets
    with no available features for a symbol are ignored and their weight is
    redistributed proportionally over the buckets that are present."""
    table = universe_factor_table(symbols, in_dir)
    weights = (("value", 0.4, VALUE_FEATURES, VALUE_SIGNS),
               ("quality", 0.4, QUALITY_FEATURES, None),
               ("growth", 0.2, GROWTH_FEATURES, None))
    out: dict[str, float] = {}
    for sym in symbols:
        zs = table.get(sym, {})
        bucket_vals: list[tuple[float, float]] = []  # (weight, value)
        for _name, w, feats, signs in weights:
            present = []
            for f in feats:
                if f in zs:
                    s = signs[f] if signs else 1.0
                    present.append(s * zs[f])
            if present:
                bucket_vals.append((w, sum(present) / len(present)))
        if not bucket_vals:
            out[sym] = 0.0
            continue
        wsum = sum(w for w, _ in bucket_vals)
        if wsum == 0:
            out[sym] = 0.0
            continue
        out[sym] = sum(w * v for w, v in bucket_vals) / wsum
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbols", nargs="+", required=True,
                    help="Tickers to fetch fundamentals for")
    ap.add_argument("--out", default="data/fundamentals/",
                    help="Output directory (default: data/fundamentals/)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)
    n_ok = 0
    for sym in args.symbols:
        sym = sym.upper()
        feats = fetch_fundamentals(sym, out_dir)
        marker = "OK " if feats else "-- "
        print(f"  {marker} {sym:<8}  {len(feats):2d} features", file=sys.stderr)
        if feats:
            n_ok += 1
    print(f"\nWrote fundamentals for {n_ok}/{len(args.symbols)} symbols "
          f"into {out_dir}/", file=sys.stderr)

    if n_ok >= 2:
        comp = composite_score([s.upper() for s in args.symbols], out_dir)
        print("\nComposite scores:", file=sys.stderr)
        for sym, score in sorted(comp.items(), key=lambda kv: -kv[1]):
            print(f"  {sym:<8}  {score:+.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
