#!/usr/bin/env python3
"""
Fetch live equity-option chains via yfinance and compute summary features
(implied-vol level, skew, put/call ratios, gamma exposure, term slope) which
are then appended to ``data/options/<SYM>_options_summary.csv`` using the
macro CSV schema (``ts_nanos,series,value``).

Usage:
    python3 scripts/data/fetch_options.py --symbols SPY QQQ XLK --out data/options/

Each invocation appends one row per feature per symbol — so a six-symbol fetch
appends ~36 rows (6 symbols x 6 features).  Symbols without a listed options
chain (FRED macro series, illiquid tickers) are silently skipped.

Black-Scholes gamma is computed locally (no scipy dependency) using the
standard formulae below, with ``r=0.05`` and ``q=0``:

    d1    = (ln(S/K) + (r - q + 0.5 sigma^2) T) / (sigma sqrt(T))
    phi   = exp(-d1^2 / 2) / sqrt(2 pi)
    gamma = exp(-q T) * phi / (S * sigma * sqrt(T))
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Make ``_common`` importable whether the script is run as a module
# (``python -m scripts.data.fetch_options``) or directly.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _common import MACRO_HEADER, ensure_dir, log  # noqa: E402

DEFAULT_SYMBOLS = ["SPY", "QQQ", "XLK", "XLF", "XLE", "GLD", "TLT"]
DEFAULT_OUT_DIR = Path("data/options")

RISK_FREE = 0.05
DIV_YIELD = 0.0

FEATURE_NAMES = (
    "iv_atm",
    "iv_skew_25d",
    "put_call_volume_ratio",
    "put_call_oi_ratio",
    "total_gamma_exposure",
    "term_structure_slope",
)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def black_scholes_gamma(
    S: float,
    K: float,
    sigma: float,
    T: float,
    r: float = RISK_FREE,
    q: float = DIV_YIELD,
) -> float:
    """Per-share Black-Scholes gamma.  Returns 0.0 when inputs are degenerate."""
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return 0.0
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    return math.exp(-q * T) * _norm_pdf(d1) / (S * sigma * sqrt_t)


# ---------------------------------------------------------------------------
# Chain feature extraction
# ---------------------------------------------------------------------------


def _safe_float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN guard
        return None
    return v


def _iter_rows(df) -> Iterable[dict]:
    """Yield each option row as a plain dict of floats (skip rows missing
    the basics: strike / iv).  Works on pandas DataFrames returned by
    yfinance.Ticker.option_chain()."""
    if df is None:
        return
    # ``DataFrame.iterrows`` returns (index, Series); access by column name.
    for _, row in df.iterrows():
        strike = _safe_float(row.get("strike"))
        iv = _safe_float(row.get("impliedVolatility"))
        if strike is None or strike <= 0 or iv is None or iv <= 0:
            continue
        yield {
            "strike": strike,
            "iv": iv,
            "volume": _safe_float(row.get("volume")) or 0.0,
            "open_interest": _safe_float(row.get("openInterest")) or 0.0,
        }


def _atm_iv(calls: list[dict], puts: list[dict], spot: float, band: float = 0.05) -> float | None:
    """Average IV of calls + puts within ``band`` of spot (5% by default)."""
    lo, hi = spot * (1.0 - band), spot * (1.0 + band)
    ivs = [r["iv"] for r in calls + puts if lo <= r["strike"] <= hi]
    if not ivs:
        return None
    return sum(ivs) / len(ivs)


def _nearest_iv(rows: list[dict], target_strike: float) -> float | None:
    if not rows:
        return None
    best = min(rows, key=lambda r: abs(r["strike"] - target_strike))
    return best["iv"]


def compute_features(
    spot: float,
    front_calls: list[dict],
    front_puts: list[dict],
    next_calls: list[dict],
    next_puts: list[dict],
    front_dte_years: float,
) -> dict[str, float]:
    """Build the six-feature dict from chain rows already filtered to dicts."""
    features: dict[str, float] = {}

    # iv_atm
    iv_atm = _atm_iv(front_calls, front_puts, spot)
    if iv_atm is not None:
        features["iv_atm"] = iv_atm

    # iv_skew_25d  ~  IV(put @ 0.92 S) - IV(call @ 1.08 S)
    put_iv = _nearest_iv(front_puts, spot * 0.92)
    call_iv = _nearest_iv(front_calls, spot * 1.08)
    if put_iv is not None and call_iv is not None:
        features["iv_skew_25d"] = put_iv - call_iv

    # put/call volume ratio
    call_vol = sum(r["volume"] for r in front_calls)
    put_vol = sum(r["volume"] for r in front_puts)
    if call_vol > 0:
        features["put_call_volume_ratio"] = put_vol / call_vol

    # put/call open-interest ratio
    call_oi = sum(r["open_interest"] for r in front_calls)
    put_oi = sum(r["open_interest"] for r in front_puts)
    if call_oi > 0:
        features["put_call_oi_ratio"] = put_oi / call_oi

    # total gamma exposure (dollar gamma per 1% move, summed across the chain)
    if front_dte_years > 0:
        gex = 0.0
        for r in front_calls + front_puts:
            g = black_scholes_gamma(spot, r["strike"], r["iv"], front_dte_years)
            gex += r["open_interest"] * g * 100.0 * spot * spot
        features["total_gamma_exposure"] = gex

    # term-structure slope: ATM IV(next) - ATM IV(front)
    front_atm = _atm_iv(front_calls, front_puts, spot)
    next_atm = _atm_iv(next_calls, next_puts, spot)
    if front_atm is not None and next_atm is not None:
        features["term_structure_slope"] = next_atm - front_atm

    return features


# ---------------------------------------------------------------------------
# yfinance integration
# ---------------------------------------------------------------------------


def _spot_from_ticker(tk) -> float | None:
    """Best-effort spot price.  Tries ``fast_info`` then a 1d history fallback."""
    spot = None
    fast = getattr(tk, "fast_info", None)
    if fast is not None:
        for key in ("last_price", "lastPrice", "regular_market_price"):
            try:
                v = fast[key] if hasattr(fast, "__getitem__") else getattr(fast, key, None)
            except (KeyError, TypeError):
                v = None
            v = _safe_float(v)
            if v is not None and v > 0:
                spot = v
                break
    if spot is None:
        try:
            hist = tk.history(period="5d", interval="1d", auto_adjust=False)
        except Exception:  # pragma: no cover - defensive only
            hist = None
        if hist is not None and not getattr(hist, "empty", True):
            try:
                spot = _safe_float(hist["Close"].iloc[-1])
            except Exception:  # pragma: no cover
                spot = None
    return spot


def _parse_expiry(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def fetch_options_for(symbol: str, out_dir: Path) -> dict[str, float]:
    """Fetch and write features for ``symbol``.  Returns the features dict
    (empty if the symbol has no options chain or the chain is unusable)."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover - environment check
        raise RuntimeError("yfinance is required: pip install yfinance") from e

    tk = yf.Ticker(symbol)
    try:
        expiries = tuple(tk.options or ())
    except Exception as e:  # network / yfinance hiccup
        log(f"[options] {symbol}: options listing failed: {e}")
        return {}
    if not expiries:
        return {}

    spot = _spot_from_ticker(tk)
    if spot is None or spot <= 0:
        log(f"[options] {symbol}: no usable spot price")
        return {}

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    # Find the front-month expiry with strictly positive DTE.
    front_expiry = None
    front_dte_years = 0.0
    for exp in expiries:
        d = _parse_expiry(exp)
        if d is None:
            continue
        dte = (d - today).days
        if dte > 0:
            front_expiry = exp
            front_dte_years = dte / 365.0
            break
    if front_expiry is None:
        return {}

    # Next expiry strictly after the front one (for term-structure slope).
    next_expiry = None
    idx = expiries.index(front_expiry)
    for exp in expiries[idx + 1 :]:
        d = _parse_expiry(exp)
        if d is None:
            continue
        if (d - today).days > 0:
            next_expiry = exp
            break

    try:
        front = tk.option_chain(front_expiry)
    except Exception as e:
        log(f"[options] {symbol}: front chain fetch failed: {e}")
        return {}
    front_calls = list(_iter_rows(getattr(front, "calls", None)))
    front_puts = list(_iter_rows(getattr(front, "puts", None)))

    next_calls: list[dict] = []
    next_puts: list[dict] = []
    if next_expiry is not None:
        try:
            nxt = tk.option_chain(next_expiry)
            next_calls = list(_iter_rows(getattr(nxt, "calls", None)))
            next_puts = list(_iter_rows(getattr(nxt, "puts", None)))
        except Exception as e:
            log(f"[options] {symbol}: next chain fetch failed: {e}")

    features = compute_features(
        spot=spot,
        front_calls=front_calls,
        front_puts=front_puts,
        next_calls=next_calls,
        next_puts=next_puts,
        front_dte_years=front_dte_years,
    )
    if not features:
        return {}

    ts_nanos = int(time.time() * 1_000_000_000)
    write_features_csv(out_dir, symbol, ts_nanos, features)
    return features


# ---------------------------------------------------------------------------
# Output (macro-schema CSV, append mode)
# ---------------------------------------------------------------------------


def write_features_csv(
    out_dir: Path,
    symbol: str,
    ts_nanos: int,
    features: dict[str, float],
) -> Path:
    """Append one row per feature for ``symbol`` to
    ``<out_dir>/<SYM>_options_summary.csv`` (macro schema)."""
    ensure_dir(out_dir)
    path = Path(out_dir) / f"{symbol}_options_summary.csv"
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(MACRO_HEADER)
        for name, value in features.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if v != v:  # NaN guard
                continue
            w.writerow([ts_nanos, f"{symbol}_{name}", f"{v:.10g}"])
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch options summaries via yfinance")
    ap.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Tickers to query (default: SPY QQQ XLK XLF XLE GLD TLT)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: data/options/)",
    )
    args = ap.parse_args()

    ensure_dir(args.out)
    total = 0
    for sym in args.symbols:
        sym_u = sym.upper()
        try:
            feats = fetch_options_for(sym_u, args.out)
        except Exception as e:  # pragma: no cover - top-level guard
            log(f"[options] {sym_u}: failed: {e}")
            continue
        if not feats:
            print(f"  - {sym_u:<8}  no options chain")
            continue
        print(f"  + {sym_u:<8}  {len(feats)} features")
        total += len(feats)
    print(f"\nWrote {total} feature rows to {args.out}/")


if __name__ == "__main__":
    main()
