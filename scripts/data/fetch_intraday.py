#!/usr/bin/env python3
"""
Fetch 1-minute intraday OHLCV bars from Yahoo Finance (via yfinance) and write
them to data/intraday/<SYMBOL>_1m.csv in the standard bars schema.

yfinance free-tier limit: at most the last 7 calendar days of 1-minute data.

Bar schema (BAR_HEADER):
    ts_nanos, symbol, open, high, low, close, volume, span_secs

For each fetched symbol we can additionally compute a small feature pack and
emit it to data/intraday/<SYMBOL>_features.csv in the macro schema
(ts_nanos, series, value).

Usage:
    python3 scripts/data/fetch_intraday.py --symbols SPY QQQ --out data/intraday/
    python3 scripts/data/fetch_intraday.py --symbols AAPL --features
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - import-time guard
    yf = None  # type: ignore[assignment]

# Make _common importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import BAR_HEADER, MACRO_HEADER, ensure_dir, log  # noqa: E402


SPAN_SECS_1M = 60
US_EASTERN = "America/New_York"

# Approximate number of 1-minute US-equity bars per trading year, used to
# annualize the realized volatility computed from 1-minute log returns:
#   390 bars/session * 252 sessions/year = 98_280.
MINUTES_PER_YEAR = 390 * 252


# ---------------------------------------------------------------------------
# Core fetcher
# ---------------------------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns a MultiIndex on columns when threads / multiple symbols
    are involved. Flatten by keeping the top-level (field) name."""
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _df_to_bar_rows(df: pd.DataFrame, symbol: str) -> list[list]:
    """Convert a yfinance OHLCV DataFrame into BAR_HEADER-shaped rows."""
    df = _flatten_columns(df)
    rows: list[list] = []
    for ts, row in df.iterrows():
        try:
            o = float(row["Open"])
            h = float(row["High"])
            lo = float(row["Low"])
            c = float(row["Close"])
            v = float(row["Volume"])
        except (KeyError, ValueError, TypeError):
            continue
        # Skip rows with NaN / non-positive prices.
        if not all(x == x and x > 0 for x in (o, h, lo, c)):
            continue
        # Timestamp.value is integer nanoseconds since epoch in UTC, regardless
        # of timezone attached to the Timestamp.
        try:
            ns = int(pd.Timestamp(ts).value)
        except Exception:
            continue
        rows.append([
            ns, symbol,
            f"{o:.6f}", f"{h:.6f}", f"{lo:.6f}", f"{c:.6f}",
            f"{v:.0f}", SPAN_SECS_1M,
        ])
    return rows


def _read_existing_bars(path: Path) -> list[list]:
    """Read an existing BAR_HEADER-shaped CSV. Returns [] if missing/empty."""
    if not path.exists():
        return []
    out: list[list] = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None:
            return []
        for row in r:
            if len(row) == len(BAR_HEADER):
                out.append(row)
    return out


def _write_bar_rows(path: Path, rows: list[list]) -> int:
    ensure_dir(path.parent)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BAR_HEADER)
        for r in rows:
            w.writerow(r)
    return len(rows)


def _merge_dedup(existing: list[list], new: list[list]) -> list[list]:
    """Combine existing + new bar rows, dedup on ts_nanos (first wins for the
    same key — but new rows overwrite existing because we put new last and use
    a dict on the key)."""
    by_key: dict[int, list] = {}
    for r in existing:
        try:
            k = int(r[0])
        except (ValueError, TypeError):
            continue
        by_key[k] = r
    for r in new:
        try:
            k = int(r[0])
        except (ValueError, TypeError):
            continue
        by_key[k] = r
    return [by_key[k] for k in sorted(by_key.keys())]


def fetch_intraday(symbol: str, out_dir: Path,
                   period: str = "7d", interval: str = "1m") -> int:
    """Fetch and write intraday bars for ``symbol``. Returns count of bars
    written. If file already exists, merge new rows (de-dupe on ``ts_nanos``)
    instead of overwriting."""
    if yf is None:
        raise RuntimeError("yfinance is not installed. pip install yfinance")
    out_dir = Path(out_dir)
    out_path = out_dir / f"{symbol.upper()}_1m.csv"

    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or len(df) == 0:
        # Don't touch the file in this case; just report zero new bars.
        return 0

    new_rows = _df_to_bar_rows(df, symbol.upper())
    if not new_rows:
        return 0

    existing = _read_existing_bars(out_path)
    merged = _merge_dedup(existing, new_rows)
    return _write_bar_rows(out_path, merged)


# ---------------------------------------------------------------------------
# Feature aggregation
# ---------------------------------------------------------------------------

def _load_bars(path: Path) -> pd.DataFrame:
    """Load a BAR_HEADER-shaped CSV into a DataFrame indexed by US/Eastern
    timestamp, with float OHLCV columns."""
    df = pd.read_csv(path)
    if df.empty:
        return df
    ts = pd.to_datetime(df["ts_nanos"].astype("int64"), unit="ns", utc=True)
    df = df.assign(ts=ts).set_index("ts").sort_index()
    df.index = df.index.tz_convert(US_EASTERN)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        sa = float(np.std(a))
        sb = float(np.std(b))
        if sa == 0.0 or sb == 0.0 or not math.isfinite(sa) or not math.isfinite(sb):
            return float("nan")
        c = np.corrcoef(a, b)
        if c.shape != (2, 2):
            return float("nan")
        val = float(c[0, 1])
    return val if math.isfinite(val) else float("nan")


def _compute_features(df: pd.DataFrame) -> dict[str, float]:
    """Compute the intraday-feature dict for a tz-aware (US/Eastern) bar frame."""
    features: dict[str, float] = {
        "realized_vol_intraday": float("nan"),
        "vwap_premium": float("nan"),
        "opening_range_pct": float("nan"),
        "afternoon_drift": float("nan"),
        "volume_clock_skew": float("nan"),
    }
    if df.empty:
        return features

    # Restrict to US regular trading hours [09:30, 16:00).
    times = df.index
    minute_of_day = times.hour * 60 + times.minute
    rth = (minute_of_day >= 9 * 60 + 30) & (minute_of_day < 16 * 60)
    sess = df.loc[rth].copy()
    if sess.empty:
        return features

    # Keep at most the trailing 5 sessions.
    session_dates = np.asarray(sess.index.date)
    unique_dates = sorted(set(session_dates.tolist()))
    keep_dates = set(unique_dates[-5:])
    sess = sess.loc[[d in keep_dates for d in session_dates]]
    if sess.empty:
        return features

    closes = sess["close"].astype(float).to_numpy()
    vols = sess["volume"].astype(float).to_numpy()

    # --- realized_vol_intraday ------------------------------------------------
    # Annualized sqrt(sum of squared 1-min log returns). Computed across the
    # whole trailing window; we don't bridge across sessions in the return
    # series to avoid the overnight jump dominating.
    sess_dates = np.asarray(sess.index.date)
    sq_sum = 0.0
    n_ret = 0
    for d in sorted(set(sess_dates.tolist())):
        mask = (sess_dates == d)
        c = closes[mask]
        if c.size < 2:
            continue
        # Guard against non-positive prices.
        c = c[c > 0]
        if c.size < 2:
            continue
        lr = np.diff(np.log(c))
        sq_sum += float(np.sum(lr ** 2))
        n_ret += int(lr.size)
    if n_ret > 0:
        # Annualize by scaling sum-of-squared-returns to the trading-year length.
        annualized_var = sq_sum * (MINUTES_PER_YEAR / n_ret)
        features["realized_vol_intraday"] = float(math.sqrt(annualized_var))

    # --- most-recent session slice -------------------------------------------
    most_recent = max(unique_dates) if unique_dates else None
    today = sess.loc[(sess_dates == most_recent)] if most_recent is not None else sess.iloc[0:0]

    if not today.empty:
        t_close = today["close"].astype(float).to_numpy()
        t_vol = today["volume"].astype(float).to_numpy()

        # --- vwap_premium ----------------------------------------------------
        denom = float(t_vol.sum())
        if denom > 0:
            vwap = float(np.dot(t_close, t_vol) / denom)
            last_close = float(t_close[-1])
            if vwap > 0:
                features["vwap_premium"] = (last_close - vwap) / vwap

        # --- opening_range_pct ----------------------------------------------
        t_min = today.index.hour * 60 + today.index.minute
        opening_mask = (t_min >= 9 * 60 + 30) & (t_min < 10 * 60)
        opening = today.loc[opening_mask]
        if not opening.empty:
            # Find the 09:30 bar specifically for the denominator open.
            open_min = opening.index.hour * 60 + opening.index.minute
            at_930 = opening.loc[open_min == 9 * 60 + 30]
            if not at_930.empty:
                o_930 = float(at_930["open"].iloc[0])
                hi = float(opening["high"].max())
                lo = float(opening["low"].min())
                if o_930 > 0:
                    features["opening_range_pct"] = (hi - lo) / o_930

        # --- afternoon_drift -------------------------------------------------
        late_mask = (t_min >= 15 * 60 + 30) & (t_min < 16 * 60)
        early_mask = (t_min >= 13 * 60 + 30) & (t_min < 14 * 60)
        late = today.loc[late_mask, "close"].astype(float)
        early = today.loc[early_mask, "close"].astype(float)
        if not late.empty and not early.empty:
            lm = float(late.mean())
            em = float(early.mean())
            if lm > 0 and em > 0:
                features["afternoon_drift"] = math.log(lm) - math.log(em)

    # --- volume_clock_skew ---------------------------------------------------
    # Pearson corr between time-of-day (minutes since 09:30) and log(volume)
    # over the trailing window. Skip zero-volume bars to keep the log defined.
    tod_min = (sess.index.hour * 60 + sess.index.minute - (9 * 60 + 30)).to_numpy(dtype=float)
    vol_arr = vols.astype(float)
    keep = vol_arr > 0
    if keep.sum() >= 2:
        features["volume_clock_skew"] = _safe_corr(
            tod_min[keep], np.log(vol_arr[keep])
        )
    return features


def aggregate_intraday_features(symbol: str, in_dir: Path,
                                out_dir: Path) -> dict[str, float]:
    """Read the most recent intraday CSV and compute a per-symbol feature
    summary for the trailing 5 sessions. Writes
    ``data/intraday/<SYM>_features.csv`` in the macro schema
    (ts_nanos, series, value) with one row per feature.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    sym = symbol.upper()
    bars_path = in_dir / f"{sym}_1m.csv"
    if not bars_path.exists():
        raise FileNotFoundError(f"intraday bar file not found: {bars_path}")
    df = _load_bars(bars_path)
    feats = _compute_features(df)

    # Tag the feature row with the timestamp of the latest bar (in UTC ns).
    if df.empty:
        ts_ns = 0
    else:
        ts_ns = int(df.index[-1].tz_convert("UTC").value)

    out_path = out_dir / f"{sym}_features.csv"
    ensure_dir(out_path.parent)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MACRO_HEADER)
        for name, value in feats.items():
            series = f"{sym}_{name}"
            # NaN serializes as the literal "nan" by default; keep that.
            w.writerow([ts_ns, series, value])
    return feats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    ap.add_argument("--symbols", nargs="+", required=True,
                    help="Tickers to fetch (e.g. --symbols SPY QQQ AAPL)")
    ap.add_argument("--out", type=Path, default=Path("data/intraday"),
                    help="Output directory (default data/intraday/)")
    ap.add_argument("--period", default="7d",
                    help="yfinance period (default 7d, the free-tier 1m max)")
    ap.add_argument("--interval", default="1m",
                    help="yfinance interval (default 1m)")
    ap.add_argument("--features", action="store_true",
                    help="Also compute per-symbol intraday feature summaries")
    args = ap.parse_args()

    ensure_dir(args.out)
    total = 0
    for sym in args.symbols:
        sym_u = sym.upper()
        try:
            n = fetch_intraday(sym_u, args.out,
                               period=args.period, interval=args.interval)
        except Exception as e:
            log(f"  ! {sym_u:<8}  fetch failed: {e}")
            n = 0
        marker = "ok" if n > 0 else "--"
        print(f"  {marker} {sym_u:<8}  {n} bars")
        total += n
        if args.features and n > 0:
            try:
                feats = aggregate_intraday_features(sym_u, args.out, args.out)
                bits = ", ".join(f"{k}={v:.4g}" for k, v in feats.items())
                print(f"     features: {bits}")
            except Exception as e:
                log(f"     feature computation failed for {sym_u}: {e}")
    print(f"\nFetched {total} total bars into {args.out}/")


if __name__ == "__main__":
    main()
