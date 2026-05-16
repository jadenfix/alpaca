#!/usr/bin/env python3
"""
Fetch daily OHLCV bars from Yahoo Finance (via yfinance) and write them to
data/real/<TICKER>.csv in the bars-CSV format the engine expects.

Usage:
    pip install yfinance pandas
    python3 scripts/fetch_yahoo.py SPY QQQ AAPL MSFT
    python3 scripts/fetch_yahoo.py            # default basket of liquid ETFs
    python3 scripts/fetch_yahoo.py --years 5 SPY QQQ
    python3 scripts/fetch_yahoo.py --merge    # also build data/real/universe.csv

Output schema:
    ts_nanos, symbol, open, high, low, close, volume, span_secs
where span_secs = 86400 (daily).
"""

import argparse
import csv
import os
import sys
from pathlib import Path

DEFAULT_TICKERS = [
    "SPY", "QQQ", "DIA", "IWM",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU",
    "GLD", "SLV", "TLT", "IEF", "HYG", "LQD",
    "EFA", "EEM", "VWO",
]

OUT_DIR = Path("data/real")


def fetch_one(ticker: str, years: int) -> int:
    """Download `years` of daily bars for `ticker`. Returns row count."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Run: pip install yfinance pandas", file=sys.stderr)
        sys.exit(1)
    df = yf.download(
        ticker,
        period=f"{years}y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return 0
    # Flatten MultiIndex columns if yfinance returned them
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{ticker}.csv"
    n = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_nanos", "symbol", "open", "high", "low", "close", "volume", "span_secs"])
        for ts, row in df.iterrows():
            ns = int(ts.timestamp() * 1_000_000_000)
            try:
                o = float(row["Open"])
                h = float(row["High"])
                lo = float(row["Low"])
                c = float(row["Close"])
                v = float(row["Volume"])
            except (KeyError, ValueError, TypeError):
                continue
            if not all(map(lambda x: x == x and x > 0, (o, h, lo, c))):
                continue
            w.writerow([ns, ticker, f"{o:.6f}", f"{h:.6f}", f"{lo:.6f}", f"{c:.6f}", f"{v:.0f}", 86400])
            n += 1
    return n


def merge_universe() -> int:
    """Concatenate every data/real/*.csv (except universe.csv) into a single
    chronologically-sorted file."""
    all_rows = []
    for p in sorted(OUT_DIR.glob("*.csv")):
        if p.name == "universe.csv":
            continue
        with open(p) as f:
            r = csv.reader(f)
            next(r, None)  # header
            for row in r:
                if len(row) == 8:
                    all_rows.append(row)
    all_rows.sort(key=lambda r: int(r[0]))
    out_path = OUT_DIR / "universe.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_nanos", "symbol", "open", "high", "low", "close", "volume", "span_secs"])
        w.writerows(all_rows)
    return len(all_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tickers", nargs="*", help="Tickers to fetch (default: liquid ETF basket)")
    ap.add_argument("--years", type=int, default=5, help="Years of history (default 5)")
    ap.add_argument("--merge", action="store_true", help="Build data/real/universe.csv afterwards")
    args = ap.parse_args()
    tickers = args.tickers or DEFAULT_TICKERS
    total = 0
    for t in tickers:
        n = fetch_one(t.upper(), args.years)
        marker = "✓" if n > 0 else "✗"
        print(f"  {marker} {t.upper():<8}  {n} rows")
        total += n
    print(f"\nFetched {total} total rows into {OUT_DIR}/")
    if args.merge:
        n = merge_universe()
        print(f"Merged universe: {n} rows → {OUT_DIR}/universe.csv")


if __name__ == "__main__":
    main()
