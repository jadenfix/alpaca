#!/usr/bin/env python3
"""
Fetch high-resolution crypto OHLCV bars from Binance's public REST API.

Source (free, no key for klines endpoint):
    https://api.binance.com/api/v3/klines?symbol=<PAIR>&interval=<INT>&limit=1000

We can grab up to 1000 bars per request. The script pages backwards from the
most recent bar through the requested history.

Output: data/binance/<SYMBOL>_<INTERVAL>.csv (bars-CSV schema)

Defaults: 1-hour bars for ~6 months across top 8 pairs by USDT volume.

Usage:
    python3 scripts/data/fetch_binance.py
    python3 scripts/data/fetch_binance.py BTCUSDT ETHUSDT --interval 5m --bars 5000
"""

from __future__ import annotations
import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from _common import DATA_ROOT, emit_manifest, ensure_dir, http_get_json, log

BINANCE_DIR = DATA_ROOT / "binance"

DEFAULT_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
]

INTERVAL_SECS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400,
}


def short_symbol(pair: str) -> str:
    base = pair.replace("USDT", "")
    return base[:12]


def fetch_one(pair: str, interval: str, target_bars: int) -> tuple[int, dict]:
    log(f"↓ Binance {pair} {interval}")
    if interval not in INTERVAL_SECS:
        return 0, {"pair": pair, "error": f"unsupported interval {interval}"}
    span = INTERVAL_SECS[interval]
    end_ms = int(time.time() * 1000)
    collected: list[list] = []
    while len(collected) < target_bars:
        chunk = min(1000, target_bars - len(collected))
        url = (
            f"https://api.binance.com/api/v3/klines?symbol={pair}"
            f"&interval={interval}&limit={chunk}&endTime={end_ms}"
        )
        try:
            data = http_get_json(url, timeout=30)
        except Exception as e:
            log(f"  ⚠ {pair}: {e}; partial result kept")
            break
        if not data:
            break
        # Binance returns ascending; we want oldest-first.
        # We'll set end_ms to the FIRST bar's open time minus 1, then continue.
        first_open = data[0][0]
        # Prepend (we're paging backwards by chunks)
        collected = data + collected
        # If we got fewer than requested, exchange ran out of history.
        if len(data) < chunk:
            break
        end_ms = first_open - 1
        time.sleep(0.15)
    # Dedupe and sort by openTime.
    seen = set()
    rows = []
    sym = short_symbol(pair)
    for r in collected:
        ot = int(r[0])
        if ot in seen:
            continue
        seen.add(ot)
        ts_ns = ot * 1_000_000  # ms → ns
        try:
            o = float(r[1]); h = float(r[2]); lo = float(r[3])
            c = float(r[4]); v = float(r[5])
        except (ValueError, TypeError):
            continue
        if min(o, h, lo, c) <= 0:
            continue
        rows.append([ts_ns, sym, f"{o:.8f}", f"{h:.8f}", f"{lo:.8f}", f"{c:.8f}", f"{v:.4f}", span])
    rows.sort(key=lambda x: x[0])
    if not rows:
        return 0, {"pair": pair, "error": "no rows"}
    out_path = BINANCE_DIR / f"{sym}_{interval}.csv"
    ensure_dir(BINANCE_DIR)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_nanos", "symbol", "open", "high", "low", "close", "volume", "span_secs"])
        w.writerows(rows)
    log(f"  ✓ {sym}: {len(rows)} bars")
    return len(rows), {
        "pair": pair, "symbol": sym, "interval": interval,
        "rows": len(rows), "path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pairs", nargs="*", help="Pairs like BTCUSDT (default basket)")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--bars", type=int, default=4000)
    args = ap.parse_args()
    pairs = args.pairs or DEFAULT_PAIRS
    entries = []
    total = 0
    for p in pairs:
        n, e = fetch_one(p, args.interval, args.bars)
        entries.append(e); total += n
        time.sleep(0.3)
    mpath = emit_manifest("binance", entries)
    log(f"\nFetched {total} crypto bars across {len(pairs)} pairs.")
    log(f"Manifest: {mpath}")


if __name__ == "__main__":
    main()
