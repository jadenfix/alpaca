#!/usr/bin/env python3
"""
Fetch daily OHLCV bars for crypto from CoinGecko's free public API.

Endpoints used (no key required):
  - /coins/{id}/ohlc?vs_currency=usd&days={n}   1-d, 7-d, 14-d, 30-d, 90-d,
                                                180-d, 365-d, max
  - /coins/{id}/market_chart?vs_currency=usd&days={n}&interval=daily
                                                 (used to pull volume)

CoinGecko rate-limits free users to ~30 calls/min. We sleep 2 s between
requests by default.

Output: data/crypto/<COIN_ID>.csv with the bars-CSV schema.

Symbol mapping for the engine: the CoinGecko id (e.g. `bitcoin`) is upper-
cased and truncated to 12 chars to fit our Symbol type. We persist the
mapping in data/crypto/_id_to_symbol.json so downstream code can resolve it.

Usage:
    python3 scripts/data/fetch_coingecko.py
    python3 scripts/data/fetch_coingecko.py bitcoin ethereum solana
"""

from __future__ import annotations
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from _common import CRYPTO_DIR, emit_manifest, ensure_dir, http_get_json, log

DEFAULT_COINS = [
    "bitcoin", "ethereum", "solana", "binancecoin",
    "ripple", "cardano", "avalanche-2", "polkadot",
]

BASE = "https://api.coingecko.com/api/v3"


def to_symbol(coin_id: str) -> str:
    """Map a CoinGecko id to a 12-char-max uppercase ticker."""
    cleaned = "".join(c for c in coin_id.upper() if c.isalnum())
    return cleaned[:12] or "CRYPTO"


def fetch_one(coin_id: str, days: int = 365) -> tuple[int, dict]:
    sym = to_symbol(coin_id)
    log(f"↓ CoinGecko {coin_id} → {sym}")
    try:
        # OHLC is candles every ~4h for days>1, but they snap to daily for days≥90.
        # We use market_chart for volume + close, then derive OHLC from candles.
        ohlc = http_get_json(
            f"{BASE}/coins/{coin_id}/ohlc?vs_currency=usd&days={days}",
            timeout=30,
        )
        time.sleep(2.0)
        chart = http_get_json(
            f"{BASE}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=daily",
            timeout=30,
        )
    except Exception as e:
        log(f"  ✗ {coin_id}: download failed ({e})")
        return 0, {"coin": coin_id, "error": str(e)}

    # Index volume by date-day for quick lookup.
    volumes_by_day: dict[str, float] = {}
    for ts_ms, v in chart.get("total_volumes", []):
        d = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        volumes_by_day[d] = v

    # Reduce OHLC candles (4h or so) to one daily bar per day.
    by_day: dict[str, list] = {}
    for row in ohlc:
        ts_ms, o, h, lo, c = row
        d = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if d not in by_day:
            by_day[d] = [o, h, lo, c]
        else:
            existing = by_day[d]
            # keep first open, max high, min low, last close
            by_day[d] = [existing[0], max(existing[1], h), min(existing[2], lo), c]

    rows = []
    for d in sorted(by_day):
        ts = int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1_000_000_000)
        o, h, lo, c = by_day[d]
        v = volumes_by_day.get(d, 0.0)
        if any(x <= 0 or x != x for x in (o, h, lo, c)):
            continue
        rows.append([ts, sym, f"{o:.6f}", f"{h:.6f}", f"{lo:.6f}", f"{c:.6f}", f"{v:.0f}", 86400])
    if not rows:
        return 0, {"coin": coin_id, "error": "no bars parsed"}
    out_path = CRYPTO_DIR / f"{sym}.csv"
    ensure_dir(out_path.parent)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_nanos", "symbol", "open", "high", "low", "close", "volume", "span_secs"])
        w.writerows(rows)
    log(f"  ✓ {sym}: {len(rows)} daily bars")
    return len(rows), {"coin": coin_id, "symbol": sym, "path": str(out_path), "rows": len(rows)}


def main():
    coins = sys.argv[1:] or DEFAULT_COINS
    ensure_dir(CRYPTO_DIR)
    entries = []
    total = 0
    id_map = {}
    for c in coins:
        n, e = fetch_one(c)
        entries.append(e)
        total += n
        if "symbol" in e:
            id_map[c] = e["symbol"]
        time.sleep(2.0)  # rate limit cushion
    with open(CRYPTO_DIR / "_id_to_symbol.json", "w") as f:
        json.dump(id_map, f, indent=2)
    mpath = emit_manifest("coingecko", entries)
    log(f"\nFetched {total} crypto bars across {len(coins)} coins.")
    log(f"Manifest: {mpath}")


if __name__ == "__main__":
    main()
