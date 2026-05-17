#!/usr/bin/env python3
"""
Fetch World Bank country indicators via the free WBdata API (no key).

Endpoint:
    https://api.worldbank.org/v2/country/<COUNTRY>/indicator/<INDICATOR>
        ?format=json&per_page=20000

We pull annual time series for the US (and a few peer economies) on a curated
set of indicators that move equity and currency markets:

  - NY.GDP.MKTP.CD          GDP (current US$)
  - NY.GDP.MKTP.KD.ZG       GDP growth (annual %)
  - FP.CPI.TOTL.ZG          Inflation, consumer prices (annual %)
  - SL.UEM.TOTL.ZS          Unemployment, total (% of labor force)
  - NE.EXP.GNFS.CD          Exports of goods and services (current US$)
  - NE.IMP.GNFS.CD          Imports of goods and services (current US$)
  - FR.INR.LEND             Lending interest rate (%)
  - GC.DOD.TOTL.GD.ZS       Central government debt (% of GDP)

Output: data/worldbank/<COUNTRY>_<INDICATOR>.csv with schema
    ts_nanos, country, indicator, value

Usage:
    python3 scripts/data/fetch_worldbank.py
    python3 scripts/data/fetch_worldbank.py USA CHN
"""

from __future__ import annotations
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from _common import DATA_ROOT, emit_manifest, ensure_dir, http_get_json, log

WB_DIR = DATA_ROOT / "worldbank"

DEFAULT_COUNTRIES = ["USA", "CHN", "JPN", "DEU", "GBR"]
DEFAULT_INDICATORS = [
    "NY.GDP.MKTP.CD",
    "NY.GDP.MKTP.KD.ZG",
    "FP.CPI.TOTL.ZG",
    "SL.UEM.TOTL.ZS",
    "NE.EXP.GNFS.CD",
    "NE.IMP.GNFS.CD",
    "FR.INR.LEND",
    "GC.DOD.TOTL.GD.ZS",
]


def fetch_one(country: str, indicator: str) -> tuple[int, dict]:
    log(f"↓ World Bank {country} {indicator}")
    url = (
        f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
        f"?format=json&per_page=20000"
    )
    try:
        payload = http_get_json(url, timeout=30)
    except Exception as e:
        log(f"  ✗ {country}/{indicator}: {e}")
        return 0, {"country": country, "indicator": indicator, "error": str(e)}
    if not isinstance(payload, list) or len(payload) < 2:
        return 0, {"country": country, "indicator": indicator, "error": "bad payload"}
    rows = []
    for item in payload[1]:
        year = item.get("date")
        v = item.get("value")
        if year is None or v is None:
            continue
        try:
            yr = int(year)
            val = float(v)
        except (TypeError, ValueError):
            continue
        dt = datetime(yr, 12, 31, tzinfo=timezone.utc)
        ns = int(dt.timestamp() * 1_000_000_000)
        rows.append([ns, country, indicator, f"{val:.6f}"])
    rows.sort(key=lambda r: r[0])
    if not rows:
        return 0, {"country": country, "indicator": indicator, "error": "no rows"}
    out_path = WB_DIR / f"{country}_{indicator.replace('.', '_')}.csv"
    ensure_dir(WB_DIR)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_nanos", "country", "indicator", "value"])
        w.writerows(rows)
    log(f"  ✓ {len(rows)} annual observations")
    return len(rows), {
        "country": country, "indicator": indicator,
        "rows": len(rows), "path": str(out_path),
    }


def main():
    args = sys.argv[1:]
    countries = args or DEFAULT_COUNTRIES
    entries = []
    total = 0
    for c in countries:
        for ind in DEFAULT_INDICATORS:
            n, e = fetch_one(c, ind)
            entries.append(e); total += n
            time.sleep(0.2)
    mpath = emit_manifest("worldbank", entries)
    log(f"\nFetched {total} World Bank observations.")
    log(f"Manifest: {mpath}")


if __name__ == "__main__":
    main()
