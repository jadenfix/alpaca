#!/usr/bin/env python3
"""
Fetch the full daily US Treasury yield curve from Treasury Direct's public
XML feed (no key required, no rate limit in practice).

Source:
    https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/<YYYY>
    ?type=daily_treasury_yield_curve

Output: data/yields/treasury_<YEAR>.csv with schema
    ts_nanos, tenor, yield_pct
Tenors:  1Mo 2Mo 3Mo 4Mo 6Mo 1Yr 2Yr 3Yr 5Yr 7Yr 10Yr 20Yr 30Yr

The yield curve is the single most important macro time series for fixed
income, regime detection, and equity factor models. Daily values back to
1990 are free.

Usage:
    python3 scripts/data/fetch_treasury_yields.py 2024 2025
    python3 scripts/data/fetch_treasury_yields.py         # current year
"""

from __future__ import annotations
import csv
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

from _common import DATA_ROOT, emit_manifest, ensure_dir, http_get_text, log

YIELDS_DIR = DATA_ROOT / "yields"

TENOR_HEADERS = [
    "1 Mo", "2 Mo", "3 Mo", "4 Mo", "6 Mo",
    "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr",
    "10 Yr", "20 Yr", "30 Yr",
]


def fetch_year(year: int) -> tuple[int, dict]:
    url = (
        f"https://home.treasury.gov/resource-center/data-chart-center/"
        f"interest-rates/daily-treasury-rates.csv/all/{year}"
        f"?type=daily_treasury_yield_curve&field_tdr_date_value={year}"
    )
    log(f"↓ Treasury yields {year}")
    try:
        text = http_get_text(url, timeout=60)
    except Exception as e:
        log(f"  ✗ {year}: download failed ({e})")
        return 0, {"year": year, "error": str(e)}
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        date_str = row.get("Date")
        if not date_str:
            continue
        try:
            dt = datetime.strptime(date_str, "%m/%d/%Y").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        ns = int(dt.timestamp() * 1_000_000_000)
        for tenor in TENOR_HEADERS:
            raw = row.get(tenor, "").strip()
            if not raw or raw == "N/A":
                continue
            try:
                y = float(raw)
            except ValueError:
                continue
            tenor_clean = tenor.replace(" ", "")
            rows.append([ns, tenor_clean, f"{y:.4f}"])
    if not rows:
        return 0, {"year": year, "error": "no parsed rows"}
    rows.sort(key=lambda r: (r[0], r[1]))
    out_path = YIELDS_DIR / f"treasury_{year}.csv"
    ensure_dir(YIELDS_DIR)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_nanos", "tenor", "yield_pct"])
        w.writerows(rows)
    log(f"  ✓ {len(rows)} (date × tenor) observations")
    return len(rows), {"year": year, "rows": len(rows), "path": str(out_path)}


def main():
    if sys.argv[1:]:
        years = [int(x) for x in sys.argv[1:]]
    else:
        years = [datetime.now(timezone.utc).year]
    entries = []
    total = 0
    for y in years:
        n, e = fetch_year(y)
        entries.append(e); total += n
    mpath = emit_manifest("treasury_yields", entries)
    log(f"\nFetched {total} yield-curve observations across {len(years)} years.")
    log(f"Manifest: {mpath}")


if __name__ == "__main__":
    main()
