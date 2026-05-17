#!/usr/bin/env python3
"""
Fetch climate / weather time series from NOAA's free Climate Data Online
service. Weather is a documented driver of energy and agricultural commodity
prices (heating-degree-days → natural gas demand; drought → corn / soy).

NOAA's official API requires a free token; many public CSV endpoints do not.
We use the Global Summary of the Day (GSOD) FTP-style HTTP mirror:
    https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/<YYYY>/<STATION>.csv

Default stations cover the major US energy-demand hubs:
  - 725300-94846  ORD (Chicago O'Hare)        — Midwest weather
  - 722950-23174  LAX (Los Angeles)           — Pacific
  - 725090-14739  BOS (Boston)                — Northeast
  - 722030-12839  MIA (Miami)                 — Southeast
  - 723070-13889  IAH (Houston)               — Gulf / oil hub

Output: data/weather/<STATION>_<YEAR>.csv with schema
    ts_nanos, station, temp_f, dewp_f, slp, wdsp_mph, prcp_in

Usage:
    python3 scripts/data/fetch_noaa.py 2024
"""

from __future__ import annotations
import csv
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

from _common import DATA_ROOT, emit_manifest, ensure_dir, http_get_text, log

WEATHER_DIR = DATA_ROOT / "weather"

DEFAULT_STATIONS = [
    ("72530094846", "ORD"),
    ("72295023174", "LAX"),
    ("72509014739", "BOS"),
    ("72202012839", "MIA"),
    ("72307013889", "IAH"),
]


def fetch_one(station_id: str, label: str, year: int) -> tuple[int, dict]:
    url = f"https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{station_id}.csv"
    log(f"↓ NOAA GSOD {label} ({station_id}) {year}")
    try:
        text = http_get_text(url, timeout=60)
    except Exception as e:
        log(f"  ✗ {label}/{year}: {e}")
        return 0, {"station": label, "year": year, "error": str(e)}
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        date_str = row.get("DATE")
        if not date_str:
            continue
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        ns = int(dt.timestamp() * 1_000_000_000)

        def f(key, missing=9999.9):
            raw = row.get(key, "")
            try:
                v = float(raw)
                if v == missing:
                    return ""
                return f"{v:.4f}"
            except ValueError:
                return ""

        rows.append([
            ns, label,
            f("TEMP"), f("DEWP"), f("SLP"),
            f("WDSP"), f("PRCP", missing=99.99),
        ])
    rows.sort(key=lambda r: r[0])
    if not rows:
        return 0, {"station": label, "year": year, "error": "no rows"}
    out_path = WEATHER_DIR / f"{label}_{year}.csv"
    ensure_dir(WEATHER_DIR)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_nanos", "station", "temp_f", "dewp_f", "slp", "wdsp_mph", "prcp_in"])
        w.writerows(rows)
    log(f"  ✓ {len(rows)} daily obs")
    return len(rows), {"station": label, "year": year, "rows": len(rows), "path": str(out_path)}


def main():
    if sys.argv[1:]:
        years = [int(x) for x in sys.argv[1:]]
    else:
        years = [datetime.now(timezone.utc).year]
    entries = []
    total = 0
    for y in years:
        for sid, label in DEFAULT_STATIONS:
            n, e = fetch_one(sid, label, y)
            entries.append(e); total += n
    mpath = emit_manifest("noaa", entries)
    log(f"\nFetched {total} weather obs across {len(years)} years × {len(DEFAULT_STATIONS)} stations.")
    log(f"Manifest: {mpath}")


if __name__ == "__main__":
    main()
