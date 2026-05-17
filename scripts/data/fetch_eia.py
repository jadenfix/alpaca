#!/usr/bin/env python3
"""
Fetch energy time series from the EIA (US Energy Information Administration)
free Open Data API.

We use the v2 endpoint:
    https://api.eia.gov/v2/<route>/data/?api_key=<KEY>&...
Set EIA_API_KEY environment variable (free at https://www.eia.gov/opendata/register.php).
If unset, we fall back to the legacy CSV download endpoints (cushing crude
spot price + weekly petroleum stocks) which need no key.

Series fetched by default (impactful for energy / commodity strategies):
  - PET.WCESTUS1.W         Weekly US crude oil ending stocks (kbbl)
  - PET.RWTC.D             Daily WTI crude price (Cushing)
  - PET.RBRTE.D            Daily Brent crude price
  - NG.RNGWHHD.D           Daily Henry Hub natural gas spot
  - ELEC.GEN.ALL-US-99.M   Monthly total US electricity generation

Output: data/energy/<SERIES>.csv with schema
    ts_nanos, series, value

Usage:
    EIA_API_KEY=... python3 scripts/data/fetch_eia.py
    python3 scripts/data/fetch_eia.py            # CSV fallback for a couple of series
"""

from __future__ import annotations
import csv
import io
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from _common import DATA_ROOT, emit_manifest, ensure_dir, http_get_json, http_get_text, log

ENERGY_DIR = DATA_ROOT / "energy"

# (route, series_id, frequency)
DEFAULT_SERIES = [
    ("petroleum/stoc/wstk", "WCESTUS1", "weekly"),
    ("petroleum/pri/spt",   "RWTC",     "daily"),
    ("petroleum/pri/spt",   "RBRTE",    "daily"),
    ("natural-gas/pri/sum", "RNGWHHD",  "daily"),
]

CSV_FALLBACKS = {
    # Cushing WTI daily spot price (no key required)
    "RWTC": "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.csv",
    "RBRTE": "https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.csv",
    "RNGWHHD": "https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.csv",
}


def parse_eia_legacy_csv(text: str, series_id: str) -> list[list]:
    """The EIA legacy XLS-as-CSV has 4 header rows + (date, value)."""
    rows = []
    reader = csv.reader(io.StringIO(text))
    started = False
    for row in reader:
        if not row or len(row) < 2:
            continue
        if not started:
            # Header rows look like "Sourcekey","RWTC" / "Date","Cushing, OK..."
            # We treat the first row whose first cell parses as a date as data.
            try:
                datetime.strptime(row[0], "%m/%d/%Y")
                started = True
            except ValueError:
                # Try alternate formats before giving up.
                try:
                    datetime.strptime(row[0], "%Y-%m-%d")
                    started = True
                except ValueError:
                    continue
        if not started:
            continue
        date_str = row[0]
        for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d-%b-%y"):
            try:
                dt = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                dt = None
        if dt is None:
            continue
        try:
            v = float(row[1])
        except ValueError:
            continue
        ns = int(dt.timestamp() * 1_000_000_000)
        rows.append([ns, series_id, f"{v:.6f}"])
    rows.sort(key=lambda r: r[0])
    return rows


def fetch_via_api(route: str, series_id: str, freq: str, key: str) -> list[list]:
    url = (
        f"https://api.eia.gov/v2/{route}/data/"
        f"?api_key={key}&frequency={freq}&data[0]=value"
        f"&facets[series][]={series_id}&sort[0][column]=period&sort[0][direction]=asc&length=5000"
    )
    data = http_get_json(url, timeout=60)
    obs = data.get("response", {}).get("data", [])
    out = []
    for o in obs:
        period = o.get("period")
        val = o.get("value")
        if period is None or val is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        # period may be YYYY, YYYY-Qn, YYYY-MM, YYYY-MM-DD, YYYY-MM-DDTHH
        ts = None
        for fmt in ("%Y-%m-%dT%H", "%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                ts = datetime.strptime(period, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        if ts is None:
            continue
        out.append([int(ts.timestamp() * 1_000_000_000), series_id, f"{v:.6f}"])
    out.sort(key=lambda r: r[0])
    return out


def fetch_one(route: str, series_id: str, freq: str) -> tuple[int, dict]:
    log(f"↓ EIA {series_id} ({freq})")
    key = os.environ.get("EIA_API_KEY")
    rows: list[list] = []
    via = "api"
    if key:
        try:
            rows = fetch_via_api(route, series_id, freq, key)
        except Exception as e:
            log(f"  ⚠ API fetch failed ({e}), trying legacy CSV")
            rows = []
    if not rows and series_id in CSV_FALLBACKS:
        via = "csv-fallback"
        try:
            text = http_get_text(CSV_FALLBACKS[series_id], timeout=60)
            rows = parse_eia_legacy_csv(text, series_id)
        except Exception as e:
            log(f"  ✗ CSV fallback failed ({e})")
            return 0, {"series": series_id, "error": str(e)}
    if not rows:
        return 0, {"series": series_id, "error": "no rows; set EIA_API_KEY"}
    out_path = ENERGY_DIR / f"{series_id}.csv"
    ensure_dir(ENERGY_DIR)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_nanos", "series", "value"])
        w.writerows(rows)
    log(f"  ✓ {len(rows)} observations via {via}")
    return len(rows), {
        "series": series_id, "route": route, "freq": freq,
        "via": via, "rows": len(rows), "path": str(out_path),
    }


def main():
    entries = []
    total = 0
    for route, sid, freq in DEFAULT_SERIES:
        n, e = fetch_one(route, sid, freq)
        entries.append(e); total += n
    mpath = emit_manifest("eia", entries)
    log(f"\nFetched {total} EIA observations across {len(DEFAULT_SERIES)} series.")
    log(f"Manifest: {mpath}")


if __name__ == "__main__":
    main()
