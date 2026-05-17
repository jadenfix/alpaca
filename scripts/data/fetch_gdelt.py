#!/usr/bin/env python3
"""
Fetch the GDELT Global Knowledge Graph daily summary export.

GDELT (https://gdeltproject.org) monitors every broadcast, print, and web
news source worldwide and classifies events. The free CSV exports include
a daily count of articles by tone, theme, and goldstein score (a -10..+10
conflict scale).

We use the Doc 2.0 free CSV download:
    http://data.gdeltproject.org/gdeltv2/<YYYYMMDDHHMMSS>.export.CSV.zip
Each file covers a 15-minute window of events globally; we aggregate per
day into a single summary line per day:
    date, n_events, avg_tone, avg_goldstein

For long-range work the daily masterfilelist allows us to pull historical
days; we default to the last 14 days as a smoke-test.

Output: data/gdelt/daily_summary.csv with schema
    ts_nanos, n_events, avg_tone, avg_goldstein

Reference: Leetaru & Schrodt (2013), "GDELT: Global Data on Events,
Location and Tone, 1979-2012".
"""

from __future__ import annotations
import csv
import io
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from _common import DATA_ROOT, emit_manifest, ensure_dir, http_get_bytes, log

GDELT_DIR = DATA_ROOT / "gdelt"

# Hardcoded subset of column indices in the v2.0 export (0-based).
COL_DATE = 1            # SQLDATE  YYYYMMDD
COL_GOLDSTEIN = 30      # GoldsteinScale
COL_AVG_TONE = 34       # AvgTone


def fetch_one_window(timestamp_str: str) -> list[tuple[int, float, float]]:
    """Fetch a 15-min window export; return (date_int, goldstein, tone) rows."""
    url = f"http://data.gdeltproject.org/gdeltv2/{timestamp_str}.export.CSV.zip"
    try:
        z = http_get_bytes(url, timeout=60)
    except Exception as e:
        return []
    rows: list[tuple[int, float, float]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(z)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as f:
                for line in f:
                    cols = line.decode("utf-8", errors="replace").rstrip("\n").split("\t")
                    if len(cols) < max(COL_DATE, COL_GOLDSTEIN, COL_AVG_TONE) + 1:
                        continue
                    try:
                        d = int(cols[COL_DATE])
                        g = float(cols[COL_GOLDSTEIN]) if cols[COL_GOLDSTEIN] else 0.0
                        t = float(cols[COL_AVG_TONE]) if cols[COL_AVG_TONE] else 0.0
                    except ValueError:
                        continue
                    rows.append((d, g, t))
    except Exception as e:
        return []
    return rows


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    log(f"↓ GDELT global events ({days} days, last 15-min window per day)")
    ensure_dir(GDELT_DIR)
    end = datetime.now(timezone.utc)
    daily_acc: dict[int, list[tuple[float, float]]] = {}
    fetched = 0
    for i in range(days):
        day = end - timedelta(days=i + 1)
        # Take a single window (00:15) per day to keep this script fast.
        ts_str = day.strftime("%Y%m%d") + "001500"
        rows = fetch_one_window(ts_str)
        if rows:
            fetched += 1
            for d, g, t in rows:
                daily_acc.setdefault(d, []).append((g, t))
        time.sleep(0.4)
    if not daily_acc:
        log("  ⚠ no GDELT windows could be fetched (rate limit or network)")
        emit_manifest("gdelt", [{"days": days, "error": "no rows fetched"}])
        return
    out = []
    for d, vals in sorted(daily_acc.items()):
        n = len(vals)
        avg_g = sum(g for g, _ in vals) / n
        avg_t = sum(t for _, t in vals) / n
        dt = datetime.strptime(str(d), "%Y%m%d").replace(tzinfo=timezone.utc)
        ns = int(dt.timestamp() * 1_000_000_000)
        out.append([ns, n, f"{avg_t:.6f}", f"{avg_g:.6f}"])
    path = GDELT_DIR / "daily_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_nanos", "n_events", "avg_tone", "avg_goldstein"])
        w.writerows(out)
    log(f"  ✓ {len(out)} daily summaries from {fetched} windows")
    emit_manifest("gdelt", [{"days": days, "rows": len(out), "path": str(path)}])


if __name__ == "__main__":
    main()
