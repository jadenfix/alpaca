#!/usr/bin/env python3
"""
Fetch macroeconomic time series from FRED (Federal Reserve Economic Data) via
the unauthenticated CSV-download endpoint:
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>

No API key required. Output: data/macro/<SERIES_ID>.csv with schema
    ts_nanos, series, value

Default series cover the macro signals most useful for regime detection and
factor models:
  - VIXCLS      CBOE VIX (implied vol)
  - DGS10       10-year Treasury yield
  - DGS2        2-year Treasury yield
  - T10Y2Y      10Y-2Y spread (recession indicator)
  - UNRATE      Unemployment rate
  - CPIAUCSL    CPI inflation
  - GDP         Nominal GDP (quarterly)
  - M2SL        Money supply M2
  - FEDFUNDS    Federal funds rate
  - DEXUSEU     USD/EUR exchange rate
  - DCOILWTICO  WTI crude oil price
  - SP500       S&P 500 index level

Usage:
    python3 scripts/data/fetch_fred.py
    python3 scripts/data/fetch_fred.py VIXCLS DGS10
"""

from __future__ import annotations
import csv
import io
import sys
from pathlib import Path

from _common import (
    MACRO_DIR, date_to_ns, emit_manifest, ensure_dir, http_get_text, log, write_macro_csv,
)

DEFAULT_SERIES = [
    "VIXCLS", "DGS10", "DGS2", "T10Y2Y", "UNRATE", "CPIAUCSL",
    "GDP", "M2SL", "FEDFUNDS", "DEXUSEU", "DCOILWTICO", "SP500",
]


def fetch_one(series_id: str) -> tuple[int, dict]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        text = http_get_text(url)
    except Exception as e:
        log(f"  ✗ {series_id}: download failed ({e})")
        return 0, {"series": series_id, "error": str(e)}
    rows = []
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if not header:
        return 0, {"series": series_id, "error": "empty response"}
    # FRED CSV columns: observation_date, <SERIES_ID>
    value_col = 1
    n_total = 0
    n_skipped = 0
    for row in reader:
        if not row or len(row) < 2:
            continue
        date_str = row[0]
        raw = row[value_col]
        if raw in ("", ".", "NA"):
            n_skipped += 1
            continue
        try:
            value = float(raw)
        except ValueError:
            n_skipped += 1
            continue
        try:
            ns = date_to_ns(date_str)
        except ValueError:
            n_skipped += 1
            continue
        rows.append([ns, series_id, f"{value:.6f}"])
        n_total += 1
    if not rows:
        return 0, {"series": series_id, "error": "no valid rows"}
    out_path = MACRO_DIR / f"{series_id}.csv"
    n_written = write_macro_csv(out_path, rows)
    return n_written, {
        "series": series_id,
        "path": str(out_path),
        "rows": n_written,
        "skipped": n_skipped,
        "first_ts_nanos": rows[0][0],
        "last_ts_nanos": rows[-1][0],
    }


def main():
    series = sys.argv[1:] or DEFAULT_SERIES
    ensure_dir(MACRO_DIR)
    entries = []
    total = 0
    for sid in series:
        log(f"↓ FRED {sid}")
        n, manifest_entry = fetch_one(sid)
        if n > 0:
            log(f"  ✓ {n} rows")
        entries.append(manifest_entry)
        total += n
    mpath = emit_manifest("fred", entries)
    log(f"\nFetched {total} total observations across {len(series)} series.")
    log(f"Manifest: {mpath}")


if __name__ == "__main__":
    main()
