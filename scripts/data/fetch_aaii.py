#!/usr/bin/env python3
"""
Fetch the AAII Investor Sentiment Survey (weekly bull / neutral / bear).
The bull-bear spread is a well-known contrarian indicator.

Public CSV (no key, no login):
    https://www.aaii.com/files/surveys/sentiment.xls
We use the .xls only as fallback; the CSV-compatible "historical data"
download page exposes a clean CSV at:
    https://www.aaii.com/sentimentsurvey/sent_results

Many AAII endpoints have started gating behind login; we try the most
permissive ones and fall back to scraping the public chart page that
embeds the same data as JSON.

Output: data/sentiment/aaii.csv with schema
    ts_nanos, bull_pct, neutral_pct, bear_pct, bull_bear_spread

This script is best-effort: if the source is unreachable, it logs a
warning and exits 0 so the pipeline keeps running.
"""

from __future__ import annotations
import csv
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from _common import DATA_ROOT, emit_manifest, ensure_dir, http_get_text, log

SENT_DIR = DATA_ROOT / "sentiment"

ENDPOINTS = [
    "https://www.aaii.com/files/surveys/sentiment.xls",  # XLS — we won't parse
    "https://www.aaii.com/sentimentsurvey/sent_results",  # HTML — we'll grep
]


def try_csv_from_html(url: str) -> list[list]:
    """Best-effort: many AAII pages embed JSON arrays of [date, bull, neut, bear]."""
    try:
        text = http_get_text(url, timeout=30)
    except Exception as e:
        log(f"  ⚠ {url}: {e}")
        return []
    # Look for a JSON array of [["YYYY-MM-DD", float, float, float], ...]
    matches = re.findall(
        r'\[\s*"(\d{4}-\d{2}-\d{2})"\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]',
        text,
    )
    out = []
    for date_str, b, n, br in matches:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            bull, neut, bear = float(b), float(n), float(br)
        except ValueError:
            continue
        ns = int(dt.timestamp() * 1_000_000_000)
        spread = bull - bear
        out.append([ns, f"{bull:.4f}", f"{neut:.4f}", f"{bear:.4f}", f"{spread:.4f}"])
    out.sort(key=lambda r: r[0])
    return out


def main():
    ensure_dir(SENT_DIR)
    log("↓ AAII sentiment (best-effort)")
    rows: list[list] = []
    for url in ENDPOINTS[1:]:  # skip XLS
        rows = try_csv_from_html(url)
        if rows:
            break
    entries = []
    if not rows:
        log("  ⚠ AAII sentiment not extractable without manual upload; skipping.")
        entries.append({"source": "aaii", "error": "not extractable; see https://www.aaii.com/sentimentsurvey"})
        emit_manifest("aaii", entries)
        return 0
    out_path = SENT_DIR / "aaii.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_nanos", "bull_pct", "neutral_pct", "bear_pct", "bull_bear_spread"])
        w.writerows(rows)
    log(f"  ✓ {len(rows)} weekly observations")
    entries.append({"source": "aaii", "rows": len(rows), "path": str(out_path)})
    mpath = emit_manifest("aaii", entries)
    log(f"\nManifest: {mpath}")


if __name__ == "__main__":
    main()
