#!/usr/bin/env python3
"""
Fetch Fama-French factor returns from the Dartmouth library.

Datasets pulled by default:
  - F-F_Research_Data_Factors_daily        Mkt-RF, SMB, HML, RF (daily)
  - F-F_Momentum_Factor_daily              Mom (daily)
  - F-F_Research_Data_5_Factors_2x3_daily  Mkt-RF, SMB, HML, RMW, CMA, RF (daily)

No API key required; the library publishes the data as ZIP files.

Output: data/factors/<DATASET>.csv with schema
    ts_nanos, factor, value      (value in decimal — input is in percent)

Example:
    python3 scripts/data/fetch_famafrench.py
"""

from __future__ import annotations
import csv
import io
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from _common import FACTORS_DIR, emit_manifest, ensure_dir, http_get_bytes, log

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

DEFAULTS = [
    ("F-F_Research_Data_Factors_daily", f"{BASE}/F-F_Research_Data_Factors_daily_CSV.zip"),
    ("F-F_Momentum_Factor_daily",        f"{BASE}/F-F_Momentum_Factor_daily_CSV.zip"),
    ("F-F_Research_Data_5_Factors_2x3_daily", f"{BASE}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"),
]


def parse_ff_csv(text: str) -> list[tuple[str, dict[str, float]]]:
    """
    The library's daily CSVs have a preamble of explanatory text, then a
    table of (YYYYMMDD, factor_1, factor_2, ...) rows, sometimes followed
    by a second annual table after a blank line. We parse only the first
    daily section.
    """
    lines = text.splitlines()
    header_idx = None
    for i, ln in enumerate(lines):
        # Header line begins with a comma (date column unnamed) then named factors.
        if ln.strip().startswith(","):
            cols = [c.strip() for c in ln.split(",")]
            if all(re.fullmatch(r"[A-Za-z\-]+", c) for c in cols[1:]) and len(cols) > 1:
                header_idx = i
                break
    if header_idx is None:
        return []
    header = [c.strip() for c in lines[header_idx].split(",")]
    factor_names = header[1:]
    out = []
    for ln in lines[header_idx + 1:]:
        ln = ln.strip()
        if not ln:
            break
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(header):
            continue
        date_token = parts[0]
        if not re.fullmatch(r"\d{8}", date_token):
            break  # second section starts, stop.
        try:
            vals = {factor_names[i]: float(parts[i + 1]) for i in range(len(factor_names))}
        except ValueError:
            continue
        out.append((date_token, vals))
    return out


def fetch_one(name: str, url: str) -> tuple[int, dict]:
    try:
        z_bytes = http_get_bytes(url)
    except Exception as e:
        log(f"  ✗ {name}: download failed ({e})")
        return 0, {"dataset": name, "error": str(e)}
    try:
        with zipfile.ZipFile(io.BytesIO(z_bytes)) as zf:
            csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            text = zf.read(csv_name).decode("utf-8", errors="replace")
    except Exception as e:
        log(f"  ✗ {name}: unzip failed ({e})")
        return 0, {"dataset": name, "error": f"unzip: {e}"}

    rows = parse_ff_csv(text)
    if not rows:
        return 0, {"dataset": name, "error": "no parsed rows"}

    out_rows = []
    for date_token, vals in rows:
        dt = datetime.strptime(date_token, "%Y%m%d").replace(tzinfo=timezone.utc)
        ns = int(dt.timestamp() * 1_000_000_000)
        for factor, v in vals.items():
            # FF reports in percent; convert to decimal.
            out_rows.append([ns, factor, f"{v / 100.0:.10f}"])
    out_path = FACTORS_DIR / f"{name}.csv"
    ensure_dir(out_path.parent)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_nanos", "factor", "value"])
        w.writerows(out_rows)
    log(f"  ✓ {len(out_rows)} factor-observations ({len(rows)} dates × {len(rows[0][1])} factors)")
    return len(out_rows), {
        "dataset": name,
        "path": str(out_path),
        "rows": len(out_rows),
        "first_date": rows[0][0],
        "last_date": rows[-1][0],
        "factors": list(rows[0][1].keys()),
    }


def main():
    total = 0
    entries = []
    for name, url in DEFAULTS:
        log(f"↓ FF {name}")
        n, e = fetch_one(name, url)
        entries.append(e)
        total += n
    mpath = emit_manifest("famafrench", entries)
    log(f"\nFetched {total} factor-observations across {len(DEFAULTS)} datasets.")
    log(f"Manifest: {mpath}")


if __name__ == "__main__":
    main()
