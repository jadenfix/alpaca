#!/usr/bin/env python3
"""
Fetch CFTC Commitments-of-Traders (COT) weekly futures positioning data.

The CFTC publishes "Disaggregated" reports weekly as plain-text TXT files
on their public website. We download the latest year-end aggregated file
("annual.txt") and parse positions for a curated futures contract list.

Source (free, no key):
    https://www.cftc.gov/sites/default/files/files/dea/history/dea_fut_disagg_xls_<YEAR>.zip

We parse the Excel-as-text (`.txt` inside the ZIP) since reading XLS
without external deps is harder. We focus on the columns:
    Market_and_Exchange_Names
    Report_Date_as_YYYY-MM-DD
    M_Money_Positions_Long_All       (managed money long)
    M_Money_Positions_Short_All      (managed money short)
    Prod_Merc_Positions_Long_ALL     (commercial long)
    Prod_Merc_Positions_Short_ALL    (commercial short)

Output: data/cot/<CONTRACT>.csv with columns
    ts_nanos, contract, mm_long, mm_short, prod_long, prod_short, mm_net,
    prod_net

`mm_net` and `prod_net` are managed-money-net and commercial-net, the most
commonly-used signal columns.

Usage:
    python3 scripts/data/fetch_cftc_cot.py 2024
"""

from __future__ import annotations
import csv
import io
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from _common import COT_DIR, emit_manifest, ensure_dir, http_get_bytes, log

# Default contracts to extract (substring match on Market_and_Exchange_Names).
DEFAULT_CONTRACTS = [
    "GOLD",
    "SILVER",
    "WTI-PHYSICAL",   # WTI crude
    "NATURAL GAS",
    "10-YEAR U.S. TREASURY",
    "S&P 500",
    "RUSSELL",
    "VIX FUTURES",
    "EURO FX",
    "BITCOIN",
]


def parse_cot_text(text: str, contract_substrings: list[str]) -> dict[str, list[list]]:
    """Returns map contract → list of CSV rows."""
    reader = csv.DictReader(io.StringIO(text))
    out: dict[str, list[list]] = {}
    for row in reader:
        name = row.get("Market_and_Exchange_Names", "").upper()
        match = None
        for sub in contract_substrings:
            if sub.upper() in name:
                match = sub
                break
        if not match:
            continue
        date_str = row.get("Report_Date_as_YYYY-MM-DD") or row.get("Report_Date_as_MM_DD_YYYY")
        if not date_str:
            continue
        try:
            if "-" in date_str:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            else:
                dt = datetime.strptime(date_str, "%m/%d/%Y")
            dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        ns = int(dt.timestamp() * 1_000_000_000)
        try:
            mm_long = float(row.get("M_Money_Positions_Long_All", "0") or 0)
            mm_short = float(row.get("M_Money_Positions_Short_All", "0") or 0)
            prod_long = float(row.get("Prod_Merc_Positions_Long_ALL", "0") or 0)
            prod_short = float(row.get("Prod_Merc_Positions_Short_ALL", "0") or 0)
        except ValueError:
            continue
        mm_net = mm_long - mm_short
        prod_net = prod_long - prod_short
        out.setdefault(match, []).append([
            ns, match, mm_long, mm_short, prod_long, prod_short, mm_net, prod_net,
        ])
    return out


def fetch_year(year: int, contracts: list[str]) -> tuple[int, dict]:
    url = f"https://www.cftc.gov/sites/default/files/files/dea/history/fut_disagg_txt_{year}.zip"
    log(f"↓ CFTC COT {year}")
    try:
        z_bytes = http_get_bytes(url, timeout=60)
    except Exception as e:
        log(f"  ✗ {year}: download failed ({e})")
        return 0, {"year": year, "error": str(e)}
    try:
        with zipfile.ZipFile(io.BytesIO(z_bytes)) as zf:
            txt_name = [n for n in zf.namelist() if n.lower().endswith(".txt")][0]
            text = zf.read(txt_name).decode("utf-8", errors="replace")
    except Exception as e:
        log(f"  ✗ {year}: unzip failed ({e})")
        return 0, {"year": year, "error": f"unzip: {e}"}

    parsed = parse_cot_text(text, contracts)
    ensure_dir(COT_DIR)
    total = 0
    for contract, rows in parsed.items():
        rows.sort(key=lambda r: r[0])
        slug = re.sub(r"[^A-Z0-9]+", "_", contract.upper()).strip("_")
        path = COT_DIR / f"{slug}.csv"
        # Append if the file exists from an earlier year.
        exists = path.exists()
        mode = "a" if exists else "w"
        with open(path, mode, newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow([
                    "ts_nanos", "contract", "mm_long", "mm_short",
                    "prod_long", "prod_short", "mm_net", "prod_net",
                ])
            w.writerows(rows)
        log(f"  ✓ {contract}: {len(rows)} weekly observations → {path.name}")
        total += len(rows)
    return total, {"year": year, "contracts": len(parsed), "rows": total}


def main():
    years = [int(x) for x in (sys.argv[1:] or ["2024"])]
    entries = []
    total = 0
    for y in years:
        n, e = fetch_year(y, DEFAULT_CONTRACTS)
        entries.append(e)
        total += n
    mpath = emit_manifest("cftc_cot", entries)
    log(f"\nFetched {total} weekly COT observations across {len(years)} years.")
    log(f"Manifest: {mpath}")


if __name__ == "__main__":
    main()
