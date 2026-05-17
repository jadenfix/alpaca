#!/usr/bin/env python3
"""
Aggregate every bar CSV under data/{real,crypto}/ into one chronologically-
ordered universe.csv per directory, AND emit a per-symbol availability
manifest at data/manifests/inventory_<ts>.json describing the date range,
row count, and asset class of each symbol.

Usage:
    python3 scripts/data/aggregate.py
    python3 scripts/data/aggregate.py --root data/crypto
"""

from __future__ import annotations
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from _common import BAR_HEADER, MANIFEST_DIR, ensure_dir, log


def aggregate_dir(root: Path) -> dict:
    """Merge every *.csv under `root` (except universe.csv and *.cleaned.csv)
    into a single chronologically-sorted universe.csv."""
    rows = []
    by_symbol: dict[str, dict] = {}
    if not root.exists():
        return {"root": str(root), "files": 0, "rows": 0, "symbols": {}}
    for p in sorted(root.rglob("*.csv")):
        if p.name in ("universe.csv",) or p.name.endswith(".cleaned.csv"):
            continue
        with open(p) as f:
            r = csv.reader(f)
            hdr = next(r, None)
            if hdr != BAR_HEADER:
                continue
            for row in r:
                if len(row) != 8:
                    continue
                rows.append(row)
                sym = row[1]
                try:
                    ts = int(row[0])
                except ValueError:
                    continue
                d = by_symbol.setdefault(sym, {
                    "rows": 0, "min_ts": ts, "max_ts": ts, "source_files": set(),
                })
                d["rows"] += 1
                if ts < d["min_ts"]:
                    d["min_ts"] = ts
                if ts > d["max_ts"]:
                    d["max_ts"] = ts
                d["source_files"].add(str(p))
    rows.sort(key=lambda r: (int(r[0]), r[1]))
    out_path = root / "universe.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BAR_HEADER)
        w.writerows(rows)
    # Stringify source_files set for JSON
    symbols_json: dict[str, dict] = {}
    for s, d in by_symbol.items():
        symbols_json[s] = {
            "rows": d["rows"],
            "first": datetime.fromtimestamp(d["min_ts"] / 1e9, tz=timezone.utc).strftime("%Y-%m-%d"),
            "last": datetime.fromtimestamp(d["max_ts"] / 1e9, tz=timezone.utc).strftime("%Y-%m-%d"),
            "source_files": sorted(d["source_files"]),
        }
    log(f"  • {root}/universe.csv: {len(rows)} bars × {len(by_symbol)} symbols")
    return {
        "root": str(root),
        "files": sum(1 for _ in root.rglob("*.csv")),
        "rows": len(rows),
        "symbols": symbols_json,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=None,
                    help="Aggregate just this root (default: data/real and data/crypto)")
    args = ap.parse_args()
    if args.root:
        roots = [Path(args.root)]
    else:
        roots = [Path("data/real"), Path("data/crypto")]
    inventory = {}
    for r in roots:
        inv = aggregate_dir(r)
        inventory[r.name] = inv
    ensure_dir(MANIFEST_DIR)
    ts = int(time.time())
    inv_path = MANIFEST_DIR / f"inventory_{ts}.json"
    with open(inv_path, "w") as f:
        json.dump({"ts": ts, "inventory": inventory}, f, indent=2, default=str)
    log(f"\nInventory written to {inv_path}")


if __name__ == "__main__":
    main()
