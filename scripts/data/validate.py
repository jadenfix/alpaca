#!/usr/bin/env python3
"""
Agentic data validator + cleaner.

Walks every bar CSV under data/{real,crypto}/ and runs a battery of
quantitative checks:

  1. Schema       : header matches the canonical bars-CSV format.
  2. Monotonicity : timestamps strictly increasing.
  3. Continuity   : detect gaps > median_gap × `gap_threshold` (configurable).
  4. Positivity   : prices > 0; volume ≥ 0.
  5. OHLC sanity  : low ≤ {open, close} ≤ high.
  6. Outliers     : flag bars whose log-return |z-score| > `outlier_z`
                    against a rolling 30-bar window.
  7. Splits       : flag close-to-close jumps > `split_ratio` as candidate
                    stock splits / dividend adjustments.
  8. Duplicates   : flag duplicate (symbol, ts) rows.

Cleaning actions (when --autofix is set):
  - Drop duplicate (symbol, ts) rows (keep the first).
  - Drop rows that fail schema / positivity / OHLC sanity.
  - Output goes to a sibling .cleaned.csv file; original is left untouched.

In all modes, every check is recorded as a structured entry in a JSON
manifest at data/manifests/validate_<ts>.json. The validator never silently
modifies the source files — `--autofix` writes alongside, not in place.

Usage:
    python3 scripts/data/validate.py                    # validate only
    python3 scripts/data/validate.py --autofix          # also write cleaned
    python3 scripts/data/validate.py --autofix --root data/real
"""

from __future__ import annotations
import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Iterator

from _common import BAR_HEADER, MANIFEST_DIR, emit_manifest, ensure_dir, log


def read_bar_rows(path: Path) -> tuple[list, list]:
    """Returns (header, rows_as_str_lists). Empty header on read failure."""
    with open(path) as f:
        r = csv.reader(f)
        header = next(r, None) or []
        rows = [row for row in r if row]
    return header, rows


def detect_gaps(timestamps: list[int], threshold: float) -> list[tuple[int, int, int]]:
    """Detect gaps where Δts > median_gap × threshold. Returns (idx, prev, curr) tuples."""
    if len(timestamps) < 3:
        return []
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    if not gaps:
        return []
    median = statistics.median(gaps)
    if median <= 0:
        return []
    out = []
    for i, g in enumerate(gaps):
        if g > median * threshold:
            out.append((i + 1, timestamps[i], timestamps[i + 1]))
    return out


def detect_outliers(closes: list[float], z_thresh: float, window: int = 30) -> list[tuple[int, float, float]]:
    """Flag closes whose log-return |z| > z_thresh against rolling window stats."""
    if len(closes) < window + 2:
        return []
    out = []
    log_rets = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0 or closes[i] <= 0:
            log_rets.append(float("nan"))
            continue
        log_rets.append(math.log(closes[i] / closes[i - 1]))
    for i in range(window, len(log_rets)):
        win = [r for r in log_rets[i - window:i] if not math.isnan(r)]
        if len(win) < window // 2:
            continue
        mu = statistics.mean(win)
        sd = statistics.pstdev(win)
        if sd <= 0:
            continue
        r = log_rets[i]
        if math.isnan(r):
            continue
        z = (r - mu) / sd
        if abs(z) > z_thresh:
            out.append((i + 1, r, z))
    return out


def detect_splits(closes: list[float], split_ratio: float = 1.5) -> list[tuple[int, float]]:
    """Flag bars where close changes by a factor of > split_ratio (likely split/div adj)."""
    out = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0 or closes[i] <= 0:
            continue
        ratio = max(closes[i] / closes[i - 1], closes[i - 1] / closes[i])
        if ratio > split_ratio:
            out.append((i, ratio))
    return out


def validate_file(path: Path, gap_threshold: float, outlier_z: float, split_ratio: float, autofix: bool) -> dict:
    entry: dict = {"path": str(path), "issues": [], "rows": 0}
    header, rows = read_bar_rows(path)
    if header != BAR_HEADER:
        entry["issues"].append({"type": "schema", "got": header, "expected": BAR_HEADER})
        return entry
    entry["rows"] = len(rows)
    if not rows:
        entry["issues"].append({"type": "empty"})
        return entry
    # Build parallel arrays
    ts_list: list[int] = []
    closes: list[float] = []
    kept: list[list[str]] = []
    dup_keys: set[tuple[str, int]] = set()
    n_dup = 0
    n_bad_schema = 0
    n_bad_pos = 0
    n_bad_ohlc = 0
    for r in rows:
        if len(r) != 8:
            n_bad_schema += 1
            continue
        try:
            ts = int(r[0])
            sym = r[1]
            o, h, lo, c, v = float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6])
        except ValueError:
            n_bad_schema += 1
            continue
        if min(o, h, lo, c) <= 0 or v < 0:
            n_bad_pos += 1
            continue
        if not (lo <= o <= h and lo <= c <= h):
            n_bad_ohlc += 1
            continue
        key = (sym, ts)
        if key in dup_keys:
            n_dup += 1
            continue
        dup_keys.add(key)
        ts_list.append(ts)
        closes.append(c)
        kept.append(r)
    if n_dup:
        entry["issues"].append({"type": "duplicate_rows", "count": n_dup})
    if n_bad_schema:
        entry["issues"].append({"type": "schema_drop", "count": n_bad_schema})
    if n_bad_pos:
        entry["issues"].append({"type": "non_positive_price_or_neg_volume", "count": n_bad_pos})
    if n_bad_ohlc:
        entry["issues"].append({"type": "ohlc_violation", "count": n_bad_ohlc})
    # Monotonicity (after dedup)
    n_non_mono = sum(1 for a, b in zip(ts_list, ts_list[1:]) if b <= a)
    if n_non_mono:
        entry["issues"].append({"type": "non_monotonic_ts", "count": n_non_mono})
    # Gaps
    gaps = detect_gaps(ts_list, gap_threshold)
    if gaps:
        entry["issues"].append({"type": "gap", "count": len(gaps), "first_gap": gaps[0]})
    # Outliers
    outliers = detect_outliers(closes, outlier_z)
    if outliers:
        entry["issues"].append({
            "type": "outlier_returns",
            "count": len(outliers),
            "first": outliers[0],
        })
    # Splits
    splits = detect_splits(closes, split_ratio)
    if splits:
        entry["issues"].append({"type": "candidate_split_or_adjustment", "count": len(splits)})
    entry["kept_rows"] = len(kept)
    entry["dropped_rows"] = entry["rows"] - len(kept)
    if autofix and (n_dup + n_bad_schema + n_bad_pos + n_bad_ohlc > 0 or n_non_mono > 0):
        out_path = path.with_suffix(".cleaned.csv")
        # Sort by ts in case of any non-monotonicity that wasn't already removed.
        kept_sorted = sorted(kept, key=lambda r: int(r[0]))
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(BAR_HEADER)
            w.writerows(kept_sorted)
        entry["cleaned_path"] = str(out_path)
    return entry


def walk_bar_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for p in root.rglob("*.csv"):
        if p.name.endswith(".cleaned.csv"):
            continue
        if p.name == "universe.csv":
            continue
        yield p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="data", help="Root directory to scan (default: data)")
    ap.add_argument("--gap-threshold", type=float, default=5.0,
                    help="Flag gaps > median_gap × this (default 5×)")
    ap.add_argument("--outlier-z", type=float, default=5.0,
                    help="Flag log-return z-scores above this (default 5σ)")
    ap.add_argument("--split-ratio", type=float, default=1.5,
                    help="Flag close-to-close moves with ratio above this (default 1.5)")
    ap.add_argument("--autofix", action="store_true",
                    help="Write cleaned .cleaned.csv alongside flagged source files")
    args = ap.parse_args()
    root = Path(args.root)
    entries = []
    n_files = 0
    n_clean = 0
    n_with_issues = 0
    for p in sorted(walk_bar_files(root)):
        # Only validate bar-shaped CSVs (presence of canonical header).
        with open(p) as f:
            first = next(csv.reader(f), None)
        if first != BAR_HEADER:
            continue
        e = validate_file(p, args.gap_threshold, args.outlier_z, args.split_ratio, args.autofix)
        entries.append(e)
        n_files += 1
        if e["issues"]:
            n_with_issues += 1
            log(f"  ⚠ {p.name}: {len(e['issues'])} issues, dropped {e.get('dropped_rows', 0)}/{e['rows']} rows")
        else:
            n_clean += 1
    mpath = emit_manifest("validate", entries)
    log(f"\nValidated {n_files} files: {n_clean} clean, {n_with_issues} with issues.")
    log(f"Manifest: {mpath}")


if __name__ == "__main__":
    main()
