#!/usr/bin/env python3
"""
Scrape constituent lists from public Wikipedia pages.

Supported lists:
  - S&P 500    https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
  - Nasdaq-100 https://en.wikipedia.org/wiki/Nasdaq-100
  - Dow Jones  https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average

Output: data/universe/<NAME>.csv with schema
    symbol, name, sector, weight     (weight is empty unless the page exposes it)

Wikipedia is scraped because there is no free machine-readable API. We use
a regex-based HTML parser (no external dependencies). Wikipedia's HTML for
these tables has been stable for years; we tolerate minor variation.

Usage:
    python3 scripts/data/fetch_wikipedia.py
    python3 scripts/data/fetch_wikipedia.py sp500
"""

from __future__ import annotations
import re
import sys
from html import unescape

from _common import UNIVERSE_DIR, emit_manifest, ensure_dir, http_get_text, log, write_universe_csv

LISTS = {
    "sp500": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "ticker_col": 0,
        "name_col": 1,
        "sector_col": 2,
    },
    "nasdaq100": {
        "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
        # The constituents table on this page has these column positions.
        "ticker_col": 1,
        "name_col": 0,
        "sector_col": 2,
    },
    "dow30": {
        "url": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
        "ticker_col": 2,
        "name_col": 0,
        "sector_col": 3,
    },
}

TABLE_RE = re.compile(r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>(.*?)</table>', re.IGNORECASE | re.DOTALL)
ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def strip_html(s: str) -> str:
    return unescape(TAG_RE.sub("", s)).strip().replace(" ", " ").replace("\n", " ").strip()


def fetch_one(name: str, cfg: dict) -> tuple[int, dict]:
    log(f"↓ Wikipedia {name}")
    try:
        html = http_get_text(cfg["url"])
    except Exception as e:
        log(f"  ✗ {name}: fetch failed ({e})")
        return 0, {"list": name, "error": str(e)}
    tables = TABLE_RE.findall(html)
    if not tables:
        return 0, {"list": name, "error": "no wikitable found"}
    # Try each wikitable until one yields a plausible constituent list.
    best_rows: list[list[str]] = []
    for tbl in tables:
        rows: list[list[str]] = []
        for row_html in ROW_RE.findall(tbl):
            cells = [strip_html(c) for c in CELL_RE.findall(row_html)]
            if len(cells) >= max(cfg["ticker_col"], cfg["name_col"], cfg["sector_col"]) + 1:
                rows.append(cells)
        # Heuristic: a constituent table has 20+ rows whose ticker column looks like a ticker.
        candidate = []
        for cells in rows:
            tic = cells[cfg["ticker_col"]].split()[0] if cells[cfg["ticker_col"]] else ""
            if not re.fullmatch(r"[A-Z][A-Z0-9\.\-]{0,6}", tic):
                continue
            candidate.append([
                tic,
                cells[cfg["name_col"]],
                cells[cfg["sector_col"]],
                "",
            ])
        if len(candidate) > len(best_rows):
            best_rows = candidate
    if not best_rows:
        return 0, {"list": name, "error": "no plausible rows after filtering"}
    out_path = UNIVERSE_DIR / f"{name}.csv"
    n = write_universe_csv(out_path, best_rows)
    log(f"  ✓ {n} constituents → {out_path}")
    return n, {"list": name, "path": str(out_path), "rows": n}


def main():
    chosen = sys.argv[1:] or list(LISTS.keys())
    ensure_dir(UNIVERSE_DIR)
    entries = []
    total = 0
    for name in chosen:
        cfg = LISTS.get(name)
        if not cfg:
            log(f"  ⚠ unknown list: {name}")
            continue
        n, e = fetch_one(name, cfg)
        entries.append(e)
        total += n
    mpath = emit_manifest("wikipedia", entries)
    log(f"\nFetched {total} constituents across {len(chosen)} lists.")
    log(f"Manifest: {mpath}")


if __name__ == "__main__":
    main()
