#!/usr/bin/env python3
"""
Build a single SQLite catalog of every dataset we have under data/.

The catalog answers questions like:
  - which series have we fetched?
  - from what source / under what license?
  - what's the date range and observation count?
  - when was it last refreshed?

Schema:
    CREATE TABLE datasets (
        dataset_id      TEXT PRIMARY KEY,   -- canonical id, e.g. fred:VIXCLS
        source          TEXT NOT NULL,      -- fred, yahoo, binance, …
        asset_class     TEXT,               -- equity, fx, crypto, macro, energy, …
        symbol          TEXT,               -- ticker / series name
        path            TEXT NOT NULL,      -- relative path under data/
        n_rows          INTEGER,
        first_ts_nanos  INTEGER,
        last_ts_nanos   INTEGER,
        frequency       TEXT,               -- daily, weekly, hourly, …
        license_url     TEXT,               -- where the license / TOS lives
        fetched_at      INTEGER,            -- unix seconds
        notes           TEXT
    );

Run after fetchers + validator. The catalog is itself versioned via
data/catalog.sqlite (gitignored; rebuild deterministically from CSVs).
"""

from __future__ import annotations
import csv
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from _common import DATA_ROOT, log

DB_PATH = DATA_ROOT / "catalog.sqlite"

# Map directory → (source, asset_class, frequency hint, license URL)
DIR_META = {
    "real":       ("yahoo",       "equity_etf", "daily",    "https://policies.yahoo.com/us/en/yahoo/terms/index.htm"),
    "crypto":     ("coingecko",   "crypto",     "daily",    "https://www.coingecko.com/en/api/documentation"),
    "binance":    ("binance",     "crypto",     "intraday", "https://www.binance.com/en/terms"),
    "macro":      ("fred",        "macro",      "varies",   "https://fred.stlouisfed.org/legal/"),
    "energy":     ("eia",         "commodity",  "varies",   "https://www.eia.gov/about/copyrights_reuse.php"),
    "factors":    ("famafrench",  "factor",     "daily",    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"),
    "yields":     ("ustreasury",  "rates",      "daily",    "https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics"),
    "weather":    ("noaa_gsod",   "alt_weather","daily",    "https://www.ncei.noaa.gov/about/legal"),
    "worldbank":  ("worldbank",   "macro",      "annual",   "https://datacatalog.worldbank.org/public-licenses"),
    "filings":    ("sec_edgar",   "fundamental","irregular","https://www.sec.gov/privacy.htm#dissemination"),
    "cot":        ("cftc",        "futures_positioning","weekly", "https://www.cftc.gov/About/Disclaimer/index.htm"),
    "pageviews":  ("wikimedia",   "alt_search", "daily",    "https://wikimediafoundation.org/wiki/Terms_of_Use"),
    "gdelt":      ("gdelt",       "alt_news",   "daily",    "https://blog.gdeltproject.org/about/"),
    "sentiment":  ("aaii",        "alt_sentiment","weekly", "https://www.aaii.com/legal"),
    "universe":   ("wikipedia",   "constituent","static",   "https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content"),
    "sample":     ("synthetic",   "synthetic",  "minute",   "internal"),
}


def init_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id      TEXT PRIMARY KEY,
            source          TEXT NOT NULL,
            asset_class     TEXT,
            symbol          TEXT,
            path            TEXT NOT NULL,
            n_rows          INTEGER,
            first_ts_nanos  INTEGER,
            last_ts_nanos   INTEGER,
            frequency       TEXT,
            license_url     TEXT,
            fetched_at      INTEGER,
            notes           TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_source ON datasets(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_asset_class ON datasets(asset_class)")


def scan_csv(path: Path) -> tuple[int, int, int]:
    """Return (n_rows, first_ts, last_ts). Skips header row."""
    n = 0
    first = None
    last = None
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                ts = int(row[0])
            except (ValueError, IndexError):
                continue
            if first is None or ts < first:
                first = ts
            if last is None or ts > last:
                last = ts
            n += 1
    return n, first or 0, last or 0


def categorize(path: Path) -> tuple[str, str, str, str, str]:
    """Returns (source, asset_class, frequency, license_url, symbol)."""
    parts = path.parts
    try:
        d_idx = parts.index("data")
    except ValueError:
        return "unknown", "unknown", "unknown", "", path.stem
    sub = parts[d_idx + 1] if len(parts) > d_idx + 1 else "unknown"
    meta = DIR_META.get(sub, (sub, "unknown", "unknown", ""))
    return meta[0], meta[1], meta[2], meta[3], path.stem


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"→ scanning {DATA_ROOT}/ ...")
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    cur = conn.cursor()
    cur.execute("DELETE FROM datasets")
    now = int(time.time())
    n_inserted = 0
    for p in sorted(DATA_ROOT.rglob("*.csv")):
        if p.name in ("universe.csv",):
            # universe.csv is an aggregate; skip to avoid double-counting symbols.
            continue
        try:
            n_rows, first_ts, last_ts = scan_csv(p)
        except Exception as e:
            log(f"  ⚠ {p}: {e}")
            continue
        if n_rows == 0:
            continue
        source, asset_class, freq, license_url, symbol = categorize(p)
        ds_id = f"{source}:{symbol}"
        cur.execute(
            "INSERT OR REPLACE INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ds_id, source, asset_class, symbol, str(p),
                n_rows, first_ts, last_ts, freq, license_url, now, "",
            ),
        )
        n_inserted += 1
    conn.commit()
    # Print a quick summary table.
    print(f"\nCatalogued {n_inserted} datasets → {DB_PATH}")
    print()
    print(f"{'SOURCE':<14}{'ASSET_CLASS':<18}{'N_DATASETS':>11}{'TOTAL_ROWS':>13}")
    print("-" * 56)
    for source, klass, n, total in conn.execute("""
        SELECT source, asset_class, COUNT(*), SUM(n_rows)
        FROM datasets GROUP BY source, asset_class
        ORDER BY source, asset_class
    """):
        print(f"{source:<14}{klass:<18}{n:>11}{total:>13}")
    print()
    print("Useful queries:")
    print(f"  sqlite3 {DB_PATH} 'SELECT * FROM datasets WHERE asset_class=\"crypto\"'")
    print(f"  sqlite3 {DB_PATH} 'SELECT source, COUNT(*) FROM datasets GROUP BY source'")
    conn.close()


if __name__ == "__main__":
    main()
