#!/usr/bin/env python3
"""
Fetch daily Wikipedia pageviews via Wikimedia's free REST API. Pageview
interest is a documented leading indicator for retail-driven trading in
single-name stocks ("Wisdom of Crowds and Markets", Da/Engelberg/Gao 2011;
"FEARS Index", Da et al. 2014).

Endpoint (no key required, no rate limit beyond polite use):
    https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/
       en.wikipedia/all-access/all-agents/<ARTICLE>/daily/<START>/<END>

Output: data/pageviews/<ARTICLE>.csv with schema
    ts_nanos, article, views

Useful articles: a ticker's Wikipedia entry, "Recession", "Cryptocurrency",
"Stock_market_crash", "Federal_Reserve", and so on — anything whose search
intensity moves macro narratives.

Usage:
    python3 scripts/data/fetch_wiki_pageviews.py
    python3 scripts/data/fetch_wiki_pageviews.py Apple_Inc. Microsoft Tesla,_Inc.
"""

from __future__ import annotations
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from _common import DATA_ROOT, emit_manifest, ensure_dir, http_get_json, log

PAGEVIEWS_DIR = DATA_ROOT / "pageviews"

DEFAULT_ARTICLES = [
    "Apple_Inc.", "Microsoft", "Amazon_(company)", "Tesla,_Inc.", "NVIDIA",
    "Bitcoin", "Ethereum",
    "Recession", "Inflation", "Federal_Reserve",
    "S%26P_500", "Stock_market_crash",
]


def fetch_one(article: str, days: int = 730) -> tuple[int, dict]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia/all-access/all-agents/{article}/daily/{start_str}/{end_str}"
    )
    log(f"↓ Wikipedia pageviews {article}")
    try:
        data = http_get_json(url, timeout=30)
    except Exception as e:
        log(f"  ✗ {article}: {e}")
        return 0, {"article": article, "error": str(e)}
    items = data.get("items", [])
    rows = []
    for it in items:
        ts = it.get("timestamp", "")
        v = it.get("views")
        if not ts or v is None or len(ts) < 10:
            continue
        try:
            dt = datetime.strptime(ts[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        ns = int(dt.timestamp() * 1_000_000_000)
        rows.append([ns, article, int(v)])
    rows.sort(key=lambda r: r[0])
    if not rows:
        return 0, {"article": article, "error": "no rows"}
    safe_name = article.replace("/", "_").replace("%26", "and").replace("%2F", "_")
    out_path = PAGEVIEWS_DIR / f"{safe_name}.csv"
    ensure_dir(PAGEVIEWS_DIR)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_nanos", "article", "views"])
        w.writerows(rows)
    log(f"  ✓ {len(rows)} daily pageview records")
    return len(rows), {"article": article, "rows": len(rows), "path": str(out_path)}


def main():
    articles = sys.argv[1:] or DEFAULT_ARTICLES
    entries = []
    total = 0
    for a in articles:
        n, e = fetch_one(a)
        entries.append(e); total += n
        time.sleep(0.3)
    mpath = emit_manifest("wiki_pageviews", entries)
    log(f"\nFetched {total} pageview observations across {len(articles)} articles.")
    log(f"Manifest: {mpath}")


if __name__ == "__main__":
    main()
