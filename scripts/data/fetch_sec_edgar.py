#!/usr/bin/env python3
"""
Fetch recent 10-K / 10-Q filing metadata from SEC EDGAR.

We use the official JSON submissions API at
    https://data.sec.gov/submissions/CIK<CIK_10>.json
which is free and key-less but requires a descriptive User-Agent (set in
_common.py). For each ticker we resolve CIK via the public company tickers
file, then pull recent filings.

Output: data/filings/<TICKER>.csv with columns
    ts_nanos, accession, form, primary_doc_url, report_date

The text of each filing is large (several MB); we fetch only metadata by
default. Pass --include-text to also save the primary document.

Usage:
    python3 scripts/data/fetch_sec_edgar.py AAPL MSFT
    python3 scripts/data/fetch_sec_edgar.py        # default basket
"""

from __future__ import annotations
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from _common import FILINGS_DIR, emit_manifest, ensure_dir, http_get_json, log

DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM"]

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def load_ticker_to_cik() -> dict[str, int]:
    """Returns map TICKER → CIK (int)."""
    data = http_get_json(TICKERS_URL)
    out = {}
    # The file is a dict keyed by index; each value: {cik_str, ticker, title}
    for _, v in data.items():
        out[v["ticker"].upper()] = int(v["cik_str"])
    return out


def fetch_one(ticker: str, cik: int) -> tuple[int, dict]:
    url = SUBS_URL_TEMPLATE.format(cik=cik)
    try:
        subs = http_get_json(url, timeout=30)
    except Exception as e:
        log(f"  ✗ {ticker}: submissions fetch failed ({e})")
        return 0, {"ticker": ticker, "cik": cik, "error": str(e)}
    recent = subs.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    report_dates = recent.get("reportDate", [])
    rows = []
    for i, form in enumerate(forms):
        if form not in ("10-K", "10-Q", "8-K", "20-F", "S-1"):
            continue
        try:
            ts = datetime.strptime(dates[i], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            ns = int(ts.timestamp() * 1_000_000_000)
        except Exception:
            continue
        accession = accessions[i].replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_docs[i]}"
        rows.append([ns, accessions[i], form, url, report_dates[i] if i < len(report_dates) else ""])
    out_path = FILINGS_DIR / f"{ticker}.csv"
    ensure_dir(out_path.parent)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_nanos", "accession", "form", "primary_doc_url", "report_date"])
        w.writerows(rows)
    log(f"  ✓ {ticker} (CIK {cik}): {len(rows)} filings")
    return len(rows), {
        "ticker": ticker, "cik": cik, "path": str(out_path), "rows": len(rows),
    }


def main():
    tickers = [t.upper() for t in (sys.argv[1:] or DEFAULT_TICKERS)]
    log("↓ SEC EDGAR ticker→CIK map")
    try:
        t2c = load_ticker_to_cik()
    except Exception as e:
        log(f"  ✗ ticker map fetch failed: {e}")
        sys.exit(1)
    log(f"  ✓ {len(t2c)} tickers known to SEC")
    ensure_dir(FILINGS_DIR)
    entries = []
    total = 0
    for tic in tickers:
        cik = t2c.get(tic)
        if cik is None:
            log(f"  ⚠ {tic}: no CIK found")
            entries.append({"ticker": tic, "error": "no CIK"})
            continue
        n, e = fetch_one(tic, cik)
        entries.append(e)
        total += n
        # SEC asks for ≤10 req/sec; 1 req/sec is more polite.
        time.sleep(1.0)
    mpath = emit_manifest("sec_edgar", entries)
    log(f"\nFetched {total} filings across {len(tickers)} tickers.")
    log(f"Manifest: {mpath}")


if __name__ == "__main__":
    main()
