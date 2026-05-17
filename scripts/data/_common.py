"""
Shared helpers for the data fetchers and validator.

Conventions:
  - Bar files live in data/real/<TICKER>.csv with schema
      ts_nanos, symbol, open, high, low, close, volume, span_secs
  - Macro / non-OHLCV series live in data/macro/<SERIES>.csv with schema
      ts_nanos, series, value
  - Constituent lists live in data/universe/<NAME>.csv with schema
      symbol, name, sector, weight
  - The validator emits cleaning manifests at data/manifests/<source>_<ts>.json
"""

from __future__ import annotations
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

DATA_ROOT = Path("data")
REAL_DIR = DATA_ROOT / "real"
MACRO_DIR = DATA_ROOT / "macro"
UNIVERSE_DIR = DATA_ROOT / "universe"
CRYPTO_DIR = DATA_ROOT / "crypto"
FACTORS_DIR = DATA_ROOT / "factors"
FILINGS_DIR = DATA_ROOT / "filings"
COT_DIR = DATA_ROOT / "cot"
MANIFEST_DIR = DATA_ROOT / "manifests"

BAR_HEADER = ["ts_nanos", "symbol", "open", "high", "low", "close", "volume", "span_secs"]
MACRO_HEADER = ["ts_nanos", "series", "value"]
UNIVERSE_HEADER = ["symbol", "name", "sector", "weight"]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def date_to_ns(date_str: str) -> int:
    """Convert YYYY-MM-DD to nanoseconds since epoch (UTC midnight)."""
    from datetime import datetime, timezone
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def http_get_text(url: str, headers: dict | None = None, timeout: int = 30) -> str:
    """GET as text with a sensible User-Agent (SEC and others require this)."""
    import urllib.request
    req = urllib.request.Request(url, headers={
        "User-Agent": (headers or {}).get(
            "User-Agent",
            "algo-research/0.1 (https://github.com/jadenfix/alpaca; contact: jaden@roe-ai.com)"
        ),
        **(headers or {}),
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def http_get_json(url: str, headers: dict | None = None, timeout: int = 30):
    return json.loads(http_get_text(url, headers, timeout))


def http_get_bytes(url: str, headers: dict | None = None, timeout: int = 60) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={
        "User-Agent": (headers or {}).get(
            "User-Agent",
            "algo-research/0.1 (https://github.com/jadenfix/alpaca; contact: jaden@roe-ai.com)"
        ),
        **(headers or {}),
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def write_bar_csv(path: Path, rows: Iterable[list]) -> int:
    ensure_dir(path.parent)
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BAR_HEADER)
        for r in rows:
            w.writerow(r)
            n += 1
    return n


def write_macro_csv(path: Path, rows: Iterable[list]) -> int:
    ensure_dir(path.parent)
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MACRO_HEADER)
        for r in rows:
            w.writerow(r)
            n += 1
    return n


def write_universe_csv(path: Path, rows: Iterable[list]) -> int:
    ensure_dir(path.parent)
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(UNIVERSE_HEADER)
        for r in rows:
            w.writerow(r)
            n += 1
    return n


def emit_manifest(source: str, entries: list[dict]) -> Path:
    ensure_dir(MANIFEST_DIR)
    ts = int(time.time())
    path = MANIFEST_DIR / f"{source}_{ts}.json"
    with open(path, "w") as f:
        json.dump({"source": source, "ts": ts, "entries": entries}, f, indent=2)
    return path


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)
