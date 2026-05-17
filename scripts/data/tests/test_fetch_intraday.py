"""Known-truth tests for scripts/data/fetch_intraday.py."""
from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Make scripts/data importable (mirrors test_walkforward.py / test_common.py).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _common import BAR_HEADER  # noqa: E402
import fetch_intraday  # noqa: E402
from fetch_intraday import (  # noqa: E402
    SPAN_SECS_1M,
    aggregate_intraday_features,
    fetch_intraday as fetch_intraday_fn,
)


def _build_synthetic_session_csv(path: Path, symbol: str = "TEST") -> None:
    """Write a 390-minute synthetic US-session CSV (09:30 -> 16:00 ET) with a
    linearly rising close and constant volume."""
    # 390 bars labeled 09:30 .. 15:59 inclusive in US/Eastern, then converted
    # to UTC nanoseconds via Timestamp.value (which is already UTC-ns).
    idx = pd.date_range("2025-01-15 09:30", periods=390, freq="1min",
                        tz="America/New_York")
    closes = np.linspace(100.0, 110.0, 390)
    # Use simple OHLC where O = previous close (or first close for first bar),
    # H/L bracket the close by a tiny epsilon, and C = the line.
    opens = np.empty_like(closes)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    highs = np.maximum(opens, closes) + 1e-4
    lows = np.minimum(opens, closes) - 1e-4
    vols = np.full_like(closes, 1000.0)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BAR_HEADER)
        for t, o, h, lo, c, v in zip(idx, opens, highs, lows, closes, vols):
            ns = int(pd.Timestamp(t).value)
            w.writerow([
                ns, symbol,
                f"{o:.6f}", f"{h:.6f}", f"{lo:.6f}", f"{c:.6f}",
                f"{v:.0f}", SPAN_SECS_1M,
            ])


def _mock_yf_df() -> pd.DataFrame:
    """A tiny tz-aware OHLCV DataFrame that mimics yfinance.download output
    (single-level columns, US/Eastern index)."""
    idx = pd.date_range("2025-01-15 09:30", periods=5, freq="1min",
                        tz="America/New_York")
    return pd.DataFrame({
        "Open":   [100.0, 100.1, 100.2, 100.3, 100.4],
        "High":   [100.5, 100.6, 100.7, 100.8, 100.9],
        "Low":    [ 99.5,  99.6,  99.7,  99.8,  99.9],
        "Close":  [100.2, 100.3, 100.4, 100.5, 100.6],
        "Volume": [1_000, 1_100, 1_050, 1_200, 1_150],
    }, index=idx)


class AggregateIntradayFeaturesTests(unittest.TestCase):
    def test_synthetic_session_yields_sensible_features(self):
        with tempfile.TemporaryDirectory() as td:
            in_dir = Path(td)
            out_dir = Path(td)
            _build_synthetic_session_csv(in_dir / "TEST_1m.csv", "TEST")

            feats = aggregate_intraday_features("TEST", in_dir, out_dir)

            # All five features should be present.
            self.assertEqual(set(feats.keys()), {
                "realized_vol_intraday",
                "vwap_premium",
                "opening_range_pct",
                "afternoon_drift",
                "volume_clock_skew",
            })

            # Rising prices over 390 bars -> realized vol > 0.
            self.assertTrue(math.isfinite(feats["realized_vol_intraday"]))
            self.assertGreater(feats["realized_vol_intraday"], 0.0)

            # Linearly rising close + constant volume -> last close above the
            # session mean (which equals VWAP for constant volume) -> positive.
            self.assertTrue(math.isfinite(feats["vwap_premium"]))
            self.assertGreater(feats["vwap_premium"], 0.0)

            # 09:30-10:00 sees a strictly increasing close, so high > low.
            self.assertTrue(math.isfinite(feats["opening_range_pct"]))
            self.assertGreater(feats["opening_range_pct"], 0.0)

            # Late-afternoon mean close > early-afternoon mean close on a
            # monotonically rising session.
            self.assertTrue(math.isfinite(feats["afternoon_drift"]))
            self.assertGreater(feats["afternoon_drift"], 0.0)

            # The features file should also be written in macro schema.
            out_path = out_dir / "TEST_features.csv"
            self.assertTrue(out_path.exists())
            with open(out_path) as f:
                r = csv.reader(f)
                header = next(r)
                self.assertEqual(header, ["ts_nanos", "series", "value"])
                rows = list(r)
            self.assertEqual(len(rows), 5)


class FetchIntradayMergeTests(unittest.TestCase):
    def test_merge_dedupes_on_ts_nanos(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            mock_df = _mock_yf_df()

            with patch.object(fetch_intraday, "yf") as mock_yf:
                mock_yf.download.return_value = mock_df
                n1 = fetch_intraday_fn("SPY", out_dir)
                n2 = fetch_intraday_fn("SPY", out_dir)

            self.assertEqual(n1, len(mock_df))
            self.assertEqual(n2, len(mock_df))  # merge -> still N, no dupes

            out_path = out_dir / "SPY_1m.csv"
            with open(out_path) as f:
                r = csv.reader(f)
                header = next(r)
                self.assertEqual(header, BAR_HEADER)
                rows = list(r)
            self.assertEqual(len(rows), len(mock_df))
            ts_values = [int(r[0]) for r in rows]
            self.assertEqual(len(ts_values), len(set(ts_values)))


class FetchIntradayEmptyTests(unittest.TestCase):
    def test_empty_dataframe_returns_zero(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            empty_df = pd.DataFrame()
            with patch.object(fetch_intraday, "yf") as mock_yf:
                mock_yf.download.return_value = empty_df
                n = fetch_intraday_fn("MUTL", out_dir)
            self.assertEqual(n, 0)
            # No file should have been created.
            self.assertFalse((out_dir / "MUTL_1m.csv").exists())


class BarCsvHeaderTests(unittest.TestCase):
    def test_csv_header_is_exactly_bar_header(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            mock_df = _mock_yf_df()
            with patch.object(fetch_intraday, "yf") as mock_yf:
                mock_yf.download.return_value = mock_df
                fetch_intraday_fn("SPY", out_dir)
            out_path = out_dir / "SPY_1m.csv"
            with open(out_path) as f:
                first_line = f.readline().rstrip("\r\n")
            self.assertEqual(first_line, ",".join(BAR_HEADER))


if __name__ == "__main__":
    unittest.main()
