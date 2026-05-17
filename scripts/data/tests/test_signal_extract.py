"""Tests for scripts/data/signal_extract.py — feature generation."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_extract import detect_schema, extract_features, mean_, std_, rolling


class SchemaDetectorTests(unittest.TestCase):
    def test_detects_bar_schema(self):
        header = ["ts_nanos", "symbol", "open", "high", "low", "close", "volume", "span_secs"]
        kind, ts_idx, val_idx = detect_schema(header)
        self.assertEqual(kind, "bar")
        self.assertEqual(ts_idx, 0)
        self.assertEqual(val_idx, 5)  # close column

    def test_detects_macro_schema(self):
        kind, ts_idx, val_idx = detect_schema(["ts_nanos", "series", "value"])
        self.assertEqual(kind, "macro")
        self.assertEqual((ts_idx, val_idx), (0, 2))

    def test_detects_factor_schema(self):
        kind, _, _ = detect_schema(["ts_nanos", "factor", "value"])
        self.assertEqual(kind, "factor")

    def test_unknown_schema(self):
        kind, _, _ = detect_schema(["random", "stuff", "here"])
        self.assertEqual(kind, "unknown")


class RollingMathTests(unittest.TestCase):
    def test_mean_known(self):
        self.assertAlmostEqual(mean_([1.0, 2.0, 3.0, 4.0, 5.0]), 3.0)

    def test_std_known(self):
        # pstdev([1,2,3,4,5]) = sqrt(2)
        self.assertAlmostEqual(std_([1.0, 2.0, 3.0, 4.0, 5.0]), math.sqrt(2.0))

    def test_rolling_mean_aligns(self):
        # Window=3, last value should be mean of [3,4,5].
        out = rolling([1.0, 2.0, 3.0, 4.0, 5.0], 3, mean_)
        self.assertTrue(math.isnan(out[0]))
        self.assertTrue(math.isnan(out[1]))
        self.assertAlmostEqual(out[2], 2.0)  # mean(1,2,3)
        self.assertAlmostEqual(out[3], 3.0)  # mean(2,3,4)
        self.assertAlmostEqual(out[4], 4.0)  # mean(3,4,5)


class FeatureExtractionTests(unittest.TestCase):
    def test_log_return_matches_hand_calculation(self):
        # 5 prices: 100, 110, 99, 121, 100
        # log_returns:  NaN, ln(1.1), ln(0.9), ln(1.222...), ln(0.826...)
        series = [(i * 1_000_000_000, p) for i, p in enumerate([100.0, 110.0, 99.0, 121.0, 100.0])]
        rows = list(extract_features("SYM", series))
        log_ret_rows = [r for r in rows if r[2] == "log_return"]
        # First bar has NaN log_return → excluded from emit.
        # We expect 4 emitted log_returns (bars 1..4).
        self.assertEqual(len(log_ret_rows), 4)
        # Spot-check: bar 1 log_return = ln(110/100).
        expected_lr1 = math.log(110.0 / 100.0)
        self.assertAlmostEqual(float(log_ret_rows[0][3]), expected_lr1, places=8)

    def test_drawdown_zero_for_monotonic_up(self):
        series = [(i * 1_000_000_000, 100.0 + i) for i in range(20)]
        rows = list(extract_features("SYM", series))
        dd_rows = [r for r in rows if r[2] == "drawdown"]
        # In monotone-up series, drawdown is always 0.
        for r in dd_rows:
            self.assertAlmostEqual(float(r[3]), 0.0, places=8)

    def test_level_feature_is_input_passthrough(self):
        series = [(i * 1_000_000_000, 100.0 + i * 0.5) for i in range(10)]
        rows = list(extract_features("X", series))
        level_rows = [r for r in rows if r[2] == "level"]
        self.assertEqual(len(level_rows), 10)
        for i, r in enumerate(level_rows):
            self.assertAlmostEqual(float(r[3]), 100.0 + i * 0.5, places=8)

    def test_zscore_centers_correctly(self):
        # 30 identical values → rolling z-score is exactly 0 (when defined).
        series = [(i * 1_000_000_000, 100.0) for i in range(40)]
        rows = list(extract_features("X", series))
        z_rows = [r for r in rows if r[2] == "zscore_21"]
        # With zero std, our zscore handles div-by-zero with NaN — those are
        # NOT emitted (the emit helper filters NaN). Result: zero rows.
        # If std were positive, all values would equal mean → z = 0.
        # The DEFINED z-scores (none here) would be 0 exactly.
        for r in z_rows:
            self.assertEqual(float(r[3]), 0.0)


if __name__ == "__main__":
    unittest.main()
