"""Tests for scripts/data/validate.py — the agentic data validator.

Each test feeds the detector a synthetic input with KNOWN issues and asserts
the validator flags the right rows. A detector that returned an empty list
unconditionally would fail every one of these.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validate import detect_gaps, detect_outliers, detect_splits


class GapDetectorTests(unittest.TestCase):
    def test_no_gaps_in_evenly_spaced(self):
        # 1-second steps for 60 ticks, no gaps.
        ts = [i * 1_000_000_000 for i in range(60)]
        self.assertEqual(detect_gaps(ts, threshold=5.0), [])

    def test_flags_a_known_gap(self):
        # 50 ticks at 1-second intervals, then a 10-second gap.
        ts = [i * 1_000_000_000 for i in range(50)]
        ts.append(ts[-1] + 10_000_000_000)
        gaps = detect_gaps(ts, threshold=5.0)
        self.assertEqual(len(gaps), 1)
        idx, prev, curr = gaps[0]
        self.assertEqual(idx, 50)
        self.assertEqual(curr - prev, 10_000_000_000)

    def test_short_series_returns_empty(self):
        self.assertEqual(detect_gaps([1, 2], threshold=5.0), [])

    def test_threshold_controls_sensitivity(self):
        # 50 ticks at 1s, one gap of 6s.
        ts = [i * 1_000_000_000 for i in range(50)]
        ts.append(ts[-1] + 6_000_000_000)
        # threshold=5 → flagged
        self.assertEqual(len(detect_gaps(ts, threshold=5.0)), 1)
        # threshold=10 → not flagged
        self.assertEqual(len(detect_gaps(ts, threshold=10.0)), 0)


class OutlierDetectorTests(unittest.TestCase):
    def test_no_outliers_in_flat_series(self):
        closes = [100.0] * 100
        self.assertEqual(detect_outliers(closes, z_thresh=5.0), [])

    def test_no_outliers_in_calm_series(self):
        # Tiny deterministic noise, no real outliers.
        closes = [100.0 + 0.01 * (i % 7 - 3) for i in range(100)]
        # In zero-or-very-low-vol windows, std may be 0 → detector skips.
        out = detect_outliers(closes, z_thresh=5.0)
        self.assertEqual(out, [])

    def test_flags_clear_outlier(self):
        # 100 small returns then one 20% jump.
        closes = [100.0 + 0.1 * (i % 7 - 3) for i in range(100)]
        closes.append(closes[-1] * 1.20)  # +20% — should be a massive z-score
        out = detect_outliers(closes, z_thresh=5.0, window=30)
        self.assertEqual(len(out), 1)
        idx, log_return, z = out[0]
        self.assertEqual(idx, 100)
        self.assertGreater(abs(z), 5.0)
        self.assertGreater(log_return, 0.0)  # was a positive jump

    def test_short_series_returns_empty(self):
        self.assertEqual(detect_outliers([100.0] * 5, z_thresh=5.0), [])


class SplitDetectorTests(unittest.TestCase):
    def test_no_splits_in_smooth_series(self):
        closes = [100.0 * (1.001 ** i) for i in range(50)]
        self.assertEqual(detect_splits(closes, split_ratio=1.5), [])

    def test_flags_2_for_1_split(self):
        # 30 bars at $100, then drops to $50 (a 2-for-1 split).
        closes = [100.0] * 30 + [50.0] * 30
        out = detect_splits(closes, split_ratio=1.5)
        self.assertEqual(len(out), 1)
        idx, ratio = out[0]
        self.assertEqual(idx, 30)
        # The ratio is max(50/100, 100/50) = 2.0
        self.assertAlmostEqual(ratio, 2.0)

    def test_threshold_controls_sensitivity(self):
        # 1.4× move
        closes = [100.0, 100.0, 100.0, 140.0]
        self.assertEqual(len(detect_splits(closes, split_ratio=1.5)), 0)
        self.assertEqual(len(detect_splits(closes, split_ratio=1.3)), 1)


if __name__ == "__main__":
    unittest.main()
