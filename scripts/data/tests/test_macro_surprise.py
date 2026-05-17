"""Known-truth tests for macro_surprise.

These tests would fail on a constant-returning or look-ahead
implementation; they exercise the causality of the EWMA forecast,
the response of the z-score to an injected jump, the false-positive
rate on stationary noise, and the I/O round-trip.
"""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Make the analysis package importable (same convention as test_analysis.py).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE.parent))


class EwmaForecastTests(unittest.TestCase):
    def test_forecast_is_causal(self):
        """forecast[:t+1] must be invariant under perturbations of
        values[t+1:]. This is the defining causality property.
        """
        import macro_surprise as ms
        rng = np.random.default_rng(7)
        base = rng.normal(size=80)
        t = 30
        f_before = ms.ewma_forecast(base, halflife=6)
        # Perturb the strict future and recompute.
        perturbed = base.copy()
        perturbed[t + 1:] += 100.0 * rng.normal(size=perturbed.size - t - 1)
        f_after = ms.ewma_forecast(perturbed, halflife=6)
        # forecast[:t+1] uses only values[:t], i.e. up to and including
        # values[t]; perturbing values[t+1:] cannot move it.
        np.testing.assert_allclose(f_before[: t + 1], f_after[: t + 1])
        # forecast[0] must be NaN (no prior data).
        self.assertTrue(np.isnan(f_before[0]))

    def test_constant_input_yields_constant_forecast(self):
        """After the burn-in (t >= 1) the EWMA of a constant series
        equals the constant exactly.
        """
        import macro_surprise as ms
        c = 3.14159
        values = np.full(100, c, dtype=np.float64)
        f = ms.ewma_forecast(values, halflife=6)
        self.assertTrue(np.isnan(f[0]))
        np.testing.assert_allclose(f[1:], c, rtol=0, atol=1e-12)


class MacroSurpriseTests(unittest.TestCase):
    def test_injected_jump_produces_large_z(self):
        """0.0 for 100 steps then 5.0 → |z[100]| > 2.

        The first 100 residuals are 0 (constant input → exact EWMA);
        the rolling std of those prior residuals is 0, so we add a
        single tiny perturbation to seed a non-zero scale, then verify
        the jump is flagged.
        """
        import macro_surprise as ms
        n = 101
        values = np.zeros(n, dtype=np.float64)
        # Tiny noise on the first chunk so the prior-residual std > 0.
        rng = np.random.default_rng(0)
        values[:100] += rng.normal(scale=1e-3, size=100)
        values[100] = 5.0
        z = ms.macro_surprises(values, halflife=6, z_window=36)
        self.assertTrue(np.isfinite(z[100]))
        self.assertGreater(abs(z[100]), 2.0)

    def test_no_false_positives_on_stationary_noise(self):
        """On 200 i.i.d. N(0,1) the fraction of |z| > 1.5 should be
        roughly the Gaussian tail (~13 %). Accept 5-25 %.
        """
        import macro_surprise as ms
        rng = np.random.default_rng(13)
        values = rng.standard_normal(200)
        z = ms.macro_surprises(values, halflife=6, z_window=36)
        valid = z[np.isfinite(z)]
        # We should have a healthy number of valid entries.
        self.assertGreater(valid.size, 150)
        frac = float(np.mean(np.abs(valid) > 1.5))
        self.assertGreater(frac, 0.05)
        self.assertLess(frac, 0.25)


class EventIndicatorTests(unittest.TestCase):
    def test_filters_threshold(self):
        import macro_surprise as ms
        z = np.array([-2.0, 0.5, 1.8, -0.3], dtype=np.float64)
        ts = np.array([10, 20, 30, 40], dtype=np.int64)
        events = ms.event_indicator(z, ts, z_threshold=1.5)
        self.assertEqual(len(events), 2)
        # Check the events are the right rows in order.
        self.assertEqual(events[0]["ts_nanos"], 10)
        self.assertEqual(events[0]["sign"], -1)
        self.assertAlmostEqual(events[0]["value"], -2.0)
        self.assertEqual(events[1]["ts_nanos"], 30)
        self.assertEqual(events[1]["sign"], 1)
        self.assertAlmostEqual(events[1]["value"], 1.8)


class SurpriseFeaturesIoTests(unittest.TestCase):
    def test_minimal_csv_round_trip(self):
        """Write a 50-row CSV, run surprise_features, assert the
        output CSV exists and follows the documented schema.
        """
        import macro_surprise as ms
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            macro_dir = tdp / "macro"
            out_dir = tdp / "out"
            macro_dir.mkdir()
            # Build a deterministic 50-row series.
            csv_path = macro_dir / "FAKESERIES.csv"
            with csv_path.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["ts_nanos", "series", "value"])
                base_ts = 1_000_000_000_000_000_000
                day = 86_400_000_000_000
                values = rng.normal(size=50).cumsum()
                for i, v in enumerate(values):
                    w.writerow([base_ts + i * day, "FAKESERIES", f"{v:.6f}"])

            results = ms.surprise_features(macro_dir, out_dir)
            self.assertIn("FAKESERIES", results)

            out_path = out_dir / "FAKESERIES_surprise.csv"
            self.assertTrue(out_path.exists())

            with out_path.open("r", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader)
                self.assertEqual(header, ["ts_nanos", "series", "value"])
                rows = list(reader)
            # At least some surprise rows should have been emitted.
            self.assertGreater(len(rows), 10)
            # All rows match schema: int ts, the labelled series, float value.
            for row in rows:
                self.assertEqual(len(row), 3)
                int(row[0])  # raises if not an int
                self.assertEqual(row[1], "FAKESERIES_surprise_z")
                self.assertTrue(np.isfinite(float(row[2])))

    def test_small_series_is_skipped(self):
        """A < 50-row series must NOT produce an output file."""
        import macro_surprise as ms
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            macro_dir = tdp / "macro"
            out_dir = tdp / "out"
            macro_dir.mkdir()
            csv_path = macro_dir / "TINY.csv"
            with csv_path.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["ts_nanos", "series", "value"])
                for i in range(10):
                    w.writerow([1000 + i, "TINY", float(i)])
            results = ms.surprise_features(macro_dir, out_dir)
            self.assertNotIn("TINY", results)
            self.assertFalse((out_dir / "TINY_surprise.csv").exists())


if __name__ == "__main__":
    unittest.main()
