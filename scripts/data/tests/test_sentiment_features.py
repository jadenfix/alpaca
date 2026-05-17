"""Known-truth tests for scripts/data/analysis/sentiment_features.py.

Each test would fail on a no-op (zeros / constant / non-causal)
implementation, so we exercise behavior, not just the API surface.
"""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Make the analysis package importable the same way test_analysis.py does.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE.parent))

import sentiment_features as sf  # noqa: E402


class AttentionZTests(unittest.TestCase):
    def test_is_causal_future_perturbation_does_not_leak(self):
        # Two series identical for the first half; only differ later.
        rng = np.random.default_rng(7)
        n = 200
        base = rng.normal(size=n)
        perturbed = base.copy()
        cut = 100
        perturbed[cut + 1:] += 50.0  # huge future shock
        z1 = sf.attention_z(base, window=30)
        z2 = sf.attention_z(perturbed, window=30)
        # All z-scores up to and including index `cut` must be identical.
        np.testing.assert_array_equal(z1[: cut + 1], z2[: cut + 1])

    def test_spikes_on_injected_jump(self):
        values = np.concatenate([np.zeros(100), np.full(100, 10.0)])
        # Window=30 → first 30 are NaN; index 100 sees a 30-bar baseline of
        # all-zeros, so the std is 0 and z[100] is NaN. To exercise the
        # injected-jump path properly, use a tiny non-zero baseline.
        values_eps = values + 1e-6 * np.arange(values.size, dtype=np.float64)
        z = sf.attention_z(values_eps, window=30)
        self.assertTrue(np.isfinite(z[100]))
        self.assertGreater(z[100], 2.0)

    def test_first_window_entries_are_nan(self):
        rng = np.random.default_rng(13)
        values = rng.normal(size=80)
        z = sf.attention_z(values, window=30)
        self.assertTrue(np.all(np.isnan(z[:30])))
        # At least some later entries should be finite.
        self.assertTrue(np.any(np.isfinite(z[30:])))


class AttentionChange1dTests(unittest.TestCase):
    def test_zero_on_constant_input_after_first_nan(self):
        values = np.full(20, 5.0)
        chg = sf.attention_change_1d(values)
        self.assertTrue(np.isnan(chg[0]))
        np.testing.assert_allclose(chg[1:], 0.0)

    def test_log_ratio_of_doubling(self):
        values = np.array([1.0, 2.0, 4.0, 8.0])
        chg = sf.attention_change_1d(values)
        self.assertTrue(np.isnan(chg[0]))
        np.testing.assert_allclose(chg[1:], np.log(2.0))

    def test_zero_views_safe(self):
        values = np.array([10.0, 0.0, 10.0])
        chg = sf.attention_change_1d(values)
        # Both transitions involve a zero — must not be -inf / +inf.
        self.assertTrue(np.isfinite(chg[1]))
        self.assertTrue(np.isfinite(chg[2]))
        self.assertEqual(chg[1], 0.0)
        self.assertEqual(chg[2], 0.0)


class ContrarianSignalTests(unittest.TestCase):
    def test_direction_matches_spec(self):
        z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        sig = sf.contrarian_attention_signal(z, z_threshold=2.0)
        np.testing.assert_array_equal(sig, np.array([1.0, 0.0, 0.0, 0.0, -1.0]))

    def test_nan_input_maps_to_zero(self):
        z = np.array([np.nan, 5.0, -5.0])
        sig = sf.contrarian_attention_signal(z, z_threshold=2.0)
        self.assertEqual(sig[0], 0.0)
        self.assertEqual(sig[1], -1.0)
        self.assertEqual(sig[2], 1.0)


class LoadPageviewsTests(unittest.TestCase):
    def test_returns_sorted_timestamps(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "TestTopic.csv"
            # Deliberately out of order.
            path.write_text(
                "ts_nanos,article,views\n"
                "300,TestTopic,30\n"
                "100,TestTopic,10\n"
                "200,TestTopic,20\n"
            )
            ts, views = sf.load_pageviews(path)
            self.assertEqual(list(ts), [100, 200, 300])
            self.assertEqual(list(views), [10.0, 20.0, 30.0])

    def test_empty_file_returns_empty_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "Empty.csv"
            path.write_text("ts_nanos,article,views\n")
            ts, views = sf.load_pageviews(path)
            self.assertEqual(ts.size, 0)
            self.assertEqual(views.size, 0)


class AttentionFeaturesEndToEndTests(unittest.TestCase):
    def test_pipeline_writes_macro_schema(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            pv_dir = tdp / "pageviews"
            out_dir = tdp / "sentiment"
            pv_dir.mkdir()
            # 80 days of synthetic views: random walk around 1000.
            rng = np.random.default_rng(7)
            n = 80
            views = 1000.0 + rng.normal(scale=50.0, size=n).cumsum()
            views = np.clip(views, 1.0, None)
            ts = np.arange(n, dtype=np.int64) * 86_400_000_000_000 + 1_700_000_000_000_000_000
            topic = "TestTopic"
            with (pv_dir / f"{topic}.csv").open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["ts_nanos", "article", "views"])
                for t, v in zip(ts.tolist(), views.tolist()):
                    w.writerow([int(t), topic, int(v)])

            results = sf.attention_features(pv_dir, out_dir, window=30)
            self.assertIn(topic, results)
            self.assertEqual(results[topic].shape, (n, 2))

            out_path = out_dir / f"{topic}_attention.csv"
            self.assertTrue(out_path.exists())

            with out_path.open("r", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader)
                self.assertEqual(header, ["ts_nanos", "series", "value"])
                rows = list(reader)

            # We expect both series names to appear.
            series_seen = {r[1] for r in rows}
            self.assertIn(f"{topic}_attention_z", series_seen)
            self.assertIn(f"{topic}_attention_chg1d", series_seen)
            # Values must parse as floats.
            for r in rows:
                float(r[2])

    def test_missing_pageviews_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            results = sf.attention_features(tdp / "does_not_exist", tdp / "out")
            self.assertEqual(results, {})


class GdeltToneFeaturesTests(unittest.TestCase):
    def test_missing_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            out = sf.gdelt_tone_features(tdp / "does_not_exist", tdp / "out")
            self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main()
