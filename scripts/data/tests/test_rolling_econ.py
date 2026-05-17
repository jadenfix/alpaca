"""Known-truth tests for the rolling econophysics features.

Mirrors the sys.path setup from `test_analysis.py` so the analysis
modules can be imported by bare name.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# Make the analysis package importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE.parent))


def _seed_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


class RollingRmtTests(unittest.TestCase):
    def test_n_above_mp_is_roughly_constant_on_noise(self):
        import rolling_econ as re
        rng = _seed_rng(7)
        t, n = 800, 12
        rets = rng.standard_normal((t, n))
        out = re.rolling_rmt(rets, window=200, step=40)
        n_above = out["n_above_mp"]
        n_above = n_above[np.isfinite(n_above)]
        self.assertGreater(n_above.size, 3)
        # On pure i.i.d. noise the count of eigenvalues above the
        # Marchenko-Pastur edge should be small and stable across windows.
        self.assertLess(float(np.var(n_above)), 2.0)
        # And output arrays are downsampled relative to T.
        self.assertLess(out["ts_idx"].size, t)


class RollingMfdfaTests(unittest.TestCase):
    def test_delta_alpha_roughly_constant_on_brownian(self):
        import rolling_econ as re
        rng = _seed_rng(11)
        # Monofractal Brownian motion: cumulative sum of i.i.d. Gaussian
        # increments. MF-DFA Δα should be ~ 0 and stable across windows.
        x = np.cumsum(rng.standard_normal(2000))
        out = re.rolling_mfdfa(x, window=400, step=80)
        d = out["delta_alpha"]
        d = d[np.isfinite(d)]
        self.assertGreater(d.size, 3)
        self.assertLess(float(np.var(d)), 0.05)


class RollingAlphaStableTests(unittest.TestCase):
    def test_alpha_stays_near_two_on_gaussian(self):
        import rolling_econ as re
        rng = _seed_rng(13)
        x = rng.standard_normal(2000)
        out = re.rolling_alpha_stable(x, window=400, step=80)
        alpha = out["alpha"]
        alpha = alpha[np.isfinite(alpha)]
        self.assertGreater(alpha.size, 3)
        # Every window should put α firmly in the Gaussian basin.
        self.assertGreater(float(alpha.min()), 1.6)


class DetectShiftsTests(unittest.TestCase):
    def test_detects_injected_jump(self):
        import rolling_econ as re
        # 80 flat points at 0.5 (with tiny noise to give nonzero std),
        # then a jump to 5.0 for the rest.
        rng = _seed_rng(17)
        baseline_len = 80
        jump_len = 20
        baseline = 0.5 + 0.01 * rng.standard_normal(baseline_len)
        jump = 5.0 + 0.01 * rng.standard_normal(jump_len)
        series = np.concatenate([baseline, jump])
        alerts = re.detect_shifts(series, z_threshold=2.0)
        self.assertTrue(len(alerts) >= 1)
        # At least one alert at or after the jump with |z| > 2.
        post_jump = [a for a in alerts
                     if a["idx"] >= baseline_len and abs(a["z"]) > 2.0]
        self.assertTrue(len(post_jump) >= 1,
                        f"no post-jump alert fired; alerts={alerts}")


class RollingEconReportTests(unittest.TestCase):
    def test_end_to_end_returns_all_keys(self):
        import rolling_econ as re
        rng = _seed_rng(23)
        t, n = 600, 5
        rets = rng.standard_normal((t, n)) * 0.01
        names = [f"s{i}" for i in range(n)]
        out = re.rolling_econ_report(rets, names, window=200, step=40)
        # Top-level shape.
        for k in ("rmt", "mfdfa_per_series", "alpha_stable_per_series",
                  "hawkes_per_series", "shifts"):
            self.assertIn(k, out)
        # RMT sub-keys.
        for k in ("ts_idx", "n_above_mp", "tw_pvalue",
                  "top_eigenvalue", "top_eigenvalue_evr", "q"):
            self.assertIn(k, out["rmt"])
        # Per-series dicts cover every name.
        self.assertEqual(set(out["mfdfa_per_series"].keys()), set(names))
        self.assertEqual(set(out["alpha_stable_per_series"].keys()),
                         set(names))
        self.assertEqual(set(out["hawkes_per_series"].keys()), set(names))
        # Per-series MF-DFA / alpha-stable / Hawkes have ts_idx aligned
        # with the RMT axis.
        rmt_len = out["rmt"]["ts_idx"].size
        for name in names:
            self.assertEqual(out["mfdfa_per_series"][name]["ts_idx"].size,
                             rmt_len)
            self.assertEqual(
                out["alpha_stable_per_series"][name]["ts_idx"].size,
                rmt_len,
            )
            self.assertEqual(
                out["hawkes_per_series"][name]["ts_idx"].size,
                rmt_len,
            )
        # shifts is a list (possibly empty) of dicts tagged with feature.
        self.assertIsInstance(out["shifts"], list)
        for a in out["shifts"]:
            self.assertIn("feature", a)
            self.assertIn("z", a)


if __name__ == "__main__":
    unittest.main()
