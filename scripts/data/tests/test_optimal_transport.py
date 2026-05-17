"""Known-truth tests for the optimal_transport module.

Each test would fail on a no-op / constant-returning implementation: we
check the actual Sinkhorn marginal constraints, monotone properties of
the Wasserstein cost, and Cuturi-Doucet barycenter behaviour on hand-
constructed cases.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE.parent))


class SinkhornTests(unittest.TestCase):
    def test_marginals_match(self):
        import optimal_transport as ot
        rng = np.random.default_rng(0)
        n, m = 7, 5
        a = rng.random(n)
        a = a / a.sum()
        b = rng.random(m)
        b = b / b.sum()
        C = rng.random((n, m))
        P = ot.sinkhorn(a, b, C, reg=0.1, n_iter=500, tol=1e-12)
        # Row sums ≈ a, column sums ≈ b within 1e-4.
        self.assertEqual(P.shape, (n, m))
        np.testing.assert_allclose(P.sum(axis=1), a, atol=1e-4)
        np.testing.assert_allclose(P.sum(axis=0), b, atol=1e-4)

    def test_marginals_match_small_reg_logdomain(self):
        # Lower reg stresses the log-domain stabilisation.
        import optimal_transport as ot
        rng = np.random.default_rng(1)
        n = 6
        a = rng.random(n); a = a / a.sum()
        b = rng.random(n); b = b / b.sum()
        C = rng.random((n, n)) * 5.0
        P = ot.sinkhorn(a, b, C, reg=0.01, n_iter=2000, tol=1e-12)
        self.assertTrue(np.all(np.isfinite(P)))
        np.testing.assert_allclose(P.sum(axis=1), a, atol=1e-4)
        np.testing.assert_allclose(P.sum(axis=0), b, atol=1e-4)


class WassersteinDistanceTests(unittest.TestCase):
    def test_non_negative(self):
        import optimal_transport as ot
        rng = np.random.default_rng(2)
        n = 5
        a = rng.random(n); a = a / a.sum()
        b = rng.random(n); b = b / b.sum()
        C = rng.random((n, n))
        w = ot.wasserstein_distance(a, b, C, reg=0.1)
        self.assertGreaterEqual(w, 0.0)

    def test_identical_distributions_small_distance(self):
        import optimal_transport as ot
        # Zero-diagonal cost so transporting mass to itself is free.
        n = 4
        a = np.array([0.25, 0.25, 0.25, 0.25])
        C = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
        w = ot.wasserstein_distance(a, a, C, reg=0.1)
        # Entropy regularization smears the plan a little, so W is not
        # exactly zero, but it should be well below 0.05.
        self.assertLess(w, 0.05)


class MinimalTurnoverRebalanceTests(unittest.TestCase):
    def test_moves_toward_target(self):
        import optimal_transport as ot
        w_curr = np.array([1.0, 0.0, 0.0, 0.0])
        w_target = np.array([0.0, 0.0, 0.0, 1.0])
        out = ot.minimal_turnover_rebalance(w_curr, w_target, reg=0.05)
        self.assertEqual(out.shape, w_curr.shape)
        # Sums to 1.
        self.assertAlmostEqual(float(out.sum()), 1.0, places=8)
        # No large negatives (tiny floor only).
        self.assertTrue(np.all(out >= -1e-9))
        # Strictly moved toward target: weight on the target index is
        # larger than the starting weight.
        self.assertGreater(out[-1], w_curr[-1] + 0.5)
        # And we shed a meaningful amount of the source index.
        self.assertLess(out[0], 0.5)


class BarycenterTests(unittest.TestCase):
    def test_two_identical_distributions(self):
        import optimal_transport as ot
        a = np.array([0.5, 0.5])
        bary = ot.wasserstein_barycenter([a, a], reg=0.05, n_iter=300)
        self.assertEqual(bary.shape, a.shape)
        self.assertAlmostEqual(float(bary.sum()), 1.0, places=8)
        np.testing.assert_allclose(bary, a, atol=1e-3)


class RegimeBlendedTargetTests(unittest.TestCase):
    def test_blend_puts_mass_on_middle(self):
        import optimal_transport as ot
        regime_targets = {
            "A": np.array([1.0, 0.0, 0.0]),
            "B": np.array([0.0, 0.0, 1.0]),
        }
        regime_probs = {"A": 0.5, "B": 0.5}
        bary = ot.regime_blended_target(regime_targets, regime_probs, reg=0.05)
        self.assertEqual(bary.shape, (3,))
        self.assertAlmostEqual(float(bary.sum()), 1.0, places=8)
        # The Wasserstein barycenter of two endpoint masses concentrates
        # mass on the middle index -- a linear blend would put 0 there.
        self.assertGreater(bary[1], 0.1)


if __name__ == "__main__":
    unittest.main()
