"""Known-truth tests for the TDA / persistent-homology module.

Each test would FAIL on a no-op or constant-returning implementation,
so they validate the math (persistence pairs, norms, causality), not
just the API surface.
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


class PairwiseDistTests(unittest.TestCase):
    def test_shape_and_symmetry(self):
        import tda
        rng = _seed_rng(0)
        X = rng.standard_normal((8, 3))
        D = tda.pairwise_dist(X)
        self.assertEqual(D.shape, (8, 8))
        np.testing.assert_allclose(D, D.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_known_distance(self):
        import tda
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        D = tda.pairwise_dist(X)
        self.assertAlmostEqual(D[0, 1], 5.0, places=12)


class PersistenceDim0Tests(unittest.TestCase):
    def test_collinear_line_of_5_points(self):
        """5 equispaced collinear points: 4 finite deaths equal to spacing."""
        import tda
        spacing = 0.7
        X = (np.arange(5, dtype=np.float64) * spacing).reshape(-1, 1)
        D = tda.pairwise_dist(X)
        pairs = tda.persistence_dim0(D)
        self.assertEqual(pairs.shape, (5, 2))
        # All births are zero.
        np.testing.assert_allclose(pairs[:, 0], 0.0, atol=1e-12)
        # Exactly one infinite death.
        inf_mask = ~np.isfinite(pairs[:, 1])
        self.assertEqual(int(inf_mask.sum()), 1)
        finite_deaths = np.sort(pairs[~inf_mask, 1])
        self.assertEqual(finite_deaths.shape, (4,))
        # Each successive merge must happen at one inter-point spacing.
        np.testing.assert_allclose(finite_deaths, spacing, atol=1e-12)

    def test_two_isolated_clusters(self):
        """Two clusters of 5 points (intra-spacing 0.1, separated by 10).

        Expect 9 small intra-cluster deaths (<= 0.5) and one large
        cluster-merge death (~10), plus 1 immortal component.
        """
        import tda
        intra = 0.1
        sep = 10.0
        c1 = (np.arange(5, dtype=np.float64) * intra).reshape(-1, 1)
        c2 = c1 + sep
        X = np.vstack([c1, c2])
        D = tda.pairwise_dist(X)
        pairs = tda.persistence_dim0(D)
        self.assertEqual(pairs.shape, (10, 2))
        finite_mask = np.isfinite(pairs[:, 1])
        self.assertEqual(int((~finite_mask).sum()), 1)
        finite_deaths = np.sort(pairs[finite_mask, 1])
        # 9 finite deaths total: 8 intra-cluster + 1 cluster-merge.
        self.assertEqual(finite_deaths.shape, (9,))
        # Smallest 8 are intra-cluster merges; bounded well below 0.5.
        self.assertTrue(np.all(finite_deaths[:8] <= 0.5))
        # The last (largest) finite death is the inter-cluster merge ~ sep.
        # Closest inter-cluster pair: dist = sep - 4*intra = 9.6 here.
        expected_merge = sep - 4 * intra
        self.assertAlmostEqual(finite_deaths[-1], expected_merge, places=10)
        # And the merge is "approximately" the separation scale.
        self.assertAlmostEqual(finite_deaths[-1], sep, delta=0.5)


class PersistenceNormTests(unittest.TestCase):
    def test_single_point_norm_is_zero(self):
        """A single point => only the immortal pair; finite-norm = 0."""
        import tda
        X = np.array([[0.0, 0.0, 0.0]])
        D = tda.pairwise_dist(X)
        pairs = tda.persistence_dim0(D)
        self.assertEqual(pairs.shape, (1, 2))
        self.assertTrue(np.isinf(pairs[0, 1]))
        self.assertEqual(tda.persistence_norm(pairs, p=2.0), 0.0)
        self.assertEqual(tda.persistence_norm(pairs, p=1.0), 0.0)

    def test_known_lp_value(self):
        """L^p norm matches hand-computed value on a synthetic diagram."""
        import tda
        # Three finite pairs with lifetimes 1, 2, 2 ; plus one infinite.
        pairs = np.array(
            [[0.0, 1.0], [0.0, 2.0], [0.0, 2.0], [0.0, np.inf]],
            dtype=np.float64,
        )
        # L^2: sqrt(1 + 4 + 4) = 3.0
        self.assertAlmostEqual(tda.persistence_norm(pairs, p=2.0), 3.0, places=12)
        # L^1: 1 + 2 + 2 = 5
        self.assertAlmostEqual(tda.persistence_norm(pairs, p=1.0), 5.0, places=12)


class RollingPersistenceNormTests(unittest.TestCase):
    def test_output_shape_correct(self):
        """T=100, N=4, window=50, step=10 => length = (100-50)/10 + 1 = 6."""
        import tda
        rng = _seed_rng(0)
        R = rng.standard_normal((100, 4))
        out = tda.rolling_persistence_norm(R, window=50, step=10)
        self.assertEqual(out["ts_idx"].shape, (6,))
        self.assertEqual(out["norm_dim0"].shape, (6,))
        self.assertEqual(out["norm_dim1"].shape, (6,))
        # All norms must be finite & non-negative.
        self.assertTrue(np.all(np.isfinite(out["norm_dim0"])))
        self.assertTrue(np.all(np.isfinite(out["norm_dim1"])))
        self.assertTrue(np.all(out["norm_dim0"] >= 0.0))
        self.assertTrue(np.all(out["norm_dim1"] >= 0.0))

    def test_window_too_large_returns_empty(self):
        import tda
        R = np.zeros((10, 4))
        out = tda.rolling_persistence_norm(R, window=50, step=5)
        self.assertEqual(out["ts_idx"].shape, (0,))
        self.assertEqual(out["norm_dim0"].shape, (0,))


class TopologyChangeScoreTests(unittest.TestCase):
    def test_causality_perturbation_does_not_leak_backward(self):
        """Perturbing input[t+1:] leaves output[:t+1] unchanged."""
        import tda
        rng = _seed_rng(0)
        T = 80
        lookback = 20
        x = rng.standard_normal(T) + 1.0  # positive so z-score well-defined
        z_orig = tda.topology_change_score(x, lookback=lookback)
        t = 50
        x_perturbed = x.copy()
        # Perturb everything strictly after index t.
        x_perturbed[t + 1:] += rng.standard_normal(T - t - 1) * 100.0
        z_pert = tda.topology_change_score(x_perturbed, lookback=lookback)
        # Indices 0..t inclusive must be identical (NaN-aware).
        for s in range(t + 1):
            a, b = z_orig[s], z_pert[s]
            if np.isnan(a) and np.isnan(b):
                continue
            self.assertEqual(a, b, msg=f"causality violated at index {s}")

    def test_first_lookback_entries_are_nan(self):
        import tda
        rng = _seed_rng(1)
        x = rng.standard_normal(60) + 5.0
        z = tda.topology_change_score(x, lookback=20)
        self.assertTrue(np.all(np.isnan(z[:20])))
        # And later entries are typically finite.
        self.assertTrue(np.any(np.isfinite(z[20:])))

    def test_spike_produces_high_z(self):
        """A real upward spike must register z > 2."""
        import tda
        rng = _seed_rng(2)
        x = rng.standard_normal(60) * 0.1 + 1.0
        x[50] = 5.0  # large positive shock vs. ~N(1, 0.1) history
        z = tda.topology_change_score(x, lookback=20)
        self.assertGreater(z[50], 2.0)


class PersistenceDim1ProxyTests(unittest.TestCase):
    def test_too_few_points_yields_empty(self):
        import tda
        D = tda.pairwise_dist(np.array([[0.0], [1.0]]))
        out = tda.persistence_dim1_proxy(D)
        self.assertEqual(out.shape, (0, 2))

    def test_first_cycle_born_at_fourth_shortest_edge(self):
        """On 4 random points whose 3 shortest edges form a spanning tree,
        the first 1-cycle is born at the 4th-shortest pairwise edge."""
        import tda
        # Seed 1 chosen so the 3 shortest edges form a tree (verified
        # offline); for 4 generic points this is the typical case.
        rng = _seed_rng(1)
        X = rng.standard_normal((4, 3))
        D = tda.pairwise_dist(X)
        # Sorted upper-triangular distances (all 6 of them).
        iu, ju = np.triu_indices(4, k=1)
        sorted_dists = np.sort(D[iu, ju])
        out = tda.persistence_dim1_proxy(D)
        self.assertGreaterEqual(out.shape[0], 1)
        # The first cycle's birth must equal the 4th-shortest edge length.
        self.assertAlmostEqual(out[0, 0], sorted_dists[3], places=12)
        # And the proxy reports an infinite death.
        self.assertTrue(np.isinf(out[0, 1]))

    def test_square_has_three_independent_cycles(self):
        """Unit square: 6 edges total, MST uses 3 => 3 cycles."""
        import tda
        X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        D = tda.pairwise_dist(X)
        out = tda.persistence_dim1_proxy(D)
        # n(n-1)/2 - (n-1) = 6 - 3 = 3 cycles.
        self.assertEqual(out.shape, (3, 2))
        # All deaths infinite in the MVP proxy.
        self.assertTrue(np.all(np.isinf(out[:, 1])))
        # Births sorted ascending; smallest at distance 1 (side length).
        births = out[:, 0]
        self.assertTrue(np.all(np.diff(births) >= -1e-12))
        self.assertAlmostEqual(births[0], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
