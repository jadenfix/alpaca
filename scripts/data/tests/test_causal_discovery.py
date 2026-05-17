"""Known-truth tests for the PC algorithm causal-discovery module."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# Make the analysis package importable, matching test_analysis.py.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE.parent))


def _seed_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


class PartialCorrTests(unittest.TestCase):
    def test_unconditional_matches_pearson(self):
        import causal_discovery as cd
        rng = _seed_rng(7)
        n = 400
        x = rng.normal(size=n)
        y = 0.5 * x + rng.normal(size=n)
        z = rng.normal(size=n)  # independent
        data = np.column_stack([x, y, z])
        rho_pc = cd.partial_corr(data, 0, 1, condset=[])
        pearson = float(np.corrcoef(x, y)[0, 1])
        self.assertAlmostEqual(rho_pc, pearson, places=9)

    def test_berkson_conditioning_on_common_effect(self):
        import causal_discovery as cd
        rng = _seed_rng(13)
        n = 600
        x = rng.normal(size=n)
        y = rng.normal(size=n)  # marginally independent of x
        z = x + y + 0.1 * rng.normal(size=n)  # common effect
        data = np.column_stack([x, y, z])
        # Marginal corr X, Y should be small.
        self.assertLess(abs(float(np.corrcoef(x, y)[0, 1])), 0.15)
        # Conditioning on the collider Z induces strong negative dependence.
        rho_cond = cd.partial_corr(data, 0, 1, condset=[2])
        self.assertLess(rho_cond, -0.3)


class FisherZTests(unittest.TestCase):
    def test_zero_rho_yields_pvalue_near_one(self):
        import causal_discovery as cd
        p = cd.fisher_z_pvalue(0.0, n=1000, k=0)
        self.assertGreater(p, 0.99)

    def test_strong_rho_yields_tiny_pvalue(self):
        import causal_discovery as cd
        p = cd.fisher_z_pvalue(0.5, n=1000, k=0)
        self.assertLess(p, 1e-30)


class PcAlgorithmTests(unittest.TestCase):
    def test_recovers_chain_x_y_z(self):
        import causal_discovery as cd
        rng = _seed_rng(7)
        n = 800
        x = rng.normal(size=n)
        y = 0.7 * x + rng.normal(size=n)
        z = 0.7 * y + rng.normal(size=n)
        data = np.column_stack([x, y, z])
        out = cd.pc_algorithm(data, names=["X", "Y", "Z"], alpha=0.05,
                              max_condset_size=2)
        adj = out["skeleton"]
        # The (X, Z) edge must be removed (Y d-separates them).
        self.assertFalse(bool(adj[0, 2]))
        self.assertFalse(bool(adj[2, 0]))
        # X-Y and Y-Z edges should remain.
        self.assertTrue(bool(adj[0, 1]))
        self.assertTrue(bool(adj[1, 2]))

    def test_recovers_v_structure(self):
        import causal_discovery as cd
        rng = _seed_rng(11)
        n = 800
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        z = x + y + 0.3 * rng.normal(size=n)
        data = np.column_stack([x, y, z])
        adj, sepsets = cd.pc_skeleton(data, alpha=0.05, max_condset_size=2)
        # Skeleton: X-Z and Y-Z present; X-Y absent.
        self.assertTrue(bool(adj[0, 2]))
        self.assertTrue(bool(adj[1, 2]))
        self.assertFalse(bool(adj[0, 1]))
        dirs = cd.orient_v_structures(adj, sepsets)
        self.assertEqual(dirs[0, 2], 1)  # X -> Z
        self.assertEqual(dirs[1, 2], 1)  # Y -> Z

    def test_no_spurious_edges_on_iid_data(self):
        import causal_discovery as cd
        rng = _seed_rng(17)
        n_samples = 500
        n_vars = 6
        data = rng.normal(size=(n_samples, n_vars))
        adj, _ = cd.pc_skeleton(data, alpha=0.05, max_condset_size=2)
        # Count surviving edges (upper triangle).
        n_edges = sum(1 for i in range(n_vars) for j in range(i + 1, n_vars)
                      if adj[i, j])
        total_possible = n_vars * (n_vars - 1) // 2  # = 15
        n_removed = total_possible - n_edges
        self.assertGreaterEqual(n_removed, 12)


if __name__ == "__main__":
    unittest.main()
