"""Known-truth tests for the tiny symbolic regression engine.

Each test would fail on a no-op or constant-returning implementation,
so they validate the math (safe ops, Spearman correctness, GP
convergence) rather than only the API surface.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# Make the analysis package importable (mirrors test_analysis.py).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE.parent))

import symbolic_regression as sr  # noqa: E402


class EvaluateTests(unittest.TestCase):
    """evaluate() agrees with numpy on every supported op."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        self.X = self.rng.normal(size=(40, 3))

    def _var(self, i: int) -> sr.Node:
        return sr.Node(kind="var", payload=i, children=[])

    def _const(self, v: float) -> sr.Node:
        return sr.Node(kind="const", payload=float(v), children=[])

    def _bin(self, name: str, a: sr.Node, b: sr.Node) -> sr.Node:
        return sr.Node(kind="op", payload=name, children=[a, b])

    def _un(self, name: str, a: sr.Node) -> sr.Node:
        return sr.Node(kind="op", payload=name, children=[a])

    def test_var_and_const_leaves(self) -> None:
        out = sr.evaluate(self._var(1), self.X)
        np.testing.assert_allclose(out, self.X[:, 1])
        out = sr.evaluate(self._const(3.5), self.X)
        np.testing.assert_allclose(out, np.full(self.X.shape[0], 3.5))

    def test_binary_ops(self) -> None:
        x0 = self._var(0)
        x1 = self._var(1)
        cases = {
            "+": self.X[:, 0] + self.X[:, 1],
            "-": self.X[:, 0] - self.X[:, 1],
            "*": self.X[:, 0] * self.X[:, 1],
            "max": np.maximum(self.X[:, 0], self.X[:, 1]),
            "min": np.minimum(self.X[:, 0], self.X[:, 1]),
        }
        for op, expected in cases.items():
            with self.subTest(op=op):
                got = sr.evaluate(self._bin(op, x0, x1), self.X)
                np.testing.assert_allclose(got, expected, atol=1e-10)

    def test_unary_ops(self) -> None:
        x0 = self._var(0)
        cases = {
            "abs": np.abs(self.X[:, 0]),
            "tanh": np.tanh(self.X[:, 0]),
            "sign": np.sign(self.X[:, 0]),
            "log_abs": np.log(np.abs(self.X[:, 0]) + 1e-9),
        }
        for op, expected in cases.items():
            with self.subTest(op=op):
                got = sr.evaluate(self._un(op, x0), self.X)
                np.testing.assert_allclose(got, expected, atol=1e-8)


class SafeDivisionTests(unittest.TestCase):
    """Division by (near-)zero produces finite output (==0)."""

    def test_div_by_zero_returns_zero_not_nan(self) -> None:
        T = 20
        X = np.column_stack([
            np.linspace(1.0, 2.0, T),
            np.zeros(T),
        ])
        tree = sr.Node(
            kind="op",
            payload="/",
            children=[
                sr.Node(kind="var", payload=0, children=[]),
                sr.Node(kind="var", payload=1, children=[]),
            ],
        )
        out = sr.evaluate(tree, X)
        self.assertEqual(out.shape, (T,))
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out, 0.0)


class SpearmanFitnessTests(unittest.TestCase):
    """Spearman fitness recognises perfect monotone agreement."""

    def test_identity_has_fitness_near_one(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(120, 2))
        y = X[:, 0].copy()
        tree = sr.Node(kind="var", payload=0, children=[])
        f = sr.fitness(tree, X, y)
        self.assertGreater(f, 0.99)

    def test_monotone_transform_still_high(self) -> None:
        # Spearman is rank-based: any monotone transform stays at ~1.
        rng = np.random.default_rng(2)
        X = rng.uniform(0.5, 2.0, size=(80, 1))
        y = np.exp(X[:, 0])
        tree = sr.Node(kind="var", payload=0, children=[])
        f = sr.fitness(tree, X, y)
        self.assertGreater(f, 0.99)


class EvolveRecoveryTests(unittest.TestCase):
    """GP recovers a high-correlation proxy for y = X0 * X1."""

    def test_evolve_recovers_product(self) -> None:
        rng = np.random.default_rng(0)
        T = 200
        X = rng.uniform(0.5, 1.5, size=(T, 3))  # 3rd col is noise
        y = X[:, 0] * X[:, 1]
        result = sr.evolve(
            X, y, n_features=3,
            population_size=300, generations=20, seed=0,
        )
        self.assertIn("best_tree", result)
        self.assertIn("best_fitness", result)
        self.assertIn("best_formula", result)
        self.assertIn("history", result)
        self.assertIsInstance(result["best_formula"], str)
        self.assertGreater(result["best_fitness"], 0.85)


class HistoryMonotoneTests(unittest.TestCase):
    """Best-fitness history never decreases generation-over-generation."""

    def test_history_is_non_decreasing(self) -> None:
        rng = np.random.default_rng(7)
        T = 120
        X = rng.normal(size=(T, 2))
        y = X[:, 0] + 0.5 * X[:, 1]
        result = sr.evolve(
            X, y, n_features=2,
            population_size=80, generations=10, seed=3,
        )
        hist = result["history"]
        self.assertEqual(len(hist), 10)
        for prev, curr in zip(hist, hist[1:]):
            self.assertGreaterEqual(curr, prev - 1e-12)


class CrossoverValidityTests(unittest.TestCase):
    """Crossover produces a finite-depth valid tree."""

    def test_crossover_yields_valid_tree(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 4))
        for trial in range(10):
            a = sr.random_tree(rng, n_features=4, max_depth=3)
            b = sr.random_tree(rng, n_features=4, max_depth=3)
            child = sr.crossover(rng, a, b)
            self.assertIsInstance(child, sr.Node)
            d = sr.tree_depth(child)
            # finite, bounded by sum of parent depths (no infinite recursion)
            self.assertGreaterEqual(d, 1)
            self.assertLessEqual(d, sr.tree_depth(a) + sr.tree_depth(b))
            # Evaluates to a finite vector of the right shape.
            out = sr.evaluate(child, X)
            self.assertEqual(out.shape, (50,))
            self.assertTrue(np.all(np.isfinite(out)))


if __name__ == "__main__":
    unittest.main()
