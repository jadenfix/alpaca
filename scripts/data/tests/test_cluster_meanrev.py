"""Known-truth tests for the within-cluster mean-reversion strategy.

These tests exercise the strategy both directly (unit-level invariants
like weight shape and gross exposure) and through the walk-forward
harness (behavioral tests on synthetic processes with a known answer).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "backtest"))
sys.path.insert(0, str(HERE.parent.parent / "backtest" / "strategies"))
sys.path.insert(0, str(HERE.parent / "analysis"))


def _seed_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _two_cluster_mean_reverting(seed: int, T: int = 800, phi: float = -0.7,
                                idio_sig: float = 0.01,
                                factor_sig: float = 0.02) -> np.ndarray:
    """Two anti-correlated cluster factors (so the *market* averages out)
    plus mean-reverting idiosyncratic AR(1) noise per asset. The result:

      * Cluster A (assets 0..3) co-moves with factor `fa`.
      * Cluster B (assets 4..7) co-moves with `-fa + fb`.
      * Within each cluster, names differ only via AR(1) idiosyncratic
        returns with strong negative autocorrelation — i.e. each name's
        deviation from its cluster mean mean-reverts bar-to-bar.
      * Σ market returns drift ≈ 0 by construction, so the market-z
        filter does NOT block the strategy.
    """
    rng = _seed_rng(seed)
    n = 8
    fa = rng.normal(0.0, factor_sig, T)
    fb = rng.normal(0.0, factor_sig, T)
    idio = np.zeros((T, n))
    prev = np.zeros(n)
    for t in range(T):
        eps = rng.normal(0.0, idio_sig, n)
        cur = phi * prev + eps
        idio[t] = cur
        prev = cur
    rets = np.zeros((T, n))
    rets[:, :4] = fa[:, None] + idio[:, :4]
    rets[:, 4:] = -fa[:, None] + fb[:, None] + idio[:, 4:]
    return rets


class ClusterMeanRevTests(unittest.TestCase):

    # ----- 1. Pure noise → near-zero PnL ----------------------------
    def test_pure_noise_near_zero_pnl(self):
        """With i.i.d. Gaussian returns the correlation distances stay
        near sqrt(2) ≈ 1.414, no MST edges fall under `distance_cutoff`,
        no clusters form, and the strategy holds zero exposure for every
        rebalance. Annual return must be tiny in absolute value.
        """
        from walkforward import walk_forward, WalkForwardConfig
        from cluster_meanrev import cluster_meanrev_strategy

        rng = _seed_rng(7)
        rets = rng.normal(0.0, 0.01, size=(600, 6))
        cfg = WalkForwardConfig(train_window=200, test_window=63, step=5,
                                cost_bps=0.5, slippage_bps=0.5,
                                min_history=60)
        r = walk_forward(rets, [f"s{i}" for i in range(6)],
                         cluster_meanrev_strategy, cfg)
        self.assertLess(abs(r.annual_return), 0.05)

    # ----- 2. Two clusters → positive Sharpe -----------------------
    def test_two_cluster_mean_reversion_positive_sharpe(self):
        """When the synthetic process has clear intra-cluster structure
        AND mean-reverting idiosyncratic deviations, the strategy should
        capture relative-value profits with Sharpe > 0.3.

        We use a lower `within_z_threshold` (0.8) because a size-4
        cluster's |z| is bounded above by sqrt(3) ≈ 1.732 — see the
        strategy docstring for the geometry. We also relax `market_z_max`
        to 1.0 because the two anti-correlated factors leave a residual
        market signal small but not always below 0.5.
        """
        from walkforward import walk_forward, WalkForwardConfig
        from cluster_meanrev import factory

        rets = _two_cluster_mean_reverting(seed=42)
        names = [f"A{i}" for i in range(4)] + [f"B{i}" for i in range(4)]
        strat = factory(distance_cutoff=1.0, within_z_threshold=0.8,
                        market_z_max=1.0, lookback=3)
        cfg = WalkForwardConfig(train_window=200, test_window=63, step=5,
                                cost_bps=0.5, slippage_bps=0.5,
                                min_history=60)
        r = walk_forward(rets, names, strat, cfg)
        # The strategy MUST be active — verify rebalances happened.
        self.assertGreater(r.n_rebalances, 0)
        self.assertGreater(r.avg_turnover, 0.0,
                           "expected some turnover from intra-cluster trades")
        self.assertGreater(r.sharpe, 0.3,
                           f"expected sharpe > 0.3, got {r.sharpe:.3f}")

    # ----- 3. Strong market trend → filter blocks all trading ------
    def test_market_filter_blocks_in_trending_market(self):
        """Inject a positive drift on every bar across every asset, so
        the equal-weight market return is consistently large relative to
        its rolling std. The market-z filter should refuse to trade.
        """
        from walkforward import walk_forward, WalkForwardConfig
        from cluster_meanrev import factory

        rng = _seed_rng(101)
        T, N = 500, 6
        drift = 0.005
        rets = rng.normal(0.0, 0.005, size=(T, N)) + drift
        strat = factory(distance_cutoff=1.0, within_z_threshold=0.8,
                        market_z_max=0.5, lookback=5)
        cfg = WalkForwardConfig(train_window=200, test_window=63, step=5,
                                cost_bps=0.5, slippage_bps=0.5,
                                min_history=60)
        r = walk_forward(rets, [f"s{i}" for i in range(N)], strat, cfg)
        # The filter must keep weights identically zero across the run.
        total_gross = float(np.sum(np.abs(r.weights)))
        self.assertEqual(total_gross, 0.0,
                         f"expected zero exposure, got total |w| = {total_gross}")
        self.assertEqual(r.avg_turnover, 0.0)

    # ----- 4. Direct call sanity ------------------------------------
    def test_direct_call_shape_and_gross(self):
        """Strategy returns (N,) and obeys the Σ|w| ≤ 1.001 invariant.
        The bound is 1 + tiny float-rounding slack.
        """
        from cluster_meanrev import cluster_meanrev_strategy, factory

        rng = _seed_rng(2026)
        N = 10
        rets = rng.normal(0.0, 0.01, size=(400, N))
        names = [f"x{i}" for i in range(N)]
        w = cluster_meanrev_strategy(rets, names)
        self.assertEqual(w.shape, (N,))
        self.assertLessEqual(float(np.sum(np.abs(w))), 1.001)

        # And the factory closure returns the same invariants.
        w2 = factory()(rets, names)
        self.assertEqual(w2.shape, (N,))
        self.assertLessEqual(float(np.sum(np.abs(w2))), 1.001)

    # ----- 5. Empty cluster output ---------------------------------
    def test_empty_clusters_returns_zeros(self):
        """When `distance_cutoff=0.0` every MST edge fails the strict
        `d < cutoff` test (distances are non-negative), `mst_clusters`
        returns an empty list, and the strategy must immediately return
        a zero vector without inspecting market_z or per-cluster z's.
        """
        from cluster_meanrev import cluster_meanrev_strategy

        # Construct returns with REAL clusters so we know the only thing
        # squashing the output is the cutoff, not lack of structure.
        rets = _two_cluster_mean_reverting(seed=3, T=300)
        names = [f"A{i}" for i in range(4)] + [f"B{i}" for i in range(4)]
        w = cluster_meanrev_strategy(rets, names, distance_cutoff=0.0)
        self.assertEqual(w.shape, (8,))
        self.assertTrue(np.allclose(w, 0.0))

    # ----- bonus: short history guard ------------------------------
    def test_short_history_returns_zeros(self):
        """Fewer than 60 train rows: defensive guard kicks in and the
        strategy returns zeros without attempting any computation.
        """
        from cluster_meanrev import cluster_meanrev_strategy

        rng = _seed_rng(5)
        rets = rng.normal(0.0, 0.01, size=(40, 5))
        w = cluster_meanrev_strategy(rets, [f"s{i}" for i in range(5)])
        self.assertTrue(np.allclose(w, 0.0))


if __name__ == "__main__":
    unittest.main()
