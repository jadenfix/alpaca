"""Known-truth tests for the RIE-cleaned residual mean-reversion strategy.

We exercise the strategy three ways:
  * directly (as a pure function)
  * through the `factory` closure
  * through the walk-forward harness on synthetic universes whose
    statistical structure is known a priori
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


class RMTResidualTests(unittest.TestCase):
    # ----- 1. pure noise should yield ~zero PnL ---------------------------
    def test_pure_noise_near_zero_pnl(self):
        from walkforward import walk_forward, WalkForwardConfig
        from rmt_residual import factory

        rng = _seed_rng(101)
        rets = rng.normal(0.0, 0.01, size=(600, 6))
        cfg = WalkForwardConfig(train_window=200, test_window=60, step=20,
                                cost_bps=0.0, slippage_bps=0.0,
                                min_history=120)
        strat = factory(z_threshold=1.5, window=60)
        r = walk_forward(rets, [f"s{i}" for i in range(6)], strat, cfg)
        self.assertLess(abs(r.annual_return), 0.05,
                        msg=f"noise alpha leaked: ann_ret={r.annual_return}")

    # ----- 2. planted AR(1) negative residuals -> positive Sharpe ---------
    def test_planted_mean_reversion_positive_sharpe(self):
        from walkforward import walk_forward, WalkForwardConfig
        from rmt_residual import factory

        rng = _seed_rng(202)
        T, N = 800, 5
        factor = rng.normal(0.0, 0.012, size=T)
        betas = rng.normal(1.0, 0.2, size=N)

        # AR(1) residuals with strong negative autocorrelation: ε_t = -0.5 ε_{t-1} + η_t
        sigma_eta = 0.008
        eps = np.zeros((T, N))
        eps[0] = rng.normal(0.0, sigma_eta, size=N)
        ar_coef = -0.5
        for tt in range(1, T):
            eps[tt] = ar_coef * eps[tt - 1] + rng.normal(0.0, sigma_eta, size=N)

        rets = (factor[:, None] * betas[None, :]) + eps

        cfg = WalkForwardConfig(train_window=200, test_window=20, step=1,
                                cost_bps=0.0, slippage_bps=0.0,
                                min_history=120)
        strat = factory(n_factors=1, z_threshold=0.5, window=60)
        r = walk_forward(rets, [f"s{i}" for i in range(N)], strat, cfg)
        self.assertGreater(r.sharpe, 0.5,
                           msg=f"expected mean-reversion alpha, got Sharpe={r.sharpe:.3f}")

    # ----- 3. direct call sanity: shape, sum-to-zero, gross<=1 ------------
    def test_direct_call_returns_valid_weights(self):
        from rmt_residual import rmt_residual_strategy

        rng = _seed_rng(303)
        train = rng.normal(0.0, 0.01, size=(300, 4))
        w = rmt_residual_strategy(train, [f"s{i}" for i in range(4)])
        self.assertEqual(w.shape, (4,))
        self.assertAlmostEqual(float(np.sum(w)), 0.0, places=10)
        self.assertLessEqual(float(np.sum(np.abs(w))), 1.0001)

    # ----- 4. too-short train returns zeros -------------------------------
    def test_too_short_train_returns_zeros(self):
        from rmt_residual import rmt_residual_strategy

        rng = _seed_rng(404)
        train = rng.normal(0.0, 0.01, size=(20, 4))
        w = rmt_residual_strategy(train, [f"s{i}" for i in range(4)])
        self.assertEqual(w.shape, (4,))
        self.assertTrue(np.all(w == 0.0))

    # ----- 5. factory parameter binding: higher threshold => smaller gross
    def test_factory_threshold_binding(self):
        from rmt_residual import factory

        rng = _seed_rng(505)
        # Build data with REAL residual signal so a low z_threshold actually
        # fires trades. Pure noise can leave both gross exposures at zero.
        T, N = 400, 5
        factor = rng.normal(0.0, 0.01, size=T)
        betas = rng.normal(1.0, 0.2, size=N)
        eps = np.zeros((T, N))
        eps[0] = rng.normal(0.0, 0.008, size=N)
        for tt in range(1, T):
            eps[tt] = -0.5 * eps[tt - 1] + rng.normal(0.0, 0.008, size=N)
        train = factor[:, None] * betas[None, :] + eps
        names = [f"s{i}" for i in range(N)]

        strat_loose = factory(z_threshold=0.5)
        strat_strict = factory(z_threshold=5.0)
        w_loose = strat_loose(train, names)
        w_strict = strat_strict(train, names)
        gross_loose = float(np.sum(np.abs(w_loose)))
        gross_strict = float(np.sum(np.abs(w_strict)))
        # Stricter threshold cannot produce MORE gross exposure.
        self.assertLessEqual(gross_strict, gross_loose + 1e-12)
        # And on this synthetic, the loose closure should actually trade.
        self.assertGreater(gross_loose, 0.0)


if __name__ == "__main__":
    unittest.main()
