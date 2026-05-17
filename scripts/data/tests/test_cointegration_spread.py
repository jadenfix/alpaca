"""Known-truth tests for the Engle-Granger cointegration spread strategy."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "backtest"))
sys.path.insert(0, str(HERE.parent.parent / "backtest" / "strategies"))
sys.path.insert(0, str(HERE.parent / "analysis"))


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _ar1(rng: np.random.Generator, phi: float, n: int,
         sigma: float = 1.0) -> np.ndarray:
    """Generate a stationary AR(1) of length n: x_t = phi*x_{t-1} + eps."""
    x = np.zeros(n)
    eps = rng.normal(0.0, sigma, size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def _random_walk(rng: np.random.Generator, n: int,
                 sigma: float = 1.0) -> np.ndarray:
    """Random-walk price series of length n starting at 0."""
    return np.cumsum(rng.normal(0.0, sigma, size=n))


class AdfTstatTests(unittest.TestCase):
    def test_stationary_ar1_rejects_unit_root(self):
        from cointegration_spread import adf_tstat
        rng = _rng(7)
        x = _ar1(rng, phi=0.3, n=250)
        t = adf_tstat(x, max_lag=1)
        self.assertLess(t, -2.86,
                        f"stationary AR(1) phi=0.3 should reject; got t={t:.3f}")

    def test_random_walk_does_not_reject(self):
        from cointegration_spread import adf_tstat
        rng = _rng(7)
        rw = _random_walk(rng, n=250)
        t = adf_tstat(rw, max_lag=1)
        self.assertGreater(t, -2.86,
                           f"random walk should NOT reject; got t={t:.3f}")


class EngleGrangerPairTests(unittest.TestCase):
    def test_recovers_known_hedge_ratio(self):
        from cointegration_spread import engle_granger_pair
        rng = _rng(3)
        n = 400
        price_b = _random_walk(rng, n=n, sigma=1.0)
        noise = rng.normal(0.0, 0.5, size=n)
        price_a = 2.0 * price_b + noise
        info = engle_granger_pair(price_a, price_b, adf_threshold=-2.86)
        self.assertAlmostEqual(info["hedge_ratio"], 2.0, delta=0.2)
        self.assertLess(info["adf_tstat"], -2.86)
        self.assertTrue(info["is_cointegrated"])

    def test_no_false_positive_on_independent_walks(self):
        from cointegration_spread import engle_granger_pair
        flags = []
        for seed in (101, 202, 303, 404, 505):
            rng = _rng(seed)
            a = _random_walk(rng, n=300, sigma=1.0)
            b = _random_walk(rng, n=300, sigma=1.0)
            info = engle_granger_pair(a, b, adf_threshold=-2.86)
            flags.append(info["is_cointegrated"])
        non_cointegrated = sum(1 for f in flags if not f)
        self.assertGreaterEqual(
            non_cointegrated, 3,
            f"expected >=3 of 5 independent walks to be non-cointegrated; "
            f"got flags={flags}",
        )


class StrategyTests(unittest.TestCase):
    def test_returns_zeros_for_tiny_train(self):
        from cointegration_spread import cointegration_spread_strategy
        rng = _rng(1)
        rets = rng.normal(0, 0.01, size=(20, 4))
        w = cointegration_spread_strategy(rets, [f"s{i}" for i in range(4)])
        self.assertEqual(w.shape, (4,))
        self.assertTrue(np.allclose(w, 0.0))

    def test_shape_and_gross_exposure(self):
        from cointegration_spread import cointegration_spread_strategy
        rng = _rng(2)
        # Build a 4-asset universe where assets 0 and 1 cointegrate so the
        # strategy actually produces non-zero weights and we can sanity check
        # the gross-exposure normalization.
        n = 300
        log_b = _random_walk(rng, n=n, sigma=0.02)
        spread = _ar1(rng, phi=0.7, n=n, sigma=0.02)
        log_a = 1.8 * log_b + spread
        log_c = _random_walk(rng, n=n, sigma=0.02)
        log_d = _random_walk(rng, n=n, sigma=0.02)
        log_prices = np.column_stack([log_a, log_b, log_c, log_d])
        rets = np.diff(log_prices, axis=0)
        w = cointegration_spread_strategy(rets, ["a", "b", "c", "d"],
                                          z_entry=0.5)
        self.assertEqual(w.shape, (4,))
        self.assertLessEqual(float(np.sum(np.abs(w))), 1.001)

    def test_planted_pair_generates_pnl(self):
        from walkforward import walk_forward, WalkForwardConfig
        from cointegration_spread import factory

        rng = _rng(17)
        n_bars = 700
        n_assets = 5

        # Strong cointegrated pair: log_a = beta*log_b + AR(1) spread.
        beta_true = 1.8
        log_b = _random_walk(rng, n=n_bars, sigma=0.015)
        spread = _ar1(rng, phi=0.75, n=n_bars, sigma=0.015)
        log_a = beta_true * log_b + spread

        # Three independent random-walk log-price series.
        log_others = np.column_stack([
            _random_walk(rng, n=n_bars, sigma=0.015) for _ in range(n_assets - 2)
        ])
        log_prices = np.column_stack([log_a, log_b, log_others])
        rets = np.diff(log_prices, axis=0)
        names = [f"s{i}" for i in range(n_assets)]

        cfg = WalkForwardConfig(
            train_window=200, test_window=40, step=10,
            cost_bps=0.0, slippage_bps=0.0, min_history=120,
        )
        strat = factory(z_entry=1.0, adf_threshold=-2.86, max_pairs=10)
        result = walk_forward(rets, names, strat, cfg)
        self.assertGreater(result.n_bars, 0)
        # Either annualized return or Sharpe should be positive enough to
        # demonstrate the planted alpha is captured.
        ok = (result.annual_return > 0.0) or (result.sharpe > 0.3)
        self.assertTrue(
            ok,
            f"expected ann_ret>0 OR Sharpe>0.3; "
            f"got ann_ret={result.annual_return:.4f}, sharpe={result.sharpe:.3f}",
        )

    def test_direct_call_shape_sanity(self):
        from cointegration_spread import cointegration_spread_strategy
        rng = _rng(31)
        n_bars, n_assets = 250, 6
        rets = rng.normal(0, 0.01, size=(n_bars, n_assets))
        w = cointegration_spread_strategy(rets, [f"s{i}" for i in range(n_assets)])
        self.assertEqual(w.shape, (n_assets,))
        self.assertLessEqual(float(np.sum(np.abs(w))), 1.001)


if __name__ == "__main__":
    unittest.main()
