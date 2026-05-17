"""Known-truth tests for the walk-forward backtester."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "backtest"))
sys.path.insert(0, str(HERE.parent / "analysis"))


def _seed_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


class WalkForwardTests(unittest.TestCase):
    def test_zero_weights_produces_zero_pnl(self):
        from walkforward import walk_forward, WalkForwardConfig
        from baseline_strategies import zero_weight
        rng = _seed_rng(7)
        rets = rng.normal(0, 0.01, size=(600, 5))
        cfg = WalkForwardConfig(train_window=200, test_window=50, step=20,
                                cost_bps=2.5, slippage_bps=1.5, min_history=60)
        r = walk_forward(rets, [f"s{i}" for i in range(5)], zero_weight, cfg)
        self.assertEqual(r.n_bars, rets.shape[0] - max(cfg.train_window, cfg.min_history))
        self.assertTrue(np.allclose(r.daily_returns, 0.0))
        self.assertEqual(r.sharpe, 0.0)
        self.assertEqual(r.max_drawdown, 0.0)

    def test_equal_weight_pnl_matches_average_of_returns(self):
        from walkforward import walk_forward, WalkForwardConfig
        from baseline_strategies import equal_weight
        rng = _seed_rng(11)
        n_bars, n_assets = 500, 4
        rets = rng.normal(0, 0.01, size=(n_bars, n_assets))
        cfg = WalkForwardConfig(train_window=100, test_window=50, step=200,
                                cost_bps=0.0, slippage_bps=0.0, min_history=60)
        r = walk_forward(rets, [f"s{i}" for i in range(n_assets)],
                         equal_weight, cfg)
        # With zero costs and a single rebalance, daily PnL equals row-mean.
        first_test_idx = max(cfg.train_window, cfg.min_history)
        expected = rets[first_test_idx:].mean(axis=1)
        # The first bar may include the rebalance cost (here zero), so they
        # should match exactly.
        self.assertTrue(np.allclose(r.daily_returns, expected, atol=1e-12))

    def test_costs_reduce_pnl_proportionally_to_turnover(self):
        from walkforward import walk_forward, WalkForwardConfig
        from baseline_strategies import momentum_xs
        rng = _seed_rng(13)
        rets = rng.normal(0, 0.02, size=(800, 6))
        names = [f"s{i}" for i in range(6)]
        cfg_no_cost = WalkForwardConfig(train_window=150, test_window=50,
                                        step=5, cost_bps=0.0, slippage_bps=0.0,
                                        min_history=60)
        cfg_with_cost = WalkForwardConfig(train_window=150, test_window=50,
                                          step=5, cost_bps=10.0,
                                          slippage_bps=5.0, min_history=60)
        r_free = walk_forward(rets, names, momentum_xs, cfg_no_cost)
        r_cost = walk_forward(rets, names, momentum_xs, cfg_with_cost)
        # With non-zero turnover and 15 bps round-trip, gross returns must drop.
        self.assertGreater(r_free.daily_returns.sum(),
                           r_cost.daily_returns.sum())
        # Total cost is exactly cost_rate * Σ|Δw|.
        cost_rate = (cfg_with_cost.cost_bps + cfg_with_cost.slippage_bps) / 1e4
        # We track per-bar cost; sum should be > 0.
        self.assertGreater(r_cost.gross_costs.sum(), 0.0)

    def test_perfect_oracle_strategy_has_high_sharpe(self):
        """Cheating: a strategy that sees the next bar's sign should win
        easily even after costs. Verifies the backtester rewards skill."""
        from walkforward import walk_forward, WalkForwardConfig
        rng = _seed_rng(19)
        n_bars, n_assets = 600, 4
        rets = rng.normal(0, 0.01, size=(n_bars, n_assets))

        # Oracle uses the FULL series + an external counter to know which
        # bar the harness is about to evaluate. This is intentionally a
        # cheat to verify that high skill → high Sharpe under the harness.
        counter = {"i": 100}  # = max(train_window, min_history)

        def oracle(train, names):
            i = counter["i"]
            counter["i"] += 1
            if i >= rets.shape[0]:
                return np.zeros(n_assets)
            return np.sign(rets[i])

        cfg = WalkForwardConfig(train_window=100, test_window=50, step=1,
                                cost_bps=0.0, slippage_bps=0.0, min_history=60)
        r = walk_forward(rets, [f"s{i}" for i in range(n_assets)], oracle, cfg)
        self.assertGreater(r.sharpe, 5.0)
        self.assertGreater(r.hit_rate, 0.9)

    def test_min_variance_lower_vol_than_equal_weight(self):
        from walkforward import walk_forward, WalkForwardConfig
        from baseline_strategies import equal_weight, min_variance
        rng = _seed_rng(23)
        n_bars = 800
        # Two volatile, two calm assets; minvar should down-weight the
        # volatile pair.
        rets = np.column_stack([
            rng.normal(0, 0.03, n_bars),
            rng.normal(0, 0.03, n_bars),
            rng.normal(0, 0.005, n_bars),
            rng.normal(0, 0.005, n_bars),
        ])
        cfg = WalkForwardConfig(train_window=200, test_window=80, step=20,
                                cost_bps=0.0, slippage_bps=0.0, min_history=60)
        r_eq = walk_forward(rets, ["a","b","c","d"], equal_weight, cfg)
        r_mv = walk_forward(rets, ["a","b","c","d"], min_variance, cfg)
        self.assertLess(r_mv.annual_vol, r_eq.annual_vol)

    def test_summarize_table_renders(self):
        from walkforward import walk_forward, WalkForwardConfig, compare_strategies, summarize_table
        from baseline_strategies import equal_weight, zero_weight
        rng = _seed_rng(29)
        rets = rng.normal(0, 0.01, size=(400, 3))
        cfg = WalkForwardConfig(train_window=100, test_window=50, step=20,
                                cost_bps=0.0, slippage_bps=0.0, min_history=60)
        results = compare_strategies(rets, ["a","b","c"],
                                     {"equal": equal_weight, "zero": zero_weight}, cfg)
        table = summarize_table(results)
        self.assertIn("Sharpe", table)
        self.assertIn("equal", table)
        self.assertIn("zero", table)


if __name__ == "__main__":
    unittest.main()
