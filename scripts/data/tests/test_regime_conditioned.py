"""Known-truth tests for regime_conditioned.py.

Each test fails on a no-op implementation, so they validate the math,
not just the API surface.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# Make the analysis package importable — same convention as test_analysis.py.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE.parent))


def _seed_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


class RegimeCoordinateTests(unittest.TestCase):
    def test_output_length_and_nan_prefix(self):
        import regime_conditioned as rc
        rng = _seed_rng(7)
        T, N, w = 200, 5, 60
        rets = rng.normal(size=(T, N)) * 0.01
        reg = rc.regime_coordinate(rets, window=w)
        self.assertEqual(reg.shape, (T,))
        # Bars before the rolling window fills must be NaN.
        self.assertTrue(np.all(np.isnan(reg[:w - 1])))
        # Bars from t = w - 1 onward must be finite.
        self.assertTrue(np.all(np.isfinite(reg[w - 1:])))

    def test_no_lookahead(self):
        """regime[t] must depend only on rets[:t+1]. Perturbing rets[t+1:]
        must leave regime[:t+1] unchanged."""
        import regime_conditioned as rc
        rng = _seed_rng(11)
        T, N, w = 150, 4, 30
        rets = rng.normal(size=(T, N)) * 0.01
        reg_a = rc.regime_coordinate(rets, window=w)
        # Pick a probe index after the rolling window has filled.
        t_probe = 80
        rets2 = rets.copy()
        # Aggressively perturb every bar AFTER t_probe.
        rets2[t_probe + 1:, :] += rng.normal(size=(T - t_probe - 1, N)) * 10.0
        reg_b = rc.regime_coordinate(rets2, window=w)
        # regime[:t_probe + 1] must be identical (NaNs equal NaNs).
        np.testing.assert_array_equal(
            np.where(np.isnan(reg_a[:t_probe + 1]), -1.0, reg_a[:t_probe + 1]),
            np.where(np.isnan(reg_b[:t_probe + 1]), -1.0, reg_b[:t_probe + 1]),
        )
        # And to make sure we actually moved something, regime values
        # AFTER t_probe should differ.
        self.assertFalse(np.allclose(reg_a[t_probe + 1:], reg_b[t_probe + 1:]))


class RegimeBucketsTests(unittest.TestCase):
    def test_buckets_cover_all_valid_entries(self):
        import regime_conditioned as rc
        rng = _seed_rng(7)
        # Uniform data => quantile buckets should be near-equal sized.
        vals = rng.uniform(0.0, 1.0, size=5000)
        b = rc.regime_buckets(vals, n_buckets=5)
        self.assertEqual(b.shape, vals.shape)
        # No NaNs in input => no -1 in output.
        self.assertTrue(np.all(b >= 0))
        # All five buckets populated.
        self.assertEqual(set(np.unique(b).tolist()), {0, 1, 2, 3, 4})
        # Roughly equal sized within ±20 %.
        counts = np.bincount(b, minlength=5)
        target = vals.size / 5.0
        for c in counts:
            self.assertLess(abs(c - target) / target, 0.20)

    def test_buckets_handle_nans(self):
        import regime_conditioned as rc
        vals = np.array([np.nan, 0.1, 0.5, 0.9, np.nan, 0.3, 0.7, 0.2, 0.8, 0.4])
        b = rc.regime_buckets(vals, n_buckets=4)
        # NaN entries must map to -1.
        self.assertEqual(b[0], -1)
        self.assertEqual(b[4], -1)
        # Non-NaN entries must be in [0, n_buckets - 1].
        for i, v in enumerate(vals):
            if not np.isnan(v):
                self.assertGreaterEqual(b[i], 0)
                self.assertLess(b[i], 4)


class ConditionalSignPredictabilityTests(unittest.TestCase):
    def test_isolates_high_vol_regime(self):
        """Build dst such that dst[t+1] = src[t] in the high-vol bucket
        and noise elsewhere. Expect the high bucket to fire and the low
        bucket to look like noise.
        """
        import regime_conditioned as rc
        rng = _seed_rng(13)
        T = 3000
        N = 4
        window = 60
        # Construct a returns matrix whose vol drifts smoothly, so
        # quantile bucketing cleanly separates high- and low-vol bars.
        vol_path = np.linspace(0.005, 0.05, T)
        # Add a little stochastic wiggle so buckets aren't trivially monotone.
        vol_path += rng.normal(size=T) * 0.002
        vol_path = np.clip(vol_path, 1e-4, None)
        market_rets = rng.normal(size=(T, N)) * vol_path[:, None]
        regime = rc.regime_coordinate(market_rets, window=window)
        bucket = rc.regime_buckets(regime, n_buckets=5)
        # Build src/dst independently of market_rets so bucketing is the
        # ONLY thing tying the signal to the regime.
        src = rng.normal(size=T)
        dst = rng.normal(size=T) * 0.5  # baseline noise
        # Where the bucket-at-time-t is the top quintile (4), set
        # dst[t+1] = src[t] so sign(src[t]) perfectly predicts sign(dst[t+1]).
        for t in range(T - 1):
            if bucket[t] == 4:
                dst[t + 1] = src[t]
        rows = rc.conditional_sign_predictability(src, dst, bucket)
        # Re-index by bucket for assertions.
        by_b = {r["bucket"]: r for r in rows}
        # High-vol bucket fires strongly.
        self.assertIn(4, by_b)
        self.assertGreater(by_b[4]["hit_rate"], 0.65)
        self.assertTrue(by_b[4]["rule_fires"])
        # Low-vol bucket looks like noise.
        self.assertIn(0, by_b)
        self.assertLess(abs(by_b[0]["hit_rate"] - 0.5), 0.10)


class ScreenConditionalEdgesTests(unittest.TestCase):
    def _build_returns_with_planted_edges(self, fires_in_buckets: set[int],
                                          seed: int = 17):
        """Build a returns matrix where the (S, D) edge fires only in
        regime buckets ∈ `fires_in_buckets`.

        Design:
          1. A smooth vol path drives all columns so the rolling EW-vol
             regime coordinate spans all five quantile buckets.
          2. Plant the signal AFTER assembling `rets` and computing its
             actual buckets — this guarantees the firing buckets match
             exactly what `screen_conditional_edges` will compute.
          3. In firing buckets we deterministically force
             sign(dst[t+1]) = sign(src[t]); planting a sign shift uses
             scale vol_path[t+1] so dst's magnitude is unchanged and
             the EW-vol regime is preserved.
        """
        import regime_conditioned as rc
        rng = _seed_rng(seed)
        T = 4000
        window = 60
        # Smooth vol path so quantile-bucketing has well-separated bins.
        vol_path = np.linspace(0.01, 0.10, T) + rng.normal(size=T) * 0.005
        vol_path = np.clip(vol_path, 1e-4, None)
        # Background columns set the regime coordinate magnitude.
        bg = rng.normal(size=(T, 8)) * vol_path[:, None]
        src = rng.normal(size=T) * vol_path
        dst = rng.normal(size=T) * vol_path
        # Iterate to a fixed point: compute buckets from the current
        # rets, replant the signal, repeat until buckets stop changing.
        # In firing buckets: sign(dst[t+1]) = +sign(src[t]) (perfect
        # positive predictor). In non-firing buckets: sign(dst[t+1]) =
        # -sign(src[t]) (perfect ANTI-predictor → hit_rate ≈ 0, well
        # under the 0.52 threshold). This pins each bucket's firing
        # status deterministically and removes sampling-noise ambiguity
        # at the keep-rule threshold.
        bucket = None
        for _ in range(20):
            rets = np.column_stack([src, dst, bg])
            regime = rc.regime_coordinate(rets, window=window)
            new_bucket = rc.regime_buckets(regime, n_buckets=5)
            if bucket is not None and np.array_equal(bucket, new_bucket):
                break
            bucket = new_bucket
            for t in range(T - 1):
                if bucket[t] in fires_in_buckets:
                    dst[t + 1] = np.sign(src[t]) * vol_path[t + 1]
                elif bucket[t] >= 0:
                    dst[t + 1] = -np.sign(src[t]) * vol_path[t + 1]
        rets = np.column_stack([src, dst, bg])
        names = ["S", "D"] + [f"X{i}" for i in range(8)]
        return rets, names

    def test_keeps_edge_firing_in_3_of_5(self):
        import regime_conditioned as rc
        rets, names = self._build_returns_with_planted_edges({2, 3, 4}, seed=17)
        kept = rc.screen_conditional_edges(
            rets, names, candidate_edges=[("S", "D")],
            n_buckets=5, min_buckets_firing=3,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["src"], "S")
        self.assertEqual(kept[0]["dst"], "D")
        self.assertGreaterEqual(kept[0]["n_buckets_firing"], 3)

    def test_drops_edge_firing_in_only_2_of_5(self):
        import regime_conditioned as rc
        rets, names = self._build_returns_with_planted_edges({3, 4}, seed=19)
        kept = rc.screen_conditional_edges(
            rets, names, candidate_edges=[("S", "D")],
            n_buckets=5, min_buckets_firing=3,
        )
        # The keep rule requires 3 of 5; with only 2 planted buckets,
        # the edge must be dropped.
        self.assertEqual(kept, [])


if __name__ == "__main__":
    unittest.main()
