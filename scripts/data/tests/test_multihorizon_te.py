"""Known-truth tests for ``multihorizon_te``.

Each test constructs a synthetic series with a planted causal structure
whose answer is known a priori; the test would FAIL on a no-op or
constant-returning implementation.
"""
from __future__ import annotations

import math
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


def _hit_by_horizon(rows: list[dict]) -> dict[int, float]:
    """Collapse rows to {h -> max hit_rate over (k, l)}."""
    out: dict[int, float] = {}
    for r in rows:
        h = int(r["h"])
        hr = float(r["hit_rate"])
        if h not in out or hr > out[h]:
            out[h] = hr
    return out


class MultiHorizonTeTests(unittest.TestCase):

    # ----- Test 1: strong h=1 lead -----
    def test_strong_h1_lead_dominates(self):
        import multihorizon_te as mh
        rng = _seed_rng(11)
        n = 1500
        src = rng.normal(size=n)
        dst = np.empty(n)
        dst[0] = rng.normal()
        # dst[t+1] = src[t] + 0.3 * noise  ⇒  sign-predictable at h=1.
        for t in range(n - 1):
            dst[t + 1] = src[t] + 0.3 * rng.normal()

        rows = mh.multi_horizon_te(src, dst,
                                   k_grid=[1, 2, 3], l_grid=[1, 2, 3],
                                   h_grid=[1, 5, 21])
        # Find the (k=1, l=1, h=1) row.
        row_111 = next(r for r in rows
                       if r["k"] == 1 and r["l"] == 1 and r["h"] == 1)
        self.assertGreater(row_111["hit_rate"], 0.70,
                           "h=1 hit-rate should be > 0.70 for strong lead")

        # The h=1 hit-rate must be strictly the highest across all h.
        per_h = _hit_by_horizon(rows)
        h1 = per_h[1]
        for h, hr in per_h.items():
            if h == 1:
                continue
            self.assertGreater(
                h1, hr,
                f"hit_rate(h=1)={h1:.3f} must exceed hit_rate(h={h})={hr:.3f}")

    # ----- Test 2: independent series — no lead at any horizon -----
    def test_independent_series_no_lead(self):
        import multihorizon_te as mh
        rng = _seed_rng(13)
        n = 1500
        src = rng.normal(size=n)
        dst = rng.normal(size=n)

        rows = mh.multi_horizon_te(src, dst,
                                   k_grid=[1, 2], l_grid=[1, 2],
                                   h_grid=[1, 5, 21])
        per_h = _hit_by_horizon(rows)
        for h, hr in per_h.items():
            self.assertLess(abs(hr - 0.5), 0.10,
                            f"hit_rate at h={h} = {hr:.3f} should be near 0.5")

        # And the binomial p-value should be high (non-significant) at
        # every horizon for the (k=1, l=1) row.
        for h in (1, 5, 21):
            r = next(r for r in rows
                     if r["k"] == 1 and r["l"] == 1 and r["h"] == h)
            self.assertGreater(r["binomial_p"], 0.05,
                               f"binomial_p at h={h} = {r['binomial_p']:.3f} "
                               "should be > 0.05 for independent series")

    # ----- Test 3: h=5 lead — h=1 should NOT fire -----
    def test_h5_lead_dominates_and_h1_does_not_fire(self):
        import multihorizon_te as mh
        rng = _seed_rng(17)
        n = 2000
        src = rng.normal(size=n)
        dst = rng.normal(size=n)
        # Plant lead at h=5: dst[t+5] = src[t] + 0.3*noise. Leave the
        # first 5 dst values as independent noise (no h=1 leak).
        for t in range(n - 5):
            dst[t + 5] = src[t] + 0.3 * rng.normal()

        rows = mh.multi_horizon_te(src, dst,
                                   k_grid=[1, 2, 3], l_grid=[1, 2, 3],
                                   h_grid=[1, 5, 21])
        row_115 = next(r for r in rows
                       if r["k"] == 1 and r["l"] == 1 and r["h"] == 5)
        self.assertGreater(row_115["hit_rate"], 0.65,
                           "h=5 hit-rate should be > 0.65 when lead is at h=5")

        per_h = _hit_by_horizon(rows)
        h5 = per_h[5]
        for h in (1, 21):
            self.assertGreater(
                h5, per_h[h],
                f"hit_rate(h=5)={h5:.3f} must exceed hit_rate(h={h})="
                f"{per_h[h]:.3f}")

        # h=1 should not fire — hit-rate near 0.5 and not significant.
        row_111 = next(r for r in rows
                       if r["k"] == 1 and r["l"] == 1 and r["h"] == 1)
        self.assertLess(abs(row_111["hit_rate"] - 0.5), 0.10,
                        f"h=1 hit-rate should be ~0.5, got {row_111['hit_rate']:.3f}")
        self.assertGreater(row_111["binomial_p"], 0.05,
                           "h=1 binomial_p should be insignificant (>0.05)")

    # ----- Test 4: screen_edges returns one monotone edge -----
    def test_screen_edges_returns_monotone_edge(self):
        import multihorizon_te as mh
        rng = _seed_rng(19)
        n = 1500
        src = rng.normal(size=n)
        # dst absorbs src with persistent memory — hit_rate decays
        # smoothly from h=1 to h=21.
        dst = np.zeros(n)
        for t in range(1, n):
            dst[t] = 0.6 * dst[t - 1] + 0.4 * src[t - 1] + 0.3 * rng.normal()

        rets = np.column_stack([src, dst])
        names = ["S", "D"]
        out = mh.screen_edges(rets, names, top_pairs=[("S", "D")],
                              k_grid=[1, 2], l_grid=[1, 2],
                              h_grid=[1, 5, 21])
        self.assertEqual(len(out), 1)
        edge = out[0]
        self.assertEqual(edge["src"], "S")
        self.assertEqual(edge["dst"], "D")
        self.assertTrue(edge["monotone"],
                        "screen_edges should flag the decaying-lead edge as monotone")
        # Sanity: the best horizon should be h=1 since that's where the
        # signal is strongest under monotone decay.
        self.assertEqual(edge["best_h"], 1,
                         f"best_h should be 1, got {edge['best_h']}")
        self.assertGreater(edge["hit_rate"], 0.55,
                           f"best hit_rate should be > 0.55, got {edge['hit_rate']:.3f}")

    # ----- Test 5: CCM convergence on Sugihara coupled logistic map -----
    def test_ccm_convergence_on_sugihara_logistic(self):
        import multihorizon_te as mh
        # Classic Sugihara (2012) coupled logistic map:
        #   x_{t+1} = x_t (rx - rx x_t - beta_xy y_t)
        #   y_{t+1} = y_t (ry - ry y_t - beta_yx x_t)
        # With beta_yx = 0 the system is one-way y -> x: x depends on
        # y but y does not depend on x. Hence x's manifold should
        # contain information about y → CCM predicting y from x's
        # manifold should converge with library size.
        rx, ry = 3.8, 3.5
        beta_xy, beta_yx = 0.02, 0.0
        n = 1500
        burn = 200
        x = np.zeros(n + burn)
        y = np.zeros(n + burn)
        x[0], y[0] = 0.4, 0.2
        for t in range(n + burn - 1):
            x[t + 1] = x[t] * (rx - rx * x[t] - beta_xy * y[t])
            y[t + 1] = y[t] * (ry - ry * y[t] - beta_yx * x[t])
        x = x[burn:]
        y = y[burn:]
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertTrue(np.all(np.isfinite(y)))

        # True direction is y -> x. multi_horizon_ccm(src=y, dst=x)
        # predicts y from x's manifold → that's the cross-map for
        # detecting y -> x causality. Should converge (positive corr
        # between library size and rho_xy).
        rows_true = mh.multi_horizon_ccm(
            y, x, e_grid=[3], tau=1,
            library_sizes=[50, 100, 200, 400, 800])
        self.assertTrue(rows_true, "multi_horizon_ccm returned no rows")
        # All rows for one E share convergence_score; just read the first.
        conv_true = rows_true[0]["convergence_score"]
        self.assertGreater(
            conv_true, 0.5,
            f"convergence_score for true direction (y -> x) should be > 0.5, "
            f"got {conv_true:.3f}")

        # Sanity: rho_xy at the largest library size should be sizeable
        # — Sugihara's hallmark for the true direction.
        largest = max(rows_true, key=lambda r: r["library_size"])
        self.assertGreater(
            largest["rho_xy"], 0.3,
            f"rho_xy at largest library should be > 0.3, got {largest['rho_xy']:.3f}")


if __name__ == "__main__":
    unittest.main()
