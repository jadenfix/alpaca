"""Known-truth tests for the multivariate exponential-kernel Hawkes
process. Mirrors the conventions used in ``test_analysis.py``.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

# Make the analysis package importable (same pattern as test_analysis.py).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE.parent))


def _seed_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ─────────────────────── thinning simulators ───────────────────────

def _sim_univariate_hawkes(mu: float, alpha: float, beta: float,
                           t_max: float, rng: np.random.Generator) -> np.ndarray:
    """Ogata thinning of a single-variate Hawkes process."""
    events: list[float] = []
    t = 0.0
    while t < t_max:
        # Upper bound on intensity is current intensity (decays only).
        lam_bar = mu + sum(alpha * math.exp(-beta * (t - ti)) for ti in events)
        u = rng.uniform(1e-9, 1.0)
        t += -math.log(u) / max(lam_bar, 1e-9)
        if t >= t_max:
            break
        lam_new = mu + sum(alpha * math.exp(-beta * (t - ti)) for ti in events)
        if rng.uniform() <= lam_new / lam_bar:
            events.append(t)
    return np.array(events, dtype=np.float64)


def _sim_mv_hawkes(mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
                   t_max: float, rng: np.random.Generator) -> list[np.ndarray]:
    """Ogata thinning of a D-dimensional exponential Hawkes."""
    D = mu.size
    ev: list[list[float]] = [[] for _ in range(D)]
    t = 0.0
    while t < t_max:
        # Compute total intensity bound (sum of per-dim intensities now).
        lam = np.array(mu, dtype=np.float64, copy=True)
        for d in range(D):
            for dp in range(D):
                if not ev[dp]:
                    continue
                # contribution from all events in dp
                lam[d] += alpha[d, dp] * sum(
                    math.exp(-beta[d] * (t - ti)) for ti in ev[dp]
                )
        lam_bar = float(np.sum(lam))
        u = rng.uniform(1e-9, 1.0)
        t += -math.log(u) / max(lam_bar, 1e-9)
        if t >= t_max:
            break
        # Recompute intensities at the new candidate time.
        lam_new = np.array(mu, dtype=np.float64, copy=True)
        for d in range(D):
            for dp in range(D):
                if not ev[dp]:
                    continue
                lam_new[d] += alpha[d, dp] * sum(
                    math.exp(-beta[d] * (t - ti)) for ti in ev[dp]
                )
        lam_total = float(np.sum(lam_new))
        if rng.uniform() <= lam_total / lam_bar:
            # Choose which dimension fired.
            p = lam_new / lam_total
            d_fire = int(np.searchsorted(np.cumsum(p), rng.uniform()))
            d_fire = min(d_fire, D - 1)
            ev[d_fire].append(t)
    return [np.array(e, dtype=np.float64) for e in ev]


# ───────────────────────── test class ────────────────────────────

class MultivariateHawkesTests(unittest.TestCase):

    def test_branching_matrix_hand_value(self):
        import multivariate_hawkes as mvh
        alpha = np.array([[0.6, 0.2], [0.1, 0.4]])
        beta = np.array([2.0, 4.0])
        K = mvh.branching_matrix(alpha, beta)
        # Row d divided by beta_d.
        self.assertAlmostEqual(K[0, 0], 0.3, places=12)
        self.assertAlmostEqual(K[0, 1], 0.1, places=12)
        self.assertAlmostEqual(K[1, 0], 0.025, places=12)
        self.assertAlmostEqual(K[1, 1], 0.1, places=12)

    def test_spectral_radius_matches_numpy(self):
        import multivariate_hawkes as mvh
        rng = _seed_rng(7)
        K = rng.normal(size=(3, 3))
        sr = mvh.spectral_radius(K)
        eigs = np.linalg.eigvals(K)
        self.assertAlmostEqual(sr, float(np.max(np.abs(eigs))), places=9)

    def test_contagion_paths_orders_by_magnitude(self):
        import multivariate_hawkes as mvh
        K = np.array([[0.1, 0.05, 0.7],
                      [0.4, 0.2, 0.01],
                      [0.02, 0.9, 0.3]])
        names = ["A", "B", "C"]
        out = mvh.contagion_paths(K, names, top_k=3)
        self.assertEqual(len(out), 3)
        # The strongest entry is K[2, 1] = 0.9.
        self.assertEqual(out[0]["trigger_dim"], 1)
        self.assertEqual(out[0]["response_dim"], 2)
        self.assertAlmostEqual(out[0]["k_value"], 0.9, places=12)
        self.assertEqual(out[0]["trigger_name"], "B")
        self.assertEqual(out[0]["response_name"], "C")
        # Second strongest is K[0, 2] = 0.7.
        self.assertEqual(out[1]["trigger_dim"], 2)
        self.assertEqual(out[1]["response_dim"], 0)
        self.assertAlmostEqual(out[1]["k_value"], 0.7, places=12)
        # Monotone non-increasing magnitudes.
        mags = [abs(r["k_value"]) for r in out]
        self.assertTrue(all(mags[i] >= mags[i + 1] for i in range(len(mags) - 1)))

    def test_mv_loglik_reduces_to_single_variate(self):
        import multivariate_hawkes as mvh
        import hawkes as hw
        rng = _seed_rng(13)
        # Simulate a small stationary 1D Hawkes.
        t = _sim_univariate_hawkes(0.5, 0.6, 1.5, t_max=120.0, rng=rng)
        self.assertGreater(t.size, 5)
        # Single-variate compensator uses t_end = t[-1].
        ll_uni = hw.hawkes_loglik(t, 1.0, 0.4, 1.7)
        ll_mv = mvh.mv_hawkes_loglik([t],
                                     mu=np.array([1.0]),
                                     alpha=np.array([[0.4]]),
                                     beta=np.array([1.7]),
                                     t_end=float(t[-1]))
        self.assertAlmostEqual(ll_uni, ll_mv, places=6)

    def test_independent_streams_recover_small_off_diagonal(self):
        import multivariate_hawkes as mvh
        rng = _seed_rng(101)
        # Two independent Hawkes streams.
        t0 = _sim_univariate_hawkes(0.5, 0.5, 1.5, t_max=2000.0, rng=rng)
        t1 = _sim_univariate_hawkes(0.7, 0.4, 1.8, t_max=2000.0, rng=rng)
        self.assertGreater(t0.size, 50)
        self.assertGreater(t1.size, 50)
        fit = mvh.fit_mv_hawkes([t0, t1], t_end=2000.0, max_iter=120)
        self.assertIsNotNone(fit)
        a = fit["alpha"]
        # Off-diagonals should be considerably smaller than the
        # corresponding diagonals.
        self.assertLess(abs(a[0, 1]), 0.5 * a[0, 0] + 1e-9)
        self.assertLess(abs(a[1, 0]), 0.5 * a[1, 1] + 1e-9)
        self.assertLess(fit["spectral_radius"], 1.0)

    def test_cross_excitation_recovered(self):
        import multivariate_hawkes as mvh
        rng = _seed_rng(202)
        # Strong α_{0,1}: events in stream 1 trigger many events in
        # stream 0. Diagonal self-excitation kept modest.
        mu_true = np.array([0.2, 0.4])
        alpha_true = np.array([[0.2, 0.7],
                               [0.05, 0.2]])
        beta_true = np.array([1.5, 1.5])
        t_max = 1500.0
        ev = _sim_mv_hawkes(mu_true, alpha_true, beta_true, t_max, rng)
        self.assertGreater(ev[0].size, 50)
        self.assertGreater(ev[1].size, 50)
        fit = mvh.fit_mv_hawkes(ev, t_end=t_max, max_iter=150)
        self.assertIsNotNone(fit)
        K = fit["branching_matrix"]
        # K_{0,1} (stream 1 → stream 0) should dominate K_{1,0}.
        self.assertGreater(K[0, 1], K[1, 0])
        self.assertLess(fit["spectral_radius"], 1.0)


if __name__ == "__main__":
    unittest.main()
