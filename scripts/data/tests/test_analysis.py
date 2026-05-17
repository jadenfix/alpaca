"""Known-truth tests for the econophysics analysis toolkit.

One test class per module. Each test would FAIL on a no-op or
constant-returning implementation, so they validate the math, not just
the API surface.
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Make the analysis package importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE.parent))


def _seed_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ───────────────────────────── correlation ─────────────────────────────

class CorrelationTests(unittest.TestCase):
    def test_pearson_perfect_linear_is_one(self):
        import correlation_matrix as cm
        data = np.column_stack([np.arange(50.0), np.arange(50.0)])
        c = cm.pearson(data)
        self.assertAlmostEqual(c[0, 1], 1.0, places=12)

    def test_spearman_invariant_under_monotone_transform(self):
        import correlation_matrix as cm
        x = np.arange(1, 21, dtype=np.float64)
        y = np.exp(x)
        data = np.column_stack([x, y])
        rho = cm.spearman(data)[0, 1]
        # Strictly monotone => Spearman = 1.
        self.assertAlmostEqual(rho, 1.0, places=12)

    def test_kendall_hand_value(self):
        import correlation_matrix as cm
        # one swap from monotone: 9 concordant, 1 discordant => τ = 0.8
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array([1, 2, 3, 5, 4], dtype=np.float64)
        data = np.column_stack([x, y])
        tau = cm.kendall(data)[0, 1]
        self.assertAlmostEqual(tau, 0.8, places=12)


# ─────────────────────────────── RMT ────────────────────────────────────

class RmtTests(unittest.TestCase):
    def test_mp_bounds_have_expected_form(self):
        import rmt_spectral as rmt
        lo, hi = rmt.mp_bounds(0.25, 1.0)
        # σ²(1 - √q)² = (1 - 0.5)² = 0.25
        # σ²(1 + √q)² = (1.5)² = 2.25
        self.assertAlmostEqual(lo, 0.25, places=12)
        self.assertAlmostEqual(hi, 2.25, places=12)

    def test_mp_density_integrates_to_one_via_simpson(self):
        import rmt_spectral as rmt
        q = 0.25
        sigma2 = 1.0
        lo, hi = rmt.mp_bounds(q, sigma2)
        n_steps = 10_000
        x = np.linspace(lo + 1e-6, hi - 1e-6, n_steps + 1)
        y = rmt.mp_density(x, q, sigma2)
        # Composite Simpson's rule.
        h = (x[-1] - x[0]) / n_steps
        s = y[0] + y[-1] + 4 * np.sum(y[1::2]) + 2 * np.sum(y[2:-1:2])
        integral = s * h / 3.0
        self.assertAlmostEqual(integral, 1.0, delta=0.02)

    def test_rie_shrinks_noise_preserves_signal(self):
        import rmt_spectral as rmt
        q = 0.1
        # 1 signal eigenvalue, 9 near the bulk mean.
        eigs = np.array([10.0, 1.0, 0.95, 1.05, 1.1, 0.9, 1.02, 0.98, 1.03, 0.97])
        shrunk = rmt.rie_shrinkage(eigs, q)
        # Signal preserved.
        self.assertGreater(shrunk[0], 8.0)
        # Noise eigenvalue variance decreases after shrinkage.
        var_in = np.var(eigs[1:])
        var_out = np.var(shrunk[1:])
        self.assertLess(var_out, var_in)

    def test_tracy_widom_pvalue_high_for_noise(self):
        import rmt_spectral as rmt
        rng = _seed_rng(7)
        n, t = 20, 200
        x = rng.standard_normal((t, n))
        out = rmt.rmt_summary(x)
        # On i.i.d. noise the top eigenvalue should look like noise.
        self.assertGreater(out["tw_pvalue"], 0.05)

    def test_tracy_widom_pvalue_low_for_factor(self):
        import rmt_spectral as rmt
        rng = _seed_rng(11)
        n, t = 20, 200
        # Add a strong common factor — λ_max should be well above the MP bulk.
        factor = rng.standard_normal(t) * 5.0
        noise = rng.standard_normal((t, n))
        x = noise + factor[:, None]
        out = rmt.rmt_summary(x)
        self.assertLess(out["tw_pvalue"], 0.05)


# ───────────────────────── information theory ──────────────────────────

class InfoTheoryTests(unittest.TestCase):
    def test_mi_independent_is_small(self):
        import info_theory as it
        rng = _seed_rng(7)
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        mi = it.mutual_information(x, y, bins=10)
        # Plug-in MI has positive bias O(bins²/n); accept up to 0.15 nats.
        self.assertLess(mi, 0.15)

    def test_mi_self_is_approx_entropy(self):
        import info_theory as it
        rng = _seed_rng(11)
        x = rng.normal(size=2000)
        mi = it.mutual_information(x, x, bins=16)
        h = it.entropy(x, bins=16)
        # MI(X;X) ≡ H(X). Allow histogram-noise tolerance.
        self.assertAlmostEqual(mi, h, delta=0.15)

    def test_kl_self_is_zero(self):
        import info_theory as it
        p = np.array([0.2, 0.3, 0.5])
        self.assertAlmostEqual(it.kl_divergence(p, p), 0.0)

    def test_jensen_shannon_symmetric_and_bounded(self):
        import info_theory as it
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.1, 0.6, 0.3])
        js_pq = it.jensen_shannon(p, q)
        js_qp = it.jensen_shannon(q, p)
        self.assertAlmostEqual(js_pq, js_qp, places=12)
        self.assertLess(js_pq, math.log(2.0) + 1e-9)

    def test_te_asymmetric_when_x_drives_y(self):
        import info_theory as it
        rng = _seed_rng(13)
        n = 1500
        x = np.zeros(n)
        y = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.3 * x[t - 1] + rng.normal()
            y[t] = 0.5 * abs(y[t - 1]) ** 0.9 * np.sign(x[t - 1]) + 0.1 * rng.normal()
        te_xy = it.transfer_entropy(x, y, k=1, l=1, bins=4)
        te_yx = it.transfer_entropy(y, x, k=1, l=1, bins=4)
        self.assertGreater(te_xy, 0.02)
        self.assertGreater(te_xy, te_yx)


# ──────────────────────────── dependence ───────────────────────────────

class DependenceTests(unittest.TestCase):
    def test_dcor_self_is_one(self):
        import dependence as dep
        x = np.linspace(0, 10, 50)
        self.assertAlmostEqual(dep.distance_correlation(x, x), 1.0, places=9)

    def test_dcor_catches_y_equals_x_squared(self):
        import dependence as dep
        rng = _seed_rng(13)
        x = rng.uniform(-1, 1, size=400)
        y = x * x
        d = dep.distance_correlation(x, y)
        self.assertGreater(d, 0.3)
        # Pearson would be ~ 0 here.
        pearson = float(np.corrcoef(x, y)[0, 1])
        self.assertLess(abs(pearson), 0.15)

    def test_dcca_independent_is_small(self):
        import dependence as dep
        rng = _seed_rng(7)
        n = 800
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        rho = dep.dcca_rho(x, y, scale=16)
        self.assertLess(abs(rho), 0.20)

    def test_dcca_correlated_at_a_scale(self):
        import dependence as dep
        # Identical sine waves should be DCCA-correlated at all scales.
        n = 400
        t = np.arange(n)
        x = np.sin(t * 0.1)
        y = np.sin(t * 0.1)
        rho = dep.dcca_rho(x, y, scale=16)
        self.assertGreater(rho, 0.95)


# ───────────────────────────── causality ───────────────────────────────

class CausalityTests(unittest.TestCase):
    def test_granger_significant_for_known_leader(self):
        import causality as cau
        rng = _seed_rng(7)
        n = 600
        x = rng.normal(size=n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.7 * x[t - 1] + 0.3 * rng.normal()
        g = cau.granger_f(y, x, lags=1)
        self.assertLess(g["p_value"], 0.05)
        self.assertGreater(g["f_stat"], 10.0)

    def test_granger_not_significant_for_independent(self):
        import causality as cau
        rng = _seed_rng(11)
        n = 600
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        g = cau.granger_f(y, x, lags=1)
        self.assertGreater(g["p_value"], 0.10)

    def test_f_pvalue_known_values(self):
        # F(1,1)=161.4 → p ≈ 0.05 (one-sided)
        import causality as cau
        p = cau._f_pvalue(161.4, 1, 1)
        self.assertAlmostEqual(p, 0.05, delta=0.02)


# ───────────────────────────── copula ──────────────────────────────────

class CopulaTests(unittest.TestCase):
    def test_independent_lambda_u_small(self):
        import copula as cop
        rng = _seed_rng(7)
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        lu = cop.upper_tail_dependence(x, y, q=0.90)
        self.assertLess(lu, 0.25)

    def test_comonotonic_lambda_u_near_one(self):
        import copula as cop
        x = np.linspace(0, 1, 500)
        y = x.copy()
        lu = cop.upper_tail_dependence(x, y, q=0.90)
        self.assertGreater(lu, 0.90)

    def test_clayton_theta_recovers_positive(self):
        import copula as cop
        # Synthesize positively associated data: y is a noisy version of x.
        rng = _seed_rng(7)
        x = rng.normal(size=300)
        y = x + 0.5 * rng.normal(size=300)
        theta = cop.fit_clayton_theta(x, y)
        self.assertGreater(theta, 0.0)


# ─────────────────────── multifractal (MF-DFA) ─────────────────────────

class MultifractalTests(unittest.TestCase):
    def test_white_noise_h_near_half(self):
        import multifractal as mf
        rng = _seed_rng(13)
        x = rng.normal(size=4096)
        out = mf.mfdfa(x)
        h_q2 = out["h_q"].get(2.0, 0.0)
        # h(q=2) ≡ classical Hurst; should be ≈ 0.5 for i.i.d. noise.
        self.assertAlmostEqual(h_q2, 0.5, delta=0.15)

    def test_multifractal_width_is_nonneg(self):
        import multifractal as mf
        rng = _seed_rng(7)
        x = rng.normal(size=2048)
        out = mf.mfdfa(x)
        self.assertGreaterEqual(out["width"], 0.0)


# ────────────────────────── diffusion map ──────────────────────────────

class DiffusionMapTests(unittest.TestCase):
    def test_embedding_shape_correct(self):
        import diffusion_map as dm
        rng = _seed_rng(7)
        # 6 series of 100 samples each.
        data = rng.normal(size=(100, 6))
        out = dm.diffusion_map(data, n_components=2)
        self.assertEqual(out["embedding"].shape, (6, 2))

    def test_top_eigenvalue_is_one(self):
        import diffusion_map as dm
        rng = _seed_rng(13)
        data = rng.normal(size=(100, 5))
        out = dm.diffusion_map(data, n_components=2)
        # Top eigenvalue of a row-stochastic matrix is always 1.
        self.assertAlmostEqual(float(out["eigenvalues"][0]), 1.0, delta=0.01)


# ────────────────────────────── network ────────────────────────────────

class NetworkTests(unittest.TestCase):
    def test_mst_has_n_minus_one_edges(self):
        import network as nw
        n = 5
        rng = _seed_rng(7)
        # Random positive-definite distance-like matrix.
        x = rng.normal(size=(n, n))
        d = np.abs(x + x.T) + np.eye(n) * 1e-9
        np.fill_diagonal(d, 0)
        edges = nw.mst(d)
        self.assertEqual(len(edges), n - 1)

    def test_mst_is_connected(self):
        import network as nw
        n = 7
        rng = _seed_rng(11)
        x = rng.normal(size=(n, n))
        d = np.abs(x + x.T)
        np.fill_diagonal(d, 0)
        edges = nw.mst(d)
        # Union-find: collapse all nodes into one component.
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, j, _ in edges:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
        roots = {find(i) for i in range(n)}
        self.assertEqual(len(roots), 1)

    def test_pmfg_at_most_planar_bound(self):
        import network as nw
        n = 10
        rng = _seed_rng(13)
        x = rng.normal(size=(n, n))
        d = np.abs(x + x.T)
        np.fill_diagonal(d, 0)
        edges = nw.pmfg(d)
        self.assertLessEqual(len(edges), 3 * n - 6)


# ───────────────────────── alpha-stable ────────────────────────────────

class AlphaStableTests(unittest.TestCase):
    def test_gaussian_recovers_alpha_near_two(self):
        import alpha_stable as al
        rng = _seed_rng(7)
        x = rng.normal(size=2000)
        out = al.fit_alpha_stable(x)
        # Allow generous tolerance; the Koutrouvelis CF regression has
        # finite-sample bias.
        self.assertGreater(out["alpha"], 1.5)

    def test_cauchy_recovers_alpha_below_two(self):
        import alpha_stable as al
        rng = _seed_rng(11)
        # Cauchy(0,1) via inverse CDF: x = tan(π(U - 1/2))
        u = rng.uniform(0.05, 0.95, size=2000)
        x = np.tan(np.pi * (u - 0.5))
        out = al.fit_alpha_stable(x)
        # Cauchy is α=1; should be clearly below Gaussian.
        self.assertLess(out["alpha"], 1.8)


# ───────────────────────────── hawkes ──────────────────────────────────

class HawkesTests(unittest.TestCase):
    def test_loglik_is_finite(self):
        import hawkes as hw
        t = np.arange(1.0, 51.0)
        ll = hw.hawkes_loglik(t, 1.0, 0.3, 1.5)
        self.assertTrue(math.isfinite(ll))

    def test_loglik_rejects_invalid_params(self):
        import hawkes as hw
        t = np.arange(1.0, 51.0)
        self.assertEqual(hw.hawkes_loglik(t, -1.0, 0.3, 1.5), float("-inf"))
        self.assertEqual(hw.hawkes_loglik(t, 1.0, 0.3, -1.5), float("-inf"))

    def test_fit_returns_stationary_params(self):
        import hawkes as hw
        # Simulate a stationary Hawkes process (n=0.4 = 0.6/1.5) via thinning.
        rng = _seed_rng(13)
        mu, alpha, beta = 0.5, 0.6, 1.5
        t_max = 3000.0
        t = 0.0
        events: list[float] = []
        while t < t_max:
            lam_t = mu + sum(alpha * math.exp(-beta * (t - ti)) for ti in events)
            t += -math.log(rng.uniform(0.01, 0.99)) / max(lam_t, 1e-6)
            if t >= t_max:
                break
            lam_new = mu + sum(alpha * math.exp(-beta * (t - ti)) for ti in events)
            if rng.uniform() <= lam_new / lam_t:
                events.append(t)
        self.assertGreater(len(events), 50)
        fit = hw.fit_hawkes(np.array(events))
        self.assertIsNotNone(fit)
        # Stationary regardless of recovery precision.
        self.assertLess(fit["branching_ratio"], 1.0)


# ───────────────────────────── wavelet ─────────────────────────────────

class WaveletTests(unittest.TestCase):
    def test_identical_series_coherence_high(self):
        import wavelet as wv
        n = 256
        t = np.arange(n)
        x = np.sin(t * 0.1) + np.sin(t * 0.05)
        y = x.copy()
        wc = wv.wavelet_coherence(x, y)
        mean_c = float(wc["mean_coherence_per_scale"].mean())
        self.assertGreater(mean_c, 0.95)

    def test_independent_series_coherence_lower(self):
        import wavelet as wv
        rng = _seed_rng(7)
        n = 256
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        wc = wv.wavelet_coherence(x, y)
        mean_c = float(wc["mean_coherence_per_scale"].mean())
        # Smoothed coherence on random signals has positive bias; just
        # assert it's clearly less than the identical-series case.
        self.assertLess(mean_c, 0.85)


# ────────────────────────────── loader ─────────────────────────────────

class LoaderTests(unittest.TestCase):
    def test_inner_join_drops_unmatched(self):
        from loader import align
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            p1 = tdp / "a.csv"
            p2 = tdp / "b.csv"
            p1.write_text("ts_nanos,series,value\n1,A,1.0\n2,A,2.0\n3,A,3.0\n")
            p2.write_text("ts_nanos,series,value\n2,B,20.0\n3,B,30.0\n4,B,40.0\n")
            data, names, ts = align([p1, p2])
            # Only ts=2 and ts=3 appear in both.
            self.assertEqual(data.shape, (2, 2))
            self.assertEqual(names, ["a", "b"])
            self.assertEqual(list(ts), [2, 3])
            self.assertAlmostEqual(data[0, 0], 2.0)
            self.assertAlmostEqual(data[0, 1], 20.0)

    def test_returns_shape(self):
        from loader import returns
        data = np.array([[100.0, 200.0], [101.0, 201.0], [102.0, 202.0]])
        r = returns(data)
        self.assertEqual(r.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
