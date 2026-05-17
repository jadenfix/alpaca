"""Known-truth tests for ``reservoir`` (Echo State Network).

Each test exercises a specific guarantee of the ESN implementation:
spectral-radius rescaling, sine-wave fit, Lorenz forecast, burn-in
shape correctness, and seed-determinism. All would FAIL on a no-op or
constant-returning implementation.
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


def _lorenz_xyz(n: int, dt: float = 0.01,
                sigma: float = 10.0, rho: float = 28.0,
                beta: float = 8.0 / 3.0,
                x0: float = 1.0, y0: float = 1.0, z0: float = 1.0,
                ) -> np.ndarray:
    """Explicit-Euler integration of the Lorenz system; no scipy."""
    out = np.empty((n, 3), dtype=np.float64)
    x, y, z = float(x0), float(y0), float(z0)
    for i in range(n):
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = z
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x = x + dt * dx
        y = y + dt * dy
        z = z + dt * dz
    return out


class ReservoirTests(unittest.TestCase):
    # ───────────────────── 1. spectral-radius rescaling ─────────────────────

    def test_spectral_radius_rescaled(self):
        import reservoir as esn
        cfg = esn.ESNConfig(n_res=100, spectral_radius=0.85,
                            sparsity=0.1, seed=7)
        res = esn.init_reservoir(n_in=1, cfg=cfg)
        rho = float(np.max(np.abs(np.linalg.eigvals(res["W_res"]))))
        self.assertAlmostEqual(rho, cfg.spectral_radius, delta=1e-6)

    # ───────────────────── 2. sine-wave one-step forecast ───────────────────

    def test_sine_wave_one_step_forecast(self):
        import reservoir as esn
        t = np.arange(1000, dtype=np.float64)
        u = np.sin(0.1 * t)
        y = np.sin(0.1 * (t + 1.0))
        U_train, Y_train = u[:700], y[:700]
        U_test, Y_test = u[700:], y[700:]
        cfg = esn.ESNConfig(n_res=200, spectral_radius=0.9, leak=0.3,
                            ridge=1e-6, burn_in=100, seed=0)
        out = esn.train_predict_esn(U_train, Y_train, U_test, cfg)
        Y_pred = out["Y_test_pred"].reshape(-1)
        rmse = float(np.sqrt(np.mean((Y_pred - Y_test) ** 2)))
        self.assertLess(rmse, 0.1,
                        f"sine-wave RMSE too high: {rmse:.4f}")

    # ───────────────────── 3. Lorenz x-coordinate forecast ──────────────────

    def test_lorenz_x_forecast_beats_baseline(self):
        import reservoir as esn
        xyz = _lorenz_xyz(1001)
        x = xyz[:, 0]
        # Predict one step ahead from current x.
        U = x[:-1]
        Y = x[1:]                  # length 1000
        U_train, Y_train = U[:800], Y[:800]
        U_test,  Y_test  = U[800:], Y[800:]
        cfg = esn.ESNConfig(n_res=300, spectral_radius=0.95, leak=0.3,
                            input_scaling=0.5, ridge=1e-6,
                            burn_in=100, seed=1)
        out = esn.train_predict_esn(U_train, Y_train, U_test, cfg)
        Y_pred = out["Y_test_pred"].reshape(-1)
        rmse = float(np.sqrt(np.mean((Y_pred - Y_test) ** 2)))
        baseline = float(np.std(Y_test))
        self.assertLess(rmse, baseline,
                        f"Lorenz RMSE {rmse:.3f} >= std(Y_test) {baseline:.3f}")

    # ───────────────────── 4. burn-in dropped correctly ─────────────────────

    def test_burn_in_dropped_in_regressor(self):
        import reservoir as esn
        T, n_in = 200, 1
        cfg = esn.ESNConfig(n_res=50, spectral_radius=0.9, sparsity=0.2,
                            burn_in=50, seed=3)
        rng = np.random.default_rng(0)
        U = rng.standard_normal((T, n_in))
        Y = rng.standard_normal((T, 1))
        res = esn.init_reservoir(n_in, cfg)
        X = esn.run_reservoir(U, res, cfg)
        self.assertEqual(X.shape, (T, cfg.n_res))

        W_out = esn.fit_readout(X, U, Y, cfg)
        # n_feat = n_res + n_in + 1 (bias column).
        expected_feat = cfg.n_res + n_in + 1
        self.assertEqual(W_out.shape, (1, expected_feat))

        # And the implied number of fitted samples = T - burn_in.
        # We can verify by re-doing the math here.
        Phi = np.concatenate(
            [X[cfg.burn_in:], U[cfg.burn_in:],
             np.ones((T - cfg.burn_in, 1))],
            axis=1,
        )
        self.assertEqual(Phi.shape, (T - cfg.burn_in, expected_feat))
        A = Phi.T @ Phi + cfg.ridge * np.eye(expected_feat)
        B = Phi.T @ Y[cfg.burn_in:]
        W_ref = np.linalg.solve(A, B).T
        np.testing.assert_allclose(W_out, W_ref, atol=1e-10)

    # ───────────────────── 5. deterministic given seed ──────────────────────

    def test_deterministic_given_seed(self):
        import reservoir as esn
        cfg = esn.ESNConfig(n_res=80, spectral_radius=0.9, seed=42)
        t = np.arange(400, dtype=np.float64)
        U = np.sin(0.07 * t).reshape(-1, 1)
        Y = np.sin(0.07 * (t + 1.0)).reshape(-1, 1)
        out1 = esn.train_predict_esn(U, Y, U[:50], cfg)
        out2 = esn.train_predict_esn(U, Y, U[:50], cfg)
        np.testing.assert_array_equal(out1["W_out"], out2["W_out"])
        np.testing.assert_array_equal(out1["res"]["W_res"],
                                      out2["res"]["W_res"])
        np.testing.assert_array_equal(out1["res"]["W_in"],
                                      out2["res"]["W_in"])
        np.testing.assert_array_equal(out1["res"]["bias"],
                                      out2["res"]["bias"])


if __name__ == "__main__":
    unittest.main()
