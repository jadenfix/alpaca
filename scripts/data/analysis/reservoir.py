"""Echo State Network (Jaeger 2001) — nonlinear forecasting without backprop.

A fixed random recurrent reservoir + a single linear readout trained by
ridge regression. Ideal for chaotic time-series forecasting where
transformer-style models are overkill.

State update (leaky integrator):
    x_t = (1 - alpha) * x_{t-1}
        + alpha * tanh(W_in u_t + W_res x_{t-1} + bias)

Readout:
    y_t = W_out @ [x_t; u_t; 1]

W_in, W_res, bias are fixed random matrices drawn once. W_res is sparse
(default 10% nonzeros) and rescaled so its spectral radius equals rho,
which controls the reservoir's memory (must be < 1 for the echo-state
property under tanh activation).

Pure-numpy implementation: no scipy / sklearn / torch / jax.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ───────────────────────────── config ─────────────────────────────


@dataclass
class ESNConfig:
    """Hyperparameters for an echo state network."""

    n_res: int = 200
    spectral_radius: float = 0.9
    leak: float = 0.3          # alpha; 0 = pure recurrence, 1 = no leak
    input_scaling: float = 1.0
    bias_scaling: float = 0.1
    sparsity: float = 0.1      # fraction of nonzero entries in W_res
    ridge: float = 1e-4
    burn_in: int = 50
    seed: int = 0


# ───────────────────────────── helpers ─────────────────────────────


def _as_2d(a: np.ndarray) -> np.ndarray:
    """Coerce 1-D arrays to column form (T, 1); leave 2-D arrays alone."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D array, got ndim={a.ndim}")
    return a


# ───────────────────────────── reservoir init ─────────────────────────────


def init_reservoir(n_in: int, cfg: ESNConfig) -> dict:
    """Build the fixed random reservoir matrices.

    Returns a dict with keys ``W_in`` (n_res, n_in), ``W_res`` (n_res, n_res),
    and ``bias`` (n_res,). ``W_res`` is sparse with the requested density
    and is rescaled so max |eigenvalue| equals cfg.spectral_radius.
    """
    if n_in <= 0:
        raise ValueError("n_in must be positive")
    if cfg.n_res <= 0:
        raise ValueError("n_res must be positive")
    if not (0.0 < cfg.sparsity <= 1.0):
        raise ValueError("sparsity must lie in (0, 1]")

    rng = np.random.default_rng(cfg.seed)
    n_res = int(cfg.n_res)

    # Input weights: dense uniform [-1, 1] * input_scaling.
    W_in = rng.uniform(-1.0, 1.0, size=(n_res, n_in)) * cfg.input_scaling

    # Bias: uniform [-1, 1] * bias_scaling.
    bias = rng.uniform(-1.0, 1.0, size=n_res) * cfg.bias_scaling

    # Recurrent weights: sparse mask * uniform [-1, 1].
    mask = rng.random((n_res, n_res)) < cfg.sparsity
    W_res = np.where(mask, rng.uniform(-1.0, 1.0, size=(n_res, n_res)), 0.0)

    # Rescale to target spectral radius.
    eigvals = np.linalg.eigvals(W_res)
    rho = float(np.max(np.abs(eigvals)))
    if rho < 1e-12:
        # Degenerate draw (all-zero matrix); reseed slightly off and retry.
        # Vanishingly unlikely for default sparsity, but guard anyway.
        raise RuntimeError("reservoir spectral radius collapsed to 0; "
                           "increase sparsity or change seed")
    W_res = W_res * (cfg.spectral_radius / rho)

    return {"W_in": W_in, "W_res": W_res, "bias": bias}


# ───────────────────────────── reservoir dynamics ─────────────────────────────


def run_reservoir(
    U: np.ndarray,
    res: dict,
    cfg: ESNConfig,
    initial_state: np.ndarray | None = None,
) -> np.ndarray:
    """Drive the reservoir with inputs U (T, n_in); return states X (T, n_res).

    If ``initial_state`` is None the reservoir starts at zero.
    """
    U2 = _as_2d(U)
    T = U2.shape[0]
    W_in = res["W_in"]
    W_res = res["W_res"]
    bias = res["bias"]
    n_res = W_res.shape[0]

    if W_in.shape[1] != U2.shape[1]:
        raise ValueError(
            f"input dim mismatch: W_in has {W_in.shape[1]} cols, U has {U2.shape[1]}"
        )

    if initial_state is None:
        x = np.zeros(n_res, dtype=np.float64)
    else:
        x = np.asarray(initial_state, dtype=np.float64).reshape(-1)
        if x.size != n_res:
            raise ValueError(f"initial_state length {x.size} != n_res {n_res}")

    alpha = float(cfg.leak)
    one_minus_alpha = 1.0 - alpha

    X = np.empty((T, n_res), dtype=np.float64)
    for t in range(T):
        pre = W_in @ U2[t] + W_res @ x + bias
        x = one_minus_alpha * x + alpha * np.tanh(pre)
        X[t] = x
    return X


# ───────────────────────────── readout fit / apply ─────────────────────────────


def _build_regressor(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Stack [state | input | bias-1] columns → (T, n_res + n_in + 1)."""
    U2 = _as_2d(U)
    if X.shape[0] != U2.shape[0]:
        raise ValueError(
            f"row mismatch: states {X.shape[0]} vs inputs {U2.shape[0]}"
        )
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    return np.concatenate([X, U2, ones], axis=1)


def fit_readout(
    X: np.ndarray,
    U: np.ndarray,
    Y: np.ndarray,
    cfg: ESNConfig,
) -> np.ndarray:
    """Ridge-regress the readout. Returns W_out of shape (n_out, n_res+n_in+1).

    The first ``cfg.burn_in`` rows of X, U, Y are discarded (cold-start
    transient before the reservoir is on the attractor).
    """
    Y2 = _as_2d(Y)
    burn = int(cfg.burn_in)
    if burn < 0:
        raise ValueError("burn_in must be non-negative")
    if burn >= X.shape[0]:
        raise ValueError(
            f"burn_in {burn} >= number of samples {X.shape[0]}"
        )

    Xb = X[burn:]
    Ub = _as_2d(U)[burn:]
    Yb = Y2[burn:]

    Phi = _build_regressor(Xb, Ub)  # (T', n_feat)
    n_feat = Phi.shape[1]

    # Solve (Phi^T Phi + ridge * I) W = Phi^T Y for W (n_feat, n_out),
    # then transpose so W_out has shape (n_out, n_feat).
    A = Phi.T @ Phi + float(cfg.ridge) * np.eye(n_feat)
    B = Phi.T @ Yb
    W = np.linalg.solve(A, B)        # (n_feat, n_out)
    return W.T                       # (n_out, n_feat)


def predict(X: np.ndarray, U: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """Apply the readout to states + inputs. Returns Y_pred (T, n_out)."""
    Phi = _build_regressor(X, U)     # (T, n_feat)
    Y_pred = Phi @ W_out.T           # (T, n_out)
    return Y_pred


# ───────────────────────────── full pipeline ─────────────────────────────


def train_predict_esn(
    U_train: np.ndarray,
    Y_train: np.ndarray,
    U_test: np.ndarray,
    cfg: ESNConfig | None = None,
) -> dict:
    """End-to-end: init reservoir, fit readout, predict on train & test.

    Returns a dict with keys ``W_out``, ``res``, ``Y_train_pred``,
    ``Y_test_pred``, ``train_rmse`` (computed on post-burn-in samples),
    and ``spectral_radius_actual``.
    """
    if cfg is None:
        cfg = ESNConfig()

    U_tr = _as_2d(U_train)
    Y_tr = _as_2d(Y_train)
    U_te = _as_2d(U_test)

    if U_tr.shape[0] != Y_tr.shape[0]:
        raise ValueError(
            f"U_train rows {U_tr.shape[0]} != Y_train rows {Y_tr.shape[0]}"
        )
    if U_te.shape[1] != U_tr.shape[1]:
        raise ValueError(
            f"U_test n_in {U_te.shape[1]} != U_train n_in {U_tr.shape[1]}"
        )

    n_in = U_tr.shape[1]
    res = init_reservoir(n_in, cfg)

    # Train pass: drive reservoir from zero and fit readout.
    X_tr = run_reservoir(U_tr, res, cfg)
    W_out = fit_readout(X_tr, U_tr, Y_tr, cfg)

    # Predict on train (post-burn-in only contributes to train RMSE).
    Y_train_pred = predict(X_tr, U_tr, W_out)

    # Continue dynamics into the test window from the final training state.
    X_te = run_reservoir(U_te, res, cfg, initial_state=X_tr[-1])
    Y_test_pred = predict(X_te, U_te, W_out)

    burn = int(cfg.burn_in)
    resid = Y_train_pred[burn:] - Y_tr[burn:]
    train_rmse = float(np.sqrt(np.mean(resid * resid)))

    spectral_radius_actual = float(
        np.max(np.abs(np.linalg.eigvals(res["W_res"])))
    )

    return {
        "W_out": W_out,
        "res": res,
        "Y_train_pred": Y_train_pred,
        "Y_test_pred": Y_test_pred,
        "train_rmse": train_rmse,
        "spectral_radius_actual": spectral_radius_actual,
    }
