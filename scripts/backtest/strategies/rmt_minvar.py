"""Minimum-variance portfolio computed on the RIE-shrunk covariance.

Combines the two strongest baselines from the discovery report:
  • min_variance is the highest-Sharpe strategy on the test universe
  • RIE shrinkage is the optimal admissible high-dim covariance cleaner

Drop-in walk-forward strategy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "data" / "analysis"))

import rmt_spectral as rmt  # noqa: E402


def _min_var_long_only(cov: np.ndarray, ridge: float = 1e-5) -> np.ndarray:
    n = cov.shape[0]
    reg = cov + ridge * np.eye(n)
    try:
        raw = np.linalg.solve(reg, np.ones(n))
    except np.linalg.LinAlgError:
        return np.full(n, 1.0 / n)
    raw = np.maximum(raw, 0.0)
    s = raw.sum()
    return raw / s if s > 0 else np.full(n, 1.0 / n)


def rmt_minvar_strategy(train_rets: np.ndarray, names: list[str],
                        ridge: float = 1e-5) -> np.ndarray:
    """Long-only min-var on RIE-shrunk Σ. Falls back to sample Σ on failure."""
    n = train_rets.shape[1]
    if train_rets.shape[0] < 60:
        return np.full(n, 1.0 / n)
    out = rmt.rmt_summary(train_rets)
    if not out:
        return _min_var_long_only(np.cov(train_rets, rowvar=False, ddof=1), ridge)
    # Recompute standardized cov + eigvecs in the same convention as rmt_summary.
    means = train_rets.mean(axis=0, keepdims=True)
    stds = train_rets.std(axis=0, keepdims=True, ddof=1)
    stds = np.where(stds > 0, stds, 1.0)
    z = (train_rets - means) / stds
    sample_cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    eigvals_s, eigvecs = np.linalg.eigh(sample_cov)
    order = np.argsort(eigvals_s)[::-1]
    eigvecs = eigvecs[:, order]
    rie_eigs = np.asarray(out["rie_eigenvalues"], dtype=np.float64)
    if rie_eigs.size != eigvals_s.size:
        return _min_var_long_only(np.cov(train_rets, rowvar=False, ddof=1), ridge)
    # Reconstruct on the standardized space, then re-scale back to raw-return
    # space via diag(std).
    rie_corr = eigvecs @ np.diag(rie_eigs) @ eigvecs.T
    s = stds.reshape(-1)
    rie_cov = rie_corr * np.outer(s, s)
    return _min_var_long_only(rie_cov, ridge)


def factory(ridge: float = 1e-5):
    def strat(train_rets, names):
        return rmt_minvar_strategy(train_rets, names, ridge=ridge)
    return strat
