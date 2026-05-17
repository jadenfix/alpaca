"""RIE-cleaned residual mean-reversion strategy.

The intuition: a returns panel decomposes into a low-rank "market modes"
component (the eigenvectors above the Marchenko-Pastur bulk after RIE
shrinkage) plus an idiosyncratic residual. The market component is
typically near a random walk; the residual is much closer to mean-
reverting. We:

  1. Run the RMT pipeline on the train window (rmt_summary), which
     standardizes, eigendecomposes the correlation matrix, then applies
     RIE shrinkage. The leading K eigenvectors are kept as factor loadings.
  2. Form factor returns F = train @ eigvecs[:, :K], regress each asset
     on F to get residuals.
  3. Z-score the latest residual against the std of the PRIOR `window`
     residuals (no look-ahead beyond the train slice).
  4. Bet AGAINST the residual: w_i ∝ -sign(z_i) * min(|z_i|/threshold, 1)
     gated by |z_i| > threshold.
  5. Demean so Σw = 0; rescale so Σ|w| ≤ 1.

This file is import-clean for both
    `from rmt_residual import rmt_residual_strategy`
(via sys.path adjustment) and
    `from scripts.backtest.strategies.rmt_residual import ...`
"""
from __future__ import annotations

import os
import sys
from typing import Callable

import numpy as np


# Make the sibling analysis package importable when callers (e.g. the
# walk-forward harness or the test suite) only put scripts/backtest on
# sys.path. We resolve the analysis directory relative to this file.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ANALYSIS = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "analysis"))
if _ANALYSIS not in sys.path:
    sys.path.insert(0, _ANALYSIS)

from rmt_spectral import rmt_summary  # noqa: E402


def _resolve_n_factors(train_rets: np.ndarray, n_factors: int | None,
                       summary: dict) -> int:
    """Decide how many leading eigenvectors to project out.

    - explicit override takes precedence
    - otherwise use the count of empirical eigenvalues above the MP bulk
    - if that count is zero, keep at least the market mode (1)
    - cap at N-1 so the residual is not identically zero
    """
    n = train_rets.shape[1]
    if n_factors is None:
        k = int(summary.get("n_above_mp", 0) or 0)
        if k <= 0:
            k = 1
    else:
        k = int(n_factors)
    k = max(1, min(k, max(1, n - 1)))
    return k


def rmt_residual_strategy(train_rets: np.ndarray, names: list[str],
                          n_factors: int | None = None,
                          z_threshold: float = 1.5,
                          window: int = 60) -> np.ndarray:
    """RIE-cleaned residual mean-reversion. Returns weight vector (N,).

    Algorithm:
      1. Compute RIE-shrunk covariance via rmt_spectral.rmt_summary on train.
      2. Find n_factors = number of eigenvalues above the MP bulk
         (or use the supplied n_factors override).
      3. The top-n_factors eigenvectors define the "market modes".
      4. For each asset i, regress its returns on the n_factors factor
         portfolios (factor returns = train @ eigvec_k). Compute the
         current residual = train[-1, i] - factor_model_fit[-1, i].
      5. Z-score the latest residual using the std of the last `window`
         residuals for that asset (PRIOR observations only — no leak).
      6. Weight = -sign(z) * min(|z| / z_threshold, 1) but only if
         |z| > z_threshold.
      7. Demean so Σw = 0, then normalize so Σ|w| ≤ 1.
    """
    train_rets = np.asarray(train_rets, dtype=np.float64)
    if train_rets.ndim != 2:
        raise ValueError("train_rets must be (T, N)")
    t, n = train_rets.shape
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Conservative short-train guard. We need at least `window+1` rows
    # so we can z-score the latest residual against `window` prior ones,
    # plus enough degrees of freedom for the factor regression.
    nf_lo = 1 if n_factors is None else max(1, int(n_factors))
    min_required = max(window + 1, 2 * nf_lo + 10)
    if t < min_required:
        return np.zeros(n, dtype=np.float64)

    summary = rmt_summary(train_rets)
    if not summary or "eigenvalues" not in summary:
        return np.zeros(n, dtype=np.float64)
    eigvecs = summary.get("eigenvectors")
    if eigvecs is None:
        # rmt_summary doesn't return eigvecs directly; recompute on z-scored
        # data so the eigenvectors match its eigenvalue ordering.
        means = train_rets.mean(axis=0, keepdims=True)
        stds = train_rets.std(axis=0, keepdims=True, ddof=1)
        stds = np.where(stds > 0, stds, 1.0)
        z = (train_rets - means) / stds
        cov = (z.T @ z) / max(t - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        eigvecs = vecs[:, ::-1]  # match rmt_summary's descending order

    k = _resolve_n_factors(train_rets, n_factors, summary)
    # Re-check the short-train guard with the resolved k.
    min_required = max(window + 1, 2 * k + 10)
    if t < min_required:
        return np.zeros(n, dtype=np.float64)

    # Build factor returns on the standardized space so the eigenvectors
    # (which live in standardized space) are applied consistently.
    means = train_rets.mean(axis=0, keepdims=True)
    stds = train_rets.std(axis=0, keepdims=True, ddof=1)
    stds_safe = np.where(stds > 0, stds, 1.0)
    z = (train_rets - means) / stds_safe

    V = eigvecs[:, :k]                  # (N, K)
    F = z @ V                            # (T, K) factor returns
    # Augment with intercept so any drift is absorbed by the model.
    X = np.column_stack([np.ones(t), F])  # (T, K+1)

    # Solve in one shot for all assets: betas (K+1, N).
    betas, *_ = np.linalg.lstsq(X, z, rcond=None)
    fit = X @ betas                      # (T, N)
    resid = z - fit                      # (T, N) residuals in z-space

    # Latest residual and its prior-window standard deviation per asset.
    latest = resid[-1, :]                # (N,)
    prior_lo = max(0, t - 1 - window)
    prior = resid[prior_lo:t - 1, :]     # excludes the latest observation
    if prior.shape[0] < 2:
        return np.zeros(n, dtype=np.float64)
    sd = prior.std(axis=0, ddof=1)
    sd_safe = np.where(sd > 0, sd, np.nan)
    zs = latest / sd_safe                # NaN where sd was zero
    zs = np.where(np.isfinite(zs), zs, 0.0)

    # Gate by threshold, scale linearly, cap at 1, bet AGAINST the residual.
    mag = np.minimum(np.abs(zs) / max(z_threshold, 1e-12), 1.0)
    fired = np.abs(zs) > z_threshold
    raw = np.where(fired, -np.sign(zs) * mag, 0.0)

    # Demean so Σw = 0 (only when at least two assets fire — otherwise
    # demeaning a single non-zero entry would zero it out spuriously).
    if int(np.sum(fired)) >= 2:
        raw = raw - raw.mean()
    elif int(np.sum(fired)) == 1:
        # Pair the single bet against an equal-and-opposite spread across
        # the rest of the universe so the book is still dollar-neutral.
        idx = int(np.argmax(fired.astype(np.int64)))
        offset = np.full(n, -raw[idx] / max(n - 1, 1))
        offset[idx] = 0.0
        raw = raw + offset
        raw[idx] = raw[idx]  # explicit no-op for clarity

    # Cap gross at 1 (but never blow up on all-zero).
    gross = float(np.sum(np.abs(raw)))
    if gross > 1.0:
        raw = raw / gross
    return raw.astype(np.float64)


def factory(n_factors: int | None = None, z_threshold: float = 1.5,
            window: int = 60) -> Callable[[np.ndarray, list[str]], np.ndarray]:
    """Return a closure with bound parameters, so callers can plug into the
    walk-forward harness with their own config: e.g. `factory(z_threshold=2.0)`.
    """
    def _strategy(train_rets: np.ndarray, names: list[str]) -> np.ndarray:
        return rmt_residual_strategy(train_rets, names,
                                     n_factors=n_factors,
                                     z_threshold=z_threshold,
                                     window=window)
    _strategy.__name__ = (f"rmt_residual(k={n_factors},"
                          f"z={z_threshold},w={window})")
    return _strategy
