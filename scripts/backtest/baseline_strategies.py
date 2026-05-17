"""Baseline strategies used both as walk-forward sanity checks and as
the apples-to-apples comparison set against the new alpha methods.

A strategy is `f(train_rets, names) -> weights_vector`. Each function
here is stateless and deterministic given its training slice.
"""
from __future__ import annotations

import numpy as np


def equal_weight(train_rets: np.ndarray, names: list[str]) -> np.ndarray:
    """1/N long-only. Sums to 1."""
    n = train_rets.shape[1]
    return np.full(n, 1.0 / n)


def zero_weight(train_rets: np.ndarray, names: list[str]) -> np.ndarray:
    """No exposure — sanity benchmark that should produce exactly zero PnL."""
    return np.zeros(train_rets.shape[1])


def min_variance(train_rets: np.ndarray, names: list[str],
                 ridge: float = 1e-4) -> np.ndarray:
    """Global minimum-variance long-only proxy: w ∝ Σ⁻¹·1, clipped at zero,
    then renormalized.
    """
    n = train_rets.shape[1]
    cov = np.cov(train_rets, rowvar=False, ddof=1)
    reg = cov + ridge * np.eye(n)
    try:
        raw = np.linalg.solve(reg, np.ones(n))
    except np.linalg.LinAlgError:
        return np.full(n, 1.0 / n)
    raw = np.maximum(raw, 0.0)
    s = raw.sum()
    return raw / s if s > 0 else np.full(n, 1.0 / n)


def momentum_xs(train_rets: np.ndarray, names: list[str],
                lookback: int = 60) -> np.ndarray:
    """Cross-sectional momentum: long top half by trailing return, short
    bottom half. Sums to 0, gross = 1.
    """
    n = train_rets.shape[1]
    look = min(lookback, train_rets.shape[0])
    if look <= 1:
        return np.zeros(n)
    tr = train_rets[-look:].sum(axis=0)
    ranks = np.argsort(tr)
    longs = ranks[n // 2:]
    shorts = ranks[:n // 2]
    w = np.zeros(n)
    if longs.size > 0:
        w[longs] = 0.5 / longs.size
    if shorts.size > 0:
        w[shorts] = -0.5 / shorts.size
    return w


def mean_reversion_xs(train_rets: np.ndarray, names: list[str],
                      lookback: int = 5) -> np.ndarray:
    """Cross-sectional short-term mean reversion: long the losers, short
    the winners over the last `lookback` bars. Sums to 0, gross = 1.
    """
    return -momentum_xs(train_rets, names, lookback=lookback)
