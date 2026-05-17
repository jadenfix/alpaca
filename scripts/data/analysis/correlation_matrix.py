"""Pearson, Spearman, and Kendall correlation matrices.

All three reduce to ±1 for perfectly (anti-)correlated input; all return
0 in expectation under independence. They catch DIFFERENT dependence
structures:
  - Pearson: linear only (affine-invariant).
  - Spearman: monotone (rank-invariant; catches y = e^x).
  - Kendall: ordinal pair concordance (robust to outliers).
"""
from __future__ import annotations

import numpy as np


def pearson(data: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix over the columns of `data` (T, N)."""
    if data.shape[0] < 2:
        return np.zeros((data.shape[1], data.shape[1]))
    return np.corrcoef(data, rowvar=False)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks with tie correction. Mirrors scipy.stats.rankdata."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    n = x.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0  # 1-based ranks
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman(data: np.ndarray) -> np.ndarray:
    """Spearman correlation matrix = Pearson on column ranks."""
    if data.shape[0] < 2:
        return np.zeros((data.shape[1], data.shape[1]))
    ranks = np.column_stack([_rankdata(data[:, j]) for j in range(data.shape[1])])
    return pearson(ranks)


def kendall(data: np.ndarray) -> np.ndarray:
    """Kendall's tau-b correlation matrix. O(N^2 * T^2) — for T <= 500, fine."""
    n = data.shape[1]
    t = data.shape[0]
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = _kendall_tau_pair(data[:, i], data[:, j])
            out[j, i] = out[i, j]
    _ = t
    return out


def _kendall_tau_pair(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    tied_x = 0
    tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx == 0 and dy == 0:
                continue
            elif dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    total = n * (n - 1) // 2
    denom = ((total - tied_x) * (total - tied_y)) ** 0.5
    if denom <= 0:
        return 0.0
    return float((concordant - discordant) / denom)
