"""Empirical copula + upper/lower tail dependence + parametric Gumbel /
Clayton MLE.

The copula C(u, v) = Pr[F_X(X) ≤ u, F_Y(Y) ≤ v] decouples the joint
distribution from the marginals. Tail dependence λ_U = lim_{u→1}
Pr[V > u | U > u] captures "crash-together" risk that Pearson misses.
"""
from __future__ import annotations

import math
import numpy as np


def _ranks(x: np.ndarray) -> np.ndarray:
    """Empirical CDF values F̂(x_i) = rank(x_i) / (n + 1)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    n = x.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks / (n + 1)


def upper_tail_dependence(x: np.ndarray, y: np.ndarray, q: float = 0.95) -> float:
    """Empirical upper-tail dependence at quantile q:
        λ_U(q) = Pr[F_Y(Y) > q | F_X(X) > q]
    A useful proxy for the limit. Bounded in [0, 1].
    """
    u = _ranks(x)
    v = _ranks(y)
    above_u = u > q
    n_above = int(np.sum(above_u))
    if n_above == 0:
        return 0.0
    return float(np.sum((v > q) & above_u) / n_above)


def lower_tail_dependence(x: np.ndarray, y: np.ndarray, q: float = 0.05) -> float:
    u = _ranks(x)
    v = _ranks(y)
    below_u = u < q
    n_below = int(np.sum(below_u))
    if n_below == 0:
        return 0.0
    return float(np.sum((v < q) & below_u) / n_below)


def fit_gumbel_theta(x: np.ndarray, y: np.ndarray) -> float:
    """Gumbel copula MLE via Kendall's tau inversion.
    Gumbel-theta and Kendall tau are related by θ = 1 / (1 - τ).
    """
    tau = _kendall_tau_pair(x, y)
    tau = max(min(tau, 0.99), 0.0)
    return 1.0 / (1.0 - tau) if tau < 0.99 else 100.0


def fit_clayton_theta(x: np.ndarray, y: np.ndarray) -> float:
    """Clayton copula via Kendall's tau:  θ = 2τ / (1 - τ)."""
    tau = _kendall_tau_pair(x, y)
    tau = max(min(tau, 0.99), -0.5)
    return 2.0 * tau / (1.0 - tau) if tau < 0.99 else 100.0


def _kendall_tau_pair(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx == 0 or dy == 0:
                continue
            if (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total
