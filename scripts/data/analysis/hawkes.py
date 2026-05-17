"""Hawkes exponential-kernel MLE — Python mirror of the Rust
crates/features/src/hawkes_exp.rs.

Intensity:  λ(t) = μ + α · Σ exp(-β(t - t_i))
Branching ratio: n = α/β; stationarity requires n < 1.

We provide log-likelihood via Ogata's O(n) recursion, and a Nelder-Mead
fit on log-parameters.
"""
from __future__ import annotations

import math
import numpy as np


def hawkes_loglik(times: np.ndarray, mu: float, alpha: float, beta: float) -> float:
    """Log-likelihood under Hawkes(mu, alpha, beta) on event times."""
    if times.size == 0 or mu <= 0 or alpha < 0 or beta <= 0:
        return float("-inf")
    t = np.asarray(times, dtype=np.float64)
    log_sum = math.log(mu)
    integral = mu * float(t[0])
    r = 0.0
    for i in range(1, t.size):
        dt = float(t[i] - t[i - 1])
        if dt < 0:
            return float("-inf")
        exp_term = math.exp(-beta * dt)
        r_prev_plus_one = 1.0 + r
        integral += mu * dt + (alpha / beta) * (1.0 - exp_term) * r_prev_plus_one
        r = exp_term * r_prev_plus_one
        lam = mu + alpha * r
        if lam <= 0:
            return float("-inf")
        log_sum += math.log(lam)
    return log_sum - integral


def fit_hawkes(times: np.ndarray) -> dict | None:
    """Fit (mu, alpha, beta) via Nelder-Mead on log-parameters.
    Returns dict {mu, alpha, beta, branching_ratio, half_life} or None.
    """
    t = np.asarray(times, dtype=np.float64)
    n = t.size
    if n < 5 or t[-1] <= t[0]:
        return None
    span = float(t[-1] - t[0])
    mean_iat = span / max(n, 1)
    mu0 = (n / span) * 0.5
    beta0 = 2.0 / max(mean_iat, 1e-9)
    alpha0 = 0.5 * beta0
    theta0 = np.array([math.log(mu0), math.log(alpha0), math.log(beta0)])

    def nll(theta: np.ndarray) -> float:
        mu, alpha, beta = math.exp(theta[0]), math.exp(theta[1]), math.exp(theta[2])
        if alpha / beta >= 1.0:
            return 1e8 + 1e6 * (alpha / beta - 1.0)
        return -hawkes_loglik(t, mu, alpha, beta)

    opt = _nelder_mead(theta0, nll, step=0.1, tol=1e-5, max_iter=800)
    mu, alpha, beta = math.exp(opt[0]), math.exp(opt[1]), math.exp(opt[2])
    if not (math.isfinite(mu) and math.isfinite(alpha) and math.isfinite(beta)):
        return None
    n_ratio = alpha / beta
    if n_ratio >= 1.0:
        return None
    return {
        "mu": mu, "alpha": alpha, "beta": beta,
        "branching_ratio": n_ratio,
        "half_life": math.log(2.0) / beta,
    }


def _nelder_mead(start: np.ndarray, f, step: float, tol: float, max_iter: int) -> np.ndarray:
    d = start.size
    simplex = [start.copy()]
    for i in range(d):
        v = start.copy()
        v[i] += step
        simplex.append(v)
    fvals = [f(v) for v in simplex]
    for _ in range(max_iter):
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        spread = max(np.abs(simplex[-1] - simplex[0]))
        if spread < tol:
            break
        centroid = np.mean(np.stack(simplex[:-1]), axis=0)
        worst = simplex[-1]
        reflected = centroid + (centroid - worst)
        f_ref = f(reflected)
        if f_ref < fvals[0]:
            expanded = centroid + 2.0 * (centroid - worst)
            f_exp = f(expanded)
            if f_exp < f_ref:
                simplex[-1] = expanded; fvals[-1] = f_exp
            else:
                simplex[-1] = reflected; fvals[-1] = f_ref
        elif f_ref < fvals[-2]:
            simplex[-1] = reflected; fvals[-1] = f_ref
        else:
            contracted = centroid - 0.5 * (centroid - worst)
            f_con = f(contracted)
            if f_con < fvals[-1]:
                simplex[-1] = contracted; fvals[-1] = f_con
            else:
                for i in range(1, d + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
    return simplex[0]
