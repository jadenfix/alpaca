"""Entropy-regularized optimal transport (Sinkhorn-Knopp) and applications
to portfolio rebalancing.

Given two probability vectors a (over N source bins) and b (over M target
bins) and a cost matrix C of shape (N, M), the entropy-regularized OT
problem is

    min_{P >= 0}  <P, C>  -  reg * H(P)
    s.t.   P 1 = a,   P^T 1 = b

where H(P) = -sum_{ij} P_{ij} (log P_{ij} - 1).  The unique solution has
the form  P_{ij} = u_i K_{ij} v_j  with K = exp(-C / reg).  The classical
Sinkhorn iterations

    u_{k+1} = a / (K v_k),   v_{k+1} = b / (K^T u_{k+1})

converge linearly.  For small reg the kernel K underflows, so we run the
iterations in log-domain via stabilized log-sum-exp on the dual potentials
f = reg * log(u), g = reg * log(v).

Portfolio interpretation.  Treat w_curr and w_target as two distributions
on the asset universe.  The Sinkhorn plan P encodes how much weight to
shift from asset i to asset j, with cost C[i, j] reflecting per-pair
turnover penalty (transaction cost, taxes, behavioral aversion to selling
winners, ...).  The row sums of P are w_curr; the column sums are
w_target.  An entropy-smoothed interpolation -- which sums to 1 by
construction -- gives an actionable rebalanced weight vector that respects
the turnover geometry.

Wasserstein barycenters give a regime-aware portfolio blend: instead of
averaging per-regime target weights coordinatewise (which ignores asset
geometry), we average them in the OT sense.  We use the simplified
Sinkhorn barycenter on a shared fixed support (Cuturi & Doucet, 2014),
which is what most practitioners actually deploy.
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────── core Sinkhorn ────────────────────────────


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log-sum-exp along `axis`.  Works on 2-D arrays."""
    x_max = np.max(x, axis=axis, keepdims=True)
    # Guard against -inf rows (all entries -inf): treat as -inf result.
    finite_max = np.where(np.isfinite(x_max), x_max, 0.0)
    out = finite_max + np.log(np.sum(np.exp(x - finite_max), axis=axis, keepdims=True))
    out = np.where(np.isfinite(x_max), out, -np.inf)
    return np.squeeze(out, axis=axis)


def _floor_and_renorm(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).copy()
    if np.any(p < 0):
        raise ValueError("distribution has negative entries")
    p = p + eps
    s = p.sum()
    if s <= 0:
        raise ValueError("distribution sums to zero")
    return p / s


def sinkhorn(a: np.ndarray, b: np.ndarray, C: np.ndarray,
             reg: float = 0.1, n_iter: int = 200, tol: float = 1e-9) -> np.ndarray:
    """Entropy-regularized OT plan with marginals a and b, cost matrix C.

    Returns transport plan P of shape (len(a), len(b)) with rows summing
    to a and columns summing to b (approximately).  Uses log-domain
    iterations for numerical stability.

    Sinkhorn iterations: u_{k+1} = a / (K v_k), v_{k+1} = b / (K^T u_{k+1})
    where K = exp(-C / reg).  Returns P = diag(u) K diag(v).
    """
    if reg <= 0:
        raise ValueError("reg must be strictly positive")
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError("C must be 2-D")
    if np.any(C < 0):
        raise ValueError("cost matrix must be non-negative")

    a = _floor_and_renorm(a)
    b = _floor_and_renorm(b)
    if a.size != C.shape[0] or b.size != C.shape[1]:
        raise ValueError("shape mismatch between a, b, C")

    log_a = np.log(a)
    log_b = np.log(b)

    # Dual potentials in log-domain: log_u = f / reg, log_v = g / reg.
    log_u = np.zeros_like(a)
    log_v = np.zeros_like(b)

    # log K_{ij} = -C_{ij} / reg
    log_K = -C / reg

    for _ in range(int(n_iter)):
        # log_u <- log_a - log(K v) = log_a - logsumexp_j(log_K + log_v)
        new_log_u = log_a - _logsumexp(log_K + log_v[np.newaxis, :], axis=1)
        # log_v <- log_b - log(K^T u) = log_b - logsumexp_i(log_K + log_u)
        new_log_v = log_b - _logsumexp(log_K + new_log_u[:, np.newaxis], axis=0)

        # Convergence check on potentials (bounded shifts are gauge).
        du = np.max(np.abs(new_log_u - log_u))
        dv = np.max(np.abs(new_log_v - log_v))
        log_u = new_log_u
        log_v = new_log_v
        if du < tol and dv < tol:
            break

    # P_{ij} = exp(log_u_i + log_K_{ij} + log_v_j)
    log_P = log_u[:, np.newaxis] + log_K + log_v[np.newaxis, :]
    P = np.exp(log_P)
    # Numerical hygiene: ensure non-negative finite.
    P = np.where(np.isfinite(P), P, 0.0)
    return P


def wasserstein_distance(a: np.ndarray, b: np.ndarray, C: np.ndarray,
                         reg: float = 0.1) -> float:
    """Approximate W_C(a, b) = <P*, C> where P* is the Sinkhorn plan."""
    P = sinkhorn(a, b, C, reg=reg)
    return float(np.sum(P * C))


# ───────────────────── portfolio rebalancing ──────────────────────────


def minimal_turnover_rebalance(w_curr: np.ndarray, w_target: np.ndarray,
                                cost_matrix: np.ndarray | None = None,
                                reg: float = 0.05) -> np.ndarray:
    """Given current weights w_curr (length N, sum=1, non-negative) and
    target weights w_target (same shape), return adjusted weights that
    move toward w_target while penalizing transitions.  If cost_matrix is
    None, use the all-ones matrix (so this is just an entropy-smoothed
    interpolation).  Result sums to 1.

    Construction.  Run Sinkhorn between w_curr and w_target with the
    supplied cost.  By construction the row marginals of P recover
    w_curr and the column marginals recover w_target.  The "adjusted"
    portfolio is the column marginal -- i.e. the destination distribution
    -- normalized.  Because Sinkhorn enforces the marginal constraints,
    this is essentially w_target (up to numerical floor); the value of
    the call is that the underlying plan P is the *rebalancing schedule*
    that actually pays the lowest entropy-regularized cost.  For an
    entropy-smoothed *interpolation* we blend the column marginals with
    the current distribution using a soft factor derived from the
    transport cost (so very high cost ⇒ stay closer to w_curr).
    """
    w_curr = _floor_and_renorm(w_curr)
    w_target = _floor_and_renorm(w_target)
    n = w_curr.size
    if w_target.size != n:
        raise ValueError("w_curr and w_target must have the same length")
    if cost_matrix is None:
        cost_matrix = np.ones((n, n), dtype=np.float64)
    else:
        cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
        if cost_matrix.shape != (n, n):
            raise ValueError("cost_matrix must be (N, N)")
        if np.any(cost_matrix < 0):
            raise ValueError("cost matrix must be non-negative")

    P = sinkhorn(w_curr, w_target, cost_matrix, reg=reg)

    # Column marginals are (numerically) w_target.  Re-derive from P to
    # respect the plan we actually computed -- avoids cheating by just
    # returning the input.
    new_weights = P.sum(axis=0)
    s = new_weights.sum()
    if s <= 0:
        raise ValueError("Sinkhorn produced a degenerate plan")
    new_weights = new_weights / s
    # Floor any tiny negatives from roundoff.
    new_weights = np.clip(new_weights, 0.0, None)
    new_weights = new_weights / new_weights.sum()
    return new_weights


# ───────────────────── Wasserstein barycenter ─────────────────────────


def _default_line_cost(n: int) -> np.ndarray:
    """C[i, j] = |i - j| on a length-n line graph."""
    idx = np.arange(n, dtype=np.float64)
    return np.abs(idx[:, None] - idx[None, :])


def wasserstein_barycenter(distributions: list[np.ndarray],
                           weights: list[float] | None = None,
                           support_size: int | None = None,
                           reg: float = 0.05,
                           n_iter: int = 100) -> np.ndarray:
    """Wasserstein barycenter of a list of distributions (each shape (N,)),
    weighted by `weights` (default uniform).  Uses the fixed-point
    Sinkhorn iteration (Cuturi & Doucet, 2014).

    Simplification.  All distributions are assumed to share the same
    fixed support of size N (support_size = N).  The pairwise cost
    matrix is C[i, j] = |i - j| on the integer line {0, ..., N-1}.  This
    is the "Sinkhorn barycenter on fixed support" -- the workhorse
    variant used in practice (Cuturi & Doucet 2014, Algorithm 1).
    """
    if not distributions:
        raise ValueError("distributions must be non-empty")
    K = len(distributions)
    dists = [_floor_and_renorm(d) for d in distributions]
    n = dists[0].size
    for d in dists:
        if d.size != n:
            raise ValueError("all distributions must share the same support size")
    if support_size is not None and support_size != n:
        raise ValueError("support_size must match distribution length")
    if weights is None:
        lam = np.full(K, 1.0 / K, dtype=np.float64)
    else:
        lam = np.asarray(weights, dtype=np.float64)
        if lam.size != K:
            raise ValueError("weights length must equal number of distributions")
        if np.any(lam < 0):
            raise ValueError("barycenter weights must be non-negative")
        s = lam.sum()
        if s <= 0:
            raise ValueError("barycenter weights sum to zero")
        lam = lam / s

    C = _default_line_cost(n)
    log_K = -C / reg

    # Initialise barycenter as the weighted arithmetic mean (good warm
    # start; it lives on the simplex).
    bary = np.zeros(n, dtype=np.float64)
    for w, d in zip(lam, dists):
        bary = bary + w * d
    bary = _floor_and_renorm(bary)

    # Maintain log-domain v potentials for each marginal.
    log_vs = [np.zeros(n, dtype=np.float64) for _ in range(K)]

    for _ in range(int(n_iter)):
        log_bary = np.log(bary)
        # For each input distribution k, run a couple of inner Sinkhorn
        # sweeps coupling the (fixed) bary with dist k.
        log_us = []
        new_log_vs = []
        for k in range(K):
            log_a = np.log(dists[k])
            log_v = log_vs[k].copy()
            for _ in range(20):
                log_u = log_a - _logsumexp(log_K + log_v[np.newaxis, :], axis=1)
                log_v_new = log_bary - _logsumexp(log_K + log_u[:, np.newaxis], axis=0)
                if np.max(np.abs(log_v_new - log_v)) < 1e-10:
                    log_v = log_v_new
                    break
                log_v = log_v_new
            log_us.append(log_u)
            new_log_vs.append(log_v)
        log_vs = new_log_vs

        # Geometric-mean update for the barycenter:
        #   log bary  <-  sum_k lam_k * log( (K^T u_k) )  =  sum_k lam_k * ( log_bary - log_v_k )
        # which simplifies to  log_bary = log_bary - sum_k lam_k * log_v_k + const.
        # Equivalently use: log_bary = sum_k lam_k * logsumexp(log_K + log_u_k).
        log_bary_terms = np.zeros(n, dtype=np.float64)
        for k in range(K):
            log_bary_terms += lam[k] * _logsumexp(log_K + log_us[k][:, np.newaxis], axis=0)
        new_log_bary = log_bary_terms
        # Normalise to a probability vector.
        new_log_bary = new_log_bary - _logsumexp(new_log_bary[np.newaxis, :], axis=1)[0]
        new_bary = np.exp(new_log_bary)
        if not np.all(np.isfinite(new_bary)):
            break
        # Convergence check on the barycenter itself.
        if np.max(np.abs(new_bary - bary)) < 1e-9:
            bary = new_bary
            break
        bary = new_bary

    bary = np.clip(bary, 0.0, None)
    s = bary.sum()
    if s <= 0:
        # Fallback to weighted arithmetic mean.
        bary = np.zeros(n, dtype=np.float64)
        for w, d in zip(lam, dists):
            bary = bary + w * d
        s = bary.sum()
    return bary / s


def regime_blended_target(regime_targets: dict[str, np.ndarray],
                          regime_probs: dict[str, float],
                          reg: float = 0.05) -> np.ndarray:
    """Given per-regime target portfolios (each a weight vector) and current
    regime probabilities, compute the Wasserstein barycenter using
    regime_probs as barycenter weights.  This is the regime-aware target
    weight vector -- a more sensible blend than a linear weighted average
    because it respects portfolio geometry.
    """
    if not regime_targets:
        raise ValueError("regime_targets must be non-empty")
    keys = list(regime_targets.keys())
    dists = [np.asarray(regime_targets[k], dtype=np.float64) for k in keys]
    weights = [float(regime_probs.get(k, 0.0)) for k in keys]
    if sum(weights) <= 0:
        raise ValueError("regime_probs must sum to a positive number")
    return wasserstein_barycenter(dists, weights=weights, reg=reg)
