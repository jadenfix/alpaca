"""Multivariate exponential-kernel Hawkes process — MLE and contagion
analysis. Companion to the single-variate :mod:`hawkes` module.

For a D-dimensional system the intensity for stream d is

    λ_d(t) = μ_d + Σ_{d'} Σ_{t_i^{d'} < t} α_{d,d'} · exp(-β_d (t - t_i^{d'}))

The branching matrix K_{d,d'} = α_{d,d'} / β_d records how strongly an
event in stream d' triggers events in stream d. Spectral radius of K
controls stationarity (must be < 1).
"""
from __future__ import annotations

import math
import numpy as np


# ───────────────────────── stability helpers ─────────────────────────

def branching_matrix(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Branching matrix K_{d,d'} = α_{d,d'} / β_d, shape (D, D)."""
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    # divide each row d by beta_d
    return a / b[:, None]


def spectral_radius(K: np.ndarray) -> float:
    """Max absolute eigenvalue of K. Process is stationary iff < 1."""
    K = np.asarray(K, dtype=np.float64)
    eigs = np.linalg.eigvals(K)
    return float(np.max(np.abs(eigs)))


# ────────────────────────── log-likelihood ───────────────────────────

def mv_hawkes_loglik(events_per_dim: list[np.ndarray],
                     mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
                     t_end: float) -> float:
    """Multivariate Ogata recursion log-likelihood.

    events_per_dim[d] is a sorted 1D array of event times in stream d.
    mu shape (D,), alpha shape (D, D), beta shape (D,).
    Returns total log-likelihood, or -inf if any parameter is
    non-positive or the kernel is non-stationary.
    """
    mu = np.asarray(mu, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    D = mu.size
    if alpha.shape != (D, D) or beta.shape != (D,):
        return float("-inf")
    if not (np.all(mu > 0) and np.all(beta > 0) and np.all(alpha >= 0)):
        return float("-inf")
    # cast event arrays
    ev = [np.asarray(e, dtype=np.float64) for e in events_per_dim]
    if len(ev) != D:
        return float("-inf")

    # ---- sum of log-intensities at events --------------------------
    # We process events in global chronological order using a recursion
    # of the per-(d, d') decaying sums R_d^{d'}(t).
    #
    # R_d^{d'}(t_now) := exp(-β_d (t_now - t_last_global))
    #                    * (R_d^{d'}(t_last_global) + #{d' events in
    #                                                  (t_last, t_now]})
    # We update R after each global step. At an event in stream d at
    # time t we evaluate λ_d(t) = μ_d + Σ_{d'} α_{d,d'} R_d^{d'}(t).

    # Build a global merged stream sorted by time, tagged by dim.
    times_list = []
    dims_list = []
    for d in range(D):
        if ev[d].size:
            times_list.append(ev[d])
            dims_list.append(np.full(ev[d].size, d, dtype=np.int64))
    if times_list:
        all_times = np.concatenate(times_list)
        all_dims = np.concatenate(dims_list)
        order = np.argsort(all_times, kind="mergesort")
        all_times = all_times[order]
        all_dims = all_dims[order]
    else:
        all_times = np.zeros(0, dtype=np.float64)
        all_dims = np.zeros(0, dtype=np.int64)

    # Check sortedness within each dim (caller contract).
    for d in range(D):
        if ev[d].size >= 2 and np.any(np.diff(ev[d]) < 0):
            return float("-inf")
    # Validate that all events lie in [0, t_end].
    if all_times.size and (all_times[0] < 0.0 or all_times[-1] > t_end + 1e-12):
        return float("-inf")

    R = np.zeros((D, D), dtype=np.float64)  # R[d, d']
    log_sum = 0.0
    t_prev = 0.0
    for k in range(all_times.size):
        t_now = float(all_times[k])
        d_now = int(all_dims[k])
        dt = t_now - t_prev
        if dt < 0.0:
            return float("-inf")
        if dt > 0.0:
            decay = np.exp(-beta * dt)            # shape (D,)
            R = R * decay[:, None]
        # Intensity at this event (use R BEFORE adding the self-arrival).
        lam = mu[d_now] + float(np.dot(alpha[d_now], R[d_now]))
        if not (lam > 0.0 and math.isfinite(lam)):
            return float("-inf")
        log_sum += math.log(lam)
        # Now register this event: it contributes to column d_now of R
        # for all rows d.
        R[:, d_now] += 1.0
        t_prev = t_now

    # ---- compensator Λ_d(t_end) ------------------------------------
    # Λ_d(T) = μ_d * T + Σ_{d'} (α_{d,d'}/β_d) *
    #          Σ_{t in stream d'} (1 - exp(-β_d (T - t)))
    integral = 0.0
    for d in range(D):
        integral += mu[d] * t_end
        for dp in range(D):
            if ev[dp].size == 0:
                continue
            contrib = np.sum(1.0 - np.exp(-beta[d] * (t_end - ev[dp])))
            integral += (alpha[d, dp] / beta[d]) * float(contrib)

    return log_sum - integral


# ───────────────────────────── fitting ───────────────────────────────

def fit_mv_hawkes(events_per_dim: list[np.ndarray], t_end: float,
                  max_iter: int = 200) -> dict | None:
    """MLE via Nelder-Mead in log-parameter space.

    Parameters fit: D (mu) + D*D (alpha) + D (beta) = D*(D+2).
    Returns dict with mu, alpha, beta, branching_matrix, spectral_radius,
    loglik — or None if no stationary fit was found.
    """
    ev = [np.asarray(e, dtype=np.float64) for e in events_per_dim]
    D = len(ev)
    if D == 0:
        return None
    total_n = sum(e.size for e in ev)
    if total_n < 5 or t_end <= 0.0:
        return None

    # ---- initial guesses ------------------------------------------
    rate = np.array([max(e.size, 1) / t_end for e in ev], dtype=np.float64)
    mu0 = np.maximum(rate * 0.5, 1e-3)
    beta0 = np.full(D, 2.0, dtype=np.float64)
    # Start with a near-diagonal triggering matrix.
    alpha0 = np.full((D, D), 0.1, dtype=np.float64)
    np.fill_diagonal(alpha0, 0.5)
    # Scale alpha so the initial spectral radius is comfortably below 1.
    K0 = branching_matrix(alpha0, beta0)
    sr0 = spectral_radius(K0)
    if sr0 >= 0.8:
        alpha0 *= 0.7 / sr0

    theta0 = _pack(mu0, alpha0, beta0)
    n_params = theta0.size

    def nll(theta: np.ndarray) -> float:
        mu, alpha, beta = _unpack(theta, D)
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(alpha))
                and np.all(np.isfinite(beta))):
            return 1e12
        K = branching_matrix(alpha, beta)
        sr = spectral_radius(K)
        ll = mv_hawkes_loglik(ev, mu, alpha, beta, t_end)
        if not math.isfinite(ll):
            return 1e12
        # Soft penalty if non-stationary (per spec).
        penalty = 0.0
        if sr >= 1.0:
            ll += -1e9 * (sr - 0.99)
            penalty = 0.0  # already baked into ll
        return -(ll + penalty)

    opt = _nelder_mead(theta0, nll, step=0.25, tol=1e-5, max_iter=max_iter * 10)
    mu, alpha, beta = _unpack(opt, D)
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(alpha))
            and np.all(np.isfinite(beta))):
        return None
    K = branching_matrix(alpha, beta)
    sr = spectral_radius(K)
    ll = mv_hawkes_loglik(ev, mu, alpha, beta, t_end)
    if not math.isfinite(ll) or sr >= 1.0:
        return None
    return {
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_matrix": K,
        "spectral_radius": sr,
        "loglik": float(ll),
    }


def _pack(mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Map (mu, alpha, beta) -> log-parameter vector."""
    D = mu.size
    out = np.empty(D + D * D + D, dtype=np.float64)
    out[:D] = np.log(np.maximum(mu, 1e-12))
    out[D:D + D * D] = np.log(np.maximum(alpha.reshape(-1), 1e-12))
    out[D + D * D:] = np.log(np.maximum(beta, 1e-12))
    return out


def _unpack(theta: np.ndarray, D: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.exp(theta[:D])
    alpha = np.exp(theta[D:D + D * D]).reshape(D, D)
    beta = np.exp(theta[D + D * D:])
    return mu, alpha, beta


# ───────────────────────── contagion ranking ─────────────────────────

def contagion_paths(K: np.ndarray, names: list[str],
                    top_k: int = 5) -> list[dict]:
    """Sort entries of K and surface the strongest cross-excitations.

    Each entry K_{d,d'} means: "shocks in stream d' trigger events in
    stream d with magnitude K_{d,d'}".
    Returns up to top_k dicts ordered by descending |K_{d,d'}| with
    keys {trigger_dim, response_dim, k_value, trigger_name, response_name}.
    """
    K = np.asarray(K, dtype=np.float64)
    D = K.shape[0]
    if K.shape != (D, D):
        raise ValueError("K must be square")
    if len(names) != D:
        raise ValueError("names length must match K dimension")
    flat = []
    for d in range(D):
        for dp in range(D):
            flat.append((d, dp, float(K[d, dp])))
    flat.sort(key=lambda r: abs(r[2]), reverse=True)
    out = []
    for d, dp, v in flat[:max(0, int(top_k))]:
        out.append({
            "trigger_dim": int(dp),
            "response_dim": int(d),
            "k_value": float(v),
            "trigger_name": names[dp],
            "response_name": names[d],
        })
    return out


# ───────────────────────── Nelder-Mead solver ────────────────────────

def _nelder_mead(start: np.ndarray, f, step: float, tol: float,
                 max_iter: int) -> np.ndarray:
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
        spread = float(np.max(np.abs(simplex[-1] - simplex[0])))
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
