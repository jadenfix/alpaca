"""Causality measures: Granger F-test (linear) + Transfer Entropy
(nonparametric) + Convergent Cross Mapping (Sugihara 2012).

Each catches a different class of causal structure:
  - Granger: linear-Gaussian causality via VAR.
  - TE: information-theoretic, model-free; reduces to Granger for
    linear-Gaussian.
  - CCM: chaos/dynamical-systems causality via Takens delay embedding;
    detects causality in deterministic nonlinear systems where Granger
    and TE often fail.
"""
from __future__ import annotations

import numpy as np

from info_theory import transfer_entropy as _te  # noqa: F401 (re-exported)


def granger_f(y: np.ndarray, x: np.ndarray, lags: int = 1) -> dict:
    """Granger causality from X -> Y. Compares restricted (Y on Y past
    only) vs unrestricted (Y on Y past + X past) OLS models.

    Returns {f_stat, p_value, ssr_r, ssr_u, dof_n, dof_d}.
    p-value is computed analytically from the F distribution CDF via
    `_f_pvalue` (no scipy).
    """
    y = np.asarray(y, dtype=np.float64).flatten()
    x = np.asarray(x, dtype=np.float64).flatten()
    n = y.size
    if x.size != n or lags < 1 or n < lags * 2 + 5:
        return {"f_stat": 0.0, "p_value": 1.0, "ssr_r": 0.0, "ssr_u": 0.0, "dof_n": 0, "dof_d": 0}
    # Build design matrices.
    n_eff = n - lags
    target = y[lags:]
    # Restricted: y on y_{t-1..t-lags} + intercept.
    xr = np.ones((n_eff, lags + 1))
    for k in range(1, lags + 1):
        xr[:, k] = y[lags - k:n - k]
    # Unrestricted adds x_{t-1..t-lags}.
    xu = np.ones((n_eff, 2 * lags + 1))
    xu[:, :lags + 1] = xr
    for k in range(1, lags + 1):
        xu[:, lags + k] = x[lags - k:n - k]
    # OLS residuals.
    br, *_ = np.linalg.lstsq(xr, target, rcond=None)
    bu, *_ = np.linalg.lstsq(xu, target, rcond=None)
    ssr_r = float(np.sum((target - xr @ br) ** 2))
    ssr_u = float(np.sum((target - xu @ bu) ** 2))
    dof_n = lags
    dof_d = n_eff - (2 * lags + 1)
    if dof_d <= 0 or ssr_u <= 0:
        return {"f_stat": 0.0, "p_value": 1.0, "ssr_r": ssr_r, "ssr_u": ssr_u,
                "dof_n": dof_n, "dof_d": dof_d}
    f_stat = ((ssr_r - ssr_u) / dof_n) / (ssr_u / dof_d)
    p = _f_pvalue(max(0.0, f_stat), dof_n, dof_d)
    return {"f_stat": float(f_stat), "p_value": float(p), "ssr_r": ssr_r, "ssr_u": ssr_u,
            "dof_n": dof_n, "dof_d": dof_d}


def _f_pvalue(f: float, d1: int, d2: int) -> float:
    """Upper-tail p-value for F(d1, d2) at f >= 0.

    Uses the relation Pr[F > f] = I_x(d2/2, d1/2)  where x = d2/(d2 + d1·f)
    and I is the regularized incomplete beta function.
    """
    if f <= 0 or d1 < 1 or d2 < 1:
        return 1.0
    x = d2 / (d2 + d1 * f)
    return _reg_incomplete_beta(x, d2 / 2.0, d1 / 2.0)


def _reg_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b) via continued fraction
    (Lentz's method) and the symmetry I_x(a,b) = 1 - I_{1-x}(b,a).
    Adequate precision for d1, d2 <= a few hundred.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Use symmetry to ensure continued fraction converges quickly.
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _reg_incomplete_beta(1 - x, b, a)
    # ln B(a, b) = lgamma(a) + lgamma(b) - lgamma(a+b)
    import math
    ln_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1 - x))
    bt = math.exp(ln_bt)
    return bt * _betacf(x, a, b) / a


def _betacf(x: float, a: float, b: float, max_iter: int = 200, eps: float = 3e-7) -> float:
    """Continued-fraction expansion for the incomplete beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


def ccm_skill(x: np.ndarray, y: np.ndarray, e: int = 3, tau: int = 1,
              library_sizes: list[int] | None = None) -> dict:
    """Convergent Cross Mapping (Sugihara 2012).

    Tests whether X causes Y by attempting to predict X from Y's Takens-
    embedded manifold. If X drives Y, then Y's manifold contains enough
    information to predict X — CCM skill ρ(library_size) increases with
    library size and converges. If X does NOT cause Y, ρ stays low.

    Args:
        x, y:        equal-length time series.
        e:           embedding dimension (typically 2-5 for financial).
        tau:         delay lag (typically 1).
        library_sizes: subsample sizes to compute ρ at; default geometric grid.

    Returns dict:
        rho_curve:        list of (library_size, ρ)
        rho_max:          best ρ observed
        converges:        True if ρ trend is monotone increasing with size
                          (slope > 0).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if y.size != n or e < 2:
        return {"rho_curve": [], "rho_max": 0.0, "converges": False}
    # Build Takens embedding of Y.
    m = n - (e - 1) * tau
    if m < 10:
        return {"rho_curve": [], "rho_max": 0.0, "converges": False}
    y_embed = np.zeros((m, e))
    for i in range(e):
        y_embed[:, i] = y[i * tau:i * tau + m]
    x_target = x[(e - 1) * tau:(e - 1) * tau + m]

    if library_sizes is None:
        # Geometric grid from 20 up to m, ~6 points.
        max_lib = m
        library_sizes = sorted(set([int(s) for s in np.geomspace(20, max_lib, num=6)]))
        library_sizes = [s for s in library_sizes if s >= e + 2 and s <= m]
    if not library_sizes:
        return {"rho_curve": [], "rho_max": 0.0, "converges": False}

    rho_curve: list[tuple[int, float]] = []
    rng = np.random.default_rng(seed=42)
    for L in library_sizes:
        # Subsample library indices.
        if L >= m:
            lib_idx = np.arange(m)
        else:
            lib_idx = rng.choice(m, size=L, replace=False)
        lib_embed = y_embed[lib_idx]
        lib_target = x_target[lib_idx]
        # Predict each non-library point using its (e+1) nearest neighbors in
        # the library (simplex projection, Sugihara 1990).
        preds: list[float] = []
        truths: list[float] = []
        for t in range(m):
            if t in lib_idx:
                continue
            v = y_embed[t]
            # Euclidean distances to all library points.
            dists = np.sum((lib_embed - v) ** 2, axis=1) ** 0.5
            # Pick e+1 nearest.
            k = min(e + 1, lib_idx.size)
            nn_idx = np.argpartition(dists, k - 1)[:k]
            nn_dists = dists[nn_idx]
            # Weights: exp(-d/d_min) per Sugihara.
            d_min = max(nn_dists.min(), 1e-12)
            w = np.exp(-nn_dists / d_min)
            w_sum = w.sum()
            if w_sum <= 0:
                continue
            w /= w_sum
            preds.append(float(np.sum(w * lib_target[nn_idx])))
            truths.append(float(x_target[t]))
        if len(preds) < 3:
            rho_curve.append((L, 0.0))
            continue
        preds_a = np.array(preds)
        truths_a = np.array(truths)
        if preds_a.std() <= 0 or truths_a.std() <= 0:
            rho_curve.append((L, 0.0))
            continue
        rho = float(np.corrcoef(preds_a, truths_a)[0, 1])
        rho_curve.append((L, rho))

    rhos = [r for _, r in rho_curve]
    rho_max = max(rhos) if rhos else 0.0
    # Trend: slope of rho vs log(library_size) should be > 0 if X causes Y.
    if len(rho_curve) >= 3:
        xs = np.log(np.array([L for L, _ in rho_curve], dtype=np.float64))
        ys = np.array(rhos)
        slope = np.polyfit(xs, ys, 1)[0]
        converges = bool(slope > 0.02 and rho_max > 0.1)
    else:
        converges = False
    return {"rho_curve": rho_curve, "rho_max": rho_max, "converges": converges}
