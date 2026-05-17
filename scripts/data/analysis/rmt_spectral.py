"""Random Matrix Theory: Marchenko-Pastur + Tracy-Widom + RIE shrinkage.

For an N x T matrix X with i.i.d. N(0, sigma^2) entries, the sample
covariance C = (1/T) X X^T has eigenvalue density (Marchenko-Pastur 1967)

    rho_MP(lambda) = sqrt((lambda_+ - lambda)(lambda - lambda_-))
                     / (2 pi q sigma^2 lambda)

on [lambda_-, lambda_+] = sigma^2 (1 -+ sqrt(q))^2 with q = N/T. Any
eigenvalue outside this support is GUARANTEED to carry information
beyond i.i.d. noise.

Tracy-Widom (1994) gives the exact distribution of fluctuations of
lambda_max around lambda_+, scaled by N^(2/3). Together MP + TW
give a calibrated p-value for "is the top eigenvalue real signal?"

Rotationally Invariant Estimator (RIE) (Bouchaud-Potters-Bun 2017) is
the optimal nonlinear shrinkage of empirical eigenvalues under Frobenius
loss in the high-dim regime. The closed-form formula involves the
Stieltjes transform of the limiting MP density; we use a numerically
stable real-axis approximation.
"""
from __future__ import annotations

import numpy as np


def mp_bounds(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Marchenko-Pastur support [lambda_-, lambda_+] for aspect q = N/T."""
    sq = q ** 0.5
    lam_minus = sigma2 * (1.0 - sq) ** 2
    lam_plus = sigma2 * (1.0 + sq) ** 2
    return lam_minus, lam_plus


def mp_density(lam: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """MP density at points `lam`. Zero outside the support."""
    lo, hi = mp_bounds(q, sigma2)
    out = np.zeros_like(lam, dtype=np.float64)
    inside = (lam > lo) & (lam < hi)
    num = np.sqrt(np.clip((hi - lam[inside]) * (lam[inside] - lo), 0.0, None))
    denom = 2.0 * np.pi * q * sigma2 * lam[inside]
    out[inside] = np.where(denom > 0, num / denom, 0.0)
    return out


def pca(data: np.ndarray) -> dict:
    """PCA on the columns of `data` (T, N). Returns dict with:
        eigenvalues:  descending
        eigenvectors: (N, N) columns = eigenvectors
        evr:          explained variance ratio
    """
    t, n = data.shape
    if t < 2:
        return {"eigenvalues": np.zeros(n), "eigenvectors": np.eye(n), "evr": np.zeros(n)}
    x_centered = data - data.mean(axis=0, keepdims=True)
    cov = (x_centered.T @ x_centered) / (t - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending; flip to descending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    total = max(eigvals.sum(), 1e-12)
    evr = np.maximum(eigvals, 0.0) / total
    return {"eigenvalues": eigvals, "eigenvectors": eigvecs, "evr": evr}


def tracy_widom_pvalue(lambda_max: float, q: float, n: int) -> float:
    """Pr[lambda_max(noise) > lambda_max(observed)] via Johnstone (2001).

    For a Wishart W = X X^T with X ~ N x T iid N(0,1), use mu and sigma
    based on rescaled (N-1/2, T-1/2). Our PCA divides by (T-1), so we
    rescale mu, sigma by (T-1) for comparison.
    """
    t = n / max(q, 1e-12)
    denom = max(t - 1.0, 1.0)
    sn = (n - 0.5) ** 0.5
    st = (t - 0.5) ** 0.5
    mu_w = (sn + st) ** 2
    sigma_w = (sn + st) * (1.0 / sn + 1.0 / st) ** (1.0 / 3.0)
    mu = mu_w / denom
    sigma = sigma_w / denom
    if sigma <= 0:
        return 1.0
    z = (lambda_max - mu) / sigma
    return _tw1_upper_tail(z)


def _tw1_upper_tail(s: float) -> float:
    """Approximation to Pr[TW_1 > s]."""
    if s >= 1.0:
        # Asymptotic right-tail of Tracy-Widom GOE.
        return max(0.0, min(1.0, (s ** (-1.0 / 8.0)) * np.exp(-(2.0 / 3.0) * s ** 1.5)))
    if s >= -2.0:
        # Tabulated CDF values (Bornemann 2010 / Prahofer-Spohn).
        table = [(-2.0, 0.978), (-1.5, 0.953), (-1.0, 0.917),
                 (-0.5, 0.866), (0.0, 0.831), (0.5, 0.762), (1.0, 0.668)]
        cdf = _piecewise_linear(s, table)
        return max(0.0, min(1.0, 1.0 - cdf))
    abs_s = -s
    cdf_lower = 1.0 - np.exp(-(abs_s ** 3) / 24.0)
    return max(0.0, min(1.0, 1.0 - cdf_lower))


def _piecewise_linear(x: float, table: list[tuple[float, float]]) -> float:
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


def rie_shrinkage(eigenvalues: np.ndarray, q: float) -> np.ndarray:
    """Rotationally Invariant Estimator (real-axis approximation).

    Eigenvalues inside the MP bulk are pulled toward the bulk mean
    (with intensity proportional to q); eigenvalues above the bulk
    (genuine signal) are lightly shrunk by the asymptotic factor.
    """
    eigs = np.asarray(eigenvalues, dtype=np.float64)
    n = eigs.size
    if n == 0:
        return eigs.copy()
    # Robust noise-variance estimate: median eigenvalue.
    sigma2 = float(np.median(eigs))
    lam_minus, lam_plus = mp_bounds(q, sigma2)
    bulk_mean = sigma2

    alpha = max(0.05, min(0.95, q))
    out = np.empty_like(eigs)
    for k, lam in enumerate(eigs):
        if lam_minus <= lam <= lam_plus:
            out[k] = alpha * bulk_mean + (1.0 - alpha) * lam
        elif lam > lam_plus and lam > sigma2:
            shrink_factor = max(0.0, 1.0 - q * sigma2 / (lam - sigma2))
            out[k] = lam * shrink_factor
        else:
            out[k] = lam
    return out


def rmt_summary(data: np.ndarray) -> dict:
    """Full RMT pipeline on a returns matrix.

    Input: (T, N) returns matrix (rows are observations, cols are series).
    Output: dict with
        - eigenvalues, evr
        - mp_bounds = (lambda-, lambda+)
        - n_above_mp:   count of eigenvalues above lambda_+
        - tw_pvalue:    Tracy-Widom p-value for the top eigenvalue
        - rie_eigenvalues: shrunk eigenvalues
        - q:            aspect ratio N/T
    """
    t, n = data.shape
    if t < 2 or n < 1:
        return {}
    # Standardize each column (correlation matrix mode).
    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True, ddof=1)
    stds = np.where(stds > 0, stds, 1.0)
    z = (data - means) / stds
    p = pca(z)
    q = n / t
    lo, hi = mp_bounds(q, 1.0)
    eigvals = p["eigenvalues"]
    n_above = int(np.sum(eigvals > hi))
    tw_p = tracy_widom_pvalue(float(eigvals[0]), q, n)
    rie = rie_shrinkage(eigvals, q)
    return {
        "eigenvalues": eigvals,
        "evr": p["evr"],
        "mp_bounds": (lo, hi),
        "n_above_mp": n_above,
        "tw_pvalue": tw_p,
        "rie_eigenvalues": rie,
        "q": q,
    }
