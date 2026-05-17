"""Alpha-stable distribution fit via Koutrouvelis (1980) characteristic-
function regression.

The Lévy α-stable family S(α, β, σ, μ) is the generalized CLT attractor
when variance is infinite. For α=2 we recover Gaussian; for α=1 (and β=0)
we get Cauchy.

Koutrouvelis's method:
    log |φ̂(t)|² = log(2σ^α) - σ^α |t|^α
linearizes in (log |t|, log(-log |φ̂(t)|²)) so OLS gives (α, σ^α).

Returns dict {alpha, sigma, mu, ks_vs_gaussian}.
"""
from __future__ import annotations

import math
import numpy as np


def fit_alpha_stable(x: np.ndarray, n_freqs: int = 12) -> dict:
    """Koutrouvelis CF regression estimator. Returns {alpha, sigma, mu, ks_vs_gaussian}.

    Scale is set from the median absolute deviation (MAD), which is finite
    for any α-stable distribution including Cauchy (variance is not).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 30:
        return {"alpha": 2.0, "sigma": float(x.std(ddof=1)),
                "mu": float(x.mean()), "ks_vs_gaussian": 0.0}
    mu = float(np.median(x))
    centered = x - mu
    # Robust scale: MAD is finite for all α-stable laws (std is not, for α<2).
    mad = float(np.median(np.abs(centered)))
    scale = max(mad, 1e-9)
    # Pick t ∈ [0.1, 2.0] / scale: for α=2, |φ(t)|² spans (~1, 0.02);
    # for α=1 (Cauchy), |φ(t)|² spans (~0.82, 0.018) — both regions where
    # log(-log|φ|²) is linear in log|t| with the right slope.
    t_grid = np.geomspace(0.1, 2.0, n_freqs) / scale
    log_t = np.log(t_grid)
    log_neg_log_phi = []
    valid_logt = []
    for t in t_grid:
        # Empirical CF: φ̂(t) = (1/n) Σ exp(i t x_k)
        cos_part = float(np.mean(np.cos(t * centered)))
        sin_part = float(np.mean(np.sin(t * centered)))
        phi_sq = cos_part ** 2 + sin_part ** 2
        if phi_sq <= 1e-12:
            continue
        nl = -np.log(phi_sq)
        if nl <= 0:
            continue
        log_neg_log_phi.append(math.log(nl))
        valid_logt.append(math.log(t))
    if len(valid_logt) < 3:
        return {"alpha": 2.0, "sigma": float(x.std(ddof=1)),
                "mu": mu, "ks_vs_gaussian": 0.0}
    # log(-log |φ|²) ≈ log(2 σ^α) + α · log|t|
    xs = np.array(valid_logt)
    ys = np.array(log_neg_log_phi)
    slope, intercept = np.polyfit(xs, ys, 1)
    alpha = float(np.clip(slope, 0.3, 2.0))
    # 2 σ^α = exp(intercept) → σ = (exp(intercept)/2)^(1/α)
    sigma = float(max(1e-9, (math.exp(intercept) / 2.0) ** (1.0 / alpha))) if alpha > 0 else float(x.std(ddof=1))

    # Compare KS distance vs Gaussian fit.
    ks_gauss = _ks_distance_vs_normal(centered, mu_offset=0.0, sigma=float(np.std(centered, ddof=1)))
    return {"alpha": alpha, "sigma": sigma, "mu": mu, "ks_vs_gaussian": ks_gauss}


def _ks_distance_vs_normal(x: np.ndarray, mu_offset: float, sigma: float) -> float:
    """Kolmogorov-Smirnov distance between empirical CDF of x and N(mu_offset, sigma^2)."""
    n = x.size
    if n == 0 or sigma <= 0:
        return 0.0
    x_sorted = np.sort(x)
    # Empirical CDF.
    ecdf = np.arange(1, n + 1) / n
    # Theoretical N(mu_offset, sigma^2) CDF via erf.
    z = (x_sorted - mu_offset) / (sigma * math.sqrt(2.0))
    theo = 0.5 * (1.0 + np.array([math.erf(zi) for zi in z]))
    d_plus = np.max(ecdf - theo)
    d_minus = np.max(theo - (ecdf - 1.0 / n))
    return float(max(d_plus, d_minus))
