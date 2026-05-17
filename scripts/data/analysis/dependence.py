"""Distance correlation (Szekely-Rizzo) and Detrended Cross-Correlation
Analysis (Podobnik-Stanley DCCA).

dCor = 0 iff X ⫫ Y; catches y = x^2 where Pearson misses.
DCCA gives a scale-resolved cross-correlation that handles non-stationary
series (catches dependence visible only at specific time horizons).
"""
from __future__ import annotations

import numpy as np


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Szekely-Rizzo distance correlation. O(n^2) memory and time."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < 4 or y.size != n:
        return 0.0
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a_centered = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    b_centered = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov_sq = float((a_centered * b_centered).mean())
    dvar_x = float((a_centered * a_centered).mean())
    dvar_y = float((b_centered * b_centered).mean())
    denom = (dvar_x * dvar_y) ** 0.5
    if denom <= 0:
        return 0.0
    r2 = dcov_sq / denom
    return float(max(0.0, r2) ** 0.5)


def dcor_matrix(data: np.ndarray) -> np.ndarray:
    """Full pairwise distance correlation matrix over the columns of `data`."""
    n = data.shape[1]
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_correlation(data[:, i], data[:, j])
            out[i, j] = d
            out[j, i] = d
    return out


def dcca_rho(x: np.ndarray, y: np.ndarray, scale: int) -> float:
    """Detrended Cross-Correlation Coefficient at scale `s`.

    Podobnik-Stanley (2008) DCCA:
      1. Build cumulative profiles X(k), Y(k).
      2. Slide a window of length `scale`; in each window detrend by an OLS
         linear (or higher-order) fit; the residuals form the detrended
         increments.
      3. F_{XY}(s) = mean over windows of (product of residuals)
         F_{XX}(s), F_{YY}(s) similarly.
      4. rho_DCCA(s) = F_{XY}(s) / sqrt(F_{XX}(s) F_{YY}(s)).

    Returns rho ∈ [-1, 1]. `scale` must be >= 4 and <= n/4.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if y.size != n or scale < 4 or scale > n // 2:
        return 0.0
    # Cumulative profiles.
    x_prof = np.cumsum(x - x.mean())
    y_prof = np.cumsum(y - y.mean())
    # Sliding non-overlapping windows.
    n_wins = n // scale
    if n_wins < 2:
        return 0.0
    fxx_sum = 0.0
    fyy_sum = 0.0
    fxy_sum = 0.0
    total_pts = 0
    t = np.arange(scale, dtype=np.float64)
    for w in range(n_wins):
        a = w * scale
        b = a + scale
        xp = x_prof[a:b]
        yp = y_prof[a:b]
        # Linear detrend.
        cx = np.polyfit(t, xp, 1)
        cy = np.polyfit(t, yp, 1)
        rx = xp - np.polyval(cx, t)
        ry = yp - np.polyval(cy, t)
        fxx_sum += float(np.sum(rx * rx))
        fyy_sum += float(np.sum(ry * ry))
        fxy_sum += float(np.sum(rx * ry))
        total_pts += scale
    fxx = fxx_sum / total_pts
    fyy = fyy_sum / total_pts
    fxy = fxy_sum / total_pts
    denom = (fxx * fyy) ** 0.5
    if denom <= 0:
        return 0.0
    return float(max(-1.0, min(1.0, fxy / denom)))


def dcca_scale_table(x: np.ndarray, y: np.ndarray, scales: list[int] | None = None) -> dict[int, float]:
    """Compute rho_DCCA(s) at several scales. Returns {scale: rho}."""
    n = x.size
    if scales is None:
        # Default: log-spaced scales from 8 to n/4.
        scales = [s for s in [8, 16, 32, 64, 128] if s <= n // 4]
    return {s: dcca_rho(x, y, s) for s in scales}
