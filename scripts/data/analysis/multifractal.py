"""Multifractal Detrended Fluctuation Analysis (MF-DFA) — Kantelhardt 2002.

For a series x_t, the q-th order fluctuation function F_q(s) scales as
    F_q(s) ~ s^h(q)
where h(q) is the generalized Hurst exponent. A monofractal has constant
h(q); a multifractal has a spectrum α(q), f(α) related by Legendre
transform. The WIDTH Δα = max(α) - min(α) measures non-Gaussian
complexity (cascade strength, intermittency).

Empirical financial returns typically have Δα ≈ 0.3–0.6 (significantly
multifractal); Brownian motion has Δα ≈ 0.
"""
from __future__ import annotations

import numpy as np


def mfdfa(x: np.ndarray, q_values: list[float] | None = None,
          scales: list[int] | None = None, poly_order: int = 1) -> dict:
    """MF-DFA.

    Args:
        x:           1-D time series.
        q_values:    moments to compute (default {-5, -4, ..., +5}).
        scales:      window sizes (default log-spaced 8 -> n/4).
        poly_order:  detrending polynomial order (1 = linear, 2 = quadratic).

    Returns dict:
        h_q:                {q: h(q)} generalized Hurst exponents.
        f_q_s:              {q: [(s, F_q(s)) ...]} fluctuation curves.
        alpha:              singularity exponents alpha(q) = h(q) + q dh/dq.
        f_alpha:            multifractal spectrum f(alpha).
        width:              max(alpha) - min(alpha) — complexity score.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 32:
        return {}
    if q_values is None:
        q_values = [-5, -3, -2, -1, 0.5, 1.0, 2.0, 3.0, 5.0]
    if scales is None:
        # log-spaced from 8 to n/4
        max_s = max(8, n // 4)
        scales = sorted(set([int(s) for s in np.geomspace(8, max_s, num=10)]))

    # Profile: cumulative sum of mean-deviations.
    y = np.cumsum(x - x.mean())

    h_q: dict[float, float] = {}
    f_q_s: dict[float, list[tuple[int, float]]] = {}
    for q in q_values:
        f_q_per_scale = []
        for s in scales:
            if s < 4 or 2 * s > n:
                continue
            # Non-overlapping windows in both directions.
            n_wins = n // s
            if n_wins < 4:
                continue
            f2_vals = []
            # Forward sweep.
            for w in range(n_wins):
                a = w * s
                b = a + s
                seg = y[a:b]
                t = np.arange(s, dtype=np.float64)
                coef = np.polyfit(t, seg, poly_order)
                fit = np.polyval(coef, t)
                f2_vals.append(float(np.mean((seg - fit) ** 2)))
            # Reverse sweep (Kantelhardt 2002 §3.2 — use the back end too).
            offset = n - n_wins * s
            for w in range(n_wins):
                a = offset + w * s
                b = a + s
                seg = y[a:b]
                t = np.arange(s, dtype=np.float64)
                coef = np.polyfit(t, seg, poly_order)
                fit = np.polyval(coef, t)
                f2_vals.append(float(np.mean((seg - fit) ** 2)))
            f2 = np.array(f2_vals)
            if f2.size == 0 or np.any(f2 < 0):
                continue
            if abs(q) < 1e-9:
                # q=0 limit: F_0(s) = exp(0.5 * mean(log F^2))
                fq = float(np.exp(0.5 * np.mean(np.log(np.maximum(f2, 1e-300)))))
            else:
                fq = float((np.mean(f2 ** (q / 2.0))) ** (1.0 / q))
            f_q_per_scale.append((s, fq))
        if len(f_q_per_scale) < 4:
            continue
        # Regress log F_q vs log s to extract h(q).
        log_s = np.log(np.array([s for s, _ in f_q_per_scale], dtype=np.float64))
        log_fq = np.log(np.array([f for _, f in f_q_per_scale], dtype=np.float64))
        slope = np.polyfit(log_s, log_fq, 1)[0]
        h_q[q] = float(slope)
        f_q_s[q] = f_q_per_scale

    # Multifractal spectrum via Legendre transform:
    #   τ(q) = q h(q) - 1
    #   α(q) = dτ/dq
    #   f(α) = q α - τ
    sorted_q = sorted(h_q.keys())
    if len(sorted_q) >= 3:
        qs = np.array(sorted_q)
        hs = np.array([h_q[q] for q in sorted_q])
        tau = qs * hs - 1.0
        # Central difference for dτ/dq.
        alpha = np.gradient(tau, qs)
        f_alpha = qs * alpha - tau
        width = float(alpha.max() - alpha.min())
    else:
        alpha = np.array([])
        f_alpha = np.array([])
        width = 0.0

    return {
        "h_q": h_q,
        "f_q_s": f_q_s,
        "alpha": alpha.tolist(),
        "f_alpha": f_alpha.tolist(),
        "width": width,
    }
