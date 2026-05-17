"""Engle-Granger cointegration pairs-trading strategy.

Plugs into the walk-forward harness in ``scripts/backtest/walkforward.py``
via the standard ``strategy(train_rets, names) -> weights`` contract.

The economic idea is classic statistical arbitrage:

  1. For every unordered pair (i, j) in the universe, regress the log
     price of asset i on the log price of asset j (with intercept) to get
     a hedge ratio beta and a residual "spread".
  2. Run an Augmented Dickey-Fuller test on the spread; if its t-stat is
     more negative than the supplied threshold (-2.86 at the 5% level for
     the no-trend specification), call the pair cointegrated.
  3. Compute the current z-score of the spread. If |z| > z_entry, take a
     dollar-balanced position: short the rich leg, long the cheap leg,
     scaled by 1/(1+|beta|) so that the gross exposure on the pair is 1.
  4. Sum across all triggered pairs, keep only the top ``max_pairs`` by
     |z|, and normalize so that sum(|w|) == 1.

Pure numpy + stdlib; OLS is solved with np.linalg.lstsq and the ADF
regression is implemented from scratch.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# ADF (Augmented Dickey-Fuller) t-statistic on the lagged-level coefficient.
# ---------------------------------------------------------------------------
def adf_tstat(series: np.ndarray, max_lag: int = 1) -> float:
    """Augmented Dickey-Fuller t-statistic on the residual series.

    Regresses Delta r_t on [1, r_{t-1}, Delta r_{t-1}, ..., Delta r_{t-p}]
    and returns the t-statistic of the lagged-level coefficient. Lower is
    more stationary; reject the unit-root null when the t-stat is more
    negative than the chosen critical value (e.g. -2.86 at 5%).
    """
    r = np.asarray(series, dtype=np.float64).reshape(-1)
    n_obs = r.size
    p = max(int(max_lag), 0)
    # Need enough rows after lagging: lose 1 for the level lag, p for the
    # difference lags, plus a few degrees of freedom for the regression.
    if n_obs < p + 5:
        return 0.0

    dr = np.diff(r)                      # length n_obs - 1
    # Align everything to the same time index. We model Delta r_t for
    # t = p+1 .. n_obs-1, so we need dr starting at index p, level
    # r_{t-1} at index p, and Delta r_{t-i} for i=1..p.
    if dr.size <= p:
        return 0.0

    y = dr[p:]                           # Delta r_t
    lag_level = r[p:-1]                  # r_{t-1}; same length as y

    # Build regressor matrix: [1, r_{t-1}, Delta r_{t-1}, ..., Delta r_{t-p}].
    n = y.size
    cols = [np.ones(n), lag_level]
    for i in range(1, p + 1):
        # Delta r_{t-i} corresponds to dr index (p - i) ... (p - i + n - 1).
        start = p - i
        cols.append(dr[start:start + n])
    X = np.column_stack(cols)

    k = X.shape[1]
    if n <= k:
        return 0.0

    # OLS via lstsq.
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    rss = float(np.dot(resid, resid))
    dof = n - k
    if dof <= 0:
        return 0.0
    sigma2 = rss / dof

    # SE of the lagged-level coefficient (column index 1).
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return 0.0
    var_b = sigma2 * xtx_inv[1, 1]
    if not np.isfinite(var_b) or var_b <= 0:
        return 0.0
    se_b = float(np.sqrt(var_b))
    if se_b == 0.0:
        return 0.0
    return float(coef[1] / se_b)


# ---------------------------------------------------------------------------
# Engle-Granger two-step cointegration test on a single pair.
# ---------------------------------------------------------------------------
def engle_granger_pair(price_a: np.ndarray, price_b: np.ndarray,
                       adf_threshold: float = -2.86) -> dict:
    """Run the Engle-Granger two-step procedure on a pair of price series.

    Regresses ``price_a`` on ``price_b`` with an intercept to obtain the
    hedge ratio ``beta``, forms the residual spread, runs an ADF test on
    the spread, and returns the diagnostics in a dict.
    """
    a = np.asarray(price_a, dtype=np.float64).reshape(-1)
    b = np.asarray(price_b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    blank = {
        "hedge_ratio": 0.0,
        "intercept": 0.0,
        "spread": np.zeros(0),
        "spread_mean": 0.0,
        "spread_std": 0.0,
        "current_z": 0.0,
        "adf_tstat": 0.0,
        "is_cointegrated": False,
    }
    if n < 10:
        return blank
    a = a[-n:]
    b = b[-n:]

    X = np.column_stack([np.ones(n), b])
    try:
        coef, *_ = np.linalg.lstsq(X, a, rcond=None)
    except np.linalg.LinAlgError:
        return blank
    alpha = float(coef[0])
    beta = float(coef[1])
    spread = a - alpha - beta * b

    mu = float(np.mean(spread))
    sigma = float(np.std(spread, ddof=1)) if spread.size > 1 else 0.0
    if sigma > 0 and np.isfinite(sigma):
        cur_z = float((spread[-1] - mu) / sigma)
    else:
        cur_z = 0.0

    t = adf_tstat(spread, max_lag=1)
    is_coint = bool(np.isfinite(t) and t < adf_threshold)

    return {
        "hedge_ratio": beta,
        "intercept": alpha,
        "spread": spread,
        "spread_mean": mu,
        "spread_std": sigma,
        "current_z": cur_z,
        "adf_tstat": float(t),
        "is_cointegrated": is_coint,
    }


# ---------------------------------------------------------------------------
# Strategy: sweep all pairs, accumulate dollar-balanced positions, normalize.
# ---------------------------------------------------------------------------
def cointegration_spread_strategy(train_rets: np.ndarray, names: list[str],
                                  z_entry: float = 2.0,
                                  adf_threshold: float = -2.86,
                                  max_pairs: int = 10) -> np.ndarray:
    """Engle-Granger pairs-trading strategy.

    For every unordered pair (i, j), reconstruct prices from log returns
    via ``exp(cumsum)``, run :func:`engle_granger_pair`, and if the pair
    is cointegrated AND the current spread |z| exceeds ``z_entry``, take
    a dollar-balanced position. Sum across triggered pairs, keep the top
    ``max_pairs`` by |z|, and normalize so that the gross exposure
    ``sum(|w|)`` equals 1.
    """
    R = np.asarray(train_rets, dtype=np.float64)
    if R.ndim != 2:
        return np.zeros(0)
    T, N = R.shape
    out = np.zeros(N, dtype=np.float64)
    if T < 60 or N < 2:
        return out

    # Reconstruct (log-cumulative) prices starting at 1.0.
    log_cum = np.cumsum(R, axis=0)
    prices = np.exp(log_cum)

    # Score every pair, then keep the top |z| performers.
    candidates: list[tuple[float, int, int, float, float]] = []
    # tuple is (abs_z, i, j, beta, signed_z)
    for i, j in combinations(range(N), 2):
        info = engle_granger_pair(prices[:, i], prices[:, j],
                                  adf_threshold=adf_threshold)
        if not info["is_cointegrated"]:
            continue
        z = info["current_z"]
        if not np.isfinite(z) or abs(z) <= z_entry:
            continue
        candidates.append((abs(z), i, j, info["hedge_ratio"], z))

    if not candidates:
        return out

    # Sort by |z| descending and keep top max_pairs.
    candidates.sort(key=lambda x: x[0], reverse=True)
    keep = candidates[:max(int(max_pairs), 0)]
    if not keep:
        return out

    for _, i, j, beta, z in keep:
        denom = 1.0 + abs(beta)
        if denom <= 0 or not np.isfinite(denom):
            continue
        # When z > 0, spread = a - alpha - beta*b is "too high" => a is rich
        # relative to beta*b. Short a, long beta*b.
        # When z < 0, the reverse: long a, short beta*b.
        sign = -1.0 if z > 0 else 1.0
        # Pair weights: w_a = sign * 1/denom, w_b = -sign * beta/denom
        # (use signed beta so we hedge in the direction implied by the
        # regression — if beta is negative, the b-leg flips automatically).
        out[i] += sign * (1.0 / denom)
        out[j] += -sign * (beta / denom)

    gross = float(np.sum(np.abs(out)))
    if gross <= 0 or not np.isfinite(gross):
        return np.zeros(N, dtype=np.float64)
    return out / gross


# ---------------------------------------------------------------------------
# Factory for plug-in style usage by the walk-forward harness.
# ---------------------------------------------------------------------------
def factory(z_entry: float = 2.0, adf_threshold: float = -2.86,
            max_pairs: int = 10) -> Callable[[np.ndarray, list[str]], np.ndarray]:
    """Return a closure with the cointegration-spread parameters bound."""
    def _strategy(train_rets: np.ndarray, names: list[str]) -> np.ndarray:
        return cointegration_spread_strategy(
            train_rets, names,
            z_entry=z_entry,
            adf_threshold=adf_threshold,
            max_pairs=max_pairs,
        )
    _strategy.__name__ = f"cointegration_spread(z={z_entry},adf={adf_threshold},k={max_pairs})"
    return _strategy
