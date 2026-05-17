"""Within-cluster mean-reversion with a market-z gate.

This strategy is plugin-compatible with the walk-forward harness at
`scripts/backtest/walkforward.py`. Its core idea fuses two econophysics
findings already in this repo:

  1. The Minimum Spanning Tree of the correlation distance matrix
     d_ij = sqrt(2(1 - rho_ij)) (Mantegna 1999) recovers the market's
     hierarchical block structure. Names connected by short MST edges
     belong to the same sector / risk cluster.

  2. Short-horizon mean reversion is robust WITHIN such clusters
     (relative-value), but is dominated by the market mode across
     clusters. We therefore project out the market by gating on an
     equal-weight market z-score: when the market has just had an
     anomalously large move, refuse to trade — otherwise we'd be buying
     losers in the middle of a crash (where everyone is a loser).

Algorithm (deterministic given train_rets):

  1. Pearson correlation of train_rets.
  2. d = sqrt(2*(1-rho)); Kruskal MST; union-find with distance_cutoff
     yields the active clusters (size >= 2 only).
  3. mkt = train_rets.mean(axis=1); market_z = mean(mkt[-lookback:]) /
     std(mkt, ddof=1). If |market_z| > market_z_max, refuse to trade.
  4. For each cluster:
       cumret_i  = sum(train_rets[-lookback:, i])  for i in cluster
       z_i       = (cumret_i - mean(cumret)) / std(cumret, ddof=1)
       if any |z_i| > within_z_threshold:
           w_i  += -sign(z_i) * clip(|z_i| - within_z_threshold, 0, 1)
  5. Normalize so Σ|w| = 1 if non-zero.

Returns a length-N weight vector. Long-short, sums roughly to zero.
"""
from __future__ import annotations

import numpy as np

from correlation_matrix import pearson
from network import correlation_to_distance, mst
from alpha_translation import mst_clusters


def cluster_meanrev_strategy(train_rets: np.ndarray, names: list[str],
                             distance_cutoff: float = 1.0,
                             within_z_threshold: float = 1.5,
                             market_z_max: float = 0.5,
                             lookback: int = 5) -> np.ndarray:
    """Within-cluster mean-reversion with market-z filter.

    See module docstring for the full algorithm.
    """
    train_rets = np.asarray(train_rets, dtype=np.float64)
    if train_rets.ndim != 2:
        return np.zeros(len(names), dtype=np.float64)
    t, n = train_rets.shape
    if n != len(names):
        return np.zeros(n, dtype=np.float64)
    # Hard floor on history. The harness's min_history defaults match,
    # but enforce locally so direct callers behave the same way.
    if t < 60:
        return np.zeros(n, dtype=np.float64)
    if lookback < 1 or lookback > t:
        return np.zeros(n, dtype=np.float64)

    # --- Step 1+2: correlation -> distance -> MST -> clusters --------
    corr = pearson(train_rets)
    dist = correlation_to_distance(corr)
    edges = mst(dist)
    clusters = mst_clusters(edges, n, names, distance_cutoff=distance_cutoff)
    if not clusters:
        return np.zeros(n, dtype=np.float64)

    # --- Step 3: market-z gate --------------------------------------
    mkt = train_rets.mean(axis=1)
    mkt_std = float(np.std(mkt, ddof=1)) if mkt.size > 1 else 0.0
    if mkt_std <= 0.0:
        return np.zeros(n, dtype=np.float64)
    recent_mkt = float(np.mean(mkt[-lookback:]))
    market_z = recent_mkt / mkt_std
    if abs(market_z) >= market_z_max:
        return np.zeros(n, dtype=np.float64)

    # --- Step 4: within-cluster z-score mean-reversion ----------------
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    weights = np.zeros(n, dtype=np.float64)
    cum_window = train_rets[-lookback:]  # (lookback, N)
    cumret_all = cum_window.sum(axis=0)

    any_fire = False
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        idxs = np.array([name_to_idx[nm] for nm in cluster if nm in name_to_idx],
                        dtype=np.int64)
        if idxs.size < 2:
            continue
        cumret = cumret_all[idxs]
        mu = float(np.mean(cumret))
        sd = float(np.std(cumret, ddof=1))
        if sd <= 0.0:
            continue
        z = (cumret - mu) / sd
        abs_z = np.abs(z)
        if not np.any(abs_z > within_z_threshold):
            continue
        magnitude = np.clip(abs_z - within_z_threshold, 0.0, 1.0)
        # Fade the move: high z -> short, low z -> long.
        contrib = -np.sign(z) * magnitude
        weights[idxs] += contrib
        any_fire = True

    if not any_fire:
        return np.zeros(n, dtype=np.float64)

    gross = float(np.sum(np.abs(weights)))
    if gross <= 0.0:
        return np.zeros(n, dtype=np.float64)
    return weights / gross


def factory(distance_cutoff: float = 1.0, within_z_threshold: float = 1.5,
            market_z_max: float = 0.5, lookback: int = 5):
    """Return a closure with bound parameters for the walk-forward harness.

    The harness calls `strategy(train_rets, names)` only — this factory
    lets callers pre-bind tuning knobs without partial-function plumbing.
    """
    def _bound(train_rets: np.ndarray, names: list[str]) -> np.ndarray:
        return cluster_meanrev_strategy(
            train_rets, names,
            distance_cutoff=distance_cutoff,
            within_z_threshold=within_z_threshold,
            market_z_max=market_z_max,
            lookback=lookback,
        )
    _bound.__name__ = (
        f"cluster_meanrev[cut={distance_cutoff},wz={within_z_threshold},"
        f"mz={market_z_max},lb={lookback}]"
    )
    return _bound
