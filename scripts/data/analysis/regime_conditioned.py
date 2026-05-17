"""Regime-conditioned sign-predictability screening.

A signal that fires only in one volatility regime is still a signal —
but the unconditional hit-rate may average it away to noise. This
module splits returns by a continuous regime coordinate (rolling
realized volatility of the equal-weighted portfolio, computed from
prior data only to avoid look-ahead), quantile-buckets that coordinate,
then re-runs the alpha_translation.te_sign_predictability test inside
each bucket.

Keep rule: a candidate edge is retained iff its conditional sign
predictability "fires" (hit_rate > 0.52 AND p < 0.10) in at least
`min_buckets_firing` of the `n_buckets` regime buckets. This is the
econophysics analogue of conditioning on a market state before
believing a lead-lag is real.

All math is pure numpy. No scipy / sklearn / pandas / matplotlib.
"""
from __future__ import annotations

import math

import numpy as np


# ───────────────────────── regime coordinate ─────────────────────────

def regime_coordinate(rets: np.ndarray, window: int = 60) -> np.ndarray:
    """1D regime indicator: rolling realized volatility of the
    equal-weighted portfolio, computed over the `window` bars ending
    at (and including) t.

    Look-ahead is strictly forbidden: regime[t] must depend only on
    rets[:t+1, :]. For t < window - 1 the coordinate is NaN.

    Args:
        rets:   (T, N) array of per-bar returns.
        window: number of trailing bars used to compute realized vol.

    Returns:
        1D float array of length T, with NaN for t < window - 1.
    """
    rets = np.asarray(rets, dtype=np.float64)
    if rets.ndim == 1:
        rets = rets[:, None]
    T, N = rets.shape
    # Equal-weighted portfolio return per bar.
    port = rets.mean(axis=1)
    out = np.full(T, np.nan, dtype=np.float64)
    if window < 2 or T < window:
        return out
    for t in range(window - 1, T):
        # Window of length `window` ending at t inclusive: indices [t-window+1, t].
        w = port[t - window + 1: t + 1]
        # Sample std (ddof=1) is the standard realized-vol estimator.
        out[t] = float(np.std(w, ddof=1))
    return out


# ───────────────────────── quantile bucketing ─────────────────────────

def regime_buckets(regime: np.ndarray, n_buckets: int = 5) -> np.ndarray:
    """Quantile-bucket the regime coordinate into `n_buckets` bins.

    NaN entries (e.g. early bars before the rolling window fills) are
    assigned bucket -1 so callers can mask them.

    Args:
        regime:    1D regime coordinate.
        n_buckets: number of equal-frequency buckets.

    Returns:
        int array of length len(regime), values in {-1, 0, …, n_buckets-1}.
    """
    regime = np.asarray(regime, dtype=np.float64)
    out = np.full(regime.shape, -1, dtype=np.int64)
    valid = ~np.isnan(regime)
    if not np.any(valid) or n_buckets < 1:
        return out
    vals = regime[valid]
    # Equal-frequency edges via empirical quantiles.
    # Use n_buckets+1 cut points; drop the outer two when calling digitize.
    qs = np.linspace(0.0, 1.0, n_buckets + 1)
    edges = np.quantile(vals, qs)
    # Inner edges only; np.digitize with `right=False` returns indices in
    # [0, n_buckets] but values equal to the max edge land in bucket n_buckets.
    inner = edges[1:-1]
    idx = np.digitize(vals, inner, right=False)
    # Clip just in case of floating-point ties at the boundary.
    idx = np.clip(idx, 0, n_buckets - 1)
    out[valid] = idx
    return out


# ───────────────────────── conditional sign predictability ─────────────────────────

def _binomial_p_two_sided(hits: int, m: int) -> float:
    """Two-sided Gaussian-approximation p-value vs H0: hit_rate = 0.5.
    Matches the convention in alpha_translation.te_sign_predictability.
    """
    if m < 1:
        return 1.0
    z = (hits - 0.5 * m) / math.sqrt(0.25 * m)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def conditional_sign_predictability(src: np.ndarray, dst: np.ndarray,
                                    regime_bucket: np.ndarray) -> list[dict]:
    """For each regime bucket b, test whether sign(src[t]) predicts
    sign(dst[t+1]) over the bars where the regime at time t falls in
    bucket b.

    The bucket label at time t determines membership for the prediction
    whose outcome materializes at t+1 — the bucketing decision must be
    made at t (no look-ahead).

    Args:
        src:           1D source series, length T.
        dst:           1D destination series, length T.
        regime_bucket: int bucket labels at time t, length T; -1 = skip.

    Returns:
        list of dicts (one per bucket present in `regime_bucket`):
        {bucket, n, hit_rate, binomial_p, rule_fires}.
        Sorted by bucket index ascending.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    regime_bucket = np.asarray(regime_bucket, dtype=np.int64)
    T = min(src.size, dst.size, regime_bucket.size)
    if T < 2:
        return []
    # Predictor: sign(src[t]); outcome: sign(dst[t+1]); bucket: regime at t.
    sx_all = np.sign(src[:T - 1])
    sy_all = np.sign(dst[1:T])
    rb_all = regime_bucket[:T - 1]
    buckets = sorted(int(b) for b in np.unique(rb_all) if b >= 0)
    out: list[dict] = []
    for b in buckets:
        mask = (rb_all == b) & (sx_all != 0) & (sy_all != 0)
        sx = sx_all[mask]
        sy = sy_all[mask]
        m = int(sx.size)
        if m < 30:
            out.append({
                "bucket": int(b),
                "n": m,
                "hit_rate": 0.5,
                "binomial_p": 1.0,
                "rule_fires": False,
            })
            continue
        hits = int(np.sum(sx == sy))
        hit_rate = hits / m
        p = _binomial_p_two_sided(hits, m)
        out.append({
            "bucket": int(b),
            "n": m,
            "hit_rate": float(hit_rate),
            "binomial_p": float(p),
            "rule_fires": bool(hit_rate > 0.52 and p < 0.10),
        })
    return out


# ───────────────────────── screen candidate edges ─────────────────────────

def screen_conditional_edges(rets: np.ndarray, names: list[str],
                             candidate_edges: list[tuple[str, str]],
                             n_buckets: int = 5,
                             min_buckets_firing: int = 3) -> list[dict]:
    """Bucket the regime coordinate, then test each candidate edge with
    `conditional_sign_predictability`. Keep the edge iff it fires in at
    least `min_buckets_firing` of the `n_buckets` buckets.

    Args:
        rets:               (T, N) returns.
        names:              N column names.
        candidate_edges:    list of (src_name, dst_name).
        n_buckets:          quantile buckets for the regime coordinate.
        min_buckets_firing: keep-rule threshold M out of K.

    Returns:
        list of {src, dst, n_buckets_firing, mean_hit_rate, min_hit_rate}
        sorted by n_buckets_firing desc, mean_hit_rate desc.
    """
    rets = np.asarray(rets, dtype=np.float64)
    name_to_idx = {n: i for i, n in enumerate(names)}
    regime = regime_coordinate(rets)
    bucket = regime_buckets(regime, n_buckets=n_buckets)
    kept: list[dict] = []
    for (src_name, dst_name) in candidate_edges:
        if src_name not in name_to_idx or dst_name not in name_to_idx:
            continue
        i = name_to_idx[src_name]
        j = name_to_idx[dst_name]
        rows = conditional_sign_predictability(rets[:, i], rets[:, j], bucket)
        if not rows:
            continue
        n_fires = sum(1 for r in rows if r["rule_fires"])
        if n_fires < min_buckets_firing:
            continue
        hit_rates = [r["hit_rate"] for r in rows]
        kept.append({
            "src": src_name,
            "dst": dst_name,
            "n_buckets_firing": int(n_fires),
            "mean_hit_rate": float(np.mean(hit_rates)),
            "min_hit_rate": float(np.min(hit_rates)),
        })
    kept.sort(key=lambda r: (-r["n_buckets_firing"], -r["mean_hit_rate"]))
    return kept
