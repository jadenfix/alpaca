"""Multi-horizon Transfer Entropy and Convergent Cross-Mapping.

Extends the (h=1) tests in ``info_theory.transfer_entropy`` and
``alpha_translation.te_sign_predictability`` to a grid of forecast
horizons ``h in {1, 5, 21}`` so an edge that fires at h=1 can be
distinguished from one that only fires at a longer horizon (slow
diffusion of information) and from a spurious lag-1 artefact.

The well-behaved case is a *monotone-decaying* hit-rate profile: the
signal is strongest at h=1 and weakens with horizon. A flat or
non-monotone profile is a red flag (over-fit, regime artifact, look-
ahead leak).

Pure numpy + stdlib. The TE estimator (binned plug-in) is reused from
``info_theory``; the CCM simplex-projection estimator is reused from
``causality``. This module only orchestrates them across grids and
returns row-oriented dicts suitable for the report layer.
"""
from __future__ import annotations

import math

import numpy as np

from info_theory import transfer_entropy
from causality import ccm_skill


# ───────────────────────── helpers ─────────────────────────

def _sign_hit_rate(src: np.ndarray, dst: np.ndarray, h: int) -> tuple[float, int, float]:
    """Hit rate of sign(src[t]) -> sign(dst[t+h]).

    Mirrors ``alpha_translation.te_sign_predictability``:
      - drop zero signs,
      - two-sided binomial p-value via Gaussian approximation
        z = (hits - 0.5 m) / sqrt(0.25 m), p = erfc(|z|/sqrt 2).
    Returns (hit_rate, n_effective, binomial_p). When ``n < 30`` returns
    (0.5, n, 1.0) — same convention as the reference.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n_total = min(src.size, dst.size)
    if h < 1 or n_total <= h + 1:
        return 0.5, 0, 1.0
    sx = np.sign(src[: n_total - h])
    sy = np.sign(dst[h: n_total])
    valid = (sx != 0) & (sy != 0)
    sx, sy = sx[valid], sy[valid]
    m = int(sx.size)
    if m < 30:
        return 0.5, m, 1.0
    hits = int(np.sum(sx == sy))
    hit_rate = hits / m
    z = (hits - 0.5 * m) / math.sqrt(0.25 * m)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return float(hit_rate), m, float(p)


def _shift_align(src: np.ndarray, dst: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Align ``src[:-h]`` with ``dst[h:]`` so that the TE estimator sees
    src "leading" dst by exactly ``h`` steps. Without this, calling
    ``transfer_entropy`` on the raw pair always measures lag-1 TE
    regardless of the intended horizon.
    """
    if h < 1:
        return src, dst
    return src[: src.size - h], dst[h:]


# ───────────────────────── multi-horizon TE ─────────────────────────

def multi_horizon_te(src: np.ndarray, dst: np.ndarray,
                     k_grid: list[int] | None = None,
                     l_grid: list[int] | None = None,
                     h_grid: list[int] | None = None,
                     bins: int = 4) -> list[dict]:
    """For each (k, l, h) triple compute TE(src -> dst shifted by h)
    AND the sign-predictability hit rate of sign(src[t]) -> sign(dst[t+h]).

    Returns rows: ``{k, l, h, te, n, hit_rate, binomial_p}``.

    Notes
    -----
    The shifted-by-h pair is built by aligning ``src[:-h]`` with
    ``dst[h:]`` BEFORE invoking ``transfer_entropy`` — otherwise the
    estimator's internal lag-1 conditioning means the answer would be
    the same for every ``h``.
    """
    if k_grid is None:
        k_grid = [1, 2, 3]
    if l_grid is None:
        l_grid = [1, 2, 3]
    if h_grid is None:
        h_grid = [1, 5, 21]
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n_total = min(src.size, dst.size)
    src = src[:n_total]
    dst = dst[:n_total]

    rows: list[dict] = []
    for h in h_grid:
        if h < 1 or h >= n_total - 5:
            for k in k_grid:
                for l in l_grid:
                    rows.append({"k": int(k), "l": int(l), "h": int(h),
                                 "te": 0.0, "n": 0,
                                 "hit_rate": 0.5, "binomial_p": 1.0})
            continue
        # Hit-rate depends only on h, so compute once per h.
        hit_rate, n_eff, p = _sign_hit_rate(src, dst, h)
        src_a, dst_a = _shift_align(src, dst, h)
        for k in k_grid:
            for l in l_grid:
                # Guard against pathological history vs sample size.
                hist = max(int(k), int(l))
                if src_a.size < hist + 3:
                    te = 0.0
                else:
                    te = float(transfer_entropy(src_a, dst_a,
                                                k=int(k), l=int(l),
                                                bins=int(bins)))
                rows.append({"k": int(k), "l": int(l), "h": int(h),
                             "te": te, "n": int(n_eff),
                             "hit_rate": float(hit_rate),
                             "binomial_p": float(p)})
    return rows


# ───────────────────────── multi-horizon CCM ─────────────────────────

def _default_library_sizes(n: int) -> list[int]:
    """Geometric grid from ~20 to ~n/2, five points, deduped."""
    if n < 40:
        return []
    hi = max(20, n // 2)
    lo = 20
    if hi <= lo:
        return [lo]
    raw = np.geomspace(lo, hi, num=5)
    out = sorted({int(round(s)) for s in raw if s >= 20})
    return [s for s in out if s <= hi]


def multi_horizon_ccm(src: np.ndarray, dst: np.ndarray,
                      e_grid: list[int] | None = None,
                      tau: int = 1,
                      library_sizes: list[int] | None = None) -> list[dict]:
    """For each embedding dim E, compute CCM skill from ``dst``'s manifold
    to ``src`` (rho_xy) and from ``src``'s manifold to ``dst`` (rho_yx)
    across library sizes.

    Returns rows: ``{E, tau, library_size, rho_xy, rho_yx,
    convergence_score}``. ``convergence_score`` is the Pearson
    correlation between library_size and rho_xy across the supplied
    library sizes for that E — a positive value indicates the monotone
    convergence Sugihara (2012) treats as evidence of causality from
    src to dst.
    """
    if e_grid is None:
        e_grid = [2, 3, 4]
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n_total = min(src.size, dst.size)
    src = src[:n_total]
    dst = dst[:n_total]

    if library_sizes is None:
        library_sizes = _default_library_sizes(n_total)
    if not library_sizes:
        return []

    rows: list[dict] = []
    for e in e_grid:
        e_int = int(e)
        # Skill estimators for both directions on the SAME library grid.
        # ccm_skill(x, y) predicts x from y's Takens manifold — high rho
        # means y's manifold contains information about x, which under
        # Sugihara is evidence that x causally drives y. So:
        #   rho_xy(src->dst) := ccm_skill(src, dst, ...).rho_curve
        #   rho_yx(dst->src) := ccm_skill(dst, src, ...).rho_curve
        xy = ccm_skill(src, dst, e=e_int, tau=int(tau),
                       library_sizes=list(library_sizes))
        yx = ccm_skill(dst, src, e=e_int, tau=int(tau),
                       library_sizes=list(library_sizes))
        xy_map = {int(L): float(r) for L, r in xy.get("rho_curve", [])}
        yx_map = {int(L): float(r) for L, r in yx.get("rho_curve", [])}

        # Convergence score: corr(library_size, rho_xy) across the grid.
        Ls_eff = [L for L in library_sizes if L in xy_map]
        if len(Ls_eff) >= 2:
            xs = np.asarray(Ls_eff, dtype=np.float64)
            ys = np.asarray([xy_map[L] for L in Ls_eff], dtype=np.float64)
            if xs.std() > 0 and ys.std() > 0:
                conv = float(np.corrcoef(xs, ys)[0, 1])
            else:
                conv = 0.0
        else:
            conv = 0.0

        for L in library_sizes:
            rows.append({
                "E": e_int,
                "tau": int(tau),
                "library_size": int(L),
                "rho_xy": float(xy_map.get(int(L), 0.0)),
                "rho_yx": float(yx_map.get(int(L), 0.0)),
                "convergence_score": float(conv),
            })
    return rows


# ───────────────────────── edge screener ─────────────────────────

def _hit_rates_by_horizon(rows: list[dict], h_grid: list[int]) -> dict[int, float]:
    """For a multi_horizon_te output that ranges over (k, l, h), return
    the (h -> max hit_rate over k,l) map. Hit-rate is constant across
    (k, l) for fixed h by construction, but using max is safe.
    """
    out: dict[int, float] = {}
    for r in rows:
        h = int(r["h"])
        hr = float(r["hit_rate"])
        if h not in out or hr > out[h]:
            out[h] = hr
    return {h: out.get(int(h), 0.5) for h in h_grid}


def _is_monotone_decay(hits_by_h: dict[int, float], h_grid: list[int]) -> bool:
    """True iff hit_rate(h1) >= hit_rate(h2) >= hit_rate(h3) >= ... across
    the supplied (assumed-ascending) h_grid. Requires at least two
    horizons; trivially True for a single horizon (no decay required).
    """
    if len(h_grid) < 2:
        return True
    prev = hits_by_h[int(h_grid[0])]
    for h in h_grid[1:]:
        cur = hits_by_h[int(h)]
        if cur > prev:
            return False
        prev = cur
    return True


def screen_edges(rets: np.ndarray, names: list[str],
                 top_pairs: list[tuple[str, str]],
                 k_grid: list[int] | None = None,
                 l_grid: list[int] | None = None,
                 h_grid: list[int] | None = None) -> list[dict]:
    """Run ``multi_horizon_te`` over the supplied (src, dst) pairs and
    collect the best ``(k, l, h)`` combination per pair.

    Selection prefers:
      1. high hit_rate,
      2. low binomial_p,
      3. consistency across at least two horizons (monotone decay of
         hit_rate as h increases — the well-behaved case).

    Returns rows: ``{src, dst, best_k, best_l, best_h, hit_rate, te,
    monotone}``.
    """
    if k_grid is None:
        k_grid = [1, 2, 3]
    if l_grid is None:
        l_grid = [1, 2, 3]
    if h_grid is None:
        h_grid = [1, 5, 21]
    name_to_idx = {n: i for i, n in enumerate(names)}
    out: list[dict] = []
    for (src_name, dst_name) in top_pairs:
        if src_name not in name_to_idx or dst_name not in name_to_idx:
            continue
        i = name_to_idx[src_name]
        j = name_to_idx[dst_name]
        rows = multi_horizon_te(rets[:, i], rets[:, j],
                                k_grid=list(k_grid),
                                l_grid=list(l_grid),
                                h_grid=list(h_grid))
        if not rows:
            continue
        hits_by_h = _hit_rates_by_horizon(rows, list(h_grid))
        monotone = _is_monotone_decay(hits_by_h, list(h_grid))

        # Composite score: prefer high hit_rate, low p. The -log p term
        # is a soft tiebreaker that does not invert the hit_rate order.
        def score(r: dict) -> float:
            return float(r["hit_rate"]) - 0.05 * math.log10(
                max(float(r["binomial_p"]), 1e-12) + 1e-12)

        best = max(rows, key=score)
        out.append({
            "src": src_name,
            "dst": dst_name,
            "best_k": int(best["k"]),
            "best_l": int(best["l"]),
            "best_h": int(best["h"]),
            "hit_rate": float(best["hit_rate"]),
            "te": float(best["te"]),
            "monotone": bool(monotone),
        })
    return out
