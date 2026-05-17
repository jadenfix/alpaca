"""Rolling econophysics features.

The point-in-time aggregates in `rmt_spectral`, `multifractal`,
`alpha_stable` and `hawkes` characterise the global statistical regime
of a returns matrix. Trading edges live in the DERIVATIVES of those
characterisations: when the top RMT eigenvalue jumps, when the
multifractal width compresses, when the Hawkes branching ratio drifts
toward criticality, when α-stable α drops out of the Gaussian basin —
those shifts are slow-moving, hard to fake and front-run sentiment.

This module slides the point-in-time estimators over a (T, N) returns
matrix and exposes a `detect_shifts` helper that flags z-score
excursions using ONLY prior observations (no look-ahead).

Pure numpy + stdlib.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from rmt_spectral import rmt_summary
from multifractal import mfdfa
from alpha_stable import fit_alpha_stable
from hawkes import fit_hawkes


# ───────────────────────── window enumeration ──────────────────────────


def _window_ends(t: int, window: int, step: int) -> list[int]:
    """Return the END indices (exclusive) of each rolling window.

    For T=500, window=252, step=21 the first valid end is 252 and we
    step by 21 until we cannot fit another full window; final entry
    equals T (the latest possible end).
    """
    if t < window or window <= 0 or step <= 0:
        return []
    ends = list(range(window, t + 1, step))
    if ends and ends[-1] != t:
        # Always include the most recent full window.
        ends.append(t)
    return ends


# ──────────────────────────── rolling RMT ──────────────────────────────


def rolling_rmt(rets: np.ndarray, window: int = 252, step: int = 21) -> dict:
    """Slide `rmt_summary` over a (T, N) returns matrix.

    Returns dict with parallel arrays of length len(window_ends):
        ts_idx              window-end indices (exclusive)
        n_above_mp          count of eigenvalues above the MP support
        tw_pvalue           Tracy-Widom p-value of the top eigenvalue
        top_eigenvalue      lambda_max
        top_eigenvalue_evr  explained variance ratio of lambda_max
        q                   aspect ratio N / window for that bin
    """
    rets = np.asarray(rets, dtype=np.float64)
    if rets.ndim != 2:
        raise ValueError("rets must be a 2-D (T, N) matrix")
    t, n = rets.shape
    ends = _window_ends(t, window, step)
    if not ends:
        return {k: np.array([]) for k in
                ("ts_idx", "n_above_mp", "tw_pvalue",
                 "top_eigenvalue", "top_eigenvalue_evr", "q")}
    ts_idx = np.array(ends, dtype=np.int64)
    n_above = np.full(len(ends), np.nan)
    tw_p = np.full(len(ends), np.nan)
    top_eig = np.full(len(ends), np.nan)
    top_evr = np.full(len(ends), np.nan)
    q_arr = np.full(len(ends), np.nan)
    for i, end in enumerate(ends):
        start = end - window
        window_data = rets[start:end, :]
        out = rmt_summary(window_data)
        if not out:
            continue
        n_above[i] = float(out["n_above_mp"])
        tw_p[i] = float(out["tw_pvalue"])
        eigs = out["eigenvalues"]
        evr = out["evr"]
        if eigs.size > 0:
            top_eig[i] = float(eigs[0])
            top_evr[i] = float(evr[0])
        q_arr[i] = float(out["q"])
    return {
        "ts_idx": ts_idx,
        "n_above_mp": n_above,
        "tw_pvalue": tw_p,
        "top_eigenvalue": top_eig,
        "top_eigenvalue_evr": top_evr,
        "q": q_arr,
    }


# ─────────────────────────── rolling MF-DFA ────────────────────────────


def rolling_mfdfa(series: np.ndarray, window: int = 252, step: int = 21) -> dict:
    """Slide MF-DFA over a 1-D series; record Δα per window."""
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    t = series.size
    ends = _window_ends(t, window, step)
    if not ends:
        return {"ts_idx": np.array([]), "delta_alpha": np.array([])}
    ts_idx = np.array(ends, dtype=np.int64)
    delta_alpha = np.full(len(ends), np.nan)
    for i, end in enumerate(ends):
        start = end - window
        seg = series[start:end]
        out = mfdfa(seg)
        if out and "width" in out:
            delta_alpha[i] = float(out["width"])
    return {"ts_idx": ts_idx, "delta_alpha": delta_alpha}


# ───────────────────────── rolling alpha-stable ────────────────────────


def rolling_alpha_stable(series: np.ndarray, window: int = 252,
                         step: int = 21) -> dict:
    """Slide the Koutrouvelis α-stable fit over a 1-D series."""
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    t = series.size
    ends = _window_ends(t, window, step)
    if not ends:
        return {"ts_idx": np.array([]), "alpha": np.array([]),
                "sigma": np.array([])}
    ts_idx = np.array(ends, dtype=np.int64)
    alpha = np.full(len(ends), np.nan)
    sigma = np.full(len(ends), np.nan)
    for i, end in enumerate(ends):
        start = end - window
        seg = series[start:end]
        out = fit_alpha_stable(seg)
        if out:
            alpha[i] = float(out.get("alpha", np.nan))
            sigma[i] = float(out.get("sigma", np.nan))
    return {"ts_idx": ts_idx, "alpha": alpha, "sigma": sigma}


# ─────────────────────── rolling Hawkes branching ──────────────────────


def rolling_hawkes_branching(series: np.ndarray, window: int = 252,
                             step: int = 21,
                             event_quantile: float = 0.90) -> dict:
    """Per-window Hawkes branching ratio.

    Treat |returns| above the within-window `event_quantile` as events
    (timestamps are integer offsets within the window). Fit a Hawkes-exp
    on those event times and record the branching ratio α/β. If fewer
    than 5 events fall in the window — too few for a stable MLE — emit
    NaN so the downstream z-score logic can drop them rather than
    treating a missing fit as a literal zero.
    """
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    t = series.size
    ends = _window_ends(t, window, step)
    if not ends:
        return {"ts_idx": np.array([]), "branching_ratio": np.array([])}
    ts_idx = np.array(ends, dtype=np.int64)
    br = np.full(len(ends), np.nan)
    for i, end in enumerate(ends):
        start = end - window
        seg = series[start:end]
        if seg.size == 0:
            continue
        abs_seg = np.abs(seg)
        if abs_seg.size == 0:
            continue
        thresh = float(np.quantile(abs_seg, event_quantile))
        if not math.isfinite(thresh) or thresh <= 0.0:
            continue
        events = np.where(abs_seg > thresh)[0].astype(np.float64)
        if events.size < 5:
            continue
        # Add a tiny jitter so equal times don't break MLE diffs.
        events = events + 1e-6 * np.arange(events.size)
        fit = fit_hawkes(events)
        if fit is None:
            continue
        br_val = fit.get("branching_ratio", np.nan)
        if br_val is None or not math.isfinite(br_val):
            continue
        br[i] = float(br_val)
    return {"ts_idx": ts_idx, "branching_ratio": br}


# ───────────────────────────── shift detector ──────────────────────────


def detect_shifts(series: np.ndarray, z_threshold: float = 2.0) -> list[dict]:
    """Flag indices where the 1-D rolling statistic departs >= `z_threshold`
    sigma from its trailing mean.

    The trailing mean / std are computed strictly over PRIOR observations
    (indices max(0, i-60) .. i, exclusive of i) — no look-ahead. NaNs are
    skipped both in the trailing window (so missing Hawkes fits don't
    poison the baseline) and in the test point itself.

    Returns a list of alerts {"idx", "value", "z"} ordered by index.
    """
    arr = np.asarray(series, dtype=np.float64).reshape(-1)
    alerts: list[dict] = []
    lookback = 60
    min_prior = 5  # need a few points to estimate mean/std meaningfully
    for i in range(arr.size):
        val = arr[i]
        if not math.isfinite(val):
            continue
        lo = max(0, i - lookback)
        prior = arr[lo:i]
        prior = prior[np.isfinite(prior)]
        if prior.size < min_prior:
            continue
        mu = float(prior.mean())
        sd = float(prior.std(ddof=1)) if prior.size > 1 else 0.0
        if sd <= 1e-12:
            continue
        z = (val - mu) / sd
        if abs(z) >= z_threshold:
            alerts.append({"idx": int(i), "value": float(val), "z": float(z)})
    return alerts


# ─────────────────────────── orchestrator ──────────────────────────────


def _shift_alerts_from(values: np.ndarray, ts_idx: np.ndarray,
                       feature: str, z_threshold: float) -> list[dict]:
    """Run `detect_shifts` on `values` and tag each alert with `feature`
    and the original `ts_idx` location."""
    out: list[dict] = []
    if values.size == 0:
        return out
    raw = detect_shifts(values, z_threshold=z_threshold)
    for a in raw:
        bin_i = a["idx"]
        ts = int(ts_idx[bin_i]) if 0 <= bin_i < ts_idx.size else bin_i
        out.append({
            "feature": feature,
            "bin_idx": bin_i,
            "ts_idx": ts,
            "value": a["value"],
            "z": a["z"],
        })
    return out


def rolling_econ_report(rets: np.ndarray, names: list[str],
                        window: int = 252, step: int = 21,
                        z_threshold: float = 2.0) -> dict:
    """End-to-end rolling pipeline suitable for the orchestrator.

    Args:
        rets:       (T, N) returns matrix.
        names:      length-N list of series names.
        window:     rolling window size in observations.
        step:       window stride.
        z_threshold: alert threshold passed to `detect_shifts`.

    Returns dict:
        rmt                      → output of `rolling_rmt`.
        mfdfa_per_series         → {name: rolling_mfdfa(series)}
        alpha_stable_per_series  → {name: rolling_alpha_stable(series)}
        hawkes_per_series        → {name: rolling_hawkes_branching(series)}
        shifts                   → list of alerts, each tagged with a
                                   `feature` string identifying the
                                   source statistic and series.
    """
    rets = np.asarray(rets, dtype=np.float64)
    if rets.ndim != 2:
        raise ValueError("rets must be (T, N)")
    t, n = rets.shape
    if len(names) != n:
        raise ValueError(f"names has {len(names)} entries, expected {n}")

    rmt_out = rolling_rmt(rets, window=window, step=step)

    mfdfa_per: dict[str, dict] = {}
    alpha_per: dict[str, dict] = {}
    hawkes_per: dict[str, dict] = {}
    shifts: list[dict] = []

    # Shift detection on RMT scalars (one curve per statistic).
    for stat in ("n_above_mp", "top_eigenvalue", "top_eigenvalue_evr",
                 "tw_pvalue"):
        vals = rmt_out.get(stat, np.array([]))
        shifts.extend(_shift_alerts_from(
            vals, rmt_out.get("ts_idx", np.array([])),
            feature=f"rmt.{stat}", z_threshold=z_threshold,
        ))

    # Per-series scalar curves.
    for j, name in enumerate(names):
        s = rets[:, j]
        mf = rolling_mfdfa(s, window=window, step=step)
        al = rolling_alpha_stable(s, window=window, step=step)
        hw = rolling_hawkes_branching(s, window=window, step=step)
        mfdfa_per[name] = mf
        alpha_per[name] = al
        hawkes_per[name] = hw

        shifts.extend(_shift_alerts_from(
            mf.get("delta_alpha", np.array([])),
            mf.get("ts_idx", np.array([])),
            feature=f"mfdfa.delta_alpha[{name}]",
            z_threshold=z_threshold,
        ))
        shifts.extend(_shift_alerts_from(
            al.get("alpha", np.array([])),
            al.get("ts_idx", np.array([])),
            feature=f"alpha_stable.alpha[{name}]",
            z_threshold=z_threshold,
        ))
        shifts.extend(_shift_alerts_from(
            al.get("sigma", np.array([])),
            al.get("ts_idx", np.array([])),
            feature=f"alpha_stable.sigma[{name}]",
            z_threshold=z_threshold,
        ))
        shifts.extend(_shift_alerts_from(
            hw.get("branching_ratio", np.array([])),
            hw.get("ts_idx", np.array([])),
            feature=f"hawkes.branching_ratio[{name}]",
            z_threshold=z_threshold,
        ))

    return {
        "rmt": rmt_out,
        "mfdfa_per_series": mfdfa_per,
        "alpha_stable_per_series": alpha_per,
        "hawkes_per_series": hawkes_per,
        "shifts": shifts,
    }


__all__ = [
    "rolling_rmt",
    "rolling_mfdfa",
    "rolling_alpha_stable",
    "rolling_hawkes_branching",
    "detect_shifts",
    "rolling_econ_report",
]
