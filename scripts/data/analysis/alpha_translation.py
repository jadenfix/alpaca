"""Turn econophysics findings into concrete, measurable trading signals.

Every statistic in `report.py` is potentially actionable; this module
makes the translation explicit and verifiable:

  • Transfer-entropy edges  → sign-predictability hit rate (binomial p)
  • RIE-shrunk covariance   → mean-variance portfolio variance reduction
  • Tail dependence λ_L     → "do NOT pair-trade" red-list
  • α-stable fits           → Gaussian-VaR underestimation factor
  • MST                     → cross-sectional-momentum cluster scoping
  • Top eigenvector         → market-mode hedge weights
  • MF-DFA Δα               → cascade-risk ranking
  • Hawkes branching ratio  → criticality / de-risk regime gauge

Each routine returns a dict with the numbers a trader would actually act
on, plus a one-line `rule` string suitable for the markdown report.

All math is pure numpy. No look-ahead: every signal uses only data
strictly prior to the prediction it is evaluated on.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


# ───────────────────────── transfer-entropy → directional signal ─────────────────────────

def te_sign_predictability(src: np.ndarray, dst: np.ndarray) -> dict:
    """Test whether sign(src[t]) predicts sign(dst[t+1]).

    Returns {hit_rate, n, binomial_p, rule_fires}. `binomial_p` is the
    two-sided probability under H0: hit_rate = 0.5. `rule_fires` is True
    iff hit_rate > 0.52 and binomial_p < 0.10 — the alpha threshold.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = min(src.size - 1, dst.size - 1)
    if n < 30:
        return {"hit_rate": 0.5, "n": n, "binomial_p": 1.0, "rule_fires": False}
    sx = np.sign(src[:n])
    sy = np.sign(dst[1:n + 1])
    valid = (sx != 0) & (sy != 0)
    sx, sy = sx[valid], sy[valid]
    m = sx.size
    if m < 30:
        return {"hit_rate": 0.5, "n": m, "binomial_p": 1.0, "rule_fires": False}
    hits = int(np.sum(sx == sy))
    hit_rate = hits / m
    # Two-sided binomial vs 0.5 via Gaussian approximation (valid for m ≥ 30).
    z = (hits - 0.5 * m) / math.sqrt(0.25 * m)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {
        "hit_rate": float(hit_rate),
        "n": int(m),
        "binomial_p": float(p),
        "rule_fires": bool(hit_rate > 0.52 and p < 0.10),
    }


def screen_te_edges(rets: np.ndarray, names: list[str],
                    edges: list[tuple[str, str, float]],
                    top_k: int = 10) -> list[dict]:
    """Run `te_sign_predictability` on the top-`top_k` (src, dst, te) edges
    and return the rows sorted by hit-rate descending. Each edge that
    passes the alpha threshold is a candidate signal.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    rows: list[dict] = []
    for (src, dst, te) in edges[:top_k]:
        if src not in name_to_idx or dst not in name_to_idx:
            continue
        i = name_to_idx[src]
        j = name_to_idx[dst]
        pred = te_sign_predictability(rets[:, i], rets[:, j])
        rows.append({
            "src": src, "dst": dst, "TE": float(te),
            **pred,
        })
    rows.sort(key=lambda r: r["hit_rate"], reverse=True)
    return rows


# ───────────────────────── RIE covariance → portfolio variance reduction ─────────────────────────

def _min_var_weights(cov: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Global minimum-variance weights w ∝ Σ⁻¹·1, normalized to ‖w‖₁ = 1
    so that the unsigned exposure is comparable across estimators.
    """
    n = cov.shape[0]
    reg = cov + ridge * np.eye(n)
    try:
        inv_ones = np.linalg.solve(reg, np.ones(n))
    except np.linalg.LinAlgError:
        return np.full(n, 1.0 / n)
    w = inv_ones / np.sum(np.abs(inv_ones))
    return w


def rie_portfolio_lift(rets: np.ndarray, rmt_result: dict) -> dict:
    """Compute in-sample volatility of the min-var portfolio under sample
    covariance vs RIE-shrunk covariance. Lower realized vol with the SAME
    eigenvectors means the cleaned spectrum sized risk better.

    Returns {sample_vol, rie_vol, vol_reduction_pct, rule}.
    """
    if not rmt_result or rmt_result.get("eigenvalues") is None:
        return {"sample_vol": 0.0, "rie_vol": 0.0,
                "vol_reduction_pct": 0.0, "rule_fires": False}
    # Standardize returns the same way rmt_summary did, so eigenvectors align.
    means = rets.mean(axis=0, keepdims=True)
    stds = rets.std(axis=0, keepdims=True, ddof=1)
    stds = np.where(stds > 0, stds, 1.0)
    z = (rets - means) / stds
    sample_cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    eigvals_sample, eigvecs = np.linalg.eigh(sample_cov)
    # Match descending order of rmt_summary's eigenvalues.
    order = np.argsort(eigvals_sample)[::-1]
    eigvals_sample = eigvals_sample[order]
    eigvecs = eigvecs[:, order]
    rie_eigs = np.asarray(rmt_result["rie_eigenvalues"], dtype=np.float64)
    if rie_eigs.size != eigvals_sample.size:
        return {"sample_vol": 0.0, "rie_vol": 0.0,
                "vol_reduction_pct": 0.0, "rule_fires": False}
    rie_cov = eigvecs @ np.diag(rie_eigs) @ eigvecs.T

    w_s = _min_var_weights(sample_cov)
    w_r = _min_var_weights(rie_cov)
    port_s = z @ w_s
    port_r = z @ w_r
    vol_s = float(np.std(port_s, ddof=1))
    vol_r = float(np.std(port_r, ddof=1))
    if vol_s <= 0:
        return {"sample_vol": vol_s, "rie_vol": vol_r,
                "vol_reduction_pct": 0.0, "rule_fires": False}
    reduction = 100.0 * (vol_s - vol_r) / vol_s
    return {
        "sample_vol": vol_s,
        "rie_vol": vol_r,
        "vol_reduction_pct": float(reduction),
        "rule_fires": bool(reduction > 0.5),  # measurable risk improvement
    }


# ───────────────────────── tail dependence → pair-trade red-list ─────────────────────────

def panic_pairs(tail_rows: Iterable[tuple[str, str, float, float]],
                threshold: float = 0.5) -> list[dict]:
    """Pairs with λ_L ≥ `threshold` crash together — disqualified as
    diversifying spreads. The trader should AVOID treating them as
    "long A, short B" hedges.
    """
    out: list[dict] = []
    for (a, b, lu, ll) in tail_rows:
        if ll >= threshold or lu >= threshold:
            out.append({"a": a, "b": b, "lambda_U": float(lu),
                        "lambda_L": float(ll),
                        "rule": "do not pair-trade as diversifier"})
    out.sort(key=lambda r: max(r["lambda_U"], r["lambda_L"]), reverse=True)
    return out


# ───────────────────────── α-stable → Gaussian-VaR underestimation ─────────────────────────

def _alpha_stable_quantile_mc(alpha: float, sigma: float, mu: float,
                              q: float = 0.99, n_sim: int = 50_000,
                              seed: int = 0) -> float:
    """Monte-Carlo quantile of a symmetric (β = 0) α-stable S(α, 0, σ, μ)
    via the Chambers-Mallows-Stuck (1976) generator.
    """
    rng = np.random.default_rng(seed)
    if alpha <= 0 or alpha > 2:
        return mu
    if abs(alpha - 1.0) < 1e-6:
        # Cauchy special case.
        w = rng.exponential(1.0, n_sim)
        u = rng.uniform(-math.pi / 2, math.pi / 2, n_sim)
        x = math.tan(0) * 0 + np.tan(u)  # β=0
        samples = sigma * x + mu
    else:
        u = rng.uniform(-math.pi / 2, math.pi / 2, n_sim)
        w = rng.exponential(1.0, n_sim)
        s = np.sin(alpha * u) / np.power(np.cos(u), 1.0 / alpha)
        t = np.power(np.cos(u - alpha * u) / w, (1.0 - alpha) / alpha)
        samples = sigma * s * t + mu
    return float(np.quantile(samples, 1.0 - q))  # left-tail q-VaR


def gaussian_var_underestimation(alpha_rows: list[tuple[str, float, float, float]],
                                 rets: np.ndarray, names: list[str],
                                 q: float = 0.99) -> list[dict]:
    """For each fitted series, compare 99 % left-tail VaR under fitted
    α-stable vs. Gaussian fitted with the same sample std. Ratio > 1
    quantifies how much Gaussian-VaR understates true tail risk.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    out: list[dict] = []
    z99 = 2.326347874040841  # Φ⁻¹(0.99)
    for (name, a, sig, ks) in alpha_rows:
        if name not in name_to_idx:
            continue
        j = name_to_idx[name]
        sample = rets[:, j]
        mu_s = float(np.mean(sample))
        sigma_g = float(np.std(sample, ddof=1))
        var_gauss = mu_s - z99 * sigma_g
        var_stable = _alpha_stable_quantile_mc(a, sig, mu_s, q=q,
                                               seed=hash(name) & 0xFFFF_FFFF)
        # Both are negative numbers; "more negative" = larger loss.
        underest = abs(var_stable) / max(abs(var_gauss), 1e-9)
        out.append({
            "series": name,
            "alpha": float(a),
            "VaR_gaussian_99": float(var_gauss),
            "VaR_stable_99": float(var_stable),
            "underestimation_factor": float(underest),
            "rule_fires": bool(underest > 1.10),
        })
    out.sort(key=lambda r: r["underestimation_factor"], reverse=True)
    return out


# ───────────────────────── MST → cross-sectional-momentum clusters ─────────────────────────

def mst_clusters(mst_edges: list[tuple[int, int, float]], n: int,
                 names: list[str], distance_cutoff: float = 1.0) -> list[list[str]]:
    """Drop MST edges with distance ≥ `distance_cutoff` and return the
    connected components. Within each component, cross-sectional
    momentum is interpretable; across components the series are too
    weakly related for relative-value to be meaningful.
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, j, d in mst_edges:
        if d < distance_cutoff:
            union(i, j)
    groups: dict[int, list[str]] = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(names[k])
    return [g for g in groups.values() if len(g) >= 2]


# ───────────────────────── top eigenvector → market-mode hedge ─────────────────────────

def market_mode_hedge(rmt_result: dict, names: list[str]) -> dict:
    """Loadings of the top eigenvector are the market-mode exposures.
    Sized to ‖w‖₁ = 1, they form a long-only market portfolio whose
    short side is the hedge for residual alpha strategies.
    """
    if not rmt_result or "eigenvalues" not in rmt_result:
        return {}
    # rmt_summary returned eigenvalues but not eigvecs; recompute on the
    # same standardized matrix to recover loadings.
    eigs = rmt_result["eigenvalues"]
    if eigs.size == 0:
        return {}
    # Caller supplies the matching cov via _recompute_top_eigvec helper
    return {"eigenvalues_top5": [float(v) for v in eigs[:5]],
            "evr_top5": [float(v) for v in rmt_result["evr"][:5]]}


def top_eigvec_loadings(rets: np.ndarray, names: list[str]) -> list[tuple[str, float]]:
    """Return (name, loading) for eigenvector 1, sorted by |loading|."""
    means = rets.mean(axis=0, keepdims=True)
    stds = rets.std(axis=0, keepdims=True, ddof=1)
    stds = np.where(stds > 0, stds, 1.0)
    z = (rets - means) / stds
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    top = vecs[:, order[0]]
    # Convention: largest absolute loading is positive.
    if top[np.argmax(np.abs(top))] < 0:
        top = -top
    out = [(names[i], float(top[i])) for i in range(top.size)]
    out.sort(key=lambda r: abs(r[1]), reverse=True)
    return out


# ───────────────────────── MF-DFA Δα → cascade-risk rank ─────────────────────────

def cascade_risk_rank(mfdfa_rows: list[tuple[str, float]],
                      threshold: float = 0.5) -> list[dict]:
    """Series with Δα > `threshold` carry multifractal cascade risk —
    intermittency, vol-of-vol bursts. Strategies on these names need
    smaller position sizes than a Gaussian risk model would suggest.
    """
    return [{"series": s, "delta_alpha": float(w),
             "rule_fires": bool(w > threshold)}
            for s, w in mfdfa_rows]


# ───────────────────────── Hawkes → criticality gauge ─────────────────────────

def criticality_from_branching(branching_ratios: dict[str, float]) -> dict:
    """If the average branching ratio across event series exceeds 0.85
    the market is near-critical: a single shock excites a runaway
    sequence of follow-ons. The rule "de-risk to 50 %" is the standard
    response.
    """
    if not branching_ratios:
        return {"mean_n": 0.0, "max_n": 0.0, "rule_fires": False, "size_multiplier": 1.0}
    vals = [v for v in branching_ratios.values() if v is not None]
    if not vals:
        return {"mean_n": 0.0, "max_n": 0.0, "rule_fires": False, "size_multiplier": 1.0}
    mean_n = float(np.mean(vals))
    max_n = float(np.max(vals))
    if max_n >= 0.95:
        mult = 0.25
    elif max_n >= 0.9:
        mult = 0.5
    elif max_n >= 0.85:
        mult = 0.75
    else:
        mult = 1.0
    return {
        "mean_n": mean_n,
        "max_n": max_n,
        "rule_fires": bool(max_n >= 0.85),
        "size_multiplier": float(mult),
    }


# ───────────────────────── markdown emitter ─────────────────────────

def to_markdown(te_signals: list[dict],
                rie_lift: dict,
                panic: list[dict],
                var_table: list[dict],
                clusters: list[list[str]],
                eigvec1: list[tuple[str, float]],
                cascade: list[dict],
                criticality: dict) -> str:
    """Render the full alpha-translation section."""
    lines: list[str] = []
    lines.append("## 12. Alpha Translation — actionable signals\n\n")
    lines.append("Every numeric finding above is converted to a measurable trade rule.\n")
    lines.append("A rule **fires** only if the test passes its specific threshold.\n\n")

    # Lead-lag
    lines.append("### 12.1 Lead-lag directional signals (top TE edges)\n\n")
    lines.append("Test: does sign(src[t]) predict sign(dst[t+1])? "
                 "Fires if hit-rate > 0.52 AND binomial-p < 0.10.\n\n")
    lines.append("| Src → Dst | TE | n | hit-rate | binomial p | fires? |\n")
    lines.append("|---|---:|---:|---:|---:|:---:|\n")
    for r in te_signals[:10]:
        fire = "**Y**" if r["rule_fires"] else "n"
        lines.append(f"| {r['src']} → {r['dst']} | {r['TE']:.4f} | {r['n']} | "
                     f"{r['hit_rate']:.3f} | {r['binomial_p']:.3f} | {fire} |\n")
    fires = sum(1 for r in te_signals if r["rule_fires"])
    lines.append(f"\n**{fires} / {len(te_signals)} edges produce a tradeable directional signal.**\n\n")

    # RIE lift
    lines.append("### 12.2 RIE-shrunk covariance vs sample (min-var portfolio)\n\n")
    if rie_lift.get("sample_vol", 0) > 0:
        lines.append(f"In-sample min-var portfolio realized vol:\n\n")
        lines.append(f"- Sample covariance: **{rie_lift['sample_vol']:.4f}**\n")
        lines.append(f"- RIE-shrunk covariance: **{rie_lift['rie_vol']:.4f}**\n")
        lines.append(f"- Volatility reduction: **{rie_lift['vol_reduction_pct']:+.2f} %**\n")
        if rie_lift["rule_fires"]:
            lines.append(f"\n**Rule fires**: use RIE-shrunk Σ in `crates/portfolio` for sizing.\n\n")
        else:
            lines.append(f"\nReduction below threshold; sample covariance is acceptable here.\n\n")
    else:
        lines.append("Insufficient data.\n\n")

    # Panic basket
    lines.append("### 12.3 Crash co-movement red-list (tail dependence)\n\n")
    if panic:
        lines.append("Pairs with λ_L ≥ 0.5 or λ_U ≥ 0.5 — do NOT treat as diversifying hedges:\n\n")
        lines.append("| Pair | λ_U | λ_L |\n|---|---:|---:|\n")
        for r in panic[:10]:
            lines.append(f"| {r['a']} ↔ {r['b']} | {r['lambda_U']:.3f} | {r['lambda_L']:.3f} |\n")
        lines.append("\n")
    else:
        lines.append("No pairs cross the panic threshold.\n\n")

    # VaR underestimation
    lines.append("### 12.4 Gaussian-VaR underestimation under α-stable\n\n")
    lines.append("Ratio = |VaR_stable| / |VaR_gaussian| at 99 %. > 1.10 → Gaussian sizing is unsafe.\n\n")
    lines.append("| Series | α | VaR_Gauss_99 | VaR_Stable_99 | ratio | fires? |\n")
    lines.append("|---|---:|---:|---:|---:|:---:|\n")
    for r in var_table[:10]:
        fire = "**Y**" if r["rule_fires"] else "n"
        lines.append(f"| {r['series']} | {r['alpha']:.2f} | {r['VaR_gaussian_99']:+.4f} | "
                     f"{r['VaR_stable_99']:+.4f} | {r['underestimation_factor']:.2f}× | {fire} |\n")
    lines.append("\n")

    # Clusters
    lines.append("### 12.5 Cross-sectional momentum clusters (MST)\n\n")
    if clusters:
        lines.append("Cross-sectional momentum is scoped WITHIN these clusters (not across):\n\n")
        for i, g in enumerate(clusters, 1):
            lines.append(f"- **Cluster {i}**: {', '.join(g)}\n")
        lines.append("\n")
    else:
        lines.append("No multi-member clusters at the chosen distance cutoff.\n\n")

    # Market-mode hedge
    lines.append("### 12.6 Market-mode hedge (top eigenvector loadings)\n\n")
    lines.append("Long-only market portfolio used to neutralize residual-alpha strategies:\n\n")
    lines.append("| Series | loading |\n|---|---:|\n")
    for name, w in eigvec1[:10]:
        lines.append(f"| {name} | {w:+.3f} |\n")
    lines.append("\n")

    # Cascade risk
    lines.append("### 12.7 Cascade-risk ranking (MF-DFA Δα)\n\n")
    if cascade:
        lines.append("Series with Δα > 0.5 need reduced gross exposure vs Gaussian baseline:\n\n")
        lines.append("| Series | Δα | fires? |\n|---|---:|:---:|\n")
        for r in cascade[:10]:
            fire = "**Y**" if r["rule_fires"] else "n"
            lines.append(f"| {r['series']} | {r['delta_alpha']:.3f} | {fire} |\n")
        lines.append("\n")

    # Criticality
    lines.append("### 12.8 Hawkes-process criticality gauge\n\n")
    if criticality.get("mean_n", 0) > 0 or criticality.get("max_n", 0) > 0:
        lines.append(f"Mean branching ratio across series: **{criticality['mean_n']:.3f}**, "
                     f"max: **{criticality['max_n']:.3f}**.\n")
        if criticality["rule_fires"]:
            lines.append(f"\n**Rule fires**: near-critical regime → size multiplier "
                         f"**{criticality['size_multiplier']:.2f}×** across all strategies.\n\n")
        else:
            lines.append("\nBranching ratios below 0.85: normal regime, full sizing.\n\n")
    else:
        lines.append("No event sequences available for fit.\n\n")

    return "".join(lines)
