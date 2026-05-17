"""End-to-end econophysics analysis orchestrator.

Loads multiple CSVs, aligns them on common timestamps, computes the full
13-module stack, and writes a markdown findings report + per-method
CSV/JSON outputs.

Usage:
    python3 scripts/data/analysis/report.py \\
        --inputs data/macro/VIXCLS.csv data/real/SPY.csv \\
        --out data/analysis/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from loader import align, returns  # noqa: E402
import correlation_matrix as cm  # noqa: E402
import rmt_spectral as rmt  # noqa: E402
import info_theory as it  # noqa: E402
import dependence as dep  # noqa: E402
import causality as cau  # noqa: E402
import copula as cop  # noqa: E402
import multifractal as mf  # noqa: E402
import diffusion_map as dm  # noqa: E402
import network as nw  # noqa: E402
import alpha_stable as al  # noqa: E402
import hawkes as hw  # noqa: E402
import wavelet as wv  # noqa: E402
import alpha_translation as at  # noqa: E402


def _write_matrix_csv(path: Path, mat: np.ndarray, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, row in enumerate(mat):
            w.writerow([names[i]] + [f"{v:.6f}" for v in row])


def _topk_pairs(mat: np.ndarray, names: list[str], k: int = 10,
                exclude_diag: bool = True) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    n = mat.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            v = float(mat[i, j])
            pairs.append((names[i], names[j], v))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:k]


def run(inputs: list[Path], out_dir: Path, max_gap_bars: int = 0) -> Path:
    """Run the full analysis. Returns path to the timestamped output dir."""
    ts_stamp = int(time.time())
    run_dir = out_dir / str(ts_stamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analysis] loading {len(inputs)} input files…")
    data, names, ts = align(inputs, max_gap_bars=max_gap_bars)
    if data.shape[0] < 30:
        raise RuntimeError(f"too few aligned timestamps: {data.shape[0]} (need ≥ 30)")
    print(f"[analysis] aligned shape: T={data.shape[0]} N={data.shape[1]}")
    rets = returns(data)
    print(f"[analysis] returns shape: T={rets.shape[0]}")

    md_lines: list[str] = []
    md_lines.append(f"# Econophysics Analysis Report\n")
    md_lines.append(f"Run timestamp (unix): `{ts_stamp}`\n")
    md_lines.append(f"Input files: {len(inputs)}\n")
    for p in inputs:
        md_lines.append(f"  - `{p}`\n")
    md_lines.append(f"\n**Aligned**: T = {data.shape[0]}, N = {data.shape[1]}\n")
    md_lines.append(f"Series order: `{names}`\n\n")

    # ─── 1. Correlation matrices ───
    print("[analysis] correlation matrices…")
    p_corr = cm.pearson(rets)
    s_corr = cm.spearman(rets)
    k_corr = cm.kendall(rets) if rets.shape[1] <= 12 else np.zeros_like(p_corr)
    _write_matrix_csv(run_dir / "correlation" / "pearson.csv", p_corr, names)
    _write_matrix_csv(run_dir / "correlation" / "spearman.csv", s_corr, names)
    _write_matrix_csv(run_dir / "correlation" / "kendall.csv", k_corr, names)
    top_pearson = _topk_pairs(p_corr, names, k=10)
    md_lines.append("## 1. Correlation (linear vs monotone)\n\n")
    md_lines.append("Top-10 by |Pearson| (with Spearman for comparison; large gap = nonlinear):\n\n")
    md_lines.append("| Pair | Pearson | Spearman | |Diff| |\n|---|---:|---:|---:|\n")
    name_to_idx = {n: i for i, n in enumerate(names)}
    for a, b, v in top_pearson:
        sp = float(s_corr[name_to_idx[a], name_to_idx[b]])
        md_lines.append(f"| {a} ↔ {b} | {v:+.3f} | {sp:+.3f} | {abs(sp - v):.3f} |\n")
    md_lines.append("\n")

    # ─── 2. RMT spectral analysis ───
    print("[analysis] RMT spectral (MP + TW + RIE)…")
    rmt_out = rmt.rmt_summary(rets)
    if rmt_out:
        eigs = rmt_out["eigenvalues"]
        lo, hi = rmt_out["mp_bounds"]
        n_above = rmt_out["n_above_mp"]
        tw_p = rmt_out["tw_pvalue"]
        with open(run_dir / "rmt_eigenvalues.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["k", "eigenvalue", "evr", "rie"])
            for i, e in enumerate(eigs):
                w.writerow([i, f"{e:.6f}", f"{rmt_out['evr'][i]:.6f}", f"{rmt_out['rie_eigenvalues'][i]:.6f}"])
        md_lines.append("## 2. Random Matrix Theory spectral analysis\n\n")
        md_lines.append(f"Aspect ratio q = N/T = {rmt_out['q']:.4f}\n")
        md_lines.append(f"Marchenko-Pastur noise bulk: [{lo:.4f}, {hi:.4f}]\n")
        md_lines.append(f"**Eigenvalues above MP bulk (signal): {n_above} / {len(eigs)}**\n")
        md_lines.append(f"Tracy-Widom p-value for λ_max = {eigs[0]:.4f}: **{tw_p:.4f}** ")
        md_lines.append("(reject H0 of pure noise iff < 0.05)\n\n")
        md_lines.append("Top eigenvalues:\n\n| k | λ | EVR | λ (RIE) |\n|---|---:|---:|---:|\n")
        for i in range(min(5, len(eigs))):
            md_lines.append(f"| {i} | {eigs[i]:.4f} | {rmt_out['evr'][i]:.3%} | {rmt_out['rie_eigenvalues'][i]:.4f} |\n")
        md_lines.append("\n")
    else:
        md_lines.append("## 2. RMT spectral\n\nInsufficient data.\n\n")

    # ─── 3. Mutual information ───
    print("[analysis] mutual information…")
    n = rets.shape[1]
    mi_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = it.mutual_information(rets[:, i], rets[:, j])
            mi_mat[i, j] = v
            mi_mat[j, i] = v
    _write_matrix_csv(run_dir / "info" / "mi.csv", mi_mat, names)
    top_mi = _topk_pairs(mi_mat, names, k=10)
    md_lines.append("## 3. Mutual Information (nonlinear-aware)\n\n")
    md_lines.append("Top-10 by MI (nats). Compare with Pearson: large MI but small |Pearson| ⇒ NONLINEAR dependence.\n\n")
    md_lines.append("| Pair | MI (nats) | |Pearson| |\n|---|---:|---:|\n")
    for a, b, v in top_mi:
        pr = abs(float(p_corr[name_to_idx[a], name_to_idx[b]]))
        md_lines.append(f"| {a} ↔ {b} | {v:.4f} | {pr:.3f} |\n")
    md_lines.append("\n")

    # ─── 4. Distance correlation ───
    print("[analysis] distance correlation…")
    dcor_mat = dep.dcor_matrix(rets)
    _write_matrix_csv(run_dir / "dependence" / "dcor.csv", dcor_mat, names)
    top_dcor = _topk_pairs(dcor_mat, names, k=10)
    md_lines.append("## 4. Distance Correlation (Szekely-Rizzo; 0 iff independent)\n\n")
    md_lines.append("| Pair | dCor | |Pearson| | Gap (catches nonlin) |\n|---|---:|---:|---:|\n")
    for a, b, v in top_dcor:
        pr = abs(float(p_corr[name_to_idx[a], name_to_idx[b]]))
        md_lines.append(f"| {a} ↔ {b} | {v:.3f} | {pr:.3f} | {abs(v - pr):.3f} |\n")
    md_lines.append("\n")

    # ─── 5. Causality (Granger + TE) ───
    print("[analysis] causality…")
    causal_rows: list[tuple[str, str, float, float, float]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            g = cau.granger_f(rets[:, j], rets[:, i], lags=1)
            te = it.transfer_entropy(rets[:, i], rets[:, j])
            causal_rows.append((names[i], names[j], g["f_stat"], g["p_value"], te))
    causal_rows.sort(key=lambda r: r[4], reverse=True)
    with open(run_dir / "causality.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["src", "dst", "granger_F", "granger_p", "TE"])
        for r in causal_rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}"])
    md_lines.append("## 5. Causality (Granger F + Transfer Entropy)\n\n")
    md_lines.append("Top-10 directed edges by Transfer Entropy:\n\n")
    md_lines.append("| Src → Dst | Granger F | Granger p | TE (nats) |\n|---|---:|---:|---:|\n")
    for r in causal_rows[:10]:
        md_lines.append(f"| {r[0]} → {r[1]} | {r[2]:.3f} | {r[3]:.4f} | {r[4]:.4f} |\n")
    md_lines.append("\n")

    # ─── 6. Tail dependence (copula) ───
    print("[analysis] tail dependence…")
    tail_rows: list[tuple[str, str, float, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            lu = cop.upper_tail_dependence(rets[:, i], rets[:, j], q=0.90)
            ll = cop.lower_tail_dependence(rets[:, i], rets[:, j], q=0.10)
            tail_rows.append((names[i], names[j], lu, ll))
    tail_rows.sort(key=lambda r: max(r[2], r[3]), reverse=True)
    with open(run_dir / "tail_dependence.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["a", "b", "lambda_U_q90", "lambda_L_q10"])
        for r in tail_rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}"])
    md_lines.append("## 6. Tail Dependence (empirical copula)\n\n")
    md_lines.append("λ_U > 0.3 means pairs that go up together in the upper tail;\nλ_L > 0.3 means they CRASH together. Both are tail-co-movement red flags.\n\n")
    md_lines.append("| Pair | λ_U (q=0.90) | λ_L (q=0.10) |\n|---|---:|---:|\n")
    for r in tail_rows[:10]:
        md_lines.append(f"| {r[0]} ↔ {r[1]} | {r[2]:.3f} | {r[3]:.3f} |\n")
    md_lines.append("\n")

    # ─── 7. Multifractal (MF-DFA) per series ───
    print("[analysis] MF-DFA…")
    mfdfa_rows: list[tuple[str, float]] = []
    for j in range(n):
        m = mf.mfdfa(rets[:, j])
        if m and m.get("width") is not None:
            mfdfa_rows.append((names[j], float(m["width"])))
    mfdfa_rows.sort(key=lambda r: r[1], reverse=True)
    with open(run_dir / "mfdfa.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["series", "multifractal_width_delta_alpha"])
        for r in mfdfa_rows:
            w.writerow([r[0], f"{r[1]:.4f}"])
    md_lines.append("## 7. Multifractal complexity (MF-DFA)\n\n")
    md_lines.append("Width Δα measures non-Gaussian / cascade complexity. Monofractal (BM) ≈ 0; financial returns typically 0.3-0.6.\n\n")
    md_lines.append("| Series | Δα |\n|---|---:|\n")
    for r in mfdfa_rows[:10]:
        md_lines.append(f"| {r[0]} | {r[1]:.3f} |\n")
    md_lines.append("\n")

    # ─── 8. Diffusion map ───
    print("[analysis] diffusion map…")
    if n >= 3:
        dmap = dm.diffusion_map(rets, n_components=min(2, n - 1))
        with open(run_dir / "diffusion_map.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["series"] + [f"dim{i+1}" for i in range(dmap["embedding"].shape[1])])
            for i, name in enumerate(names):
                w.writerow([name] + [f"{v:.6f}" for v in dmap["embedding"][i]])
        md_lines.append("## 8. Diffusion map (nonlinear manifold embedding)\n\n")
        md_lines.append("Top eigenvalues of the random-walk kernel:\n")
        md_lines.append(f"`{[f'{v:.4f}' for v in dmap['eigenvalues'][:5]]}`\n\n")

    # ─── 9. Correlation network (MST + PMFG) ───
    print("[analysis] MST + PMFG…")
    mst_edges: list[tuple[int, int, float]] = []
    if n >= 3:
        dist = nw.correlation_to_distance(p_corr)
        mst_edges = nw.mst(dist)
        pmfg_edges = nw.pmfg(dist)
        with open(run_dir / "mst_edges.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["a", "b", "distance"])
            for i, j, d in mst_edges:
                w.writerow([names[i], names[j], f"{d:.4f}"])
        deg = nw.node_degree_centrality(mst_edges, n)
        md_lines.append("## 9. MST / PMFG correlation network\n\n")
        md_lines.append(f"MST: {len(mst_edges)} edges (must be N-1 = {n-1}).\n")
        md_lines.append(f"PMFG: {len(pmfg_edges)} edges (bound: 3N-6 = {3*n-6}).\n\n")
        deg_pairs = sorted(zip(names, deg.tolist()), key=lambda x: x[1], reverse=True)
        md_lines.append("Node degree centrality (MST):\n\n| Series | Degree |\n|---|---:|\n")
        for name, d in deg_pairs[:10]:
            md_lines.append(f"| {name} | {d} |\n")
        md_lines.append("\n")

    # ─── 10. α-stable per series ───
    print("[analysis] alpha-stable fits…")
    alpha_rows: list[tuple[str, float, float, float]] = []
    for j in range(n):
        f = al.fit_alpha_stable(rets[:, j])
        alpha_rows.append((names[j], f["alpha"], f["sigma"], f["ks_vs_gaussian"]))
    alpha_rows.sort(key=lambda r: r[1])  # smallest alpha = heaviest tails
    with open(run_dir / "alpha_stable.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["series", "alpha", "sigma", "ks_vs_gaussian"])
        for r in alpha_rows:
            w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.6f}", f"{r[3]:.4f}"])
    md_lines.append("## 10. Lévy α-stable fits (heavy-tail diagnostic)\n\n")
    md_lines.append("α = 2 → Gaussian. α < 2 → heavy-tailed (var infinite below α=2).\n\n")
    md_lines.append("| Series | α | σ | KS vs Gaussian |\n|---|---:|---:|---:|\n")
    for r in alpha_rows[:10]:
        md_lines.append(f"| {r[0]} | {r[1]:.3f} | {r[2]:.5f} | {r[3]:.3f} |\n")
    md_lines.append("\n")

    # ─── 11. Wavelet coherence — show pair with strongest mean coherence ───
    print("[analysis] wavelet coherence (top pair only)…")
    if n >= 2 and rets.shape[0] >= 32:
        # Pick the pair with strongest mean coherence at scale ≈ 8.
        best_pair = None
        best_mean = -1.0
        for i in range(n):
            for j in range(i + 1, n):
                wc = wv.wavelet_coherence(rets[:, i], rets[:, j])
                m = float(wc["mean_coherence_per_scale"].mean())
                if m > best_mean:
                    best_mean = m
                    best_pair = (names[i], names[j], wc)
        if best_pair is not None:
            a, b, wc = best_pair
            (run_dir / "wavelet").mkdir(exist_ok=True)
            with open(run_dir / "wavelet" / f"coherence_{a}_{b}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["scale", "mean_coherence"])
                for s, c in zip(wc["scales"], wc["mean_coherence_per_scale"]):
                    w.writerow([f"{s:.3f}", f"{c:.4f}"])
            md_lines.append("## 11. Wavelet coherence (top pair)\n\n")
            md_lines.append(f"Strongest mean wavelet coherence: **{a} ↔ {b}** (mean R² = {best_mean:.3f}).\n\n")

    # ─── 12. Alpha translation: turn findings into measurable signals ───
    print("[analysis] alpha translation…")
    # 12.1 — TE lead-lag → directional hit rate
    te_edges_raw = [(r[0], r[1], r[4]) for r in causal_rows[:10]]
    te_signals = at.screen_te_edges(rets, names, te_edges_raw, top_k=10)
    # 12.2 — RIE portfolio variance reduction
    rie_lift = at.rie_portfolio_lift(rets, rmt_out) if rmt_out else {}
    # 12.3 — crash co-movement panic basket
    panic = at.panic_pairs(tail_rows, threshold=0.5)
    # 12.4 — Gaussian-VaR underestimation under α-stable
    var_table = at.gaussian_var_underestimation(alpha_rows, rets, names, q=0.99)
    # 12.5 — MST clusters for XS-momentum scoping
    clusters: list[list[str]] = []
    if n >= 3 and mst_edges:
        clusters = at.mst_clusters(mst_edges, n, names, distance_cutoff=1.0)
    # 12.6 — market-mode hedge weights
    eigvec1 = at.top_eigvec_loadings(rets, names)
    # 12.7 — cascade risk
    cascade = at.cascade_risk_rank(mfdfa_rows, threshold=0.5)
    # 12.8 — Hawkes criticality on large-return events per series
    branching: dict[str, float] = {}
    for j in range(n):
        col = rets[:, j]
        thr = float(np.quantile(np.abs(col), 0.90)) if col.size > 30 else 0.0
        events = np.where(np.abs(col) >= thr)[0].astype(float)
        if events.size < 30:
            continue
        params = hw.fit_hawkes(events)
        if params is not None:
            branching[names[j]] = float(params["branching_ratio"])
    criticality = at.criticality_from_branching(branching)

    # Append the alpha-translation section + persist its rows as CSV.
    alpha_dir = run_dir / "alpha"
    alpha_dir.mkdir(exist_ok=True)
    with open(alpha_dir / "te_signals.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "TE", "n", "hit_rate", "binomial_p", "rule_fires"])
        for r in te_signals:
            w.writerow([r["src"], r["dst"], f"{r['TE']:.4f}", r["n"],
                        f"{r['hit_rate']:.4f}", f"{r['binomial_p']:.4f}",
                        int(r["rule_fires"])])
    with open(alpha_dir / "var_underestimation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "alpha", "VaR_gauss_99", "VaR_stable_99",
                    "ratio", "rule_fires"])
        for r in var_table:
            w.writerow([r["series"], f"{r['alpha']:.4f}",
                        f"{r['VaR_gaussian_99']:.6f}",
                        f"{r['VaR_stable_99']:.6f}",
                        f"{r['underestimation_factor']:.3f}",
                        int(r["rule_fires"])])
    with open(alpha_dir / "rie_lift.json", "w") as f:
        json.dump(rie_lift, f, indent=2)
    with open(alpha_dir / "clusters.json", "w") as f:
        json.dump(clusters, f, indent=2)
    with open(alpha_dir / "criticality.json", "w") as f:
        json.dump(criticality, f, indent=2)
    with open(alpha_dir / "panic_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "lambda_U", "lambda_L"])
        for r in panic:
            w.writerow([r["a"], r["b"],
                        f"{r['lambda_U']:.4f}", f"{r['lambda_L']:.4f}"])

    md_lines.append(at.to_markdown(te_signals, rie_lift, panic, var_table,
                                   clusters, eigvec1, cascade, criticality))

    # ─── Write report.md ───
    report_path = run_dir / "report.md"
    with open(report_path, "w") as f:
        f.writelines(md_lines)

    print(f"[analysis] wrote report to {report_path}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True, help="Input CSV files.")
    ap.add_argument("--out", required=True, help="Output directory (timestamped subdir created).")
    ap.add_argument("--max-gap-bars", type=int, default=0,
                    help="Forward-fill gap budget (default 0 = strict inner join).")
    args = ap.parse_args()
    inputs = [Path(p) for p in args.inputs]
    out = Path(args.out)
    run(inputs, out, max_gap_bars=args.max_gap_bars)


if __name__ == "__main__":
    main()
