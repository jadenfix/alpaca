"""Alpha-discovery orchestrator.

Wires together every Tier-1 component:
  • Walk-forward backtest of all candidate strategies vs baselines
  • Multi-horizon TE/CCM screen
  • Regime-conditioned sign predictability
  • Rolling econophysics with shift detection
  • Multivariate Hawkes contagion graph

Produces a single markdown report ranking strategies and signals by
out-of-sample evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "strategies"))
sys.path.insert(0, str(HERE.parent / "data" / "analysis"))
sys.path.insert(0, str(HERE.parent / "data"))

from loader import align, returns  # noqa: E402
from walkforward import (WalkForwardConfig, compare_strategies,  # noqa: E402
                          summarize_table)
import baseline_strategies as bl  # noqa: E402
import rmt_residual  # noqa: E402
import rmt_minvar  # noqa: E402
import cluster_meanrev  # noqa: E402
import cointegration_spread  # noqa: E402
import multihorizon_te as mh  # noqa: E402
import regime_conditioned as rc  # noqa: E402
import rolling_econ as re_mod  # noqa: E402
import multivariate_hawkes as mvh  # noqa: E402


def _topk_pairs(mat: np.ndarray, names: list[str], k: int = 10) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str, float]] = []
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append((names[i], names[j], float(mat[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return [(a, b) for a, b, _ in pairs[:k]]


def run(inputs: list[Path], out_dir: Path, max_gap_bars: int = 0) -> Path:
    ts = int(time.time())
    run_dir = out_dir / f"alpha_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[discover] loading {len(inputs)} CSVs…")
    data, names, _ = align(inputs, max_gap_bars=max_gap_bars)
    if data.shape[0] < 120:
        raise RuntimeError(f"too few aligned rows: {data.shape[0]} (need >=120)")
    rets = returns(data)
    t, n = rets.shape
    print(f"[discover] returns shape: T={t}, N={n}")

    md: list[str] = [f"# Alpha Discovery Report\nUnix ts: `{ts}`\n",
                     f"Aligned: T = {t}, N = {n}\n",
                     f"Universe: `{names}`\n\n"]

    # ─────────── 1. Walk-forward strategy comparison ───────────
    print("[discover] walk-forward strategy comparison…")
    cfg = WalkForwardConfig(train_window=min(252, t // 2),
                            test_window=min(63, t // 4),
                            step=21,
                            cost_bps=2.5,
                            slippage_bps=1.5,
                            min_history=60)
    strategies = {
        "equal_weight": bl.equal_weight,
        "min_variance": bl.min_variance,
        "rmt_minvar": rmt_minvar.factory(),
        "xs_momentum_60d": bl.momentum_xs,
        "xs_mean_reversion_5d": bl.mean_reversion_xs,
        "rmt_residual_z1.5": rmt_residual.factory(z_threshold=1.5),
        "rmt_residual_z2.0": rmt_residual.factory(z_threshold=2.0),
        "cluster_meanrev_default": cluster_meanrev.factory(),
        "cluster_meanrev_z0.8": cluster_meanrev.factory(within_z_threshold=0.8),
        "cointegration_z2.0": cointegration_spread.factory(z_entry=2.0),
    }
    results = compare_strategies(rets, names, strategies, cfg)
    md.append("## 1. Strategy walk-forward results (NET of 4 bps round-trip)\n\n")
    md.append(summarize_table(results) + "\n")

    with open(run_dir / "strategy_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "sharpe", "sortino", "annual_return",
                    "annual_vol", "max_drawdown", "hit_rate",
                    "avg_turnover", "n_rebalances", "n_bars"])
        for sname, r in sorted(results.items(),
                                key=lambda kv: kv[1].sharpe, reverse=True):
            w.writerow([sname, f"{r.sharpe:.4f}", f"{r.sortino:.4f}",
                        f"{r.annual_return:.6f}", f"{r.annual_vol:.6f}",
                        f"{r.max_drawdown:.6f}", f"{r.hit_rate:.4f}",
                        f"{r.avg_turnover:.4f}", r.n_rebalances, r.n_bars])

    # Persist equity curves per strategy for downstream charting.
    eq_dir = run_dir / "equity_curves"
    eq_dir.mkdir(exist_ok=True)
    for sname, r in results.items():
        with open(eq_dir / f"{sname}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bar", "equity", "daily_return", "turnover"])
            for i, (e, dr, to) in enumerate(zip(r.equity_curve,
                                                 r.daily_returns,
                                                 r.turnover)):
                w.writerow([i, f"{e:.6f}", f"{dr:.6f}", f"{to:.6f}"])

    # ─────────── 2. Multi-horizon TE / sign predictability ───────────
    print("[discover] multi-horizon TE screen…")
    # Pick the top-15 correlated pairs as candidates to limit compute.
    from correlation_matrix import pearson as cm_pearson
    corr = cm_pearson(rets)
    cand_pairs = _topk_pairs(np.abs(corr), names, k=15)
    mh_results = mh.screen_edges(rets, names, cand_pairs,
                                  k_grid=[1, 2], l_grid=[1, 2],
                                  h_grid=[1, 5, 21])
    md.append("## 2. Multi-horizon sign predictability (top correlated pairs)\n\n")
    md.append("Edge fires if it monotonically decays (signal strongest at smallest h).\n\n")
    md.append("| Src → Dst | best k | best l | best h | TE | hit rate | monotone? |\n"
              "|---|:--:|:--:|:--:|---:|---:|:--:|\n")
    for r in mh_results[:15]:
        mono = "**Y**" if r["monotone"] else "n"
        md.append(f"| {r['src']} → {r['dst']} | {r['best_k']} | {r['best_l']} | "
                  f"{r['best_h']} | {r['te']:.4f} | {r['hit_rate']:.3f} | {mono} |\n")
    monotone_count = sum(1 for r in mh_results if r["monotone"])
    md.append(f"\n**{monotone_count} / {len(mh_results)} edges are monotone "
              f"(signal weakens with horizon — the well-behaved class).**\n\n")
    with open(run_dir / "multihorizon_te.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "best_k", "best_l", "best_h", "te",
                    "hit_rate", "monotone"])
        for r in mh_results:
            w.writerow([r["src"], r["dst"], r["best_k"], r["best_l"],
                        r["best_h"], f"{r['te']:.4f}",
                        f"{r['hit_rate']:.4f}", int(r["monotone"])])

    # ─────────── 3. Regime-conditioned screen ───────────
    print("[discover] regime-conditioned screen…")
    regime_signals = rc.screen_conditional_edges(rets, names, cand_pairs,
                                                  n_buckets=5,
                                                  min_buckets_firing=3)
    md.append("## 3. Regime-conditioned signal robustness\n\n")
    md.append("Keep rule: fires in ≥ 3 of 5 vol-regime buckets.\n\n")
    if regime_signals:
        md.append("| Src → Dst | buckets firing | mean hit | min hit |\n"
                  "|---|:--:|---:|---:|\n")
        for r in regime_signals[:15]:
            md.append(f"| {r['src']} → {r['dst']} | {r['n_buckets_firing']}/5 | "
                      f"{r['mean_hit_rate']:.3f} | {r['min_hit_rate']:.3f} |\n")
        md.append("\n")
    else:
        md.append("No edges fire in ≥ 3 buckets — none robust across regimes.\n\n")
    with open(run_dir / "regime_signals.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "buckets_firing", "mean_hit", "min_hit"])
        for r in regime_signals:
            w.writerow([r["src"], r["dst"], r["n_buckets_firing"],
                        f"{r['mean_hit_rate']:.4f}",
                        f"{r['min_hit_rate']:.4f}"])

    # ─────────── 4. Rolling econophysics + shift detection ───────────
    print("[discover] rolling econophysics + shift detection…")
    if t >= 252:
        rolling = re_mod.rolling_econ_report(rets, names,
                                              window=min(252, t // 2),
                                              step=21)
        md.append("## 4. Rolling econophysics shifts (leading indicators)\n\n")
        shifts = rolling.get("shifts", [])
        if shifts:
            md.append(f"**{len(shifts)} alerts** where a rolling stat moved > 2σ "
                      "from its trailing 60-window mean.\n\n")
            md.append("| Feature | Bar idx | Value | z |\n|---|---:|---:|---:|\n")
            for s in shifts[:20]:
                md.append(f"| {s.get('feature', '?')} | "
                          f"{s.get('ts_idx', s.get('bin_idx', '?'))} | "
                          f"{s['value']:.4f} | {s['z']:+.2f} |\n")
            md.append("\n")
        else:
            md.append("No rolling-feature shifts crossed the z=2 threshold.\n\n")
        with open(run_dir / "shifts.json", "w") as f:
            json.dump(shifts, f, indent=2)
    else:
        md.append("## 4. Rolling econophysics\n\nInsufficient bars (need ≥ 252).\n\n")

    # ─────────── 5. Multivariate Hawkes contagion ───────────
    print("[discover] multivariate Hawkes (top-3 series by |ret| events)…")
    # Pick a subset of series with most extreme events to keep MLE tractable.
    abs_means = np.mean(np.abs(rets), axis=0)
    pick = np.argsort(abs_means)[-3:]
    pick_names = [names[i] for i in pick]
    events_per_dim: list[np.ndarray] = []
    for i in pick:
        col = rets[:, i]
        thr = float(np.quantile(np.abs(col), 0.90))
        ev = np.where(np.abs(col) >= thr)[0].astype(np.float64)
        events_per_dim.append(ev)
    t_end = float(rets.shape[0])
    mvh_result = None
    try:
        if all(e.size >= 30 for e in events_per_dim):
            mvh_result = mvh.fit_mv_hawkes(events_per_dim, t_end)
    except Exception as e:
        print(f"[discover] mv_hawkes skipped: {e}")
    md.append("## 5. Multivariate Hawkes contagion (3 most-volatile series)\n\n")
    if mvh_result is not None:
        K = mvh_result["branching_matrix"]
        sr = mvh_result["spectral_radius"]
        md.append(f"Spectral radius ρ(K) = **{sr:.3f}** "
                  f"(must be < 1 for stationarity; > 0.9 = near-critical).\n\n")
        md.append(f"Series: `{pick_names}`\n\n")
        md.append("Branching matrix K (rows = response, cols = trigger):\n\n")
        md.append("| | " + " | ".join(pick_names) + " |\n")
        md.append("|---|" + "|".join(["---:"] * len(pick_names)) + "|\n")
        for r_idx, rn in enumerate(pick_names):
            md.append(f"| **{rn}** | " +
                      " | ".join(f"{K[r_idx, c]:.3f}" for c in range(len(pick_names))) +
                      " |\n")
        md.append("\n")
        cont = mvh.contagion_paths(K, pick_names, top_k=5)
        if cont:
            md.append("Top cross-excitations:\n\n")
            for c in cont:
                md.append(f"- {c['trigger_name']} → {c['response_name']} "
                          f"(K = {c['k_value']:.3f})\n")
            md.append("\n")
        with open(run_dir / "mv_hawkes.json", "w") as f:
            json.dump({"mu": mvh_result["mu"].tolist(),
                       "alpha": mvh_result["alpha"].tolist(),
                       "beta": mvh_result["beta"].tolist(),
                       "branching_matrix": K.tolist(),
                       "spectral_radius": sr,
                       "loglik": mvh_result["loglik"],
                       "series": pick_names}, f, indent=2)
    else:
        md.append("Hawkes fit unavailable (too few events or fit failed).\n\n")

    # ─────────── 6. Recommendation ───────────
    md.append("## 6. Recommendation\n\n")
    ranked = sorted(results.items(), key=lambda kv: kv[1].sharpe, reverse=True)
    top = ranked[0]
    md.append(f"Best walk-forward Sharpe: **{top[0]}** at **{top[1].sharpe:+.2f}** "
              f"(annual return {top[1].annual_return:+.2%}, max DD "
              f"{top[1].max_drawdown:.2%}).\n\n")
    md.append(f"Robust signals (regime-conditioned, ≥3/5 buckets): "
              f"**{len(regime_signals)}**.\n")
    md.append(f"Monotone TE edges (signal decays with horizon): "
              f"**{monotone_count}/{len(mh_results)}**.\n")
    if mvh_result is not None:
        md.append(f"Hawkes criticality: ρ(K) = {mvh_result['spectral_radius']:.2f}.\n")
    md.append("\n")

    report_path = run_dir / "report.md"
    with open(report_path, "w") as f:
        f.write("".join(md))
    print(f"[discover] wrote {report_path}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gap-bars", type=int, default=0)
    args = ap.parse_args()
    run([Path(p) for p in args.inputs], Path(args.out),
        max_gap_bars=args.max_gap_bars)


if __name__ == "__main__":
    main()
