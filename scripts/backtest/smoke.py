"""Deterministic offline research workflow with machine-readable evidence."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import baseline_strategies as baselines
from walkforward import WalkForwardConfig, compare_strategies, summarize_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/research-smoke.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    names = [f"SYN{i}" for i in range(8)]
    # Shared market factor plus independent idiosyncratic returns.
    returns = rng.normal(0.0001, 0.006, (720, 1)) + rng.normal(0, 0.008, (720, 8))
    config = WalkForwardConfig(train_window=252, test_window=63, step=21)
    results = compare_strategies(returns, names, {
        "cash": baselines.zero_weight,
        "equal_weight": baselines.equal_weight,
        "momentum": baselines.momentum_xs,
        "min_variance": baselines.min_variance,
    }, config)
    payload = {"dataset": "seeded synthetic daily returns", "seed": args.seed,
               "symbols": names, "config": asdict(config), "strategies": {}}
    for name, result in results.items():
        fields = asdict(result)
        fields = {key: value.tolist() if isinstance(value, np.ndarray) else value
                  for key, value in fields.items()}
        if not np.isfinite(result.equity_curve).all() or result.n_bars != 468:
            raise RuntimeError(f"Invalid held-out result for {name}")
        payload["strategies"][name] = fields
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(summarize_table(results))
    print(f"Saved actual held-out equity curves to {args.out}")


if __name__ == "__main__":
    main()
