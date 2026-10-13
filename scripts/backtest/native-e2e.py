#!/usr/bin/env python3
"""Run the actual CPU CLI through generation, CSV storage and repeatable backtests."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    subprocess.run(["cargo", "build", "--locked", "--bin", "algo-backtest"], cwd=ROOT, check=True)
    metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=ROOT))
    binary = str(Path(metadata["target_directory"]) / "debug/algo-backtest")
    with tempfile.TemporaryDirectory(prefix="alpaca-native-e2e-") as directory:
        root = Path(directory)
        bars = root / "bars.csv"
        subprocess.run([binary, "gen-data", "--out", str(bars), "--symbols", "4", "--bars", "300", "--seed", "42"], check=True)
        with bars.open() as handle:
            assert len(list(csv.DictReader(handle))) == 1200
        curves = []
        for attempt in range(2):
            path = root / f"equity-{attempt}.csv"
            report = subprocess.run([binary, "csv", "--path", str(bars), "--strategy", "xs_momentum", "--equity-csv", str(path)], text=True, capture_output=True, check=True)
            assert "Backtest Report (xs_momentum)" in report.stdout
            with path.open() as handle:
                curve = [(int(row["ts_nanos"]), float(row["nav"])) for row in csv.DictReader(handle)]
            assert curve and all(math.isfinite(value) and value > 0 for _, value in curve)
            assert all(left[0] <= right[0] for left, right in zip(curve, curve[1:]))
            curves.append(curve)
        # Randomized hash-map summation order may change last floating bits.
        # Financially equivalent curves must agree within numerical roundoff.
        assert len(curves[0]) == len(curves[1])
        for first, second in zip(*curves):
            assert first[0] == second[0]
            assert math.isclose(first[1], second[1], rel_tol=1e-12, abs_tol=1e-8)
        invalid = subprocess.run([binary, "csv", "--path", str(bars), "--strategy", "does-not-exist"], capture_output=True)
        assert invalid.returncode != 0
        print(json.dumps({"status": "pass", "persisted_bars": 1200, "equity_rows": len(curves[0]), "repeatability": "within floating-point roundoff", "live_orders": False}, indent=2))


if __name__ == "__main__":
    main()
