"""Walk-forward backtester with a realistic cost model.

The contract is intentionally minimal so any strategy — econophysics-
derived or otherwise — can plug in by exposing a single function:

    strategy(train_rets: np.ndarray, names: list[str]) -> np.ndarray

The harness slides a train/test window across the returns matrix,
re-fits the strategy on each train slice, applies the resulting weights
to the held-out test slice, and tracks PnL net of trade-size-proportional
transaction costs and slippage.

Outputs are sufficient for downstream evaluation:

    BacktestResult:
        equity_curve     cumulative wealth at each test bar
        daily_returns    per-bar net returns
        weights          weights held at the start of each test bar
        turnover         |Δw|/2 per bar (one-way)
        gross_costs      cost charged per bar (in returns units)
        sharpe / sortino / max_drawdown / hit_rate / n_rebalances
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


StrategyFn = Callable[[np.ndarray, list[str]], np.ndarray]


@dataclass
class WalkForwardConfig:
    """Knobs for the backtester."""
    train_window: int = 252       # bars used to fit each iteration
    test_window: int = 63         # bars to hold each fitted weight vector
    step: int = 21                # rebalance cadence inside the test window
    cost_bps: float = 2.5         # one-way commission per unit notional
    slippage_bps: float = 1.5     # linear impact per unit notional
    risk_free_rate: float = 0.0   # per-bar (already in test-period units)
    min_history: int = 60         # minimum train rows before we activate

    @property
    def total_cost_bps(self) -> float:
        return self.cost_bps + self.slippage_bps


@dataclass
class BacktestResult:
    """Output of a walk-forward run."""
    equity_curve: np.ndarray = field(default_factory=lambda: np.zeros(0))
    daily_returns: np.ndarray = field(default_factory=lambda: np.zeros(0))
    weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    turnover: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gross_costs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    avg_turnover: float = 0.0
    annual_return: float = 0.0
    annual_vol: float = 0.0
    n_rebalances: int = 0
    n_bars: int = 0


def _summarize(daily_returns: np.ndarray, turnover: np.ndarray,
               periods_per_year: int = 252) -> dict:
    """Compute Sharpe, Sortino, drawdown from a daily PnL series."""
    r = np.asarray(daily_returns, dtype=np.float64)
    n = r.size
    if n == 0:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
                "hit_rate": 0.0, "avg_turnover": 0.0,
                "annual_return": 0.0, "annual_vol": 0.0}
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1)) if n > 1 else 0.0
    downside = r[r < 0.0]
    sigma_dn = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    sharpe = (mu / sigma) * np.sqrt(periods_per_year) if sigma > 0 else 0.0
    sortino = (mu / sigma_dn) * np.sqrt(periods_per_year) if sigma_dn > 0 else 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    drawdown = float(np.min(equity / peak - 1.0)) if equity.size > 0 else 0.0
    hit = float(np.mean(r > 0)) if n > 0 else 0.0
    ann_ret = mu * periods_per_year
    ann_vol = sigma * np.sqrt(periods_per_year)
    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": drawdown,
        "hit_rate": hit,
        "avg_turnover": float(np.mean(turnover)) if turnover.size > 0 else 0.0,
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
    }


def walk_forward(rets: np.ndarray, names: list[str], strategy: StrategyFn,
                 cfg: WalkForwardConfig | None = None) -> BacktestResult:
    """Run `strategy` walk-forward across `rets` (T, N).

    `strategy(train_rets, names)` returns a weight vector of length N.
    Weights may be long-only or long-short; they are NOT normalized by
    the harness — that's the strategy's choice. The harness will warn
    only if the gross exposure exceeds 10× (almost certainly a bug).
    """
    cfg = cfg or WalkForwardConfig()
    rets = np.asarray(rets, dtype=np.float64)
    if rets.ndim != 2:
        raise ValueError("rets must be (T, N)")
    t, n = rets.shape
    if t < cfg.min_history + cfg.test_window:
        raise ValueError(f"need at least {cfg.min_history + cfg.test_window} bars; got {t}")
    if len(names) != n:
        raise ValueError("names length must equal rets.shape[1]")

    start = max(cfg.train_window, cfg.min_history)

    daily_r: list[float] = []
    daily_to: list[float] = []
    daily_costs: list[float] = []
    weights_path: list[np.ndarray] = []
    current_w = np.zeros(n, dtype=np.float64)
    bars_since_rebalance = cfg.step  # force rebalance on first bar
    n_rebalances = 0

    cost_rate = cfg.total_cost_bps / 1e4

    for i in range(start, t):
        # Rebalance every `step` bars (and on the first eligible bar).
        if bars_since_rebalance >= cfg.step:
            train_lo = max(0, i - cfg.train_window)
            train_hi = i  # exclusive — strictly prior data only
            train = rets[train_lo:train_hi]
            try:
                target = strategy(train, names)
            except Exception as e:
                # A strategy that crashes mid-walk holds zero weights.
                target = np.zeros(n)
            target = np.asarray(target, dtype=np.float64).reshape(-1)
            if target.shape[0] != n:
                target = np.zeros(n)
            # Safety: cap exposure at 10× gross.
            gross = float(np.sum(np.abs(target)))
            if gross > 10.0:
                target = target * (10.0 / gross)
            delta = target - current_w
            turn = 0.5 * float(np.sum(np.abs(delta)))  # one-way
            cost = cost_rate * float(np.sum(np.abs(delta)))
            current_w = target
            bars_since_rebalance = 0
            n_rebalances += 1
        else:
            turn = 0.0
            cost = 0.0
        # PnL of the current weights on this bar.
        bar_ret = float(np.dot(current_w, rets[i])) - cost
        daily_r.append(bar_ret)
        daily_to.append(turn)
        daily_costs.append(cost)
        weights_path.append(current_w.copy())
        bars_since_rebalance += 1

    daily_returns = np.asarray(daily_r, dtype=np.float64)
    turnover = np.asarray(daily_to, dtype=np.float64)
    gross_costs = np.asarray(daily_costs, dtype=np.float64)
    weights_arr = np.asarray(weights_path, dtype=np.float64) if weights_path else np.zeros((0, n))
    equity = np.cumprod(1.0 + daily_returns) if daily_returns.size > 0 else np.zeros(0)

    stats = _summarize(daily_returns, turnover)
    return BacktestResult(
        equity_curve=equity,
        daily_returns=daily_returns,
        weights=weights_arr,
        turnover=turnover,
        gross_costs=gross_costs,
        sharpe=stats["sharpe"],
        sortino=stats["sortino"],
        max_drawdown=stats["max_drawdown"],
        hit_rate=stats["hit_rate"],
        avg_turnover=stats["avg_turnover"],
        annual_return=stats["annual_return"],
        annual_vol=stats["annual_vol"],
        n_rebalances=n_rebalances,
        n_bars=int(daily_returns.size),
    )


def compare_strategies(rets: np.ndarray, names: list[str],
                       strategies: dict[str, StrategyFn],
                       cfg: WalkForwardConfig | None = None) -> dict[str, BacktestResult]:
    """Run a dict of strategies and return their results keyed by name."""
    out: dict[str, BacktestResult] = {}
    for sname, fn in strategies.items():
        out[sname] = walk_forward(rets, names, fn, cfg)
    return out


def summarize_table(results: dict[str, BacktestResult]) -> str:
    """Pretty-print a comparison table as markdown."""
    rows = []
    rows.append("| Strategy | Sharpe | Sortino | AnnRet | AnnVol | MaxDD | HitRate | AvgTO | Rebals |")
    rows.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sname, r in sorted(results.items(), key=lambda kv: kv[1].sharpe, reverse=True):
        rows.append(f"| {sname} | {r.sharpe:+.3f} | {r.sortino:+.3f} | "
                    f"{r.annual_return:+.3%} | {r.annual_vol:.3%} | "
                    f"{r.max_drawdown:.3%} | {r.hit_rate:.3f} | "
                    f"{r.avg_turnover:.3f} | {r.n_rebalances} |")
    return "\n".join(rows) + "\n"
