"""Retail-attention and news-tone sentiment features.

This module turns two alt-data sources into clean, tradeable signals:

  * Wikipedia daily pageviews — a documented proxy for retail/search
    attention ("FEARS Index", Da/Engelberg/Gao 2014). A surge of views
    for a name or theme tends to precede capitulation or mania moves.
  * GDELT global news tone — the average sentiment of news mentions
    for an actor/country. Tone z-scores and news-volume z-scores form
    a fast macro narrative gauge (Leetaru & Schrodt 2013).

Both feeds are converted into the project's canonical macro schema
(ts_nanos, series, value) so downstream analytics can join them with
every other macro feature without special-casing.

Pure numpy + stdlib; no scipy/pandas/sklearn dependency.

Schema in / out:
    input  data/pageviews/<topic>.csv  (ts_nanos, article, views)
    input  data/news/gdelt/<actor>.csv (ts_nanos, actor, avg_tone, num_events)
    output data/sentiment/<topic>_attention.csv  (ts_nanos, series, value)
    output data/sentiment/gdelt_<actor>_features.csv (ts_nanos, series, value)
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


# ──────────────────────────── I/O helpers ──────────────────────────────


def load_pageviews(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a pageviews CSV; return (ts_nanos, views) sorted by ts_nanos.

    Schema: ``ts_nanos,article,views`` (header optional but expected).
    Non-finite or unparseable rows are dropped silently.
    """
    ts_list: list[int] = []
    val_list: list[float] = []
    path = Path(path)
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        try:
            ts_idx = header.index("ts_nanos")
            val_idx = header.index("views")
        except ValueError:
            ts_idx, val_idx = 0, 2
            # The "header" was really a data row.
            try:
                ts_list.append(int(header[ts_idx]))
                val_list.append(float(header[val_idx]))
            except (ValueError, IndexError):
                pass
        for row in reader:
            if not row:
                continue
            try:
                ts = int(row[ts_idx])
                val = float(row[val_idx])
            except (ValueError, IndexError):
                continue
            if not math.isfinite(val):
                continue
            ts_list.append(ts)
            val_list.append(val)

    if not ts_list:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    ts_arr = np.asarray(ts_list, dtype=np.int64)
    val_arr = np.asarray(val_list, dtype=np.float64)
    order = np.argsort(ts_arr, kind="stable")
    return ts_arr[order], val_arr[order]


def _load_gdelt_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a GDELT per-actor CSV; return (ts, avg_tone, num_events) sorted.

    Tolerates either the documented per-actor schema
    ``ts_nanos,actor,avg_tone,num_events`` or the project's existing
    daily-summary schema ``ts_nanos,n_events,avg_tone,avg_goldstein``.
    """
    ts_list: list[int] = []
    tone_list: list[float] = []
    vol_list: list[float] = []
    path = Path(path)
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            empty = np.empty(0, dtype=np.float64)
            return np.empty(0, dtype=np.int64), empty, empty
        # Resolve column indices from the header.
        try:
            ts_idx = header.index("ts_nanos")
        except ValueError:
            ts_idx = 0
        if "avg_tone" in header:
            tone_idx = header.index("avg_tone")
        else:
            tone_idx = 2
        if "num_events" in header:
            vol_idx = header.index("num_events")
        elif "n_events" in header:
            vol_idx = header.index("n_events")
        else:
            vol_idx = 3
        for row in reader:
            if not row:
                continue
            try:
                ts = int(row[ts_idx])
                tone = float(row[tone_idx])
                vol = float(row[vol_idx])
            except (ValueError, IndexError):
                continue
            if not (math.isfinite(tone) and math.isfinite(vol)):
                continue
            ts_list.append(ts)
            tone_list.append(tone)
            vol_list.append(vol)

    if not ts_list:
        empty = np.empty(0, dtype=np.float64)
        return np.empty(0, dtype=np.int64), empty, empty

    ts_arr = np.asarray(ts_list, dtype=np.int64)
    tone_arr = np.asarray(tone_list, dtype=np.float64)
    vol_arr = np.asarray(vol_list, dtype=np.float64)
    order = np.argsort(ts_arr, kind="stable")
    return ts_arr[order], tone_arr[order], vol_arr[order]


# ───────────────────────── attention features ──────────────────────────


def attention_z(values: np.ndarray, window: int = 60) -> np.ndarray:
    """Causal rolling z-score of ``values`` over the trailing ``window``.

    Baseline mean and std use ONLY ``values[t-window:t]`` — the current
    bar is excluded so the score is strictly observable at time t.

    Returns a same-length array; the first ``window`` entries are NaN
    (insufficient history). Entries where the baseline std is zero are
    also NaN (a constant prior window has no scale).
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or window <= 0:
        return out

    w = int(window)
    for t in range(w, n):
        prior = values[t - w:t]
        # Only require window-many finite samples — drop the bar if not.
        if not np.all(np.isfinite(prior)):
            prior = prior[np.isfinite(prior)]
            if prior.size < 2:
                continue
        mean = float(prior.mean())
        sd = float(prior.std(ddof=1)) if prior.size >= 2 else 0.0
        v = values[t]
        if not math.isfinite(v) or not math.isfinite(sd) or sd <= 0.0:
            continue
        out[t] = (v - mean) / sd
    return out


def attention_change_1d(values: np.ndarray) -> np.ndarray:
    """1-bar log-change: ``log(values[t] / values[t-1])``.

    Returns NaN at t=0 (no prior bar). When either side is zero or
    negative the result is 0.0 — pageview counts can legitimately drop
    to 0 (article not fetched) and a -inf return would propagate
    through downstream rolling stats.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    for t in range(1, n):
        prev = values[t - 1]
        curr = values[t]
        if not (math.isfinite(prev) and math.isfinite(curr)):
            out[t] = 0.0
            continue
        if prev <= 0.0 or curr <= 0.0:
            out[t] = 0.0
            continue
        out[t] = math.log(curr / prev)
    return out


# ────────────────────── batch attention features ───────────────────────


def attention_features(pageviews_dir: Path, out_dir: Path,
                       window: int = 60) -> dict[str, np.ndarray]:
    """For each ``.csv`` in ``pageviews_dir`` compute attention_z and
    attention_change_1d; write to ``<out_dir>/<topic>_attention.csv`` in
    the macro schema (``ts_nanos,series,value``) — two rows per ts (one
    for the z-score, one for the 1d log-change).

    Series naming inside the output CSV:
        <topic>_attention_z
        <topic>_attention_chg1d

    Returns a dict of feature arrays keyed by topic, where each entry
    is a 2-D array stacked as ``[z, chg1d]`` along axis 1. Topics with
    no usable rows are skipped silently.
    """
    pageviews_dir = Path(pageviews_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, np.ndarray] = {}
    if not pageviews_dir.exists():
        return results

    for csv_path in sorted(pageviews_dir.glob("*.csv")):
        topic = csv_path.stem
        ts, views = load_pageviews(csv_path)
        if ts.size == 0:
            continue
        z = attention_z(views, window=window)
        chg = attention_change_1d(views)
        results[topic] = np.column_stack([z, chg])

        out_path = out_dir / f"{topic}_attention.csv"
        z_label = f"{topic}_attention_z"
        chg_label = f"{topic}_attention_chg1d"
        with out_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["ts_nanos", "series", "value"])
            for ts_i, z_i, chg_i in zip(ts.tolist(), z.tolist(), chg.tolist()):
                if math.isfinite(z_i):
                    writer.writerow([int(ts_i), z_label, f"{z_i:.10f}"])
                if math.isfinite(chg_i):
                    writer.writerow([int(ts_i), chg_label, f"{chg_i:.10f}"])

    return results


# ─────────────────────── GDELT news-tone features ──────────────────────


def _rolling_mean_window(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean over the trailing ``window`` (excludes t)."""
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for t in range(window, n):
        prior = values[t - window:t]
        prior = prior[np.isfinite(prior)]
        if prior.size >= 1:
            out[t] = float(prior.mean())
    return out


def gdelt_tone_features(gdelt_dir: Path, out_dir: Path) -> dict[str, np.ndarray]:
    """For each CSV in ``gdelt_dir`` compute three features:

      * ``tone_z``         — causal rolling z of avg_tone over 30 bars
      * ``tone_momentum``  — ``tone[t] - mean(tone[t-7:t])``
      * ``news_volume_z``  — causal rolling z of num_events over 30 bars

    Output goes to ``<out_dir>/gdelt_<actor>_features.csv`` in the
    macro schema. If ``gdelt_dir`` does not exist (data not fetched
    yet) the function returns ``{}`` without error so it can be wired
    into a pipeline unconditionally.

    Returns a dict keyed by actor stem with a 2-D array
    ``[tone_z, tone_momentum, news_volume_z]`` stacked along axis 1.
    """
    gdelt_dir = Path(gdelt_dir)
    out_dir = Path(out_dir)
    results: dict[str, np.ndarray] = {}
    if not gdelt_dir.exists():
        return results
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(gdelt_dir.glob("*.csv")):
        actor = csv_path.stem
        ts, tone, vol = _load_gdelt_csv(csv_path)
        if ts.size == 0:
            continue
        tone_z = attention_z(tone, window=30)
        tone_mom = tone - _rolling_mean_window(tone, window=7)
        vol_z = attention_z(vol, window=30)
        results[actor] = np.column_stack([tone_z, tone_mom, vol_z])

        out_path = out_dir / f"gdelt_{actor}_features.csv"
        labels = (
            f"gdelt_{actor}_tone_z",
            f"gdelt_{actor}_tone_momentum",
            f"gdelt_{actor}_news_volume_z",
        )
        with out_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["ts_nanos", "series", "value"])
            for ts_i, z_i, m_i, v_i in zip(
                ts.tolist(), tone_z.tolist(), tone_mom.tolist(), vol_z.tolist()
            ):
                if math.isfinite(z_i):
                    writer.writerow([int(ts_i), labels[0], f"{z_i:.10f}"])
                if math.isfinite(m_i):
                    writer.writerow([int(ts_i), labels[1], f"{m_i:.10f}"])
                if math.isfinite(v_i):
                    writer.writerow([int(ts_i), labels[2], f"{v_i:.10f}"])

    return results


# ────────────────────────── contrarian signal ──────────────────────────


def contrarian_attention_signal(topic_attention_z: np.ndarray,
                                z_threshold: float = 2.0) -> np.ndarray:
    """Classic contrarian attention rule.

    Extreme retail attention (``z > +threshold``) often coincides with
    short-term tops (mania); extreme apathy (``z < -threshold``) often
    marks capitulation lows. We translate the z-score into a discrete
    position recommendation:

        z >  +threshold  -> -1  (short / fade the crowd)
        z <  -threshold  -> +1  (buy capitulation)
        otherwise        ->  0  (neutral)

    NaN inputs map to 0 (no signal).
    """
    z = np.asarray(topic_attention_z, dtype=np.float64)
    thr = float(z_threshold)
    out = np.zeros(z.shape, dtype=np.float64)
    finite = np.isfinite(z)
    out[finite & (z > thr)] = -1.0
    out[finite & (z < -thr)] = 1.0
    return out
