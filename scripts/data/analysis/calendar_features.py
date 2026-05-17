"""Calendar-effect features for US equity markets.

These are small but persistent anomalies that have been documented in the
empirical-finance literature for decades. None of them are large enough
to trade in isolation, but they are nearly free to compute, orthogonal
to most price-based factors, and often surface as useful conditioning
variables inside a larger feature pipeline:

* Turn-of-month (Lakonishok & Smidt, 1988): equity returns concentrate
  around the month boundary.
* Day-of-week / month-of-year: classic seasonality dummies.
* Pre-FOMC drift (Lucca & Moench, 2015): abnormal positive returns in
  the 24 hours preceding a scheduled FOMC announcement.
* Santa-Claus rally: the last week of December plus the first two
  trading days of January.

All inputs are int64 nanoseconds-since-epoch UTC timestamps that are
assumed to already correspond to trading days (weekends and US holidays
absent). Outputs are float64 arrays aligned 1:1 with the input.

Pure numpy + Python stdlib only.
"""
from __future__ import annotations

import datetime as _dt
from typing import List, Tuple

import numpy as np


# ───────────────────────── timestamp helpers ──────────────────────────

def _ts_to_dates(ts_nanos: np.ndarray) -> List[_dt.date]:
    """Convert int64 ns-since-epoch UTC -> list of datetime.date."""
    out: List[_dt.date] = []
    for ts in ts_nanos:
        # utcfromtimestamp avoids local-tz drift; ts // 1e9 keeps us in seconds.
        out.append(_dt.datetime.utcfromtimestamp(int(ts) / 1e9).date())
    return out


# ──────────────────────────── turn of month ───────────────────────────

def turn_of_month(ts_nanos: np.ndarray) -> np.ndarray:
    """1.0 on the last trading day of each month and the first 3 trading
    days of the next month; 0.0 otherwise.

    Detection rule: a date is the "last trading day of month" iff the
    NEXT timestamp in the input lies in a different calendar month. The
    "first three trading days of the next month" are then simply the
    three subsequent rows of the input array.
    """
    ts_nanos = np.asarray(ts_nanos, dtype=np.int64)
    n = ts_nanos.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    dates = _ts_to_dates(ts_nanos)
    months = np.array([d.month for d in dates], dtype=np.int64)

    # Last trading day of month: months[i] != months[i+1].
    # The final row has no successor and we conservatively mark it as a
    # month-end too (calendar-end of the series — month-end either way).
    for i in range(n):
        is_last_of_month = (i == n - 1) or (months[i] != months[i + 1])
        if is_last_of_month:
            out[i] = 1.0
            # First three trading days of the next month.
            for k in (1, 2, 3):
                if i + k < n:
                    out[i + k] = 1.0
    return out


# ───────────────────────── day-of-week dummies ────────────────────────

def day_of_week_dummies(ts_nanos: np.ndarray) -> np.ndarray:
    """One-hot day-of-week (Mon..Fri only), shape (T, 5).

    Mon=col0, Tue=col1, Wed=col2, Thu=col3, Fri=col4. Weekend rows (which
    should not appear in a trading-day series) get an all-zero row.
    """
    ts_nanos = np.asarray(ts_nanos, dtype=np.int64)
    n = ts_nanos.shape[0]
    out = np.zeros((n, 5), dtype=np.float64)
    if n == 0:
        return out
    dates = _ts_to_dates(ts_nanos)
    for i, d in enumerate(dates):
        wd = d.weekday()  # Mon=0..Sun=6
        if 0 <= wd <= 4:
            out[i, wd] = 1.0
    return out


# ──────────────────────── month-of-year dummies ───────────────────────

def month_of_year_dummies(ts_nanos: np.ndarray) -> np.ndarray:
    """One-hot month, shape (T, 12). Jan=col0, ..., Dec=col11."""
    ts_nanos = np.asarray(ts_nanos, dtype=np.int64)
    n = ts_nanos.shape[0]
    out = np.zeros((n, 12), dtype=np.float64)
    if n == 0:
        return out
    dates = _ts_to_dates(ts_nanos)
    for i, d in enumerate(dates):
        out[i, d.month - 1] = 1.0
    return out


# ──────────────────────────── pre-FOMC drift ──────────────────────────

def fomc_drift(ts_nanos: np.ndarray, fomc_dates: list) -> np.ndarray:
    """1.0 on the trading day BEFORE an FOMC announcement, 0.0 otherwise.

    `fomc_dates` is a list of "YYYY-MM-DD" strings. For each FOMC date
    we mark the largest input timestamp whose date is strictly less than
    the FOMC date. That works even if the FOMC day itself is missing
    from the input (e.g. a holiday or weekend) and avoids look-ahead.
    """
    ts_nanos = np.asarray(ts_nanos, dtype=np.int64)
    n = ts_nanos.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or not fomc_dates:
        return out

    dates = _ts_to_dates(ts_nanos)
    # Map each row -> ordinal day count for fast comparison.
    date_ords = np.array([d.toordinal() for d in dates], dtype=np.int64)

    for s in fomc_dates:
        try:
            fomc = _dt.date.fromisoformat(s)
        except ValueError:
            continue
        target = fomc.toordinal()
        # Index of largest input strictly before FOMC date.
        mask = date_ords < target
        if not np.any(mask):
            continue
        idx = int(np.flatnonzero(mask)[-1])
        out[idx] = 1.0
    return out


# ───────────────────────── santa-claus rally ──────────────────────────

def santa_rally_window(ts_nanos: np.ndarray) -> np.ndarray:
    """1.0 from Dec 26 through Jan 2 each year, 0.0 otherwise.

    Inclusive on both ends; weekends/holidays inside the window are
    naturally skipped because they don't appear in the input.
    """
    ts_nanos = np.asarray(ts_nanos, dtype=np.int64)
    n = ts_nanos.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    dates = _ts_to_dates(ts_nanos)
    for i, d in enumerate(dates):
        m, day = d.month, d.day
        if (m == 12 and day >= 26) or (m == 1 and day <= 2):
            out[i] = 1.0
    return out


# ──────────────────────────── stacked matrix ──────────────────────────

def calendar_feature_matrix(
    ts_nanos: np.ndarray,
    fomc_dates: list | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Stack all calendar features into a single (T, F) matrix.

    Column layout (20 cols total):
        0:        turn_of_month
        1..5:     dow_mon, dow_tue, dow_wed, dow_thu, dow_fri
        6..17:    moy_jan, moy_feb, ..., moy_dec
        18:       fomc_pre
        19:       santa_rally
    """
    if fomc_dates is None:
        fomc_dates = []

    tom = turn_of_month(ts_nanos).reshape(-1, 1)
    dow = day_of_week_dummies(ts_nanos)
    moy = month_of_year_dummies(ts_nanos)
    fomc = fomc_drift(ts_nanos, fomc_dates).reshape(-1, 1)
    santa = santa_rally_window(ts_nanos).reshape(-1, 1)

    mat = np.concatenate([tom, dow, moy, fomc, santa], axis=1)
    names = (
        ["turn_of_month"]
        + ["dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri"]
        + [
            "moy_jan", "moy_feb", "moy_mar", "moy_apr", "moy_may", "moy_jun",
            "moy_jul", "moy_aug", "moy_sep", "moy_oct", "moy_nov", "moy_dec",
        ]
        + ["fomc_pre", "santa_rally"]
    )
    return mat, names


# ────────────────────── hardcoded FOMC calendar ───────────────────────

def hardcoded_fomc_dates_2020_2025() -> List[str]:
    """Actual FOMC meeting announcement dates 2020-01-01 to 2025-12-31.

    Eight scheduled meetings per year (the announcement day is the
    second day of a two-day meeting). The 2020 list also includes the
    March 15 emergency cut. Dates from the Federal Reserve calendar.
    """
    return [
        # 2020 — scheduled meetings + the 2020-03-15 emergency Sunday cut.
        "2020-01-29", "2020-03-03", "2020-03-15", "2020-03-18",
        "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16",
        "2020-11-05", "2020-12-16",
        # 2021
        "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
        "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
        # 2022
        "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
        "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
        # 2023
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
        "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        # 2024
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        # 2025
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    ]
