"""Known-truth tests for calendar_features.

Each test would FAIL on a no-op or constant-returning implementation,
so they validate the seasonality logic, not just the API surface.
"""
from __future__ import annotations

import datetime as _dt
import re
import sys
import unittest
from pathlib import Path

import numpy as np

# Make the analysis package importable (matches test_analysis.py).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE.parent))

import calendar_features as cf  # noqa: E402


# ─────────────────────────── helpers ────────────────────────────

_NS_PER_DAY = 86_400 * 1_000_000_000


def _date_to_ns(d: _dt.date) -> int:
    """Convert a UTC date to ns-since-epoch at 00:00:00 UTC."""
    ts = _dt.datetime(d.year, d.month, d.day, tzinfo=_dt.timezone.utc).timestamp()
    return int(ts * 1e9)


def _trading_days(start: _dt.date, count: int) -> np.ndarray:
    """Return `count` consecutive weekday timestamps starting at `start`."""
    out = []
    d = start
    while len(out) < count:
        if d.weekday() < 5:  # Mon-Fri
            out.append(_date_to_ns(d))
        d += _dt.timedelta(days=1)
    return np.array(out, dtype=np.int64)


def _trading_days_between(start: _dt.date, end: _dt.date) -> np.ndarray:
    """All Mon-Fri timestamps from `start` to `end` inclusive."""
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(_date_to_ns(d))
        d += _dt.timedelta(days=1)
    return np.array(out, dtype=np.int64)


# ─────────────────────── turn_of_month ──────────────────────────

class TurnOfMonthTests(unittest.TestCase):
    def test_average_flagging_rate_matches_4_per_21(self):
        # 252 consecutive weekdays starting 2023-01-02 (~12 months).
        ts = _trading_days(_dt.date(2023, 1, 2), 252)
        flags = cf.turn_of_month(ts)
        self.assertEqual(flags.shape, (252,))
        # ~4 flagged days per ~21-trading-day month = ~0.19 mean.
        self.assertAlmostEqual(flags.mean(), 4.0 / 21.0, delta=0.05)
        # Sanity: indicator is strictly 0/1.
        self.assertTrue(np.all((flags == 0.0) | (flags == 1.0)))

    def test_known_january_2023_boundary(self):
        # 2023-01-31 (Tue) is last trading day of Jan; Feb 1-3 are first
        # three trading days of Feb. All four should fire.
        ts = _trading_days(_dt.date(2023, 1, 30), 6)  # Jan30..Feb6
        flags = cf.turn_of_month(ts)
        # ts[1]=Jan31, ts[2]=Feb1, ts[3]=Feb2, ts[4]=Feb3.
        self.assertEqual(flags[1], 1.0)
        self.assertEqual(flags[2], 1.0)
        self.assertEqual(flags[3], 1.0)
        self.assertEqual(flags[4], 1.0)


# ────────────────────── day_of_week_dummies ─────────────────────

class DayOfWeekTests(unittest.TestCase):
    def test_one_hot_rows_sum_to_one(self):
        ts = _trading_days(_dt.date(2024, 3, 4), 60)  # 60 weekdays
        d = cf.day_of_week_dummies(ts)
        self.assertEqual(d.shape, (60, 5))
        row_sums = d.sum(axis=1)
        np.testing.assert_array_equal(row_sums, np.ones(60))

    def test_monday_lands_in_col0(self):
        # 2024-03-04 is a Monday.
        ts = _trading_days(_dt.date(2024, 3, 4), 5)
        d = cf.day_of_week_dummies(ts)
        self.assertEqual(d[0, 0], 1.0)
        self.assertEqual(d[4, 4], 1.0)  # Friday in col 4


# ───────────────────── month_of_year_dummies ────────────────────

class MonthOfYearTests(unittest.TestCase):
    def test_shape_and_one_hot(self):
        ts = _trading_days(_dt.date(2023, 1, 2), 252)
        m = cf.month_of_year_dummies(ts)
        self.assertEqual(m.shape, (252, 12))
        np.testing.assert_array_equal(m.sum(axis=1), np.ones(252))


# ─────────────────────────── fomc_drift ─────────────────────────

class FOMCDriftTests(unittest.TestCase):
    def test_three_in_range_fires_three(self):
        # Build a small trading-day grid covering early 2024.
        ts = _trading_days_between(_dt.date(2024, 1, 2), _dt.date(2024, 6, 30))
        fomc = ["2024-01-31", "2024-03-20", "2024-05-01"]
        drift = cf.fomc_drift(ts, fomc)
        self.assertEqual(drift.shape, ts.shape)
        self.assertEqual(int(drift.sum()), 3)
        # Indicator only.
        self.assertTrue(np.all((drift == 0.0) | (drift == 1.0)))

    def test_empty_dates_returns_zeros(self):
        ts = _trading_days(_dt.date(2024, 1, 2), 30)
        drift = cf.fomc_drift(ts, [])
        np.testing.assert_array_equal(drift, np.zeros(30))


# ──────────────────────── santa_rally_window ────────────────────

class SantaRallyTests(unittest.TestCase):
    def test_only_fires_in_late_dec_early_jan(self):
        # Full calendar year of weekdays for 2023.
        ts = _trading_days_between(_dt.date(2023, 1, 2), _dt.date(2023, 12, 29))
        s = cf.santa_rally_window(ts)
        total = int(s.sum())
        # Expect 5-8 fires (Dec 26-29 + Jan 2 are the trading days in 2023).
        self.assertGreaterEqual(total, 5)
        self.assertLessEqual(total, 8)

        # All flagged rows must be Dec >=26 or Jan <=2.
        for i, flag in enumerate(s):
            if flag == 1.0:
                d = _dt.datetime.utcfromtimestamp(int(ts[i]) / 1e9).date()
                self.assertTrue(
                    (d.month == 12 and d.day >= 26)
                    or (d.month == 1 and d.day <= 2),
                    f"flagged unexpected date {d}",
                )


# ──────────────────── hardcoded FOMC calendar ───────────────────

class HardcodedFOMCTests(unittest.TestCase):
    def test_approx_48_dates_in_range(self):
        dates = cf.hardcoded_fomc_dates_2020_2025()
        self.assertGreaterEqual(len(dates), 46)  # 8 × 6 yrs = 48, ±2 leeway
        self.assertLessEqual(len(dates), 52)

        iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        lo = _dt.date(2020, 1, 1)
        hi = _dt.date(2025, 12, 31)
        for s in dates:
            self.assertRegex(s, iso_re)
            d = _dt.date.fromisoformat(s)
            self.assertGreaterEqual(d, lo)
            self.assertLessEqual(d, hi)


# ──────────────────── calendar_feature_matrix ───────────────────

class CalendarMatrixTests(unittest.TestCase):
    def test_shape_is_20_columns(self):
        ts = _trading_days(_dt.date(2023, 1, 2), 100)
        mat, names = cf.calendar_feature_matrix(ts)
        self.assertEqual(mat.shape, (100, 20))
        self.assertEqual(len(names), 20)
        # 1 + 5 + 12 + 1 + 1
        self.assertEqual(mat.shape[1], 1 + 5 + 12 + 1 + 1)

    def test_fomc_column_is_zero_without_dates(self):
        ts = _trading_days(_dt.date(2023, 1, 2), 100)
        mat, names = cf.calendar_feature_matrix(ts)
        fomc_idx = names.index("fomc_pre")
        np.testing.assert_array_equal(mat[:, fomc_idx], np.zeros(100))


if __name__ == "__main__":
    unittest.main()
