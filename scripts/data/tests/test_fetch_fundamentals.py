"""Known-truth tests for scripts/data/fetch_fundamentals.py.

All tests mock yfinance — no network calls.
"""
from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fetch_fundamentals as ff  # noqa: E402
from _common import MACRO_HEADER  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic Ticker stubs used to monkeypatch yfinance.Ticker
# ---------------------------------------------------------------------------

class _StubTicker:
    def __init__(self, info_dict):
        self._info = info_dict

    @property
    def info(self):
        return self._info


def _full_info():
    """A complete .info dict so every one of the 12 features is computable."""
    return {
        # value
        "trailingPE": 25.0,           # earnings_yield = 1/25 = 0.04
        "enterpriseValue": 2_000_000,
        "ebitda": 100_000,            # ev_to_ebitda = 20.0
        "bookValue": 50.0,
        "marketCap": 1_000_000,       # book_to_market = 5.0e-5; log_mcap = ln(1e6)
        # quality
        "returnOnEquity": 0.30,
        "returnOnAssets": 0.15,
        "grossMargins": 0.55,
        "currentRatio": 1.8,
        # growth
        "revenueGrowth": 0.12,
        "earningsGrowth": 0.20,
        # profitability
        "operatingMargins": 0.25,
        "profitMargins": 0.18,
        # leverage (reported as %)
        "debtToEquity": 60.0,         # → 0.60
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class FetchFundamentalsHappyPathTests(unittest.TestCase):
    def test_full_info_computes_all_twelve_features(self):
        info = _full_info()
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            with patch.object(ff.yf, "Ticker",
                              return_value=_StubTicker(info)):
                feats = ff.fetch_fundamentals("AAPL", out_dir)

            expected_names = {
                "earnings_yield", "ev_to_ebitda", "book_to_market",
                "roe", "roa", "gross_margin", "current_ratio",
                "revenue_growth_yoy", "earnings_growth_yoy",
                "operating_margin", "profit_margin",
                "debt_to_equity", "log_market_cap",
            }
            self.assertEqual(set(feats.keys()), expected_names)

            # Spot-check the math.
            self.assertAlmostEqual(feats["earnings_yield"], 1.0 / 25.0, places=10)
            self.assertAlmostEqual(feats["ev_to_ebitda"], 20.0, places=10)
            self.assertAlmostEqual(feats["book_to_market"], 50.0 / 1_000_000, places=12)
            self.assertAlmostEqual(feats["roe"], 0.30, places=12)
            self.assertAlmostEqual(feats["debt_to_equity"], 0.60, places=12)
            self.assertAlmostEqual(feats["log_market_cap"], math.log(1_000_000), places=10)

            # File written in macro schema.
            csv_path = out_dir / "AAPL_features.csv"
            self.assertTrue(csv_path.exists())
            with open(csv_path) as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], MACRO_HEADER)
            data_rows = rows[1:]
            self.assertEqual(len(data_rows), len(expected_names))
            for row in data_rows:
                self.assertEqual(len(row), 3)
                # ts_nanos is an int, series is a string, value is a float
                int(row[0])
                self.assertIn(row[1], expected_names)
                float(row[2])

    def test_missing_fields_returns_empty_and_writes_no_file(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            with patch.object(ff.yf, "Ticker",
                              return_value=_StubTicker({})):
                feats = ff.fetch_fundamentals("SPY", out_dir)
            self.assertEqual(feats, {})
            self.assertFalse((out_dir / "SPY_features.csv").exists())

    def test_etf_ticker_raises_returns_empty_cleanly(self):
        def _boom(_sym):
            raise RuntimeError("no fundamentals for ETF")
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            with patch.object(ff.yf, "Ticker", side_effect=_boom):
                feats = ff.fetch_fundamentals("SPY", out_dir)
            self.assertEqual(feats, {})
            self.assertFalse((out_dir / "SPY_features.csv").exists())

    def test_partial_info_yields_only_computable_features(self):
        # Only ROE + a non-finite (None) bookValue and missing marketCap →
        # only roe survives.  book_to_market needs both bookValue and
        # marketCap; log_market_cap needs marketCap; neither should appear.
        info = {"returnOnEquity": 0.25, "bookValue": None}
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            with patch.object(ff.yf, "Ticker",
                              return_value=_StubTicker(info)):
                feats = ff.fetch_fundamentals("XYZ", out_dir)
            self.assertEqual(set(feats.keys()), {"roe"})
            self.assertAlmostEqual(feats["roe"], 0.25, places=12)


class UniverseFactorTableTests(unittest.TestCase):
    def _write_features(self, out_dir: Path, sym: str,
                        feats: dict[str, float]) -> None:
        ff._write_features_csv(out_dir / f"{sym}_features.csv", feats)

    def test_median_symbol_gets_zero_z_score(self):
        """Build a 5-symbol cross-section. The middle symbol holds the exact
        median of every feature → robust z must be 0 for it on every feature."""
        symbols = ["A", "B", "C", "D", "E"]
        # Two features, increasing across the universe so 'C' is the median.
        # roe: 0.10, 0.15, 0.20, 0.25, 0.30  → median = 0.20 (C)
        # gross_margin: 0.40, 0.45, 0.50, 0.55, 0.60 → median = 0.50 (C)
        roe_vals = [0.10, 0.15, 0.20, 0.25, 0.30]
        gm_vals = [0.40, 0.45, 0.50, 0.55, 0.60]
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            for sym, r, g in zip(symbols, roe_vals, gm_vals):
                self._write_features(out_dir, sym,
                                     {"roe": r, "gross_margin": g})
            table = ff.universe_factor_table(symbols, out_dir)

        # C is the median symbol → z ≈ 0 on every feature.
        self.assertAlmostEqual(table["C"]["roe"], 0.0, places=12)
        self.assertAlmostEqual(table["C"]["gross_margin"], 0.0, places=12)

        # And the symmetric pair A/E should have opposite-sign z-scores of
        # equal magnitude.
        self.assertAlmostEqual(table["A"]["roe"], -table["E"]["roe"], places=12)
        self.assertAlmostEqual(table["A"]["gross_margin"],
                               -table["E"]["gross_margin"], places=12)
        self.assertLess(table["A"]["roe"], 0.0)
        self.assertGreater(table["E"]["roe"], 0.0)

    def test_mad_zero_yields_zero_z_no_division_error(self):
        # All symbols identical → mad = 0; we must return 0, not blow up.
        symbols = ["X", "Y", "Z"]
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            for s in symbols:
                self._write_features(out_dir, s, {"roe": 0.20})
            table = ff.universe_factor_table(symbols, out_dir)
        for s in symbols:
            self.assertEqual(table[s]["roe"], 0.0)

    def test_missing_feature_for_one_symbol_skipped(self):
        # B lacks gross_margin → it should be absent from B's z-row but the
        # other two still compute fine.
        symbols = ["A", "B", "C"]
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            self._write_features(out_dir, "A", {"roe": 0.10, "gross_margin": 0.40})
            self._write_features(out_dir, "B", {"roe": 0.20})
            self._write_features(out_dir, "C", {"roe": 0.30, "gross_margin": 0.60})
            table = ff.universe_factor_table(symbols, out_dir)
        self.assertIn("roe", table["B"])
        self.assertNotIn("gross_margin", table["B"])
        self.assertIn("gross_margin", table["A"])
        self.assertIn("gross_margin", table["C"])


class CompositeScoreTests(unittest.TestCase):
    def test_hand_verified_three_symbol_case(self):
        """Use a degenerate-but-controllable cross-section to hand-verify the
        composite-score arithmetic:

        We hand-craft features so that, for each of the three buckets, the
        z-scores fall at exactly -0.6745, 0, +0.6745 (the classic MAD-z values
        for the 25th/50th/75th percentiles in a 3-point sample with the
        Normal-consistent scaling we use).

        Specifically, for any three values v_low < v_med < v_high with
        symmetric spacing, the MAD = (v_high - v_low)/2 and
            z = 0.6745 * (x - v_med) / MAD
        yields z(v_low) = -0.6745, z(v_med) = 0, z(v_high) = +0.6745.
        """
        # We give each symbol exactly ONE value-bucket feature, ONE quality
        # feature and ONE growth feature, all in the "low/med/high" pattern.
        symbols = ["LOW", "MID", "HIGH"]
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            ff._write_features_csv(out_dir / "LOW_features.csv", {
                "earnings_yield": 0.02,
                "roe": 0.05,
                "revenue_growth_yoy": 0.01,
            })
            ff._write_features_csv(out_dir / "MID_features.csv", {
                "earnings_yield": 0.05,
                "roe": 0.15,
                "revenue_growth_yoy": 0.10,
            })
            ff._write_features_csv(out_dir / "HIGH_features.csv", {
                "earnings_yield": 0.08,
                "roe": 0.25,
                "revenue_growth_yoy": 0.19,
            })

            # First sanity-check the z-table.
            table = ff.universe_factor_table(symbols, out_dir)
            for sym, sign in [("LOW", -1.0), ("MID", 0.0), ("HIGH", +1.0)]:
                expected = sign * 0.6745
                self.assertAlmostEqual(table[sym]["earnings_yield"],
                                       expected, places=10)
                self.assertAlmostEqual(table[sym]["roe"],
                                       expected, places=10)
                self.assertAlmostEqual(table[sym]["revenue_growth_yoy"],
                                       expected, places=10)

            comp = ff.composite_score(symbols, out_dir)

        # Each symbol's value_z = +z(earnings_yield)  (only value feature
        # present, and earnings_yield has sign +1 in VALUE_SIGNS).
        # quality_z = z(roe); growth_z = z(revenue_growth_yoy).
        # Composite = 0.4 * value_z + 0.4 * quality_z + 0.2 * growth_z.
        for sym, sign in [("LOW", -1.0), ("MID", 0.0), ("HIGH", +1.0)]:
            z = sign * 0.6745
            expected = 0.4 * z + 0.4 * z + 0.2 * z  # = z
            self.assertAlmostEqual(comp[sym], expected, places=10)

    def test_ev_to_ebitda_is_negated_in_value_bucket(self):
        """Higher ev_to_ebitda is *worse*; verify the composite reflects
        that by negating its z-score."""
        symbols = ["CHEAP", "MID", "EXPENSIVE"]
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            # Only the ev_to_ebitda value feature is present; cheap < mid < expensive.
            ff._write_features_csv(out_dir / "CHEAP_features.csv",
                                   {"ev_to_ebitda": 5.0})
            ff._write_features_csv(out_dir / "MID_features.csv",
                                   {"ev_to_ebitda": 10.0})
            ff._write_features_csv(out_dir / "EXPENSIVE_features.csv",
                                   {"ev_to_ebitda": 20.0})
            comp = ff.composite_score(symbols, out_dir)
        # CHEAP should have a higher composite than EXPENSIVE because lower
        # EV/EBITDA = better.
        self.assertGreater(comp["CHEAP"], comp["EXPENSIVE"])


if __name__ == "__main__":
    unittest.main()
