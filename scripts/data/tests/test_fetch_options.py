"""Known-truth tests for scripts/data/fetch_options.py.

No tests touch the network — yfinance.Ticker is replaced with a stub via
``unittest.mock.patch`` so the fetcher operates on hand-built chains.
"""
from __future__ import annotations

import csv
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# Add scripts/data to sys.path (same pattern test_walkforward.py uses
# for backtest/analysis dirs).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import fetch_options  # noqa: E402
from fetch_options import (  # noqa: E402
    black_scholes_gamma,
    compute_features,
    fetch_options_for,
    write_features_csv,
)


def _make_chain(rows):
    """Build a yfinance-shaped DataFrame from a list of dicts."""
    return pd.DataFrame(
        rows,
        columns=["strike", "impliedVolatility", "volume", "openInterest"],
    )


class BlackScholesGammaTests(unittest.TestCase):
    def test_atm_hand_calculation(self):
        # S=K=100, sigma=20%, r=5%, q=0, T=0.25  -> gamma ~ 0.0393
        # Spec gives "~ 0.0398 within 1%"; allow a small absolute tolerance
        # which admits both the textbook value and the spec's rounded one.
        g = black_scholes_gamma(S=100, K=100, sigma=0.20, T=0.25, r=0.05, q=0.0)
        self.assertAlmostEqual(g, 0.0398, delta=0.001)

    def test_degenerate_inputs_return_zero(self):
        self.assertEqual(black_scholes_gamma(100, 100, 0.0, 0.25), 0.0)
        self.assertEqual(black_scholes_gamma(100, 100, 0.2, 0.0), 0.0)
        self.assertEqual(black_scholes_gamma(0.0, 100, 0.2, 0.25), 0.0)


class FeatureExtractionTests(unittest.TestCase):
    def test_iv_skew_25d_positive_when_puts_richer(self):
        spot = 100.0
        # Puts skewed high (rich downside vol) -> skew > 0.
        front_calls = [
            {"strike": 100.0, "iv": 0.20, "volume": 10, "open_interest": 10},
            {"strike": 108.0, "iv": 0.18, "volume": 10, "open_interest": 10},
        ]
        front_puts = [
            {"strike": 100.0, "iv": 0.22, "volume": 10, "open_interest": 10},
            {"strike": 92.0, "iv": 0.30, "volume": 10, "open_interest": 10},
        ]
        feats = compute_features(
            spot=spot,
            front_calls=front_calls,
            front_puts=front_puts,
            next_calls=[],
            next_puts=[],
            front_dte_years=30 / 365.0,
        )
        self.assertIn("iv_skew_25d", feats)
        self.assertGreater(feats["iv_skew_25d"], 0.0)
        # IV(put 92) 0.30  -  IV(call 108) 0.18  =  0.12
        self.assertAlmostEqual(feats["iv_skew_25d"], 0.12, places=6)

    def test_put_call_volume_ratio_exact(self):
        # sum(put vol) = 200, sum(call vol) = 100  -> ratio = 2.0
        front_calls = [
            {"strike": 100.0, "iv": 0.20, "volume": 40, "open_interest": 5},
            {"strike": 105.0, "iv": 0.20, "volume": 60, "open_interest": 5},
        ]
        front_puts = [
            {"strike": 100.0, "iv": 0.20, "volume": 80, "open_interest": 5},
            {"strike": 95.0, "iv": 0.20, "volume": 120, "open_interest": 5},
        ]
        feats = compute_features(
            spot=100.0,
            front_calls=front_calls,
            front_puts=front_puts,
            next_calls=[],
            next_puts=[],
            front_dte_years=30 / 365.0,
        )
        self.assertIn("put_call_volume_ratio", feats)
        self.assertAlmostEqual(feats["put_call_volume_ratio"], 2.0, places=9)


class FetchOptionsForTests(unittest.TestCase):
    def test_no_options_symbol_returns_empty_dict(self):
        """A FRED-style symbol with `Ticker.options == ()` must yield {}."""

        class _StubTicker:
            options = ()  # empty tuple = no listed expiries

            def option_chain(self, expiry):  # pragma: no cover - never called
                raise AssertionError("option_chain should not be called")

        fake_yf = types.SimpleNamespace(Ticker=lambda sym: _StubTicker())
        with patch.dict(sys.modules, {"yfinance": fake_yf}):
            with tempfile.TemporaryDirectory() as td:
                out = fetch_options_for("FRED_SERIES", Path(td))
        self.assertEqual(out, {})

    def test_full_fetch_writes_macro_rows(self):
        """End-to-end: stub yfinance with a synthetic chain and check that
        fetch_options_for writes macro-schema rows for the computed features."""
        spot = 100.0
        # Front expiry 14 days out, next expiry 45 days out.
        front_dt = datetime.now(timezone.utc) + timedelta(days=14)
        next_dt = datetime.now(timezone.utc) + timedelta(days=45)
        front_expiry = front_dt.strftime("%Y-%m-%d")
        next_expiry = next_dt.strftime("%Y-%m-%d")

        front_calls = _make_chain([
            {"strike": 100.0, "impliedVolatility": 0.20, "volume": 50, "openInterest": 100},
            {"strike": 108.0, "impliedVolatility": 0.18, "volume": 50, "openInterest": 100},
        ])
        front_puts = _make_chain([
            {"strike": 100.0, "impliedVolatility": 0.22, "volume": 100, "openInterest": 200},
            {"strike": 92.0,  "impliedVolatility": 0.30, "volume": 100, "openInterest": 200},
        ])
        next_calls = _make_chain([
            {"strike": 100.0, "impliedVolatility": 0.21, "volume": 10, "openInterest": 10},
        ])
        next_puts = _make_chain([
            {"strike": 100.0, "impliedVolatility": 0.23, "volume": 10, "openInterest": 10},
        ])

        chains = {
            front_expiry: types.SimpleNamespace(calls=front_calls, puts=front_puts),
            next_expiry: types.SimpleNamespace(calls=next_calls, puts=next_puts),
        }

        class _StubTicker:
            options = (front_expiry, next_expiry)
            fast_info = {"last_price": spot}

            def option_chain(self, expiry):
                return chains[expiry]

        fake_yf = types.SimpleNamespace(Ticker=lambda sym: _StubTicker())
        with patch.dict(sys.modules, {"yfinance": fake_yf}):
            with tempfile.TemporaryDirectory() as td:
                feats = fetch_options_for("XYZ", Path(td))
                csv_path = Path(td) / "XYZ_options_summary.csv"
                self.assertTrue(csv_path.exists())
                with open(csv_path) as f:
                    rows = list(csv.reader(f))

        self.assertGreater(len(feats), 0)
        # Header is intact and series labels are namespaced by symbol.
        self.assertEqual(rows[0], ["ts_nanos", "series", "value"])
        for r in rows[1:]:
            self.assertEqual(len(r), 3)
            self.assertTrue(r[1].startswith("XYZ_"))
        # Term-structure slope should be present (we provided a next expiry).
        self.assertIn("term_structure_slope", feats)


class WriteFeaturesCsvTests(unittest.TestCase):
    def test_macro_header_is_written(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            write_features_csv(
                out_dir, "TEST", ts_nanos=1_700_000_000_000_000_000,
                features={"iv_atm": 0.21, "put_call_oi_ratio": 1.5},
            )
            path = out_dir / "TEST_options_summary.csv"
            self.assertTrue(path.exists())
            with open(path) as f:
                content = f.read()
        first_line = content.splitlines()[0]
        self.assertEqual(first_line, "ts_nanos,series,value")
        # Two feature rows written; both should reference the TEST_ prefix.
        rows = content.splitlines()[1:]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r.split(",")[1].startswith("TEST_") for r in rows))

    def test_append_mode_does_not_duplicate_header(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            write_features_csv(out_dir, "AA", 1, {"iv_atm": 0.2})
            write_features_csv(out_dir, "AA", 2, {"iv_atm": 0.3})
            with open(out_dir / "AA_options_summary.csv") as f:
                lines = f.read().splitlines()
        # 1 header + 2 data rows.
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "ts_nanos,series,value")


if __name__ == "__main__":
    unittest.main()
