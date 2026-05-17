"""Tests for scripts/data/_common.py — basic helpers + serialization."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Make the parent dir importable so we can `from _common import ...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import (
    BAR_HEADER,
    MACRO_HEADER,
    UNIVERSE_HEADER,
    date_to_ns,
    write_bar_csv,
    write_macro_csv,
    write_universe_csv,
    emit_manifest,
)


class DateToNsTests(unittest.TestCase):
    def test_epoch_is_zero(self):
        self.assertEqual(date_to_ns("1970-01-01"), 0)

    def test_known_date(self):
        # 2025-01-01 00:00:00 UTC = 1_735_689_600 unix seconds
        self.assertEqual(date_to_ns("2025-01-01"), 1_735_689_600 * 1_000_000_000)

    def test_invalid_date_raises(self):
        with self.assertRaises(ValueError):
            date_to_ns("not-a-date")


class CsvWriteRoundtripTests(unittest.TestCase):
    def test_bar_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bars.csv"
            rows = [
                [1_700_000_000_000_000_000, "AAPL", "150.0", "151.0", "149.5", "150.5", "1000", 60],
                [1_700_000_060_000_000_000, "AAPL", "150.5", "151.5", "150.0", "151.0", "1100", 60],
            ]
            n = write_bar_csv(path, rows)
            self.assertEqual(n, 2)
            with open(path) as f:
                r = csv.reader(f)
                header = next(r)
                self.assertEqual(header, BAR_HEADER)
                read_rows = list(r)
                self.assertEqual(len(read_rows), 2)
                self.assertEqual(read_rows[0][1], "AAPL")

    def test_macro_csv_writes_header(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "macro.csv"
            write_macro_csv(path, [[1, "VIXCLS", "17.5"], [2, "VIXCLS", "18.0"]])
            with open(path) as f:
                header = next(csv.reader(f))
                self.assertEqual(header, MACRO_HEADER)

    def test_universe_csv_writes_header(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "u.csv"
            write_universe_csv(path, [["AAPL", "Apple", "Technology", ""]])
            with open(path) as f:
                header = next(csv.reader(f))
                self.assertEqual(header, UNIVERSE_HEADER)


class ManifestTests(unittest.TestCase):
    def test_emit_manifest_round_trips(self):
        # emit_manifest writes to a relative path; chdir to a temp dir for isolation.
        import os
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            try:
                entries = [{"series": "VIXCLS", "rows": 100, "path": "data/macro/VIXCLS.csv"}]
                path = emit_manifest("fred", entries)
                self.assertTrue(path.exists())
                with open(path) as f:
                    data = json.load(f)
                self.assertEqual(data["source"], "fred")
                self.assertEqual(data["entries"], entries)
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
