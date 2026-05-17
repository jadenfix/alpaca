#!/usr/bin/env bash
# Run all Python tests under scripts/data/tests/ using stdlib unittest.
# No third-party deps required (no pytest).
set -e
cd "$(dirname "$0")/../../.."
python3 -m unittest discover -s scripts/data/tests -p 'test_*.py' -v
