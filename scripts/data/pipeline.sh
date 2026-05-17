#!/usr/bin/env bash
# End-to-end research data pipeline.
#
# Steps (each best-effort; failures are logged, never aborts the pipeline):
#   1. Yahoo Finance        equity bars
#   2. FRED                 macro indicators
#   3. Fama-French          factor returns
#   4. CoinGecko            daily crypto bars
#   5. Binance              high-res (1h) crypto bars
#   6. Wikipedia            constituent lists (SP500, NDX, DJI)
#   7. SEC EDGAR            10-K/10-Q metadata
#   8. CFTC COT             weekly futures positioning
#   9. EIA                  energy series (oil, gas)
#  10. US Treasury          daily yield curve
#  11. NOAA GSOD            weather time series
#  12. World Bank           country indicators
#  13. Wikipedia pageviews  search-interest proxies
#  14. AAII                 investor sentiment survey (best effort)
#  15. GDELT                global news/events (last 14 days)
#  16. validate (autofix)
#  17. aggregate            per-class universe.csv + inventory.json
#  18. catalog              SQLite catalog of every dataset
#
# Re-running is idempotent; manifests append, downloads overwrite.
#
# Usage:
#   scripts/data/pipeline.sh
#   YEARS=5 scripts/data/pipeline.sh
#   SKIP=binance,gdelt scripts/data/pipeline.sh

set -uo pipefail

YEARS=${YEARS:-2}
SKIP=${SKIP:-}

cd "$(dirname "$0")/../.."

skip_step() {
  [[ ",$SKIP," == *",$1,"* ]]
}

run_step() {
  local name="$1"
  shift
  if skip_step "$name"; then
    echo "═══ skip: $name"
    return 0
  fi
  echo ""
  echo "═══ $name"
  "$@" || echo "  ⚠ $name failed (continuing)"
}

run_step yahoo    python3 scripts/data/fetch_yahoo.py --years "$YEARS" --merge \
                    SPY QQQ DIA IWM XLK XLF XLE XLV TLT GLD HYG VTI
run_step fred     python3 scripts/data/fetch_fred.py
run_step ff       python3 scripts/data/fetch_famafrench.py
run_step crypto   python3 scripts/data/fetch_coingecko.py
run_step binance  python3 scripts/data/fetch_binance.py --interval 1h --bars 2000
run_step wiki     python3 scripts/data/fetch_wikipedia.py
run_step edgar    python3 scripts/data/fetch_sec_edgar.py
run_step cot      python3 scripts/data/fetch_cftc_cot.py 2024
run_step eia      python3 scripts/data/fetch_eia.py
run_step yields   python3 scripts/data/fetch_treasury_yields.py "$(date +%Y)"
run_step noaa     python3 scripts/data/fetch_noaa.py "$(date +%Y)"
run_step wb       python3 scripts/data/fetch_worldbank.py
run_step pvw      python3 scripts/data/fetch_wiki_pageviews.py
run_step aaii     python3 scripts/data/fetch_aaii.py
run_step gdelt    python3 scripts/data/fetch_gdelt.py 7

echo ""
echo "═══ validate (autofix)"
python3 scripts/data/validate.py --autofix --root data

echo ""
echo "═══ aggregate"
python3 scripts/data/aggregate.py

echo ""
echo "═══ catalog"
python3 scripts/data/catalog.py

echo ""
echo "═══ done"
echo ""
echo "Browse:"
echo "  sqlite3 data/catalog.sqlite 'SELECT source, asset_class, COUNT(*), SUM(n_rows) FROM datasets GROUP BY source, asset_class'"
echo ""
echo "Extract features for any series:"
echo "  python3 scripts/data/signal_extract.py data/macro/VIXCLS.csv"
echo "  python3 scripts/data/signal_extract.py data/real/SPY.csv --out features/spy.csv"
