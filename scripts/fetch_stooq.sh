#!/usr/bin/env bash
# Fetch daily OHLCV bars from Stooq.com (no API key required) for a set of
# US tickers and convert them into our bars-CSV format.
#
# Usage:
#   scripts/fetch_stooq.sh SPY QQQ AAPL MSFT
#   scripts/fetch_stooq.sh                # default basket
#
# Output: data/real/<TICKER>.csv with columns
#   ts_nanos,symbol,open,high,low,close,volume,span_secs
#
# Each row is one trading day's daily bar (span_secs = 86400).

set -euo pipefail

if [[ $# -eq 0 ]]; then
  TICKERS=(SPY QQQ DIA IWM XLK XLF XLE XLV TLT GLD HYG)
else
  TICKERS=("$@")
fi

OUT_DIR="data/real"
mkdir -p "$OUT_DIR"

for raw in "${TICKERS[@]}"; do
  TICK_UPPER=$(echo "$raw" | tr '[:lower:]' '[:upper:]')
  # Stooq's US symbols carry a .us suffix.
  STOOQ_SYM=$(echo "$TICK_UPPER" | tr '[:upper:]' '[:lower:]').us
  URL="https://stooq.com/q/d/l/?s=${STOOQ_SYM}&i=d"
  echo "↓ ${TICK_UPPER} ← ${URL}"
  TMP=$(mktemp)
  if ! curl -fsSL "$URL" -o "$TMP"; then
    echo "  ⚠ download failed for ${TICK_UPPER}, skipping"
    rm -f "$TMP"
    continue
  fi
  # Stooq CSV header: Date,Open,High,Low,Close,Volume
  # Convert to our bars-CSV format.
  OUT="${OUT_DIR}/${TICK_UPPER}.csv"
  awk -v sym="$TICK_UPPER" -F, '
    NR == 1 { print "ts_nanos,symbol,open,high,low,close,volume,span_secs"; next }
    NF >= 6 {
      # Convert YYYY-MM-DD to seconds since epoch via system date.
      cmd = "date -j -f \"%Y-%m-%d\" \"" $1 "\" \"+%s\" 2>/dev/null || date -d \"" $1 "\" \"+%s\"";
      cmd | getline secs; close(cmd);
      ns = secs * 1000000000;
      printf "%s,%s,%s,%s,%s,%s,%s,86400\n", ns, sym, $2, $3, $4, $5, $6;
    }
  ' "$TMP" > "$OUT"
  rm -f "$TMP"
  ROWS=$(wc -l < "$OUT")
  echo "  ✓ wrote ${OUT} (${ROWS} rows)"
done

echo ""
echo "Done. To merge multiple symbols into one universe CSV (sorted by ts):"
echo "  python3 -c \"import csv,glob,sys; rows=[]; [rows.extend(list(csv.reader(open(p)))[1:]) for p in glob.glob('${OUT_DIR}/*.csv')]; rows.sort(key=lambda r:int(r[0])); w=csv.writer(sys.stdout); w.writerow(['ts_nanos','symbol','open','high','low','close','volume','span_secs']); [w.writerow(r) for r in rows]\" > ${OUT_DIR}/universe.csv"
