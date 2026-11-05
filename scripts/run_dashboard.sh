#!/usr/bin/env bash
# Launch the G1Nav dashboards (needs `pip install -e ".[dashboard]"`, done by scripts/install.sh).
#   8501  g1nav.apps.results_dashboard     — replay recorded results (always works offline)
#   8502  g1nav.apps.interactive_dashboard — drive the pipeline yourself (local GPU or remote machine)
#
#   bash scripts/run_dashboard.sh               # both
#   bash scripts/run_dashboard.sh interactive   # only 8502
#   bash scripts/run_dashboard.sh results       # only 8501
set -uo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate 2>/dev/null || true

WHICH="${1:-both}"
COMMON="--server.headless true --browser.gatherUsageStats false"
LOG_DIR="${G1NAV_LOG_DIR:-/tmp}"

if [ "$WHICH" = "both" ] || [ "$WHICH" = "results" ]; then
  echo "[dash] results dashboard  -> http://localhost:8501"
  streamlit run src/g1nav/apps/results_dashboard.py --server.port 8501 $COMMON \
    > "$LOG_DIR/g1nav_results.log" 2>&1 &
fi
if [ "$WHICH" = "both" ] || [ "$WHICH" = "interactive" ]; then
  echo "[dash] interactive (run it yourself) -> http://localhost:8502"
  streamlit run src/g1nav/apps/interactive_dashboard.py --server.port 8502 $COMMON \
    > "$LOG_DIR/g1nav_interactive.log" 2>&1 &
fi

echo "[dash] launched. Logs: $LOG_DIR/g1nav_results.log  $LOG_DIR/g1nav_interactive.log"
echo "[dash] stop with:  pkill -f 'streamlit run'"
wait
