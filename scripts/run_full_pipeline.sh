#!/usr/bin/env bash
set -euo pipefail

# Run from repo root regardless of where this script is invoked
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "${REPO_ROOT}"

# Defaults
CLEAN_RESULTS=false
REBUILD_DATA=false
LAGS="0 1 2"
GRID_N=5
TRAIN_START="2011-01-01"
TRAIN_END="2017-12-31"
TEST_START="2018-01-01"
TEST_END="2022-12-31"
ENTRY_RANGE="1.5:2.5"
EXIT_RANGE="0.25:0.75"
STOP_RANGE="2.5:3.5"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --clean                 Remove results/ before running
  --rebuild-data          Remove data/processed and data/derived before running
  --lags "0 1 2"          Space-separated lags for POWER correlations (default: "${LAGS}")
  --grid-n N              Grid size for tuning per range (default: ${GRID_N})
  --train-start DATE      Train start (default: ${TRAIN_START})
  --train-end DATE        Train end (default: ${TRAIN_END})
  --test-start DATE       Test start (default: ${TEST_START})
  --test-end DATE         Test end (default: ${TEST_END})
  --entry-range A:B       Entry z range (default: ${ENTRY_RANGE})
  --exit-range A:B        Exit z range (default: ${EXIT_RANGE})
  --stop-range A:B        Stop z range (default: ${STOP_RANGE})
  -h, --help              Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN_RESULTS=true; shift ;;
    --rebuild-data) REBUILD_DATA=true; shift ;;
    --lags) LAGS="$2"; shift 2 ;;
    --grid-n) GRID_N="$2"; shift 2 ;;
    --train-start) TRAIN_START="$2"; shift 2 ;;
    --train-end) TRAIN_END="$2"; shift 2 ;;
    --test-start) TEST_START="$2"; shift 2 ;;
    --test-end) TEST_END="$2"; shift 2 ;;
    --entry-range) ENTRY_RANGE="$2"; shift 2 ;;
    --exit-range) EXIT_RANGE="$2"; shift 2 ;;
    --stop-range) STOP_RANGE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

echo "[1/9] Cleaning (optional)";
if [[ "$CLEAN_RESULTS" == true ]]; then
  rm -rf results || true
  echo " - removed results/"
fi
if [[ "$REBUILD_DATA" == true ]]; then
  rm -rf data/processed data/derived || true
  echo " - removed data/processed and data/derived"
fi
 # Always fetch fresh FX in build step (handled by code)

echo "[2/9] Build datasets + FX cache"
python scripts/build_datasets.py

echo "[3/9] Build POWER point panel"
python scripts/build_power_point_panel.py

echo "[4/9] POWER correlations (lags: ${LAGS})"
python scripts/eval_power_point_correlations.py --lags ${LAGS}

echo "[5/9] Rain signals evaluation"
python scripts/eval_rain_signals.py --lag 15 --horizon 20

echo "[6/9] Tune backtest parameters"
python scripts/tune_backtest_params.py \
  --entry-range "${ENTRY_RANGE}" \
  --exit-range "${EXIT_RANGE}" \
  --stop-range "${STOP_RANGE}" \
  --grid-n ${GRID_N} \
  --train-start ${TRAIN_START} \
  --train-end ${TRAIN_END} \
  --test-start ${TEST_START} \
  --test-end ${TEST_END}

echo "[7/9] Run backtests (standard + causal)"
python scripts/run_backtest.py --compare-causal \
  --train-start ${TRAIN_START} --train-end ${TRAIN_END} \
  --test-start ${TEST_START} --test-end ${TEST_END}

echo "[8/9] Run backtests with tilting (standard + causal)"
python scripts/run_backtest.py --compare-causal --tilt \
  --train-start ${TRAIN_START} --train-end ${TRAIN_END} \
  --test-start ${TEST_START} --test-end ${TEST_END}

echo "[9/9] Compare metrics + make report"
python scripts/compare_backtest_metrics.py
python scripts/make_report.py

echo "Done. Key outputs:"
echo " - results/tables/backtest_compare.csv"
echo " - results/backtests/standard/base/{metrics,equity,pnl,positions}.csv"
echo " - results/backtests/causal/base/{metrics,equity,pnl,positions}.csv"
echo " - results/backtests/standard/tilted/{metrics,equity,pnl,positions}.csv (if --tilt ran)"
echo " - results/backtests/causal/tilted/{metrics,equity,pnl,positions}.csv (if --tilt ran)"
echo " - results/REPORT.md"
