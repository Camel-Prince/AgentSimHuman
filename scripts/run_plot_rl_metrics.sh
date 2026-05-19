#!/usr/bin/env bash
set -euo pipefail

LOG_PATH_DEFAULT="/home/wangzixu/Search-R1/logs/paper-writing-grpo-qwen2_5-3B-instruct-arxiv-writing-20260416_234503-run0.log"
OUT_DIR_DEFAULT="/home/wangzixu/Search-R1/monitor_outputs"

LOG_PATH="${1:-$LOG_PATH_DEFAULT}"
OUT_DIR="${2:-$OUT_DIR_DEFAULT}"
MODE="${3:-key}"  # key | all

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="/home/wangzixu/anaconda3/envs/searchr1/bin/python"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"

echo "[plot] log: $LOG_PATH"
echo "[plot] out_dir: $OUT_DIR"
echo "[plot] mode: $MODE"

if [[ "$MODE" == "all" ]]; then
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/plot_rl_metrics.py" --log "$LOG_PATH" --out-dir "$OUT_DIR" --all-metrics
else
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/plot_rl_metrics.py" --log "$LOG_PATH" --out-dir "$OUT_DIR"
fi
