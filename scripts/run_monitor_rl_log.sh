#!/usr/bin/env bash
set -euo pipefail

LOG_PATH_DEFAULT="/home/wangzixu/Search-R1/paper-writing-grpo-qwen2_5-3B-instruct-arxiv-writing.log"
LOG_PATH="${1:-$LOG_PATH_DEFAULT}"
WINDOW="${2:-20}"
REFRESH_SEC="${3:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[monitor] log: $LOG_PATH"
echo "[monitor] window: $WINDOW"
echo "[monitor] refresh_sec: $REFRESH_SEC"

python "$PROJECT_ROOT/scripts/monitor_rl_log.py" \
  --log "$LOG_PATH" \
  --window "$WINDOW" \
  --follow \
  --refresh-sec "$REFRESH_SEC"

