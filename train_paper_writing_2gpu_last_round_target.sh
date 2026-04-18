#!/usr/bin/env bash
set -euo pipefail

# 兼容入口：保留旧脚本名，内部统一转调到 train_paper_writing_2gpu.sh
# 推荐新用法：
#   TASK_TYPE=paper_writing_last_round_target EXPERIMENT_TAG=last_round_target \
#   RUN_ID=1 CUDA_VISIBLE_DEVICES=5,6 bash train_paper_writing_2gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 旧脚本默认语义：last_round_target + 5,6 卡
export TASK_TYPE="${TASK_TYPE:-paper_writing_last_round_target}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-last_round_target}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6}"
# 为并发默认预留不同 RUN_ID，避免与主脚本默认 RUN_ID=0 冲突
export RUN_ID="${RUN_ID:-1}"

exec bash "${SCRIPT_DIR}/train_paper_writing_2gpu.sh" "$@"
