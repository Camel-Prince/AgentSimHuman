#!/bin/bash
# 每隔2小时自动同步 wandb offline 数据
# 用法: nohup bash scripts/wandb_sync_loop.sh &

cd "$(dirname "$0")/.."

while true; do
    echo "[$(date)] Syncing wandb offline runs..."
    wandb sync wandb/offline-run-* --append 2>&1 | tail -5
    echo "[$(date)] Sync done. Sleeping 2h..."
    sleep 7200
done
