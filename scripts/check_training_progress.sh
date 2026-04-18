#!/bin/bash
# Quick training progress checker

LOG_FILE="paper-writing-grpo-qwen2_5-7b-instruct-debug.log"
CKPT_DIR="verl_checkpoints/paper-writing-grpo-qwen2_5-7b-instruct-debug"

echo "=========================================="
echo "Training Progress Summary"
echo "=========================================="
echo ""

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

# Extract latest step/epoch
echo "📊 Latest Progress:"
grep -E "step|epoch" "$LOG_FILE" | tail -5
echo ""

# Extract latest reward
echo "🎯 Latest Rewards:"
grep -i "reward\|score" "$LOG_FILE" | tail -5
echo ""

# Extract latest loss
echo "📉 Latest Loss:"
grep -i "loss" "$LOG_FILE" | tail -5
echo ""

# Check checkpoints
echo "💾 Saved Checkpoints:"
if [ -d "$CKPT_DIR" ]; then
    ls -lh "$CKPT_DIR" | grep "checkpoint" | tail -5
    echo ""
    echo "Total checkpoints: $(ls -d $CKPT_DIR/checkpoint_* 2>/dev/null | wc -l)"
else
    echo "No checkpoints found yet"
fi
echo ""

# Check training status
echo "🔄 Training Status:"
if pgrep -f "verl.trainer.main_ppo" > /dev/null; then
    echo "✅ Training is RUNNING"
    echo "PID: $(pgrep -f verl.trainer.main_ppo)"
else
    echo "⏸️  Training is NOT running"
fi
echo ""

echo "=========================================="
echo "For detailed analysis, run:"
echo "  python scripts/analyze_training_log.py $LOG_FILE"
echo ""
echo "For TensorBoard visualization, run:"
echo "  tensorboard --logdir=$CKPT_DIR/tensorboard/ --port=6006"
echo "=========================================="
