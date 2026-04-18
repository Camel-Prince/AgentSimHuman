export CUDA_VISIBLE_DEVICES=1,2,3,7
export DATA_DIR='data_paper_writing/processed'

# 并发运行配置：
# - RUN_ID 用于区分同机并行任务（必须不同）
# - 端口与 TMPDIR 会根据 RUN_ID 自动偏移，避免 Ray session 冲突
RUN_ID=${RUN_ID:-0}
RAY_BASE_GCS_PORT=${RAY_BASE_GCS_PORT:-6385}
RAY_BASE_DASHBOARD_PORT=${RAY_BASE_DASHBOARD_PORT:-8266}
RAY_TMPDIR_BASE=${RAY_TMPDIR_BASE:-/tmp/ray_wangzixu_pw4g}

export RAY_ADDRESS=""
export RAY_GCS_SERVER_PORT=$((RAY_BASE_GCS_PORT + RUN_ID))
export RAY_DASHBOARD_PORT=$((RAY_BASE_DASHBOARD_PORT + RUN_ID))
export RAY_TMPDIR="${RAY_TMPDIR_BASE}_${RUN_ID}"

# Ray 端口预检查，防止多个任务误用同一个 RUN_ID
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${RAY_GCS_SERVER_PORT}$"; then
    echo "[ERROR] RAY_GCS_SERVER_PORT=${RAY_GCS_SERVER_PORT} already in use. Please change RUN_ID."
    exit 1
fi

mkdir -p "${RAY_TMPDIR}"
echo "[INFO] RUN_ID=${RUN_ID}"
echo "[INFO] Ray config: GCS_PORT=${RAY_GCS_SERVER_PORT}, DASHBOARD_PORT=${RAY_DASHBOARD_PORT}, TMPDIR=${RAY_TMPDIR}"

# API Key 配置 (保持不变)
export COMMENTER_API_KEY="sk-b870c071cce248ab825a9c213779cd68"
export COMMENTER_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export COMMENTER_MODEL="qwen-plus"
export RUBRIC_API_KEY="sk-b870c071cce248ab825a9c213779cd68"
export RUBRIC_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export RUBRIC_MODEL="qwen-max"

WAND_PROJECT='Search-R1-PaperWriting'
export BASE_MODEL='/home/wangzixu/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export EXPERIMENT_NAME=paper-writing-grpo-qwen2_5-7B-instruct-arxiv-writing-${TIMESTAMP}-run${RUN_ID}

export VLLM_ATTENTION_BACKEND=XFORMERS

TBS=128
VBS=32
PPO_MINI_BATCH_SIZE=64
PPO_MICRO_BATCH_SIZE=8
LOG_PROB_MICRO_BATCH_SIZE=16
KL_LOSS=true
KL_LOSS_COEF=0.001
GROUP_SIZE=16
MAX_TOKEN_LEN_PER_GPU=4096
WARM_UP_RATIO=0

# 选择 rollout 路径:
# paper_writing / paper_writing_per_segment / paper_writing_last_round_target / paper_writing_train_commenter / paper_writing_arena_seeded
TASK_TYPE=${TASK_TYPE:-paper_writing}
# 选择 reward 路径:
# paper_writing / paper_writing_arena_hybrid
REWARD_TYPE=paper_writing
# 修订轮数（所有 paper-writing rollout 共用）
NUM_REVISION_ROUNDS=3
# API generator 并发（仅在 paper_writing_train_commenter 中使用）
GENERATOR_MAX_CONCURRENCY=64
# Arena Swiss 单轮排序配置（仅在 paper_writing_arena_seeded + arena_hybrid 奖励下生效）
ARENA_SEED_MODE=swiss_single_round
ARENA_SEED=20260413
ARENA_GROUP_SIZE=$GROUP_SIZE
# Arena + Rubric 融合权重（仅 reward_type=paper_writing_arena_hybrid 生效）
ARENA_WEIGHT=0.7
RUBRIC_WEIGHT=0.3
PAPER_WRITING_SAVE_SFT_CANDIDATES=${PAPER_WRITING_SAVE_SFT_CANDIDATES:-true}
PAPER_WRITING_SFT_SCORE_THRESHOLD=${PAPER_WRITING_SFT_SCORE_THRESHOLD:-0.78}
PAPER_WRITING_SFT_OUTPUT_DIR=${PAPER_WRITING_SFT_OUTPUT_DIR:-outputs/sft_candidates}

EXPERIMENT_TAG=${EXPERIMENT_TAG:-}
if [ -z "${EXPERIMENT_TAG}" ] && [ "${TASK_TYPE}" = "paper_writing_per_segment" ]; then
    EXPERIMENT_TAG="per_segment"
fi
if [ -n "${EXPERIMENT_TAG}" ]; then
    export EXPERIMENT_NAME=paper-writing-grpo-qwen2_5-7B-instruct-arxiv-writing-${TIMESTAMP}-run${RUN_ID}-${EXPERIMENT_TAG}
fi


PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/arxiv_writing_train.parquet \
    data.val_files=$DATA_DIR/arxiv_writing_valid.parquet \
    data.train_batch_size=$TBS \
    data.val_batch_size=$VBS \
    data.max_prompt_length=3072 \
    data.max_response_length=1024 \
    data.max_start_length=768 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$WARM_UP_RATIO \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=$KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n_agent=$GROUP_SIZE \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    algorithm.no_think_rl=false \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=false \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    +task_type=$TASK_TYPE \
    +reward_type=$REWARD_TYPE \
    +num_revision_rounds=$NUM_REVISION_ROUNDS \
    +generator_max_concurrency=$GENERATOR_MAX_CONCURRENCY \
    +arena_seed_mode=$ARENA_SEED_MODE \
    +arena_seed=$ARENA_SEED \
    +arena_group_size=$ARENA_GROUP_SIZE \
    +arena_weight=$ARENA_WEIGHT \
    +rubric_weight=$RUBRIC_WEIGHT \
    +commenter_api_key=$COMMENTER_API_KEY \
    +commenter_base_url=$COMMENTER_BASE_URL \
    +commenter_model=$COMMENTER_MODEL \
    +rubric_api_key=$RUBRIC_API_KEY \
    +rubric_api_base=$RUBRIC_API_BASE \
    +rubric_model=$RUBRIC_MODEL \
    +paper_writing_save_sft_candidates=$PAPER_WRITING_SAVE_SFT_CANDIDATES \
    +paper_writing_sft_score_threshold=$PAPER_WRITING_SFT_SCORE_THRESHOLD \
    +paper_writing_sft_output_dir=$PAPER_WRITING_SFT_OUTPUT_DIR \
    2>&1 | tee $EXPERIMENT_NAME.log
