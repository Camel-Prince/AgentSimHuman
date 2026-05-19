# 可外部覆盖；默认使用两张卡
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2}
export DATA_DIR='data_paper_writing/processed'

# 缓解 FSDP backward 的显存碎片，避免 reserved-but-unallocated 导致 OOM
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# 本机无直连外网路由，必须走代理访问 dashscope。这里做两件事：
# 1) 清理 SOCKS 变量（httpx+openai 会因缺 socksio 报错，且 SOCKS 常把握手搞坏）
# 2) 若当前 http_proxy/https_proxy 仍是失效的 ofey 代理 (10.128.208.19)，
#    自动切回本机 clash (127.0.0.1:17890)
unset ALL_PROXY all_proxy
# if [[ -z "$http_proxy" || "$http_proxy" == *"10.128.208.19"* ]]; then
#     export http_proxy="http://127.0.0.1:17890"
#     export https_proxy="http://127.0.0.1:17890"
#     export HTTP_PROXY="$http_proxy"
#     export HTTPS_PROXY="$https_proxy"
# fi
echo "[INFO] Proxy: http_proxy=$http_proxy https_proxy=$https_proxy"

# 并发运行配置：
# - RUN_ID 用于区分同机并行任务（必须不同）
# - 端口与 TMPDIR 会根据 RUN_ID 自动偏移，避免 Ray session 冲突
# 示例：
#   RUN_ID=0 CUDA_VISIBLE_DEVICES=3,4 bash train_paper_writing_2gpu.sh
#   RUN_ID=1 CUDA_VISIBLE_DEVICES=5,6 bash train_paper_writing_2gpu.sh
#   TASK_TYPE=paper_writing_per_segment EXPERIMENT_TAG=per_segment \
#     RUN_ID=1 CUDA_VISIBLE_DEVICES=5,6 bash train_paper_writing_2gpu.sh
RUN_ID=${RUN_ID:-0}
RAY_BASE_GCS_PORT=${RAY_BASE_GCS_PORT:-6385}
RAY_BASE_DASHBOARD_PORT=${RAY_BASE_DASHBOARD_PORT:-8266}
RAY_TMPDIR_BASE=${RAY_TMPDIR_BASE:-/tmp/ray_wangzixu_pw2g}

export RAY_ADDRESS=""
export RAY_GCS_SERVER_PORT=$((RAY_BASE_GCS_PORT + RUN_ID))
export RAY_DASHBOARD_PORT=$((RAY_BASE_DASHBOARD_PORT + RUN_ID))
export RAY_TMPDIR="${RAY_TMPDIR_BASE}_${RUN_ID}"

# Ray 端口预检查，防止两个任务误用同一个 RUN_ID
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${RAY_GCS_SERVER_PORT}$"; then
    echo "[ERROR] RAY_GCS_SERVER_PORT=${RAY_GCS_SERVER_PORT} already in use. Please change RUN_ID."
    exit 1
fi

mkdir -p "${RAY_TMPDIR}"
echo "[INFO] RUN_ID=${RUN_ID}"
echo "[INFO] Ray config: GCS_PORT=${RAY_GCS_SERVER_PORT}, DASHBOARD_PORT=${RAY_DASHBOARD_PORT}, TMPDIR=${RAY_TMPDIR}"

# API Key 配置 (保持不变)
export COMMENTER_API_KEY="sk-c19178e6b0054b94ba68fa80c25e54bf"
export COMMENTER_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export COMMENTER_MODEL="qwen-plus"
export RUBRIC_API_KEY="sk-b870c071cce248ab825a9c213779cd68"
export RUBRIC_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export RUBRIC_MODEL="qwen-max"

WAND_PROJECT='Search-R1-PaperWriting'
export BASE_MODEL="${MODELSCOPE_CACHE:-/data1/wangzixu/.cache/modelscope}/hub/models/Qwen/Qwen2___5-3B-Instruct"
# warm cktp 是一个已经能够较好遵守format的模型
# WARM_CKPT_ROOT=/home/wangzixu/Search-R1/verl_checkpoints/paper-writing-grpo-qwen2_5-3B-instruct-arxiv-writing-20260408_000243                                                                                               
# export BASE_MODEL=${WARM_CKPT_ROOT}/actor/global_step_20
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

export VLLM_ATTENTION_BACKEND=XFORMERS

TBS=128
VBS=32
PPO_MINI_BATCH_SIZE=64
PPO_MICRO_BATCH_SIZE=2
LOG_PROB_MICRO_BATCH_SIZE=8

# GRPO-like, KL as loss
KL_LOSS=true
KL_LOSS_COEF=0.000

# PPO-like, KL correct the score to get reward;
# the critic/kl_coeff in wandb
KL_COEF=0.000
GROUP_SIZE=8

# 选择 rollout 路径:
# paper_writing_autonomous /paper_writing / paper_writing_per_segment / paper_writing_last_round_target / paper_writing_train_commenter / paper_writing_arena_seeded
TASK_TYPE=${TASK_TYPE:-paper_writing_autonomous}
# 选择 reward 路径:
# paper_writing / paper_writing_arena_hybrid
REWARD_TYPE=${REWARD_TYPE:-paper_writing}
# 实验名附加后缀（可选），例如 last_round_target
EXPERIMENT_TAG=${EXPERIMENT_TAG:-}
# 修订轮数（所有 paper-writing rollout 共用）
NUM_REVISION_ROUNDS=3
# API generator 并发（仅在 paper_writing_train_commenter 中使用）
GENERATOR_MAX_CONCURRENCY=64
# API 速率限制（RPM），commenter 和 rubric 共用
API_RPM=${API_RPM:-120}
COMMENTER_MAX_CONCURRENCY=${COMMENTER_MAX_CONCURRENCY:-48}
RUBRIC_RPM=${RUBRIC_RPM:-120}
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

# 当使用 last_round_target 且未显式设置 tag 时，自动追加默认后缀，方便区分实验目录
if [ -z "${EXPERIMENT_TAG}" ] && [ "${TASK_TYPE}" = "paper_writing_last_round_target" ]; then
    EXPERIMENT_TAG="last_round_target"
fi
if [ -z "${EXPERIMENT_TAG}" ] && [ "${TASK_TYPE}" = "paper_writing_per_segment" ]; then
    EXPERIMENT_TAG="per_segment"
fi

# 统一实验名拼接，支持可选 tag
if [ -n "${EXPERIMENT_TAG}" ]; then
    export EXPERIMENT_NAME=paper-writing-grpo-qwen2_5-3B-instruct-arxiv-writing-${TIMESTAMP}-run${RUN_ID}-${EXPERIMENT_TAG}
else
    export EXPERIMENT_NAME=paper-writing-grpo-qwen2_5-3B-instruct-arxiv-writing-${TIMESTAMP}-run${RUN_ID}
fi

echo "[INFO] Training config: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, TASK_TYPE=${TASK_TYPE}, REWARD_TYPE=${REWARD_TYPE}, EXPERIMENT_NAME=${EXPERIMENT_NAME}"

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
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=$KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.state_masking=true \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n_agent=$GROUP_SIZE \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    algorithm.no_think_rl=false \
    trainer.logger=['console'] \
    +trainer.val_before_train=false \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.total_epochs=15 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    +task_type=$TASK_TYPE \
    +reward_type=$REWARD_TYPE \
    +num_revision_rounds=$NUM_REVISION_ROUNDS \
    +generator_max_concurrency=$GENERATOR_MAX_CONCURRENCY \
    +api_rpm=$API_RPM \
    +commenter_max_concurrency=$COMMENTER_MAX_CONCURRENCY \
    +rubric_rpm=$RUBRIC_RPM \
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
    max_turns=2 \
    2>&1 | tee $EXPERIMENT_NAME.log
