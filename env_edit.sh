#!/usr/bin/env bash
# 在 Paladin 父镜像（已有 /opt/conda + CUDA 12.1）基础上，创建 searchr1 conda 环境
# 用法：在容器里执行  bash env_edit.sh
set -euo pipefail

# ===== 配置区：按你们公司实际情况修改 =====
PIP_INDEX_URL="https://<内网pypi>/simple"
PIP_TRUSTED_HOST="<内网pypi主机名>"          # 例如 mirrors.xxx.com
SEARCHR1_DIR="/workspace/Search-R1"          # 仓库在容器里的路径（按挂载点改）
ENV_NAME="searchr1"
# ==========================================

CONDA=/opt/conda/bin/conda
source /opt/conda/etc/profile.d/conda.sh

# 1) 新建 py3.9 环境（不动 base）
#    需要 conda 能拉到 python=3.9。若父镜像默认 channel 不通，
#    请先在 ~/.condarc 配好内网 conda mirror。
$CONDA create -n "$ENV_NAME" -y python=3.9 pip

conda activate "$ENV_NAME"
python -V   # 应输出 Python 3.9.x

# 2) 配置内网 pip 源（仅作用于当前 env）
pip config set global.index-url "$PIP_INDEX_URL"
pip config set global.trusted-host "$PIP_TRUSTED_HOST"
pip install --upgrade pip setuptools wheel

# 3) 装 torch / vllm（父镜像已带 CUDA 12.1）
#    前提：内网 pypi 已同步 cu121 版 torch。否则改成本地 wheel：
#    pip install /path/to/torch-2.4.0+cu121-*.whl
pip install torch==2.4.0
pip install vllm==0.6.3

# 4) 装 Search-R1 / verl 及其依赖（按 requirements.txt 拉 transformers<4.48 / ray / hydra-core 等）
if [ ! -d "$SEARCHR1_DIR" ]; then
    echo "[ERROR] 仓库目录不存在: $SEARCHR1_DIR"
    echo "        请把代码挂载/拷贝到该路径，或修改本脚本顶部的 SEARCHR1_DIR"
    exit 1
fi
pip install -e "$SEARCHR1_DIR"

# 5) flash-attn 需要编译（依赖父镜像里的 nvcc 12.1）
pip install ninja packaging
pip install flash-attn --no-build-isolation

# 6) 训练日志工具
pip install wandb

# 7) 验证
python - <<'PY'
import torch, vllm, flash_attn
print("torch       :", torch.__version__, "| cuda:", torch.version.cuda)
print("vllm        :", vllm.__version__)
print("flash_attn  :", flash_attn.__version__)
print("cuda avail  :", torch.cuda.is_available(), "| device count:", torch.cuda.device_count())
PY

cat <<EOF

[OK] 环境 ${ENV_NAME} 准备完成。后续使用：
    source /opt/conda/etc/profile.d/conda.sh
    conda activate ${ENV_NAME}
    export VLLM_ATTENTION_BACKEND=XFORMERS
    cd ${SEARCHR1_DIR}
    bash train_ppo.sh   # 或 train_grpo.sh / train_paper_writing_*.sh
EOF