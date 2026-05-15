# 1) 新建 py3.9 环境（不动 base）
/opt/conda/bin/conda create -n searchr1 -y python=3.9 pip
source /opt/conda/etc/profile.d/conda.sh
conda activate searchr1
python -V   # 确认是 3.9.x

# 2) 装 torch / vllm（CUDA 12.1 已就绪）
pip install --upgrade pip setuptools wheel
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3

# 3) 装 Search-R1 / verl
cd /workspace                       # 或你想放代码的位置
# 如果代码已经挂载进来就跳过 git clone
# git clone <你的内网仓库地址> Search-R1
pip install -e /workspace/Search-R1   # 会按 requirements.txt 安装 transformers<4.48 / ray / hydra-core 等

# 4) flash-attn（编译耗时，建议在容器里一次装好）
pip install flash-attn --no-build-isolation
pip install wandb

# 5) 验证
python -c "import torch, vllm, flash_attn; \
print(torch.__version__, torch.version.cuda, vllm.__version__, flash_attn.__version__)"