# veRL 框架学习笔记

> 目标：把 veRL 框架的"模块架构 → 训练数据流 → 关键代码片段"梳理成可以反复回查的笔记。配合 Search-R1 / paper-writing rollout 的实际用法。

---

## 一、模块架构

veRL 的设计哲学：**driver 端排兵布阵 + Ray worker group 执行 + DataProto 做协议**。整个仓库分 4 层。

### 1. Driver / 训练编排层

- **入口**：`verl/trainer/main_ppo.py`（Hydra config）→ 实例化 `RayPPOTrainer`
- **核心**：[verl/trainer/ppo/ray_trainer.py](../verl/trainer/ppo/ray_trainer.py) 的 `RayPPOTrainer.fit()`
- driver 不持有模型，只通过 worker group 句柄远程调用：
  - `actor_rollout_wg`：合体的 actor + rollout（同一组 GPU 进程，靠 sharding_manager 切换）
  - `ref_policy_wg`：冻结的 reference policy
  - `critic_wg`：value 网络（PPO 用）
  - `rm_wg`：神经 reward model（可选）
- driver 上几乎没有 GPU 运算，只跑 advantage、KL 注入、reward 合成等轻量逻辑

### 2. 算法纯函数层

**位置**：[verl/trainer/ppo/core_algos.py](../verl/trainer/ppo/core_algos.py)

完全和分布式解耦的纯 torch 函数：

| 函数 | 作用 |
|---|---|
| `compute_gae_advantage_return` | PPO 用，需要 critic values |
| `compute_grpo_outcome_advantage` | GRPO 用，按 prompt 组做 z-score 归一化 |
| `compute_grpo_paper_writing_advantage` | paper-writing 自定义，按 `segment_ids` 做"camera-ready 广播 + 每个 draft 局部 broadcast" |
| `compute_policy_loss` | PPO clip 目标 |
| `kl_penalty` | 五种 KL 估计器（`kl/abs/mse/low_var_kl/full`） |
| `AdaptiveKLController / FixedKLController` | KL 系数自适应 |

### 3. Worker 层（Ray actor）

| Worker | 文件 | 内容 |
|---|---|---|
| Actor (训练) | [verl/workers/actor/dp_actor.py](../verl/workers/actor/dp_actor.py) `DataParallelPPOActor` | FSDP 包裹；提供 `compute_log_prob` 和 `update_policy` |
| Critic | `verl/workers/critic/` | Value head；`compute_values` 和 `update_critic` |
| Reward model | `verl/workers/reward_model/` | 神经 RM（可选） |
| Rollout | `verl/workers/rollout/vllm_rollout/` | vLLM 推理；`generate_sequences` |
| Sharding manager | [verl/workers/sharding_manager/fsdp_vllm.py](../verl/workers/sharding_manager/fsdp_vllm.py) | FSDP shard ↔ vLLM 权重切换 |

**关键事实**：actor + rollout **共置在同一组 GPU 进程**，互相切换显存占用。ref policy 是独立 worker group。原因：vLLM 只出 token，loss 必须由 FSDP 重算。

### 4. 协议层 / 工具

- [verl/protocol.py](../verl/protocol.py) `DataProto`：包了 `TensorDict`（按 batch dim 对齐的张量）+ `non_tensor_batch`（object 数组，如 uid、camera_ready）+ `meta_info`（dict，全局 scalar）。`repeat / chunk / pop / union / select` 保证三者并行变形
- [verl/utils/torch_functional.py](../verl/utils/torch_functional.py)：`logprobs_from_logits / entropy_from_logits / masked_mean / masked_whiten` 等
- `verl/utils/seqlen_balancing.py`：dynamic micro-batching
- `verl/single_controller/`：Ray worker group 抽象

**直觉**：driver 不持有大张量，worker 不持有训练循环；张量通过 DataProto 走 Ray object store 流转。

---

## 二、vLLM 与 FSDP 的分工

一个 `actor_rollout_wg` worker 在同一 GPU 上同时挂两份"模型形态"，由 `FSDPVLLMShardingManager` 做切换：

| 引擎 | 形态 | 负责 | 输入 | 输出 |
|---|---|---|---|---|
| **vLLM** | 推理引擎，PagedAttention，权重 TP-shard | Rollout 采样 | `input_ids` | `responses` token ids（**不返回 logits/logprob**） |
| **FSDP actor** | 训练形态，ZeRO-3 sharded，挂优化器/梯度 | (a) `compute_log_prob` 算 `old_log_probs`；(b) `update_policy` 反传 loss | `(input_ids, attention_mask, position_ids, responses)` | (a) `old_log_probs [B, T_resp]`；(b) 梯度 |
| **FSDP ref** | 冻结的参考策略 | `compute_ref_log_prob` | 同上 | `ref_log_prob [B, T_resp]` |

**切换时机**（每个 step 内）：

1. rollout 阶段：`FSDPVLLMShardingManager.__enter__` → FSDP full state_dict gather → `inference_engine.sync_model_weights(...)` → vLLM 生成
2. logprob / 训练阶段：`__exit__` → vLLM 释放 → FSDP reshard → `compute_log_prob` / `update_policy`

**关键事实**：vLLM 永远不算 loss，也不返回完整 logits / 分布；它只吐 token。任何概率/分布的量必须由 FSDP 那边重放一遍 forward 才能拿到。

**为什么 vLLM 不能直接算 logprob？**
1. vLLM 走 PagedAttention + 增量解码，没有 teacher-forcing 路径
2. 它的权重和 FSDP 那份不同步——只有 rollout 上下文期间一致；actor update 一次，vLLM 权重就过期

---

## 三、典型训练 step 的数据流

### 高层管线

```
[Driver] dataloader yields batch_dict          ← parquet
       ▼
[Driver] DataProto.from_single_dict → batch
[Driver] batch.repeat(n_agent)
[Driver] gen_batch = batch.pop(input_ids,...)
       ▼
─────── ① ROLLOUT ──────────────────────────────
[actor_rollout_wg] (vLLM 形态)
   in : gen_batch  (input_ids,attention_mask,position_ids)
   out: gen_batch_output (responses)
       ▼
[Driver] batch = batch.repeat(rollout.n).union(gen_batch_output)
       ▼
─────── ② OLD LOGPROB ──────────────────────────
[actor_rollout_wg] (FSDP 形态)
   in : (input_ids+responses, attn_mask, position_ids, responses)
   out: old_log_probs        [B, T_resp]
       ▼
─────── ③ REF LOGPROB (可选) ────────────────────
[ref_policy_wg] (FSDP, 冻结)
   in : 同上
   out: ref_log_prob         [B, T_resp]
       ▼
─────── ④ VALUES (仅 PPO) ───────────────────────
[critic_wg] (FSDP)
   out: values               [B, T_resp]
       ▼
─────── ⑤ REWARD ───────────────────────────────
[Driver]  reward_fn(batch)  -> token_level_scores  [B, T_resp]
       ▼
─────── ⑥ KL 注入 / 留给 actor ──────────────────
[Driver] if not use_kl_loss:
              apply_kl_penalty: token_level_rewards = scores - β·KL
          else:
              token_level_rewards = scores   (KL 由 actor 当 loss 项)
       ▼
─────── ⑦ ADVANTAGE ────────────────────────────
[Driver] compute_advantage:
          gae:  GAE(values, rewards) → advantages, returns
          grpo: 组内 z-score → advantages
       ▼
─────── ⑧ UPDATE CRITIC (仅 PPO) ────────────────
[critic_wg] update_critic(batch)     ← MSE loss
       ▼
─────── ⑨ UPDATE ACTOR ─────────────────────────
[actor_rollout_wg] (FSDP) update_actor(batch)
   inside: forward → log_prob → PPO clip loss (+kl loss)
       ▼
       回到 ① 下一个 step
```

### A. PPO (on-policy, 带 critic)

| 步骤 | 引擎 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| ① rollout | **vLLM** | prompt | response | `rollout.n=1` |
| ② old_log_probs | FSDP actor | (prompt+resp) | `old_log_probs [B,T]` | PPO ratio 的分母 |
| ③ ref_log_prob | FSDP ref | 同 ② | `ref_log_prob [B,T]` | KL 教师 |
| ④ values | FSDP critic | 同 ② | `values [B,T]` | 每 token V 值 |
| ⑤ reward | driver | responses, gt | `token_level_scores [B,T]` | rule-based 或 RM |
| ⑥ KL 注入 | driver | scores, old_lp, ref_lp | `token_level_rewards` | `rewards = scores - β·(old-ref)` |
| ⑦ advantage | driver | rewards, values | `advantages, returns` | GAE 反向递推 |
| ⑧ update_critic | FSDP critic | (input, returns, values) | grad | MSE clip |
| ⑨ update_actor | FSDP actor | (input, responses, old_lp, advantages) | grad | PPO clip loss |

**on-policy 体现**：`old_log_probs` 在 ② 用刚采样时的权重算；只要 PPO mini-epoch ≤ 1 或 ratio 在 clip 区间内，近似 on-policy。

### B. GRPO (on-policy, 无 critic)

砍掉 ④/⑧，advantage 改为组内归一化：

| 步骤 | 引擎 | 输入 | 输出 |
|---|---|---|---|
| ① rollout | vLLM | prompt | `rollout.n=k`，每个 prompt 产 k 条 |
| ② old_log_probs | FSDP actor | 同 PPO | `old_log_probs` |
| ③ ref_log_prob | FSDP ref | 同 PPO | `ref_log_prob` |
| ⑤ reward | driver | responses | `token_level_scores`（EOS 处一个标量） |
| ⑥ KL | 一般 `use_kl_loss=True` | — | rewards = scores（KL 进 actor loss） |
| ⑦ advantage | driver | scores, mask, uid | 组内 z-score → broadcast 回 token |
| ⑨ update_actor | FSDP actor | + ref_log_prob | `pg_loss + kl_loss_coef * KL(low_var_kl)` |

**关键差异**：advantage 来自"同 prompt 内 k 条样本的相对优劣"，不需 critic。

### C. OPD（reverse-KL distillation）

OPD = GRPO 的 PG 项被弱化或去掉，**主目标变成 reverse-KL = KL(π_θ ‖ π_ref)**；ref 是更强的 teacher。

实现路径：
- `use_kl_loss=True`、`kl_loss_coef` 调大（成为主项）
- `kl_loss_type=low_var_kl`（hard-token reverse-KL）；或补 `"full"` 分支做 full-vocab
- reward 权重压低或 `token_level_scores=0`（纯 distillation）

数据流和 GRPO 同形，只是 ⑨ 里 KL 项主导。

#### C.1 Hard-token reverse-KL（默认支持）

`update_policy` 一个 micro-batch 内：

```python
entropy, log_prob = self._forward_micro_batch(data, temperature)   # [B, T_resp]
# ref_log_prob 已在 ③ 算好
kld = core_algos.kl_penalty(log_prob, ref_log_prob, kl_penalty=kl_loss_type)
kl_loss = masked_mean(kld, response_mask)
policy_loss = pg_loss - entropy*ec + kl_loss * kl_loss_coef
loss.backward()
```

- 引擎：vLLM 采 token，FSDP 跑两次 teacher-forcing
- 只产生 1-D logprob 张量；通信量小
- 是采样轨迹上的（低方差/有偏）reverse-KL 估计
- 和 PPO ratio 完全兼容

#### C.2 Full-vocab reverse-KL

定义：

$$
\mathrm{KL}\big(\pi_\theta(\cdot|s_t)\,\|\,\pi_{\text{ref}}(\cdot|s_t)\big)
= \sum_{v\in\mathcal V}\pi_\theta(v|s_t)\big(\log\pi_\theta(v|s_t)-\log\pi_{\text{ref}}(v|s_t)\big)
$$

veRL 在 `kl_penalty` 里给 `"full"` 留了入口但 `raise NotImplementedError`。要走通需要：

1. **Rollout 不变**：vLLM 还是只吐 token
2. **学生 forward**：返回 `[B, T_resp, V]` 而不是 gather 后的 `[B, T_resp]`
3. **教师 forward**：同样返回 `[B, T_resp, V]`；**必须和学生同 micro-batch 即用即弃**，否则爆显存（如 B=8, T=1024, V=152k 的 Qwen ≈ 2.3 GB / micro-batch / 模型 / fp16）
4. **KL 计算**：
   ```python
   p_theta = log_p_theta.exp()
   kld_per_step = (p_theta * (log_p_theta - log_p_ref)).sum(-1)   # [B, T]
   kl_loss = masked_mean(kld_per_step, response_mask)
   ```
5. **PG 项**：依然只需 gather 取采样 token 的 `log_prob` 做 ratio

| 维度 | Hard-token KL | Full-vocab KL |
|---|---|---|
| 训练引擎产物 | `[B,T]` scalar logprob | `[B,T,V]` logits |
| 教师存储 | 可跨阶段缓存 | 必须即用即弃 |
| KL 性质 | 采样轨迹估计（有偏） | 真值期望 |
| 梯度信号 | 仅采样 token | 整个词表 soft 信号 |
| 显存/通信 | 小 | 大 (×V) |
| veRL 现状 | 全部就绪 | `"full"` 分支 NotImplemented |

---

## 四、必须熟练的关键代码

### 1. logits → log_probs

[verl/utils/torch_functional.py](../verl/utils/torch_functional.py)

```python
def logprobs_from_logits_naive(logits, labels):
    logp = F.log_softmax(logits, dim=-1)           # [B, T, V]
    logpy = gather_from_labels(logp, labels)        # [B, T]
    return logpy

def gather_from_labels(data, label):
    return torch.gather(data, -1, label.unsqueeze(-1)).squeeze(-1)
```

省显存版（不存 V 维 softmax）：

```python
def logprobs_of_labels_v2(logits, labels):
    gathered = torch.gather(logits, -1, labels.unsqueeze(-1)).squeeze(-1)
    return gathered - torch.logsumexp(logits, dim=-1)
```

### 2. Teacher-forcing forward 的 shift 技巧

[verl/workers/actor/dp_actor.py](../verl/workers/actor/dp_actor.py) 非 rmpad 分支：

```python
output = self.actor_module(input_ids=..., attention_mask=..., position_ids=...,
                           use_cache=False)
logits = output.logits                          # [B, L, V]   L = prompt+resp
logits.div_(temperature)
logits = logits[:, -response_length - 1:-1]     # 关键!
log_probs = logprobs_from_logits(logits, micro_batch['responses'])
```

**为什么 `[-response_length-1 : -1]`**？LM 是 next-token prediction：位置 `t` 的 logits 用来预测位置 `t+1` 的 token。response 占序列尾部 `T_resp` 个 token，要预测它们就得取它们 **前一个位置** 的 logits，所以左移 1 位。

rmpad（flash-attn varlen）分支用 `torch.roll` 等效实现：

```python
input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
log_probs = logprobs_from_logits(logits_rmpad, input_ids_rmpad_rolled)
```

### 3. 熵（数值稳定）

```python
def entropy_from_logits(logits):
    pd = torch.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
```

等价于 `-sum p log p`，但避免 small p 处的 -inf。

### 4. masked_mean / masked_whiten

```python
def masked_mean(values, mask, axis=None):
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)

def masked_whiten(values, mask, shift_mean=True):
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened
```

**所有"对 response 段做平均/归一化"几乎都走 `masked_mean`**，别用 `tensor.mean()`——pad 位置会污染统计。`masked_whiten` 给 GAE 后的 advantage 归一化（PPO 训稳）。

### 5. PPO clip loss

[verl/trainer/ppo/core_algos.py](../verl/trainer/ppo/core_algos.py)

```python
negative_approx_kl = log_prob - old_log_prob       # 注意符号
ratio = torch.exp(negative_approx_kl)              # π_new / π_old
pg_losses  = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1-cliprange, 1+cliprange)
pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
```

`torch.max` 是 PPO 精髓：取"两个 loss 的更大者" ≡ 取"两个目标的更小者"，保证 ratio 偏离 1 时被惩罚。

### 6. KL 估计器集合

```python
if kl_penalty == 'kl':         return logprob - ref_logprob          # 朴素,高方差
if kl_penalty == 'abs':        return (logprob - ref_logprob).abs()
if kl_penalty == 'mse':        return 0.5*(logprob - ref_logprob)**2
if kl_penalty == 'low_var_kl':                                       # Schulman 近似
    kl = ref_logprob - logprob
    ratio = torch.exp(kl)
    return torch.clamp(ratio - kl - 1, -10, 10)                      # >=0, low-var
if kl_penalty == 'full':       raise NotImplementedError             # 全 vocab
```

注意 sign：`'kl'` 返回 `log π_θ − log π_ref`，正好和 reverse-KL 方向一致。

### 7. GRPO advantage

```python
# 1) token reward → seq reward
scores = (token_level_rewards * (token_level_rewards != 0)).sum(dim=-1)  # [B]
# 2) 按 prompt 组（uid）做 z-score
for idx in groups:
    id2mean[idx] = mean(group_scores)
    id2std[idx]  = std(group_scores)
scores[i] = (scores[i] - id2mean[uid_i]) / (id2std[uid_i] + eps)
# 3) broadcast 回 [B, T_resp]，再乘 response mask
advantages = scores.unsqueeze(-1).tile([1, T_resp]) * eos_mask
```

**broadcast 回 token 级**是关键：PG 公式 `∇log π(a_t) * Â` 期望 `Â` per-token，但 reward 只有 seq 级，所以直接复制。这也是 GRPO 没法做 step-level credit assignment 的原因（paper-writing 的 `compute_grpo_paper_writing_advantage` 加了 `segment_ids` 做细粒度归因）。

### 8. FSDP ↔ vLLM 权重切换

[verl/workers/sharding_manager/fsdp_vllm.py](../verl/workers/sharding_manager/fsdp_vllm.py)

```python
def __enter__(self):
    params = self.module.state_dict()              # FSDP gather: sharded → full / dtensor
    self.inference_engine.sync_model_weights(
        params, load_format='hf' if self.full_params else 'dtensor')
    del params; torch.cuda.empty_cache()

def __exit__(self, ...):
    self.inference_engine.offload_model_weights()  # 释放 vLLM 权重 + KV cache
    self.module.train()
    torch.cuda.empty_cache()
```

**三个事实**：
1. rollout 上下文 `with FSDPVLLMShardingManager(...):` 包住的就是 `generate_sequences`
2. 进入时 FSDP 权重被 gather 后拷贝进 vLLM；FSDP 本身只剩 shard 占用（小）
3. 退出时 vLLM 释放，FSDP 继续训练。`compute_log_prob` / `update_policy` 一定发生在 `__exit__` 之后

### 9. DataProto 的小动作

```python
gen_batch = batch.pop(batch_keys=['input_ids','attention_mask','position_ids'])
batch = batch.repeat(n, interleave=True)
batch = batch.union(gen_batch_output)
batch.non_tensor_batch['uid'] = ...
data = data.select(batch_keys=[...]).batch         # 拿到底层 TensorDict
micro_batches = batch.split(micro_bsz)
```

**非张量字段一定要放 `non_tensor_batch`**，这样 `repeat/chunk/reorder` 会自动跟着张量同序变换——paper-writing 代码里大量依赖这个不变量。

### 10. 多轮 rollout 中 per-segment logprob 拼接

[search_r1/llm_agent/generation.py](../search_r1/llm_agent/generation.py) 的 `_build_logprob_dataproto` 和 `_update_dense_right_side_aligned`：把 ②③ 在多轮 rollout 的每一段 segment 上做一次，并按 pad-right 规则把不等长 segment 拼到一个 `responses_with_info_mask` 对齐的 dense 张量里。这是 veRL 原生没有的扩展，但完全建立在 `compute_log_prob` 接口上。

---

## 五、推荐阅读顺序

1. [verl/protocol.py](../verl/protocol.py)：浏览 `DataProto` 的 `repeat / pop / union / select`
2. [verl/trainer/ppo/ray_trainer.py](../verl/trainer/ppo/ray_trainer.py) 的 `fit()` 主循环：把"哪一步打哪个 wg 的哪个 method"对照本节图
3. [verl/workers/actor/dp_actor.py](../verl/workers/actor/dp_actor.py) 的 `_forward_micro_batch / compute_log_prob / update_policy`：搞清 shift、masked_mean、PPO clip、KL loss 怎么落地
4. [verl/trainer/ppo/core_algos.py](../verl/trainer/ppo/core_algos.py)：一次看完 GAE、GRPO advantage、policy loss、kl penalty
5. [verl/workers/sharding_manager/fsdp_vllm.py](../verl/workers/sharding_manager/fsdp_vllm.py)：理解权重切换边界
6. [verl/utils/torch_functional.py](../verl/utils/torch_functional.py) 的 `logprobs_from_logits / entropy_from_logits / masked_*`：手写一遍

跑熟后再回头看 `search_r1/llm_agent/generation.py` 的 `run_llm_loop*` 系列——它们只干两件事：(a) 多次调 vLLM 生成段；(b) 维护 `responses / responses_with_info_mask / segment_ids` 这套 dense 张量，让最终 DataProto 喂回 actor 时格式和单轮 PPO 完全一致。框架对它们透明，所以才能复用 PPO/GRPO 全部基础设施。

---

## 六、速查清单（常错点）

- **shift -1**：teacher-forcing 算 response logprob 时一定要 `logits[:, -T_resp-1:-1]`，response 段不能直接对齐 logits
- **温度除法**：`logits.div_(temperature)` 必须在 logprob/entropy 计算之前，且 rollout 和 logprob 重算的 temperature 要一致（meta_info 里传）
- **masked_mean**：response 段聚合永远用它，pad 位置会污染
- **`old_log_probs` 在哪算**：必须在 rollout **之后**、actor update **之前**，由 FSDP actor 用当前权重算一遍；这是 PPO ratio 的分母
- **`use_kl_loss` 二选一**：True 时 KL 作为 actor loss 项，`token_level_rewards = scores`；False 时 KL 注入 reward（`apply_kl_penalty`）
- **GRPO 必须 `rollout.n > 1`**：否则组内 std 为 0，advantage 全 0
- **vLLM 权重过期**：actor update 一次后 vLLM 那份就过期，下次 rollout 前必须重新 `sync_model_weights`
- **DataProto 字段对齐**：`repeat / chunk` 只对 `batch` + `non_tensor_batch` 有效，自己手动 append 的字段要放对位置
- **dynamic_bsz 的副作用**：`rearrange_micro_batches` 会打乱顺序，结束时必须 `revert_indices` 还原
