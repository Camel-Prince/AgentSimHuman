# GRPO 和 PPO Loss 计算流程详解

本文档详细介绍 veRL 框架中 GRPO 和 PPO 算法的 loss 计算流程，包括 advantage 计算、log probability、KL 散度、entropy loss 以及 value loss 的完整实现。

---

## 目录

1. [整体流程对比](#整体流程对比)
2. [Log Probability 计算](#log-probability-计算)
3. [GRPO Loss 计算](#grpo-loss-计算)
4. [PPO Loss 计算](#ppo-loss-计算)
5. [代码位置索引](#代码位置索引)

---

## 整体流程对比

### GRPO 训练流程

```
Step 1: Rollout (生成数据)
├─ 采样 prompts
├─ 每个 prompt 生成 n_agent 个 responses
└─ 计算 old_log_probs

Step 2: Reward 计算
├─ 调用 reward function 得到 scores
└─ rewards = scores (不减 KL)

Step 3: Advantage 计算 (Group-based)
├─ 按 prompt 分组
├─ 计算每组的 mean 和 std
└─ advantage = (score - group_mean) / (group_std + ε)

Step 4: Actor 更新
├─ 计算 policy loss (PPO-Clip)
├─ 计算 entropy loss
├─ 计算 KL loss (可选)
└─ actor_loss = pg_loss - entropy_loss + kl_loss
```

### PPO 训练流程

```
Step 1: Rollout (生成数据)
├─ 采样 prompts
├─ 生成 responses
└─ 计算 old_log_probs

Step 2: Reward 计算
├─ 调用 reward function 得到 scores
├─ 计算 KL 散度
└─ rewards = scores - β × KL (KL 惩罚 reward)

Step 3: Critic 计算 Values
└─ values = critic(states)

Step 4: Advantage 计算 (GAE)
├─ 使用 value function
└─ advantage = GAE(rewards, values, γ, λ)

Step 5: Critic 更新
└─ critic_loss = value_loss

Step 6: Actor 更新
├─ 计算 policy loss (PPO-Clip)
├─ 计算 entropy loss
└─ actor_loss = pg_loss - entropy_loss
```

---

## Log Probability 计算

### 数学公式

对于生成的每个 token，计算其条件概率的对数：

```
log π_θ(o_t | o_{<t}, q)
```

其中：
- `π_θ`：策略（语言模型），参数为 θ
- `o_t`：第 t 个位置生成的 token
- `o_{<t}`：前面所有已生成的 tokens (o_1, ..., o_{t-1})
- `q`：query/prompt（输入）

### 代码实现

**位置：`verl/utils/torch_functional.py:49-73`**

```python
def logprobs_from_logits(logits, labels):
    """计算 token-level log probabilities"""
    # 使用 log_softmax 计算整个 vocab 的 log 概率
    logp = F.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)

    # 提取实际生成的 token 的 log 概率
    logpy = gather_from_labels(logp, labels)  # (batch, seq_len)

    return logpy
```

### 详细步骤

**位置：`verl/workers/actor/dp_actor.py:52-141`**

```python
# Step 1: Forward pass
output = model(input_ids, attention_mask, position_ids)
logits = output.logits  # (batch_size, seq_len, vocab_size)

# Step 2: 应用 temperature
logits = logits / temperature

# Step 3: 提取 response 部分的 logits
# 注意：取 response 前一个位置开始，因为 logits[t] 预测 token[t+1]
logits = logits[:, -response_length - 1 : -1, :]

# Step 4: 计算 log_softmax
log_probs_all = F.log_softmax(logits, dim=-1)
# shape: (batch_size, response_length, vocab_size)

# Step 5: 提取实际生成的 token 的 log_prob
log_probs = gather_from_labels(log_probs_all, responses)
# shape: (batch_size, response_length)
```

### 关键理解

1. **Autoregressive 特性**：位置 t 的 logits 预测位置 t+1 的 token
2. **条件概率**：模型的 attention 机制自动处理了条件依赖
3. **Softmax 归一化**：必须对整个 vocab 计算 softmax，然后提取实际 token 的概率

---

## GRPO Loss 计算

GRPO (Group Relative Policy Optimization) 不使用 critic，通过 group-based advantage 进行训练。

### 1. Advantage 计算 (Group-based)

**位置：`verl/trainer/ppo/core_algos.py:111-156`**

```python
def compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index, epsilon=1e-6):
    """
    GRPO 的核心：基于同一 prompt 的多个 responses 计算 group advantage

    Args:
        token_level_rewards: (batch_size, response_length) - token 级别的 reward
        eos_mask: (batch_size, response_length) - 有效 token 的 mask
        index: (batch_size,) - prompt ID，用于分组
        epsilon: 防止除零的小常数

    Returns:
        advantages: (batch_size, response_length) - 归一化的 advantage
        returns: (batch_size, response_length) - 与 advantages 相同
    """
    # Step 1: 计算每个 response 的总 score
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)  # (batch_size,)

    # Step 2: 按 prompt ID 分组
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    # Step 3: 计算每组的均值和标准差
    for idx in id2score:
        if len(id2score[idx]) == 1:
            # 只有一个样本，无法计算 std
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            id2std[idx] = torch.std(torch.tensor([id2score[idx]]))

    # Step 4: 归一化每个 response 的 score
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

    # Step 5: 将 scalar advantage 扩展到所有 token 位置
    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores
```

**核心思想**：
- 同一 prompt 的多个 responses 形成一组
- 使用组内均值作为 baseline
- 归一化：`advantage = (score - group_mean) / (group_std + ε)`

### 2. Policy Loss (PPO-Clip)

**位置：`verl/trainer/ppo/core_algos.py:163-194`**

```python
def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """
    计算 PPO-Clip policy loss

    Args:
        old_log_prob: (batch_size, response_length) - rollout 时的 log_prob
        log_prob: (batch_size, response_length) - 当前的 log_prob
        advantages: (batch_size, response_length) - advantage 估计
        eos_mask: (batch_size, response_length) - 有效 token 的 mask
        cliprange: float - clip 范围 (通常是 0.2)

    Returns:
        pg_loss: scalar - policy gradient loss
        pg_clipfrac: float - 被 clip 的比例
        ppo_kl: float - 近似 KL 散度
    """
    # Step 1: 计算 ratio = π_new / π_old
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, eos_mask)

    # Step 2: 计算两种 loss
    pg_losses = -advantages * ratio  # 标准 policy gradient
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    # Step 3: 取最大值（悲观更新）
    pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)

    return pg_loss, pg_clipfrac, ppo_kl
```

**PPO-Clip 公式**：
```
L^{CLIP}(θ) = E[min(r_t(θ) × A_t, clip(r_t(θ), 1-ε, 1+ε) × A_t)]
```
其中 `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` 是概率比率。

### 3. Entropy Loss

**位置：`verl/trainer/ppo/core_algos.py:197-213`**

```python
def compute_entropy_loss(logits, eos_mask):
    """
    计算 entropy bonus，鼓励探索

    Args:
        logits: (batch_size, response_length, vocab_size)
        eos_mask: (batch_size, response_length)

    Returns:
        entropy_loss: scalar - 平均 entropy
    """
    entropy = verl_F.entropy_from_logits(logits)  # (batch, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss
```

**作用**：
- 鼓励模型保持一定的随机性
- 防止过早收敛到单一策略
- 通常系数很小（如 0.001）

### 4. KL Loss (正则项)

**位置：`verl/workers/actor/dp_actor.py:263-271`**

```python
if self.config.use_kl_loss:
    ref_log_prob = data['ref_log_prob']

    # 计算 KL 散度
    kld = core_algos.kl_penalty(
        logprob=log_prob,
        ref_logprob=ref_log_prob,
        kl_penalty=self.config.kl_loss_type  # 'low_var_kl'
    )
    kl_loss = masked_mean(kld, response_mask)

    # 加到 policy loss 上
    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
```

**KL 散度计算** (`verl/trainer/ppo/core_algos.py:242-274`)：

```python
def kl_penalty(logprob, ref_logprob, kl_penalty):
    """计算 KL 散度"""
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "low_var_kl":
        # 低方差近似（推荐）
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
```

### 5. GRPO 完整 Loss

**位置：`verl/workers/actor/dp_actor.py:235-276`**

```python
def update_policy(self, data: DataProto):
    """GRPO Actor 更新"""
    for data in micro_batches:
        # 1. 计算 policy gradient loss
        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=response_mask,
            cliprange=clip_ratio
        )

        # 2. 计算 entropy loss
        entropy_loss = verl_F.masked_mean(entropy, response_mask)

        # 3. 组合 policy loss
        policy_loss = pg_loss - entropy_loss * entropy_coeff

        # 4. 可选：加上 KL loss
        if self.config.use_kl_loss:
            kl_loss = masked_mean(kld, response_mask)
            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

        # 5. Backward
        loss = policy_loss / self.gradient_accumulation
        loss.backward()

    # 6. 更新参数
    self.actor_optimizer.step()
```

**GRPO 完整公式**：
```
L_GRPO = L_pg^{CLIP} - c_1 × H[π_θ] + c_2 × KL[π_θ || π_ref]
```

---

## PPO Loss 计算

PPO 使用 critic 网络来估计 value function，需要同时优化 actor 和 critic。

### 1. Reward 计算（KL 惩罚）

**位置：`verl/trainer/ppo/ray_trainer.py:88-120`**

```python
def apply_kl_penalty(data, kl_ctrl, kl_penalty='kl'):
    """
    PPO 中 KL 用于惩罚 reward

    与 GRPO 的区别：
    - PPO: reward = score - β × KL (KL 在 reward 阶段)
    - GRPO: loss = policy_loss + β × KL (KL 在 loss 阶段)
    """
    token_level_scores = data.batch['token_level_scores']
    response_length = token_level_scores.shape[-1]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # 计算 KL 散度
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch['old_log_probs'],
            data.batch['ref_log_prob'],
            kl_penalty=kl_penalty
        )
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    # 关键：KL 直接从 reward 中减去
    token_level_rewards = token_level_scores - beta * kld

    data.batch['token_level_rewards'] = token_level_rewards

    return data, {'critic/kl': current_kl, 'critic/kl_coeff': beta}
```

### 2. Advantage 计算 (GAE)

**位置：`verl/trainer/ppo/core_algos.py:70-107`**

```python
def compute_gae_advantage_return(token_level_rewards, values, eos_mask, gamma, lam):
    """
    计算 Generalized Advantage Estimation (GAE)

    Args:
        token_level_rewards: (batch_size, response_length) - token 级别的 reward
        values: (batch_size, response_length) - critic 预测的 value
        eos_mask: (batch_size, response_length) - 有效 token 的 mask
        gamma: float - 折扣因子（通常是 1.0）
        lam: float - GAE 的 λ 参数（通常是 1.0）

    Returns:
        advantages: (batch_size, response_length) - 归一化的 advantage
        returns: (batch_size, response_length) - 目标 return
    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        # 从后往前计算 GAE
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0

            # TD error: δ_t = r_t + γ × V(s_{t+1}) - V(s_t)
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]

            # GAE: A_t = δ_t + γλ × A_{t+1}
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)

        # 反转回正序
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        # Returns: R_t = A_t + V(s_t)
        returns = advantages + values

        # 归一化 advantages
        advantages = verl_F.masked_whiten(advantages, eos_mask)

    return advantages, returns
```

**GAE 公式**：
```
A_t^{GAE(γ,λ)} = Σ_{l=0}^∞ (γλ)^l × δ_{t+l}
其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

### 3. Value Loss (Critic)

**位置：`verl/trainer/ppo/core_algos.py:216-239`**

```python
def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """
    计算 value function loss (clipped MSE)

    Args:
        vpreds: (batch_size, response_length) - 当前 critic 预测的 value
        returns: (batch_size, response_length) - 目标 return (ground truth)
        values: (batch_size, response_length) - 旧的 critic 预测的 value
        eos_mask: (batch_size, response_length) - 有效 token 的 mask
        cliprange_value: float - clip 范围（通常是 0.2）

    Returns:
        vf_loss: scalar - value function loss
        vf_clipfrac: float - 被 clip 的比例
    """
    # Step 1: Clip 预测值，防止更新太激进
    vpredclipped = verl_F.clip_by_value(
        vpreds,
        values - cliprange_value,
        values + cliprange_value
    )

    # Step 2: 计算两种 MSE loss
    vf_losses1 = (vpreds - returns)**2          # 未 clip 的 MSE
    vf_losses2 = (vpredclipped - returns)**2    # clip 后的 MSE

    # Step 3: 取最大值（悲观更新）
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)

    return vf_loss, vf_clipfrac
```

**为什么要 Clip Value？**
- 与 policy clip 保持一致，确保 actor 和 critic 同步更新
- 防止 value function 过拟合单个 batch
- 提升训练稳定性

**为什么取 max？**
- 悲观更新策略
- 如果更新太大 → loss1 > loss2 → 用 loss1 惩罚
- 如果更新合理 → loss1 ≈ loss2 → 正常更新

### 4. Critic 更新

**位置：`verl/workers/critic/dp_critic.py:146-199`**

```python
def update_critic(self, data: DataProto):
    """PPO Critic 更新"""
    self.critic_module.train()

    for data in micro_batches:
        # 1. Forward pass 得到 value predictions
        vpreds = self._forward_micro_batch(data)

        # 2. 计算 value loss
        vf_loss, vf_clipfrac = core_algos.compute_value_loss(
            vpreds=vpreds,
            values=values,      # old values
            returns=returns,    # ground truth returns
            eos_mask=eos_mask,
            cliprange_value=self.config.cliprange_value
        )

        # 3. Backward
        loss = vf_loss / self.gradient_accumulation
        loss.backward()

    # 4. 更新 critic 参数
    self.critic_optimizer.step()
```

### 5. PPO 完整 Loss

**Actor Loss**：
```python
actor_loss = pg_loss - entropy_loss * entropy_coeff
```

**Critic Loss**：
```python
critic_loss = 0.5 × max(MSE(vpreds, returns), MSE(vpreds_clipped, returns))
```

**PPO 完整公式**：
```
L_PPO = L_pg^{CLIP} - c_1 × H[π_θ] + c_2 × L_vf^{CLIP}
```

---

## 代码位置索引

### 核心算法

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| **Log Probability 计算** | `verl/utils/torch_functional.py` | 49-73 |
| **GRPO Advantage** | `verl/trainer/ppo/core_algos.py` | 111-156 |
| **PPO Advantage (GAE)** | `verl/trainer/ppo/core_algos.py` | 70-107 |
| **Policy Loss (PPO-Clip)** | `verl/trainer/ppo/core_algos.py` | 163-194 |
| **Value Loss** | `verl/trainer/ppo/core_algos.py` | 216-239 |
| **Entropy Loss** | `verl/trainer/ppo/core_algos.py` | 197-213 |
| **KL Penalty** | `verl/trainer/ppo/core_algos.py` | 242-274 |

### Worker 实现

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| **Actor Forward** | `verl/workers/actor/dp_actor.py` | 52-141 |
| **Actor Update (GRPO/PPO)** | `verl/workers/actor/dp_actor.py` | 203-290 |
| **Critic Forward** | `verl/workers/critic/dp_critic.py` | 52-101 |
| **Critic Update** | `verl/workers/critic/dp_critic.py` | 146-199 |

### Trainer 流程

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| **KL 惩罚 Reward (PPO)** | `verl/trainer/ppo/ray_trainer.py` | 88-120 |
| **Advantage 计算调用** | `verl/trainer/ppo/ray_trainer.py` | 123-156 |
| **训练主循环** | `verl/trainer/ppo/ray_trainer.py` | 667-878 |

---

## 总结对比

### GRPO vs PPO 关键区别

| 特性 | GRPO | PPO |
|------|------|-----|
| **Advantage 计算** | Group-based，同 prompt 的多个样本共享 baseline | GAE，使用 value function |
| **是否需要 Critic** | ❌ 不需要 | ✅ 需要 |
| **KL 使用位置** | Loss 阶段（正则项） | Reward 阶段（惩罚项） |
| **Baseline** | Group mean | Value function V(s) |
| **样本效率** | 需要多个样本（n_agent > 1） | 单个样本即可 |
| **适用场景** | Outcome reward（单个 scalar） | Token-level reward |

### Loss 公式总结

**GRPO**：
```
L_GRPO = L_pg^{CLIP} - c_1 × H[π_θ] + c_2 × KL[π_θ || π_ref]

其中：
- L_pg^{CLIP}: PPO-Clip policy loss
- H[π_θ]: Entropy bonus (探索)
- KL[π_θ || π_ref]: KL 正则项（约束）
```

**PPO**：
```
L_PPO = L_pg^{CLIP} - c_1 × H[π_θ] + c_2 × L_vf^{CLIP}

其中：
- L_pg^{CLIP}: PPO-Clip policy loss
- H[π_θ]: Entropy bonus (探索)
- L_vf^{CLIP}: Clipped value loss (critic)
```

---

## 参考文献

1. **PPO 论文**: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
2. **GAE 论文**: Schulman et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
3. **GRPO 实现**: veRL 框架 - https://github.com/volcengine/verl

---

*文档生成时间: 2026-03-28*
*基于 veRL 框架源码分析*
