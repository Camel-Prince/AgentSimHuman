# GRPO 学习笔记

> 对应代码：[verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py) 中的 `compute_grpo_outcome_advantage`

---

## 1. GRPO 是什么 / 为什么有它

GRPO（Group Relative Policy Optimization，DeepSeekMath 提出）是 PPO 的一个**去 critic**变体。

### 痛点：PPO 需要 critic
PPO 的 actor loss 需要 token 级 advantage $\hat A_t$，而 advantage 需要价值函数 $V(s_t)$。这意味着：
- 训练一个和 actor 同尺寸的 critic 模型 → 显存 ×2、计算 ×2
- critic 在 LLM 场景下并不好训练（token 级稀疏 reward，价值预测信号弱）

### GRPO 的思路：用"组内均值"替代 $V$
对同一个 prompt 采样 $G$ 条 rollout（一组），用这 $G$ 个序列级 reward 的**组内统计量**当作 baseline：

$$\hat A_i = \frac{R_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon}$$

- $\mu_{\text{group}}$ 充当"在这个 prompt 上，当前策略的平均表现" → 即 baseline，对应 $V(s_0)$
- 除以 $\sigma$ 做白化，稳定梯度尺度（和 GAE 末尾 `masked_whiten` 同样动机）

**取消 critic 的代价**：advantage 是序列级常量，所有 response token 共享同一个 $\hat A$（没有 token 级信用分配）。

---

## 2. 公式

设一个 prompt 的 group 内有 $G$ 条 rollout，第 $i$ 条序列的最终标量 reward 为 $R_i$（outcome reward，只在 EOS 或最后一个 token 给）。

**Step 1: 序列级 reward 聚合**

$$R_i = \sum_{t=0}^{T_i-1} r_{i,t}$$

由于 outcome reward 只有最后位置非零，sum 等价于取最后那个标量。

**Step 2: 组内白化**

$$\mu_g = \frac{1}{|g|}\sum_{i\in g} R_i, \qquad \sigma_g = \text{std}_{i\in g}(R_i)$$

$$\hat A_i = \frac{R_i - \mu_g}{\sigma_g + \epsilon}$$

**Step 3: 广播到 token 级**

$$\hat A_{i,t} = \hat A_i \cdot \text{mask}_{i,t}, \quad t=0,\dots,T_i-1$$

**Step 4: 复用 PPO clip loss**

$$L^{\text{GRPO}} = -\mathbb{E}_{i,t}\Big[\min\big(r_{i,t}(\theta)\hat A_{i,t},\ \text{clip}(r_{i,t},1\pm\epsilon)\hat A_{i,t}\big)\Big] + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

其中 $r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\text{old}}(a_{i,t}|s_{i,t})}$。

KL 项通常作为 token 级惩罚加到 loss 上（`use_kl_loss=True`），或直接扣到 reward 里（`apply_kl_penalty`）。

---

## 3. 与 PPO 的对比

| 维度 | PPO | GRPO |
|---|---|---|
| 是否需要 critic | 需要 | 不需要 |
| baseline | $V(s_t)$（critic 输出） | $\mu_{\text{group}}$（组内均值） |
| advantage 时间粒度 | token 级（GAE 沿时间积分） | 序列级常量广播 |
| 标准化 | `masked_whiten`（batch 全局） | 组内 z-score（按 prompt 分组） |
| rollout 数 | 通常 1 | 通常 $G=4\sim8$ |
| 显存 | actor + critic + ref | actor + ref |

**几何直觉**：PPO 的 advantage 既评估"在这条轨迹里第 t 步比平均好多少"，又评估"这条轨迹整体比平均好多少"；GRPO 只评估后者，靠组内多次采样来估计后者的"平均水平"。

---

## 4. verl 代码逐行解读

```python
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)   # (bs,) 序列级 reward
```

**形参**：
- `token_level_rewards`: `(bs, response_length)`，outcome reward 稀疏地放在最后位置
- `eos_mask`: `(bs, response_length)`，response 真实 token 为 1，prompt/padding 为 0
- `index`: `(bs,)`，每条样本所属 prompt 的 uid，用来分组

```python
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])           # 按 uid 分组收集 reward
```

```python
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)            # 组内只 1 条无法白化，退化为原值
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
```

```python
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)  # 组内白化
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask              # 广播 + mask
    return scores, scores   # 第二个返回值是 returns，GRPO 里不用 critic，所以也等于 advantage
```

返回 `scores, scores` 是因为接口要和 PPO 对齐：PPO 返回 `(advantages, returns)`，GRPO 没有 critic 训练目标，就直接复用 advantages 占位（critic worker 在 GRPO 配置下不会被启用）。

---

## 5. 三个代码细节 Q&A

### Q1: 为什么用 `non_zero_mask` 辅助求和？直接 sum 不行吗？

```python
non_zero_mask = (token_level_rewards != 0)
scores = (token_level_rewards * non_zero_mask).sum(dim=-1)
```

**数学上完全等价**：0 × mask 还是 0，直接 `token_level_rewards.sum(dim=-1)` 得到相同结果。

这行**实际是冗余的**，只表达语义"只聚合真正打了 reward 的位置"。在 outcome reward 场景下，整条序列只有最后一个 token 有非零 reward，sum 出来就是那个标量。

唯一不同的情况：如果 reward 含 NaN/Inf，mask 也不能屏蔽（因为 `nan * 0.0 = nan`）。所以这个 mask 没有实际功能作用，纯粹是作者的可读性偏好。

### Q2: 为什么用 `defaultdict(list)` 而不是普通 `dict`？

```python
id2score = defaultdict(list)
for i in range(bsz):
    id2score[index[i]].append(scores[i])
```

`defaultdict(list)` 在**首次访问不存在的 key** 时自动调用 `list()` 创建空列表作为默认值。

普通 dict 写法：

```python
id2score = {}
for i in range(bsz):
    if index[i] not in id2score:
        id2score[index[i]] = []
    id2score[index[i]].append(scores[i])
```

GRPO 场景下 `index[i]` 是 prompt 的 uid，每个 prompt 被 rollout 多次（默认 `n=5`），要把同 uid 的若干 score 聚到一个列表做组内统计。`defaultdict(list)` 让"分组收集"代码简洁。

### Q3: `scores.unsqueeze(-1).tile([1, response_length]) * eos_mask` 在做什么？

把**序列级标量 advantage 广播成 token 级 advantage**。

**形状变化**（示例 bs=2, response_length=4）：

| 操作 | 形状 | 示例 |
|---|---|---|
| 起点 `scores` | `(2,)` | `[0.8, -0.3]` |
| `.unsqueeze(-1)` | `(2, 1)` | `[[0.8], [-0.3]]` |
| `.tile([1, 4])` | `(2, 4)` | `[[0.8, 0.8, 0.8, 0.8], [-0.3, -0.3, -0.3, -0.3]]` |
| `* eos_mask` | `(2, 4)` | `[[0.8, 0.8, 0.8, 0], [-0.3, -0.3, 0, 0]]` |

**`tile` 是什么**：`tensor.tile([n1, n2, ...])` 在每个维度上重复 ni 次。`.tile([1, 4])` 即第 0 维不动、第 1 维复制 4 份。
- 等价写法：`scores.unsqueeze(-1).expand(-1, response_length)` （只是视图，不分配新内存）
- 或：`scores[:, None].repeat(1, response_length)`

**为什么要广播**：`compute_policy_loss` 需要 `advantages` 形状 `(bs, response_length)`，每个 token 都要有 advantage 值。GRPO 的 outcome reward 是序列级标量——所以把这个标量"复制"到该序列的每个 response token 上，让 token-level PPO loss 接口完全兼容。

**`* eos_mask` 的作用**：把 prompt token、padding token 位置的 advantage 强制置 0，避免下游误用（虽然 `masked_mean` 也会过滤，这里是保险）。

---

## 6. B=4 G=2 数值例子

设一个 batch 含 2 个 prompt（uid=A, B），每个 prompt rollout 2 次（G=2），response_length=3。

```
索引 i:    0       1       2       3
uid:       A       A       B       B
R_i:       1.0     0.0     0.5     0.5
```

**Step 1**：分组
- group A: [1.0, 0.0] → μ=0.5, σ=0.707
- group B: [0.5, 0.5] → μ=0.5, σ=0.0

**Step 2**：白化
- $\hat A_0 = (1.0-0.5)/(0.707+ε) ≈ +0.707$
- $\hat A_1 = (0.0-0.5)/(0.707+ε) ≈ -0.707$
- $\hat A_2 = (0.5-0.5)/(0.0+ε) ≈ 0$
- $\hat A_3 = (0.5-0.5)/(0.0+ε) ≈ 0$

**直观解读**：
- prompt A 上两条 rollout 有差异 → 给"较好的那条"正向梯度、"较差的那条"负向梯度
- prompt B 上两条完全一样 → 没有偏好信号，advantage=0 → 对梯度无贡献（合理：组内没区分度时不应更新）

**Step 3**：广播到 `(4, 3)`
```
[[ 0.707,  0.707,  0.707],
 [-0.707, -0.707, -0.707],
 [ 0.0,    0.0,    0.0  ],
 [ 0.0,    0.0,    0.0  ]]
```
再乘 `eos_mask` 屏蔽 padding。

---

## 7. 一句话总结

**GRPO = PPO 去掉 critic + 用同 prompt 多 rollout 的组内 z-score 作为 token 级 advantage 的常量广播**。所有花活都在 advantage 计算上，loss、KL、clip 完全复用 PPO 接口。

---

## 8. 相关代码位置

| 功能 | 文件:行 |
|---|---|
| GRPO advantage 计算 | [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py) `compute_grpo_outcome_advantage` |
| paper-writing 多段信用分配变体 | [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py) `compute_grpo_paper_writing_advantage` |
| PPO clip loss（GRPO 复用） | [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py) `compute_policy_loss` |
| 调用入口 / `adv_estimator` 分发 | [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) `compute_advantage` |
| GRPO 训练脚本 | [train_grpo.sh](train_grpo.sh) |
