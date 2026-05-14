# eos_mask 学习笔记

> 对应代码：
> - 定义：[verl/utils/torch_functional.py](verl/utils/torch_functional.py) `get_eos_mask`
> - 使用：[verl/workers/rollout/vllm_rollout/vllm_rollout.py](verl/workers/rollout/vllm_rollout/vllm_rollout.py)、[verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) `compute_advantage`

---

## 1. 含义

`eos_mask` 是一个形状为 `(bs, response_length)` 的 0/1 张量：

- **第一个 EOS token 及其之前的位置** = 1（有效 token，参与训练）
- **第一个 EOS 之后的位置** = 0（padding 或多余生成，不参与训练）

示例（设 EOS 的 token id = 1）：

```
response_id: [tok_a, tok_b, tok_c, EOS, PAD, PAD]
eos_mask:    [  1,     1,     1,    1,   0,   0]
                                    ↑ EOS 本身保留为 1（要训练它）
                                       ↑ 之后全 0
```

> **命名提醒**：函数 / 形参叫 `eos_mask`，但实际语义是 "**response 段有效 token mask**"。这是从 trl 继承的历史命名，不是只标记 EOS 那一个位置。

---

## 2. 作用：在 PPO/GRPO 训练全链路屏蔽无效 token

`eos_mask` 在数据流中扮演 "哪些 token 算数" 的统一开关，被以下几处使用：

| 使用位置 | 文件 | 作用 |
|---|---|---|
| Advantage 白化 | `core_algos.compute_gae_advantage_return` | `masked_whiten` 只用有效 token 统计 μ/σ |
| GRPO 广播 | `core_algos.compute_grpo_outcome_advantage` | 把序列级标量 advantage 复制到每个有效 token，无效位置归零 |
| Policy loss | `core_algos.compute_policy_loss` | `masked_mean` 对 PPO clip loss 做按有效 token 求均值 |
| Value loss | `core_algos.compute_value_loss` | 同上 |
| Entropy loss | `core_algos.compute_entropy_loss` | 同上 |
| KL 罚项 | `ray_trainer.apply_kl_penalty` | KL 散度只在有效 token 上累加 |
| 序列级指标 | `ray_trainer` 各处 `masked_mean` | response 长度、平均 reward 等都按 mask 加权 |

### 为什么不能用整段 attention_mask？

`attention_mask` 包含 prompt + response 的所有有效 token。但 advantage、PPO ratio 等只在 **模型自己生成的 response token** 上才有意义（prompt 部分模型没采样动作，不需要梯度）。所以 verl 一律切 `attention_mask[:, -response_length:]`，再交给 advantage / loss 函数。

### 为什么不能用整段 response_id != pad_id？

EOS 本身被模型采样出来，是模型预测的**最后一个有效 action**，必须参与训练。而 EOS 之后的 padding 不是模型的输出，不能算。`response_id != pad_id` 这种判定方式在某些 tokenizer 里 PAD = EOS（如 GPT-2），会出错。基于 EOS 截断的方式更鲁棒。

---

## 3. 在 verl 数据流中的位置

```
vLLM rollout
    └─ 生成 response 的 token id
    └─ get_eos_mask(response_id, eos_token=tok.eos_token_id)  ← 此处构造
    └─ torch.cat([prompt_attention_mask, response_attention_mask])  ← 拼成完整 attention_mask
            │
            ▼
DataProto 流转到 trainer
            │
            ▼
ray_trainer.compute_advantage:
    response_mask = attention_mask[:, -response_length:]   ← 切回 response 段
    compute_gae_advantage_return(..., eos_mask=response_mask)
    或 compute_grpo_outcome_advantage(..., eos_mask=response_mask)
            │
            ▼
core_algos.compute_policy_loss(..., eos_mask=response_mask)
core_algos.compute_value_loss(..., eos_mask=response_mask)
```

---

## 4. 计算方法（`get_eos_mask` 逐步解析）

```python
def get_eos_mask(response_id, eos_token=2, dtype=torch.int64):
    eos_mask = response_id.eq(eos_token).long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask
```

设 `response_id = [0, 0, 2, 42, 3, 5, 1, 0, 0]`，`eos_token=1`，期望输出 `[1, 1, 1, 1, 1, 1, 1, 0, 0]`。

### Step 1：标记 EOS 位置

```python
eos_mask = response_id.eq(eos_token).long()
```

逐元素比较，等于 EOS 的位置=1。

```
response_id: [0, 0, 2, 42, 3, 5, 1, 0, 0]
.eq(1):      [0, 0, 0, 0,  0, 0, 1, 0, 0]
                                  ↑ EOS 位置
```

### Step 2：`cumsum` 累加

```python
torch.cumsum(eos_mask, dim=1)
```

**`cumsum` 定义**：沿指定维度做累积求和，`out[i] = in[0] + in[1] + ... + in[i]`。

```
输入:    [0, 0, 0, 0, 0, 0, 1, 0, 0]
cumsum:  [0, 0, 0, 0, 0, 0, 1, 1, 1]
                              ↑ 第一次遇到 EOS 起，累加值变成 1，并一直保持
```

**关键性质**：cumsum 之后，**第一个 EOS 及其之后的所有位置都 ≥ 1**，之前都 = 0。这样就分离出 "EOS 及之后" 的区域。

### Step 3：`cumsum - eos_mask` —— 把 EOS 本身从 "之后" 区域剔除

```python
eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
```

```
cumsum:           [0, 0, 0, 0, 0, 0, 1, 1, 1]
减原 eos_mask:   -[0, 0, 0, 0, 0, 0, 1, 0, 0]
                  ───────────────────────────
结果:             [0, 0, 0, 0, 0, 0, 0, 1, 1]
                                     ↑ EOS 位置变 0
                                        ↑ 严格 EOS 之后保持 1
.bool():          [F, F, F, F, F, F, F, T, T]
```

**含义**：此时 True 表示 "**严格在 EOS 之后**"。EOS 自己被减回 0，因为我们要让 EOS 进入训练区域。

### Step 4：取反 + 类型转换

```python
eos_mask = torch.logical_not(eos_mask).to(dtype)
```

```
取反前:   [F, F, F, F, F, F, F, T, T]
取反后:   [T, T, T, T, T, T, T, F, F]
.to(int): [1, 1, 1, 1, 1, 1, 1, 0, 0]   ← 最终输出
```

把 "严格在 EOS 之后" 翻转成 "EOS 及之前"，得到目标 mask。

### 步骤汇总

| 步骤 | 操作 | 中间结果 | 含义 |
|---|---|---|---|
| 1 | `eq(eos)` | `[0,0,0,0,0,0,1,0,0]` | 标记 EOS 位置 |
| 2 | `cumsum` | `[0,0,0,0,0,0,1,1,1]` | EOS 及之后累加 ≥ 1 |
| 3 | `- eos_mask` | `[0,0,0,0,0,0,0,1,1]` | 严格 EOS 之后 = 1 |
| 4 | `logical_not` + dtype | `[1,1,1,1,1,1,1,0,0]` | EOS 及之前 = 1，输出 |

---

## 5. 等价的朴素实现（仅供理解）

```python
def get_eos_mask_naive(response_id, eos_token):
    bs, L = response_id.shape
    mask = torch.zeros_like(response_id)
    for b in range(bs):
        for t in range(L):
            mask[b, t] = 1
            if response_id[b, t] == eos_token:
                break   # EOS 之后不再标 1
    return mask
```

verl 的 `cumsum` 写法是这个 for 循环的向量化等价版本，无 Python 循环，GPU 并行、batch 友好。

---

## 6. 边界情况

| 情况 | cumsum 行为 | 最终 mask | 说明 |
|---|---|---|---|
| 没有 EOS（生成到 max_length 仍未 EOS） | 全 0 | 全 1 | 整条都是有效 token ✓ |
| 第 0 个 token 就是 EOS | `[1,1,1,...]` | `[1,0,0,...]` | 只有 EOS 这一位有效 ✓ |
| 多个 EOS（如 `[a, EOS, b, EOS, c]`） | `[0,1,1,2,2]` → `-eq` → `[0,0,1,1,2]` → `.bool()` → `[F,F,T,T,T]` → not → `[1,1,0,0,0]` | 只在**第一个 EOS** 处截断 ✓ |

---

## 7. 与 attention_mask 的关系

vLLM rollout 完成后，最终的 `attention_mask`（[verl/workers/rollout/vllm_rollout/vllm_rollout.py](verl/workers/rollout/vllm_rollout/vllm_rollout.py) ≈ 第 207 行附近）：

```python
response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, ...)
attention_mask = torch.cat((attention_mask_prompt, response_attention_mask), dim=-1)
```

整条 attention_mask 全貌：

```
位置:          [ ← prompt 段 (左 padding) → ] [ ← response 段 → ]
attention:     [0, 0, 0, p1, p2, p3, p4, p5]  [1, 1, 1, 1, 1, 0, 0]
                ↑ 左 padding   ↑ prompt 内容    ↑ 真实生成+EOS  ↑ pad
切片 [-resp_len:]:                            ⤷ [1, 1, 1, 1, 1, 0, 0]   ← 这就是 advantage 计算用的 eos_mask
```

trainer 端调用 `attention_mask[:, -response_length:]` 即可拿回 `eos_mask`。

---

## 8. 一句话总结

`eos_mask` = "**response 段第一个 EOS（含）之前为 1，之后为 0**" 的 token 级 0/1 mask。通过 `eq → cumsum → 减去 → 取反` 四步向量化构造，是 verl 在 advantage / policy / value / KL 全链路中统一区分 "有效 token vs padding" 的标准开关。
