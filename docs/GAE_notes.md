# GAE Advantage 学习笔记

> 按照手写思路整理：从 advantage 的定义出发，先看 MC 估计是怎么来的，再看 GAE 的两个极端 λ=0 / λ=1 分别是什么，最后理解 λ∈(0,1) 的物理意义；最后逐行解读 verl 的实现。

---

## 1. Advantage 的定义

$$
A(s_t, a_t) \;:=\; Q(s_t, a_t) \;-\; V(s_t)
$$

两个量都不能直接拿到，所以必须分别做近似：

```
        A(s_t, a_t)  :=  Q(s_t, a_t)   -   V(s_t)
                         ↑                   ↑
                       近似                  建模
                         │                   │
                   G_t = Σ γ^k r_{t+k}    V̂_t = V_φ(s_t)
                   (k=0..T-t)             (critic 网络)
```

- $Q$：用**真实跑出来的轨迹回报** $G_t$ 来近似（蒙特卡洛）
- $V$：用 critic 网络 $V_\phi$ 来**建模**

---

## 2. 蒙特卡洛 advantage

$G_t$ 是从 $t$ 开始一直累到终止的真实折扣回报：

$$
G_t \;=\; \sum_{k=0}^{T-t} \gamma^k \, r_{t+k}
$$

代入 $A$ 的定义：

$$
\boxed{\;\hat A_t^{\text{MC}} \;\approx\; G_t - \hat V_t \;=\; \sum_{k=0}^{T-t} \gamma^k r_{t+k} \;-\; \hat V_t\;}
$$

**特点**：
- $G_t$ 是 $Q$ 的**无偏**样本（单条轨迹方差极大）
- 减 $\hat V_t$ 只减方差不引入偏差（baseline 不依赖动作 $a_t$）

---

## 3. GAE 定义

$$
\hat A_t^{\text{GAE}} \;:=\; \sum_{k=0}^{\infty} (\gamma\lambda)^k \, \delta_{t+k}, \qquad
\delta_t = \gamma \hat V_{t+1} + r_t - \hat V_t \quad\text{(TD Error)}
$$

$\delta_t$ 就是经典的一步 TD 残差。GAE 把当前及之后所有 TD 残差用 $(\gamma\lambda)^k$ 加权累加。

---

## 4. 极端情况一：λ = 0

只剩 $k=0$ 那一项：

$$
\hat A_t^{\text{GAE}} \;=\; \delta_t \;=\; \gamma \hat V_{t+1} + r_t - \hat V_t
$$

**就是一步 TD 残差**。完全依赖 critic 来 bootstrap，方差最小、偏差最大。

---

## 5. 极端情况二：λ = 1（关键推导）

代入定义并展开 $\delta$：

$$
\hat A_t^{\text{GAE}} \;=\; \sum_{k=0}^{\infty}\gamma^k \delta_{t+k}
\;\approx\; \sum_{k=0}^{T-t}\gamma^k \delta_{t+k}
$$

$$
=\;\sum_{k=0}^{T-t}\gamma^k\Big(\gamma \hat V_{t+k+1} + r_{t+k} - \hat V_{t+k}\Big)
$$

把括号内每一项乘上 $\gamma^k$ 后分拆：

$$
=\;\sum_{k=0}^{T-t}\underbrace{\gamma^{k+1}\hat V_{t+k+1}}_{B_{k+1}}
\;+\;\sum_{k=0}^{T-t}\gamma^k r_{t+k}
\;-\;\sum_{k=0}^{T-t}\underbrace{\gamma^k \hat V_{t+k}}_{B_k}
$$

中间那一坨就是 $G_t$；两个 $V$ 的求和形成**伸缩相消（telescoping）**：

$$
\sum_{k=0}^{T-t} B_{k+1} \;-\; \sum_{k=0}^{T-t} B_k
\;=\; B_{T-t+1} - B_0
$$

代回去：

$$
\hat A_t^{\text{GAE}}\big|_{\lambda=1}
\;=\; G_t \;+\; B_{T-t+1} \;-\; B_0
\;=\; G_t \;+\; \gamma^{T-t+1}\hat V_{T+1} \;-\; \hat V_t
$$

episodic 任务里终止后 $\hat V_{T+1}=0$，于是：

$$
\boxed{\;\hat A_t^{\text{GAE}}\big|_{\lambda=1} \;=\; G_t + 0 - \hat V_t \;=\; G_t - \hat V_t \;}
$$

**正好就是 MC 估计**。中间所有 $\hat V_{t+1},\hat V_{t+2},\ldots$ 全被伸缩相消掉了——也就是说 λ=1 时 GAE **完全不信中间 critic**。

---

## 6. λ ∈ (0, 1)：物理意义

> **λ 的指数衰减权重决定了"我信任多远的 TD"。**

- $\lambda$ 越小 → 衰减越快 → 只信最近几步 TD → **更依赖 critic**（接近一步 TD）
- $\lambda$ 越大 → 衰减越慢 → 累积更多步的真实 reward → **更接近 MC**
- 常用 $\lambda \in [0.9, 0.99]$：在 bias 和 variance 之间折中

| $\lambda$ | 退化形态 | 偏差 | 方差 | 对 critic 的依赖 |
|---|---|---|---|---|
| 0 | 一步 TD | 高 | 低 | 完全依赖 |
| 1 | MC | 低（无偏） | 高 | 完全不信中间值 |
| 0.95 | GAE 折中 | 中 | 中 | 信任近期 TD |

---

## 7. $\hat A_t$ 与 $\hat A_{t+1}$ 的递推关系（O(T) 实现的关键）

直接按级数定义算 GAE 是 $O(T^2)$。利用错位关系可以化成 $O(T)$ 的反向递推。

把定义拆出 $k=0$ 那一项：

$$
\hat A_t = \sum_{k=0}^{\infty}(\gamma\lambda)^k \delta_{t+k}
       = \delta_t + \sum_{k=1}^{\infty}(\gamma\lambda)^k \delta_{t+k}
$$

第二项里令 $j = k-1$，把 $k$ 换成 $j+1$：

$$
\sum_{k=1}^{\infty}(\gamma\lambda)^k \delta_{t+k}
= (\gamma\lambda)\sum_{j=0}^{\infty}(\gamma\lambda)^j \delta_{t+1+j}
= (\gamma\lambda)\,\hat A_{t+1}
$$

合起来得到**核心递推式**：

$$
\boxed{\;\hat A_t \;=\; \delta_t \;+\; \gamma\lambda\,\hat A_{t+1}, \qquad \hat A_{T+1}=0\;}
$$

边界条件 $\hat A_{T+1}=0$ 来自 episodic 任务：终止后没有未来 reward 也没有未来 value。

这正是 verl 代码里 `lastgaelam = delta + gamma*lam*lastgaelam` 一行做的事。

---

## 8. verl 官方代码逐行解读

源码 [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py):

```python
def compute_gae_advantage_return(token_level_rewards, values, eos_mask, gamma, lam):
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns
```

### 输入张量

- `token_level_rewards: (B, T)` — per-token reward $r_t$。LLM RL 里通常稀疏，只有 EOS / segment 末端非零
- `values: (B, T)` — critic 输出的 $\hat V(s_t)$
- `eos_mask: (B, T)` — response 段的 attention mask（EOS 之后为 0），保证 prompt 段和 pad 不参与统计
- `gamma`, `lam` — 标量超参 $\gamma, \lambda$

### 逐行解读

| 代码 | 对应数学 | 解读 |
|---|---|---|
| `with torch.no_grad():` | — | advantage 不参与梯度。它在 actor loss 里作为常数：$\nabla\log\pi(a_t) \cdot \hat A_t$。如果反传到这里会污染 critic |
| `lastgaelam = 0` | $\hat A_{T+1}=0$ | 递推的初始值（"未来的 advantage"） |
| `for t in reversed(range(gen_len)):` | $t = T-1, T-2, \ldots, 0$ | **反向**遍历是为了复用 $\hat A_{t+1}$ |
| `nextvalues = values[:, t+1] if t < gen_len-1 else 0.0` | $\hat V_{t+1}$，终末取 0 | 终末位置（$t = T-1$）后面没有 token，$\hat V_T = 0$。这是 GAE 的边界条件，**容易写错的地方** |
| `delta = r_t + γ·V_{t+1} − V_t` | $\delta_t = r_t + \gamma\hat V_{t+1} - \hat V_t$ | TD 残差 |
| `lastgaelam = delta + γλ·lastgaelam` | $\hat A_t = \delta_t + \gamma\lambda\,\hat A_{t+1}$ | **核心递推式**，一行完成 |
| `advantages_reversed.append(lastgaelam)` | 反序收集 | 因为是从后往前算 |
| `torch.stack(advantages_reversed[::-1], dim=1)` | $[\hat A_0, \hat A_1, \ldots, \hat A_{T-1}]$ | `[::-1]` 把反向列表翻正 |
| `returns = advantages + values` | $R_t = \hat A_t + \hat V_t$ | critic MSE 的 target；本质是 $\lambda$-return |
| `masked_whiten(advantages, eos_mask)` | 在 response mask 内 z-score | **训稳关键**：让 advantage 均值 0、方差 1，避免不同样本量纲不同搞炸 PPO ratio |
| `return advantages, returns` | — | 返回两个 `(B, T)` 张量 |

### 关键细节追问

**Q1：为什么 `returns = advantages + values`？**

因为：

$$
R_t \;=\; \hat A_t + \hat V_t \;=\; (Q - V) + V \;=\; Q
$$

也就是 critic 应该拟合的目标。GAE 给出的 $R_t$ 实际上是 $\lambda$-return $G_t^\lambda$ —— 把 $\lambda$ 取 0 时 $R_t = r_t + \gamma \hat V_{t+1}$（TD target），$\lambda=1$ 时 $R_t = G_t$（MC target），中间则是平滑组合。这样 actor 用的 advantage 和 critic 学的 target **方向一致**，训练更稳。

**Q2：为什么终末 `nextvalues = 0`？**

LLM RL 是 episodic（EOS 终止）。终止后没有后续状态，$\hat V_{T} = 0$ 是定义。如果填 `values[:, t+1]` 会越界；如果按某种 bootstrap 取 $\hat V_T$，相当于把后面"虚构"的未来回报算进来，对 episodic 任务是错的。

**Q3：稀疏 reward 怎么传到前面？**

`token_level_rewards` 大部分位置为 0，只有 EOS 处有标量 reward。靠两步传播：

1. critic `values` 在每个位置都有非零估计，$\delta_t$ 通过 $\hat V_{t+1} - \hat V_t$ 把"未来期望回报变化"注入到每一步
2. 反向递推 `lastgaelam = δ + γλ·lastgaelam` 把末端的 reward 信号沿着 $(\gamma\lambda)^k$ 衰减系数向前广播

所以**有 critic 的 GAE 不怕稀疏 reward**；没有 critic 的 GRPO 就只能把 seq-level reward broadcast 到所有 token（参考 `compute_grpo_outcome_advantage`）。

**Q4：`masked_whiten` 为什么放在 GAE 之后而不是 reward 之前？**

如果先归一化 reward，会破坏 $\delta_t = r_t + \gamma\hat V_{t+1} - \hat V_t$ 的物理意义（$r$ 和 $V$ 的量纲必须一致）。而 advantage 是无量纲的相对量，归一化只是数值稳定技巧，所以放在最后。

**Q5：`with torch.no_grad()` 包多大范围？**

包整段：advantage、returns 都不该带梯度。critic 的 MSE 是 `(vpreds - returns.detach())^2` 风格——`returns` 当作常数 target，critic 通过 `vpreds` 那边自己的 forward 反传。actor 同理。

---

## 9. 一个最小数值示例

$\gamma=1, \lambda=0.95$，response 长度 4：

- `rewards = [0, 0, 0, 1.0]`（只 EOS 给 1）
- `values  = [0.4, 0.5, 0.6, 0.7]`

反向递推：

| $t$ | $\hat V_{t+1}$ | $\delta_t$ | $\hat A_t$ |
|---|---|---|---|
| 3 | 0 | $1 + 0 - 0.7 = 0.3$ | $0.3$ |
| 2 | 0.7 | $0 + 0.7 - 0.6 = 0.1$ | $0.1 + 0.95\cdot 0.3 = 0.385$ |
| 1 | 0.6 | $0 + 0.6 - 0.5 = 0.1$ | $0.1 + 0.95\cdot 0.385 \approx 0.466$ |
| 0 | 0.5 | $0 + 0.5 - 0.4 = 0.1$ | $0.1 + 0.95\cdot 0.466 \approx 0.542$ |

得到：

- `advantages = [0.542, 0.466, 0.385, 0.300]`
- `returns    = advantages + values = [0.942, 0.966, 0.985, 1.000]`

可以看到 `returns` 平滑地朝最终 reward = 1 收敛——这就是"信号从 EOS 反向传播"的直观体现。

> 建议在 ipython 里 $B=2, T=4$ 手敲一次，把代码输出和上表对齐，肌肉记忆就有了。

---

## 10. 一句话总结

> **Advantage = "Q − V"。MC 用真实回报 $G_t$ 估 Q；GAE 通过对 TD 残差做 $(\gamma\lambda)^k$ 加权累加，用一个 λ 在"信 critic（TD）"和"信真实回报（MC）"之间连续滑动。代码用 $\hat A_t = \delta_t + \gamma\lambda\,\hat A_{t+1}$ 的反向递推在 O(T) 内算完。**
