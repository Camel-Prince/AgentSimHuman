# Paper Writing Rollout/LogProb Alignment Plan

## Goal
在 paper-writing 任务中，同时满足：
1. `rollings` 与 `original_right_side` 采用同一条、单向累计轨迹。
2. rollout / old_log_prob / ref_log_prob / current_log_prob 使用一致上下文。
3. 每轮都明确约束模型只输出 `<draft>...</draft>`，降低“续写 `<comment>`”问题。

## Core Principle
不要双轨（一个给生成、一个给训练）。
只保留一条真实轨迹，并把所有用于控制生成行为的文本都显式写入这条轨迹。

---

## Proposed Design

### 1) Keep single-track accumulative trajectory
每轮同时更新两者：
- `rollings`：下一轮生成输入
- `original_right_side`：最终训练 `responses` / `responses_with_info_mask`

累积顺序建议：
`q, d1, c1, inst2, d2, c2, inst3, d3, ...`

其中：
- `d_i`：第 i 轮 draft
- `c_i`：commenter 反馈
- `inst_{i+1}`：下一轮控制指令（例如“只输出单个 `<draft>`”）

### 2) Introduce action-observation loop (borrowed from run_llm_loop)
参考 `run_llm_loop` 的关键思想：不是只靠 prompt，而是靠“合法输出判定 + 纠错反馈”。

定义唯一合法输出动作：
- 只允许一个 `<draft>...</draft>` block

每轮生成后：
- 若合法：进入 commenter 评分流程
- 若不合法（例如输出 `<comment>`、缺失闭合、多个块等）：
  - 本轮跳过 commenter
  - 直接注入纠错 observation（下一轮提示模型修正格式）

### 3) Keep round-specific instruction at the tail
每轮末尾追加简短控制指令作为 observation：
- 示例：`Round i: Please output only one <draft>...</draft>. Do not output <comment>.`

关键点：
- 该指令必须同时进入 `rollings` 和 `original_right_side`
- 否则 rollout 与 logprob 上下文不一致

### 4) Loss masking policy
为了不把 comment/控制提示当训练目标：
- `draft` 段：`loss_mask = 1`
- `comment` 段：`loss_mask = 0`
- `inst` 段：`loss_mask = 0`

### 5) Logprob computation strategy
停用“段级 old/ref 拼接”路径（可先注释保留代码）。
统一在最终累计序列上计算：
- `old_log_probs`
- `ref_log_prob`

并由 update 阶段计算 `current_log_prob`。

这样三者自然在同一条累计序列上对齐。

---

## Why this should work

1. **对齐正确性**
- 只要 `rollings` 与 `original_right_side` 同步累计，且 old/ref/current 基于同一最终序列计算，语义与位置就一致。

2. **指令遵循稳定性更高**
- 从“单纯 prompt 约束”升级为“动作合法性 + 纠错 observation”闭环，能更稳地把模型拉回 `<draft>` 通道。

3. **工程复杂度可控**
- 不需要维护两套上下文表示。
- 保留已有代码骨架（可通过注释停用旧逻辑），便于回退和对照实验。

---

## Minimal Implementation Steps

1. 新增 `validate_draft_output()`：
- 检查是否为单个 `<draft>...</draft>`
- 不合法时给出结构化错误类型（用于纠错 observation）

2. 在每轮生成后：
- 合法分支：正常 commenter
- 非法分支：跳过 commenter，注入纠错 observation

3. 每轮将 `inst_{i+1}` 追加到 observation（与 comment 一起进入累计轨迹）

4. 确保 `loss_mask` 屏蔽 `comment + inst`

5. 保持 trainer 的统一 old/ref 计算路径；不在 generation 中拼段级 old/ref

6. 加断言：
- `responses.shape == old_log_probs.shape == ref_log_prob.shape == loss_mask.shape`
- `input_ids[:, -R:] == responses`

---

## Risks & Mitigations

1. **上下文变长导致开销上涨**
- 通过 `max_prompt_length` 截断控制上限；必要时减少 revision rounds。

2. **纠错 observation 过长**
- 保持模板短且固定，避免占用过多上下文。

3. **仍有格式漂移**
- 增加非法输出统计（每 step 非法率）用于评估 prompt/策略效果。

---

## Open Questions for Advisor Discussion

1. 是否需要在 reward 中加入格式奖励（合法 `<draft>` 加分）？
2. 最大可接受的 revision rounds 与上下文长度预算是多少？
3. 是否需要将 commenter 输出也结构化（例如固定 3 条 bullet）来减少漂移？
