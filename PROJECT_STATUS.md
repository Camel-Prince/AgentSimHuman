# Project Status

This file is a compact handoff map for the current project state. Keep it short and current.

Maintenance rules:
- Record current state, key files, and design constraints only.
- Do not append full project history, long logs, or chat transcripts.
- Do not record secrets, API keys, private URLs with credentials, or full W&B metadata.
- Updating this file is not permission to modify code. Code changes still require an explicit request.
- Prefer replacing stale bullets over adding new historical bullets.

## Current Focus

- Main task area: Search-R1 paper-writing RL with multi-round draft/comment rollout.
- Current preferred rollout mode: `TASK_TYPE=paper_writing_per_segment`.
- Per-segment credit assignment: drafts and camera-ready participate in loss (info_mask=1); comments/instructions are masked (info_mask=0). Each segment gets independent reward signal via `segment_ids`.
- Segment label encoding: draft_i = `2*round_idx+1`, comment_i = `2*round_idx+2`, camera_ready = `2*num_rounds+1`.
- Evaluator is being moved toward ground-truth-aware scoring so short/generic abstracts do not receive high reward.
- High-quality trajectories should be saved for possible later SFT only when format, length, and score filters pass.
- SFT candidate JSONL schema: `comments` field has been removed; `comment_observations` (XML-wrapped comment + round instruction) is the only comment representation kept.

## Workspace Map

- `search_r1/llm_agent/generation.py`: rollout loops, paper-writing interaction flow, `info_mask` construction, draft/comment/camera-ready metadata.
- `verl/trainer/main_ppo.py`: reward manager, paper-writing rubric scoring, length/format penalties, SFT candidate filtering.
- `verl/trainer/ppo/ray_trainer.py`: PPO/GRPO training loop, task-type dispatch, KL/reward/advantage plumbing.
- `verl/trainer/ppo/core_algos.py`: GRPO advantage computation and sequence-level reward aggregation.
- `verl/workers/actor/dp_actor.py`: actor policy loss, KL loss, masking, token-level loss aggregation.
- `train_paper_writing_2gpu.sh` and `train_paper_writing_4gpu.sh`: main paper-writing launch scripts and Hydra overrides.
- `monitor_outputs/rl_metrics_parsed.csv`: parsed local metric snapshot used for quick training diagnosis.
- `skills/rl-training-diagnoser/`: local skill for RL metrics and run-state diagnosis.
- `outputs/sft_candidates/`: target directory for filtered SFT candidate JSONL files.
- `scripts/clean_sft_candidates.py`: post-processing script to strip redundant fields from SFT candidate JSONL files; supports `--watch` mode for use alongside a running training job.

## Critical Code Paths

- `run_llm_loop_paper_writing`: legacy accumulated-context paper-writing rollout; drafts and camera-ready can be trainable depending on mask construction.
- `run_llm_loop_paper_writing_per_segment`: accumulated-context rollout where draft and camera-ready tokens are trainable with per-segment credit assignment; comment/instruction tokens are masked.
- `_compose_final_output`: combines prompts, responses, attention mask, position ids, and `info_mask`.
- `_compute_paper_writing_reward`: per-segment reward placement. Camera-ready reward (rubric + length penalty) at camera-ready last token; per-draft penalties (format + length) at each draft's last token via `segment_ids`. Falls back to single last-token reward when `segment_ids` absent.
- `_call_rubric_scoring_api`: calls the rubric evaluator; current intent is ground-truth-aware scoring with dimension subscores.
- `_paper_writing_length_penalties`: computes soft length penalties relative to ground-truth length for drafts and camera-ready output. Draft length weights: [0.25, 0.15, 0.10], camera-ready weight: 0.50.
- `_paper_writing_format_penalty`: penalizes invalid draft format by round. Format weights: [0.08, 0.03, 0.02] for rounds 1-3.
- `_maybe_save_sft_candidates`: saves high-quality trajectories for later SFT when filters pass. No longer writes `comments` field (removed as redundant; `comment_observations` already contains the same text wrapped in context).
- `compute_grpo_outcome_advantage`: sums token-level rewards to sequence-level rewards, normalizes by prompt group, and broadcasts advantage to valid tokens.
- `compute_grpo_paper_writing_advantage`: per-segment advantage. Camera-ready advantage normalized and broadcast to ALL response tokens; draft_i advantage normalized and broadcast ONLY to draft_i tokens. Final advantage = camera_advantage + draft_advantage.
- `compute_policy_loss` and actor update path in `dp_actor.py`: actor loss is reduced over masked valid response tokens; `info_mask` determines trainable regions when `state_masking=true`.

## Algorithm Notes

- With `use_kl_loss=false`, KL is subtracted in the reward path: effective sequence reward for paper writing is close to `final_score - beta * sum_token_KL`.
- With `use_kl_loss=true`, reward is not KL-penalized; KL is added as actor loss via `actor.kl_loss_coef`.
- `algorithm.kl_ctrl.kl_coef` matters mainly for reward-path KL; `actor.kl_loss_coef` matters for actor-loss KL.
- In GRPO, paper-writing scalar reward is sequence-level after aggregation; credit assignment across draft rounds is weak unless masking or reward design localizes the signal.
- `info_mask=0` prevents actor loss on those tokens, but the tokens remain visible in rollout context.
- `state_masking=true` uses `info_mask` to create the actor loss mask for search/paper-writing style rollouts.
- Per-segment RL trains both draft and camera-ready tokens. Camera-ready reward provides a global learning signal; per-draft penalties localize format/length feedback to the responsible draft tokens.
- Ground truth should be used by evaluator/reward and candidate filtering, not injected into generator-visible rollout context unless explicitly designing a supervised task.
- Equal strong format penalties across rounds can improve later rounds first because later rounds receive corrective observations; Round 1 usually needs separate handling or SFT.
- Token-mean actor loss aggregation is preferred over seq-mean-token-mean for this project because sequence length differences should not overweight short trajectories.

## Active Run And Monitoring

- Key stability metrics: `actor/kl_loss`, `actor/ppo_kl`, `actor/pg_clipfrac`, `actor/grad_norm`, `actor/entropy_loss`.
- Key quality metrics: `critic/score/mean`, `critic/rewards/mean`, `critic/advantages/mean`, rubric dimension means.
- Key behavior metrics: `response_length/mean`, `global_seqlen/mean`, per-round invalid draft rates, length penalty mean, format penalty mean.
- Watch for this bad pattern: KL rises, entropy drops, response length drops, Round 1 invalid rises, score does not improve.
- For current diagnostics, use `$rl-training-diagnoser` or run `skills/rl-training-diagnoser/scripts/summarize_rl_metrics.py`.
- Previous run (2026-04-17, camera_ready_only): collapsed around step 20 — entropy/score/response_len all dropped. Root cause: draft tokens had zero gradient (info_mask=0) and zero KL, causing unconstrained drift and format breakdown.
- Per-segment implementation ready (2026-04-18): segment_ids tracking, per-segment rewards, per-segment advantage. Not yet launched as a training run.

## Do Not Forget

- Do not expose API keys in logs, status files, candidate JSONL, or summaries.
- Do not assume clipping alone can stop policy drift; monitor KL and entropy together.
- Do not treat `critic/rewards/mean` the same way under `use_kl_loss=true` and `use_kl_loss=false`.
- Do not conclude `info_mask` hides text from the model; it only masks training loss.
- Do not let SFT candidates include trajectories with invalid draft tags, large length mismatch, or low rubric score.
- Do not re-add `comments` to the SFT candidate schema; `comment_observations` already contains the same text plus XML tags and round instruction.
- Do not turn this file into an experiment diary; keep it as a current-state map.
