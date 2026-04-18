---
name: rl-training-diagnoser
description: Use when analyzing RL training status from W&B or local logged metrics, especially actor KL loss, PPO KL, grad norm, clipfrac, critic score/reward/advantages, response length, global sequence length, entropy, invalid draft rates, and diagnosing failures by checking run config and Search-R1/verl implementation.
---

# RL Training Diagnoser

Use this skill to diagnose whether a Search-R1/verl RL run is healthy, unstable, reward-hacking, or misconfigured.

## Workflow

1. Locate the run artifacts.
   - Prefer a user-specified metrics CSV or log.
   - If omitted, look for `monitor_outputs/rl_metrics_parsed.csv`, recent `*.log`, and `wandb/run-*/files/output.log`.
   - Do not expose secrets from logs; redact API keys and tokens.

2. Generate a metrics summary.
   - Run `python skills/rl-training-diagnoser/scripts/summarize_rl_metrics.py --metrics-csv <csv> --log <optional-log> --format markdown`.
   - Use `--format json` when a structured intermediate is easier to reason over.
   - Treat missing metrics as a signal about logging coverage, not as zero values.

3. Inspect training configuration when needed.
   - Check `use_kl_loss`, `actor.kl_loss_coef`, `algorithm.kl_ctrl.kl_coef`, `adv_estimator`, `actor.state_masking`, learning rate, batch size, rollout count, loss aggregation, and reward weights.
   - For paper writing runs, inspect reward/rollout code if symptoms involve length, draft format, or camera-ready behavior.

4. Connect metrics to implementation.
   - Read `references/search-r1-rl-diagnostics.md` for project-specific mappings.
   - Verify the relevant code before making strong claims if the user asks for root cause.

5. Report with this structure.
   - Run identity and important config.
   - Metric diagnosis with concrete first/mid/last or slope evidence.
   - Behavioral symptoms.
   - Likely causes tied to config/code.
   - Recommended fixes ordered by expected impact and risk.
   - Unknowns or missing logs that limit confidence.

## Key Files To Check

- `verl/trainer/main_ppo.py` for paper-writing reward and logged custom metrics.
- `verl/trainer/ppo/ray_trainer.py` for KL penalty application and reward path.
- `verl/trainer/ppo/core_algos.py` for GRPO advantage aggregation.
- `verl/workers/actor/dp_actor.py` for actor loss, KL loss, and masking.
- `search_r1/llm_agent/generation.py` for rollout loop, inserted instructions, and `info_mask`.
- `train_paper_writing_*.sh` for default run parameters.
