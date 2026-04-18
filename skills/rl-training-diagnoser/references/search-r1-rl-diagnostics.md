# Search-R1 RL Diagnostics

This reference captures Search-R1/verl-specific interpretations. Always prefer local code truth over this file if they diverge.

## KL Paths

- With `use_kl_loss=false`, KL is applied in the reward path: `token_level_rewards = token_level_scores - beta * token_kl`. For paper writing, score is usually nonzero only at the final trainable token, and GRPO aggregates token rewards into a sequence reward. The effective sequence objective is close to `final_score - beta * sum_token_KL`.
- With `use_kl_loss=true`, KL is not subtracted from reward. The actor adds a KL loss term, usually controlled by `actor.kl_loss_coef`. In this mode, interpret `critic/rewards/mean` closer to task score and monitor `actor/kl_loss`, `actor/ppo_kl`, `actor/pg_clipfrac`, and `actor/grad_norm` for drift.
- `algorithm.kl_ctrl.kl_coef` is the reward-path KL coefficient. It matters mainly when `use_kl_loss=false`.
- `actor.kl_loss_coef` is the actor-loss KL coefficient. It matters when `use_kl_loss=true`.

## GRPO Credit Assignment

- `compute_grpo_outcome_advantage` sums token-level rewards over each response, normalizes scores within a prompt group, then broadcasts the resulting sequence-level advantage back to valid response tokens.
- If a penalty is computed from a whole trajectory, GRPO does not know which round or token caused it unless the reward design or rollout structure makes that obvious.
- Token-level actor loss aggregation can still be `token_mean`; that does not remove sequence-level credit assignment in GRPO advantage construction.

## Common Metric Patterns

- `response_length/mean` or `global_seqlen/mean` steadily down plus entropy down usually means the policy is becoming conservative or exploiting shorter outputs.
- `critic/score/mean` up but `critic/rewards/mean` and `critic/advantages/mean` down under `use_kl_loss=false` usually means KL penalty is larger than score improvement.
- `actor/pg_clipfrac`, `actor/ppo_kl`, `actor/kl_loss`, and `actor/grad_norm` rising together indicates actor updates are increasingly constrained or unstable.
- `actor/pg_loss` oscillating upward with rising clipfrac and KL is usually not a clean improvement signal; inspect reward variance, KL coefficient, learning rate, and batch composition.
- `actor/entropy_loss` decreasing usually means lower output diversity. Combined with shorter responses, it supports a mode-collapse or template-collapse hypothesis.
- `critic/advantages/mean` decreasing does not directly mean the model is optimizing to reduce advantages. Advantages are batch-relative training signals; next-rollout reward quality and normalization determine whether they improve.

## Paper Writing Specifics

- Paper-writing reward has historically placed the rubric score on the last trainable token. If length or format penalties are added as scalar score modifications, they affect the whole trajectory after GRPO aggregation.
- A draft format penalty that is equal across rounds may improve later rounds first because later rounds receive corrective observations and are easier to fix. Round 1 has weaker credit assignment and no previous correction unless explicitly added.
- First-round reminder or inserted instruction text can affect generation even if `info_mask=0`. `state_masking=true` prevents training loss on masked inserted tokens; it does not hide them from the model context.
- If Round 1 invalid increases while Round 2/3 invalid decrease, inspect whether the rollout correction loop creates a beneficial path: invalid early draft, receive correction, recover later, still score well on final camera-ready output.
- Length penalties should usually be mild and applied to each draft plus final camera-ready output if the failure mode is globally shortening responses.

## Recommended Interventions

- For uncontrolled KL drift, prefer `use_kl_loss=true`, raise `actor.kl_loss_coef` gradually, lower actor learning rate, or reduce PPO epochs/minibatch aggressiveness.
- For short-output reward hacking, add a mild length reward relative to ground truth and monitor score, response length, and invalid format together.
- For Round 1 draft invalidity, use a stronger but still moderate first-round format penalty than later rounds, rather than equal large weights.
- For noisy reward or high variance, inspect score distribution, group size, invalid samples per prompt, and whether penalties dominate rubric score.
- For loss aggregation concerns, verify actual actor loss masking and reduction in `dp_actor.py` and `core_algos.py` before changing aggregation.
