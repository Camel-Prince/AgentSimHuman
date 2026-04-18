
## Experiment log

### Preliminary results 

Resources: [wandb](https://wandb.ai/peterjin/Search-R1-open)


The preliminary experiment is conducted only on natural question (NQ) dataset (+ PPO) with a small number of training steps.


### v0.1

Resources: [wandb](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train), [docs](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa), [scripts](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa/v0.1)


We extend the experiments from NQ to seven datasets with both PPO and GRPO methods. The studies are still on a small number of training steps with a big learning rate warm up ratio.


### v0.2

Resources: [wandb](https://wandb.ai/peterjin/Search-R1-v0.2), [docs](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa), [scripts](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa/v0.2), [paper](https://arxiv.org/abs/2503.09516)


We fix several bugs including [retrieved token masking](https://github.com/PeterGriffinJin/Search-R1/pull/21) and [GRPO sample indexing](https://github.com/PeterGriffinJin/Search-R1/commit/9ec2fa9892fbf0315d0c67b4dc08ae8f6cf5f378). 
The former can largely improve the stablity of RL training. 
Then we adjust the training scripts, increasing the number of training steps and decreasing the learning rate warm up ratio, to obtain a better performance, and conduct experiments on different scale of LLMs (3B, 7B, 14B).


### v0.3

Resources: [wandb](https://wandb.ai/peterjin/Search-R1-v0.3), [docs](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa), [scripts](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa/v0.3), [paper](https://arxiv.org/abs/2505.15117)

We conduct studies on (1) reward design; (2) LLM backbone; and (3) search engine.

- Reward design
  - Format reward
  - Intermediate retrieval reward
- LLM backbone
  - LLM type (e.g., general LLM or reasoning LLM)
  - LLM scale (3B/7B/14B/32B)
- Search engine
  - RL training dynamics
  - generalization during inference
- Data scaling

Details can be found in the [paper](https://arxiv.org/abs/2505.15117).

### Paper Writing Rollout Extensions (internal memory)

This repo now contains three paper-writing rollout variants for controlled ablations:

- `paper_writing_last_round_target`
  - Motivation: reduce overfitting to long multi-round templates.
  - Behavior: rollout still uses multi-round context, but final training trace keeps only the last round draft/comment pair.

- `paper_writing_train_commenter`
  - Motivation: switch optimization target from generator to commenter.
  - Behavior: generator uses API calls; local actor is trained to output `<comment>...</comment>` only.

- `paper_writing_arena_seeded`
  - Motivation: reduce evaluator noise using relative ranking with an anchor.
  - Behavior: adds seeded single-round Swiss arena scores to `meta_info` with dataset `ground_truth` as anchor.

Hybrid reward path:

- `reward_type=paper_writing_arena_hybrid`
  - Score = `0.7 * arena_score + 0.3 * rubric_score` by default.
  - Falls back to rubric-only scoring if arena scores are missing.
