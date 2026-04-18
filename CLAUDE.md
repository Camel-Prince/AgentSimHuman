# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Search-R1 is a reinforcement learning framework for training LLMs to reason and call search engines interleaved. It extends DeepSeek-R1 with integrated search engine access, enabling multi-turn tool-augmented reasoning on QA tasks (NQ, HotpotQA, etc.).

## Environment Setup

Two separate conda environments are required:

**Training environment** (Python 3.9):
```bash
conda create -n searchr1 python=3.9
pip install torch==2.4.0 vllm==0.6.3 flash-attn wandb
pip install -e .  # installs verl in editable mode
```

**Retriever environment** (Python 3.10, separate due to dependency conflicts):
```bash
conda create -n retriever python=3.10
pip install transformers datasets pyserini faiss-gpu uvicorn fastapi
```

## Common Commands

```bash
# Download corpus and pre-built index
python scripts/download.py --save_path $save_path

# Launch retrieval server (run in retriever env, port 8000)
bash retrieval_launch.sh

# Train with PPO
bash train_ppo.sh

# Train with GRPO
bash train_grpo.sh

# Run inference
python infer.py
```

## Architecture

The project has three layers:

**1. veRL Framework** (`verl/`)
The underlying distributed RL training infrastructure. Key entry points:
- `verl/trainer/main_ppo.py` — main training entry point (Hydra config)
- `verl/trainer/ppo/ray_trainer.py` — Ray-based distributed trainer coordinating actor/critic/rollout workers
- `verl/trainer/ppo/core_algos.py` — PPO/GRPO/reinforce algorithm implementations
- `verl/workers/` — Actor, critic, rollout (vLLM), and reward model workers
- `verl/trainer/config/ppo_trainer.yaml` — base Hydra config (overridden by train scripts)

**2. Search-R1 Agent Layer** (`search_r1/`)
- `search_r1/llm_agent/generation.py` — `LLMGenerationManager`: orchestrates multi-turn generation loops, parses `<search>query</search>` and `<answer>content</answer>` tags from LLM output, calls the retrieval server, and formats results as `<information>...</information>` context
- `search_r1/search/retrieval_server.py` — FastAPI server for local dense (e5) and sparse (BM25/pyserini) retrieval with FAISS indexing
- `search_r1/search/serp_search_server.py` / `google_search_server.py` — online search backends (SerpAPI, Google Custom Search)
- `search_r1/search/rerank_server.py` — optional neural reranker

**3. Data Pipeline** (`scripts/data_process/`)
Converts NQ/HotpotQA datasets to parquet format with prompts and ground-truth answers.

## Training Data Flow

```
Parquet dataset (prompts + ground truth)
    → LLMGenerationManager (multi-turn loop, up to max_turns)
        → LLM generates reasoning + <search>query</search> or <answer>...</answer>
        → If search: POST to retrieval_server → FAISS/BM25 → top-k passages
        → Append <information>...</information> to context, repeat
    → RewardManager (exact match score vs ground truth)
    → PPO/GRPO update via veRL (Ray distributed, vLLM rollout, FSDP)
```

## Key Configuration Parameters

Set via shell script overrides to Hydra (see `train_ppo.sh`, `train_grpo.sh`):

| Parameter | Typical Value | Notes |
|---|---|---|
| `max_prompt_length` | 4096 | Input context length |
| `max_response_length` | 500 | Per-turn generation length |
| `max_obs_length` | 500 | Search result truncation |
| `max_turns` | 2–4 | Multi-turn search iterations |
| `retriever.url` | `http://127.0.0.1:8000/retrieve` | Retrieval server endpoint |
| `retriever.topk` | 3 | Passages returned per query |
| `actor_rollout_ref.rollout.n` | 1 (PPO) / 5 (GRPO) | Rollout samples per prompt |

## Retriever Backends

Three options documented in `docs/retriever.md`:
- **Local dense** (default): e5 embeddings + FAISS — requires pre-built index
- **Local sparse**: BM25 via pyserini — requires Elasticsearch or Lucene index
- **Online**: SerpAPI or Google Custom Search — requires API keys, no local index needed

## Multi-node Training

See `docs/multinode.md` and `example/` for Ray cluster setup. The Ray head node address must be set before launching workers.

## Paper-Writing Memory (Project-Specific)

Paper-writing task now supports multiple rollout modes via `task_type`:

- `paper_writing`: baseline multi-round draft-comment loop
- `paper_writing_last_round_target`: only last draft/comment kept in training trace
- `paper_writing_train_commenter`: generator via API, local model trained as commenter
- `paper_writing_arena_seeded`: baseline rollout plus seeded Swiss arena ranking metadata

Reward options:

- `paper_writing`: rubric-only reward on camera-ready text
- `paper_writing_arena_hybrid`: `0.7 * arena + 0.3 * rubric` (default weights)

Common extra Hydra knobs used by these modes:

- `num_revision_rounds`
- `generator_max_concurrency`
- `arena_seed_mode`, `arena_seed`, `arena_group_size`
- `arena_weight`, `rubric_weight`
