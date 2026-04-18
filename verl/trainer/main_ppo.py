# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import openai
import pdb
import json
import os


def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0.,
                 reward_type='paper_writing',
                 rubric_api_key=None, rubric_api_base=None, rubric_model='qwen-max',
                 rubric_max_concurrency=32,
                 arena_weight=0.7,
                 rubric_weight=0.3,
                 experiment_name='default',
                 save_sft_candidates=False,
                 sft_score_threshold=0.78,
                 sft_output_dir='outputs/sft_candidates') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.reward_type = reward_type  # 'qa_em' or 'paper_writing'
        self.rubric_api_key = rubric_api_key
        self.rubric_api_base = rubric_api_base
        self.rubric_model = rubric_model
        # Max concurrent rubric API calls; configurable via Hydra config.
        self.rubric_max_concurrency = rubric_max_concurrency
        self.arena_weight = arena_weight
        self.rubric_weight = rubric_weight
        self.experiment_name = experiment_name
        if isinstance(save_sft_candidates, str):
            self.save_sft_candidates = save_sft_candidates.lower() in ('1', 'true', 'yes', 'y', 'on')
        else:
            self.save_sft_candidates = bool(save_sft_candidates)
        self.sft_score_threshold = float(sft_score_threshold)
        self.sft_output_dir = sft_output_dir
        # Long-lived rubric client, lazily initialized on first use.
        self._rubric_client = None

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # Route to different reward functions based on reward_type
        if self.reward_type == 'paper_writing':
            return self._compute_paper_writing_reward(data)
        if self.reward_type == 'paper_writing_arena_hybrid':
            return self._compute_paper_writing_arena_hybrid_reward(data)
        else:
            return self._compute_qa_em_reward(data)

    def _get_last_trainable_response_index(self, data_item) -> int:
        """Return the last response index that is trainable under info_mask.

        Fallback to the last valid response token if all response tokens are masked.
        """
        prompt_len = data_item.batch['prompts'].shape[-1]
        attn_resp = data_item.batch['attention_mask'][prompt_len:]
        valid_len = int(attn_resp.sum().item())
        if valid_len <= 0:
            return 0

        if 'info_mask' not in data_item.batch:
            return valid_len - 1

        info_resp = data_item.batch['info_mask'][prompt_len:prompt_len + valid_len]
        trainable_indices = (info_resp > 0).nonzero(as_tuple=False).flatten()
        if trainable_indices.numel() == 0:
            return valid_len - 1
        return int(trainable_indices[-1].item())

    def _compute_qa_em_reward(self, data: DataProto):
        """Original QA exact match reward function."""
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor

    @staticmethod
    def _to_plain_list(value):
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    @staticmethod
    def _single_length_penalty(output_len, gt_len):
        """Soft length penalty relative to the ground-truth abstract length."""
        try:
            output_len = float(output_len)
            gt_len = float(gt_len)
        except (TypeError, ValueError):
            return 0.0
        if gt_len <= 0:
            return 0.0

        ratio = output_len / gt_len
        if ratio < 0.80:
            return -0.50 * min((0.80 - ratio) / 0.80, 1.0)
        if ratio > 1.25:
            return -0.35 * min((ratio - 1.25) / 1.25, 1.0)
        return 0.0

    def _paper_writing_length_penalties(self, data: DataProto, sample_idx: int):
        draft_lengths_by_round = self._to_plain_list(
            data.meta_info.get('paper_writing_draft_lengths', [])
        )
        camera_ready_lengths = self._to_plain_list(
            data.meta_info.get('paper_writing_camera_ready_lengths', [])
        )
        ground_truth_lengths = self._to_plain_list(
            data.meta_info.get('paper_writing_ground_truth_lengths', [])
        )

        if sample_idx >= len(ground_truth_lengths):
            return 0.0, 0.0, 0.0
        gt_len = ground_truth_lengths[sample_idx]

        draft_weights = [0.25, 0.15, 0.10]
        weighted_penalty = 0.0
        total_weight = 0.0
        draft_weighted_penalty = 0.0
        draft_total_weight = 0.0

        for round_idx, round_lengths in enumerate(draft_lengths_by_round):
            if sample_idx >= len(round_lengths):
                continue
            weight = draft_weights[round_idx] if round_idx < len(draft_weights) else 0.05
            penalty = self._single_length_penalty(round_lengths[sample_idx], gt_len)
            weighted_penalty += weight * penalty
            total_weight += weight
            draft_weighted_penalty += weight * penalty
            draft_total_weight += weight

        camera_penalty = 0.0
        if sample_idx < len(camera_ready_lengths):
            camera_penalty = self._single_length_penalty(camera_ready_lengths[sample_idx], gt_len)
            weighted_penalty += 0.50 * camera_penalty
            total_weight += 0.50

        if total_weight <= 0:
            total_penalty = 0.0
        else:
            total_penalty = weighted_penalty / total_weight

        draft_penalty = draft_weighted_penalty / draft_total_weight if draft_total_weight > 0 else 0.0
        return total_penalty, draft_penalty, camera_penalty

    def _paper_writing_format_penalty(self, data: DataProto, sample_idx: int):
        valid_flags_by_round = self._to_plain_list(
            data.meta_info.get('paper_writing_draft_valid_flags', [])
        )
        weights = [0.08, 0.03, 0.02]
        penalty = 0.0
        for round_idx, round_flags in enumerate(valid_flags_by_round):
            if sample_idx >= len(round_flags):
                continue
            is_valid = bool(round_flags[sample_idx])
            if not is_valid:
                penalty -= weights[round_idx] if round_idx < len(weights) else 0.01
        return penalty

    def _print_paper_writing_reward_stats(self, data: DataProto, length_penalties,
                                          draft_length_penalties, camera_length_penalties,
                                          format_penalties, rubric_details=None):
        valid_flags_by_round = self._to_plain_list(
            data.meta_info.get('paper_writing_draft_valid_flags', [])
        )
        pieces = []
        for round_idx, round_flags in enumerate(valid_flags_by_round[:3]):
            if round_flags:
                invalid_rate = 1.0 - float(np.mean([bool(x) for x in round_flags]))
                pieces.append(f"round{round_idx + 1}_invalid_rate={invalid_rate:.4f}")
        if length_penalties:
            pieces.append(f"length_penalty_mean={float(np.mean(length_penalties)):.4f}")
            pieces.append(f"draft_length_penalty_mean={float(np.mean(draft_length_penalties)):.4f}")
            pieces.append(f"camera_ready_length_penalty_mean={float(np.mean(camera_length_penalties)):.4f}")
        if format_penalties:
            pieces.append(f"format_penalty_mean={float(np.mean(format_penalties)):.4f}")
        if rubric_details:
            dimensions = [
                'Problem & Motivation',
                'Method & Contribution Coverage',
                'Results & Evidence Coverage',
                'Faithfulness to Reference',
                'Clarity & Conciseness',
                'Length Appropriateness',
            ]
            for dim in dimensions:
                vals = [
                    float(item.get('subscores', {}).get(dim))
                    for item in rubric_details
                    if item.get('subscores', {}).get(dim) is not None
                ]
                if vals:
                    metric_name = dim.lower().replace(' & ', '_').replace(' ', '_')
                    pieces.append(f"rubric_{metric_name}_mean={float(np.mean(vals)):.4f}")
        if pieces:
            print("[Paper Writing Reward] " + " ".join(pieces))

        # Build metrics dict for wandb logging
        wandb_metrics = {}
        for piece in pieces:
            key, val = piece.split('=')
            wandb_metrics[f'paper_writing_reward/{key}'] = float(val)
        return wandb_metrics

    def _length_ratio_ok(self, length_value, gt_len, low=0.75, high=1.35):
        try:
            length_value = float(length_value)
            gt_len = float(gt_len)
        except (TypeError, ValueError):
            return False, None
        if gt_len <= 0:
            return False, None
        ratio = length_value / gt_len
        return low <= ratio <= high, ratio

    def _get_nested_round_value(self, nested, round_idx, sample_idx, default=None):
        if round_idx >= len(nested):
            return default
        round_values = nested[round_idx]
        if sample_idx >= len(round_values):
            return default
        return round_values[sample_idx]

    def _maybe_save_sft_candidates(self, data: DataProto, final_scores, rubric_details,
                                   length_penalties, format_penalties):
        if not self.save_sft_candidates:
            return

        valid_flags_by_round = self._to_plain_list(
            data.meta_info.get('paper_writing_draft_valid_flags', [])
        )
        draft_lengths_by_round = self._to_plain_list(
            data.meta_info.get('paper_writing_draft_lengths', [])
        )
        camera_ready_lengths = self._to_plain_list(
            data.meta_info.get('paper_writing_camera_ready_lengths', [])
        )
        ground_truth_lengths = self._to_plain_list(
            data.meta_info.get('paper_writing_ground_truth_lengths', [])
        )
        draft_texts_by_round = self._to_plain_list(
            data.meta_info.get('paper_writing_draft_texts', [])
        )
        comment_obs_by_round = self._to_plain_list(
            data.meta_info.get('paper_writing_comment_obs_texts', [])
        )
        camera_ready_texts = self._to_plain_list(data.meta_info.get('camera_ready_texts', []))
        ground_truth_texts = self._to_plain_list(
            data.meta_info.get('paper_writing_ground_truth_texts', [])
        )
        domains = self._to_plain_list(data.meta_info.get('paper_writing_domains', []))
        topics = self._to_plain_list(data.meta_info.get('paper_writing_topics', []))

        records = []
        for i, score in enumerate(final_scores):
            rubric_overall = float(rubric_details[i].get('overall', 0.0)) if i < len(rubric_details) else 0.0
            if rubric_overall < self.sft_score_threshold or float(score) == 0.0:
                continue
            if i >= len(ground_truth_lengths) or i >= len(camera_ready_lengths):
                continue
            gt_len = ground_truth_lengths[i]
            all_valid = True
            draft_ratios = []
            for round_idx, round_flags in enumerate(valid_flags_by_round):
                if i >= len(round_flags) or not bool(round_flags[i]):
                    all_valid = False
                    break
                draft_len = self._get_nested_round_value(draft_lengths_by_round, round_idx, i)
                ok, ratio = self._length_ratio_ok(draft_len, gt_len)
                if not ok:
                    all_valid = False
                    break
                draft_ratios.append(ratio)
            if not all_valid:
                continue
            camera_ok, camera_ratio = self._length_ratio_ok(camera_ready_lengths[i], gt_len)
            if not camera_ok:
                continue

            records.append({
                'experiment_name': self.experiment_name,
                'sample_index': i,
                'domain': domains[i] if i < len(domains) else None,
                'topic': topics[i] if i < len(topics) else None,
                'ground_truth': ground_truth_texts[i] if i < len(ground_truth_texts) else None,
                'drafts': [
                    self._get_nested_round_value(draft_texts_by_round, r, i, '')
                    for r in range(len(draft_texts_by_round))
                ],
                'comment_observations': [
                    self._get_nested_round_value(comment_obs_by_round, r, i, '')
                    for r in range(len(comment_obs_by_round))
                ],
                'camera_ready': camera_ready_texts[i] if i < len(camera_ready_texts) else None,
                'rubric_overall': rubric_overall,
                'rubric_subscores': rubric_details[i].get('subscores', {}) if i < len(rubric_details) else {},
                'draft_length_ratios': draft_ratios,
                'camera_ready_length_ratio': camera_ratio,
                'length_penalty': float(length_penalties[i]) if i < len(length_penalties) else None,
                'format_penalty': float(format_penalties[i]) if i < len(format_penalties) else None,
                'final_reward': float(score),
            })

        if not records:
            return

        os.makedirs(self.sft_output_dir, exist_ok=True)
        safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(self.experiment_name or 'default'))
        path = os.path.join(self.sft_output_dir, f'{safe_name}.jsonl')
        with open(path, 'a', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"[Paper Writing SFT] saved_candidates={len(records)} path={path}")

    def _compute_paper_writing_reward(self, data: DataProto):
        """
        Compute reward for paper writing task with per-segment credit assignment.

        Reward placement strategy (requires ``segment_ids`` in batch):
        - Camera-ready last token: rubric score + camera-ready length penalty
        - Draft i last token: draft_i format penalty + draft_i length penalty

        When ``segment_ids`` is absent, falls back to legacy single-reward-at-last-token.
        """

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # Get camera-ready texts from meta_info
        camera_ready_texts = data.meta_info.get('camera_ready_texts', [])

        if not camera_ready_texts:
            print("[WARNING] No camera_ready_texts found in meta_info, returning zero rewards")
            return reward_tensor

        ground_truth_texts = data.meta_info.get('paper_writing_ground_truth_texts', None)

        # Call rubric scoring API for all samples
        scores, rubric_details = self._call_rubric_scoring_api(
            camera_ready_texts,
            ground_truth_texts=ground_truth_texts,
            return_details=True,
        )


        has_segment_ids = 'segment_ids' in data.batch

        # Pre-compute per-round penalties
        draft_lengths_by_round = self._to_plain_list(
            data.meta_info.get('paper_writing_draft_lengths', []))
        camera_ready_lengths = self._to_plain_list(
            data.meta_info.get('paper_writing_camera_ready_lengths', []))
        ground_truth_lengths = self._to_plain_list(
            data.meta_info.get('paper_writing_ground_truth_lengths', []))
        valid_flags_by_round = self._to_plain_list(
            data.meta_info.get('paper_writing_draft_valid_flags', []))
        num_rounds = data.meta_info.get('num_revision_rounds', len(draft_lengths_by_round))

        length_penalties = []
        draft_length_penalties = []
        camera_length_penalties = []
        format_penalties = []
        final_scores = []

        prompt_len = data.batch['prompts'].shape[-1]

        for i in range(len(data)):
            data_item = data[i]
            gt_len = ground_truth_lengths[i] if i < len(ground_truth_lengths) else 0

            # --- Camera-ready reward ---
            camera_penalty = 0.0
            if i < len(camera_ready_lengths):
                camera_penalty = self._single_length_penalty(camera_ready_lengths[i], gt_len)
            camera_score = float(scores[i]) + camera_penalty

            # --- Per-draft penalties ---
            format_weights = [0.30, 0.20, 0.10]
            draft_len_weights = [0.50, 0.35, 0.25]
            per_draft_penalties = []
            total_format_penalty = 0.0
            total_draft_len_penalty = 0.0
            total_draft_len_weight = 0.0
            for r in range(num_rounds):
                fp = 0.0
                if r < len(valid_flags_by_round) and i < len(valid_flags_by_round[r]):
                    if not bool(valid_flags_by_round[r][i]):
                        fp = -(format_weights[r] if r < len(format_weights) else 0.01)
                lp = 0.0
                if r < len(draft_lengths_by_round) and i < len(draft_lengths_by_round[r]):
                    w = draft_len_weights[r] if r < len(draft_len_weights) else 0.05
                    lp = w * self._single_length_penalty(draft_lengths_by_round[r][i], gt_len)
                    total_draft_len_weight += w
                per_draft_penalties.append(fp + lp)
                total_format_penalty += fp
                total_draft_len_penalty += lp
            
            # --- Collect aggregate metrics (for logging compatibility) ---
            agg_draft_len_penalty = total_draft_len_penalty / total_draft_len_weight if total_draft_len_weight > 0 else 0.0
            total_weight = total_draft_len_weight + 0.50
            agg_len_penalty = (total_draft_len_penalty + 0.50 * camera_penalty) / total_weight if total_weight > 0 else 0.0
            length_penalties.append(agg_len_penalty)
            draft_length_penalties.append(agg_draft_len_penalty)
            camera_length_penalties.append(camera_penalty)
            format_penalties.append(total_format_penalty)
            # final_score for logging / SFT filtering (legacy aggregate)
            final_score = float(scores[i]) + agg_len_penalty + total_format_penalty
            final_scores.append(final_score)

            if has_segment_ids:
                # Place rewards at per-segment last tokens
                seg = data_item.batch['segment_ids'][prompt_len:]
                attn = data_item.batch['attention_mask'][prompt_len:]
                valid_len = int(attn.sum().item())
                seg_valid = seg[:valid_len]

                # Camera-ready: segment_label = 2*num_rounds + 1
                cam_label = 2 * num_rounds + 1
                cam_positions = (seg_valid == cam_label).nonzero(as_tuple=False).flatten()
                if cam_positions.numel() > 0:
                    reward_tensor[i, int(cam_positions[-1].item())] = camera_score
                else:
                    # Fallback: last valid token
                    reward_tensor[i, valid_len - 1] = camera_score

                # Per-draft: segment_label = 2*round_idx + 1
                for r in range(num_rounds):
                    draft_label = 2 * r + 1
                    draft_positions = (seg_valid == draft_label).nonzero(as_tuple=False).flatten()
                    if draft_positions.numel() > 0 and abs(per_draft_penalties[r]) > 1e-8:
                        reward_tensor[i, int(draft_positions[-1].item())] = per_draft_penalties[r]
            else:
                # Legacy fallback: single reward at last trainable token
                reward_index = self._get_last_trainable_response_index(data_item)
                reward_tensor[i, reward_index] = final_score
        os.environ.get('PW_DEBUG') and __import__('pdb').set_trace()
        pw_metrics = self._print_paper_writing_reward_stats(
            data,
            length_penalties,
            draft_length_penalties,
            camera_length_penalties,
            format_penalties,
            rubric_details=rubric_details,
        )
        self.paper_writing_metrics = pw_metrics if pw_metrics else {}
        self._maybe_save_sft_candidates(
            data,
            final_scores,
            rubric_details,
            length_penalties,
            format_penalties,
        )
        return reward_tensor

    def _compute_paper_writing_arena_hybrid_reward(self, data: DataProto):
        """Compute hybrid reward = arena_weight * arena + rubric_weight * rubric."""
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        camera_ready_texts = data.meta_info.get('camera_ready_texts', [])
        if not camera_ready_texts:
            print("[WARNING] No camera_ready_texts found in meta_info, returning zero rewards")
            return reward_tensor

        rubric_scores = self._call_rubric_scoring_api(
            camera_ready_texts,
            ground_truth_texts=data.meta_info.get('paper_writing_ground_truth_texts', None),
        )
        arena_scores = data.meta_info.get('arena_scores', None)
        if arena_scores is None or len(arena_scores) != len(camera_ready_texts):
            print("[WARNING] arena_scores missing or mismatched; fallback to rubric-only scores")
            arena_scores = [0.0 for _ in camera_ready_texts]
            arena_weight = 0.0
            rubric_weight = 1.0
        else:
            arena_weight = float(self.arena_weight)
            rubric_weight = float(self.rubric_weight)

        for i in range(len(data)):
            score = arena_weight * float(arena_scores[i]) + rubric_weight * float(rubric_scores[i])
            reward_index = self._get_last_trainable_response_index(data[i])
            reward_tensor[i, reward_index] = score
        return reward_tensor
    
    def _call_rubric_scoring_api(self, camera_ready_texts, ground_truth_texts=None, return_details=False):
        """
        Call external API to score papers based on rubric.
        Returns a list of scores (one per paper).
        """
        # Lazily create a long-lived client so we don't recreate it per sample.
        if self._rubric_client is None:
            self._rubric_client = openai.OpenAI(
                api_key=self.rubric_api_key,
                base_url=self.rubric_api_base
            )
        client = self._rubric_client
        
        rubric_prompt = (
            "You are a senior program committee member reviewing generated camera-ready abstracts for a top-tier academic venue "
            "(e.g., NeurIPS, ICML, ICLR, ACL). Evaluate the generated abstract strictly by comparing it with the reference abstract. "
            "The reference abstract is ground truth for the paper content, but the generated abstract may use different wording. "
            "Score each dimension from 0.0 to 1.0 (two decimal places).\n\n"

            "## Scoring Rubric\n\n"

            "### 1. Problem & Motivation\n"
            "Does the generated abstract preserve the reference's problem setting and motivation?\n\n"

            "### 2. Method & Contribution Coverage\n"
            "Does it cover the reference's main method, technical idea, and claimed contribution without replacing them with generic claims?\n\n"

            "### 3. Results & Evidence Coverage\n"
            "Does it include the reference's important empirical/theoretical results, evidence, and impact when present?\n\n"

            "### 4. Faithfulness to Reference\n"
            "Is it factually consistent with the reference, without hallucinating unsupported methods, datasets, results, or claims?\n\n"

            "### 5. Clarity & Conciseness\n"
            "Is it clear, coherent, self-contained, and written like a real academic abstract?\n\n"

            "### 6. Length Appropriateness\n"
            "Is its level of detail and length appropriate relative to the reference, without being much shorter, much longer, or overly template-like?\n\n"

            "Use 1.0 for excellent, 0.8 for good with minor omissions, 0.6 for partially adequate, "
            "0.4 for weak, 0.2 for very poor, and 0.0 for missing or unusable.\n\n"

            "## Output Format\n"
            "You MUST output scores in exactly the following format (no extra text before the scores):\n"
            "Problem & Motivation: [score]\n"
            "Method & Contribution Coverage: [score]\n"
            "Results & Evidence Coverage: [score]\n"
            "Faithfulness to Reference: [score]\n"
            "Clarity & Conciseness: [score]\n"
            "Length Appropriateness: [score]\n"
            "Summary: [2-3 sentence overall assessment]\n\n"

            "Example:\n"
            "Problem & Motivation: 0.80\n"
            "Method & Contribution Coverage: 0.70\n"
            "Results & Evidence Coverage: 0.60\n"
            "Faithfulness to Reference: 0.85\n"
            "Clarity & Conciseness: 0.75\n"
            "Length Appropriateness: 0.70\n"
            "Summary: The generated abstract preserves the main problem and is mostly faithful to the reference, "
            "but it omits some result details and is slightly too generic."
        ) 

        # Preserve output order: one score per paper in the same order as input.
        num_papers = len(camera_ready_texts)
        scores = [0.0] * num_papers
        details = [{'overall': 0.0, 'subscores': {}, 'summary': ''} for _ in range(num_papers)]
        if ground_truth_texts is None:
            ground_truth_texts = [''] * num_papers

        def _score_single(idx, paper_text, ground_truth_text):
            """Score a single paper using the rubric API."""
            try:
                user_content = (
                    "Reference abstract:\n\n"
                    f"{ground_truth_text or '[missing reference]'}\n\n"
                    "Generated camera-ready abstract to review:\n\n"
                    f"{paper_text}"
                )
                messages = [
                    {"role": "system", "content": rubric_prompt},
                    {"role": "user", "content": user_content}
                ]

                response = client.chat.completions.create(
                    model=self.rubric_model,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.3,
                    extra_body={"enable_thinking": False}
                )

                # Parse scores from response
                score_text = response.choices[0].message.content
                detail = self._parse_rubric_scores(score_text)
                score = float(detail.get('overall', 0.0))
                # if score == 0:
                #     pdb.set_trace()
                return score, detail
            except Exception as e:
                # Keep existing behavior style: print error and return 0.0 for this sample.
                print(f"[ERROR] Rubric scoring API failed: {e}")
                return 0.0, {'overall': 0.0, 'subscores': {}, 'summary': ''}

        max_workers = min(self.rubric_max_concurrency, num_papers) if num_papers > 0 else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _score_single,
                    idx,
                    paper_text,
                    ground_truth_texts[idx] if idx < len(ground_truth_texts) else ''
                ): idx
                for idx, paper_text in enumerate(camera_ready_texts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                score, detail = future.result()
                scores[idx] = score
                details[idx] = detail

        if return_details:
            return scores, details
        return scores
    
    def _parse_rubric_scores(self, score_text):
        """
        Parse rubric scores from API response and compute average.
        Expected format:
        Soundness: 0.75
        Significance: 0.60
        ...
        """
        import re
        
        dimensions = [
            'Problem & Motivation',
            'Method & Contribution Coverage',
            'Results & Evidence Coverage',
            'Faithfulness to Reference',
            'Clarity & Conciseness',
            'Length Appropriateness',
        ]
        scores = []
        subscores = {}
        
        for dim in dimensions:
            pattern = rf'{dim}:\s*([0-9.]+)'
            match = re.search(pattern, score_text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    scores.append(score)
                    subscores[dim] = score
                except ValueError:
                    pass
        
        if not scores:
            # pdb.set_trace()
            print(f"[WARNING] Failed to parse scores from: {score_text[:200]}...")
            return {'overall': 0.0, 'subscores': {}, 'summary': ''}
        
        # Return average of all dimensions
        avg_score = sum(scores) / len(scores)
        summary_match = re.search(r'Summary:\s*(.*)', score_text, re.IGNORECASE | re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ''
        return {'overall': avg_score, 'subscores': subscores, 'summary': summary}



import ray
import hydra
import os


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    # import os
    # print("DEBUG ENV - CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET'))
    # print("DEBUG ENV - RAY_NUM_GPUS:", os.environ.get('RAY_NUM_GPUS', 'NOT SET'))
    if not ray.is_initialized():
        # this is for local ray cluster
        print(
            f"[Ray Init] RAY_ADDRESS={os.environ.get('RAY_ADDRESS', '')}, "
            f"RAY_GCS_SERVER_PORT={os.environ.get('RAY_GCS_SERVER_PORT', '')}, "
            f"RAY_DASHBOARD_PORT={os.environ.get('RAY_DASHBOARD_PORT', '')}, "
            f"RAY_TMPDIR={os.environ.get('RAY_TMPDIR', '')}"
        )
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config)) # 正常流程，ray worker进程，断点不会触发
    # main_task(config) # DEBUG: 主进程中运行，断点正常


@ray.remote  # DEBUG: commented out to run in main process for breakpoint debugging
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        reward_type=config.get('reward_type', 'paper_writing'),
        rubric_api_key=config.get('rubric_api_key'),
        rubric_api_base=config.get('rubric_api_base'),
        rubric_model=config.get('rubric_model', 'qwen-max'),
        rubric_max_concurrency=config.get('rubric_max_concurrency', 32),
        arena_weight=config.get('arena_weight', 0.7),
        rubric_weight=config.get('rubric_weight', 0.3),
        experiment_name=config.trainer.get('experiment_name', 'default'),
        save_sft_candidates=config.get('paper_writing_save_sft_candidates', False),
        sft_score_threshold=config.get('paper_writing_sft_score_threshold', 0.78),
        sft_output_dir=config.get('paper_writing_sft_output_dir', 'outputs/sft_candidates'),
    )

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        reward_type=config.get('reward_type', 'paper_writing'),
        rubric_api_key=config.get('rubric_api_key'),
        rubric_api_base=config.get('rubric_api_base'),
        rubric_model=config.get('rubric_model', 'qwen-max'),
        rubric_max_concurrency=config.get('rubric_max_concurrency', 32),
        arena_weight=config.get('arena_weight', 0.7),
        rubric_weight=config.get('rubric_weight', 0.3),
        experiment_name=config.trainer.get('experiment_name', 'default'),
        save_sft_candidates=False,
        sft_score_threshold=config.get('paper_writing_sft_score_threshold', 0.78),
        sft_output_dir=config.get('paper_writing_sft_output_dir', 'outputs/sft_candidates'),
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
