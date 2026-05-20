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
import threading
import time


def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class _SimpleRateLimiter:
    """线程安全的滑动窗口 RPM 限流器（独立版，避免跨文件依赖）。"""

    def __init__(self, rpm: int = 120):
        self.rpm = max(1, rpm)
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) >= self.rpm:
                wait = self._timestamps[0] - cutoff
                if wait > 0:
                    time.sleep(wait)
                now = time.monotonic()
                cutoff = now - 60.0
                self._timestamps = [t for t in self._timestamps if t > cutoff]
            self._timestamps.append(now)


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0.,
                 reward_type='paper_writing',
                 rubric_api_key=None, rubric_api_base=None, rubric_model='qwen-max',
                 rubric_max_concurrency=32,
                 rubric_rpm=120,
                 arena_weight=0.7,
                 rubric_weight=0.3,
                 experiment_name='default',
                 save_sft_candidates=False,
                 sft_score_threshold=0.78,
                 sft_output_dir='outputs/sft_candidates',
                 gpu_filler_enabled=True,
                 gpu_filler_gpu_ids=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.reward_type = reward_type  # 'qa_em' or 'paper_writing'
        self.rubric_api_key = rubric_api_key
        self.rubric_api_base = rubric_api_base
        self.rubric_model = rubric_model
        # Max concurrent rubric API calls; configurable via Hydra config.
        self.rubric_max_concurrency = rubric_max_concurrency
        self.rubric_rpm = rubric_rpm
        self._rubric_rate_limiter = _SimpleRateLimiter(rpm=rubric_rpm)
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

        from verl.utils.gpu_idle_filler import GPUIdleFiller
        from pathlib import Path
        _agent_sas_path = Path(__file__).parent.parent.parent / 'agent_SAS.py'
        self._gpu_filler = GPUIdleFiller(
            agent_sas_path=str(_agent_sas_path),
            gpu_ids=gpu_filler_gpu_ids,
            enabled=gpu_filler_enabled,
        )

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

    def _print_paper_writing_reward_stats(self, scores, rubric_details=None):
        """Log rubric sub-scores and final outcome reward only."""
        pieces = []
        if rubric_details:
            dimensions = [
                'Problem & Motivation',
                'Method & Contribution Coverage',
                'Results & Evidence Coverage',
                'Topic Consistency',
                'Clarity & Conciseness',
                'Length Appropriateness',
                'Format & Presentation',
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
        if scores:
            format_fail_rate = sum(1 for s in scores if s == -1.0) / len(scores)
            valid_scores = [s for s in scores if s != -1.0]
            pieces.append(f"format_fail_rate={format_fail_rate:.4f}")
            if valid_scores:
                pieces.append(f"reward_mean={float(np.mean(valid_scores)):.4f}")
            pieces.append(f"reward_mean_with_format_gate={float(np.mean(scores)):.4f}")
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

    @staticmethod
    def _extract_camera_ready_body(text: str):
        """Relaxed format gate + body extraction for camera-ready outputs.

        Rules:
          - Exactly one ``<camera-ready>...</camera-ready>`` block (accepted).
          - Else exactly one ``<draft>...</draft>`` block (also accepted as
            camera-ready body — the model sometimes submits inside ``<draft>``).
          - Plain prose with no XML tags (accepted as-is).
          - Anything else (multiple blocks, nested/other tags, body still
            containing ``<...>``) → format_bad, reward -1.

        Returns:
            (body: str, format_bad: bool)
        """
        if not isinstance(text, str):
            return ('', True)
        stripped = text.strip()
        tag_pattern = re.compile(r'<[^>]+>')
        cr_blocks = re.findall(r'<camera-ready>(.*?)</camera-ready>', stripped, re.DOTALL)
        draft_blocks = re.findall(r'<draft>(.*?)</draft>', stripped, re.DOTALL)
        total_tags = len(tag_pattern.findall(stripped))

        if len(cr_blocks) == 1 and len(draft_blocks) == 0:
            # Exactly one <camera-ready> block, no draft.
            if total_tags != 2:  # must be exactly <camera-ready> and </camera-ready>
                return (cr_blocks[0].strip(), True)
            body = cr_blocks[0].strip()
            if tag_pattern.search(body):
                return (body, True)
            return (body, False)

        if len(draft_blocks) == 1 and len(cr_blocks) == 0:
            # Exactly one <draft> block used as camera-ready fallback.
            if total_tags != 2:
                return (draft_blocks[0].strip(), True)
            body = draft_blocks[0].strip()
            if tag_pattern.search(body):
                return (body, True)
            return (body, False)

        if not cr_blocks and not draft_blocks:
            # No recognized tags at all.
            if total_tags == 0:
                return (stripped, False)
            return (stripped, True)

        # Multiple blocks, or mixed camera-ready + draft.
        body = (cr_blocks[0] if cr_blocks else draft_blocks[0]).strip()
        return (body, True)

    def _compute_paper_writing_reward(self, data: DataProto):
        """
        Pure outcome reward for paper writing.

        The rubric score for the camera-ready abstract is placed at the last
        trainable token; all other token rewards are zero.
        Format gate: if the camera-ready text still contains XML tags (i.e. the
        model failed to submit via <camera-ready>), reward is -1.0.
        """

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # Get camera-ready texts and format flags from non_tensor_batch (set during rollout).
        nt = data.non_tensor_batch or {}
        camera_ready_arr = nt.get('camera_ready', None)
        format_ok_arr = nt.get('format_ok', None)

        if camera_ready_arr is None:
            print("[WARNING] No camera_ready found in non_tensor_batch, returning zero rewards")
            return reward_tensor

        batch_size = len(camera_ready_arr)
        camera_ready_texts = [str(x) for x in camera_ready_arr]
        # format gate already applied in generation; invert format_ok to get format_bad.
        format_bad = [not bool(x) for x in format_ok_arr] if format_ok_arr is not None \
            else [True] * batch_size

        ground_truth_texts = data.meta_info.get('paper_writing_ground_truth_texts', None)

        clean_indices = [i for i, bad in enumerate(format_bad) if not bad]
        clean_texts = [camera_ready_texts[i] for i in clean_indices]
        clean_gt = (
            [ground_truth_texts[i] for i in clean_indices]
            if ground_truth_texts is not None else None
        )
        _fail_detail = {
            'overall': -1.0, 'subscores': {},
            'summary': 'FORMAT ERROR: XML tags detected in camera-ready output'
        }
        scores = [-1.0 if bad else 0.0 for bad in format_bad]
        rubric_details = [
            _fail_detail if bad else {'overall': 0.0, 'subscores': {}, 'summary': ''}
            for bad in format_bad
        ]
        all_domains = self._to_plain_list(data.meta_info.get('paper_writing_domains', []))
        all_topics = self._to_plain_list(data.meta_info.get('paper_writing_topics', []))
        clean_domains = [all_domains[i] if i < len(all_domains) else '' for i in clean_indices]
        clean_topics = [all_topics[i] if i < len(all_topics) else '' for i in clean_indices]
        if clean_texts:
            clean_scores, clean_details = self._call_rubric_scoring_api(
                clean_texts,
                ground_truth_texts=clean_gt,
                domains=clean_domains,
                topics=clean_topics,
                return_details=True,
            )
            for pos, orig_idx in enumerate(clean_indices):
                scores[orig_idx] = clean_scores[pos]
                rubric_details[orig_idx] = clean_details[pos]
        if any(format_bad):
            print(f"[Paper Writing] Format gate: {sum(format_bad)}/{len(format_bad)} samples "
                  f"have XML tags in camera-ready, assigned reward -1")

        # Place outcome reward at last trainable token; all other tokens stay zero.
        for i in range(len(data)):
            reward_index = self._get_last_trainable_response_index(data[i])
            reward_tensor[i, reward_index] = float(scores[i])

        os.environ.get('PW_DEBUG') and __import__('pdb').set_trace()
        pw_metrics = self._print_paper_writing_reward_stats(scores, rubric_details=rubric_details)
        self.paper_writing_metrics = pw_metrics if pw_metrics else {}
        # Expose per-sample details so downstream code (e.g. rollout dump) can use them.
        data.meta_info['paper_writing_rubric_details'] = rubric_details
        data.meta_info['paper_writing_format_bad'] = format_bad
        # self._maybe_save_sft_candidates(data, scores, rubric_details, [], [])
        return reward_tensor

    def _compute_paper_writing_arena_hybrid_reward(self, data: DataProto):
        """Compute hybrid reward = arena_weight * arena + rubric_weight * rubric."""
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # Get camera-ready texts and format flags from non_tensor_batch (set during rollout).
        nt = data.non_tensor_batch or {}
        camera_ready_arr = nt.get('camera_ready', None)
        format_ok_arr = nt.get('format_ok', None)

        if camera_ready_arr is None:
            print("[WARNING] No camera_ready found in non_tensor_batch, returning zero rewards")
            return reward_tensor

        batch_size = len(camera_ready_arr)
        camera_ready_texts = [str(x) for x in camera_ready_arr]
        # format gate already applied in generation; invert format_ok to get format_bad.
        _format_bad = [not bool(x) for x in format_ok_arr] if format_ok_arr is not None \
            else [True] * batch_size

        _clean_indices = [i for i, bad in enumerate(_format_bad) if not bad]
        _clean_texts = [camera_ready_texts[i] for i in _clean_indices]
        _clean_gt = data.meta_info.get('paper_writing_ground_truth_texts', None)
        _clean_gt_filtered = (
            [_clean_gt[i] for i in _clean_indices] if _clean_gt is not None else None
        )
        _all_domains = self._to_plain_list(data.meta_info.get('paper_writing_domains', []))
        _all_topics = self._to_plain_list(data.meta_info.get('paper_writing_topics', []))
        _clean_domains = [_all_domains[i] if i < len(_all_domains) else '' for i in _clean_indices]
        _clean_topics = [_all_topics[i] if i < len(_all_topics) else '' for i in _clean_indices]
        rubric_scores = [-1.0 if bad else 0.0 for bad in _format_bad]
        _fail_detail = {
            'overall': -1.0, 'subscores': {},
            'summary': 'FORMAT ERROR: XML tags detected in camera-ready output',
            'raw': '',
        }
        rubric_details = [
            dict(_fail_detail) if bad else {'overall': 0.0, 'subscores': {}, 'summary': '', 'raw': ''}
            for bad in _format_bad
        ]
        if _clean_texts:
            _clean_rubric, _clean_details = self._call_rubric_scoring_api(
                _clean_texts,
                ground_truth_texts=_clean_gt_filtered,
                domains=_clean_domains,
                topics=_clean_topics,
                return_details=True,
            )
            for pos, orig_idx in enumerate(_clean_indices):
                rubric_scores[orig_idx] = _clean_rubric[pos]
                rubric_details[orig_idx] = _clean_details[pos]
        if any(_format_bad):
            print(f"[Paper Writing/Arena] Format gate: {sum(_format_bad)}/{len(_format_bad)} samples "
                  f"have XML tags in camera-ready, assigned reward -1")
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
            if float(rubric_scores[i]) == -1.0:
                score = -1.0  # format gate
            else:
                score = arena_weight * float(arena_scores[i]) + rubric_weight * float(rubric_scores[i])
            reward_index = self._get_last_trainable_response_index(data[i])
            reward_tensor[i, reward_index] = score
        # Expose per-sample details so downstream code (e.g. rollout dump) can use them.
        data.meta_info['paper_writing_rubric_details'] = rubric_details
        data.meta_info['paper_writing_format_bad'] = _format_bad
        return reward_tensor
    
    def _call_rubric_scoring_api(self, camera_ready_texts, ground_truth_texts=None,
                                 domains=None, topics=None, return_details=False):
        """
        Call external API to score papers based on rubric.
        Evaluation is based on the paper's keywords and title; ground truth is
        provided as a reference only, not as an authoritative answer.
        Returns a list of scores (one per paper).
        """
        # Lazily create a long-lived client so we don't recreate it per sample.
        if self._rubric_client is None:
            self._rubric_client = openai.OpenAI(
                api_key=self.rubric_api_key,
                base_url=self.rubric_api_base,
                timeout=60,  # Set a reasonable timeout for rubric API calls
                max_retries=3  # Disable retries to fail fast on issues
            )
        client = self._rubric_client

        rubric_prompt = (
            "You are an expert evaluator for academic paper abstracts at top-tier venues "
            "(e.g., NeurIPS, ICML, ICLR, ACL). "
            "You will be given a paper's keywords, title, and a generated abstract to evaluate. "
            "A reference abstract is also provided for context — it is one possible way to write "
            "the abstract, but it is NOT the ground truth or the only correct answer. "
            "Do NOT penalize the generated abstract merely for phrasing differently from the reference. "
            "Evaluate the generated abstract independently based on the keywords and title, "
            "using the reference only as a helpful guide to the paper's content.\n\n"
            "Score each dimension from 0.0 to 1.0 (two decimal places).\n\n"

            "## Scoring Rubric\n\n"

            "### 1. Problem & Motivation\n"
            "Does the abstract clearly identify a research problem or gap, and explain why it matters? "
            "Is the motivation well-grounded in the context implied by the title and keywords?\n\n"

            "### 2. Method & Contribution Coverage\n"
            "Does it describe the main approach, method, or technical idea? "
            "Does it state what is novel or what the paper contributes, without relying on vague or generic claims?\n\n"

            "### 3. Results & Evidence Coverage\n"
            "Does it report key empirical or theoretical results, performance numbers, or evidence of the contribution's effectiveness?\n\n"

            "### 4. Topic Consistency\n"
            "Is the abstract's content consistent with the given title and keywords? "
            "Does it stay on-topic, without drifting to unrelated problems or fabricating claims not plausible given the topic?\n\n"

            "### 5. Clarity & Conciseness\n"
            "Is it clear, coherent, and self-contained? Is it written in fluent academic prose "
            "appropriate for a top-tier venue abstract?\n\n"

            "### 6. Length Appropriateness\n"
            "Is the length and level of detail appropriate for an abstract — "
            "neither too brief to be informative nor too long and unfocused?\n\n"

            "### 7. Format & Presentation\n"
            "Is the abstract presented as clean plain-text academic prose?\n"
            "Penalize heavily for: residual XML tags (e.g. <draft>, <camera-ready>), markdown formatting, "
            "bullet points, numbered lists, or any non-academic structural artifacts.\n"
            "Score 1.0 if completely clean prose; 0.0 if heavily polluted with tags or non-prose artifacts.\n\n"

            "Use 1.0 for excellent, 0.8 for good with minor issues, 0.6 for partially adequate, "
            "0.4 for weak, 0.2 for very poor, and 0.0 for missing or unusable.\n\n"

            "## Output Format\n"
            "You MUST output scores in exactly the following format (no extra text before the scores):\n"
            "Problem & Motivation: [score]\n"
            "Method & Contribution Coverage: [score]\n"
            "Results & Evidence Coverage: [score]\n"
            "Topic Consistency: [score]\n"
            "Clarity & Conciseness: [score]\n"
            "Length Appropriateness: [score]\n"
            "Format & Presentation: [score]\n"
            "Summary: [2-3 sentence overall assessment]\n\n"

            "Example:\n"
            "Problem & Motivation: 0.80\n"
            "Method & Contribution Coverage: 0.70\n"
            "Results & Evidence Coverage: 0.60\n"
            "Topic Consistency: 0.85\n"
            "Clarity & Conciseness: 0.75\n"
            "Length Appropriateness: 0.70\n"
            "Format & Presentation: 1.00\n"
            "Summary: The abstract clearly addresses the stated topic and presents a coherent contribution, "
            "but omits key result details and is slightly generic in its method description."
        )

        # Preserve output order: one score per paper in the same order as input.
        num_papers = len(camera_ready_texts)
        scores = [0.0] * num_papers
        details = [{'overall': 0.0, 'subscores': {}, 'summary': ''} for _ in range(num_papers)]
        if ground_truth_texts is None:
            ground_truth_texts = [''] * num_papers
        if domains is None:
            domains = [''] * num_papers
        if topics is None:
            topics = [''] * num_papers

        def _score_single(idx, paper_text, ground_truth_text, domain, topic):
            """Score a single paper using the rubric API."""
            try:
                ref_block = (
                    f"Reference abstract (for reference only — not the ground truth):\n{ground_truth_text}"
                    if ground_truth_text and ground_truth_text.strip()
                    else "Reference abstract: [not provided]"
                )
                user_content = (
                    f"Keywords: {domain or '[not provided]'}\n"
                    f"Title: {topic or '[not provided]'}\n\n"
                    f"{ref_block}\n\n"
                    f"Abstract to evaluate:\n{paper_text}"
                )
                messages = [
                    {"role": "system", "content": rubric_prompt},
                    {"role": "user", "content": user_content}
                ]

                self._rubric_rate_limiter.acquire()
                response = client.chat.completions.create(
                    model=self.rubric_model,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.3,
                    extra_body={"enable_thinking": False}
                )

                score_text = response.choices[0].message.content
                detail = self._parse_rubric_scores(score_text)
                detail['raw'] = score_text
                score = float(detail.get('overall', 0.0))
                return score, detail
            except Exception as e:
                print(f"[ERROR] Rubric scoring API failed: {e}")
                return 0.0, {'overall': 0.0, 'subscores': {}, 'summary': '', 'raw': f'ERROR: {e}'}

        max_workers = min(self.rubric_max_concurrency, num_papers) if num_papers > 0 else 1
        with self._gpu_filler:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _score_single,
                        idx,
                        paper_text,
                        ground_truth_texts[idx] if idx < len(ground_truth_texts) else '',
                        domains[idx] if idx < len(domains) else '',
                        topics[idx] if idx < len(topics) else '',
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
            'Topic Consistency',
            'Clarity & Conciseness',
            'Length Appropriateness',
            'Format & Presentation',
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
            return {'overall': 0.0, 'subscores': {}, 'summary': '', 'raw': score_text}
        
        # Return average of all dimensions
        avg_score = sum(scores) / len(scores)
        summary_match = re.search(r'Summary:\s*(.*)', score_text, re.IGNORECASE | re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ''
        return {'overall': avg_score, 'subscores': subscores, 'summary': summary, 'raw': score_text}



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
        rubric_rpm=config.get('rubric_rpm', 120),
        arena_weight=config.get('arena_weight', 0.7),
        rubric_weight=config.get('rubric_weight', 0.3),
        experiment_name=config.trainer.get('experiment_name', 'default'),
        save_sft_candidates=config.get('paper_writing_save_sft_candidates', False),
        sft_score_threshold=config.get('paper_writing_sft_score_threshold', 0.78),
        sft_output_dir=config.get('paper_writing_sft_output_dir', 'outputs/sft_candidates'),
        gpu_filler_enabled=config.get('gpu_filler_enabled', True),
        gpu_filler_gpu_ids=config.get('gpu_filler_gpu_ids', None),
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
        rubric_rpm=config.get('rubric_rpm', 120),
        arena_weight=config.get('arena_weight', 0.7),
        rubric_weight=config.get('rubric_weight', 0.3),
        experiment_name=config.trainer.get('experiment_name', 'default'),
        save_sft_candidates=False,
        sft_score_threshold=config.get('paper_writing_sft_score_threshold', 0.78),
        sft_output_dir=config.get('paper_writing_sft_output_dir', 'outputs/sft_candidates'),
        gpu_filler_enabled=config.get('gpu_filler_enabled', True),
        gpu_filler_gpu_ids=config.get('gpu_filler_gpu_ids', None),
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
