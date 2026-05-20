import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import hashlib
import numpy as np
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
import asyncio
import pdb
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def _extract_camera_ready_body_and_source(text: str) -> Tuple[str, bool, str]:
    """Stage-6 format gate used at rollout end for paper_writing_autonomous.

    Mirrors PaperWritingRewardManager._extract_camera_ready_body in main_ppo.py,
    but additionally reports a ``final_source`` label so downstream (dumper /
    analytics) can tell how the body was obtained.

    Returns:
        (body, format_ok, final_source) where final_source ∈ {
            'camera_ready',        # exactly one clean <camera-ready>...</camera-ready>
            'draft_as_camera_ready',  # exactly one clean <draft>...</draft> treated as final
            'raw_plain',           # plain text with no XML tags
            'raw_fallback',        # anything else (multiple/nested/dirty tags) -> format_bad
        }
    """
    if not isinstance(text, str):
        return ('', False, 'raw_fallback')
    stripped = text.strip()
    tag_pattern = re.compile(r'<[^>]+>')
    cr_blocks = re.findall(r'<camera-ready>(.*?)</camera-ready>', stripped, re.DOTALL)
    draft_blocks = re.findall(r'<draft>(.*?)</draft>', stripped, re.DOTALL)
    total_tags = len(tag_pattern.findall(stripped))

    if len(cr_blocks) == 1 and len(draft_blocks) == 0:
        body = cr_blocks[0].strip()
        if total_tags == 2 and not tag_pattern.search(body):
            return (body, True, 'camera_ready')
        return (body, False, 'raw_fallback')

    if len(draft_blocks) == 1 and len(cr_blocks) == 0:
        body = draft_blocks[0].strip()
        if total_tags == 2 and not tag_pattern.search(body):
            return (body, True, 'draft_as_camera_ready')
        return (body, False, 'raw_fallback')

    if not cr_blocks and not draft_blocks:
        if total_tags == 0:
            return (stripped, True, 'raw_plain')
        return (stripped, False, 'raw_fallback')

    body = (cr_blocks[0] if cr_blocks else draft_blocks[0]).strip()
    return (body, False, 'raw_fallback')


class RateLimiter:
    """线程安全的滑动窗口 RPM 限流器。"""

    def __init__(self, rpm: int = 120):
        self.rpm = max(1, rpm)
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    def acquire(self):
        """阻塞直到 60s 窗口内有可用配额。"""
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


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    # Paper writing specific configs
    num_revision_rounds: int = 3
    commenter_api_key: str = None
    commenter_base_url: str = None
    commenter_model: str = "qwen-plus"
    generator_max_concurrency: int = 64
    commenter_max_concurrency: int = 48
    api_rpm: int = 120
    arena_seed_mode: str = "swiss_single_round"
    arena_seed: int = 20260413
    arena_group_size: int = 8
    # GPU idle filler configs
    gpu_filler_enabled: bool = True
    gpu_filler_gpu_ids: list = None

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        ref_policy_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.ref_policy_wg = ref_policy_wg
        self.config = config
        self.is_validation = is_validation
        self._commenter_client = None
        self._generator_client = None
        self._commenter_rate_limiter = RateLimiter(rpm=getattr(config, 'api_rpm', 120))

        from verl.utils.gpu_idle_filler import GPUIdleFiller
        from pathlib import Path
        _agent_sas_path = Path(__file__).parent.parent.parent / 'agent_SAS.py'
        self._gpu_filler = GPUIdleFiller(
            agent_sas_path=str(_agent_sas_path),
            gpu_ids=getattr(config, 'gpu_filler_gpu_ids', None),
            enabled=getattr(config, 'gpu_filler_enabled', True),
        )

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _build_logprob_dataproto(self, ctx: DataProto, responses: torch.Tensor) -> DataProto:
        """Build a DataProto for segment-level logprob computation under current context."""
        cut_ctx_batch = self.tensor_fn.cut_to_effective_len(
            ctx.batch,
            keys=['input_ids', 'attention_mask', 'position_ids']
        )
        ctx_ids = cut_ctx_batch['input_ids']
        ctx_attn = cut_ctx_batch['attention_mask']
        response_attn = self.tensor_fn.create_attention_mask(responses)
        input_ids = torch.cat([ctx_ids, responses], dim=1)
        attention_mask = torch.cat([ctx_attn, response_attn], dim=1)
        position_ids = self.tensor_fn.create_position_ids(attention_mask)

        batch = {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.long(),
            'position_ids': position_ids.long(),
            'responses': responses.long(),
        }
        data = DataProto.from_dict(batch)
        data.meta_info.update(dict(getattr(ctx, 'meta_info', {}) or {}))
        return data

    def _compute_segment_old_ref_logprob(self, ctx: DataProto, responses: torch.Tensor):
        """Compute old/ref logprob for one generated segment under the exact rollout context."""
        data = self._build_logprob_dataproto(ctx, responses)
        old_output = self.actor_rollout_wg.compute_log_prob(data)
        old_lp = old_output.batch['old_log_probs'].float()

        if self.ref_policy_wg is not None:
            ref_output = self.ref_policy_wg.compute_ref_log_prob(data)
            ref_lp = ref_output.batch['ref_log_prob'].float()
        else:
            ref_lp = torch.zeros_like(old_lp)
        return old_lp, ref_lp

    def _update_dense_right_side_aligned(self,
                                         prev_tokens: torch.Tensor,
                                         prev_dense: torch.Tensor,
                                         cur_tokens: torch.Tensor,
                                         cur_dense: torch.Tensor,
                                         info_tokens: torch.Tensor = None,
                                         info_dense: torch.Tensor = None,
                                         target_len: int = None) -> torch.Tensor:
        """Update dense right-side tensor (old/ref logprob) with the same compaction
        rule used by `_info_masked_concatenate_with_padding(..., pad_to_left=False)`.
        """
        token_tensors = [prev_tokens, cur_tokens]
        dense_tensors = [prev_dense, cur_dense]
        if info_tokens is not None:
            token_tensors.append(info_tokens)
            dense_tensors.append(info_dense)

        token_concat = torch.cat(token_tensors, dim=1)
        dense_concat = torch.cat(dense_tensors, dim=1)
        pad_id = self.tokenizer.pad_token_id
        # pad_to_left=False in right-side update: move non-pad to the left, pads to the right.
        mask = token_concat == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        dense_compacted = dense_concat.gather(1, sorted_indices)

        if target_len is None:
            target_len = min(self.config.max_prompt_length, dense_compacted.shape[1])
        return dense_compacted[:, :target_len]

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _normalize_draft_texts(self, raw_texts: List[str], round_idx: int = -1) -> Tuple[List[str], List[str]]:
        """Strictly normalize draft outputs without auto-fix.

        Returns:
            normalized_texts: exact-match outputs are canonicalized to <draft>...</draft>;
                invalid outputs are kept as raw text (no auto-fix).
            statuses: one of exact_match / invalid_raw
        """
        normalized_texts = []
        statuses = []

        for text in raw_texts:
            raw = text if isinstance(text, str) else ('' if text is None else str(text))
            status = 'invalid_raw'

            # 1) Best case: exact <draft>...</draft> pair.
            match = re.search(r'<draft>(.*?)</draft>', raw, re.DOTALL)
            is_single_block = re.fullmatch(r'\s*<draft>.*?</draft>\s*', raw, flags=re.DOTALL) is not None
            if match and is_single_block:
                clean_text = match.group(1).strip()
                status = 'exact_match'
                normalized_texts.append(f'<draft>{clean_text}</draft>')
            else:
                # Keep invalid model output as-is so rollout data is not modified by auto-fix.
                normalized_texts.append(raw)
            statuses.append(status)

        if statuses:
            exact = sum(1 for s in statuses if s == 'exact_match')
            invalid_raw = sum(1 for s in statuses if s == 'invalid_raw')
            round_msg = f'round {round_idx}' if round_idx >= 0 else 'final'
            print(f"[Paper Writing] Draft normalization ({round_msg}): exact={exact}, invalid_raw={invalid_raw}")

        return normalized_texts, statuses

    def _is_strict_single_draft(self, text: str) -> bool:
        """Return True iff text is exactly one <draft>...</draft> block (ignoring outer spaces)."""
        raw = text if isinstance(text, str) else ('' if text is None else str(text))
        if not re.fullmatch(r'\s*<draft>.*?</draft>\s*', raw, flags=re.DOTALL):
            return False
        blocks = re.findall(r'<draft>.*?</draft>', raw, flags=re.DOTALL)
        return len(blocks) == 1

    def _draft_content_for_length(self, text: str) -> str:
        """Use inner draft text for valid drafts; otherwise use raw output length."""
        raw = text if isinstance(text, str) else ('' if text is None else str(text))
        match = re.fullmatch(r'\s*<draft>(.*?)</draft>\s*', raw, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw.strip()

    def _text_token_lengths(self, texts: List[str]) -> List[int]:
        """Token lengths without special tokens, robust to empty strings."""
        if not texts:
            return []
        normalized = [text if isinstance(text, str) else ('' if text is None else str(text)) for text in texts]
        encoded = self.tokenizer(normalized, add_special_tokens=False, padding=False)
        return [len(ids) for ids in encoded['input_ids']]

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """postprocess to tokenize."""
        responses_str = self.tokenizer.batch_decode(
            responses,
            skip_special_tokens=True
        )

        # Truncate at the first closing tag found
        processed_responses = []
        for resp in responses_str:
            if '</search>' in resp:
                processed_responses.append(resp.split('</search>')[0] + '</search>')
            elif '</answer>' in resp:
                processed_responses.append(resp.split('</answer>')[0] + '</answer>')
            else:
                processed_responses.append(resp)

        responses_str = processed_responses

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _postprocess_responses_paper_writing_autonomous(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Postprocess responses for autonomous paper writing loop.

        Truncates at the first </draft> or </camera-ready> closing tag, mirroring
        how _postprocess_responses truncates at </search> or </answer>.
        """
        responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        processed_responses = []
        for resp in responses_str:
            if '</draft>' in resp:
                processed_responses.append(resp.split('</draft>')[0] + '</draft>')
            elif '</camera-ready>' in resp:
                processed_responses.append(resp.split('</camera-ready>')[0] + '</camera-ready>')
            else:
                processed_responses.append(resp)
        responses_str = processed_responses
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment, into token ids, avoid truncation."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _rebuild_rolling_state(self,
                               query_ids: torch.Tensor,
                               draft_ids: torch.Tensor,
                               comment_obs_ids: torch.Tensor,
                               meta_info: Dict = None) -> DataProto:
        """
        Rebuild rolling state from scratch (non-accumulative).
        Used in paper writing to keep only the latest context.

        Args:
            query_ids: Initial query tokens
            draft_ids: Current draft tokens
            comment_obs_ids: Current comment tokens
            meta_info: Optional meta_info to preserve

        Returns:
            DataProto with input_ids, attention_mask, position_ids
        """
        # Concatenate and handle padding
        tensors = [draft_ids, comment_obs_ids, query_ids]
        new_input_ids = self.tensor_fn.concatenate_with_padding(tensors)

        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })

        # Preserve meta_info if provided
        if meta_info is not None:
            new_rollings.meta_info.update(meta_info)

        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True,
                extra_tensors: Dict[str, torch.Tensor] = None,
            ) -> tuple:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists.
        
        Args:
            extra_tensors: optional dict of tensors to sort in parallel (same layout as
                ``[prompt, response]`` or ``[prompt, response, info]``). Each value is a
                list of tensors matching that layout.  They will be concatenated and
                reordered with the same ``sorted_indices``.
        """
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        if extra_tensors:
            sorted_extras = {}
            for key, parts in extra_tensors.items():
                cat = torch.cat(parts, dim=1)
                sorted_extras[key] = cat.gather(1, sorted_indices)
            return padded_tensor, padded_tensor_with_info, sorted_extras

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None,
                          cur_segment_id: int = 0,
                          next_obs_segment_id: int = 0) -> Dict:
        """Update right side state.
        
        Args:
            cur_segment_id: segment label for ``cur_responses`` tokens in ``segment_ids``.
            next_obs_segment_id: segment label for ``next_obs_ids`` tokens.
        """
        # Build extra segment_ids parts that mirror [prompt, response(, info)]
        has_seg = 'segment_ids' in right_side
        extra = None
        if has_seg or cur_segment_id or next_obs_segment_id:
            prev_seg = right_side.get('segment_ids',
                                      torch.zeros_like(right_side['responses']))
            cur_seg = torch.full_like(cur_responses, cur_segment_id)
            parts = [prev_seg, cur_seg]
            if next_obs_ids is not None:
                parts.append(torch.full_like(next_obs_ids, next_obs_segment_id))
            extra = {'segment_ids': parts}

        if next_obs_ids is not None:
            result = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False,
                    extra_tensors=extra,
                )
        else:
            result = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False,
                    extra_tensors=extra,
                )

        if extra:
            responses, responses_with_info_mask, sorted_extras = result
        else:
            responses, responses_with_info_mask = result
            sorted_extras = {}

        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        # Preserve auxiliary right-side tensors (e.g. old/ref log-probs) while
        # keeping sequence length aligned with responses.
        new_right_side = {
            'responses': responses[:, :max_len],
            'responses_with_info_mask': responses_with_info_mask[:, :max_len],
        }
        if 'segment_ids' in sorted_extras:
            new_right_side['segment_ids'] = sorted_extras['segment_ids'][:, :max_len]
        for key, value in right_side.items():
            if key in ('responses', 'responses_with_info_mask', 'segment_ids'):
                continue
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                new_right_side[key] = value[:, :max_len]
            else:
                new_right_side[key] = value
        return new_right_side

    def _append_right_side_segment(self,
                                   right_side: Dict,
                                   segment_ids: torch.Tensor,
                                   trainable: bool,
                                   segment_label: int = 0) -> Dict:
        """按“是否可训练”把一个片段追加到 right_side。

        Args:
            right_side: 当前累计的 right_side，包含 responses / responses_with_info_mask。
            segment_ids: 待追加片段的 token ids。
            trainable:
                - True: 该片段参与策略损失（保留在 responses_with_info_mask）
                - False: 该片段不参与策略损失（在 responses_with_info_mask 中被 pad 掉）
        """
        empty = segment_ids[:, :0]
        if trainable:
            return self._update_right_side(right_side, segment_ids, next_obs_ids=None,
                                           cur_segment_id=segment_label)
        return self._update_right_side(right_side, empty, next_obs_ids=segment_ids,
                                       next_obs_segment_id=segment_label)

    def _make_round_instruction(self, next_round: int) -> str:
        """生成下一轮固定指令。"""
        return (
            f"Round {next_round}: Please write your draft now. "
            "Output only one <draft>...</draft> block."
        )

    def _safe_similarity_to_anchor(self, candidate: str, anchor: str) -> float:
        """计算候选文本与锚点文本的轻量词汇相似度（范围 [0,1]）。"""
        if not isinstance(candidate, str):
            candidate = '' if candidate is None else str(candidate)
        if not isinstance(anchor, str):
            anchor = '' if anchor is None else str(anchor)
        cand_tokens = set(re.findall(r'\w+', candidate.lower()))
        anchor_tokens = set(re.findall(r'\w+', anchor.lower()))
        if not cand_tokens or not anchor_tokens:
            return 0.0
        inter = len(cand_tokens & anchor_tokens)
        union = len(cand_tokens | anchor_tokens)
        return float(inter) / float(union) if union > 0 else 0.0

    def _seeded_tiebreak_score(self, seed: int, global_idx: int) -> float:
        """基于 seed 和样本索引生成确定性极小扰动，用于打破完全平分。"""
        digest = hashlib.sha256(f"{seed}:{global_idx}".encode("utf-8")).hexdigest()
        # Keep a very small range so it only breaks exact ties.
        return (int(digest[:8], 16) % 1000000) / 1e12

    def _compute_seeded_swiss_arena_scores(self,
                                           candidates: List[str],
                                           anchors: List[str],
                                           group_size: int,
                                           seed: int) -> List[float]:
        """计算“带种子单轮 Swiss”组内相对分数。

        规则：
        1. 每组先按 seed 分排序（anchor 相似度 + 确定性微扰）
        2. 相邻样本仅对战 1 次（单轮）
        3. 胜者 1.0，平局双方各 0.5
        4. 组内样本数为奇数时，最后一个样本获得 bye 分 0.5
        """
        n = len(candidates)
        if n == 0:
            return []
        if group_size <= 1:
            return [0.5] * n

        scores = [0.5] * n
        for start in range(0, n, group_size):
            end = min(start + group_size, n)
            group_indices = list(range(start, end))
            seeded = []
            for idx in group_indices:
                sim = self._safe_similarity_to_anchor(candidates[idx], anchors[idx])
                tie = self._seeded_tiebreak_score(seed=seed, global_idx=idx)
                seeded.append((idx, sim + tie))
            seeded.sort(key=lambda x: x[1], reverse=True)
            ranked = [idx for idx, _ in seeded]

            group_points = {idx: 0.0 for idx in ranked}
            cursor = 0
            while cursor + 1 < len(ranked):
                left = ranked[cursor]
                right = ranked[cursor + 1]
                left_sim = self._safe_similarity_to_anchor(candidates[left], anchors[left])
                right_sim = self._safe_similarity_to_anchor(candidates[right], anchors[right])
                if abs(left_sim - right_sim) < 1e-8:
                    group_points[left] += 0.5
                    group_points[right] += 0.5
                elif left_sim > right_sim:
                    group_points[left] += 1.0
                else:
                    group_points[right] += 1.0
                cursor += 2

            # Odd sample gets a bye in one-round Swiss.
            if len(ranked) % 2 == 1:
                group_points[ranked[-1]] += 0.5

            for idx in ranked:
                scores[idx] = group_points[idx]
        return scores

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop.
        - 样本 0：问题 "Who invented the telephone?"                                                    
        - 样本 1：问题 "What is the capital of France?" 
        """
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool) # 标记哪些样本还在生成，即没有生成<answer>也没有达到最大_turns
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int) # 记录每个样本当前的turn数
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int) # 记录每个样本当前的valid_action数
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int) # 记录每个样本当前的valid_search数
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch # 滚动上下文，每个turn都会追加新的内容，作为下一轮的输入

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            ) # 剪掉左侧所有样本的共同padding部分，节省显存
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            """
             rolings.batch:
             4: batch_size
             143: max_prompt_length after cutting to effective_len
             TensorDict(
                fields={
                    attention_mask: Tensor(shape=torch.Size([4, 143]), device=cpu, dtype=torch.int64, is_shared=False),
                    input_ids: Tensor(shape=torch.Size([4, 143]), device=cpu, dtype=torch.int64, is_shared=False),
                    position_ids: Tensor(shape=torch.Size([4, 143]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([4]),
                device=None,
                is_shared=False)
            """
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active) # 自己封装的调用VeRL的LLM生成
            """
            - 样本 0："<think>Let me search for this.</think><search>telephone inventor</search>"
            - 样本 1："<think>I know this.</think><answer>Paris</answer>" 
            """
            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # 🌟Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask,
                do_search = True,
            )
            """
            以一个batch-size = 2的为例子，第一个问题调用了外部retriever，第二个问题没有。
            next_obs      = ['<information>...</information>', '']                                          
            dones         = [0, 1]                                                                          
            valid_action  = [1, 1]                                                                          
            is_search     = [1, 0] 
            """
            # 更新active_mask，记录哪些样本还在生成； 更新每个样本当前的turn数，更新每个样本的valid_action数和valid_search数
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            # 处理下一个观察，将文本转换为token_ids； 比如
            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            ) # 将作为下一个turn的输入
            """
             [SYSTEM_prompt] + [<think>...</think><search>telephone inventor</search>] +
                [<information>Doc1...</information>] 
            """
            
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            ) # 更新responses和rewponses_with_info_mask, <information>...</information>的内容在responses_with_info_mask中被pad掉，只保留LLM生成的内容
            
        # final LLM rollout，,如果还有 active 样本（比如 max_turns 用完了还没给 <answer>），这里会做最后一次生成，但do_search=False，不再调用检索，强制收尾。
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False # 超出max_turns的样本，不再调用检索，强制回答得到最后的答案
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)

        # Propagate segment_ids if available (used for per-draft credit assignment)
        if 'segment_ids' in right_side:
            prompt_len = left_side['input_ids'].shape[1]
            final_output['segment_ids'] = torch.cat([
                torch.zeros(right_side['segment_ids'].shape[0], prompt_len,
                            dtype=right_side['segment_ids'].dtype,
                            device=right_side['segment_ids'].device),
                right_side['segment_ids']
            ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

    def _execute_paper_writing_autonomous(
        self,
        responses_str: List[str],
        active_mask,
        domains: List[str],
        topics: List[str],
        ground_truths: List[str],
    ) -> Tuple[List[str], List[int], List[int], List[int], List[str]]:
        """Execute one step of the autonomous paper writing loop.

        Mirrors execute_predictions for QA search/answer:
          - action='draft'        -> call commenter API, obs=<comment>...</comment>, done=False
          - action='camera-ready' -> done=True, obs=''
          - action=None           -> invalid-action feedback, done=False

        Returns:
            next_obs, dones, valid_action, is_comment, camera_ready_contents
            camera_ready_contents: per-sample extracted <camera-ready> content ('' if not done)
        """
        pattern = r'<(draft|camera-ready)>(.*?)</\1>'
        actions = []
        contents = []
        for resp in responses_str:
            match = re.search(pattern, resp, re.DOTALL)
            if match:
                actions.append(match.group(1))
                contents.append(match.group(2).strip())
            else:
                actions.append(None)
                contents.append('')

        # Collect all active non-camera-ready samples for commenter feedback.
        # For draft actions: pass extracted inner content.
        # For invalid actions: pass raw response so commenter can identify format issues.
        commenter_indices = [
            i for i, (action, active) in enumerate(zip(actions, active_mask))
            if active and action != 'camera-ready'
        ]
        if commenter_indices:
            commenter_domains = [domains[i] for i in commenter_indices]
            commenter_topics = [topics[i] for i in commenter_indices]
            commenter_texts = [
                contents[i] if actions[i] == 'draft' else responses_str[i]
                for i in commenter_indices
            ]
            commenter_ground_truths = [ground_truths[i] for i in commenter_indices]
            all_comments = self._call_commenter_batch(
                commenter_domains, commenter_topics, commenter_texts, commenter_ground_truths
            )
            comment_map = {idx: comment for idx, comment in zip(commenter_indices, all_comments)}
        else:
            comment_map = {}

        next_obs, dones, valid_action, is_comment, camera_ready_contents = [], [], [], [], []
        for i, (action, active) in enumerate(zip(actions, active_mask)):
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_comment.append(0)
                camera_ready_contents.append('')
            elif action == 'camera-ready':
                next_obs.append('')
                dones.append(1)
                valid_action.append(1)
                is_comment.append(0)
                camera_ready_contents.append(contents[i])
            elif action == 'draft':
                feedback = comment_map[i]
                next_obs.append(f'\n\n<comment>{feedback}</comment>\n\n')
                dones.append(0)
                valid_action.append(1)
                is_comment.append(1)
                camera_ready_contents.append('')
            else:
                feedback = comment_map[i]
                next_obs.append(f'\n\n<comment>{feedback}</comment>\n\n')
                dones.append(0)
                valid_action.append(0)
                is_comment.append(1)
                camera_ready_contents.append('')

        return next_obs, dones, valid_action, is_comment, camera_ready_contents


    # ========== Paper Writing Specific Methods ==========
    def run_llm_loop_paper_writing_train_commenter(self, gen_batch, initial_input_ids: torch.Tensor,
                                                    num_revision_rounds: int = None):
        """
        多轮写作循环（Train-Commenter 版本，Autonomous 架构）。

        流程镜像 run_llm_loop_paper_writing_autonomous，但角色互换：
        - API generator 负责 <draft> 和 <camera-ready>（masked，不参与梯度）；
        - 本地 actor 只负责 <comment>（unmasked，参与梯度）。

        每一步：
        1. API generator 对所有 active 样本生成 (<draft> 或 <camera-ready>)；
        2. <camera-ready> → 该样本 done；
        3. <draft> → 本地模型生成 <comment>（trainable）；
        4. 更新 rolling state 和 right_side；
        5. active_mask 更新。

        强制最终轮：exhausted max_turns 的样本由 API 生成 fallback，masked。

        meta_info 只写 turns_stats, active_mask, valid_action_stats,
        valid_comment_stats, trace_mode（不写 camera_ready_texts 等）。
        non_tensor_batch 写 camera_ready, format_ok, final_source, raw_final。

        说明：num_revision_rounds 参数保留签名兼容性，本模式以 max_turns 控制终止。
        """
        print(f"\n[Paper Writing/TrainCommenter] Starting autonomous-style loop (max_turns={self.config.max_turns})")
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            'responses': initial_input_ids[:, []],
            'responses_with_info_mask': initial_input_ids[:, []]
        }

        batch_size = gen_batch.batch['input_ids'].shape[0]
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.ones(batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_comment_stats = torch.zeros(batch_size, dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        raw_last_responses: List[str] = [''] * batch_size
        rollings = gen_batch

        domains, topics = self._extract_domain_topic_from_batch(gen_batch)
        ground_truths = self._extract_ground_truth_from_batch(gen_batch)
        meta_info = {}

        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            rolling_texts = self.tokenizer.batch_decode(
                rollings.batch["input_ids"], skip_special_tokens=True
            )

            # Step 1: API 生成（仅对 active 样本）。
            active_indices = [i for i in range(batch_size) if active_mask[i]]
            api_results = self._call_generator_batch(
                contexts=[rolling_texts[i] for i in active_indices],
                domains=[domains[i] for i in active_indices],
                topics=[topics[i] for i in active_indices],
            )

            # 归一化并散射回完整 batch（inactive 样本保持 ''）。
            api_texts = [''] * batch_size
            for pos, idx in enumerate(active_indices):
                text = api_results[pos]
                if re.fullmatch(r'\s*<camera-ready>.*?</camera-ready>\s*', text, re.DOTALL):
                    api_texts[idx] = text.strip()
                elif self._is_strict_single_draft(text):
                    api_texts[idx] = text.strip()
                else:
                    api_texts[idx] = "<draft>Please revise the abstract to satisfy formatting requirements.</draft>"

            # 解析 API 动作：'camera-ready' | 'draft'。
            api_actions = [None] * batch_size
            for i in range(batch_size):
                text = api_texts[i]
                if re.search(r'<camera-ready>', text, re.DOTALL):
                    api_actions[i] = 'camera-ready'
                elif re.search(r'<draft>', text, re.DOTALL):
                    api_actions[i] = 'draft'

            # done：inactive 或 camera-ready。
            dones = [
                1 if (not bool(active_mask[i].item()) or api_actions[i] == 'camera-ready') else 0
                for i in range(batch_size)
            ]

            # 记录已完成样本的最后一次原始输出。
            for i, done in enumerate(dones):
                if done and bool(active_mask[i].item()) and not raw_last_responses[i]:
                    raw_last_responses[i] = api_texts[i]

            # 统计 valid API action。
            for i in range(batch_size):
                if bool(active_mask[i].item()) and api_actions[i] in ('camera-ready', 'draft'):
                    valid_action_stats[i] += 1

            # Tokenize API 输出（full batch，masked in right_side）。
            api_ids = self._batch_tokenize(api_texts)

            # Step 2: 本地模型对 draft 样本生成 <comment>（trainable）。
            draft_indices = [i for i in active_indices if api_actions[i] == 'draft']
            comment_block_texts = [''] * batch_size   # 用于 right_side
            comment_obs_texts = [''] * batch_size     # 用于 rolling context

            if draft_indices:
                comment_instruction_str = (
                    "\n\nPlease provide a concise reviewer comment for the previous draft. "
                    "Output exactly one <comment>...</comment> block.\n\n"
                )
                comment_instruction_ids = self._process_next_obs(
                    [comment_instruction_str] * batch_size
                )
                commenter_rollings_full = self._update_rolling_state(
                    rollings, api_ids, comment_instruction_ids
                )
                commenter_rollings_full.batch = self.tensor_fn.cut_to_effective_len(
                    commenter_rollings_full.batch,
                    keys=['input_ids', 'attention_mask', 'position_ids']
                )
                draft_mask = torch.tensor(
                    [i in set(draft_indices) for i in range(batch_size)], dtype=torch.bool
                )
                commenter_rollings_active = DataProto.from_dict({
                    k: v[draft_mask] for k, v in commenter_rollings_full.batch.items()
                })
                comment_output = self._generate_with_gpu_padding(commenter_rollings_active)
                meta_info = dict(getattr(comment_output, 'meta_info', {}) or {})

                raw_comment_strs = self.tokenizer.batch_decode(
                    comment_output.batch["responses"], skip_special_tokens=True
                )
                # 归一化并散射回完整 batch。
                for pos, idx in enumerate(draft_indices):
                    text = raw_comment_strs[pos]
                    m = re.search(r'<comment>(.*?)</comment>', text, re.DOTALL)
                    content = m.group(1).strip() if m else text.strip()
                    if not content:
                        content = "Please improve clarity, method detail, and result specificity."
                    comment_block_texts[idx] = f"<comment>{content}</comment>"
                    comment_obs_texts[idx] = f"\n\n<comment>{content}</comment>\n\n"
                    valid_comment_stats[idx] += 1

            # Tokenize comment（full batch；非 draft 样本为 '' → padding）。
            comment_ids = self._batch_tokenize(comment_block_texts)
            comment_obs_ids = self._process_next_obs(comment_obs_texts)

            # 更新 active_mask。
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1

            # Rolling state：追加 API output + comment obs。
            rollings = self._update_rolling_state(rollings, api_ids, comment_obs_ids)

            # Right-side：api draft → masked；comment → trainable。
            original_right_side = self._append_right_side_segment(
                original_right_side, api_ids, trainable=False
            )
            original_right_side = self._append_right_side_segment(
                original_right_side, comment_ids, trainable=True
            )

        # 强制最终轮：耗尽 max_turns 的样本由 API 生成（masked）。
        if active_mask.sum():
            print(f"[Paper Writing/TrainCommenter] {active_mask.sum().item()} sample(s) exhausted "
                  f"max_turns, forcing final API generation")
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            rolling_texts = self.tokenizer.batch_decode(
                rollings.batch["input_ids"], skip_special_tokens=True
            )
            active_indices = [i for i in range(batch_size) if active_mask[i]]
            api_results = self._call_generator_batch(
                contexts=[rolling_texts[i] for i in active_indices],
                domains=[domains[i] for i in active_indices],
                topics=[topics[i] for i in active_indices],
            )
            final_api_texts = [''] * batch_size
            for pos, idx in enumerate(active_indices):
                text = api_results[pos]
                if re.fullmatch(r'\s*<camera-ready>.*?</camera-ready>\s*', text, re.DOTALL):
                    final_api_texts[idx] = text.strip()
                elif self._is_strict_single_draft(text):
                    final_api_texts[idx] = text.strip()
                else:
                    final_api_texts[idx] = "<draft>Please revise the abstract to satisfy formatting requirements.</draft>"
            for i, active in enumerate(active_mask.tolist()):
                if active and not raw_last_responses[i]:
                    raw_last_responses[i] = final_api_texts[i]
            final_api_ids = self._batch_tokenize(final_api_texts)
            original_right_side = self._append_right_side_segment(
                original_right_side, final_api_ids, trainable=False
            )

        # Format gate：与 autonomous 版本保持一致。
        bodies: List[str] = [''] * batch_size
        format_oks: List[bool] = [False] * batch_size
        final_sources: List[str] = ['raw_fallback'] * batch_size
        for i in range(batch_size):
            raw = raw_last_responses[i] or ''
            body, ok, source = _extract_camera_ready_body_and_source(raw)
            bodies[i] = body
            format_oks[i] = ok
            final_sources[i] = source

        meta_info = dict(meta_info)
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_comment_stats'] = valid_comment_stats.tolist()
        meta_info['trace_mode'] = 'train_commenter'

        print(f"[Paper Writing/TrainCommenter] ACTIVE_TRAJ_NUM: {active_num_list}")
        final_output = self._compose_final_output(original_left_side, original_right_side, meta_info)
        final_output.non_tensor_batch['camera_ready'] = np.array(bodies, dtype=object)
        final_output.non_tensor_batch['format_ok'] = np.array(format_oks, dtype=object)
        final_output.non_tensor_batch['final_source'] = np.array(final_sources, dtype=object)
        final_output.non_tensor_batch['raw_final'] = np.array(raw_last_responses, dtype=object)
        return final_output
    def run_llm_loop_paper_writing_arena_seeded(self, gen_batch, initial_input_ids: torch.Tensor,
                                                 num_revision_rounds: int = None):
        """
        多轮写作循环（Arena Seeded 版本）。

        核心思想：
        - 先复用基线 `run_llm_loop_paper_writing` 完成 rollout；
        - 再基于最终 `camera_ready_texts` 做组内“带种子单轮 Swiss 排序”；
        - 使用数据集 `ground_truth` 作为 anchor，得到更干净的相对排序信号；
        - 将排序分写入 `meta_info['arena_scores']`，供 hybrid reward 融合使用。
        """
        base_output = self.run_llm_loop_paper_writing_autonomous(
            gen_batch=gen_batch,
            initial_input_ids=initial_input_ids,
        )
        camera_ready_arr = base_output.non_tensor_batch.get('camera_ready', None)
        camera_ready_texts = [str(x) for x in camera_ready_arr] if camera_ready_arr is not None else []
        anchors = self._extract_ground_truth_from_batch(gen_batch)
        seed = int(getattr(self.config, 'arena_seed', 20260413))
        group_size = int(getattr(self.config, 'arena_group_size', 8))
        arena_scores = self._compute_seeded_swiss_arena_scores(
            candidates=camera_ready_texts,
            anchors=anchors,
            group_size=group_size,
            seed=seed
        )
        base_output.meta_info['arena_scores'] = arena_scores
        base_output.meta_info['arena_seed_mode'] = getattr(self.config, 'arena_seed_mode', 'swiss_single_round')
        base_output.meta_info['trace_mode'] = 'arena_seeded'
        return base_output

    def run_llm_loop_paper_writing_autonomous(self, gen_batch, initial_input_ids: torch.Tensor):
        """
        自主决策论文写作循环（Autonomous 版本）。

        完全镜像 run_llm_loop 的 active_mask 架构：
          - LLM 输出 <draft>text</draft>         -> 触发 commenter API，返回 <comment>feedback</comment>，继续
          - LLM 输出 <camera-ready>text</camera-ready> -> 完成，记录 camera_ready_texts
          - rollout 中间不插入任何额外 instruction，LLM 自主决定何时终止。

        强制最终轮（max_turns 用完仍 active）：
          直接 append 原始输出到 right_side，格式由 reward evaluator 判断。

        Per-sample rollout 产出放 ``non_tensor_batch`` (DataProto 原生对齐)：
          - camera_ready   (object str)       已经过 Stage-6 format gate 清洗的最终正文
          - format_ok      (object bool)      True 表示格式合法可评分
          - final_source   (object str)       'camera_ready' / 'draft_as_camera_ready' /
                                              'raw_plain' / 'raw_fallback'
          - raw_final      (object str)       最后一轮的未清洗原始输出，调试用

        写入 ``meta_info``：
          - turns_stats, active_mask, valid_action_stats, valid_comment_stats, trace_mode
        """
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            'responses': initial_input_ids[:, []],
            'responses_with_info_mask': initial_input_ids[:, []]
        }

        batch_size = gen_batch.batch['input_ids'].shape[0]
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.ones(batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_comment_stats = torch.zeros(batch_size, dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        camera_ready_texts = [''] * batch_size
        # Raw last-turn output per sample: the untouched response string from the
        # turn in which the sample finished (or the forced final turn). Used at
        # the end to run the Stage-6 format gate once, consistently.
        raw_last_responses: List[str] = [''] * batch_size
        # Per-turn trace, indexed as [round_idx][sample_idx]. Used for rollout dumping.
        draft_texts_by_round: List[List[str]] = []
        comment_texts_by_round: List[List[str]] = []
        rollings = gen_batch

        domains, topics = self._extract_domain_topic_from_batch(gen_batch)
        ground_truths = self._extract_ground_truth_from_batch(gen_batch)

        print(f"\n[Paper Writing/Autonomous] Starting autonomous loop (max_turns={self.config.max_turns})")
        meta_info = {}

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info

            responses_ids, responses_str = self._postprocess_responses_paper_writing_autonomous(
                gen_output.batch['responses']
            )
            responses_ids, responses_str = self.tensor_fn._example_level_pad(
                responses_ids, responses_str, active_mask
            )

            next_obs, dones, valid_action, is_comment, camera_ready_contents = \
                self._execute_paper_writing_autonomous(
                    responses_str, active_mask,
                    domains=domains, topics=topics, ground_truths=ground_truths
                )

            # Record camera-ready texts + raw final output for samples that
            # finished via <camera-ready> in this step.
            for i, (content, done) in enumerate(zip(camera_ready_contents, dones)):
                if content and not camera_ready_texts[i]:
                    camera_ready_texts[i] = content
                if done and bool(active_mask[i].item()) and not raw_last_responses[i]:
                    raw_last_responses[i] = responses_str[i] if i < len(responses_str) else ''

            # Record per-turn drafts (from response) and comments (from commenter obs).
            round_drafts: List[str] = ['' for _ in range(batch_size)]
            round_comments: List[str] = ['' for _ in range(batch_size)]
            for i, resp in enumerate(responses_str):
                if not bool(active_mask[i].item()):
                    continue
                d_match = re.search(r'<draft>(.*?)</draft>', resp, re.DOTALL)
                if d_match:
                    round_drafts[i] = d_match.group(1).strip()
            for i, obs in enumerate(next_obs):
                if i >= batch_size or not bool(active_mask[i].item()):
                    continue
                c_match = re.search(r'<comment>(.*?)</comment>', obs, re.DOTALL)
                if c_match:
                    round_comments[i] = c_match.group(1).strip()
            draft_texts_by_round.append(round_drafts)
            comment_texts_by_round.append(round_comments)

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_comment_stats += torch.tensor(is_comment, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            original_right_side = self._update_right_side(
                original_right_side, responses_ids, next_obs_ids
            )
            # pdb.set_trace()

        # Forced final generation for samples that exhausted max_turns without <camera-ready>
        if active_mask.sum():
            print(f"[Paper Writing/Autonomous] {active_mask.sum().item()} sample(s) exhausted "
                  f"max_turns, forcing final generation")
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info

            responses_ids, responses_str = self._postprocess_responses_paper_writing_autonomous(
                gen_output.batch['responses']
            )
            responses_ids, responses_str = self.tensor_fn._example_level_pad(
                responses_ids, responses_str, active_mask
            )

            # Stash the raw final-turn output; body/format_ok extraction happens
            # in the unified Stage-6 gate below.
            for i, active in enumerate(active_mask.tolist()):
                if active:
                    raw_last_responses[i] = responses_str[i] if i < len(responses_str) else ''

            original_right_side = self._update_right_side(
                original_right_side, responses_ids
            )

        # format gate: derive a single clean body + format_ok +
        # final_source for every sample from its raw last-turn output. This
        # replaces the old dumper-side _xml_pattern tripwire and accepts
        # <draft>...</draft> in the final turn as a valid camera-ready.
        bodies: List[str] = [''] * batch_size
        format_oks: List[bool] = [False] * batch_size
        final_sources: List[str] = ['raw_fallback'] * batch_size
        for i in range(batch_size):
            raw = raw_last_responses[i] or ''
            body, ok, source = _extract_camera_ready_body_and_source(raw)
            bodies[i] = body
            format_oks[i] = ok
            final_sources[i] = source
            camera_ready_texts[i] = body

        meta_info = dict(meta_info)
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_comment_stats'] = valid_comment_stats.tolist()
        meta_info['trace_mode'] = 'autonomous'

        print(f"[Paper Writing/Autonomous] ACTIVE_TRAJ_NUM: {active_num_list}")
        final_output = self._compose_final_output(original_left_side, original_right_side, meta_info)

        # Attach per-sample fields to non_tensor_batch so that DataProto's
        # reorder/repeat/pop/union automatically keep them aligned with the
        # batch tensors (Stage 2 of the refactor).
        final_output.non_tensor_batch['camera_ready'] = np.array(bodies, dtype=object)
        final_output.non_tensor_batch['format_ok'] = np.array(format_oks, dtype=object)
        final_output.non_tensor_batch['final_source'] = np.array(final_sources, dtype=object)
        final_output.non_tensor_batch['raw_final'] = np.array(raw_last_responses, dtype=object)
        os.environ.get('PW_DEBUG') and __import__('pdb').set_trace()
        responses_str = self.tokenizer.batch_decode(original_right_side['responses'],skip_special_tokens=True)
        responses_with_info_mask_str = self.tokenizer.batch_decode(original_right_side['responses_with_info_mask'],skip_special_tokens=True)
        return final_output

    def _extract_queries_from_batch(self, gen_batch):
        """Extract query text from non_tensor_batch (saved during data processing)."""
        queries = []
        batch_size = gen_batch.batch['input_ids'].shape[0]

        # Access non_tensor_batch to get extra_info
        if hasattr(gen_batch, 'non_tensor_batch') and gen_batch.non_tensor_batch:
            if 'extra_info' in gen_batch.non_tensor_batch:
                extra_infos = gen_batch.non_tensor_batch['extra_info']
                for extra_info in extra_infos:
                    query = extra_info.get('query', '')
                    queries.append(query)
            else:
                raise ValueError("extra_info not found in non_tensor_batch")
        else:
            raise ValueError("non_tensor_batch not available in gen_batch")

        return queries

    def _extract_domain_topic_from_batch(self, gen_batch):
        """Extract domain and topic from non_tensor_batch (saved during data processing).
        在arxiv数据集上，domain 是keywords，topic 是title
        """
        domains = []
        topics = []

        # Access non_tensor_batch to get extra_info
        if hasattr(gen_batch, 'non_tensor_batch') and gen_batch.non_tensor_batch:
            if 'extra_info' in gen_batch.non_tensor_batch:
                extra_infos = gen_batch.non_tensor_batch['extra_info']
                for extra_info in extra_infos:
                    domain = extra_info.get('keywords', extra_info.get('domain', 'General'))
                    topic = extra_info.get('title', extra_info.get('topic', 'research topic'))
                    domains.append(domain)
                    topics.append(topic)
            else:
                raise ValueError("extra_info not found in non_tensor_batch")
        else:
            raise ValueError("non_tensor_batch not available in gen_batch")

        return domains, topics

    def _extract_ground_truth_from_batch(self, gen_batch):
        """Extract optional ground truth from non_tensor_batch['reward_model'].

        Returns:
            List[str]: one ground_truth per sample; empty string if not provided.
        """
        batch_size = gen_batch.batch['input_ids'].shape[0]
        ground_truths = [''] * batch_size

        if not (hasattr(gen_batch, 'non_tensor_batch') and gen_batch.non_tensor_batch):
            return ground_truths

        reward_models = gen_batch.non_tensor_batch.get('reward_model', None)
        if reward_models is None:
            return ground_truths

        for i in range(min(batch_size, len(reward_models))):
            reward_model = reward_models[i]
            if isinstance(reward_model, dict):
                gt = reward_model.get('ground_truth', '')
                ground_truths[i] = gt if isinstance(gt, str) else ('' if gt is None else str(gt))

        return ground_truths
    
    def _extract_drafts_with_fallback(self, responses_str, round_idx):
        """
        Extract <draft>...</draft> content from responses.
        If missing, use empty string and log warning.
        
        Returns:
            drafts: List of draft texts (empty string if tag missing)
            failed_indices: List of sample indices that failed
        """
        drafts = []
        failed_indices = []
        
        for i, resp in enumerate(responses_str):
            match = re.search(r'<draft>(.*?)</draft>', resp, re.DOTALL)
            if match:
                drafts.append(match.group(1).strip())
            else:
                drafts.append('')
                failed_indices.append(i)
                print(f"\n[WARNING] Sample {i} missing <draft> tag in round {round_idx}")
                print(f"Response text:\n{resp[:500]}...")
                print()
        
        if failed_indices:
            print(f"[Paper Writing] {len(failed_indices)}/{len(responses_str)} samples missing <draft> tag, continuing with empty drafts")
        
        return drafts, failed_indices
    
    def _extract_camera_ready_with_fallback(self, responses_str):
        """
        Extract <camera-ready>...</camera-ready> content from responses.
        If missing, use empty string and log warning.
        
        Returns:
            camera_readys: List of camera-ready texts (empty string if tag missing)
            failed_indices: List of sample indices that failed
        """
        camera_readys = []
        failed_indices = []
        
        for i, resp in enumerate(responses_str):
            match = re.search(r'<camera-ready>(.*?)</camera-ready>', resp, re.DOTALL)
            if match:
                camera_readys.append(match.group(1).strip())
            else:
                camera_readys.append('')
                failed_indices.append(i)
                print(f"\n[WARNING] Sample {i} missing <camera-ready> tag in final round")
                print(f"Response text:\n{resp[:500]}...")
                print()
        
        if failed_indices:
            print(f"[Paper Writing] {len(failed_indices)}/{len(responses_str)} samples missing <camera-ready> tag")
        
        return camera_readys, failed_indices
    
    def _get_commenter_client(self):
        """Get or create a long-lived commenter client to avoid frequent transport teardown."""
        if self._commenter_client is None:
            from openai import OpenAI
            self._commenter_client = OpenAI(
                api_key=self.config.commenter_api_key,
                base_url=self.config.commenter_base_url,
                timeout=60,  # Set a reasonable timeout for commenter API calls
                max_retries=3  # Disable retries to fail fast on issues
            )
        return self._commenter_client

    def _get_generator_client(self):
        """获取/复用 API generator 客户端（长连接）。

        这里复用 commenter 的 API 配置，避免额外引入一套独立配置。
        """
        if self._generator_client is None:
            from openai import OpenAI
            self._generator_client = OpenAI(
                api_key=self.config.commenter_api_key,
                base_url=self.config.commenter_base_url,
                timeout=60,  # Set a reasonable timeout for generator API calls
                max_retries=3  # Disable retries to fail fast on issues
            )
        return self._generator_client

    def _call_generator_single(self, client, context: str, domain: str, topic: str) -> str:
        """调用外部 API 生成单条 draft，要求返回 `<draft>...</draft>`。"""
        system_prompt = (
            "You are an academic writing assistant. Output exactly one <draft>...</draft> block "
            "and nothing else."
        )
        user_content = (
            f"Keywords: {domain}\n"
            f"Title: {topic}\n\n"
            "Conversation context:\n"
            f"{context}\n\n"
            "Write the next abstract draft now."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        try:
            resp = client.chat.completions.create(
                model=self.config.commenter_model,
                messages=messages,
                max_tokens=512,
                temperature=0.8,
                extra_body={"enable_thinking": False}
            )
            content = resp.choices[0].message.content
            if isinstance(content, str) and content.strip():
                return content
        except Exception as e:
            print(f"[ERROR] Generator API call failed: {e}")
        return "<draft>Please provide a complete abstract with motivation, method, results, and conclusion.</draft>"

    def _call_generator_batch(self, contexts: List[str], domains: List[str], topics: List[str]) -> List[str]:
        """批量调用 API 生成 draft（线程池并发）。"""
        if not contexts:
            return []
        client = self._get_generator_client()
        pairs = list(zip(contexts, domains, topics))
        max_workers = min(max(1, int(self.config.generator_max_concurrency)), len(pairs))
        with self._gpu_filler:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._call_generator_single, client, context, domain, topic)
                    for context, domain, topic in pairs
                ]
                return [f.result() for f in futures]

    def _call_commenter_single(self, client, system_prompt, domain, topic, draft, ground_truth=''):
        """Call Commenter API for a single sample (sync)."""
        has_ground_truth = isinstance(ground_truth, str) and len(ground_truth.strip()) > 0
        if has_ground_truth:
            user_content = (
                f"Keywords: {domain}\n"
                f"Title: {topic}\n\n"
                f"Current Draft:\n{draft}\n\n"
                "Reference Text (from dataset ground_truth; use only as a potentially helpful reference, "
                "not as an absolute correct answer):\n"
                f"{ground_truth}\n\n"
                "Please compare the current draft with the reference text and provide specific, constructive "
                "revision suggestions. The reference may be imperfect, so do not blindly copy it."
            )
        else:
            user_content = (
                f"Keywords: {domain}\n"
                f"Title: {topic}\n\n"
                f"Current Draft:\n{draft}\n\n"
                "Please provide specific, constructive revision suggestions."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            self._commenter_rate_limiter.acquire()
            resp = client.chat.completions.create(
                model=self.config.commenter_model,
                messages=messages,
                max_tokens=256,
                temperature=0.8,
                extra_body={"enable_thinking": False}
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] Commenter API call failed: {e}")
            return "Please revise your draft to improve clarity and academic rigor."
    
    def _call_commenter_batch(self, domains, topics, drafts, ground_truths=None):
        """Call Commenter API for a batch of samples (threaded parallel)."""
        client = self._get_commenter_client()

        if ground_truths is None:
            ground_truths = [''] * len(drafts)

        system_prompt = (
            "You are an experienced academic reviewer. Your task is to read abstract drafts "
            "and provide specific, constructive revision suggestions.\n\n"
            "FORMAT CHECK (do this first):\n"
            "- Check whether the input is clean academic prose: no XML tags (e.g. <draft>, "
            "<camera-ready>), no markdown, no bullet points, no template placeholders.\n"
            "- If format issues are found, start your feedback with \"FORMAT NOTE: [describe the issue]\" "
            "on the first line, then continue with content suggestions as usual.\n"
            "- If the format is clean, skip the FORMAT NOTE line entirely.\n\n"
            "CRITICAL LENGTH CONSTRAINT:\n"
            "- Your feedback MUST be 50-80 words (approximately 70-120 tokens)\n"
            "- Be concise and focus on the most important issues only\n"
            "- Prioritize: 1) Format issues (if any), 2) Clarity issues, 3) Missing key elements, "
            "4) Structural problems\n\n"
            "IMPORTANT OUTPUT FORMAT:\n"
            "- Output ONLY plain text revision suggestions\n"
            "- Do NOT use any XML tags (like <comment>, <draft>, <camera-ready>, etc.) in your response\n"
            "- Do NOT repeat or include the draft content in your response\n"
            "- Provide 2-3 specific, actionable suggestions directly"
        )

        pairs = list(zip(domains, topics, drafts, ground_truths))
        if not pairs:
            return []

        max_workers = min(max(1, int(getattr(self.config, 'commenter_max_concurrency', 48))), len(pairs))
        with self._gpu_filler:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._call_commenter_single,
                        client,
                        system_prompt,
                        domain,
                        topic,
                        draft,
                        ground_truth
                    )
                    for domain, topic, draft, ground_truth in pairs
                ]
                comments = [f.result() for f in futures]

        return comments
