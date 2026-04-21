import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import hashlib
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
import asyncio
import pdb
from concurrent.futures import ThreadPoolExecutor

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
    arena_seed_mode: str = "swiss_single_round"
    arena_seed: int = 20260413
    arena_group_size: int = 8

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

        # Collect drafts that need commenter feedback (active samples only)
        draft_indices = [
            i for i, (action, active) in enumerate(zip(actions, active_mask))
            if active and action == 'draft'
        ]
        if draft_indices:
            draft_domains = [domains[i] for i in draft_indices]
            draft_topics = [topics[i] for i in draft_indices]
            draft_texts = [contents[i] for i in draft_indices]
            draft_ground_truths = [ground_truths[i] for i in draft_indices]
            comments = self._call_commenter_batch(
                draft_domains, draft_topics, draft_texts, draft_ground_truths
            )
        else:
            comments = []

        comment_iter = iter(comments)
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
                feedback = next(comment_iter)
                next_obs.append(f'\n\n<comment>{feedback}</comment>\n\n')
                dones.append(0)
                valid_action.append(1)
                is_comment.append(1)
                camera_ready_contents.append('')
            else:
                next_obs.append(
                    '\nMy previous action is invalid. '
                    'If I want to request reviewer feedback, I should put my draft between <draft> and </draft>. '
                    'If I want to submit my final abstract, I should put it between <camera-ready> and </camera-ready>. '
                    'Let me try again.\n'
                )
                dones.append(0)
                valid_action.append(0)
                is_comment.append(0)
                camera_ready_contents.append('')

        return next_obs, dones, valid_action, is_comment, camera_ready_contents


    # ========== Paper Writing Specific Methods ==========
    def run_llm_loop_paper_writing(self, gen_batch, initial_input_ids: torch.Tensor,
                                   num_revision_rounds: int = None):
        """
        Run multi-turn paper writing loop with fixed revision rounds.
        
        Flow:
        1. Generate initial draft -> get comment from Commenter API
        2. Repeat K times: generate revised draft -> get comment  
        3. Generate final camera-ready version
        
        Args:
            gen_batch: DataProto with input_ids, attention_mask, position_ids
            initial_input_ids: Initial prompt tokens
            num_revision_rounds: Number of draft-comment cycles (default: use config)
        
        Returns:
            final_output: DataProto with complete generation
            
         ---             
        最终输出                                                                                                                   
                        
        调用 _compose_final_output 组合最终结果：
                                                                                                                                    
        final_output = {
            'prompts': query,                                                                                                      
            'responses': draft_0 + comment_0 + draft_1 + comment_1 + draft_2 + comment_2 + camera_ready,
            'responses_with_info_mask': draft_0 + [pad_mask] + draft_1 + [pad_mask] + draft_2 + [pad_mask] + camera_ready,         
            'input_ids': query + responses,                                                                                        
            'attention_mask': [1, 1, 1, ...],  # 基于 input_ids                                                                    
            'info_mask': [1, 1, 1, 0, 0, 1, 1, 0, 0, ...],  # 基于 responses_with_info_mask，comments 位置为 0                     
            'position_ids': [0, 1, 2, 3, ...],                                                                                     
            'meta_info': {                                                                                                         
                'num_revision_rounds': 3,                                                                                          
                'camera_ready_texts': [camera_ready 的文本内容]                                                                    
            }                                                                                                                      
        }                                                                                                                          
                                                                                                                                    
        ---             
        关键点总结                                                                                                                 
                                                                                                                                    
        1. rollings（滚动上下文）：
            - 累积，每轮保留：query + 所有的（ draft + comment + 指令遵循prompt）
            - 用于下一轮 gen-agent 的输入                                                                                            
        2. original_right_side（最终输出）：                                                                                       
            - 累积所有 drafts 和 comments                                                                                            
            - responses：完整的生成序列                                                                                              
            - responses_with_info_mask：所有非generator生成的内容 被替换为 pad_mask                                                                   
        3. mask 机制：                                                                                                             
            - gen-agent 生成的内容（drafts, camera-ready）：不加 mask                                                                
            - commenter 生成的内容（comments）：加 mask（替换为 pad_id）                                                             
            - 用于训练时只对 gen-agent 的输出计算梯度                                                                                
        4. 输入逻辑：                                                                                                              
            - 第 0 轮生成 draft_0：输入 = query                                                                                      
            - 第 1 轮生成 draft_1：输入 = query + draft_0 + comment_0                                                                
            - 第 2 轮生成 draft_2：输入 = query + draft_0 + comment_0 + draft_1 + comment_1                                                                
            - 第 i 轮生成 draft_i：输入 = query + draft_0 + comment_0 + ... + draft_{i-1} + comment_{i-1}                                                        
            - 最终生成 camera-ready：输入 = query + draft_0 + comment_0 + ... + draft_{最后一轮} + comment_{最后一轮}  
        
        1）question: 训练多轮长trace，实际测试并非多轮 -> 可能导致模型过拟合在多轮交互的模版，而非真正的写作能力； 
        可能的方案：最终的输出（original_right_side）只保留 query + 最后一轮的 draft + comment
        
        2）训commenter的版本；
        
        3）Qwe-Max as Evaluator: noisy; 真实的abstract没有利用起来；-> 参考arenaRL
        
        4）自己决定多少个turn；[TBD] 
        
        
        5）待定；data方面：
        auto-rubric, 检查数据是否都是写作任务的？ ❓
        
        """
        
        
        if num_revision_rounds is None:
            num_revision_rounds = self.config.num_revision_rounds
        
        print(f"\n[Paper Writing] Starting {num_revision_rounds}-round revision process")
        
        # Initialize
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            'responses': initial_input_ids[:, []], 
            'responses_with_info_mask': initial_input_ids[:, []]
        }
        # Segment-level old/ref append path is temporarily disabled.
        # We will recompute old/ref once on the final accumulated sequence in ray_trainer.
        # if not self.is_validation:
        #     original_right_side['old_log_probs'] = initial_input_ids[:, []].float()
        #     original_right_side['ref_log_prob'] = initial_input_ids[:, []].float()
        rollings = gen_batch
        
        # Extract queries from non_tensor_batch; system prompt 和 query应该分离；手动控制循环轮次的时候，gen的system-prompt无需说明循环过程，只需告诉它如果有前轮的drfat和comment，就据此修改，如果没有就重新写draft；
        queries = self._extract_queries_from_batch(gen_batch)
        
        # Extract domain and topic for Commenter API
        domains, topics = self._extract_domain_topic_from_batch(gen_batch)
        
        # Extract optional ground truth references for commenter.
        # Ground truth is only a reference, not necessarily better than current draft.
        ground_truths = self._extract_ground_truth_from_batch(gen_batch)
        ground_truth_lengths = self._text_token_lengths(ground_truths)
        draft_valid_flags_by_round = []
        draft_lengths_by_round = []
        draft_texts_by_round = []
        comments_by_round = []
        comment_obs_by_round = []

        # pdb.set_trace()
        # ===== Phase 1: K rounds of draft-comment cycles =====
        for round_idx in range(num_revision_rounds):
            print(f"\n[Paper Writing] Round {round_idx + 1}/{num_revision_rounds}")
            
            # Cut to effective length
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            # Debug: print current rolling context, the input of gen-agent each turn
            gen_input_str = self.tokenizer.batch_decode(rollings.batch["input_ids"], skip_special_tokens=True)
            
            # Generate draft
            gen_output = self._generate_with_gpu_padding(rollings)
    
            draft_str = self.tokenizer.batch_decode(gen_output.batch["responses"], skip_special_tokens=True)
            draft_ids = self._batch_tokenize(draft_str)
            valid_flags = [self._is_strict_single_draft(x) for x in draft_str]
            draft_valid_flags_by_round.append(valid_flags)
            draft_texts_by_round.append(draft_str)
            draft_length_texts = [self._draft_content_for_length(x) for x in draft_str]
            draft_lengths_by_round.append(self._text_token_lengths(draft_length_texts))
            invalid_count = sum(1 for ok in valid_flags if not ok)
            if invalid_count > 0:
                print(f"[Paper Writing] Round {round_idx + 1}: invalid draft format count = {invalid_count}/{len(draft_str)}")
            
            # Call Commenter API only for valid draft outputs.
            comments = [''] * len(draft_str)
            valid_indices = [i for i, ok in enumerate(valid_flags) if ok]
            invalid_indices = [i for i, ok in enumerate(valid_flags) if not ok]
            if valid_indices:
                valid_domains = [domains[i] for i in valid_indices]
                valid_topics = [topics[i] for i in valid_indices]
                valid_drafts = [draft_str[i] for i in valid_indices]
                valid_ground_truths = [ground_truths[i] for i in valid_indices] if ground_truths is not None else None
                valid_comments = self._call_commenter_batch(valid_domains, valid_topics, valid_drafts, valid_ground_truths)
                for i, idx in enumerate(valid_indices):
                    comments[idx] = valid_comments[i]
            comments_by_round.append(comments)
            # print(f'round_idx: {round_idx}, comments[0]: {comments[0]}')
            
            # Build next observations:
            # - valid output: reviewer comment + next-round instruction at tail
            # - invalid output: corrective feedback + next-round instruction at tail
            next_round = round_idx + 2
            if round_idx < num_revision_rounds - 1:
                round_instruction = (
                    f"Round {next_round}: Please write your draft now. "
                    "Output only one <draft>...</draft> block."
                )
            else:
                round_instruction = (
                    "Final step: Based on the latest draft and reviewer comments, write the final camera-ready abstract. "
                    "Output only the abstract text. Do not include XML tags, comments, explanations, or markdown."
                )
            comment_obs = []
            for i, ok in enumerate(valid_flags):
                if ok:
                    feedback = comments[i]
                else:
                    feedback = (
                        "Your previous output format is invalid. "
                        "You must output exactly one <draft>...</draft> block and nothing else."
                    )
                comment_obs.append(f'\n\n<comment>{feedback}</comment>\n\n{round_instruction}\n\n')
            comment_obs_by_round.append(comment_obs)
            comment_obs_ids = self._process_next_obs(comment_obs)

            # Accumulative rolling context:
            # q -> q,d1,c1 -> q,d1,c1,d2,c2 -> ...
            rollings = self._update_rolling_state(
                rollings,
                draft_ids,
                comment_obs_ids
            )

            # Update right_side (accumulative: for final output and mask)
            prev_responses = original_right_side['responses']
            original_right_side = self._update_right_side(
                original_right_side, draft_ids, comment_obs_ids
            )
            
            responses_str = self.tokenizer.batch_decode(original_right_side["responses"], skip_special_tokens=True)
            responses_with_info_mask_str = self.tokenizer.batch_decode(original_right_side["responses_with_info_mask"], skip_special_tokens=True)
            
        # ===== Phase 2: Generate final camera-ready =====
        print(f"\n[Paper Writing] Generating final camera-ready version")
        

        rollings.batch = self.tensor_fn.cut_to_effective_len(
            rollings.batch,
            keys=['input_ids', 'attention_mask', 'position_ids']
        )
        
        gen_output = self._generate_with_gpu_padding(rollings)
        final_str = self.tokenizer.batch_decode(gen_output.batch["responses"], skip_special_tokens=True)
        camera_ready_lengths = self._text_token_lengths(final_str)
        # final_str, _ = self._normalize_draft_texts(raw_final_str, round_idx=-1)
        final_ids = self._batch_tokenize(final_str)
        # Segment-level old/ref computation is disabled for accumulative rolling mode.
        # if not self.is_validation:
        #     final_old_lp, final_ref_lp = self._compute_segment_old_ref_logprob(rollings, final_ids)
        
        # Extract camera-ready  content nothing different from draft
        # camera_readys, failed_indices = self._extract_drafts_with_fallback(final_str, round_idx=num_revision_rounds)

        # Update right_side (no comment this time)
        prev_responses = original_right_side['responses']
        original_right_side = self._update_right_side(
            original_right_side, final_ids, next_obs_ids=None
        )
        # Segment-level old/ref append is disabled for accumulative rolling mode.
        # if not self.is_validation:
        #     target_len = original_right_side['responses'].shape[1]
        #     original_right_side['old_log_probs'] = self._update_dense_right_side_aligned(
        #         prev_tokens=prev_responses,
        #         prev_dense=original_right_side['old_log_probs'],
        #         cur_tokens=final_ids,
        #         cur_dense=final_old_lp,
        #         target_len=target_len
        #     )
        #     original_right_side['ref_log_prob'] = self._update_dense_right_side_aligned(
        #         prev_tokens=prev_responses,
        #         prev_dense=original_right_side['ref_log_prob'],
        #         cur_tokens=final_ids,
        #         cur_dense=final_ref_lp,
        #         target_len=target_len
        #     )
        #     assert original_right_side['responses'].shape == original_right_side['old_log_probs'].shape, \
        #         f"responses vs old_log_probs mismatch: {original_right_side['responses'].shape} vs {original_right_side['old_log_probs'].shape}"
        #     assert original_right_side['responses'].shape == original_right_side['ref_log_prob'].shape, \
        #         f"responses vs ref_log_prob mismatch: {original_right_side['responses'].shape} vs {original_right_side['ref_log_prob'].shape}"
        
        # Compose final output
        # Keep rollout meta_info (e.g. micro_batch_size/temperature/use_dynamic_bsz/max_token_len)
        # to stay compatible with downstream compute_log_prob in custom training paths.
        meta_info = dict(getattr(gen_output, 'meta_info', {}) or {})
        meta_info['num_revision_rounds'] = num_revision_rounds
        meta_info['camera_ready_texts'] = final_str
        meta_info['paper_writing_draft_valid_flags'] = draft_valid_flags_by_round
        meta_info['paper_writing_draft_lengths'] = draft_lengths_by_round
        meta_info['paper_writing_camera_ready_lengths'] = camera_ready_lengths
        meta_info['paper_writing_ground_truth_lengths'] = ground_truth_lengths
        meta_info['paper_writing_draft_texts'] = draft_texts_by_round
        meta_info['paper_writing_comment_texts'] = comments_by_round
        meta_info['paper_writing_comment_obs_texts'] = comment_obs_by_round
        meta_info['paper_writing_ground_truth_texts'] = ground_truths
        meta_info['paper_writing_domains'] = domains
        meta_info['paper_writing_topics'] = topics
        print(f"[Paper Writing] Completed {num_revision_rounds} revision rounds\n")
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def run_llm_loop_paper_writing_last_round_target(self, gen_batch, initial_input_ids: torch.Tensor,
                                                      num_revision_rounds: int = None):
        """
        多轮写作循环（Last-Round Target 版本）。

        目标：
        1. rollout 侧仍保留“多轮 draft-comment”交互，保证生成过程不变；
        2. 训练侧只保留最后一轮的 `draft + comment`，避免长 trace 模板过拟合；
        3. 最终 camera-ready 文本仍会生成并写入 `meta_info['camera_ready_texts']` 供 reward 使用，
           但不会拼入 responses（即不作为训练 token）。

        输入：
        - gen_batch: 含 input_ids/attention_mask/position_ids 的 DataProto
        - initial_input_ids: 起始 query token
        - num_revision_rounds: 修订轮次（默认取 config）

        输出：
        - DataProto（通过 _compose_final_output 组织）
        - 其中 right_side 只包含最后一轮 `draft + comment(+round_instruction)`；
          comment/instruction 在 responses_with_info_mask 中会被 mask（不可训练）。
        """
        if num_revision_rounds is None:
            num_revision_rounds = self.config.num_revision_rounds

        print(f"\n[Paper Writing/LastRoundTarget] Starting {num_revision_rounds}-round revision process")
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        rollings = gen_batch

        domains, topics = self._extract_domain_topic_from_batch(gen_batch)
        ground_truths = self._extract_ground_truth_from_batch(gen_batch)
        ground_truth_lengths = self._text_token_lengths(ground_truths)

        last_draft_ids = initial_input_ids[:, :0]
        last_comment_obs_ids = initial_input_ids[:, :0]

        for round_idx in range(num_revision_rounds):
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            gen_output = self._generate_with_gpu_padding(rollings)
            draft_str = self.tokenizer.batch_decode(gen_output.batch["responses"], skip_special_tokens=True)
            draft_ids = self._batch_tokenize(draft_str)
            valid_flags = [self._is_strict_single_draft(x) for x in draft_str]

            comments = [''] * len(draft_str)
            valid_indices = [i for i, ok in enumerate(valid_flags) if ok]
            if valid_indices:
                valid_domains = [domains[i] for i in valid_indices]
                valid_topics = [topics[i] for i in valid_indices]
                valid_drafts = [draft_str[i] for i in valid_indices]
                valid_ground_truths = [ground_truths[i] for i in valid_indices]
                valid_comments = self._call_commenter_batch(
                    valid_domains, valid_topics, valid_drafts, valid_ground_truths
                )
                for i, idx in enumerate(valid_indices):
                    comments[idx] = valid_comments[i]

            next_round = round_idx + 2
            round_instruction = self._make_round_instruction(next_round)
            comment_obs = []
            for i, ok in enumerate(valid_flags):
                if ok:
                    feedback = comments[i]
                else:
                    feedback = (
                        "Your previous output format is invalid. "
                        "You must output exactly one <draft>...</draft> block and nothing else."
                    )
                comment_obs.append(f'\n\n<comment>{feedback}</comment>\n\n{round_instruction}\n\n')
            comment_obs_ids = self._process_next_obs(comment_obs)

            rollings = self._update_rolling_state(rollings, draft_ids, comment_obs_ids)
            last_draft_ids = draft_ids
            last_comment_obs_ids = comment_obs_ids

        # 额外生成一次最终稿，仅用于 reward 与日志，不加入训练序列。
        rollings.batch = self.tensor_fn.cut_to_effective_len(
            rollings.batch,
            keys=['input_ids', 'attention_mask', 'position_ids']
        )
        final_output = self._generate_with_gpu_padding(rollings)
        final_str = self.tokenizer.batch_decode(final_output.batch["responses"], skip_special_tokens=True)

        # right_side 仅保留最后一轮内容（draft + comment/instruction）。
        original_right_side = {
            'responses': initial_input_ids[:, []],
            'responses_with_info_mask': initial_input_ids[:, []]
        }
        original_right_side = self._update_right_side(original_right_side, last_draft_ids, last_comment_obs_ids)

        meta_info = dict(getattr(final_output, 'meta_info', {}) or {})
        meta_info['num_revision_rounds'] = num_revision_rounds
        meta_info['camera_ready_texts'] = final_str
        meta_info['paper_writing_ground_truth_texts'] = ground_truths
        meta_info['paper_writing_ground_truth_lengths'] = ground_truth_lengths
        meta_info['paper_writing_domains'] = domains
        meta_info['paper_writing_topics'] = topics
        meta_info['trace_mode'] = 'last_round_target'
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def run_llm_loop_paper_writing_per_segment(self, gen_batch, initial_input_ids: torch.Tensor,
                                                     num_revision_rounds: int = None):
        """
        多轮写作循环（Per-Segment 信用分配版本）。

        rollout 侧保持累积上下文；训练侧让 draft 和 camera-ready 参与 loss，
        comment/instruction 通过 info_mask 遮掉。每段有独立的 segment_label，
        用于 per-draft 奖励信号和信用分配。
        """
        if num_revision_rounds is None:
            num_revision_rounds = self.config.num_revision_rounds

        print(f"\n[Paper Writing/PerSegment] Starting {num_revision_rounds}-round revision process")
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            'responses': initial_input_ids[:, []],
            'responses_with_info_mask': initial_input_ids[:, []]
        }
        rollings = gen_batch

        domains, topics = self._extract_domain_topic_from_batch(gen_batch)
        ground_truths = self._extract_ground_truth_from_batch(gen_batch)
        ground_truth_lengths = self._text_token_lengths(ground_truths)

        draft_valid_flags_by_round = []
        draft_lengths_by_round = []
        draft_texts_by_round = []
        comments_by_round = []
        comment_obs_by_round = []

        for round_idx in range(num_revision_rounds):
            print(f"\n[Paper Writing/PerSegment] Round {round_idx + 1}/{num_revision_rounds}")
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            gen_output = self._generate_with_gpu_padding(rollings)
            draft_str = self.tokenizer.batch_decode(gen_output.batch["responses"], skip_special_tokens=True)
            draft_ids = self._batch_tokenize(draft_str)
            valid_flags = [self._is_strict_single_draft(x) for x in draft_str]
            draft_valid_flags_by_round.append(valid_flags)
            draft_texts_by_round.append(draft_str)
            draft_length_texts = [self._draft_content_for_length(x) for x in draft_str]
            draft_lengths_by_round.append(self._text_token_lengths(draft_length_texts))

            invalid_count = sum(1 for ok in valid_flags if not ok)
            if invalid_count > 0:
                print(f"[Paper Writing/PerSegment] Round {round_idx + 1}: invalid draft format count = {invalid_count}/{len(draft_str)}")

            comments = [''] * len(draft_str)
            valid_indices = [i for i, ok in enumerate(valid_flags) if ok]
            if valid_indices:
                valid_domains = [domains[i] for i in valid_indices]
                valid_topics = [topics[i] for i in valid_indices]
                valid_drafts = [draft_str[i] for i in valid_indices]
                valid_ground_truths = [ground_truths[i] for i in valid_indices] if ground_truths is not None else None
                valid_comments = self._call_commenter_batch(valid_domains, valid_topics, valid_drafts, valid_ground_truths)
                for i, idx in enumerate(valid_indices):
                    comments[idx] = valid_comments[i]
            comments_by_round.append(comments)

            next_round = round_idx + 2
            if round_idx < num_revision_rounds - 1:
                round_instruction = self._make_round_instruction(next_round)
            else:
                round_instruction = (
                    "Final step: Based on the latest draft and reviewer comments, write the final camera-ready abstract. "
                    "Output only the abstract text. Do not include XML tags, comments, explanations, or markdown."
                )

            comment_obs = []
            for i, ok in enumerate(valid_flags):
                if ok:
                    feedback = comments[i]
                else:
                    feedback = (
                        "Your previous output format is invalid. "
                        "You must output exactly one <draft>...</draft> block and nothing else."
                    )
                comment_obs.append(f'\n\n<comment>{feedback}</comment>\n\n{round_instruction}\n\n')
            comment_obs_by_round.append(comment_obs)
            comment_obs_ids = self._process_next_obs(comment_obs)

            rollings = self._update_rolling_state(rollings, draft_ids, comment_obs_ids)
            # segment_label: draft_i = 2*round_idx+1, comment_i = 2*round_idx+2
            # Drafts are trainable (participate in loss); comments are not.
            original_right_side = self._append_right_side_segment(
                original_right_side, draft_ids, trainable=True,
                segment_label=2 * round_idx + 1,
            )
            original_right_side = self._append_right_side_segment(
                original_right_side, comment_obs_ids, trainable=False,
                segment_label=2 * round_idx + 2,
            )

        print(f"\n[Paper Writing/PerSegment] Generating final camera-ready version")
        rollings.batch = self.tensor_fn.cut_to_effective_len(
            rollings.batch,
            keys=['input_ids', 'attention_mask', 'position_ids']
        )
        gen_output = self._generate_with_gpu_padding(rollings)
        final_str = self.tokenizer.batch_decode(gen_output.batch["responses"], skip_special_tokens=True)
        final_ids = self._batch_tokenize(final_str)
        camera_ready_lengths = self._text_token_lengths(final_str)
        camera_ready_label = 2 * num_revision_rounds + 1
        original_right_side = self._append_right_side_segment(
            original_right_side, final_ids, trainable=True,
            segment_label=camera_ready_label,
        )

        meta_info = dict(getattr(gen_output, 'meta_info', {}) or {})
        meta_info['num_revision_rounds'] = num_revision_rounds
        meta_info['camera_ready_texts'] = final_str
        meta_info['paper_writing_draft_valid_flags'] = draft_valid_flags_by_round
        meta_info['paper_writing_draft_lengths'] = draft_lengths_by_round
        meta_info['paper_writing_camera_ready_lengths'] = camera_ready_lengths
        meta_info['paper_writing_ground_truth_lengths'] = ground_truth_lengths
        meta_info['paper_writing_draft_texts'] = draft_texts_by_round
        meta_info['paper_writing_comment_texts'] = comments_by_round
        meta_info['paper_writing_comment_obs_texts'] = comment_obs_by_round
        meta_info['paper_writing_ground_truth_texts'] = ground_truths
        meta_info['paper_writing_domains'] = domains
        meta_info['paper_writing_topics'] = topics
        meta_info['trace_mode'] = 'per_segment'
        print(f"[Paper Writing/PerSegment] Completed {num_revision_rounds} revision rounds\n")
        os.environ.get('PW_DEBUG') and __import__('pdb').set_trace()
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def run_llm_loop_paper_writing_train_commenter(self, gen_batch, initial_input_ids: torch.Tensor,
                                                    num_revision_rounds: int = None):
        """
        多轮写作循环（Train-Commenter 版本）。

        目标：
        1. 将“写 draft”的角色切换为 API generator；
        2. 本地 actor 只负责生成 `<comment>...</comment>`；
        3. 训练目标从 draft 切到 comment（只训 commenter）。

        训练 mask 策略：
        - API 生成的 draft：mask（不参与梯度）
        - 本地模型生成的 comment：不 mask（参与梯度）
        - 控制指令/轮次提示：mask（不参与梯度）

        说明：
        - 为减少配置改动，API generator 复用 commenter 的 API 配置
          (`commenter_api_key/base_url/model`)。
        """
        if num_revision_rounds is None:
            num_revision_rounds = self.config.num_revision_rounds

        print(f"\n[Paper Writing/TrainCommenter] Starting {num_revision_rounds}-round revision process")
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            'responses': initial_input_ids[:, []],
            'responses_with_info_mask': initial_input_ids[:, []]
        }
        rollings = gen_batch

        domains, topics = self._extract_domain_topic_from_batch(gen_batch)
        _ = self._extract_ground_truth_from_batch(gen_batch)
        last_local_meta = {}

        for round_idx in range(num_revision_rounds):
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            rolling_texts = self.tokenizer.batch_decode(rollings.batch["input_ids"], skip_special_tokens=True)

            # Step 1: API 生成 draft（该段不参与训练）。
            draft_texts = self._call_generator_batch(
                contexts=rolling_texts,
                domains=domains,
                topics=topics
            )
            normalized_drafts = []
            for text in draft_texts:
                if self._is_strict_single_draft(text):
                    normalized_drafts.append(text)
                else:
                    normalized_drafts.append("<draft>Please revise the abstract to satisfy formatting requirements.</draft>")
            draft_ids = self._batch_tokenize(normalized_drafts)

            # Step 2: 本地模型生成 comment（该段参与训练）。
            comment_instruction = [
                "\n\nPlease provide a concise reviewer comment for the previous draft. "
                "Output exactly one <comment>...</comment> block.\n\n"
                for _ in normalized_drafts
            ]
            comment_instruction_ids = self._process_next_obs(comment_instruction)
            commenter_rollings = self._update_rolling_state(rollings, draft_ids, comment_instruction_ids)
            commenter_rollings.batch = self.tensor_fn.cut_to_effective_len(
                commenter_rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            comment_output = self._generate_with_gpu_padding(commenter_rollings)
            last_local_meta = dict(getattr(comment_output, 'meta_info', {}) or {})
            raw_comment_texts = self.tokenizer.batch_decode(comment_output.batch["responses"], skip_special_tokens=True)
            comment_blocks = []
            for text in raw_comment_texts:
                match = re.search(r'<comment>(.*?)</comment>', text, re.DOTALL)
                if match:
                    content = match.group(1).strip()
                else:
                    content = text.strip()
                if not content:
                    content = "Please improve clarity, method detail, and result specificity."
                comment_blocks.append(f"<comment>{content}</comment>")
            comment_ids = self._process_next_obs([f"\n\n{c}\n\n" for c in comment_blocks])

            # Step 3: 组装下一轮观察（comment + round instruction），交给 API 继续写 draft。
            next_round = round_idx + 2
            round_instruction = self._make_round_instruction(next_round)
            instruction_ids = self._process_next_obs([f"{round_instruction}\n\n" for _ in comment_blocks])
            combined_obs_ids = self._process_next_obs([
                f"\n\n{comment_blocks[i]}\n\n{round_instruction}\n\n" for i in range(len(comment_blocks))
            ])
            rollings = self._update_rolling_state(rollings, draft_ids, combined_obs_ids)

            # 右侧按“可训练性”追加：
            # draft -> mask；comment -> train；instruction -> mask。
            original_right_side = self._append_right_side_segment(original_right_side, draft_ids, trainable=False)
            original_right_side = self._append_right_side_segment(original_right_side, comment_ids, trainable=True)
            original_right_side = self._append_right_side_segment(original_right_side, instruction_ids, trainable=False)

        # 最终稿由 API 生成，仅用于 reward/logging，在训练序列中保持 mask。
        rollings.batch = self.tensor_fn.cut_to_effective_len(
            rollings.batch,
            keys=['input_ids', 'attention_mask', 'position_ids']
        )
        final_contexts = self.tokenizer.batch_decode(rollings.batch["input_ids"], skip_special_tokens=True)
        final_str = self._call_generator_batch(contexts=final_contexts, domains=domains, topics=topics)
        final_ids = self._batch_tokenize(final_str)
        original_right_side = self._append_right_side_segment(original_right_side, final_ids, trainable=False)

        meta_info = dict(last_local_meta)
        meta_info['num_revision_rounds'] = num_revision_rounds
        meta_info['camera_ready_texts'] = final_str
        meta_info['trace_mode'] = 'train_commenter'
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

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
        base_output = self.run_llm_loop_paper_writing(
            gen_batch=gen_batch,
            initial_input_ids=initial_input_ids,
            num_revision_rounds=num_revision_rounds
        )
        camera_ready_texts = base_output.meta_info.get('camera_ready_texts', [])
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

        final_output meta_info:
          - camera_ready_texts: List[str]  (供 reward 使用)
          - trace_mode: 'autonomous'
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

            # Record camera-ready texts for samples that just finished this step
            for i, content in enumerate(camera_ready_contents):
                if content and not camera_ready_texts[i]:
                    camera_ready_texts[i] = content

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

            # Record raw output as camera_ready for still-active samples (evaluator handles format check)
            for i, (active, text) in enumerate(zip(active_mask.tolist(), responses_str)):
                if active:
                    camera_ready_texts[i] = text

            original_right_side = self._update_right_side(
                original_right_side, responses_ids
            )

        meta_info = dict(meta_info)
        meta_info['camera_ready_texts'] = camera_ready_texts
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_comment_stats'] = valid_comment_stats.tolist()
        meta_info['trace_mode'] = 'autonomous'

        print(f"[Paper Writing/Autonomous] ACTIVE_TRAJ_NUM: {active_num_list}")
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

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
                base_url=self.config.commenter_base_url
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
                base_url=self.config.commenter_base_url
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
            "CRITICAL LENGTH CONSTRAINT:\n"
            "- Your feedback MUST be 50-80 words (approximately 70-120 tokens)\n"
            "- Be concise and focus on the most important issues only\n"
            "- Prioritize: 1) Clarity issues, 2) Missing key elements, 3) Structural problems\n\n"
            "IMPORTANT OUTPUT FORMAT:\n"
            "- Output ONLY plain text revision suggestions\n"
            "- Do NOT use any XML tags (like <comment>, <draft>, <camera-ready>, etc.)\n"
            "- Do NOT repeat or include the draft content in your response\n"
            "- Provide 2-3 specific, actionable suggestions directly"
        )

        pairs = list(zip(domains, topics, drafts, ground_truths))
        if not pairs:
            return []

        max_workers = min(256, len(pairs))
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
