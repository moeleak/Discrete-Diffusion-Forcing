from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from d2f_vllm.engine.model_runner import _config_dtype
from d2f_vllm.fastdllm_engine import (
    FastDLLMDreamEngine,
    _StaticMaskSeq,
)
from d2f_vllm.multimodal.lladao_gui import LLaDAOGuiPrefixEncoder
from d2f_vllm.utils.context import (
    reset_context_diffusion_lm,
    set_context_diffusion_lm,
)


@dataclass
class LLaDAOGuiEngineOutput:
    text: str
    token_ids: list[int]
    n_diff_steps: int
    image_tokens: int
    prompt_tokens: int
    image_seconds: float
    prompt_seconds: float
    generation_seconds: float
    total_seconds: float
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_generation_attention_mask(
    context_len: int,
    active_len: int,
    block_length: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if active_len <= 0 or active_len % block_length:
        raise ValueError("active_len must be a positive multiple of block_length")
    mask = torch.zeros(
        (active_len, context_len + active_len), dtype=torch.bool, device=device
    )
    mask[:, :context_len] = True
    for start in range(0, active_len, block_length):
        end = start + block_length
        mask[start:end, context_len : context_len + end] = True
    return mask


class LLaDAOGuiD2FEngine(FastDLLMDreamEngine):
    """Native d2f_vllm Non-PD engine for the GUI-grounding LLaDA-o model."""

    def __init__(
        self,
        model: str | Path,
        *,
        max_model_len: int = 16384,
        block_length: int = 16,
        max_new_tokens: int = 64,
        mask_token_id: int = 126336,
        block_add_threshold: float = 0.1,
        decoded_token_threshold: float = 0.95,
        skip_threshold: float = 0.9,
        temperature: float = 0.0,
        gpu_memory_utilization: float = 0.75,
        master_port: int = 2333,
    ) -> None:
        if max_new_tokens <= 0 or max_new_tokens % block_length:
            raise ValueError("max_new_tokens must be a positive block multiple")
        self.max_new_tokens = int(max_new_tokens)
        self.block_add_threshold = float(block_add_threshold)
        self.decoded_token_threshold = float(decoded_token_threshold)
        self.skip_threshold = float(skip_threshold)
        page_count = math.ceil(max_model_len / 256) + 4
        super().__init__(
            str(model),
            max_model_len=max_model_len,
            block_length=block_length,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_model_len,
            max_num_seqs=1,
            mask_token_id=mask_token_id,
            threshold=skip_threshold,
            temperature=temperature,
            enforce_eager=True,
            kv_cache_layout="unified",
            master_port=master_port,
            model_name="lladao_gui",
            num_kvcache_blocks=page_count,
            skip_model_warmup=True,
        )
        if self.config.tensor_parallel_size != 1:
            raise ValueError("LLaDA-o GUI Non-PD currently supports TP=1 only")
        self.prefix_encoder = LLaDAOGuiPrefixEncoder(
            model,
            self.model.model.embed_tokens,
            device=torch.device("cuda", torch.cuda.current_device()),
            dtype=_config_dtype(self.config.hf_config),
        )

    def _set_active_context(
        self,
        *,
        context_len: int,
        active_len: int,
        start_token: int,
        page_ids: list[int],
    ) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        slot_mapping = self._range_slot_mapping(page_ids, start_token, active_len)
        block_tables = torch.tensor(
            page_ids, dtype=torch.int32, device=device
        ).view(1, -1)
        mask = build_generation_attention_mask(
            context_len,
            active_len,
            self.block_length,
            device=device,
        )
        seq = _StaticMaskSeq(mask, self.block_length)
        set_context_diffusion_lm(
            False,
            cu_seqlens_q=torch.tensor(
                [0, active_len], dtype=torch.int32, device=device
            ),
            cu_seqlens_k=torch.tensor(
                [0, context_len + active_len], dtype=torch.int32, device=device
            ),
            max_seqlen_q=active_len,
            max_seqlen_k=context_len + active_len,
            slot_mapping=slot_mapping,
            context_lens=torch.tensor([context_len], dtype=torch.int32, device=device),
            block_tables=block_tables,
            seqs=[seq],
            seq_lens=[active_len],
            seq_lens_ts=torch.tensor(
                [active_len], dtype=torch.int32, device=device
            ),
            kv_cache_layout="unified",
            need_kv_cache_store=True,
        )

    def _forward_image_prefix(
        self,
        embeddings: torch.Tensor,
        positions: list[int],
        page_ids: list[int],
    ) -> None:
        slot_mapping = self._range_slot_mapping(page_ids, 0, embeddings.size(0))
        self._set_full_prefill_context(embeddings.size(0), slot_mapping)
        try:
            self.model(
                None,
                self._positions_tensor(positions),
                input_embeds=embeddings,
            )
        finally:
            reset_context_diffusion_lm()

    def _forward_active(
        self,
        token_ids: torch.Tensor,
        positions: list[int],
        *,
        context_len: int,
        start_token: int,
        page_ids: list[int],
    ) -> torch.Tensor:
        active_len = int(token_ids.numel())
        self._set_active_context(
            context_len=context_len,
            active_len=active_len,
            start_token=start_token,
            page_ids=page_ids,
        )
        try:
            hidden = self.model(
                token_ids.reshape(-1), self._positions_tensor(positions)
            )
            return self.model.compute_logits(hidden)
        finally:
            reset_context_diffusion_lm()

    @torch.inference_mode()
    def generate_gui(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        max_iterations: int = 256,
    ) -> LLaDAOGuiEngineOutput:
        total_started = time.perf_counter()
        max_new_tokens = int(max_new_tokens or self.max_new_tokens)
        if max_new_tokens <= 0 or max_new_tokens % self.block_length:
            raise ValueError("max_new_tokens must be a positive block multiple")
        prefix_started = time.perf_counter()
        prefix = self.prefix_encoder.encode(image, prompt)
        torch.cuda.synchronize()
        full_length = prefix.length + max_new_tokens
        if full_length > self.config.max_model_len:
            raise ValueError(
                f"image+prompt+generation length {full_length} exceeds "
                f"max_model_len={self.config.max_model_len}"
            )
        pages_needed = math.ceil(full_length / self.page_size)
        page_ids = self._prefix_cache.allocate_pages(pages_needed)
        try:
            self._forward_image_prefix(
                prefix.image_embeddings, prefix.image_positions, page_ids
            )
            torch.cuda.synchronize()
            image_cached = time.perf_counter()
            self._forward_append_tokens_paged(
                prefix.prompt_ids,
                prefix.prompt_positions,
                context_len=len(prefix.image_ids),
                all_page_ids=page_ids,
                start_token=len(prefix.image_ids),
            )
            torch.cuda.synchronize()
            prompt_cached = time.perf_counter()

            device = torch.device("cuda", torch.cuda.current_device())
            tokens = torch.full(
                (max_new_tokens,),
                self.mask_token_id,
                dtype=torch.long,
                device=device,
            )
            tokens[0] = self.prefix_encoder.bos_token_id
            eos_token_id = self.prefix_encoder.eos_token_id
            rope_start = prefix.prompt_positions[-1] + 1
            max_blocks = max_new_tokens // self.block_length
            states: list[dict[str, int]] = []
            blocks_added = 0
            blocks_cached = 0
            eos_position: int | None = None
            trace: list[dict[str, Any]] = []
            generation_started = time.perf_counter()

            for iteration in range(1, max_iterations + 1):
                if blocks_added == 0:
                    states.append(
                        {
                            "masks": self.block_length - 1,
                            "total": self.block_length - 1,
                        }
                    )
                    blocks_added = 1
                elif blocks_added < max_blocks and eos_position is None:
                    previous = states[blocks_added - 1]
                    progress = 1.0 - previous["masks"] / max(
                        previous["total"], 1
                    )
                    if progress >= self.block_add_threshold:
                        states.append(
                            {
                                "masks": self.block_length,
                                "total": self.block_length,
                            }
                        )
                        blocks_added += 1

                while (
                    blocks_cached < blocks_added
                    and states[blocks_cached]["masks"] == 0
                ):
                    start = blocks_cached * self.block_length
                    end = start + self.block_length
                    cache_length = prefix.length + start
                    self._forward_append_tokens_paged(
                        tokens[start:end].tolist(),
                        list(range(rope_start + start, rope_start + end)),
                        context_len=cache_length,
                        all_page_ids=page_ids,
                        start_token=cache_length,
                    )
                    blocks_cached += 1

                required_end = eos_position + 1 if eos_position is not None else None
                if required_end is not None and not bool(
                    (tokens[:required_end] == self.mask_token_id).any()
                ):
                    break
                if blocks_cached == blocks_added:
                    if blocks_added == max_blocks or eos_position is not None:
                        break
                    continue

                active_start = blocks_cached * self.block_length
                active_end = blocks_added * self.block_length
                cache_length = prefix.length + active_start
                logits = self._forward_active(
                    tokens[active_start:active_end],
                    list(range(rope_start + active_start, rope_start + active_end)),
                    context_len=cache_length,
                    start_token=cache_length,
                    page_ids=page_ids,
                )

                ready_blocks = {blocks_cached}
                for block_id in range(blocks_cached + 1, blocks_added):
                    previous = states[block_id - 1]
                    progress = 1.0 - previous["masks"] / max(
                        previous["total"], 1
                    )
                    if progress >= self.decoded_token_threshold:
                        ready_blocks.add(block_id)

                updates = 0
                for block_id in range(blocks_cached, blocks_added):
                    start = block_id * self.block_length
                    end = start + self.block_length
                    local_start = start - active_start
                    block_tokens = tokens[start:end]
                    masked = (block_tokens == self.mask_token_id).nonzero(
                        as_tuple=True
                    )[0]
                    if masked.numel() == 0:
                        states[block_id]["masks"] = 0
                        continue
                    block_logits = logits[local_start + masked].float()
                    if self.temperature > 0:
                        probabilities = F.softmax(
                            block_logits / self.temperature, dim=-1
                        )
                        predictions = torch.multinomial(probabilities, 1).squeeze(-1)
                        confidence = probabilities.gather(
                            1, predictions[:, None]
                        ).squeeze(1)
                    else:
                        probabilities = F.softmax(block_logits, dim=-1)
                        confidence, predictions = probabilities.max(dim=-1)
                    selected = (confidence >= self.skip_threshold).nonzero(
                        as_tuple=True
                    )[0]
                    if selected.numel() == 0 and block_id in ready_blocks:
                        selected = confidence.argmax().reshape(1)
                    if selected.numel() == 0:
                        continue
                    absolute = start + masked[selected]
                    chosen = predictions[selected]
                    tokens[absolute] = chosen
                    states[block_id]["masks"] -= int(selected.numel())
                    updates += int(selected.numel())
                    eos_hits = absolute[chosen == eos_token_id]
                    if eos_hits.numel():
                        candidate = int(eos_hits.min().item())
                        eos_position = (
                            candidate
                            if eos_position is None
                            else min(eos_position, candidate)
                        )

                trace.append(
                    {
                        "iteration": iteration,
                        "blocks_added": blocks_added,
                        "blocks_cached": blocks_cached,
                        "updates": updates,
                        "remaining_masks": int(
                            (
                                tokens[: blocks_added * self.block_length]
                                == self.mask_token_id
                            )
                            .sum()
                            .item()
                        ),
                    }
                )
                if updates == 0:
                    raise RuntimeError("D2F scheduler made no progress")
            else:
                raise RuntimeError(
                    f"D2F generation exceeded {max_iterations} iterations"
                )

            torch.cuda.synchronize()
            generation_finished = time.perf_counter()
            output_end = (
                eos_position + 1
                if eos_position is not None
                else blocks_added * self.block_length
            )
            output = tokens[:output_end]
            if bool((output == self.mask_token_id).any()):
                raise RuntimeError("D2F generation stopped with unresolved masks")
            text = self.tokenizer.decode(
                output.tolist(), skip_special_tokens=False
            )
            return LLaDAOGuiEngineOutput(
                text=text,
                token_ids=output.tolist(),
                n_diff_steps=len(trace),
                image_tokens=len(prefix.image_ids),
                prompt_tokens=len(prefix.prompt_ids),
                image_seconds=image_cached - prefix_started,
                prompt_seconds=prompt_cached - image_cached,
                generation_seconds=generation_finished - generation_started,
                total_seconds=generation_finished - total_started,
                trace=trace,
            )
        finally:
            self._prefix_cache.release_pages(page_ids)
