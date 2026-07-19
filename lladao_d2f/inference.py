from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .masking import build_suffix_attention_bias
from .modeling import (
    adapter_disabled,
    add_lladao_repo,
    load_adapter,
    load_base_model,
    unwrap_lladao,
)


def _sync(device: torch.device | str | None = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


class LLaDAOGuiD2FInference:
    def __init__(
        self,
        *,
        lladao_repo: str | Path,
        model_path: str | Path,
        checkpoint: str | Path,
        adapter_path: str | Path | None = None,
        device: str = "cuda:0",
    ):
        repo = add_lladao_repo(lladao_repo)
        from data.transforms import ImageTransform

        base, self.tokenizer, self.special_tokens = load_base_model(
            repo, model_path, checkpoint
        )
        self.model = (
            load_adapter(base, adapter_path) if adapter_path is not None else base
        ).to(device).eval()
        self.base = unwrap_lladao(self.model)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.image_transform = ImageTransform(980, 378, 14, max_pixels=2_007_040)
        self.mask_id = int(
            self.special_tokens.get(
                "mask_token_id", self.tokenizer.mask_token_id or 126336
            )
        )

    @torch.inference_mode()
    def generate_baseline(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        diffusion_steps: int = 64,
        confidence_threshold: float | None = 0.95,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        with adapter_disabled(self.model):
            cache, cache_length, rope_start, image_seconds, prompt_seconds = self._prefix_cache(
                image, prompt
            )
            bos_id = int(self.special_tokens["bos_token_id"])
            eos_id = int(self.special_tokens["eos_token_id"])
            tokens = torch.full(
                (max_new_tokens,), self.mask_id, dtype=torch.long, device=self.device
            )
            tokens[0] = bos_id
            stats: dict[str, Any] = {}
            _sync(self.device)
            generation_started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = self.base._generate_with_full_cache(
                    past_key_values=cache,
                    cached_kvlens=[cache_length],
                    initial_sequence=tokens,
                    gen_position_ids=torch.arange(
                        rope_start, rope_start + max_new_tokens, device=self.device
                    ),
                    gen_text_indexes=torch.arange(max_new_tokens, device=self.device),
                    sample_lens=[max_new_tokens],
                    bos_token_id=bos_id,
                    eos_token_id=eos_id,
                    steps=diffusion_steps,
                    block_length=max_new_tokens,
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    mask_id=self.mask_id,
                    confidence_threshold=confidence_threshold,
                    generation_stats=stats,
                )[0]
        _sync(self.device)
        generation_seconds = time.perf_counter() - generation_started
        eos = (output == eos_id).nonzero(as_tuple=True)[0]
        if eos.numel():
            output = output[: int(eos[0].item()) + 1]
        raw_text = self.tokenizer.decode(output, skip_special_tokens=False)
        total_seconds = time.perf_counter() - total_started
        return {
            "raw_text": raw_text,
            "tokens": output.detach().cpu().tolist(),
            "image_cache_seconds": image_seconds,
            "prompt_cache_seconds": prompt_seconds,
            "generation_seconds": generation_seconds,
            "total_seconds": total_seconds,
            "iterations": sum(item["iterations"] for item in stats.get("blocks", [])),
            "trace": stats,
        }

    @torch.inference_mode()
    def _prefix_cache(self, image: Image.Image, prompt: str):
        from modeling.lladao.llada_navit import NaiveCache

        cache = NaiveCache(self.base.config.llm_config.num_hidden_layers)
        lengths = [0]
        rope = [0]
        _sync(self.device)
        image_started = time.perf_counter()
        generation_input, lengths, rope = self.base.prepare_vit_images(
            lengths,
            rope,
            [image.convert("RGB")],
            self.image_transform,
            self.special_tokens,
        )
        generation_input = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in generation_input.items()
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cache = self.base.forward_cache_update_vit(cache, **generation_input)
        _sync(self.device)
        image_seconds = time.perf_counter() - image_started

        prompt_started = time.perf_counter()
        generation_input, lengths, rope = self.base.prepare_prompts(
            lengths, rope, [prompt], self.tokenizer, self.special_tokens
        )
        generation_input = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in generation_input.items()
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cache = self.base.forward_cache_update_text(cache, **generation_input)
        _sync(self.device)
        prompt_seconds = time.perf_counter() - prompt_started
        return cache, lengths[0], rope[0], image_seconds, prompt_seconds

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        block_size: int = 16,
        block_add_threshold: float = 0.1,
        decoded_token_threshold: float = 0.95,
        skip_threshold: float = 0.9,
        temperature: float = 0.0,
        max_iterations: int = 256,
    ) -> dict[str, Any]:
        if max_new_tokens <= 0 or max_new_tokens % block_size:
            raise ValueError("max_new_tokens must be a positive multiple of block_size")
        total_started = time.perf_counter()
        cache, cache_length, rope_start, image_seconds, prompt_seconds = self._prefix_cache(
            image, prompt
        )
        eos_id = int(self.special_tokens["eos_token_id"])
        bos_id = int(self.special_tokens["bos_token_id"])
        tokens = torch.full(
            (max_new_tokens,), self.mask_id, dtype=torch.long, device=self.device
        )
        tokens[0] = bos_id
        max_blocks = max_new_tokens // block_size
        states: list[dict[str, Any]] = []
        blocks_added = 0
        blocks_cached = 0
        eos_position: int | None = None
        trace: list[dict[str, Any]] = []

        _sync(self.device)
        generation_started = time.perf_counter()
        for iteration in range(1, max_iterations + 1):
            if blocks_added == 0:
                states.append({"masks": block_size - 1, "total": block_size - 1})
                blocks_added = 1
            elif blocks_added < max_blocks and eos_position is None:
                previous = states[blocks_added - 1]
                progress = 1.0 - previous["masks"] / max(previous["total"], 1)
                if progress >= block_add_threshold:
                    states.append({"masks": block_size, "total": block_size})
                    blocks_added += 1

            while blocks_cached < blocks_added and states[blocks_cached]["masks"] == 0:
                start = blocks_cached * block_size
                end = start + block_size
                packed_key_indexes = torch.arange(cache_length, device=self.device)
                packed_text_indexes = torch.arange(
                    cache_length, cache_length + block_size, device=self.device
                )
                cache_input = {
                    "text_token_lens": torch.tensor([block_size], dtype=torch.int32, device=self.device),
                    "packed_text_ids": tokens[start:end],
                    "packed_text_position_ids": torch.arange(
                        rope_start + start, rope_start + end, device=self.device
                    ),
                    "packed_text_indexes": packed_text_indexes,
                    "packed_key_value_indexes": packed_key_indexes,
                    "key_values_lens": torch.tensor([cache_length], dtype=torch.int32, device=self.device),
                }
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cache = self.base.forward_cache_update_text(cache, **cache_input)
                cache_length += block_size
                blocks_cached += 1

            required_end = eos_position + 1 if eos_position is not None else None
            if required_end is not None and not bool((tokens[:required_end] == self.mask_id).any()):
                break
            if blocks_cached == blocks_added:
                if blocks_added == max_blocks or eos_position is not None:
                    break
                continue

            active_start = blocks_cached * block_size
            active_end = blocks_added * block_size
            active_ids = tokens[active_start:active_end]
            active_length = active_end - active_start
            attention_bias = build_suffix_attention_bias(
                cache_length,
                active_length,
                block_size,
                device=self.device,
                dtype=torch.bfloat16,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.base.forward_for_generation_with_cache(
                    past_key_values=cache,
                    sequence_length=active_length,
                    packed_text_ids=active_ids,
                    packed_text_indexes=torch.arange(active_length, device=self.device),
                    sample_lens=[active_length],
                    packed_position_ids=torch.arange(
                        rope_start + active_start,
                        rope_start + active_end,
                        device=self.device,
                    ),
                    packed_key_value_indexes=torch.arange(cache_length, device=self.device),
                    key_values_lens=torch.tensor([cache_length], dtype=torch.int32, device=self.device),
                    update_cache=False,
                    attention_bias=attention_bias,
                )

            # Match the official scheduler: block-completion eligibility is
            # computed before this iteration's token updates, rather than
            # allowing readiness to cascade through several blocks at once.
            ready_blocks = {blocks_cached}
            for block_id in range(blocks_cached + 1, blocks_added):
                previous = states[block_id - 1]
                previous_progress = 1.0 - previous["masks"] / max(
                    previous["total"], 1
                )
                if previous_progress >= decoded_token_threshold:
                    ready_blocks.add(block_id)

            updates = 0
            for block_id in range(blocks_cached, blocks_added):
                start = block_id * block_size
                end = start + block_size
                local_start = start - active_start
                block_tokens = tokens[start:end]
                masked_locations = (block_tokens == self.mask_id).nonzero(as_tuple=True)[0]
                if masked_locations.numel() == 0:
                    states[block_id]["masks"] = 0
                    continue
                block_logits = logits[local_start + masked_locations]
                if temperature > 0:
                    probabilities = torch.softmax(block_logits.float() / temperature, dim=-1)
                    predictions = torch.multinomial(probabilities, 1).squeeze(-1)
                    confidence = probabilities.gather(1, predictions[:, None]).squeeze(1)
                else:
                    probabilities = torch.softmax(block_logits.float(), dim=-1)
                    confidence, predictions = probabilities.max(dim=-1)
                selected = (confidence >= skip_threshold).nonzero(as_tuple=True)[0]
                previous_ready = block_id in ready_blocks
                if selected.numel() == 0 and previous_ready:
                    selected = confidence.argmax().reshape(1)
                if selected.numel() == 0:
                    continue
                absolute = start + masked_locations[selected]
                chosen = predictions[selected]
                tokens[absolute] = chosen
                states[block_id]["masks"] -= int(selected.numel())
                updates += int(selected.numel())
                eos_hits = absolute[chosen == eos_id]
                if eos_hits.numel():
                    candidate = int(eos_hits.min().item())
                    eos_position = candidate if eos_position is None else min(eos_position, candidate)

            trace.append(
                {
                    "iteration": iteration,
                    "blocks_added": blocks_added,
                    "blocks_cached": blocks_cached,
                    "updates": updates,
                    "remaining_masks": int((tokens[: blocks_added * block_size] == self.mask_id).sum().item()),
                }
            )
            if updates == 0:
                raise RuntimeError("D2F scheduler made no progress")
        else:
            raise RuntimeError(f"D2F generation exceeded {max_iterations} iterations")

        _sync(self.device)
        generation_seconds = time.perf_counter() - generation_started
        output_end = eos_position + 1 if eos_position is not None else blocks_added * block_size
        output_tokens = tokens[:output_end]
        if bool((output_tokens == self.mask_id).any()):
            raise RuntimeError("D2F generation stopped with masks before EOS")
        raw_text = self.tokenizer.decode(output_tokens, skip_special_tokens=False)
        _sync(self.device)
        total_seconds = time.perf_counter() - total_started
        return {
            "raw_text": raw_text,
            "tokens": output_tokens.detach().cpu().tolist(),
            "image_cache_seconds": image_seconds,
            "prompt_cache_seconds": prompt_seconds,
            "generation_seconds": generation_seconds,
            "total_seconds": total_seconds,
            "iterations": len(trace),
            "trace": trace,
        }
