from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from d2f_vllm.engine.model_runner import _config_dtype
from d2f_vllm.fastdllm_engine import (
    FastDLLMDreamEngine,
    _StaticMaskSeq,
)
from d2f_vllm.multimodal.lladao_gui import (
    LLaDAOGuiImageSpan,
    LLaDAOGuiPrefix,
    LLaDAOGuiPrefixEncoder,
)
from d2f_vllm.utils.context import (
    reset_context_diffusion_lm,
    set_context_diffusion_lm,
)


@dataclass(frozen=True)
class LLaDAOGuiKVCompressionConfig:
    enabled: bool = False
    vision_tile_size: int = 16
    vision_topk_tiles: int = 20
    vision_token_keep_ratio: float = 0.75
    vision_score_query_window: int = 32
    vision_score_layers: int = 4
    vision_score_layer_mode: str = "last"
    vision_score_pool_kernel: int = 7

    def __post_init__(self) -> None:
        if self.vision_tile_size <= 0:
            raise ValueError("vision_tile_size must be positive")
        if self.vision_topk_tiles < 0:
            raise ValueError("vision_topk_tiles must be non-negative")
        if not 0.0 < self.vision_token_keep_ratio <= 1.0:
            raise ValueError("vision_token_keep_ratio must be in (0, 1]")
        if self.vision_score_query_window < 0:
            raise ValueError("vision_score_query_window must be non-negative")
        if self.vision_score_layers < 0:
            raise ValueError("vision_score_layers must be non-negative")
        if self.vision_score_layer_mode not in {"all", "first", "last"}:
            raise ValueError(
                "vision_score_layer_mode must be one of: all, first, last"
            )
        if (
            self.vision_score_pool_kernel <= 0
            or self.vision_score_pool_kernel % 2 == 0
        ):
            raise ValueError("vision_score_pool_kernel must be a positive odd integer")


@dataclass
class LLaDAOGuiEngineOutput:
    text: str
    token_ids: list[int]
    n_diff_steps: int
    image_tokens: int
    prompt_tokens: int
    dense_prefix_tokens: int
    cached_prefix_tokens: int
    kv_cache_compression_ratio: float
    kv_cache_compression_seconds: float
    vision_tiles: int
    vision_selected_tiles: int
    input_images: int
    source_width: int
    source_height: int
    image_seconds: float
    prompt_seconds: float
    generation_seconds: float
    total_seconds: float
    peak_memory_allocated_gib: float
    peak_memory_reserved_gib: float
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_vision_tiles(
    grid_height: int,
    grid_width: int,
    tile_size: int,
) -> list[list[int]]:
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("vision grid dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    tiles: dict[tuple[int, int], list[int]] = {}
    for row in range(grid_height):
        for column in range(grid_width):
            tile = (row // tile_size, column // tile_size)
            tiles.setdefault(tile, []).append(row * grid_width + column)
    return [tiles[key] for key in sorted(tiles)]


def select_top_vision_tiles(
    patch_scores: torch.Tensor,
    tiles: Sequence[Sequence[int]],
    topk_tiles: int,
) -> list[int]:
    if patch_scores.ndim != 1:
        raise ValueError("patch_scores must be one-dimensional")
    if not tiles:
        return []
    if topk_tiles <= 0 or topk_tiles >= len(tiles):
        return list(range(len(tiles)))
    tile_scores = torch.stack(
        [
            patch_scores.index_select(
                0,
                torch.as_tensor(
                    tile,
                    dtype=torch.long,
                    device=patch_scores.device,
                ),
            ).amax()
            for tile in tiles
        ]
    )
    selected = torch.topk(tile_scores, k=topk_tiles, largest=True).indices
    return sorted(int(index) for index in selected.tolist())


def select_patch_tokens_per_head(
    scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    keep_count: int,
) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [num_kv_heads, num_patches]")
    candidates = candidate_indices.to(device=scores.device, dtype=torch.long)
    if candidates.ndim != 1 or candidates.numel() == 0:
        raise ValueError("candidate_indices must be a non-empty vector")
    count = min(max(1, int(keep_count)), int(candidates.numel()))
    if count == int(candidates.numel()):
        return candidates.sort().values.unsqueeze(0).expand(scores.shape[0], -1)
    candidate_scores = scores.index_select(1, candidates)
    selected_local = torch.topk(
        candidate_scores, k=count, dim=1, largest=True
    ).indices
    selected = candidates[selected_local]
    return selected.sort(dim=1).values


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
        kv_compression: LLaDAOGuiKVCompressionConfig | None = None,
        kv_cache_capacity: int | None = None,
        rope_scaling: dict | None = None,
        allow_unscaled_max_model_len: bool = False,
    ) -> None:
        if max_new_tokens <= 0 or max_new_tokens % block_length:
            raise ValueError("max_new_tokens must be a positive block multiple")
        self.max_new_tokens = int(max_new_tokens)
        self.block_add_threshold = float(block_add_threshold)
        self.decoded_token_threshold = float(decoded_token_threshold)
        self.skip_threshold = float(skip_threshold)
        self.kv_compression = (
            kv_compression
            if kv_compression is not None
            else LLaDAOGuiKVCompressionConfig()
        )
        self.kv_cache_capacity = int(kv_cache_capacity or max_model_len)
        if self.kv_cache_capacity <= 0:
            raise ValueError("kv_cache_capacity must be positive")
        if self.kv_cache_capacity > max_model_len:
            raise ValueError(
                "kv_cache_capacity cannot exceed max_model_len"
            )
        page_count = math.ceil(self.kv_cache_capacity / 256) + 4
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
            rope_scaling_override=rope_scaling,
            allow_unscaled_max_model_len=allow_unscaled_max_model_len,
        )
        if self.config.tensor_parallel_size != 1:
            raise ValueError("LLaDA-o GUI Non-PD currently supports TP=1 only")
        try:
            self.prefix_encoder = LLaDAOGuiPrefixEncoder(
                model,
                self.model.model.embed_tokens,
                device=torch.device("cuda", torch.cuda.current_device()),
                dtype=_config_dtype(self.config.hf_config),
            )
            self.tokenizer = self.prefix_encoder.tokenizer
        except BaseException:
            self.close()
            raise

    def _compression_reduces_context(self) -> bool:
        config = self.kv_compression
        return config.enabled and (
            config.vision_topk_tiles > 0
            or config.vision_token_keep_ratio < 1.0
        )

    def _prefix_keys(
        self,
        page_ids: list[int],
        prefix_len: int,
        layer_index: int,
    ) -> torch.Tensor:
        device = torch.device("cuda", torch.cuda.current_device())
        token_indices = torch.arange(prefix_len, dtype=torch.long, device=device)
        page_table = torch.tensor(page_ids, dtype=torch.long, device=device)
        pages = page_table.index_select(0, token_indices // self.page_size)
        offsets = token_indices % self.page_size
        return self.runner.kv_cache[0, layer_index, pages, offsets, :, :]

    def _pool_vision_scores(
        self,
        scores: torch.Tensor,
        span: LLaDAOGuiImageSpan,
    ) -> torch.Tensor:
        expected = span.grid_height * span.grid_width
        if scores.shape[-1] != expected:
            raise ValueError(
                f"vision score length mismatch: got {scores.shape[-1]}, "
                f"expected {expected}"
            )
        kernel = self.kv_compression.vision_score_pool_kernel
        if kernel == 1:
            return scores
        leading = scores.shape[:-1]
        pooled = F.max_pool2d(
            scores.reshape(
                -1,
                1,
                span.grid_height,
                span.grid_width,
            ),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )
        return pooled.reshape(*leading, expected)

    def _score_vision_tokens(
        self,
        prefix: LLaDAOGuiPrefix,
        page_ids: list[int],
        query_capture: dict[int, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        config = self.kv_compression
        num_layers = len(self.model.model.layers)
        layer_indices = self._select_attention_layer_indices(
            num_layers,
            config.vision_score_layers,
            config.vision_score_layer_mode,
        )
        if not layer_indices:
            raise RuntimeError("KV compression did not select any scoring layers")

        prompt_len = len(prefix.prompt_ids)
        query_end = max(1, prompt_len - 1)
        query_start = 1
        if query_end <= query_start:
            query_start, query_end = 0, prompt_len
        if config.vision_score_query_window > 0:
            query_start = max(
                query_start,
                query_end - config.vision_score_query_window,
            )
        scores_by_layer: dict[int, torch.Tensor] = {}

        for layer_index in layer_indices:
            query = query_capture.get(layer_index)
            if query is None:
                raise RuntimeError(
                    f"missing captured prompt query for layer {layer_index}"
                )
            if query.shape[0] != prompt_len:
                raise ValueError(
                    f"captured query length mismatch at layer {layer_index}: "
                    f"got {query.shape[0]}, expected {prompt_len}"
                )
            layer = self.model.model.layers[layer_index]
            attention = layer.self_attn
            query = query[query_start:query_end]
            keys = self._prefix_keys(page_ids, prefix.length, layer_index)
            num_query_heads = int(query.shape[1])
            num_kv_heads = int(keys.shape[1])
            if num_query_heads % num_kv_heads:
                raise ValueError(
                    f"query heads {num_query_heads} are not divisible by "
                    f"KV heads {num_kv_heads}"
                )
            group_size = num_query_heads // num_kv_heads
            grouped_query = (
                query.float()
                .reshape(query.shape[0], num_kv_heads, group_size, query.shape[2])
                .permute(1, 0, 2, 3)
            )
            grouped_keys = keys.float().permute(1, 2, 0)
            logits = torch.einsum(
                "hqgd,hds->hqgs", grouped_query, grouped_keys
            )
            probabilities = torch.softmax(
                logits * float(attention.scaling), dim=-1
            )
            span_scores = []
            for span in prefix.image_spans:
                scores = (
                    probabilities[..., span.patch_start : span.patch_end]
                    .sum(dim=1)
                    .mean(dim=1)
                )
                span_scores.append(self._pool_vision_scores(scores, span))
            scores_by_layer[layer_index] = torch.cat(
                span_scores, dim=-1
            )
        return scores_by_layer

    def _build_vision_keep_indices(
        self,
        prefix: LLaDAOGuiPrefix,
        scores_by_layer: dict[int, torch.Tensor],
    ) -> tuple[list[torch.Tensor], dict[str, int | float]]:
        config = self.kv_compression
        patch_count = prefix.image_patch_count
        if patch_count != len(prefix.image_ids) - 2 * len(prefix.image_spans):
            raise ValueError(
                "vision grid does not match the number of image patch tokens"
            )
        scored_layers = sorted(scores_by_layer)
        if not scored_layers:
            raise RuntimeError("vision token scoring produced no layer scores")
        aggregate_scores = torch.stack(
            [scores_by_layer[index] for index in scored_layers]
        ).mean(dim=(0, 1))
        span_plans: list[tuple[int, int, int, torch.Tensor, int]] = []
        score_offset = 0
        total_tiles = 0
        total_selected_tiles = 0
        total_candidates = 0
        total_kept = 0
        for span in prefix.image_spans:
            span_count = span.patch_count
            span_scores = aggregate_scores[
                score_offset : score_offset + span_count
            ]
            tiles = build_vision_tiles(
                span.grid_height,
                span.grid_width,
                config.vision_tile_size,
            )
            selected_tiles = select_top_vision_tiles(
                span_scores,
                tiles,
                config.vision_topk_tiles,
            )
            candidates = torch.tensor(
                sorted(
                    patch_index
                    for tile_index in selected_tiles
                    for patch_index in tiles[tile_index]
                ),
                dtype=torch.long,
                device=aggregate_scores.device,
            )
            requested_keep = max(
                1,
                math.ceil(
                    span_count * config.vision_token_keep_ratio
                ),
            )
            keep_count = min(requested_keep, int(candidates.numel()))
            span_plans.append(
                (
                    score_offset,
                    span_count,
                    span.patch_start,
                    candidates,
                    keep_count,
                )
            )
            score_offset += span_count
            total_tiles += len(tiles)
            total_selected_tiles += len(selected_tiles)
            total_candidates += int(candidates.numel())
            total_kept += keep_count
        if score_offset != patch_count:
            raise ValueError("vision score spans do not cover all patches")

        prompt_indices = torch.arange(
            len(prefix.image_ids),
            prefix.length,
            dtype=torch.long,
            device=aggregate_scores.device,
        )
        boundary_indices = torch.tensor(
            [
                index
                for span in prefix.image_spans
                for index in (span.token_start, span.token_end - 1)
            ],
            dtype=torch.long,
            device=aggregate_scores.device,
        )
        num_layers = len(self.model.model.layers)
        keep_indices: list[torch.Tensor] = []
        for layer_index in range(num_layers):
            score_layer = min(
                scored_layers,
                key=lambda selected: abs(selected - layer_index),
            )
            per_span_keeps = []
            for (
                score_start,
                span_count,
                token_start,
                candidates,
                keep_count,
            ) in span_plans:
                local_scores = scores_by_layer[score_layer][
                    :, score_start : score_start + span_count
                ]
                selected = select_patch_tokens_per_head(
                    local_scores,
                    candidates,
                    keep_count,
                )
                per_span_keeps.append(selected + token_start)
            patch_keep = torch.cat(per_span_keeps, dim=1)
            layer_keep = []
            for head_index in range(patch_keep.shape[0]):
                layer_keep.append(
                    torch.cat(
                        (
                            boundary_indices,
                            patch_keep[head_index],
                            prompt_indices,
                        )
                    ).sort().values
                )
            keep_indices.append(torch.stack(layer_keep, dim=0).contiguous())

        cached_prefix_tokens = (
            total_kept
            + 2 * len(prefix.image_spans)
            + len(prefix.prompt_ids)
        )
        return keep_indices, {
            "dense_prefix_tokens": prefix.length,
            "cached_prefix_tokens": cached_prefix_tokens,
            "vision_patches": patch_count,
            "vision_kept_patches": total_kept,
            "vision_tiles": total_tiles,
            "vision_selected_tiles": total_selected_tiles,
            "candidate_patches": total_candidates,
            "compression_ratio": cached_prefix_tokens / prefix.length,
        }

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
            need_kv_cache_store=False,
        )

    def _forward_image_prefix(
        self,
        prefix: LLaDAOGuiPrefix,
        page_ids: list[int],
    ) -> None:
        # Each full-page tile is an independent visual document.  Running
        # bidirectional prefill per tile preserves that native multi-image
        # boundary and avoids quadratic attention across unrelated tiles.
        # The subsequent text prompt attends all stored tile KV in one request.
        for span in prefix.image_spans:
            start, end = span.token_start, span.token_end
            embeddings = prefix.image_embeddings[start:end]
            positions = prefix.image_positions[start:end]
            slot_mapping = self._range_slot_mapping(
                page_ids, start, end - start
            )
            self._set_full_prefill_context(end - start, slot_mapping)
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
    def diagnose_absolute_positions(
        self,
        prompt: str,
        offsets: Sequence[int],
    ) -> list[dict[str, Any]]:
        """Run a short identical text at several absolute RoPE offsets."""

        ids = [
            self.prefix_encoder.bos_token_id,
            *self.tokenizer.encode(prompt, add_special_tokens=False),
            self.prefix_encoder.eos_token_id,
        ]
        if len(ids) > self.kv_cache_capacity:
            raise ValueError("diagnostic prompt exceeds kv_cache_capacity")
        page_ids = self._prefix_cache.allocate_pages(
            math.ceil(len(ids) / self.page_size)
        )
        logits_by_offset: list[tuple[int, torch.Tensor]] = []
        try:
            for raw_offset in offsets:
                offset = int(raw_offset)
                if offset < 0 or offset + len(ids) > self.config.max_model_len:
                    raise ValueError(
                        f"diagnostic offset {offset} with {len(ids)} tokens "
                        f"exceeds max_model_len={self.config.max_model_len}"
                    )
                slot_mapping = self._range_slot_mapping(
                    page_ids, 0, len(ids)
                )
                self._set_full_prefill_context(len(ids), slot_mapping)
                try:
                    hidden = self.model(
                        self._ids_tensor(ids),
                        self._positions_tensor(
                            range(offset, offset + len(ids))
                        ),
                    )
                    logits = self.model.compute_logits(hidden[-1:]).float()[0]
                finally:
                    reset_context_diffusion_lm()
                logits_by_offset.append((offset, logits.cpu()))
        finally:
            self._prefix_cache.release_pages(page_ids)

        reference = logits_by_offset[0][1]
        results: list[dict[str, Any]] = []
        for offset, logits in logits_by_offset:
            probabilities = torch.softmax(logits, dim=-1)
            top = torch.topk(probabilities, k=5)
            results.append(
                {
                    "offset": offset,
                    "finite": bool(torch.isfinite(logits).all()),
                    "top_token_ids": [
                        int(value) for value in top.indices.tolist()
                    ],
                    "top_probabilities": [
                        float(value) for value in top.values.tolist()
                    ],
                    "entropy": float(
                        -(probabilities * probabilities.clamp_min(1e-30).log())
                        .sum()
                        .item()
                    ),
                    "cosine_to_offset_0": float(
                        F.cosine_similarity(
                            logits.unsqueeze(0),
                            reference.unsqueeze(0),
                        ).item()
                    ),
                }
            )
        return results

    @torch.inference_mode()
    def generate_gui(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        max_iterations: int = 256,
        full_page: bool = False,
        full_page_tile_size: int = 980,
    ) -> LLaDAOGuiEngineOutput:
        total_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        max_new_tokens = int(max_new_tokens or self.max_new_tokens)
        if max_new_tokens <= 0 or max_new_tokens % self.block_length:
            raise ValueError("max_new_tokens must be a positive block multiple")
        prefix_started = time.perf_counter()
        prefix = (
            self.prefix_encoder.encode_full_page(
                image,
                prompt,
                tile_size=full_page_tile_size,
            )
            if full_page
            else self.prefix_encoder.encode(image, prompt)
        )
        torch.cuda.synchronize()
        full_length = prefix.length + max_new_tokens
        if full_length > self.config.max_model_len:
            raise ValueError(
                f"image+prompt+generation length {full_length} exceeds "
                f"max_model_len={self.config.max_model_len}"
            )
        if full_length > self.kv_cache_capacity:
            raise ValueError(
                f"image+prompt+generation length {full_length} exceeds "
                f"kv_cache_capacity={self.kv_cache_capacity}"
            )
        pages_needed = math.ceil(full_length / self.page_size)
        page_ids = self._prefix_cache.allocate_pages(pages_needed)
        try:
            self._forward_image_prefix(prefix, page_ids)
            torch.cuda.synchronize()
            image_cached = time.perf_counter()
            query_capture = {} if self._compression_reduces_context() else None
            self._forward_append_tokens_paged(
                prefix.prompt_ids,
                prefix.prompt_positions,
                context_len=len(prefix.image_ids),
                all_page_ids=page_ids,
                start_token=len(prefix.image_ids),
                query_capture=query_capture,
            )
            torch.cuda.synchronize()
            dense_prompt_cached = time.perf_counter()

            tiles = [
                tile
                for span in prefix.image_spans
                for tile in build_vision_tiles(
                    span.grid_height,
                    span.grid_width,
                    self.kv_compression.vision_tile_size,
                )
            ]
            compression_stats: dict[str, int | float] = {
                "dense_prefix_tokens": prefix.length,
                "cached_prefix_tokens": prefix.length,
                "vision_patches": prefix.image_patch_count,
                "vision_kept_patches": prefix.image_patch_count,
                "vision_tiles": len(tiles),
                "vision_selected_tiles": len(tiles),
                "candidate_patches": prefix.image_patch_count,
                "compression_ratio": 1.0,
            }
            cached_prefix_len = prefix.length
            if query_capture is not None:
                scores_by_layer = self._score_vision_tokens(
                    prefix,
                    page_ids,
                    query_capture,
                )
                keep_indices, compression_stats = self._build_vision_keep_indices(
                    prefix,
                    scores_by_layer,
                )
                cached_prefix_len = int(
                    compression_stats["cached_prefix_tokens"]
                )
                if cached_prefix_len < prefix.length:
                    compacted = self._compact_prompt_cache_per_layer_per_head(
                        page_ids,
                        keep_indices,
                    )
                    if compacted != cached_prefix_len:
                        raise RuntimeError(
                            f"KV compaction length mismatch: got {compacted}, "
                            f"expected {cached_prefix_len}"
                        )
                    active_pages = math.ceil(
                        (cached_prefix_len + max_new_tokens) / self.page_size
                    )
                    if active_pages < len(page_ids):
                        self._prefix_cache.release_pages(page_ids[active_pages:])
                        page_ids = page_ids[:active_pages]
                query_capture.clear()
            torch.cuda.synchronize()
            prompt_cached = time.perf_counter()
            compression_seconds = prompt_cached - dense_prompt_cached

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
                    cache_length = cached_prefix_len + start
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
                cache_length = cached_prefix_len + active_start
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
            peak_allocated = torch.cuda.max_memory_allocated() / 2**30
            peak_reserved = torch.cuda.max_memory_reserved() / 2**30
            return LLaDAOGuiEngineOutput(
                text=text,
                token_ids=output.tolist(),
                n_diff_steps=len(trace),
                image_tokens=len(prefix.image_ids),
                prompt_tokens=len(prefix.prompt_ids),
                dense_prefix_tokens=prefix.length,
                cached_prefix_tokens=cached_prefix_len,
                kv_cache_compression_ratio=float(
                    compression_stats["compression_ratio"]
                ),
                kv_cache_compression_seconds=compression_seconds,
                vision_tiles=int(compression_stats["vision_tiles"]),
                vision_selected_tiles=int(
                    compression_stats["vision_selected_tiles"]
                ),
                input_images=len(prefix.image_spans),
                source_width=prefix.source_width,
                source_height=prefix.source_height,
                image_seconds=image_cached - prefix_started,
                prompt_seconds=prompt_cached - image_cached,
                generation_seconds=generation_finished - generation_started,
                total_seconds=generation_finished - total_started,
                peak_memory_allocated_gib=peak_allocated,
                peak_memory_reserved_gib=peak_reserved,
                trace=trace,
            )
        finally:
            self._prefix_cache.release_pages(page_ids)
