#!/usr/bin/env python3
"""Run sharded LLaDA-o GUI-grounding inference with baseline or D2F decoding."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lladao_d2f.inference import LLaDAOGuiD2FInference
from lladao_d2f.modeling import add_lladao_repo
from lladao_d2f.residual_grounding import (
    ResidualGroundingContractError,
    load_adapter_contract,
    sha256_file,
    validate_sha256,
)


DEFAULT_BENCHMARKS = "mind2web"


def optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "off", "fixed"}:
        return None
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be in [0, 1] or 'none'")
    return result


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("baseline", "d2f", "d2f_vllm"), required=True
    )
    parser.add_argument("--lladao-repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        help="bind evaluation to one exact full Planner checkpoint",
    )
    parser.add_argument(
        "--require-residual-adapter-contract",
        action="store_true",
        help=(
            "require a release-eligible residual-grounder epoch bound to the "
            "exact checkpoint SHA; intended for release benchmarks"
        ),
    )
    parser.add_argument("--runtime-model", type=Path)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    parser.add_argument(
        "--rank",
        type=int,
        default=env_int("RANK", env_int("SLURM_PROCID", 0)),
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=env_int("WORLD_SIZE", env_int("SLURM_NTASKS", 1)),
    )
    parser.add_argument("--device", help="default: cuda:<rank modulo GPU count>")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=optional_float, default=0.95)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--block-add-threshold", type=float, default=0.1)
    parser.add_argument("--decoded-token-threshold", type=float, default=0.95)
    parser.add_argument("--skip-threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-iterations", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--kv-cache-capacity", type=int)
    parser.add_argument(
        "--rope-scaling", choices=("none", "yarn"), default="none"
    )
    parser.add_argument("--rope-factor", type=float, default=8.0)
    parser.add_argument(
        "--original-max-position-embeddings",
        type=int,
        default=16384,
    )
    parser.add_argument("--allow-unscaled-max-model-len", action="store_true")
    parser.add_argument(
        "--full-page-tiles",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "override the prepared sample input protocol; omit to follow the "
            "manifest, pass --no-full-page-tiles for native single-image resize"
        ),
    )
    parser.add_argument("--full-page-tile-size", type=int, default=980)
    parser.add_argument(
        "--full-page-overview",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "append a checkpoint-native whole-page resize after all exact "
            "full-resolution tiles as a global coordinate anchor"
        ),
    )
    parser.add_argument(
        "--truncate-full-page-tiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "drop trailing complete full-resolution tiles until prompt and "
            "generation fit the resident context capacity"
        ),
    )
    parser.add_argument(
        "--full-page-position-mode",
        choices=("native", "strided", "sequential"),
        default="native",
        help=(
            "native shares one small LLM RoPE position per image; strided "
            "keeps a shared position within each image at its dense token "
            "offset; sequential gives every visual token an absolute position"
        ),
    )
    parser.add_argument("--master-port", type=int, default=2333)
    parser.add_argument(
        "--attention-backend", choices=("sdpa", "flex"), default="sdpa"
    )
    parser.add_argument(
        "--rms-norm-backend", choices=("torch", "vllm"), default="torch"
    )
    parser.add_argument(
        "--kv-cache-compression",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--kv-cache-retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "retrieve complete image KV spans by prompt self-information; "
            "selected spans retain every token, layer, and KV head"
        ),
    )
    parser.add_argument("--kv-retrieval-topk-images", type=int, default=4)
    parser.add_argument(
        "--kv-retrieval-score-mode",
        choices=(
            "masked_self_information",
            "cached_masked_self_information",
            "causal_masked_self_information",
            "causal_self_information",
        ),
        default="masked_self_information",
        help=(
            "whole-image retrieval scorer; cached_masked_self_information "
            "keeps bidirectional query attention while reusing visual KV, "
            "causal_masked_self_information is the one-way control, and "
            "causal_self_information is the retired clear-query legacy proxy"
        ),
    )
    parser.add_argument(
        "--kv-retrieval-mask-rounds",
        type=int,
        default=2,
        help=(
            "number of complementary masked-query rounds used by "
            "bidirectional dLLM retrieval scoring"
        ),
    )
    parser.add_argument(
        "--kv-retrieval-packed-scoring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "batch independent masked candidate/round documents with "
            "FlashAttention varlen; disable for sequential equivalence tests"
        ),
    )
    parser.add_argument(
        "--kv-retrieval-max-batch-tokens",
        type=int,
        default=65_536,
        help="soft token budget for each packed masked-scoring forward",
    )
    parser.add_argument(
        "--kv-retrieval-keep-overview",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--vision-tile-size", type=int, default=16)
    parser.add_argument("--vision-topk-tiles", type=int, default=20)
    parser.add_argument("--vision-token-keep-ratio", type=float, default=0.75)
    parser.add_argument("--vision-score-query-window", type=int, default=32)
    parser.add_argument("--vision-score-layers", type=int, default=4)
    parser.add_argument(
        "--vision-score-layer-mode",
        choices=("all", "first", "last"),
        default="last",
    )
    parser.add_argument("--vision-score-pool-kernel", type=int, default=7)
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.rank < 0 or args.world_size <= 0 or args.rank >= args.world_size:
        parser.error("rank must satisfy 0 <= rank < world-size")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.warmup < 0 or args.flush_every <= 0:
        parser.error("--warmup must be non-negative and --flush-every must be positive")
    if args.max_new_tokens <= 0 or args.diffusion_steps <= 0:
        parser.error("generation length and diffusion steps must be positive")
    if args.backend in {"d2f", "d2f_vllm"} and args.max_new_tokens % args.block_size:
        parser.error("D2F max-new-tokens must be divisible by block-size")
    if args.backend == "d2f_vllm" and args.runtime_model is None:
        parser.error("--runtime-model is required for d2f_vllm")
    if args.require_residual_adapter_contract:
        if args.adapter is None or args.expected_checkpoint_sha256 is None:
            parser.error(
                "--require-residual-adapter-contract requires --adapter and "
                "--expected-checkpoint-sha256"
            )
        if args.backend == "d2f_vllm":
            parser.error(
                "a converted runtime cannot prove a separately loaded adapter; "
                "use baseline/d2f for the residual release gate"
            )
        if args.limit is None or args.limit > 100:
            parser.error("the residual release gate requires --limit in [1, 100]")
    if args.kv_cache_capacity is not None:
        if args.kv_cache_capacity <= 0:
            parser.error("--kv-cache-capacity must be positive")
        if args.kv_cache_capacity > args.max_model_len:
            parser.error(
                "--kv-cache-capacity cannot exceed --max-model-len"
            )
    if args.rope_scaling == "yarn" and args.rope_factor <= 1.0:
        parser.error("--rope-factor must be greater than 1 for YaRN")
    if (
        args.max_model_len > args.original_max_position_embeddings
        and args.rope_scaling == "none"
        and not args.allow_unscaled_max_model_len
        and args.backend == "d2f_vllm"
    ):
        parser.error(
            "an extended unscaled run requires "
            "--allow-unscaled-max-model-len"
        )
    if args.full_page_overview and args.truncate_full_page_tiles:
        parser.error(
            "--full-page-overview and --truncate-full-page-tiles "
            "are mutually exclusive"
        )
    if args.kv_cache_retrieval and args.kv_cache_compression:
        parser.error(
            "--kv-cache-retrieval and --kv-cache-compression are mutually "
            "exclusive"
        )
    if args.kv_retrieval_topk_images < 0:
        parser.error("--kv-retrieval-topk-images must be non-negative")
    if args.kv_retrieval_mask_rounds <= 0:
        parser.error("--kv-retrieval-mask-rounds must be positive")
    if args.kv_retrieval_max_batch_tokens <= 0:
        parser.error("--kv-retrieval-max-batch-tokens must be positive")
    for name in (
        "block_add_threshold",
        "decoded_token_threshold",
        "skip_threshold",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_response_text(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.replace("<|endoftext|>", "").strip()


def native_resize_prompt(sample: dict[str, Any]) -> str:
    """Return the checkpoint-native prompt for a prepared full-page sample."""

    native_prompt = sample.get("native_prompt")
    if isinstance(native_prompt, str) and native_prompt.strip():
        return native_prompt
    prompt = str(sample["prompt"])
    layout = sample.get("tile_layout")
    tile_count = len(layout) if isinstance(layout, list) else None
    width = sample.get("image_width")
    height = sample.get("image_height")
    if not all(isinstance(value, int) for value in (tile_count, width, height)):
        raise ValueError(
            "native resize of a full-page sample requires native_prompt or "
            "integer tile/image metadata"
        )
    prefix = (
        f"The following {tile_count} images are non-overlapping tiles from one "
        f"{width}x{height} webpage screenshot, ordered left-to-right and then "
        "top-to-bottom. Treat them as one complete page. "
    )
    suffix = (
        " Return the action and bounding box with coordinates normalized to "
        "the complete original screenshot in [0,1000]."
    )
    if not prompt.startswith(prefix) or not prompt.endswith(suffix):
        raise ValueError(
            "cannot recover the checkpoint-native prompt from the full-page "
            "benchmark wrapper"
        )
    return prompt[len(prefix) : -len(suffix)]


def full_page_tile_count(sample: dict[str, Any], tile_size: int) -> int:
    """Return the number of runtime full-page tiles for a prepared sample."""

    width = sample.get("image_width")
    height = sample.get("image_height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("full-page input requires integer image dimensions")
    if width <= 0 or height <= 0:
        raise ValueError("full-page image dimensions must be positive")
    if not 1 <= tile_size <= 980:
        raise ValueError("full-page tile size must be in [1, 980]")
    return math.ceil(width / tile_size) * math.ceil(height / tile_size)


def full_page_grounding_prompt(
    sample: dict[str, Any],
    tile_size: int = 980,
) -> str:
    """Describe the exact runtime tiles rather than the prepared 980px layout."""

    width = sample.get("image_width")
    height = sample.get("image_height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("full-page input requires integer image dimensions")
    tile_count = full_page_tile_count(sample, tile_size)
    instruction = native_resize_prompt(sample)
    return (
        f"The following {tile_count} images are non-overlapping tiles from one "
        f"{width}x{height} webpage screenshot, ordered left-to-right and then "
        "top-to-bottom. Treat them as one complete page. "
        f"{instruction} Return the action and bounding box with coordinates "
        "normalized to the complete original screenshot in [0,1000]."
    )


def resolve_sample_input(
    sample: dict[str, Any],
    full_page_override: bool | None,
    full_page_tile_size: int = 980,
) -> tuple[bool, str, str]:
    """Resolve the actual image preprocessing and prompt used for inference."""

    prepared_full_page = sample.get("input_protocol") == "full_page_tiles"
    full_page = (
        prepared_full_page
        if full_page_override is None
        else full_page_override
    )
    if full_page:
        prompt = (
            full_page_grounding_prompt(sample, full_page_tile_size)
            if prepared_full_page
            else str(sample["prompt"])
        )
        return True, prompt, "full_page_tiles"
    prompt = (
        native_resize_prompt(sample)
        if prepared_full_page
        else str(sample["prompt"])
    )
    return False, prompt, "native_resize"


def overview_grounding_prompt(
    sample: dict[str, Any],
    tile_size: int = 980,
) -> str:
    """Describe exact tiles plus the final whole-page overview without a crop."""

    width = sample.get("image_width")
    height = sample.get("image_height")
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
    ):
        raise ValueError(
            "full-page overview requires integer image dimensions"
        )
    tile_count = full_page_tile_count(sample, tile_size)
    instruction = native_resize_prompt(sample)
    return (
        f"The first {tile_count} images are exact non-overlapping tiles from "
        f"one {width}x{height} webpage screenshot, ordered left-to-right and "
        "then top-to-bottom. The final image is a resized overview of that "
        "same complete screenshot. Treat all images as one page and use the "
        "final overview as the global coordinate reference. "
        f"{instruction} Return the action and bounding box with coordinates "
        "normalized to the complete original screenshot in [0,1000]."
    )


def select_device(args: argparse.Namespace) -> str:
    if args.device:
        return args.device
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("LLaDA-o inference requires CUDA")
    return f"cuda:{args.rank % count}"


def selected_benchmarks(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> list[str]:
    requested = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    available = manifest.get("benchmarks", {})
    missing = [item for item in requested if item not in available]
    if missing:
        print(
            "Skipping unavailable benchmarks: " + ", ".join(missing),
            file=sys.stderr,
            flush=True,
        )
    selected = [item for item in requested if item in available]
    if not selected:
        raise RuntimeError("none of the requested benchmarks is prepared")
    return selected


def iter_samples(
    root: Path,
    manifest: dict[str, Any],
    benchmark: str,
    *,
    rank: int,
    world_size: int,
    limit: int | None,
) -> Iterator[dict[str, Any]]:
    path = root / manifest["benchmarks"][benchmark]["path"]
    with path.open(encoding="utf-8") as handle:
        logical_index = 0
        for line in handle:
            if not line.strip():
                continue
            if limit is not None and logical_index >= limit:
                break
            if logical_index % world_size == rank:
                yield json.loads(line)
            logical_index += 1


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                completed.add(str(json.loads(line)["sample_id"]))
            except (json.JSONDecodeError, KeyError) as exc:
                raise RuntimeError(
                    f"cannot resume malformed {path}:{line_number}: {exc}"
                ) from exc
    return completed


def load_protocol(lladao_repo: Path):
    add_lladao_repo(lladao_repo)
    from eval.gui_grounding.metrics import parse_action
    from eval.gui_grounding.reproducibility import paired_sample_seed

    return parse_action, paired_sample_seed, iter_samples, load_completed, selected_benchmarks


def model_generate(
    engine,
    image: Image.Image,
    prompt: str,
    args: argparse.Namespace,
    *,
    full_page: bool = False,
    retrieval_query: str | None = None,
) -> dict[str, Any]:
    if args.backend == "baseline":
        return engine.generate_baseline(
            image,
            prompt,
            max_new_tokens=args.max_new_tokens,
            diffusion_steps=args.diffusion_steps,
            confidence_threshold=args.confidence_threshold,
        )
    if args.backend == "d2f":
        return engine.generate(
            image,
            prompt,
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            block_add_threshold=args.block_add_threshold,
            decoded_token_threshold=args.decoded_token_threshold,
            skip_threshold=args.skip_threshold,
            temperature=args.temperature,
            max_iterations=args.max_iterations,
        )
    output = engine.generate_gui(
        image,
        prompt,
        max_new_tokens=args.max_new_tokens,
        max_iterations=args.max_iterations,
        full_page=full_page,
        full_page_tile_size=args.full_page_tile_size,
        full_page_position_mode=args.full_page_position_mode,
        full_page_overview=args.full_page_overview,
        truncate_full_page_tiles=args.truncate_full_page_tiles,
        retrieval_query=retrieval_query,
    )
    return {
        "raw_text": output.text,
        "tokens": output.token_ids,
        "image_cache_seconds": output.image_seconds,
        "prompt_cache_seconds": output.prompt_seconds,
        "generation_seconds": output.generation_seconds,
        "total_seconds": output.total_seconds,
        "dense_prefix_tokens": output.dense_prefix_tokens,
        "cached_prefix_tokens": output.cached_prefix_tokens,
        "kv_cache_compression_ratio": output.kv_cache_compression_ratio,
        "kv_cache_compression_seconds": output.kv_cache_compression_seconds,
        "kv_cache_retrieval_enabled": output.kv_cache_retrieval_enabled,
        "kv_cache_retrieval_candidates": (
            output.kv_cache_retrieval_candidates
        ),
        "kv_cache_retrieval_selected": output.kv_cache_retrieval_selected,
        "kv_cache_retrieval_indices": output.kv_cache_retrieval_indices,
        "kv_cache_retrieval_scores": output.kv_cache_retrieval_scores,
        "kv_cache_retrieval_query": output.kv_cache_retrieval_query,
        "kv_cache_retrieval_query_tokens": (
            output.kv_cache_retrieval_query_tokens
        ),
        "kv_cache_retrieval_score_mode": (
            output.kv_cache_retrieval_score_mode
        ),
        "kv_cache_retrieval_mask_rounds": (
            output.kv_cache_retrieval_mask_rounds
        ),
        "kv_cache_retrieval_packed_scoring": (
            output.kv_cache_retrieval_packed_scoring
        ),
        "kv_cache_retrieval_score_batches": (
            output.kv_cache_retrieval_score_batches
        ),
        "kv_cache_retrieval_max_batch_tokens": (
            output.kv_cache_retrieval_max_batch_tokens
        ),
        "kv_cache_retrieval_ratio": output.kv_cache_retrieval_ratio,
        "kv_cache_retrieval_seconds": output.kv_cache_retrieval_seconds,
        "vision_tiles": output.vision_tiles,
        "vision_selected_tiles": output.vision_selected_tiles,
        "input_images": output.input_images,
        "source_images": output.source_images,
        "truncated_images": output.truncated_images,
        "source_width": output.source_width,
        "source_height": output.source_height,
        "peak_memory_allocated_gib": output.peak_memory_allocated_gib,
        "peak_memory_reserved_gib": output.peak_memory_reserved_gib,
        "position_mode": output.position_mode,
        "max_prefill_position": output.max_prefill_position,
        "max_generation_position": output.max_generation_position,
        "iterations": output.n_diff_steps,
        "trace": output.trace,
    }


def infer_one(
    engine,
    root: Path,
    sample: dict[str, Any],
    args: argparse.Namespace,
    parse_action,
    paired_sample_seed,
) -> dict[str, Any]:
    inference_seed = paired_sample_seed(sample, args.seed)
    set_seed(inference_seed)
    full_page, prompt, runtime_input_protocol = resolve_sample_input(
        sample,
        args.full_page_tiles,
        args.full_page_tile_size,
    )
    if full_page and args.full_page_overview:
        prompt = overview_grounding_prompt(sample, args.full_page_tile_size)
        runtime_input_protocol = "full_page_tiles_with_overview"
    elif full_page and args.truncate_full_page_tiles:
        runtime_input_protocol = "full_page_tiles_truncated"
    retrieval_query = (
        native_resize_prompt(sample)
        if args.backend == "d2f_vllm" and args.kv_cache_retrieval
        else None
    )
    if args.backend == "d2f_vllm" and full_page:
        sequence = sample.get("sequence_tokens")
        expected_total = (
            sequence.get("total") if isinstance(sequence, dict) else None
        )
        capacity = args.kv_cache_capacity or args.max_model_len
        if (
            not args.truncate_full_page_tiles
            and not args.kv_cache_retrieval
            and isinstance(expected_total, (int, float))
            and expected_total > capacity
        ):
            raise ValueError(
                f"prepared sequence length {int(expected_total)} exceeds "
                f"kv_cache_capacity={capacity}"
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with Image.open(root / sample["image"]) as source:
        image = source.convert("RGB")
        result = model_generate(
            engine,
            image,
            prompt,
            args,
            full_page=full_page,
            retrieval_query=retrieval_query,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - started
    prediction = clean_response_text(result["raw_text"])
    parsed = parse_action(prediction)
    return {
        "sample_id": sample["sample_id"],
        "benchmark": sample["benchmark"],
        "split": sample["split"],
        "backend": args.backend,
        "prediction": prediction,
        "raw_prediction": result["raw_text"],
        "predicted_action": parsed.action,
        "predicted_bbox_1000": list(parsed.bbox_1000) if parsed.bbox_1000 else None,
        "predicted_value": parsed.value,
        "parse_error": parsed.error,
        "target_action": sample["target_action"],
        "target_bbox_1000": sample["target_bbox_1000"],
        "target_value": sample.get("target_value", ""),
        "runtime_input_protocol": runtime_input_protocol,
        "latency_seconds": latency,
        "model_elapsed_seconds": result["total_seconds"],
        "image_cache_seconds": result["image_cache_seconds"],
        "prompt_cache_seconds": result["prompt_cache_seconds"],
        "generation_seconds": result["generation_seconds"],
        "dense_prefix_tokens": result.get("dense_prefix_tokens"),
        "cached_prefix_tokens": result.get("cached_prefix_tokens"),
        "kv_cache_compression_ratio": result.get("kv_cache_compression_ratio"),
        "kv_cache_compression_seconds": result.get(
            "kv_cache_compression_seconds"
        ),
        "kv_cache_retrieval_enabled": result.get(
            "kv_cache_retrieval_enabled"
        ),
        "kv_cache_retrieval_candidates": result.get(
            "kv_cache_retrieval_candidates"
        ),
        "kv_cache_retrieval_selected": result.get(
            "kv_cache_retrieval_selected"
        ),
        "kv_cache_retrieval_indices": result.get(
            "kv_cache_retrieval_indices"
        ),
        "kv_cache_retrieval_scores": result.get(
            "kv_cache_retrieval_scores"
        ),
        "kv_cache_retrieval_query": result.get(
            "kv_cache_retrieval_query"
        ),
        "kv_cache_retrieval_query_tokens": result.get(
            "kv_cache_retrieval_query_tokens"
        ),
        "kv_cache_retrieval_score_mode": result.get(
            "kv_cache_retrieval_score_mode"
        ),
        "kv_cache_retrieval_mask_rounds": result.get(
            "kv_cache_retrieval_mask_rounds"
        ),
        "kv_cache_retrieval_packed_scoring": result.get(
            "kv_cache_retrieval_packed_scoring"
        ),
        "kv_cache_retrieval_score_batches": result.get(
            "kv_cache_retrieval_score_batches"
        ),
        "kv_cache_retrieval_max_batch_tokens": result.get(
            "kv_cache_retrieval_max_batch_tokens"
        ),
        "kv_cache_retrieval_ratio": result.get(
            "kv_cache_retrieval_ratio"
        ),
        "kv_cache_retrieval_seconds": result.get(
            "kv_cache_retrieval_seconds"
        ),
        "vision_tiles": result.get("vision_tiles"),
        "vision_selected_tiles": result.get("vision_selected_tiles"),
        "input_images": result.get("input_images"),
        "source_images": result.get("source_images"),
        "truncated_images": result.get("truncated_images"),
        "source_width": result.get("source_width"),
        "source_height": result.get("source_height"),
        "peak_memory_allocated_gib": result.get(
            "peak_memory_allocated_gib"
        ),
        "peak_memory_reserved_gib": result.get(
            "peak_memory_reserved_gib"
        ),
        "position_mode": result.get("position_mode"),
        "max_prefill_position": result.get("max_prefill_position"),
        "max_generation_position": result.get(
            "max_generation_position"
        ),
        "convergence_steps": result["iterations"],
        "valid_tokens": len(result["tokens"]),
        "generated_tokens": args.max_new_tokens,
        "runtime_sequence_tokens": (
            result.get("dense_prefix_tokens", 0) + args.max_new_tokens
            if isinstance(result.get("dense_prefix_tokens"), int)
            else None
        ),
        "generation_stats": result["trace"],
        "inference_seed": inference_seed,
        "error": None,
    }


def error_record(sample, args, paired_sample_seed, exc: BaseException) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "benchmark": sample["benchmark"],
        "split": sample["split"],
        "backend": args.backend,
        "prediction": "",
        "raw_prediction": "",
        "predicted_action": None,
        "predicted_bbox_1000": None,
        "predicted_value": "",
        "parse_error": "inference_error",
        "target_action": sample["target_action"],
        "target_bbox_1000": sample["target_bbox_1000"],
        "target_value": sample.get("target_value", ""),
        "runtime_input_protocol": None,
        "latency_seconds": None,
        "model_elapsed_seconds": None,
        "image_cache_seconds": None,
        "prompt_cache_seconds": None,
        "generation_seconds": None,
        "dense_prefix_tokens": None,
        "cached_prefix_tokens": None,
        "kv_cache_compression_ratio": None,
        "kv_cache_compression_seconds": None,
        "kv_cache_retrieval_enabled": args.kv_cache_retrieval,
        "kv_cache_retrieval_candidates": None,
        "kv_cache_retrieval_selected": None,
        "kv_cache_retrieval_indices": None,
        "kv_cache_retrieval_scores": None,
        "kv_cache_retrieval_query": None,
        "kv_cache_retrieval_query_tokens": None,
        "kv_cache_retrieval_score_mode": None,
        "kv_cache_retrieval_mask_rounds": None,
        "kv_cache_retrieval_packed_scoring": None,
        "kv_cache_retrieval_score_batches": None,
        "kv_cache_retrieval_max_batch_tokens": None,
        "kv_cache_retrieval_ratio": None,
        "kv_cache_retrieval_seconds": None,
        "vision_tiles": None,
        "vision_selected_tiles": None,
        "input_images": None,
        "source_images": None,
        "truncated_images": None,
        "source_width": None,
        "source_height": None,
        "peak_memory_allocated_gib": None,
        "peak_memory_reserved_gib": None,
        "position_mode": args.full_page_position_mode,
        "max_prefill_position": None,
        "max_generation_position": None,
        "convergence_steps": None,
        "valid_tokens": None,
        "generated_tokens": None,
        "runtime_sequence_tokens": None,
        "generation_stats": None,
        "inference_seed": paired_sample_seed(sample, args.seed),
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(limit=20),
    }


def run_config(args: argparse.Namespace, benchmarks: list[str], device: str) -> dict[str, Any]:
    return {
        "backend": args.backend,
        "lladao_repo": str(args.lladao_repo.expanduser().resolve()),
        "model_path": str(args.model_path.expanduser().resolve()),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_sha256": getattr(args, "checkpoint_sha256", None),
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256,
        "adapter": str(args.adapter.expanduser().resolve()) if args.adapter else None,
        "residual_adapter_contract": getattr(
            args, "residual_adapter_contract", None
        ),
        "runtime_model": (
            str(args.runtime_model.expanduser().resolve())
            if args.runtime_model
            else None
        ),
        "benchmark_root": str(args.benchmark_root.expanduser().resolve()),
        "benchmarks": benchmarks,
        "rank": args.rank,
        "world_size": args.world_size,
        "device": device,
        "limit_per_benchmark": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "diffusion_steps": args.diffusion_steps,
        "confidence_threshold": args.confidence_threshold,
        "block_size": args.block_size,
        "block_add_threshold": args.block_add_threshold,
        "decoded_token_threshold": args.decoded_token_threshold,
        "skip_threshold": args.skip_threshold,
        "temperature": args.temperature,
        "max_model_len": args.max_model_len,
        "kv_cache_capacity": args.kv_cache_capacity,
        "rope_scaling": args.rope_scaling,
        "rope_factor": args.rope_factor,
        "original_max_position_embeddings": (
            args.original_max_position_embeddings
        ),
        "allow_unscaled_max_model_len": (
            args.allow_unscaled_max_model_len
        ),
        "full_page_tiles": args.full_page_tiles,
        "full_page_tile_size": args.full_page_tile_size,
        "full_page_position_mode": args.full_page_position_mode,
        "full_page_overview": args.full_page_overview,
        "truncate_full_page_tiles": args.truncate_full_page_tiles,
        "attention_backend": args.attention_backend,
        "rms_norm_backend": args.rms_norm_backend,
        "kv_cache_compression": args.kv_cache_compression,
        "kv_cache_retrieval": args.kv_cache_retrieval,
        "kv_retrieval_topk_images": args.kv_retrieval_topk_images,
        "kv_retrieval_score_mode": args.kv_retrieval_score_mode,
        "kv_retrieval_mask_rounds": (
            args.kv_retrieval_mask_rounds
            if args.kv_retrieval_score_mode
            in {
                "masked_self_information",
                "cached_masked_self_information",
                "causal_masked_self_information",
            }
            else 0
        ),
        "kv_retrieval_packed_scoring": (
            args.kv_retrieval_packed_scoring
        ),
        "kv_retrieval_max_batch_tokens": (
            args.kv_retrieval_max_batch_tokens
        ),
        "kv_retrieval_keep_overview": args.kv_retrieval_keep_overview,
        "kv_retrieval_query_source": "native_resize_prompt",
        "vision_tile_size": args.vision_tile_size,
        "vision_topk_tiles": args.vision_topk_tiles,
        "vision_token_keep_ratio": args.vision_token_keep_ratio,
        "vision_score_query_window": args.vision_score_query_window,
        "vision_score_layers": args.vision_score_layers,
        "vision_score_layer_mode": args.vision_score_layer_mode,
        "vision_score_pool_kernel": args.vision_score_pool_kernel,
        "seed": args.seed,
        "sample_seed_policy": "sha256(base_seed, provenance.action_uid || sample_id)",
        "latency_scope": "synchronized image decode, preprocessing, cache construction, and generation",
    }


def main() -> None:
    args = parse_args()
    args.checkpoint_sha256 = None
    args.residual_adapter_contract = None
    if args.expected_checkpoint_sha256 is not None:
        expected_checkpoint_sha256 = validate_sha256(
            args.expected_checkpoint_sha256,
            name="--expected-checkpoint-sha256",
        )
        args.checkpoint_sha256 = sha256_file(args.checkpoint)
        if args.checkpoint_sha256 != expected_checkpoint_sha256:
            raise ResidualGroundingContractError(
                "evaluation Planner checkpoint SHA-256 mismatch: "
                f"expected={expected_checkpoint_sha256} "
                f"actual={args.checkpoint_sha256}"
            )
    if args.require_residual_adapter_contract:
        args.residual_adapter_contract = load_adapter_contract(
            args.adapter,
            expected_backbone_sha256=args.checkpoint_sha256,
            require_complete=True,
        )
    protocol = load_protocol(args.lladao_repo)
    parse_action, paired_sample_seed, iter_samples, load_completed, selected_benchmarks = protocol
    root = args.benchmark_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    benchmarks = selected_benchmarks(args, manifest)
    device = select_device(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = run_config(args, benchmarks, device)
    (output_dir / f"run-config-rank-{args.rank:05d}.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.require_residual_adapter_contract:
        print(
            "Residual release contract: loading Planner "
            f"sha256={args.checkpoint_sha256} and its bound adapter epoch",
            flush=True,
        )
    print(f"Rank {args.rank}/{args.world_size}: loading {args.backend} on {device}", flush=True)
    if args.backend == "d2f_vllm":
        os.environ["D2F_VLLM_ATTENTION_BACKEND"] = args.attention_backend
        os.environ["D2F_VLLM_RMS_NORM_BACKEND"] = args.rms_norm_backend
        from d2f_vllm.lladao_gui_engine import (
            LLaDAOGuiD2FEngine,
            LLaDAOGuiKVCompressionConfig,
            LLaDAOGuiKVRetrievalConfig,
        )

        rope_scaling = None
        if args.rope_scaling == "yarn":
            rope_scaling = {
                "rope_type": "yarn",
                "factor": args.rope_factor,
                "original_max_position_embeddings": (
                    args.original_max_position_embeddings
                ),
                "beta_fast": 32.0,
                "beta_slow": 1.0,
            }
        engine = LLaDAOGuiD2FEngine(
            args.runtime_model,
            max_model_len=args.max_model_len,
            block_length=args.block_size,
            max_new_tokens=args.max_new_tokens,
            block_add_threshold=args.block_add_threshold,
            decoded_token_threshold=args.decoded_token_threshold,
            skip_threshold=args.skip_threshold,
            temperature=args.temperature,
            master_port=args.master_port,
            kv_cache_capacity=args.kv_cache_capacity,
            rope_scaling=rope_scaling,
            allow_unscaled_max_model_len=(
                args.allow_unscaled_max_model_len
            ),
            kv_compression=LLaDAOGuiKVCompressionConfig(
                enabled=args.kv_cache_compression,
                vision_tile_size=args.vision_tile_size,
                vision_topk_tiles=args.vision_topk_tiles,
                vision_token_keep_ratio=args.vision_token_keep_ratio,
                vision_score_query_window=args.vision_score_query_window,
                vision_score_layers=args.vision_score_layers,
                vision_score_layer_mode=args.vision_score_layer_mode,
                vision_score_pool_kernel=args.vision_score_pool_kernel,
            ),
            kv_retrieval=LLaDAOGuiKVRetrievalConfig(
                enabled=args.kv_cache_retrieval,
                topk_images=args.kv_retrieval_topk_images,
                score_mode=args.kv_retrieval_score_mode,
                mask_rounds=args.kv_retrieval_mask_rounds,
                packed_scoring=args.kv_retrieval_packed_scoring,
                max_batch_tokens=args.kv_retrieval_max_batch_tokens,
                keep_overview=args.kv_retrieval_keep_overview,
            ),
        )
    else:
        engine = LLaDAOGuiD2FInference(
            lladao_repo=args.lladao_repo,
            model_path=args.model_path,
            checkpoint=args.checkpoint,
            adapter_path=args.adapter,
            device=device,
        )

    warmup_samples = []
    for benchmark in benchmarks:
        warmup_samples.extend(
            itertools.islice(
                iter_samples(
                    root,
                    manifest,
                    benchmark,
                    rank=args.rank,
                    world_size=args.world_size,
                    limit=args.limit,
                ),
                args.warmup,
            )
        )
        if len(warmup_samples) >= args.warmup:
            break
    for sample in warmup_samples[: args.warmup]:
        print(f"Rank {args.rank}: warmup {sample['sample_id']}", flush=True)
        infer_one(engine, root, sample, args, parse_action, paired_sample_seed)

    total_written = 0
    for benchmark in benchmarks:
        benchmark_dir = output_dir / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        output_path = benchmark_dir / f"part-{args.rank:05d}.jsonl"
        if args.no_resume and output_path.exists():
            output_path.unlink()
        completed = load_completed(output_path)
        pending = [
            sample
            for sample in iter_samples(
                root,
                manifest,
                benchmark,
                rank=args.rank,
                world_size=args.world_size,
                limit=args.limit,
            )
            if sample["sample_id"] not in completed
        ]
        print(
            f"Rank {args.rank}: {benchmark}: {len(pending)} pending, "
            f"{len(completed)} complete",
            flush=True,
        )
        with output_path.open("a", encoding="utf-8", buffering=1) as handle:
            for index, sample in enumerate(pending, start=1):
                try:
                    record = infer_one(
                        engine, root, sample, args, parse_action, paired_sample_seed
                    )
                except Exception as exc:
                    print(
                        f"Rank {args.rank}: failed {sample['sample_id']}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if args.fail_fast:
                        raise
                    record = error_record(sample, args, paired_sample_seed, exc)
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                if index % args.flush_every == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                total_written += 1
                if index == 1 or index % 10 == 0 or index == len(pending):
                    latency = record.get("latency_seconds")
                    latency_text = (
                        f"{latency:.3f}s"
                        if isinstance(latency, (int, float)) and math.isfinite(latency)
                        else "error"
                    )
                    print(
                        f"Rank {args.rank}: {benchmark} {index}/{len(pending)} "
                        f"{record.get('prediction')!r} {latency_text}",
                        flush=True,
                    )
    if hasattr(engine, "close"):
        engine.close()
    print(f"Rank {args.rank}: wrote {total_written} predictions", flush=True)


if __name__ == "__main__":
    main()
