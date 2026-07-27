from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GUI-grounding LLaDA-o through native d2f_vllm Non-PD"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--kv-cache-capacity", type=int)
    parser.add_argument(
        "--rope-scaling",
        choices=("none", "yarn"),
        default="none",
    )
    parser.add_argument("--rope-factor", type=float, default=8.0)
    parser.add_argument(
        "--original-max-position-embeddings",
        type=int,
        default=16384,
    )
    parser.add_argument(
        "--allow-unscaled-max-model-len",
        action="store_true",
    )
    parser.add_argument(
        "--full-page-tiles",
        action=argparse.BooleanOptionalAction,
        default=False,
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
            "drop trailing complete full-resolution tiles until the prompt "
            "and generation fit the resident context capacity"
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
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--block-add-threshold", type=float, default=0.1)
    parser.add_argument("--decoded-token-threshold", type=float, default=0.95)
    parser.add_argument("--skip-threshold", type=float, default=0.9)
    parser.add_argument("--mask-token-id", type=int, default=126336)
    parser.add_argument("--master-port", type=int, default=2333)
    parser.add_argument("--attention-backend", choices=("sdpa", "flex"), default="sdpa")
    parser.add_argument(
        "--rms-norm-backend", choices=("torch", "vllm"), default="vllm"
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
            "does not evict tokens within a selected span"
        ),
    )
    parser.add_argument("--kv-retrieval-topk-images", type=int, default=4)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kv_cache_retrieval and args.kv_cache_compression:
        raise SystemExit(
            "--kv-cache-retrieval and --kv-cache-compression are mutually "
            "exclusive"
        )
    if args.kv_retrieval_mask_rounds <= 0:
        raise SystemExit("--kv-retrieval-mask-rounds must be positive")
    if (
        args.max_model_len > args.original_max_position_embeddings
        and args.rope_scaling == "none"
        and not args.allow_unscaled_max_model_len
    ):
        raise SystemExit(
            "an extended unscaled run requires "
            "--allow-unscaled-max-model-len"
        )
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
        args.model,
        max_model_len=args.max_model_len,
        block_length=args.block_length,
        max_new_tokens=args.max_new_tokens,
        mask_token_id=args.mask_token_id,
        block_add_threshold=args.block_add_threshold,
        decoded_token_threshold=args.decoded_token_threshold,
        skip_threshold=args.skip_threshold,
        master_port=args.master_port,
        kv_cache_capacity=args.kv_cache_capacity,
        rope_scaling=rope_scaling,
        allow_unscaled_max_model_len=args.allow_unscaled_max_model_len,
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
            mask_rounds=args.kv_retrieval_mask_rounds,
            keep_overview=args.kv_retrieval_keep_overview,
        ),
    )
    try:
        with Image.open(args.image) as image:
            result = engine.generate_gui(
                image.convert("RGB"),
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                full_page=args.full_page_tiles,
                full_page_tile_size=args.full_page_tile_size,
                full_page_position_mode=args.full_page_position_mode,
                full_page_overview=args.full_page_overview,
                truncate_full_page_tiles=args.truncate_full_page_tiles,
                retrieval_query=args.prompt,
            ).to_dict()
    finally:
        engine.close()
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload, flush=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n")


if __name__ == "__main__":
    main()
