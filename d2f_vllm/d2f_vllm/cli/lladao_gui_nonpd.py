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
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--block-add-threshold", type=float, default=0.1)
    parser.add_argument("--decoded-token-threshold", type=float, default=0.95)
    parser.add_argument("--skip-threshold", type=float, default=0.9)
    parser.add_argument("--mask-token-id", type=int, default=126336)
    parser.add_argument("--master-port", type=int, default=2333)
    parser.add_argument("--attention-backend", choices=("sdpa", "flex"), default="sdpa")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["D2F_VLLM_ATTENTION_BACKEND"] = args.attention_backend
    from d2f_vllm.lladao_gui_engine import LLaDAOGuiD2FEngine

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
    )
    try:
        with Image.open(args.image) as image:
            result = engine.generate_gui(
                image.convert("RGB"),
                args.prompt,
                max_new_tokens=args.max_new_tokens,
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
