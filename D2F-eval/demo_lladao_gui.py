#!/usr/bin/env python3
"""Run a single-image LLaDA-o GUI-grounding demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lladao_d2f.inference import LLaDAOGuiD2FInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("baseline", "d2f", "both"), default="d2f")
    parser.add_argument("--lladao-repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--block-add-threshold", type=float, default=0.1)
    parser.add_argument("--decoded-token-threshold", type=float, default=0.95)
    parser.add_argument("--skip-threshold", type=float, default=0.9)
    return parser.parse_args()


def clean(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.replace("<|endoftext|>", "").strip()


def summary(result: dict) -> dict:
    return {
        "text": clean(result["raw_text"]),
        "raw_text": result["raw_text"],
        "tokens": len(result["tokens"]),
        "iterations": result["iterations"],
        "generation_seconds": result["generation_seconds"],
        "total_seconds": result["total_seconds"],
    }


def main() -> None:
    args = parse_args()
    engine = LLaDAOGuiD2FInference(
        lladao_repo=args.lladao_repo,
        model_path=args.model_path,
        checkpoint=args.checkpoint,
        adapter_path=args.adapter,
        device=args.device,
    )
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    output = {}
    if args.backend in {"baseline", "both"}:
        output["baseline"] = summary(
            engine.generate_baseline(
                image,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                diffusion_steps=args.diffusion_steps,
                confidence_threshold=args.confidence_threshold,
            )
        )
    if args.backend in {"d2f", "both"}:
        output["d2f"] = summary(
            engine.generate(
                image,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                block_size=args.block_size,
                block_add_threshold=args.block_add_threshold,
                decoded_token_threshold=args.decoded_token_threshold,
                skip_threshold=args.skip_threshold,
            )
        )
    if args.backend == "both":
        baseline = output["baseline"]["generation_seconds"]
        d2f = output["d2f"]["generation_seconds"]
        output["generation_speedup"] = baseline / d2f
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
