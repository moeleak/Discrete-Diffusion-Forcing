#!/usr/bin/env python3
"""Compare unscaled and YaRN logits at absolute positions up to 128K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="Click on the requested web interface element.",
    )
    parser.add_argument("--master-port", type=int, default=32403)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from d2f_vllm.lladao_gui_engine import LLaDAOGuiD2FEngine

    max_model_len = 131_072
    offsets = [0, 16_384, 32_768, 65_536]
    payload = {
        "model": str(args.model.expanduser().resolve()),
        "max_model_len": max_model_len,
        "original_max_position_embeddings": 16_384,
        "factor": 8.0,
        "prompt": args.prompt,
        "runs": {},
    }
    for index, mode in enumerate(("unscaled", "yarn")):
        scaling = (
            None
            if mode == "unscaled"
            else {
                "rope_type": "yarn",
                "factor": 8.0,
                "original_max_position_embeddings": 16_384,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
            }
        )
        engine = LLaDAOGuiD2FEngine(
            args.model,
            max_model_len=max_model_len,
            kv_cache_capacity=256,
            rope_scaling=scaling,
            allow_unscaled_max_model_len=mode == "unscaled",
            master_port=args.master_port + index,
        )
        try:
            prompt_token_count = len(
                engine.tokenizer.encode(
                    args.prompt, add_special_tokens=False
                )
            ) + 2
            run_offsets = [
                *offsets,
                max_model_len - prompt_token_count,
            ]
            payload["runs"][mode] = engine.diagnose_absolute_positions(
                args.prompt, run_offsets
            )
        finally:
            engine.close()
            del engine
            torch.cuda.empty_cache()

    for unscaled, yarn in zip(
        payload["runs"]["unscaled"], payload["runs"]["yarn"]
    ):
        if unscaled["offset"] != yarn["offset"]:
            raise RuntimeError("diagnostic offsets do not align")
        unscaled["top1_matches_yarn"] = (
            unscaled["top_token_ids"][0] == yarn["top_token_ids"][0]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
