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
    parser.add_argument("--master-port", type=int, default=2333)
    parser.add_argument(
        "--attention-backend", choices=("sdpa", "flex"), default="sdpa"
    )
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
    )
    return {
        "raw_text": output.text,
        "tokens": output.token_ids,
        "image_cache_seconds": output.image_seconds,
        "prompt_cache_seconds": output.prompt_seconds,
        "generation_seconds": output.generation_seconds,
        "total_seconds": output.total_seconds,
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with Image.open(root / sample["image"]) as source:
        image = source.convert("RGB")
        result = model_generate(engine, image, sample["prompt"], args)
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
        "latency_seconds": latency,
        "model_elapsed_seconds": result["total_seconds"],
        "image_cache_seconds": result["image_cache_seconds"],
        "prompt_cache_seconds": result["prompt_cache_seconds"],
        "generation_seconds": result["generation_seconds"],
        "convergence_steps": result["iterations"],
        "valid_tokens": len(result["tokens"]),
        "generated_tokens": args.max_new_tokens,
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
        "latency_seconds": None,
        "model_elapsed_seconds": None,
        "image_cache_seconds": None,
        "prompt_cache_seconds": None,
        "generation_seconds": None,
        "convergence_steps": None,
        "valid_tokens": None,
        "generated_tokens": None,
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
        "adapter": str(args.adapter.expanduser().resolve()) if args.adapter else None,
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
        "attention_backend": args.attention_backend,
        "seed": args.seed,
        "sample_seed_policy": "sha256(base_seed, provenance.action_uid || sample_id)",
        "latency_scope": "synchronized image decode, preprocessing, cache construction, and generation",
    }


def main() -> None:
    args = parse_args()
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

    print(f"Rank {args.rank}/{args.world_size}: loading {args.backend} on {device}", flush=True)
    if args.backend == "d2f_vllm":
        os.environ["D2F_VLLM_ATTENTION_BACKEND"] = args.attention_backend
        from d2f_vllm.lladao_gui_engine import LLaDAOGuiD2FEngine

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
