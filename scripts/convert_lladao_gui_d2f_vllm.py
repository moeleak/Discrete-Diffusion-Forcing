#!/usr/bin/env python3
"""Convert a GUI-grounding LLaDA-o checkpoint for native d2f_vllm inference.

The source checkpoint contains both understanding and visual-generation
experts.  The runtime only needs the understanding language path and the
vision prefix encoder.  This converter also merges the trained PEFT LoRA into
the four attention projections so serving never depends on PEFT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterator


LANGUAGE_PREFIX = "language_model."
VISION_PREFIXES = ("vit_model.", "connector.", "vit_pos_embed.")
LORA_TARGET = re.compile(
    r"^language_model\.model\.layers\.\d+\.self_attn\."
    r"(?:q_proj|k_proj|v_proj|o_proj)\.weight$"
)
TOKENIZER_FILES = (
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


def runtime_language_key(source_key: str) -> str | None:
    """Map an understanding-path checkpoint key to the runtime namespace."""
    if not source_key.startswith(LANGUAGE_PREFIX):
        return None
    if "_moe_gen" in source_key:
        return None
    key = source_key[len(LANGUAGE_PREFIX) :]
    if key in {
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
    }:
        return key
    if not key.startswith("model.layers."):
        return None
    allowed_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".self_attn.q_norm.weight",
        ".self_attn.k_norm.weight",
    )
    return key if key.endswith(allowed_suffixes) else None


def adapter_keys_for_source(source_key: str) -> tuple[str, str] | None:
    if not LORA_TARGET.match(source_key):
        return None
    stem = source_key.removesuffix(".weight")
    prefix = f"base_model.model.{stem}"
    return f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"


def _checkpoint_files(path: Path, basename: str) -> list[Path]:
    if path.is_file():
        return [path]
    direct = path / f"{basename}.safetensors"
    if direct.is_file():
        return [direct]
    index_path = path / f"{basename}.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        return sorted({path / item for item in index["weight_map"].values()})
    files = sorted(path.glob(f"{basename}-*.safetensors"))
    if files:
        return files
    raise FileNotFoundError(f"cannot find {basename} safetensors under {path}")


def _tensor_locations(files: list[Path]) -> dict[str, Path]:
    from safetensors import safe_open

    result: dict[str, Path] = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in result:
                    raise ValueError(f"duplicate tensor {key} in {file} and {result[key]}")
                result[key] = file
    return result


def _read_tensor(path: Path, key: str):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _load_adapter(adapter_dir: Path) -> tuple[dict, dict]:
    from safetensors import safe_open

    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    tensors = {}
    files = sorted(adapter_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no adapter safetensors found under {adapter_dir}")
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
    return config, tensors


def _write_language_shards(
    output_dir: Path,
    tensors: Iterator[tuple[str, object]],
    *,
    max_shard_bytes: int,
) -> tuple[dict[str, str], int]:
    from safetensors.torch import save_file

    pending: dict[str, object] = {}
    pending_bytes = 0
    temporary_files: list[Path] = []
    temporary_weight_map: dict[str, str] = {}
    total_size = 0

    def flush() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        filename = f"model-{len(temporary_files) + 1:05d}.safetensors"
        save_file(pending, str(output_dir / filename), metadata={"format": "pt"})
        temporary_files.append(output_dir / filename)
        temporary_weight_map.update({key: filename for key in pending})
        pending = {}
        pending_bytes = 0

    for key, tensor in tensors:
        size = tensor.numel() * tensor.element_size()
        if pending and pending_bytes + size > max_shard_bytes:
            flush()
        pending[key] = tensor.contiguous()
        pending_bytes += size
        total_size += size
    flush()

    count = len(temporary_files)
    final_weight_map: dict[str, str] = {}
    for index, temporary in enumerate(temporary_files, 1):
        final_name = f"model-{index:05d}-of-{count:05d}.safetensors"
        temporary.rename(output_dir / final_name)
        for key, old_name in temporary_weight_map.items():
            if old_name == temporary.name:
                final_weight_map[key] = final_name
    return final_weight_map, total_size


def _copy_tokenizer_files(model_dir: Path, output_dir: Path) -> list[str]:
    copied = []
    for name in TOKENIZER_FILES:
        source = model_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
            copied.append(name)
    if not copied:
        raise FileNotFoundError(f"no tokenizer files found under {model_dir}")
    return copied


def convert(args: argparse.Namespace) -> None:
    import torch
    from safetensors.torch import save_file

    model_dir = args.model_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    adapter_dir = args.adapter.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    temporary_dir = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    try:
        source_files = _checkpoint_files(checkpoint, "ema")
        source_locations = _tensor_locations(source_files)
        adapter_config, adapter_tensors = _load_adapter(adapter_dir)
        rank = int(adapter_config["r"])
        alpha = float(adapter_config["lora_alpha"])
        scale = alpha / rank
        merged_modules: list[str] = []

        def language_tensors():
            for source_key in sorted(source_locations):
                runtime_key = runtime_language_key(source_key)
                if runtime_key is None:
                    continue
                tensor = _read_tensor(source_locations[source_key], source_key)
                adapter_keys = adapter_keys_for_source(source_key)
                if adapter_keys is not None:
                    key_a, key_b = adapter_keys
                    if key_a not in adapter_tensors or key_b not in adapter_tensors:
                        raise KeyError(f"missing LoRA pair for {source_key}")
                    delta = torch.mm(
                        adapter_tensors[key_b].float(), adapter_tensors[key_a].float()
                    ).mul_(scale)
                    tensor = tensor.float().add_(delta).to(tensor.dtype)
                    merged_modules.append(runtime_key.removesuffix(".weight"))
                yield runtime_key, tensor

        weight_map, total_size = _write_language_shards(
            temporary_dir,
            language_tensors(),
            max_shard_bytes=int(args.max_shard_size_gib * 2**30),
        )
        expected_merges = int(args.expected_layers) * 4
        if len(merged_modules) != expected_merges:
            raise RuntimeError(
                f"expected {expected_merges} merged attention modules, got {len(merged_modules)}"
            )
        (temporary_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        vision = {
            key: _read_tensor(path, key)
            for key, path in sorted(source_locations.items())
            if key.startswith(VISION_PREFIXES)
        }
        if not vision:
            raise RuntimeError("source checkpoint has no vision-prefix tensors")
        save_file(vision, str(temporary_dir / "vision.safetensors"), metadata={"format": "pt"})

        llm_config = json.loads((model_dir / "llm_config.json").read_text())
        llm_config.update(
            {
                "architectures": ["LLaDAOGuiForDiffusionLM"],
                "model_type": "lladao_gui",
                "qk_norm": True,
                "tie_word_embeddings": False,
            }
        )
        llm_config.pop("auto_map", None)
        (temporary_dir / "config.json").write_text(
            json.dumps(llm_config, indent=2, sort_keys=True) + "\n"
        )

        vision_config = json.loads((model_dir / "vit_config.json").read_text())
        checkpoint_layers = {
            int(key.split(".")[4])
            for key in vision
            if key.startswith("vit_model.vision_model.encoder.layers.")
        }
        vision_config["num_hidden_layers"] = len(checkpoint_layers)
        vision_config["rope"] = False
        (temporary_dir / "vision_config.json").write_text(
            json.dumps(vision_config, indent=2, sort_keys=True) + "\n"
        )
        tokenizer_files = _copy_tokenizer_files(model_dir, temporary_dir)
        manifest = {
            "format": "lladao-gui-d2f-vllm-merged-v1",
            "source_checkpoint": str(checkpoint),
            "source_adapter": str(adapter_dir),
            "adapter_rank": rank,
            "adapter_alpha": alpha,
            "adapter_scale": scale,
            "merged_module_count": len(merged_modules),
            "merged_modules": merged_modules,
            "language_tensor_count": len(weight_map),
            "language_bytes": total_size,
            "vision_tensor_count": len(vision),
            "tokenizer_files": tokenizer_files,
        }
        (temporary_dir / "runtime_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary_dir.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    print(
        f"converted {len(weight_map)} language tensors, merged {len(merged_modules)} LoRA modules, "
        f"and copied {len(vision)} vision tensors to {output_dir}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-layers", type=int, default=32)
    parser.add_argument("--max-shard-size-gib", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
