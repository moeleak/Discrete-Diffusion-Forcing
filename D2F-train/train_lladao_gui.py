#!/usr/bin/env python3
"""Train the multimodal LLaDA-o GUI backend with D2F distillation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Keep variable-length packed batches from fragmenting the CUDA allocator.
# This changes allocation strategy only; model and optimizer precision stay
# identical to the configured bf16-base/fp32-LoRA training recipe.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch._dynamo
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lladao_d2f.modeling import LLaDAOGuiD2FModel, add_lladao_repo, add_lora, load_base_model


def as_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: as_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [as_namespace(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help=(
            "stop and checkpoint at this absolute step while retaining the "
            "--max-steps optimizer schedule (used by distributed smoke tests)"
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="step directory containing adapter/ and training_state.pt",
    )
    return parser.parse_args()


def build_loader(config, tokenizer, special_tokens, accelerator):
    add_lladao_repo(config.paths.lladao_repo)
    os.environ["LLADAO_GUI_GROUNDING_DIR"] = str(
        Path(config.paths.train_data).resolve()
    )
    from data.dataset_base import DataConfig, PackedDataset, collate_wrapper

    with Path(config.paths.dataset_config).open() as handle:
        grouped = yaml.safe_load(handle)
    data_config = DataConfig(grouped_datasets=grouped)
    data_config.visual_und = True
    data_config.visual_und_sft = True
    data_config.merge_vit_text_segments = True
    data_config.vit_patch_size = 14
    data_config.max_num_patch_per_side = 70
    data_config.loss_reduction = "square"
    dataset = PackedDataset(
        data_config,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        local_rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        num_workers=config.data.num_workers,
        expected_num_tokens=config.data.expected_num_tokens,
        max_num_tokens_per_sample=config.data.max_num_tokens_per_sample,
        max_num_tokens=config.data.max_num_tokens,
        prefer_buffer_before=config.data.prefer_buffer_before,
        max_buffer_size=config.data.max_buffer_size,
        use_flex=True,
    )
    dataset.set_epoch(config.seed)
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        prefetch_factor=config.data.prefetch_factor,
    )


def save_checkpoint(accelerator, model, optimizer, scheduler, output_root: Path, step: int):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    checkpoint = output_root / f"step-{step:07d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.peft_model.save_pretrained(checkpoint / "adapter", safe_serialization=True)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint / "training_state.pt",
    )


def restore_checkpoint(peft_model, optimizer, scheduler, checkpoint: Path) -> int:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    checkpoint = checkpoint.expanduser().resolve()
    adapter_file = checkpoint / "adapter" / "adapter_model.safetensors"
    state_file = checkpoint / "training_state.pt"
    if not adapter_file.is_file() or not state_file.is_file():
        raise FileNotFoundError(
            f"resume checkpoint must contain {adapter_file.name} and {state_file.name}: "
            f"{checkpoint}"
        )
    adapter_state = load_file(str(adapter_file), device="cpu")
    incompatible = set_peft_model_state_dict(peft_model, adapter_state)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected adapter keys while resuming: {incompatible.unexpected_keys[:8]}"
        )
    training_state = torch.load(state_file, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(training_state["optimizer"])
    scheduler.load_state_dict(training_state["scheduler"])
    return int(training_state["step"])


def main() -> None:
    # FlexAttention is a higher-order op, which Torch's DDP graph optimizer
    # cannot partition. Keep Dynamo enabled for FlexAttention itself.
    torch._dynamo.config.optimize_ddp = False
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    config = as_namespace(raw_config)
    train_data = Path(config.paths.train_data).expanduser().resolve()
    if not train_data.is_dir() or next(train_data.rglob("*.parquet"), None) is None:
        raise FileNotFoundError(f"training data contains no parquet shards: {train_data}")
    os.environ["LLADAO_GUI_GROUNDING_DIR"] = str(train_data)
    max_steps = int(
        args.max_steps if args.max_steps is not None else config.train.max_steps
    )
    stop_after_step = int(
        args.stop_after_step if args.stop_after_step is not None else max_steps
    )
    if max_steps <= 0 or stop_after_step <= 0 or stop_after_step > max_steps:
        raise ValueError("steps must satisfy 0 < stop-after-step <= max-steps")
    output_root = Path(config.paths.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    project_config = ProjectConfiguration(project_dir=str(output_root), logging_dir=str(output_root / "logs"))
    ddp = DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        project_config=project_config,
        kwargs_handlers=[ddp],
    )
    set_seed(int(config.seed) + accelerator.process_index)
    if accelerator.is_main_process:
        with (output_root / "resolved-config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(raw_config, handle, sort_keys=False)

    base, tokenizer, special_tokens = load_base_model(
        config.paths.lladao_repo,
        config.paths.model_path,
        config.paths.checkpoint,
    )
    peft_model = add_lora(
        base,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
    )
    model = LLaDAOGuiD2FModel(
        peft_model,
        mask_id=config.model.mask_id,
        block_size=config.model.block_size,
        distill_weight=config.train.distill_weight,
        hard_ce_weight=config.train.hard_ce_weight,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.train.lr,
        betas=tuple(config.train.betas),
        eps=config.train.eps,
        weight_decay=config.train.weight_decay,
    )
    warmup_steps = max(1, round(max_steps * float(config.train.warmup_ratio)))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    step = 0
    if args.resume_from is not None:
        step = restore_checkpoint(
            peft_model,
            optimizer,
            scheduler,
            args.resume_from,
        )
        if step >= max_steps:
            raise ValueError(f"resume step {step} must be less than max steps {max_steps}")
        if step >= stop_after_step:
            raise ValueError(
                f"resume step {step} must be less than stop-after-step {stop_after_step}"
            )
    loader = build_loader(config, tokenizer, special_tokens, accelerator)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    model.train()

    iterator = iter(loader)
    log_path = output_root / "train.jsonl"
    while step < stop_after_step:
        try:
            packed = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            packed = next(iterator)
        batch = packed.cuda(accelerator.device).to_dict()
        batch.pop("batch_data_indexes", None)
        with accelerator.accumulate(model):
            metrics = model(batch)
            accelerator.backward(metrics["loss"])
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable, float(config.train.max_grad_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if not accelerator.sync_gradients:
            continue
        step += 1
        reduced = {
            key: accelerator.gather(value.detach().reshape(1)).float().mean().item()
            for key, value in metrics.items()
        }
        if accelerator.is_main_process and (step == 1 or step % int(config.train.log_every) == 0):
            record = {"step": step, "lr": scheduler.get_last_lr()[0], **reduced}
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
        if step % int(config.train.save_every) == 0 or step == stop_after_step:
            save_checkpoint(accelerator, model, optimizer, scheduler, output_root, step)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
