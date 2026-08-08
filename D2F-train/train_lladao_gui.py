#!/usr/bin/env python3
"""Train the multimodal LLaDA-o GUI backend with D2F distillation."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch._dynamo
import yaml
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lladao_d2f.modeling import LLaDAOGuiD2FModel, add_lladao_repo, add_lora, load_base_model
from lladao_d2f.training import (
    advance_scheduler_for_optimizer_update,
    validate_scheduler_global_step,
)
from lladao_d2f.residual_grounding import (
    adapter_contract,
    audit_understanding_checkpoint,
    audit_zero_initialized_lora,
    domain_for_microstep,
    load_adapter_contract,
    validate_domain_schedule,
    write_json_atomic,
)


def as_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: as_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [as_namespace(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override paths.output_dir from the YAML config",
    )
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


def build_loader(config, tokenizer, special_tokens, accelerator, train_data: Path):
    add_lladao_repo(config.paths.lladao_repo)
    os.environ["LLADAO_GUI_GROUNDING_DIR"] = str(train_data)
    from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
    from data.dataset_info import DATASET_INFO

    # dataset_info captures LLADAO_GUI_GROUNDING_DIR at import time. Two-domain
    # residual training constructs two independent PackedDatasets in one
    # process, so bind the already-imported registry immediately before each
    # dataset captures its path.
    DATASET_INFO["vlm_parquet"]["gui_grounding_table1"]["data_dir"] = str(
        train_data
    )

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
    num_workers = int(config.data.num_workers)
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(config.data.prefetch_factor)
        loader_kwargs["timeout"] = float(getattr(config.data, "timeout_seconds", 0))
    return DataLoader(**loader_kwargs)


def resolve_training_domains(config) -> tuple[tuple[str, Path, bool], ...]:
    configured = getattr(config.data, "domains", None)
    if configured is None:
        return (
            (
                "mind2web",
                Path(config.paths.train_data).expanduser().resolve(),
                True,
            ),
        )
    domains = tuple(
        (
            name,
            Path(domain.path).expanduser().resolve(),
            bool(domain.distill),
        )
        for name, domain in vars(configured).items()
    )
    validate_domain_schedule(
        [(name, distill) for name, _, distill in domains],
        int(config.train.gradient_accumulation_steps),
    )
    return domains


def arm_stall_trace(timeout_seconds: float, trace_file) -> None:
    """Dump every Python thread if an optimizer step stops making progress."""
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(
        timeout_seconds,
        repeat=True,
        file=trace_file,
    )


def save_checkpoint(
    accelerator,
    model,
    optimizer,
    scheduler,
    output_root: Path,
    step: int,
    *,
    max_steps: int,
    backbone_audit: dict | None = None,
    lora_audit: dict | None = None,
    domain_counts: dict[str, int] | None = None,
    config_sha256: str | None = None,
    release_eligible: bool = False,
):
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
            # Every rank follows the same deterministic domain alternation.
            # Persist the rank-local counters so a resumed main process can
            # continue them without multiplying an already-global value.
            "domain_counts": dict(domain_counts or {}),
        },
        checkpoint / "training_state.pt",
    )
    if backbone_audit is not None:
        if lora_audit is None or config_sha256 is None:
            raise RuntimeError("residual adapter checkpoint audit is incomplete")
        contract = adapter_contract(
            backbone_audit=backbone_audit,
            lora_audit=lora_audit,
            step=step,
            max_steps=max_steps,
            domain_counts={
                name: int(count) * accelerator.num_processes
                for name, count in (domain_counts or {}).items()
            },
            config_sha256=config_sha256,
            release_eligible=release_eligible,
        )
        write_json_atomic(checkpoint / "adapter" / "training_contract.json", contract)


def restore_checkpoint(
    peft_model,
    optimizer,
    scheduler,
    checkpoint: Path,
    *,
    expected_backbone_sha256: str | None = None,
) -> tuple[int, dict[str, int]]:
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
    if expected_backbone_sha256 is not None:
        load_adapter_contract(
            checkpoint / "adapter",
            expected_backbone_sha256=expected_backbone_sha256,
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
    return int(training_state["step"]), {
        str(name): int(count)
        for name, count in training_state.get("domain_counts", {}).items()
    }


def main() -> None:
    # FlexAttention is a higher-order op, which Torch's DDP graph optimizer
    # cannot partition. Keep Dynamo enabled for FlexAttention itself.
    torch._dynamo.config.optimize_ddp = False
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    if args.output_dir is not None:
        raw_config["paths"]["output_dir"] = str(
            args.output_dir.expanduser().resolve()
        )
    config = as_namespace(raw_config)
    domains = resolve_training_domains(config)
    for domain_name, train_data, _ in domains:
        if (
            not train_data.is_dir()
            or next(train_data.rglob("*.parquet"), None) is None
        ):
            raise FileNotFoundError(
                f"{domain_name} training data contains no parquet shards: "
                f"{train_data}"
            )
    residual_training = len(domains) == 2
    expected_checkpoint_sha256 = (
        str(config.model.expected_checkpoint_sha256)
        if residual_training
        else None
    )
    expected_active_parameters = int(
        getattr(config.model, "expected_active_parameters", 8_459_716_512)
    )
    config_bytes = yaml.safe_dump(raw_config, sort_keys=True).encode("utf-8")
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
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
    # torchrun provides ranks before torch.distributed is initialized.  Build
    # and load the large CPU model first so its several-minute startup does not
    # consume the process-group collective timeout.
    process_index = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    set_seed(int(config.seed) + process_index)
    if process_index == 0:
        with (output_root / "resolved-config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(raw_config, handle, sort_keys=False)

    backbone_audit = None
    if residual_training and process_index == 0:
        backbone_audit = audit_understanding_checkpoint(
            config.paths.checkpoint,
            expected_sha256=expected_checkpoint_sha256,
            expected_parameters=expected_active_parameters,
        )
        write_json_atomic(output_root / "backbone-audit.json", backbone_audit)

    if residual_training and (
        int(config.lora.rank),
        int(config.lora.alpha),
        float(config.lora.dropout),
    ) != (32, 32, 0.1):
        raise ValueError(
            "residual grounding requires rank=32, alpha=32, dropout=0.1"
        )

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
    lora_audit = audit_zero_initialized_lora(peft_model)
    if residual_training and process_index == 0:
        write_json_atomic(output_root / "initial-lora-audit.json", lora_audit)
    model = LLaDAOGuiD2FModel(
        peft_model,
        mask_id=config.model.mask_id,
        block_size=config.model.block_size,
        distill_weight=config.train.distill_weight,
        hard_ce_weight=config.train.hard_ce_weight,
        action_ce_weight=float(getattr(config.train, "action_ce_weight", 0.0)),
        content_ce_weight=float(getattr(config.train, "content_ce_weight", 0.0)),
        full_response_mask_probability=float(
            getattr(config.train, "full_response_mask_probability", 0.0)
        ),
        content_ce_use_action_class_weight=bool(
            getattr(config.train, "content_ce_use_action_class_weight", True)
        ),
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
    domain_counts: dict[str, int] = {name: 0 for name, _, _ in domains}
    validate_scheduler_global_step(scheduler, step)
    if args.resume_from is not None:
        step, domain_counts = restore_checkpoint(
            peft_model,
            optimizer,
            scheduler,
            args.resume_from,
            expected_backbone_sha256=expected_checkpoint_sha256,
        )
        validate_scheduler_global_step(scheduler, step)
        if step >= max_steps:
            raise ValueError(f"resume step {step} must be less than max steps {max_steps}")
        if step >= stop_after_step:
            raise ValueError(
                f"resume step {step} must be less than stop-after-step {stop_after_step}"
            )
    project_config = ProjectConfiguration(
        project_dir=str(output_root), logging_dir=str(output_root / "logs")
    )
    ddp = DistributedDataParallelKwargs(
        find_unused_parameters=False, broadcast_buffers=False
    )
    distributed_config = getattr(config, "distributed", SimpleNamespace())
    process_group = InitProcessGroupKwargs(
        timeout=timedelta(
            seconds=float(getattr(distributed_config, "timeout_seconds", 300))
        )
    )
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        project_config=project_config,
        kwargs_handlers=[ddp, process_group],
    )
    loaders = {
        name: build_loader(config, tokenizer, special_tokens, accelerator, path)
        for name, path, _ in domains
    }
    # Keep the scheduler outside Accelerate. AcceleratedScheduler compensates
    # for non-split distributed batches by stepping once per process, but this
    # job defines max_steps in global optimizer updates and partitions its
    # iterable dataset explicitly by rank.
    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()

    iterators = {name: iter(loader) for name, loader in loaders.items()}
    domain_schedule = tuple((name, distill) for name, _, distill in domains)
    microstep = step * int(config.train.gradient_accumulation_steps)
    log_path = output_root / "train.jsonl"
    progress_handle = None
    if accelerator.is_main_process:
        progress_path = output_root / "progress.log"
        progress_handle = progress_path.open("a", encoding="utf-8", buffering=1)
        progress_handle.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"starting at step {step}/{stop_after_step}\n"
        )
    progress = tqdm(
        total=stop_after_step,
        initial=step,
        desc="D2F training",
        unit="step",
        dynamic_ncols=False,
        mininterval=float(getattr(config.train, "progress_refresh_seconds", 10)),
        miniters=1,
        smoothing=0.1,
        file=progress_handle if progress_handle is not None else sys.stderr,
        disable=not accelerator.is_main_process,
    )
    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    trace_handle = (diagnostics_dir / f"rank-{accelerator.process_index}.stack.log").open(
        "a", encoding="utf-8", buffering=1
    )
    trace_handle.write(
        f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"watching rank {accelerator.process_index} from step {step}\n"
    )
    stall_trace_seconds = float(getattr(config.train, "stall_trace_seconds", 120))
    arm_stall_trace(stall_trace_seconds, trace_handle)
    try:
        while step < stop_after_step:
            domain_name, distill_enabled = domain_for_microstep(
                microstep, domain_schedule
            )
            try:
                packed = next(iterators[domain_name])
            except StopIteration:
                iterators[domain_name] = iter(loaders[domain_name])
                packed = next(iterators[domain_name])
            batch = packed.cuda(accelerator.device).to_dict()
            batch.pop("batch_data_indexes", None)
            sample_count = len(batch["sample_lens"])
            batch["distill_sample_mask"] = torch.full(
                (sample_count,),
                bool(distill_enabled),
                dtype=torch.bool,
                device=accelerator.device,
            )
            microstep += 1
            domain_counts[domain_name] = domain_counts.get(domain_name, 0) + 1
            optimizer_updated = False
            with accelerator.accumulate(model):
                metrics = model(batch)
                accelerator.backward(metrics["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, float(config.train.max_grad_norm))
                optimizer.step()
                optimizer_updated = advance_scheduler_for_optimizer_update(
                    scheduler,
                    sync_gradients=accelerator.sync_gradients,
                    optimizer_step_was_skipped=accelerator.optimizer_step_was_skipped,
                )
                optimizer.zero_grad(set_to_none=True)
            if not optimizer_updated:
                continue
            step += 1
            validate_scheduler_global_step(scheduler, step)
            gathered = {
                key: accelerator.gather(value.detach().reshape(1)).float()
                for key, value in metrics.items()
            }
            reduced = {key: value.mean().item() for key, value in gathered.items()}
            count_metrics = (
                "full_response_masked_count",
                "d2f_response_count",
                "full_response_token_correct",
                "full_response_token_count",
                "full_response_exact",
                "full_response_count",
            )
            for key in count_metrics:
                reduced[key] = gathered[key].sum().item()
            reduced["full_response_masked_rate"] = (
                reduced["full_response_masked_count"]
                / max(reduced["d2f_response_count"], 1.0)
            )
            reduced["full_response_token_accuracy"] = (
                reduced["full_response_token_correct"]
                / max(reduced["full_response_token_count"], 1.0)
            )
            reduced["full_response_exact_rate"] = (
                reduced["full_response_exact"]
                / max(reduced["full_response_count"], 1.0)
            )
            current_lr = scheduler.get_last_lr()[0]
            if accelerator.is_main_process:
                progress.set_postfix(
                    loss=f"{reduced['loss']:.4f}",
                    lr=f"{current_lr:.2e}",
                    masked=f"{reduced['masked_tokens']:.1f}",
                    refresh=False,
                )
                progress.update(1)
                if step == 1 or step % int(config.train.log_every) == 0:
                    record = {"step": step, "lr": current_lr, **reduced}
                    record["domain_microbatches_per_rank"] = dict(domain_counts)
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    print(json.dumps(record, sort_keys=True), flush=True)
            arm_stall_trace(stall_trace_seconds, trace_handle)
            if step % int(config.train.save_every) == 0 or step == stop_after_step:
                save_checkpoint(
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    output_root,
                    step,
                    max_steps=max_steps,
                    backbone_audit=backbone_audit,
                    lora_audit=lora_audit,
                    domain_counts=domain_counts,
                    config_sha256=config_sha256,
                    release_eligible=bool(
                        getattr(config.train, "release_eligible", False)
                    ),
                )
    finally:
        faulthandler.cancel_dump_traceback_later()
        progress.close()
        trace_handle.close()
        if progress_handle is not None:
            progress_handle.close()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
