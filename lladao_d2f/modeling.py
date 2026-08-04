from __future__ import annotations

import importlib
import inspect
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .masking import create_full_document_mask, create_training_block_mask
from .noise import rebuild_and_corrupt_responses


LLM_LORA_PATTERN = (
    r"language_model\.model\.layers\.\d+\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj)"
)


def strip_unused_generation_experts(model) -> int:
    """Remove visual-generation experts from an understanding-only model.

    LLaDA-o MoT checkpoints contain a second set of attention, MLP, and norm
    modules whose names end in ``_moe_gen``.  GUI grounding runs exclusively
    with ``visual_gen=False`` and empty generation-token indexes, so those
    modules receive only empty tensors during training and are never reached
    during understanding-mode inference.  Replacing them after strict
    checkpoint validation preserves the GUI computation exactly while
    avoiding moving roughly seven billion unused parameters to every GPU.
    """
    if getattr(getattr(model, "config", None), "visual_gen", None) is not False:
        raise ValueError("generation experts may only be stripped with visual_gen=False")

    removed_parameters = 0
    removed_modules: list[str] = []
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not child_name.endswith("_moe_gen"):
                continue
            removed_parameters += sum(parameter.numel() for parameter in child.parameters())
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            removed_modules.append(full_name)
            setattr(module, child_name, nn.Identity())

    remaining = [name for name, _ in model.named_parameters() if "_moe_gen" in name]
    if remaining:
        raise RuntimeError(f"generation parameters remain after pruning: {remaining[:4]}")
    if not removed_modules or removed_parameters == 0:
        raise RuntimeError("expected LLaDA-o MoT generation experts but found none")
    return removed_parameters


def _summarize_checkpoint_mismatches(values: list[str], *, limit: int = 8) -> str:
    shown = values[:limit]
    suffix = f", ... ({len(values)} total)" if len(values) > limit else ""
    return ", ".join(shown) + suffix


def _validate_checkpoint_tensors(
    expected: dict[str, torch.Tensor],
    checkpoint: dict[str, torch.Tensor],
) -> None:
    """Reject every key, shape, or dtype mismatch before assigning tensors."""

    expected_keys = set(expected)
    checkpoint_keys = set(checkpoint)
    missing = sorted(expected_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - expected_keys)
    common = sorted(expected_keys & checkpoint_keys)
    shape_mismatches = [
        f"{key}: expected {tuple(expected[key].shape)}, got {tuple(checkpoint[key].shape)}"
        for key in common
        if expected[key].shape != checkpoint[key].shape
    ]
    dtype_mismatches = [
        f"{key}: expected {expected[key].dtype}, got {checkpoint[key].dtype}"
        for key in common
        if expected[key].dtype != checkpoint[key].dtype
    ]
    problems = []
    for name, values in (
        ("missing", missing),
        ("unexpected", unexpected),
        ("shape mismatches", shape_mismatches),
        ("dtype mismatches", dtype_mismatches),
    ):
        if values:
            problems.append(f"{name}=[{_summarize_checkpoint_mismatches(values)}]")
    if problems:
        raise RuntimeError("checkpoint mismatch: " + "; ".join(problems))


def _assign_strict_checkpoint(
    model: nn.Module,
    checkpoint: dict[str, torch.Tensor],
) -> None:
    expected = model.state_dict()
    _validate_checkpoint_tensors(expected, checkpoint)
    del expected
    incompatible = model.load_state_dict(checkpoint, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint mismatch while assigning tensors: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    meta_tensors = [
        f"parameter:{name}"
        for name, parameter in model.named_parameters()
        if parameter.is_meta
    ]
    meta_tensors.extend(
        f"buffer:{name}" for name, buffer in model.named_buffers() if buffer.is_meta
    )
    if meta_tensors:
        raise RuntimeError(
            "checkpoint load left tensors on meta: "
            + _summarize_checkpoint_mismatches(meta_tensors)
        )


def load_strict_pruned_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> int:
    """Strictly bind a safetensors checkpoint to a pruned meta model.

    Validation is deliberately performed against the complete model before
    pruning.  This prevents generation-expert keys, or any other unexpected
    checkpoint content, from being silently ignored.  Once validated, only
    understanding-path tensors are assigned, so generation experts never
    need an allocated CPU copy.
    """
    from safetensors.torch import load_file

    checkpoint = load_file(
        str(Path(checkpoint_path).expanduser().resolve()),
        device="cpu",
    )
    expected = model.state_dict()
    _validate_checkpoint_tensors(expected, checkpoint)

    removed_parameters = strip_unused_generation_experts(model)
    retained = {key: checkpoint[key] for key in model.state_dict()}
    del checkpoint, expected
    _assign_strict_checkpoint(model, retained)
    del retained
    return removed_parameters


def load_strict_understanding_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> int:
    """Load a full-parameter understanding-only checkpoint into a meta model.

    Unlike :func:`load_strict_pruned_checkpoint`, this format intentionally
    omits every unused ``*_moe_gen`` tensor.  The omission is accepted only as
    a complete set: after pruning, all remaining keys, shapes, and dtypes must
    match exactly.
    """
    from safetensors.torch import load_file

    removed_parameters = strip_unused_generation_experts(model)
    checkpoint = load_file(
        str(Path(checkpoint_path).expanduser().resolve()),
        device="cpu",
    )
    _assign_strict_checkpoint(model, checkpoint)
    del checkpoint
    return removed_parameters


def load_strict_understanding_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> tuple[int, str]:
    """Load either supported strict checkpoint format for understanding use.

    Original LLaDA-o checkpoints contain generation experts and are validated
    in full before those experts are pruned.  Full-parameter checkpoints
    exported by understanding-only training contain no generation-expert keys
    and are validated against the exact pruned model instead.  A checkpoint
    may not silently mix the two formats.
    """
    from safetensors import safe_open

    resolved = Path(checkpoint_path).expanduser().resolve()
    with safe_open(str(resolved), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    if any("_moe_gen" in key for key in keys):
        return load_strict_pruned_checkpoint(model, resolved), "complete"
    return load_strict_understanding_checkpoint(model, resolved), "understanding-only"


def _convert_conv2d_to_linear_on_meta(embeddings: nn.Module, config: Any) -> None:
    conversion = getattr(embeddings, "convert_conv2d_to_linear", None)
    if conversion is None:
        raise RuntimeError("LLaDA-o vision embeddings lack convert_conv2d_to_linear")
    try:
        parameters = inspect.signature(conversion).parameters
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "cannot verify LLaDA-o meta vision-conversion compatibility"
        ) from error
    if "meta" not in parameters:
        raise RuntimeError(
            "LLaDA-o convert_conv2d_to_linear must explicitly support meta=True"
        )
    conversion(config, meta=True)


def add_lladao_repo(lladao_repo: str | Path) -> Path:
    path = Path(lladao_repo).expanduser().resolve()
    if not (path / "modeling" / "lladao" / "lladao.py").is_file():
        raise FileNotFoundError(f"not a LLaDA-o checkout: {path}")
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
    return path


def load_base_model(
    lladao_repo: str | Path,
    model_path: str | Path,
    checkpoint_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
):
    add_lladao_repo(lladao_repo)
    from data.data_utils import add_special_tokens
    from modeling.lladao import (
        LLaDAO,
        LLaDAOConfig,
        LLaDAConfig,
        LLaDAModelLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from accelerate import init_empty_weights
    from transformers import AutoTokenizer

    model_path = Path(model_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    llm_config = LLaDAConfig.from_json_file(str(model_path / "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "LLaDAMoTDecoderLayer"
    llm_config.freeze_und = False
    vit_config = SiglipVisionConfig.from_json_file(str(model_path / "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    config = LLaDAOConfig(
        visual_gen=False,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=None,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )
    # Keep parameters on meta while retaining real, non-persistent runtime
    # buffers.  The latter is required for RoPE buffers that are absent from
    # the checkpoint and therefore cannot be materialized by state loading.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with init_empty_weights(include_buffers=False):
            language_model = LLaDAModelLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = LLaDAO(language_model, vit_model, None, config)
            _convert_conv2d_to_linear_on_meta(
                model.vit_model.vision_model.embeddings,
                vit_config,
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    removed_parameters, checkpoint_format = load_strict_understanding_model_checkpoint(
        model, checkpoint_path
    )
    print(
        f"loaded {checkpoint_format} LLaDA-o checkpoint; "
        "stripped unused generation experts: "
        f"{removed_parameters:,} parameters "
        f"({removed_parameters * 2 / 2**30:.2f} GiB at bf16)",
        flush=True,
    )
    model.to(dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer, special_tokens, _ = add_special_tokens(tokenizer)
    return model, tokenizer, special_tokens


def add_lora(
    model,
    *,
    rank: int = 32,
    alpha: int = 32,
    dropout: float = 0.1,
):
    from peft import LoraConfig, get_peft_model

    for parameter in model.parameters():
        parameter.requires_grad = False
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LLM_LORA_PATTERN,
        bias="none",
    )
    peft_model = get_peft_model(model, config)
    matched = [
        name
        for name, module in peft_model.named_modules()
        if hasattr(module, "lora_A") and re.search(r"self_attn\.(q_proj|k_proj|v_proj|o_proj)$", name)
    ]
    if len(matched) != 128:
        raise RuntimeError(f"expected 128 language attention LoRA modules, found {len(matched)}")
    bad = [name for name in matched if "vit_model" in name or "moe_gen" in name]
    if bad:
        raise RuntimeError(f"LoRA leaked outside the understanding attention path: {bad[:4]}")
    return peft_model


def load_adapter(model, adapter_path: str | Path):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(Path(adapter_path).expanduser().resolve()))


def unwrap_lladao(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def prepare_understanding_sequence(model, batch: dict[str, Any]) -> torch.Tensor:
    base = unwrap_lladao(model)
    packed_text_ids = batch["packed_text_ids"]
    packed_text_indexes = batch["packed_text_indexes"]
    packed_text_embedding = base.language_model.model.embed_tokens(packed_text_ids)
    sequence = packed_text_embedding.new_zeros((int(batch["sequence_length"]), base.hidden_size))
    sequence[packed_text_indexes] = packed_text_embedding

    vit_lens = batch["vit_token_seqlens"]
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_lens, dim=0), (1, 0)).to(torch.int32)
    vit_embed = base.vit_model(
        packed_pixel_values=batch["packed_vit_tokens"],
        packed_flattened_position_ids=batch["packed_vit_position_ids"],
        cu_seqlens=cu_seqlens,
        max_seqlen=int(torch.max(vit_lens).item()),
    )
    vit_embed = base.connector(vit_embed)
    vit_embed = vit_embed + base.vit_pos_embed(batch["packed_vit_position_ids"])
    sequence[batch["packed_vit_token_indexes"]] = vit_embed.to(sequence.dtype)
    return sequence


def forward_masked_logits(
    model,
    packed_sequence: torch.Tensor,
    batch: dict[str, Any],
    attention_mask,
) -> torch.Tensor:
    base = unwrap_lladao(model)
    understanding_indexes = torch.cat(
        [batch["packed_text_indexes"], batch["packed_vit_token_indexes"]], dim=0
    )
    empty_generation_indexes = torch.empty(
        0, dtype=torch.long, device=understanding_indexes.device
    )
    # This path always consumes the packed training representation, including
    # the block/bidirectional attention mask.  Reconstruction diagnostics put
    # the wrapper in eval mode to disable LoRA dropout, but LLaDA-o uses the
    # training flag at every nested model/layer/attention module to choose
    # between packed and KV-cache signatures.  Flip only those dispatchers'
    # flags directly (not recursively via train()) so dropout stays disabled.
    dispatchers = [
        module
        for module in base.language_model.modules()
        if callable(getattr(type(module), "forward_train", None))
        and callable(getattr(type(module), "forward_inference", None))
    ]
    dispatcher_states = [module.training for module in dispatchers]
    try:
        for module in dispatchers:
            module.training = True
        hidden, _ = base.language_model(
            packed_sequence=packed_sequence,
            sample_lens=batch["sample_lens"],
            attention_mask=attention_mask,
            packed_position_ids=batch["packed_position_ids"],
            packed_und_token_indexes=understanding_indexes,
            packed_gen_token_indexes=empty_generation_indexes,
        )
    finally:
        for module, training in zip(dispatchers, dispatcher_states, strict=True):
            module.training = training
    return base.language_model.lm_head(hidden[batch["ce_loss_indexes"].long()])


def adapter_disabled(model):
    target = model.module if hasattr(model, "module") else model
    return target.disable_adapter() if hasattr(target, "disable_adapter") else nullcontext()


def combine_d2f_and_action_losses(
    *,
    distill: torch.Tensor,
    hard_ce: torch.Tensor,
    d2f_weights: torch.Tensor,
    action_mask: torch.Tensor,
    distill_weight: float,
    hard_ce_weight: float,
    action_ce_weight: float,
    action_class_weight: torch.Tensor,
    content_mask: torch.Tensor | None = None,
    content_ce_weight: float = 0.0,
    content_ce_use_action_class_weight: bool = True,
) -> dict[str, torch.Tensor]:
    if content_mask is None:
        content_mask = torch.zeros_like(action_mask, dtype=torch.bool)
    if not (
        distill.shape
        == hard_ce.shape
        == d2f_weights.shape
        == action_mask.shape
        == content_mask.shape
    ):
        raise ValueError("token loss tensors must have identical shapes")
    action_mask = action_mask.bool()
    content_mask = content_mask.bool()
    if bool((action_mask & content_mask).any()):
        raise ValueError("action and content CE masks must be disjoint")
    denominator = d2f_weights.sum()
    if not bool(denominator > 0):
        raise ValueError("D2F random-mask weights must have positive mass")
    token_loss = distill_weight * distill + hard_ce_weight * hard_ce
    d2f_loss = (token_loss * d2f_weights).sum() / denominator
    weighted_distill = (distill * d2f_weights).sum() / denominator
    weighted_hard_ce = (hard_ce * d2f_weights).sum() / denominator
    if bool(action_mask.any()):
        action_ce = hard_ce[action_mask].mean()
        balanced_action_ce = action_ce * action_class_weight.float().reshape(())
    else:
        if action_ce_weight:
            raise ValueError("action CE is enabled but the batch has no action tokens")
        action_ce = hard_ce.new_zeros(())
        balanced_action_ce = hard_ce.new_zeros(())
    if bool(content_mask.any()):
        content_ce = hard_ce[content_mask].mean()
        if content_ce_use_action_class_weight:
            balanced_content_ce = (
                content_ce * action_class_weight.float().reshape(())
            )
        else:
            # Open-vocabulary payload tokens should not be downweighted merely
            # because their enclosing action class (commonly CLICK) is frequent.
            balanced_content_ce = content_ce
    else:
        # Some actions intentionally have no payload (for example BACK/HOME).
        # Enabling content CE must therefore remain valid for contentless
        # samples and contribute an exact zero.
        content_ce = hard_ce.new_zeros(())
        balanced_content_ce = hard_ce.new_zeros(())
    return {
        "loss": (
            d2f_loss
            + action_ce_weight * balanced_action_ce
            + content_ce_weight * balanced_content_ce
        ),
        "d2f_loss": d2f_loss,
        "distill_loss": weighted_distill,
        "hard_ce_loss": weighted_hard_ce,
        "action_ce_loss": action_ce,
        "balanced_action_ce_loss": balanced_action_ce,
        "content_ce_loss": content_ce,
        "balanced_content_ce_loss": balanced_content_ce,
    }


def full_response_reconstruction_metrics(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_mask: torch.Tensor,
    group_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Count token and whole-response reconstruction for full-mask examples."""
    if logits.ndim != 2 or labels.shape != logits.shape[:1]:
        raise ValueError("full-response logits and labels have incompatible shapes")
    if token_mask.shape != labels.shape or group_ids.shape != labels.shape:
        raise ValueError("full-response token metadata must match labels")
    token_mask = token_mask.bool()
    selected_group_ids = group_ids[token_mask].long()
    if bool((selected_group_ids < 0).any()):
        raise ValueError("full-response tokens must have non-negative group IDs")
    token_correct = logits.argmax(dim=-1).eq(labels)
    selected_correct = token_correct[token_mask]
    groups = torch.unique(selected_group_ids, sorted=True)
    if len(groups):
        exact = torch.stack(
            [selected_correct[selected_group_ids == group].all() for group in groups]
        ).sum()
    else:
        exact = labels.new_zeros(())
    return {
        "full_response_token_correct": selected_correct.sum(),
        "full_response_token_count": token_mask.sum(),
        "full_response_exact": exact,
        "full_response_count": labels.new_tensor(len(groups)),
    }


def teacher_distillation_loss(
    model,
    packed_sequence: torch.Tensor,
    batch: dict[str, Any],
    student_log_probabilities: torch.Tensor,
    *,
    num_heads: int,
    distill_weight: float,
) -> torch.Tensor:
    """Return tokenwise distillation loss without invoking a zero-weight teacher."""
    if distill_weight == 0.0:
        return student_log_probabilities.new_zeros(
            student_log_probabilities.shape[0]
        )
    teacher_mask = create_full_document_mask(
        batch["sample_lens"],
        num_heads=num_heads,
        device=packed_sequence.device,
    )
    with torch.no_grad(), adapter_disabled(model):
        teacher_logits = forward_masked_logits(
            model,
            packed_sequence,
            batch,
            teacher_mask,
        )
        teacher_probabilities = torch.softmax(teacher_logits.float(), dim=-1)
    return -(teacher_probabilities * student_log_probabilities).sum(dim=-1)


class LLaDAOGuiD2FModel(nn.Module):
    """Distributed-safe D2F wrapper around a PEFT or full LLaDA-o model."""

    def __init__(
        self,
        model,
        *,
        mask_id: int = 126336,
        block_size: int = 16,
        distill_weight: float = 1.0,
        hard_ce_weight: float = 0.1,
        action_ce_weight: float = 0.0,
        content_ce_weight: float = 0.0,
        full_response_mask_probability: float = 0.0,
        content_ce_use_action_class_weight: bool = True,
    ):
        super().__init__()
        self.model = model
        self.mask_id = mask_id
        self.block_size = block_size
        self.distill_weight = distill_weight
        self.hard_ce_weight = hard_ce_weight
        self.action_ce_weight = action_ce_weight
        self.content_ce_weight = content_ce_weight
        self.full_response_mask_probability = full_response_mask_probability
        self.content_ce_use_action_class_weight = content_ce_use_action_class_weight

    @property
    def peft_model(self):
        """Compatibility alias for callers written before full-model training."""

        return self.model

    def forward(self, raw_batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        batch = rebuild_and_corrupt_responses(
            raw_batch,
            mask_id=self.mask_id,
            block_size=self.block_size,
            full_response_mask_probability=self.full_response_mask_probability,
        )
        base = unwrap_lladao(self.model)
        packed_sequence = prepare_understanding_sequence(self.model, batch)
        student_mask = create_training_block_mask(
            batch["sample_lens"],
            batch["d2f_response_spans"],
            self.block_size,
            prefix_segments=batch.get("d2f_prefix_segments"),
            num_heads=base.num_heads,
            device=packed_sequence.device,
        )
        student_logits = forward_masked_logits(
            self.model, packed_sequence, batch, student_mask
        )
        student_log_probabilities = torch.log_softmax(student_logits.float(), dim=-1)
        distill = teacher_distillation_loss(
            self.model,
            packed_sequence,
            batch,
            student_log_probabilities,
            num_heads=base.num_heads,
            distill_weight=self.distill_weight,
        )
        hard_ce = torch.nn.functional.cross_entropy(
            student_logits.float(), batch["packed_label_ids"].long(), reduction="none"
        )
        weights = batch["ce_loss_weights"].float()
        action_mask = batch["action_ce_mask"].bool()
        content_mask = batch["content_ce_mask"].bool()
        class_weight = batch.get("action_class_weight")
        if class_weight is None:
            class_weight = hard_ce.new_ones(())
        metrics = combine_d2f_and_action_losses(
            distill=distill,
            hard_ce=hard_ce,
            d2f_weights=weights,
            action_mask=action_mask,
            distill_weight=self.distill_weight,
            hard_ce_weight=self.hard_ce_weight,
            action_ce_weight=self.action_ce_weight,
            action_class_weight=class_weight,
            content_mask=content_mask,
            content_ce_weight=self.content_ce_weight,
            content_ce_use_action_class_weight=(
                self.content_ce_use_action_class_weight
            ),
        )
        metrics.update(
            full_response_reconstruction_metrics(
                logits=student_logits.detach(),
                labels=batch["packed_label_ids"].long(),
                token_mask=batch["full_response_ce_mask"],
                group_ids=batch["full_response_group_ids"],
            )
        )
        response_count = batch["d2f_response_count"]
        full_response_count = batch["full_response_masked_count"]
        metrics.update({
            "masked_tokens": torch.tensor(
                len(batch["packed_label_ids"]), device=hard_ce.device, dtype=torch.long
            ),
            "d2f_random_masked_tokens": (weights > 0).sum(),
            "action_tokens": action_mask.sum(),
            "content_tokens": content_mask.sum(),
            "action_class_weight": class_weight.float().reshape(()),
            "full_response_masked_count": full_response_count,
            "d2f_response_count": response_count,
            "full_response_masked_rate": (
                full_response_count.float() / response_count.clamp_min(1).float()
            ),
        })
        return metrics
