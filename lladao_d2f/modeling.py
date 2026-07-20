from __future__ import annotations

import importlib
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
    from safetensors.torch import load_model
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
    # Construct directly in the checkpoint dtype. Building both DDP replicas
    # in FP32 first peaks above the 128 GB host-memory limit before either
    # process can move its model to the GPU.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        language_model = LLaDAModelLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model = LLaDAO(language_model, vit_model, None, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    missing, unexpected = load_model(model, str(checkpoint_path), strict=True, device="cpu")
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
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


@torch.no_grad()
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
    hidden, _ = base.language_model(
        packed_sequence=packed_sequence,
        sample_lens=batch["sample_lens"],
        attention_mask=attention_mask,
        packed_position_ids=batch["packed_position_ids"],
        packed_und_token_indexes=understanding_indexes,
        packed_gen_token_indexes=empty_generation_indexes,
    )
    return base.language_model.lm_head(hidden[batch["ce_loss_indexes"].long()])


def adapter_disabled(model):
    target = model.module if hasattr(model, "module") else model
    return target.disable_adapter() if hasattr(target, "disable_adapter") else nullcontext()


class LLaDAOGuiD2FModel(nn.Module):
    """DDP-safe teacher/student wrapper around a single PEFT LLaDA-o model."""

    def __init__(
        self,
        peft_model,
        *,
        mask_id: int = 126336,
        block_size: int = 16,
        distill_weight: float = 1.0,
        hard_ce_weight: float = 0.1,
    ):
        super().__init__()
        self.peft_model = peft_model
        self.mask_id = mask_id
        self.block_size = block_size
        self.distill_weight = distill_weight
        self.hard_ce_weight = hard_ce_weight

    def forward(self, raw_batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        batch = rebuild_and_corrupt_responses(
            raw_batch,
            mask_id=self.mask_id,
            block_size=self.block_size,
        )
        base = unwrap_lladao(self.peft_model)
        packed_sequence = prepare_understanding_sequence(self.peft_model, batch)
        student_mask = create_training_block_mask(
            batch["sample_lens"],
            batch["d2f_response_spans"],
            self.block_size,
            num_heads=base.num_heads,
            device=packed_sequence.device,
        )
        teacher_mask = create_full_document_mask(
            batch["sample_lens"],
            num_heads=base.num_heads,
            device=packed_sequence.device,
        )
        with torch.no_grad(), adapter_disabled(self.peft_model):
            teacher_logits = forward_masked_logits(
                self.peft_model, packed_sequence, batch, teacher_mask
            )
            teacher_probabilities = torch.softmax(teacher_logits.float(), dim=-1)
            del teacher_logits
        student_logits = forward_masked_logits(
            self.peft_model, packed_sequence, batch, student_mask
        )
        student_log_probabilities = torch.log_softmax(student_logits.float(), dim=-1)
        distill = -(teacher_probabilities * student_log_probabilities).sum(dim=-1)
        hard_ce = torch.nn.functional.cross_entropy(
            student_logits.float(), batch["packed_label_ids"].long(), reduction="none"
        )
        weights = batch["ce_loss_weights"].float()
        token_loss = self.distill_weight * distill + self.hard_ce_weight * hard_ce
        loss = (token_loss * weights).sum() / weights.sum().clamp_min(1e-8)
        return {
            "loss": loss,
            "distill_loss": (distill * weights).sum() / weights.sum().clamp_min(1e-8),
            "hard_ce_loss": (hard_ce * weights).sum() / weights.sum().clamp_min(1e-8),
            "masked_tokens": torch.tensor(
                len(batch["packed_label_ids"]), device=loss.device, dtype=torch.long
            ),
        }
