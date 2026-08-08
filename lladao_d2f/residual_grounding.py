"""Contracts for retraining a residual grounder on a full Planner backbone."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_ACTIVE_PARAMETERS = 8_459_716_512
EXPECTED_LORA_MODULES = 128
GROUNDING_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class ResidualGroundingContractError(ValueError):
    """Raised when a backbone, adapter, or two-domain recipe is ambiguous."""


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(sample_ids: Iterable[str]) -> str:
    payload = "".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ResidualGroundingContractError(f"{name} must be one lowercase SHA-256")
    return value


def audit_understanding_checkpoint(
    checkpoint: str | Path,
    *,
    expected_sha256: str,
    expected_parameters: int = EXPECTED_ACTIVE_PARAMETERS,
) -> dict[str, Any]:
    """Strictly bind training to one standalone full-parameter Planner file."""

    from safetensors import safe_open

    resolved = Path(checkpoint).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"Planner checkpoint is missing: {resolved}")
    expected_digest = validate_sha256(
        expected_sha256, name="model.expected_checkpoint_sha256"
    )
    actual_digest = sha256_file(resolved)
    if actual_digest != expected_digest:
        raise ResidualGroundingContractError(
            "Planner checkpoint SHA-256 mismatch: "
            f"expected={expected_digest} actual={actual_digest} path={resolved}"
        )

    parameter_count = 0
    dtypes: set[str] = set()
    keys: list[str] = []
    with safe_open(str(resolved), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            tensor_slice = handle.get_slice(key)
            parameter_count += math.prod(tensor_slice.get_shape())
            dtype = str(tensor_slice.get_dtype()).casefold()
            dtypes.add({"bf16": "bfloat16"}.get(dtype, dtype))
    if any("_moe_gen" in key for key in keys):
        raise ResidualGroundingContractError(
            "residual grounding requires an understanding-only Planner checkpoint"
        )
    if any("lora_" in key.casefold() for key in keys):
        raise ResidualGroundingContractError(
            "Planner checkpoint must not contain an already-active LoRA"
        )
    if parameter_count != int(expected_parameters):
        raise ResidualGroundingContractError(
            f"Planner parameter count is {parameter_count:,}, expected "
            f"{int(expected_parameters):,}"
        )
    if dtypes != {"bfloat16"}:
        raise ResidualGroundingContractError(
            f"Planner checkpoint must be uniformly bfloat16, found {sorted(dtypes)}"
        )
    return {
        "path": str(resolved),
        "sha256": actual_digest,
        "size_bytes": resolved.stat().st_size,
        "tensor_count": len(keys),
        "parameter_count": parameter_count,
        "dtype": "bfloat16",
        "contains_generation_experts": False,
        "contains_lora": False,
        "format": "understanding-only-full-safetensors",
    }


def audit_zero_initialized_lora(model: torch.nn.Module) -> dict[str, Any]:
    """Verify PEFT's zero-delta initialization and frozen shared backbone."""

    modules: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        if not re.search(
            r"self_attn\.(q_proj|k_proj|v_proj|o_proj)$", name
        ):
            raise ResidualGroundingContractError(
                f"LoRA module escaped q/k/v/o attention: {name}"
            )
        modules.append((name, module))
    if len(modules) != EXPECTED_LORA_MODULES:
        raise ResidualGroundingContractError(
            f"expected {EXPECTED_LORA_MODULES} grounding LoRA modules, "
            f"found {len(modules)}"
        )

    nonzero_b: list[str] = []
    ranks: set[int] = set()
    for name, module in modules:
        if set(module.lora_A) != {"default"} or set(module.lora_B) != {"default"}:
            raise ResidualGroundingContractError(
                f"unexpected PEFT adapter names in {name}"
            )
        a_weight = module.lora_A["default"].weight.detach()
        b_weight = module.lora_B["default"].weight.detach()
        ranks.add(int(a_weight.shape[0]))
        if bool(torch.count_nonzero(b_weight)):
            nonzero_b.append(name)
    if nonzero_b:
        raise ResidualGroundingContractError(
            "new residual adapter has a non-zero initial delta: "
            + ", ".join(nonzero_b[:4])
        )
    if ranks != {32}:
        raise ResidualGroundingContractError(f"grounding LoRA rank drift: {ranks}")

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    illegal = [name for name in trainable if "lora_" not in name]
    if illegal:
        raise ResidualGroundingContractError(
            "shared Planner backbone is not completely frozen: "
            + ", ".join(illegal[:4])
        )
    frozen_parameters = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "module_count": len(modules),
        "rank": 32,
        "alpha": 32,
        "dropout": 0.1,
        "targets": list(GROUNDING_LORA_TARGETS),
        "zero_delta": True,
        "frozen_backbone_parameters": frozen_parameters,
        "trainable_lora_parameters": trainable_parameters,
        "trainable_parameter_tensors": len(trainable),
    }


def validate_domain_schedule(
    domains: Sequence[tuple[str, bool]], gradient_accumulation_steps: int
) -> tuple[tuple[str, bool], ...]:
    """Require exact 50:50 optimizer-step composition for two domains."""

    normalized = tuple((str(name), bool(distill)) for name, distill in domains)
    if normalized != (("mind2web", True), ("mobile", False)):
        raise ResidualGroundingContractError(
            "domain order must be mind2web(distill) then mobile(hard-CE-only)"
        )
    accumulation = int(gradient_accumulation_steps)
    if accumulation <= 0 or accumulation % 2:
        raise ResidualGroundingContractError(
            "gradient accumulation must be a positive even number for exact 50:50 mixing"
        )
    return normalized


def domain_for_microstep(
    microstep: int, domains: Sequence[tuple[str, bool]]
) -> tuple[str, bool]:
    if microstep < 0:
        raise ResidualGroundingContractError("microstep must be non-negative")
    if not domains:
        raise ResidualGroundingContractError("domain schedule is empty")
    return tuple(domains)[microstep % len(domains)]


def distillation_token_mask(
    *,
    sample_lens: Sequence[int],
    target_indexes: torch.Tensor,
    sample_mask: torch.Tensor | Sequence[bool] | None,
) -> torch.Tensor:
    """Expand a per-packed-sample teacher policy to supervised token positions."""

    if sample_mask is None:
        return torch.ones_like(target_indexes, dtype=torch.bool)
    mask = torch.as_tensor(sample_mask, dtype=torch.bool, device=target_indexes.device)
    lengths = torch.as_tensor(sample_lens, dtype=torch.long, device=target_indexes.device)
    if lengths.ndim != 1 or bool((lengths < 0).any()):
        raise ResidualGroundingContractError("sample lengths must be non-negative")
    if mask.ndim != 1 or len(mask) != len(lengths):
        raise ResidualGroundingContractError(
            "distill sample mask must match packed sample lengths"
        )
    if target_indexes.ndim != 1:
        raise ResidualGroundingContractError("target indexes must be one vector")
    ends = torch.cumsum(lengths, dim=0)
    if len(target_indexes) and (
        bool((target_indexes < 0).any())
        or not len(ends)
        or bool((target_indexes >= ends[-1]).any())
    ):
        raise ResidualGroundingContractError(
            "supervised token index falls outside packed samples"
        )
    sample_indexes = torch.searchsorted(ends, target_indexes, right=True)
    return mask[sample_indexes]


def adapter_contract(
    *,
    backbone_audit: dict[str, Any],
    lora_audit: dict[str, Any],
    step: int,
    max_steps: int,
    domain_counts: dict[str, int],
    config_sha256: str,
    release_eligible: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "format": "lladao-residual-grounding-lora-v1",
        "backbone": dict(backbone_audit),
        "adapter": dict(lora_audit),
        "training": {
            "step": int(step),
            "max_steps": int(max_steps),
            "domain_microbatches": dict(sorted(domain_counts.items())),
            "domain_mix": {"mind2web": 0.5, "mobile": 0.5},
            "mind2web_objective": "d2f_distillation_plus_hard_ce",
            "mobile_objective": "hard_ce_only",
            "config_sha256": validate_sha256(config_sha256, name="config_sha256"),
            "release_eligible": bool(release_eligible),
        },
    }


def write_json_atomic(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_adapter_contract(
    adapter: str | Path,
    *,
    expected_backbone_sha256: str,
    require_complete: bool = False,
) -> dict[str, Any]:
    root = Path(adapter).expanduser().resolve()
    path = root / "training_contract.json"
    if not path.is_file():
        raise ResidualGroundingContractError(
            f"residual adapter has no training contract: {path}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != "lladao-residual-grounding-lora-v1":
        raise ResidualGroundingContractError("unsupported residual adapter contract")
    actual = (value.get("backbone") or {}).get("sha256")
    expected = validate_sha256(
        expected_backbone_sha256, name="expected_backbone_sha256"
    )
    if actual != expected:
        raise ResidualGroundingContractError(
            f"adapter was trained for backbone {actual}, not {expected}"
        )
    adapter_audit = value.get("adapter") or {}
    if (
        adapter_audit.get("module_count") != EXPECTED_LORA_MODULES
        or adapter_audit.get("rank") != 32
        or adapter_audit.get("alpha") != 32
        or adapter_audit.get("dropout") != 0.1
        or adapter_audit.get("targets") != list(GROUNDING_LORA_TARGETS)
        or adapter_audit.get("zero_delta") is not True
    ):
        raise ResidualGroundingContractError(
            "residual adapter architecture does not match r32/alpha32/dropout0.1 qkvo"
        )
    training = value.get("training") or {}
    counts = training.get("domain_microbatches") or {}
    if (
        training.get("domain_mix") != {"mind2web": 0.5, "mobile": 0.5}
        or set(counts) != {"mind2web", "mobile"}
        or int(counts.get("mind2web", 0)) != int(counts.get("mobile", -1))
    ):
        raise ResidualGroundingContractError(
            "residual adapter contract does not prove an exact 50:50 domain mix"
        )
    step = int(training.get("step", -1))
    max_steps = int(training.get("max_steps", -2))
    if require_complete and (
        not 0 < step <= max_steps
        or int(counts.get("mind2web", 0)) <= 0
        or training.get("release_eligible") is not True
    ):
        raise ResidualGroundingContractError(
            "adapter is not a release-eligible residual grounding epoch"
        )
    return value


__all__ = [
    "EXPECTED_ACTIVE_PARAMETERS",
    "ResidualGroundingContractError",
    "adapter_contract",
    "audit_understanding_checkpoint",
    "audit_zero_initialized_lora",
    "distillation_token_mask",
    "domain_for_microstep",
    "load_adapter_contract",
    "ordered_ids_sha256",
    "sha256_file",
    "validate_domain_schedule",
    "write_json_atomic",
]
