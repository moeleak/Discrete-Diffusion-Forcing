from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lladao_d2f.residual_grounding import (
    ResidualGroundingContractError,
    audit_understanding_checkpoint,
    distillation_token_mask,
    domain_for_microstep,
    load_adapter_contract,
    validate_domain_schedule,
)


def test_distillation_policy_expands_across_packed_samples() -> None:
    mask = distillation_token_mask(
        sample_lens=[3, 4, 2],
        target_indexes=torch.tensor([1, 3, 6, 8]),
        sample_mask=torch.tensor([True, False, True]),
    )

    assert mask.tolist() == [True, False, False, True]


def test_domain_schedule_is_exactly_alternating_and_even() -> None:
    schedule = validate_domain_schedule(
        [("mind2web", True), ("mobile", False)], 8
    )

    assert [domain_for_microstep(index, schedule) for index in range(4)] == [
        ("mind2web", True),
        ("mobile", False),
        ("mind2web", True),
        ("mobile", False),
    ]
    with pytest.raises(ResidualGroundingContractError, match="positive even"):
        validate_domain_schedule(schedule, 3)
    with pytest.raises(ResidualGroundingContractError, match="domain order"):
        validate_domain_schedule(
            [("mobile", False), ("mind2web", True)], 8
        )


def adapter_contract_fixture(backbone_sha: str, *, step: int = 10) -> dict:
    return {
        "schema_version": 1,
        "format": "lladao-residual-grounding-lora-v1",
        "backbone": {"sha256": backbone_sha},
        "adapter": {
            "module_count": 128,
            "rank": 32,
            "alpha": 32,
            "dropout": 0.1,
            "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "zero_delta": True,
        },
        "training": {
            "step": step,
            "max_steps": 10,
            "domain_microbatches": {"mind2web": 80, "mobile": 80},
            "domain_mix": {"mind2web": 0.5, "mobile": 0.5},
            "release_eligible": True,
        },
    }


def test_completed_adapter_contract_is_bound_to_one_backbone(tmp_path: Path) -> None:
    expected = "a" * 64
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "training_contract.json").write_text(
        json.dumps(adapter_contract_fixture(expected)), encoding="utf-8"
    )

    value = load_adapter_contract(
        adapter, expected_backbone_sha256=expected, require_complete=True
    )
    assert value["backbone"]["sha256"] == expected
    with pytest.raises(ResidualGroundingContractError, match="not b+"):
        load_adapter_contract(adapter, expected_backbone_sha256="b" * 64)

    smoke = adapter_contract_fixture(expected, step=1)
    smoke["training"]["release_eligible"] = False
    (adapter / "training_contract.json").write_text(json.dumps(smoke), encoding="utf-8")
    with pytest.raises(ResidualGroundingContractError, match="not a release-eligible"):
        load_adapter_contract(
            adapter, expected_backbone_sha256=expected, require_complete=True
        )


def test_checkpoint_audit_checks_exact_sha_dtype_and_keys(tmp_path: Path) -> None:
    checkpoint = tmp_path / "planner.safetensors"
    save_file({"model.weight": torch.zeros(2, 3, dtype=torch.bfloat16)}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    audit = audit_understanding_checkpoint(
        checkpoint, expected_sha256=digest, expected_parameters=6
    )
    assert audit["parameter_count"] == 6
    assert audit["dtype"] == "bfloat16"

    with pytest.raises(ResidualGroundingContractError, match="SHA-256 mismatch"):
        audit_understanding_checkpoint(
            checkpoint, expected_sha256="0" * 64, expected_parameters=6
        )

    lora_checkpoint = tmp_path / "bad.safetensors"
    save_file(
        {"model.lora_A.weight": torch.zeros(1, dtype=torch.bfloat16)},
        lora_checkpoint,
    )
    with pytest.raises(ResidualGroundingContractError, match="already-active LoRA"):
        audit_understanding_checkpoint(
            lora_checkpoint,
            expected_sha256=hashlib.sha256(lora_checkpoint.read_bytes()).hexdigest(),
            expected_parameters=1,
        )
