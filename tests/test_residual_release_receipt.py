from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "D2F-eval/build_residual_release_receipt.py"
SPEC = importlib.util.spec_from_file_location("build_residual_release_receipt", SCRIPT)
RECEIPT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = RECEIPT
SPEC.loader.exec_module(RECEIPT)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_root = tmp_path / "run"
    adapter = run_root / "step-0000010/adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    write_json(
        adapter / "adapter_config.json",
        {
            "r": 32,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "bias": "none",
            "peft_type": "LORA",
            "use_dora": False,
            "use_rslora": False,
            "modules_to_save": None,
            "target_modules": RECEIPT.EXPECTED_TARGET_MODULES,
        },
    )
    backbone = "a" * 64
    contract = {
        "schema_version": 1,
        "format": "lladao-residual-grounding-lora-v1",
        "backbone": {
            "sha256": backbone,
            "parameter_count": 8_459_716_512,
            "dtype": "bfloat16",
            "contains_generation_experts": False,
            "contains_lora": False,
            "format": "understanding-only-full-safetensors",
        },
        "adapter": {
            "module_count": 128,
            "rank": 32,
            "alpha": 32,
            "dropout": 0.1,
            "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "zero_delta": True,
            "frozen_backbone_parameters": 8_459_716_512,
            "trainable_lora_parameters": 33_554_432,
            "trainable_parameter_tensors": 256,
        },
        "training": {
            "step": 10,
            "max_steps": 30,
            "domain_microbatches": {"mind2web": 80, "mobile": 80},
            "domain_mix": {"mind2web": 0.5, "mobile": 0.5},
            "mind2web_objective": "d2f_distillation_plus_hard_ce",
            "mobile_objective": "hard_ce_only",
            "config_sha256": "b" * 64,
            "release_eligible": True,
        },
    }
    write_json(adapter / "training_contract.json", contract)
    selection_path = run_root / "benchmark/checkpoint-selection.json"
    candidates = [
        {
            "epoch": 1,
            "step": 10,
            "adapter": str(adapter),
            "mind2web_validation_ssr": 0.8,
            "mind2web_validation_joint_ssr": 0.75,
            "mobile_validation_ssr": 0.7,
            "mobile_validation_joint_ssr": 0.65,
            "worst_domain_ssr": 0.7,
            "mean_domain_ssr": 0.75,
            "worst_domain_joint_ssr": 0.65,
        },
        {
            "epoch": 2,
            "step": 20,
            "adapter": str(run_root / "step-0000020/adapter"),
            "mind2web_validation_ssr": 0.6,
            "mind2web_validation_joint_ssr": 0.58,
            "mobile_validation_ssr": 0.8,
            "mobile_validation_joint_ssr": 0.7,
            "worst_domain_ssr": 0.6,
            "mean_domain_ssr": 0.7,
            "worst_domain_joint_ssr": 0.58,
        },
        {
            "epoch": 3,
            "step": 30,
            "adapter": str(run_root / "step-0000030/adapter"),
            "mind2web_validation_ssr": 0.5,
            "mind2web_validation_joint_ssr": 0.45,
            "mobile_validation_ssr": 0.5,
            "mobile_validation_joint_ssr": 0.48,
            "worst_domain_ssr": 0.5,
            "mean_domain_ssr": 0.5,
            "worst_domain_joint_ssr": 0.45,
        },
    ]
    write_json(
        selection_path,
        {
            "schema_version": 1,
            "selection_metric": RECEIPT.EXPECTED_SELECTION_METRIC,
            "tie_breakers": RECEIPT.EXPECTED_SELECTION_TIE_BREAKERS,
            "test_data_used_for_selection": False,
            "candidates": candidates,
            "selected": candidates[0],
        },
    )

    runs = []
    for label, benchmark in (
        ("mind2web-test", "mind2web"),
        ("mobile-test", "mobile_test"),
    ):
        root = run_root / "benchmark" / label
        benchmark_root = run_root / "benchmark-data" / label
        write_json(benchmark_root / "manifest.json", {"benchmarks": {benchmark: {}}})
        run_config = root / "run-config-rank-00000.json"
        scores = root / "scores/results.json"
        write_json(
            run_config,
            {
                **RECEIPT.EXPECTED_RUN_CONFIG,
                "checkpoint_sha256": backbone,
                "expected_checkpoint_sha256": backbone,
                "adapter": str(adapter),
                "residual_adapter_contract": contract,
                "limit_per_benchmark": 100,
                "benchmarks": [benchmark],
                "benchmark_root": str(benchmark_root),
            },
        )
        write_json(
            scores,
            {
                "benchmark_manifest": str(benchmark_root / "manifest.json"),
                "benchmarks": {
                    benchmark: {
                        "num_samples": 100,
                        "ssr_point_only": 0.8,
                        "joint_step_success": 0.75,
                        "action_f1_macro_present": 1.0,
                        "parse_rate": 1.0,
                        "latency_seconds": {"mean": 1.25},
                    }
                },
                "coverage": {
                    benchmark: {
                        "targets": 100,
                        "predictions": 100,
                        "joined": 100,
                        "missing": 0,
                    }
                },
            },
        )
        runs.append(
            {
                "label": label,
                "benchmark": benchmark,
                "run_config": str(run_config),
                "scores": str(scores),
            }
        )
    index_path = run_root / "benchmark/index.json"
    write_json(index_path, {"backbone_sha256": backbone, "runs": runs})
    return index_path, selection_path, adapter


def test_release_receipt_binds_selected_epoch_and_test_100(tmp_path: Path) -> None:
    index, selection, adapter = fixture(tmp_path)

    receipt = RECEIPT.build_release_receipt(index, selection)

    assert receipt["format"] == "lladao-residual-grounding-release-v1"
    assert receipt["status"] == "benchmark-complete"
    assert receipt["release_eligible"] is True
    assert receipt["selected_adapter"]["path"] == str(adapter.resolve())
    assert receipt["selected_adapter"]["step"] == 10
    assert set(receipt["benchmarks"]) == {"mind2web-test", "mobile-test"}
    assert all(row["num_samples"] == 100 for row in receipt["benchmarks"].values())


def test_release_receipt_rejects_nonselected_test_adapter(tmp_path: Path) -> None:
    index, selection, _ = fixture(tmp_path)
    value = json.loads(index.read_text(encoding="utf-8"))
    run_config = Path(value["runs"][0]["run_config"])
    config = json.loads(run_config.read_text(encoding="utf-8"))
    config["adapter"] = str(tmp_path / "wrong-adapter")
    write_json(run_config, config)

    with pytest.raises(RECEIPT.ResidualReleaseReceiptError, match="non-selected"):
        RECEIPT.build_release_receipt(index, selection)


def test_release_receipt_rejects_partial_test(tmp_path: Path) -> None:
    index, selection, _ = fixture(tmp_path)
    value = json.loads(index.read_text(encoding="utf-8"))
    scores = Path(value["runs"][1]["scores"])
    payload = json.loads(scores.read_text(encoding="utf-8"))
    payload["benchmarks"]["mobile_test"]["num_samples"] = 99
    write_json(scores, payload)

    with pytest.raises(RECEIPT.ResidualReleaseReceiptError, match="exactly 100"):
        RECEIPT.build_release_receipt(index, selection)


def test_release_receipt_rejects_test_based_selection(tmp_path: Path) -> None:
    index, selection, _ = fixture(tmp_path)
    value = json.loads(selection.read_text(encoding="utf-8"))
    value["test_data_used_for_selection"] = True
    write_json(selection, value)

    with pytest.raises(RECEIPT.ResidualReleaseReceiptError, match="test labels"):
        RECEIPT.build_release_receipt(index, selection)


def test_release_receipt_rejects_adapter_architecture_drift(tmp_path: Path) -> None:
    index, selection, adapter = fixture(tmp_path)
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["target_modules"] = "q_proj"
    write_json(config_path, config)

    with pytest.raises(RECEIPT.ResidualReleaseReceiptError, match="q/k/v/o"):
        RECEIPT.build_release_receipt(index, selection)


def test_release_receipt_rejects_forged_validation_selection(tmp_path: Path) -> None:
    index, selection, _ = fixture(tmp_path)
    value = json.loads(selection.read_text(encoding="utf-8"))
    value["selected"] = value["candidates"][1]
    write_json(selection, value)

    with pytest.raises(RECEIPT.ResidualReleaseReceiptError, match="does not maximize"):
        RECEIPT.build_release_receipt(index, selection)


def test_release_receipt_rejects_nonfinite_metric(tmp_path: Path) -> None:
    index, selection, _ = fixture(tmp_path)
    value = json.loads(index.read_text(encoding="utf-8"))
    scores = Path(value["runs"][0]["scores"])
    payload = json.loads(scores.read_text(encoding="utf-8"))
    payload["benchmarks"]["mind2web"]["ssr_point_only"] = float("nan")
    write_json(scores, payload)

    with pytest.raises(RECEIPT.ResidualReleaseReceiptError, match="must be finite"):
        RECEIPT.build_release_receipt(index, selection)


def test_release_receipt_rejects_decoding_protocol_drift(tmp_path: Path) -> None:
    index, selection, _ = fixture(tmp_path)
    value = json.loads(index.read_text(encoding="utf-8"))
    run_config = Path(value["runs"][0]["run_config"])
    config = json.loads(run_config.read_text(encoding="utf-8"))
    config["block_size"] = 64
    write_json(run_config, config)

    with pytest.raises(RECEIPT.ResidualReleaseReceiptError, match="decoding protocol"):
        RECEIPT.build_release_receipt(index, selection)


def test_release_receipt_rejects_negative_latency(tmp_path: Path) -> None:
    index, selection, _ = fixture(tmp_path)
    value = json.loads(index.read_text(encoding="utf-8"))
    scores = Path(value["runs"][0]["scores"])
    payload = json.loads(scores.read_text(encoding="utf-8"))
    payload["benchmarks"]["mind2web"]["latency_seconds"]["mean"] = -1.0
    write_json(scores, payload)

    with pytest.raises(RECEIPT.ResidualReleaseReceiptError, match="cannot be negative"):
        RECEIPT.build_release_receipt(index, selection)
