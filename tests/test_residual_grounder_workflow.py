from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts/run_mllm_lladao_residual_grounder.sh"
TRAINER = ROOT / "D2F-train/train_lladao_gui.py"
SELECTOR_PATH = ROOT / "D2F-eval/select_residual_grounder.py"
SELECTOR_SPEC = importlib.util.spec_from_file_location(
    "select_residual_grounder", SELECTOR_PATH
)
SELECTOR = importlib.util.module_from_spec(SELECTOR_SPEC)
assert SELECTOR_SPEC.loader is not None
sys.modules[SELECTOR_SPEC.name] = SELECTOR
SELECTOR_SPEC.loader.exec_module(SELECTOR)
RECEIPT_PATH = ROOT / "D2F-eval/build_residual_release_receipt.py"
SUMMARY_PATH = ROOT / "D2F-eval/summarize_residual_grounder.py"
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_residual_grounder", SUMMARY_PATH
)
SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
assert SUMMARY_SPEC.loader is not None
sys.modules[SUMMARY_SPEC.name] = SUMMARY
SUMMARY_SPEC.loader.exec_module(SUMMARY)


def test_launcher_is_one_log_strict_final_planner_and_2_or_8_gpu() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    source = LAUNCHER.read_text(encoding="utf-8")

    assert RECEIPT_PATH.is_file()
    assert 'exec >>"${LOG_FILE}" 2>&1' in source
    assert "FINAL_PLANNER_RESULT" in source
    assert 'result.get("status") != "passed"' in source
    assert "RESIDUAL_SMOKE_ONLY" in source
    assert "release residual training requires exactly 3 epochs" in source
    assert "FINAL_PLANNER_CHECKPOINT" in source
    assert "FINAL_PLANNER_SHA256" in source
    assert "MAX_STEPS=1" in source
    assert "not release eligible" in source
    assert "latest_checkpoint" not in source
    assert "CUDA_VISIBLE_DEVICES" not in source
    assert "2) GRADIENT_ACCUMULATION_STEPS=8" in source
    assert "8) GRADIENT_ACCUMULATION_STEPS=2" in source
    assert source.count("--limit 100") >= 1
    assert "--require-residual-adapter-contract" in source
    assert "--no-kv-cache-compression" in source
    assert "select_residual_grounder.py" in source
    assert "build_residual_release_receipt.py" in source
    assert "release-receipt.json" in source
    assert "Mind2Web validation-100 overlaps Mind2Web test-100" in source


def test_residual_recipe_is_fixed_r32_and_exact_two_domain_objectives() -> None:
    config = yaml.safe_load(
        (ROOT / "D2F-train/config/lladao_gui_residual.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["lora"] == {"rank": 32, "alpha": 32, "dropout": 0.1}
    assert config["train"]["global_batch_size"] == 16
    assert config["train"]["release_eligible"] is True
    assert config["train"]["hard_ce_weight"] == 1.0
    assert config["data"]["domains"]["mind2web"]["distill"] is True
    assert config["data"]["domains"]["mobile"]["distill"] is False

    source = TRAINER.read_text(encoding="utf-8")
    assert 'batch["distill_sample_mask"] = torch.full' in source
    assert "domain_for_microstep" in source
    assert "OptimizerStepMetricAccumulator" in source
    assert 'reduced[f"{domain_name}_{key}"]' in source
    assert "save_embedding_layers=False" in source
    assert "accelerator.end_training()" in source
    assert "audit_understanding_checkpoint" in source
    assert "audit_zero_initialized_lora" in source


def test_checkpoint_selection_maximizes_worst_validation_domain(
    tmp_path: Path,
) -> None:
    def scores(path: Path, benchmark: str, ssr: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "benchmarks": {
                        benchmark: {
                            "num_samples": 100,
                            "ssr_point_only": ssr,
                            "joint_step_success": ssr,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    for epoch, mind2web, mobile in ((1, 0.90, 0.50), (2, 0.70, 0.68)):
        root = tmp_path / "benchmark/validation" / f"epoch-{epoch:02d}"
        scores(
            root / "mind2web/scores/results.json",
            "mind2web_validation",
            mind2web,
        )
        scores(root / "mobile/scores/results.json", "mobile_validation", mobile)

    result = SELECTOR.select(
        tmp_path,
        epochs=2,
        save_every=10,
        mind2web_benchmark="mind2web_validation",
    )

    assert result["selected"]["epoch"] == 2
    assert result["test_data_used_for_selection"] is False


def test_release_summary_rejects_partial_test_run(tmp_path: Path) -> None:
    backbone = "a" * 64
    adapter_contract = {"backbone": {"sha256": backbone}}
    config = tmp_path / "run-config.json"
    scores = tmp_path / "scores.json"
    index = tmp_path / "index.json"
    config.write_text(
        json.dumps(
            {
                "checkpoint_sha256": backbone,
                "residual_adapter_contract": adapter_contract,
            }
        ),
        encoding="utf-8",
    )
    scores.write_text(
        json.dumps(
            {
                "benchmarks": {
                    "mind2web": {
                        "num_samples": 99,
                        "ssr_point_only": 0.8,
                        "joint_step_success": 0.75,
                        "action_f1_macro_present": 1.0,
                        "parse_rate": 1.0,
                        "latency_seconds": {"mean": 1.0},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    index.write_text(
        json.dumps(
            {
                "backbone_sha256": backbone,
                "runs": [
                    {
                        "label": "mind2web-test",
                        "benchmark": "mind2web",
                        "run_config": str(config),
                        "scores": str(scores),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        SUMMARY.summarize(index)
    except ValueError as error:
        assert "exactly 100" in str(error)
    else:
        raise AssertionError("partial release benchmark was accepted")
