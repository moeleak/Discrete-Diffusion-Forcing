#!/usr/bin/env python3
"""Bind the selected residual adapter to its two completed test-100 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_RUNS = {
    "mind2web-test": "mind2web",
    "mobile-test": "mobile_test",
}
EXPECTED_SELECTION_METRIC = (
    "max(min(mind2web_validation_ssr,mobile_validation_ssr))"
)
EXPECTED_SELECTION_TIE_BREAKERS = [
    "mean_domain_ssr",
    "worst_domain_joint_ssr",
    "earlier_epoch",
]
EXPECTED_TARGET_MODULES = (
    r"language_model\.model\.layers\.\d+\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj)"
)
EXPECTED_RUN_CONFIG = {
    "backend": "d2f",
    "seed": 42,
    "max_new_tokens": 64,
    "diffusion_steps": 64,
    "confidence_threshold": 0.95,
    "block_size": 16,
    "block_add_threshold": 0.1,
    "decoded_token_threshold": 0.95,
    "skip_threshold": 0.9,
    "temperature": 0.0,
    "max_model_len": 16_384,
    "kv_cache_capacity": None,
    "rope_scaling": "none",
    "allow_unscaled_max_model_len": False,
    "full_page_tiles": None,
    "full_page_position_mode": "native",
    "full_page_overview": False,
    "truncate_full_page_tiles": False,
    "attention_backend": "sdpa",
    "rms_norm_backend": "torch",
    "kv_cache_compression": False,
    "kv_cache_retrieval": False,
    "sample_seed_policy": "sha256(base_seed, provenance.action_uid || sample_id)",
}


class ResidualReleaseReceiptError(ValueError):
    """Raised when selection and held-out benchmark evidence do not agree."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ResidualReleaseReceiptError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResidualReleaseReceiptError(f"{label} must contain a JSON object")
    return value


def resolve_evidence_path(raw: Any, *, relative_to: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ResidualReleaseReceiptError(f"{label} path is missing")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ResidualReleaseReceiptError(f"{label} must be one lowercase SHA-256")
    return value


def require_finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ResidualReleaseReceiptError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ResidualReleaseReceiptError(f"{label} must be finite")
    return result


def validate_adapter_config(config: dict[str, Any]) -> None:
    expected = {
        "r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "bias": "none",
        "peft_type": "LORA",
        "use_dora": False,
        "use_rslora": False,
        "modules_to_save": None,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ResidualReleaseReceiptError(
                f"selected PEFT adapter has unexpected {key}: {config.get(key)!r}"
            )
    if config.get("target_modules") != EXPECTED_TARGET_MODULES:
        raise ResidualReleaseReceiptError(
            "selected PEFT adapter does not target only language q/k/v/o attention"
        )


def validate_training_contract(
    contract: dict[str, Any],
    *,
    backbone_sha256: str,
    selected_step: int,
    expected_max_steps: int,
) -> None:
    if contract.get("schema_version") != 1:
        raise ResidualReleaseReceiptError("selected adapter contract schema is unsupported")
    if contract.get("format") != "lladao-residual-grounding-lora-v1":
        raise ResidualReleaseReceiptError("selected adapter contract format is unsupported")
    backbone = contract.get("backbone") or {}
    if backbone.get("sha256") != backbone_sha256:
        raise ResidualReleaseReceiptError(
            "selected adapter contract targets a different Planner checkpoint"
        )
    if (
        backbone.get("parameter_count") != 8_459_716_512
        or backbone.get("dtype") != "bfloat16"
        or backbone.get("contains_generation_experts") is not False
        or backbone.get("contains_lora") is not False
        or backbone.get("format") != "understanding-only-full-safetensors"
    ):
        raise ResidualReleaseReceiptError(
            "selected adapter contract does not prove one clean full Planner backbone"
        )
    adapter = contract.get("adapter") or {}
    if (
        adapter.get("module_count") != 128
        or adapter.get("rank") != 32
        or adapter.get("alpha") != 32
        or adapter.get("dropout") != 0.1
        or adapter.get("targets") != ["q_proj", "k_proj", "v_proj", "o_proj"]
        or adapter.get("zero_delta") is not True
        or adapter.get("frozen_backbone_parameters") != 8_459_716_512
        or adapter.get("trainable_lora_parameters") != 33_554_432
        or adapter.get("trainable_parameter_tensors") != 256
    ):
        raise ResidualReleaseReceiptError(
            "selected adapter contract does not prove the fixed residual LoRA architecture"
        )
    training = contract.get("training") or {}
    try:
        step = int(training.get("step", -1))
        max_steps = int(training.get("max_steps", -1))
        counts = training.get("domain_microbatches") or {}
        mind2web_count = int(counts.get("mind2web", 0))
        mobile_count = int(counts.get("mobile", -1))
    except (TypeError, ValueError) as error:
        raise ResidualReleaseReceiptError(
            "selected adapter contract contains invalid training counters"
        ) from error
    if (
        not 0 < step <= max_steps
        or step != selected_step
        or max_steps != expected_max_steps
    ):
        raise ResidualReleaseReceiptError(
            "selected adapter step does not match its complete training contract"
        )
    if (
        set(counts) != {"mind2web", "mobile"}
        or mind2web_count <= 0
        or mind2web_count != mobile_count
        or training.get("domain_mix") != {"mind2web": 0.5, "mobile": 0.5}
        or training.get("mind2web_objective")
        != "d2f_distillation_plus_hard_ce"
        or training.get("mobile_objective") != "hard_ce_only"
        or SHA256_RE.fullmatch(str(training.get("config_sha256", ""))) is None
        or training.get("release_eligible") is not True
    ):
        raise ResidualReleaseReceiptError(
            "selected adapter contract does not prove the fixed two-domain recipe"
        )


def validate_selection(selection: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if selection.get("schema_version") != 1:
        raise ResidualReleaseReceiptError("checkpoint selection schema is unsupported")
    if selection.get("selection_metric") != EXPECTED_SELECTION_METRIC:
        raise ResidualReleaseReceiptError("checkpoint selection metric drifted")
    if selection.get("tie_breakers") != EXPECTED_SELECTION_TIE_BREAKERS:
        raise ResidualReleaseReceiptError("checkpoint selection tie-breakers drifted")
    if selection.get("test_data_used_for_selection") is not False:
        raise ResidualReleaseReceiptError(
            "checkpoint selection must prove test labels were not used"
        )
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ResidualReleaseReceiptError(
            "checkpoint selection must contain the three validation epochs"
        )
    validated: list[dict[str, Any]] = []
    steps: list[int] = []
    for expected_epoch, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise ResidualReleaseReceiptError("selection candidates must be objects")
        try:
            epoch = int(candidate.get("epoch", -1))
            step = int(candidate.get("step", -1))
        except (TypeError, ValueError) as error:
            raise ResidualReleaseReceiptError(
                "selection candidate epoch and step must be integers"
            ) from error
        if epoch != expected_epoch or step <= 0:
            raise ResidualReleaseReceiptError(
                "selection candidates must contain ordered positive epochs and steps"
            )
        mind = require_finite_float(
            candidate.get("mind2web_validation_ssr"),
            label=f"epoch {epoch} Mind2Web validation SSR",
        )
        mobile = require_finite_float(
            candidate.get("mobile_validation_ssr"),
            label=f"epoch {epoch} mobile validation SSR",
        )
        mind_joint = require_finite_float(
            candidate.get("mind2web_validation_joint_ssr"),
            label=f"epoch {epoch} Mind2Web validation joint SSR",
        )
        mobile_joint = require_finite_float(
            candidate.get("mobile_validation_joint_ssr"),
            label=f"epoch {epoch} mobile validation joint SSR",
        )
        declared = (
            require_finite_float(
                candidate.get("worst_domain_ssr"),
                label=f"epoch {epoch} worst-domain SSR",
            ),
            require_finite_float(
                candidate.get("mean_domain_ssr"),
                label=f"epoch {epoch} mean-domain SSR",
            ),
            require_finite_float(
                candidate.get("worst_domain_joint_ssr"),
                label=f"epoch {epoch} worst-domain joint SSR",
            ),
        )
        source_values = (mind, mobile, mind_joint, mobile_joint)
        if any(value < 0.0 or value > 1.0 for value in source_values):
            raise ResidualReleaseReceiptError(
                f"epoch {epoch} validation metrics must be fractions in [0, 1]"
            )
        recomputed = (
            min(mind, mobile),
            (mind + mobile) / 2.0,
            min(mind_joint, mobile_joint),
        )
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(declared, recomputed, strict=True)
        ):
            raise ResidualReleaseReceiptError(
                f"epoch {epoch} validation selection metrics are inconsistent"
            )
        validated.append(candidate)
        steps.append(step)
    if steps != sorted(set(steps)):
        raise ResidualReleaseReceiptError(
            "selection candidate steps must be unique and strictly increasing"
        )
    recomputed_selected = max(
        validated,
        key=lambda item: (
            item["worst_domain_ssr"],
            item["mean_domain_ssr"],
            item["worst_domain_joint_ssr"],
            -item["epoch"],
        ),
    )
    selected = selection.get("selected")
    if not isinstance(selected, dict) or selected != recomputed_selected:
        raise ResidualReleaseReceiptError(
            "selected adapter does not maximize the validation-only metric"
        )
    return selected, steps[-1]


def selected_adapter_evidence(
    selection_path: Path,
    *,
    backbone_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = read_json(selection_path, label="checkpoint selection")
    selected, expected_max_steps = validate_selection(selection)
    # checkpoint-selection.json lives below <run>/benchmark/. Relative adapter
    # paths are resolved from the run root rather than the current directory.
    run_root = selection_path.parent.parent
    adapter = resolve_evidence_path(
        selected.get("adapter"), relative_to=run_root, label="selected adapter"
    )
    adapter_model = adapter / "adapter_model.safetensors"
    adapter_config = adapter / "adapter_config.json"
    training_contract_path = adapter / "training_contract.json"
    contract = read_json(training_contract_path, label="training contract")
    config = read_json(adapter_config, label="PEFT adapter config")
    if not adapter_model.is_file() or adapter_model.stat().st_size <= 0:
        raise ResidualReleaseReceiptError(
            f"selected adapter weights are missing: {adapter_model}"
        )
    try:
        selected_step = int(selected.get("step", -2))
        selected_epoch = int(selected.get("epoch", -1))
    except (TypeError, ValueError) as error:
        raise ResidualReleaseReceiptError(
            "checkpoint selection contains invalid epoch or step"
        ) from error
    if selected_epoch <= 0:
        raise ResidualReleaseReceiptError("checkpoint selection epoch must be positive")
    validate_adapter_config(config)
    validate_training_contract(
        contract,
        backbone_sha256=backbone_sha256,
        selected_step=selected_step,
        expected_max_steps=expected_max_steps,
    )
    return selection, {
        "path": str(adapter),
        "epoch": selected_epoch,
        "step": selected_step,
        "adapter_model_sha256": sha256_file(adapter_model),
        "adapter_config_sha256": sha256_file(adapter_config),
        "training_contract_sha256": sha256_file(training_contract_path),
    }


def benchmark_evidence(
    index: dict[str, Any],
    index_path: Path,
    *,
    backbone_sha256: str,
    selected_adapter: Path,
    training_contract: dict[str, Any],
) -> dict[str, Any]:
    runs = index.get("runs")
    if not isinstance(runs, list):
        raise ResidualReleaseReceiptError("benchmark index runs must be a list")
    by_label: dict[str, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict) or run.get("label") in by_label:
            raise ResidualReleaseReceiptError("benchmark run labels must be unique objects")
        by_label[str(run.get("label"))] = run
    if set(by_label) != set(EXPECTED_RUNS):
        raise ResidualReleaseReceiptError(
            "release requires exactly mind2web-test and mobile-test benchmark evidence"
        )

    evidence: dict[str, Any] = {}
    for label, expected_benchmark in EXPECTED_RUNS.items():
        run = by_label[label]
        benchmark = run.get("benchmark")
        if not isinstance(benchmark, str) or not benchmark:
            raise ResidualReleaseReceiptError(f"{label} benchmark name is missing")
        if benchmark != expected_benchmark:
            raise ResidualReleaseReceiptError(
                f"{label} must use benchmark {expected_benchmark!r}"
            )
        run_config_path = resolve_evidence_path(
            run.get("run_config"), relative_to=index_path.parent, label=f"{label} run config"
        )
        scores_path = resolve_evidence_path(
            run.get("scores"), relative_to=index_path.parent, label=f"{label} scores"
        )
        run_config = read_json(run_config_path, label=f"{label} run config")
        scores = read_json(scores_path, label=f"{label} scores")
        drifted = {
            key: run_config.get(key)
            for key, expected in EXPECTED_RUN_CONFIG.items()
            if run_config.get(key) != expected
        }
        if drifted:
            raise ResidualReleaseReceiptError(
                f"{label} changed the fixed native D2F decoding protocol: {drifted}"
            )
        if run_config.get("checkpoint_sha256") != backbone_sha256:
            raise ResidualReleaseReceiptError(f"{label} changed the Planner checkpoint")
        if run_config.get("expected_checkpoint_sha256") != backbone_sha256:
            raise ResidualReleaseReceiptError(
                f"{label} did not enforce the Planner checkpoint SHA-256"
            )
        if run_config.get("benchmarks") != [benchmark]:
            raise ResidualReleaseReceiptError(
                f"{label} run config contains the wrong benchmark selection"
            )
        benchmark_root = resolve_evidence_path(
            run_config.get("benchmark_root"),
            relative_to=run_config_path.parent,
            label=f"{label} benchmark root",
        )
        benchmark_manifest = (benchmark_root / "manifest.json").resolve()
        scored_manifest = resolve_evidence_path(
            scores.get("benchmark_manifest"),
            relative_to=scores_path.parent,
            label=f"{label} scored benchmark manifest",
        )
        if not benchmark_manifest.is_file() or scored_manifest != benchmark_manifest:
            raise ResidualReleaseReceiptError(
                f"{label} scores are not bound to the configured benchmark manifest"
            )
        run_adapter = resolve_evidence_path(
            run_config.get("adapter"), relative_to=run_config_path.parent, label=f"{label} adapter"
        )
        if run_adapter != selected_adapter:
            raise ResidualReleaseReceiptError(f"{label} evaluated a non-selected adapter")
        if run_config.get("residual_adapter_contract") != training_contract:
            raise ResidualReleaseReceiptError(
                f"{label} did not record the selected adapter contract"
            )
        if int(run_config.get("limit_per_benchmark", -1)) != 100:
            raise ResidualReleaseReceiptError(f"{label} was not capped at 100 samples")
        score_benchmarks = scores.get("benchmarks") or {}
        if not isinstance(score_benchmarks, dict) or set(score_benchmarks) != {benchmark}:
            raise ResidualReleaseReceiptError(
                f"{label} scores contain an unexpected benchmark set"
            )
        coverage = (scores.get("coverage") or {}).get(benchmark) or {}
        if coverage != {
            "targets": 100,
            "predictions": 100,
            "joined": 100,
            "missing": 0,
        }:
            raise ResidualReleaseReceiptError(
                f"{label} does not prove complete 100/100 prediction coverage"
            )
        metrics = score_benchmarks.get(benchmark)
        if not isinstance(metrics, dict) or int(metrics.get("num_samples", -1)) != 100:
            raise ResidualReleaseReceiptError(
                f"{label} must contain exactly 100 scored samples"
            )
        latency = metrics.get("latency_seconds") or {}
        metric_values = {
            "ssr_point_only": require_finite_float(
                metrics.get("ssr_point_only"), label=f"{label} SSR"
            ),
            "joint_step_success": require_finite_float(
                metrics.get("joint_step_success"), label=f"{label} joint SSR"
            ),
            "action_f1_macro_present": require_finite_float(
                metrics.get("action_f1_macro_present"), label=f"{label} action F1"
            ),
            "parse_rate": require_finite_float(
                metrics.get("parse_rate"), label=f"{label} parse rate"
            ),
            "mean_latency_seconds": require_finite_float(
                latency.get("mean"), label=f"{label} mean latency"
            ),
        }
        if any(
            not 0.0 <= metric_values[key] <= 1.0
            for key in (
                "ssr_point_only",
                "joint_step_success",
                "action_f1_macro_present",
                "parse_rate",
            )
        ):
            raise ResidualReleaseReceiptError(
                f"{label} quality metrics must be fractions in [0, 1]"
            )
        if metric_values["mean_latency_seconds"] < 0.0:
            raise ResidualReleaseReceiptError(f"{label} mean latency cannot be negative")
        evidence[label] = {
            "benchmark": benchmark,
            "num_samples": 100,
            "run_config_path": str(run_config_path),
            "run_config_sha256": sha256_file(run_config_path),
            "scores_path": str(scores_path),
            "scores_sha256": sha256_file(scores_path),
            "metrics": metric_values,
        }
    return evidence


def build_release_receipt(index_path: Path, selection_path: Path) -> dict[str, Any]:
    index_path = index_path.expanduser().resolve()
    selection_path = selection_path.expanduser().resolve()
    index = read_json(index_path, label="benchmark index")
    backbone_sha256 = require_sha256(
        index.get("backbone_sha256"), label="benchmark backbone_sha256"
    )
    selection, adapter_evidence = selected_adapter_evidence(
        selection_path, backbone_sha256=backbone_sha256
    )
    selected_adapter = Path(adapter_evidence["path"])
    training_contract_path = selected_adapter / "training_contract.json"
    training_contract = read_json(training_contract_path, label="training contract")
    benchmarks = benchmark_evidence(
        index,
        index_path,
        backbone_sha256=backbone_sha256,
        selected_adapter=selected_adapter,
        training_contract=training_contract,
    )
    return {
        "schema_version": 1,
        "format": "lladao-residual-grounding-release-v1",
        "status": "benchmark-complete",
        "release_eligible": True,
        "backbone": {"sha256": backbone_sha256},
        "selected_adapter": adapter_evidence,
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
            "metric": selection.get("selection_metric"),
        },
        "benchmarks": benchmarks,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = build_release_receipt(args.index, args.selection)
    output = args.output.expanduser().resolve()
    write_json_atomic(output, receipt)
    print(output)


if __name__ == "__main__":
    main()
