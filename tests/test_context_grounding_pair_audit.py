from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "D2F-eval/audit_context_grounding_pairs.py"
SPEC = importlib.util.spec_from_file_location("audit_context_grounding_pairs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
RECEIPT_PATH = ROOT / "D2F-eval/build_residual_release_receipt.py"
RECEIPT_SPEC = importlib.util.spec_from_file_location(
    "build_residual_release_receipt_for_context_test", RECEIPT_PATH
)
RECEIPT = importlib.util.module_from_spec(RECEIPT_SPEC)
assert RECEIPT_SPEC.loader is not None
sys.modules[RECEIPT_SPEC.name] = RECEIPT
RECEIPT_SPEC.loader.exec_module(RECEIPT)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    benchmark = tmp_path / "benchmark"
    clean_rows = []
    hard_rows = []
    for index in range(3):
        common = {
            "sample_id": f"sample-{index}",
            "target_action": "lclick",
            "target_bbox_1000": [700, 300, 800, 400],
        }
        clean_rows.append({**common, "hint_is_hard_negative": False})
        hard_rows.append(
            {
                **common,
                "hint_is_hard_negative": True,
                "hard_negative_bbox_1000": [100, 300, 200, 400],
            }
        )
    write_jsonl(benchmark / "clean.jsonl", clean_rows)
    write_jsonl(benchmark / "hard.jsonl", hard_rows)
    (benchmark / "manifest.json").write_text(
        json.dumps(
            {
                "benchmarks": {
                    "clean": {"path": "clean.jsonl", "rows": 3},
                    "hard": {"path": "hard.jsonl", "rows": 3},
                }
            }
        ),
        encoding="utf-8",
    )
    clean_predictions = tmp_path / "clean_predictions"
    hard_predictions = tmp_path / "hard_predictions"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    return benchmark, clean_predictions, hard_predictions, adapter


def prediction(benchmark: str, sample: int, bbox: list[int]) -> dict:
    return {
        "benchmark": benchmark,
        "sample_id": f"sample-{sample}",
        "error": None,
        "parse_error": None,
        "predicted_action": "lclick",
        "predicted_bbox_1000": bbox,
    }


def run_audit(
    benchmark: Path,
    clean_predictions: Path,
    hard_predictions: Path,
    adapter: Path,
) -> dict:
    return MODULE.audit(
        benchmark_root=benchmark,
        clean_benchmark="clean",
        hard_benchmark="hard",
        clean_predictions=clean_predictions,
        hard_predictions=hard_predictions,
        adapter=adapter,
        backbone_sha256="a" * 64,
        min_clean_ssr=0.70,
        min_hard_ssr=0.70,
        max_hard_ssr_drop=0.05,
        max_hard_distractor_rate=0.10,
        min_parse_rate=0.98,
    )


def test_pair_audit_accepts_hint_invariant_grounding(tmp_path: Path) -> None:
    benchmark, clean_predictions, hard_predictions, adapter = fixture(tmp_path)
    target = [710, 310, 790, 390]
    write_jsonl(
        clean_predictions / "clean/part-00000.jsonl",
        [prediction("clean", index, target) for index in range(3)],
    )
    write_jsonl(
        hard_predictions / "hard/part-00000.jsonl",
        [prediction("hard", index, target) for index in range(3)],
    )

    result = run_audit(benchmark, clean_predictions, hard_predictions, adapter)

    assert result["status"] == "passed"
    assert result["release_eligible"] is True
    assert result["metrics"]["hard_distractor_hits"] == 0
    assert result["metrics"]["paired_hit_consistency"] == 1.0


def test_pair_audit_rejects_predictions_that_follow_bad_hints(tmp_path: Path) -> None:
    benchmark, clean_predictions, hard_predictions, adapter = fixture(tmp_path)
    target = [710, 310, 790, 390]
    distractor = [110, 310, 190, 390]
    write_jsonl(
        clean_predictions / "clean/part-00000.jsonl",
        [prediction("clean", index, target) for index in range(3)],
    )
    write_jsonl(
        hard_predictions / "hard/part-00000.jsonl",
        [prediction("hard", index, distractor) for index in range(3)],
    )

    result = run_audit(benchmark, clean_predictions, hard_predictions, adapter)

    assert result["status"] == "failed"
    assert result["release_eligible"] is False
    assert result["checks"]["hard_ssr"] is False
    assert result["checks"]["hard_distractor_rate"] is False
    assert result["metrics"]["hard_distractor_hits"] == 3


def test_release_receipt_binds_passing_pair_audit_to_selected_adapter(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    adapter_model = adapter / "adapter_model.safetensors"
    adapter_model.write_bytes(b"adapter")
    audit_path = tmp_path / "context-audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "lladao-context-grounding-pair-audit-v1",
                "status": "passed",
                "release_eligible": True,
                "backbone_sha256": "a" * 64,
                "adapter": {
                    "path": str(adapter),
                    "adapter_model_sha256": MODULE.sha256_file(adapter_model),
                },
                "benchmark": {"samples": 69},
                "checks": {"hard_ssr": True, "hard_distractor_rate": True},
                "thresholds": {"min_hard_ssr": 0.70},
                "metrics": {"hard_ssr": 0.80},
            }
        ),
        encoding="utf-8",
    )

    evidence = RECEIPT.context_audit_evidence(
        audit_path,
        backbone_sha256="a" * 64,
        selected_adapter=adapter.resolve(),
        adapter_model_sha256=MODULE.sha256_file(adapter_model),
    )

    assert evidence["samples_per_arm"] == 69
    assert evidence["metrics"] == {"hard_ssr": 0.80}


def test_release_receipt_rejects_failed_pair_audit(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    adapter_model = adapter / "adapter_model.safetensors"
    adapter_model.write_bytes(b"adapter")
    audit_path = tmp_path / "context-audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "lladao-context-grounding-pair-audit-v1",
                "status": "failed",
                "release_eligible": False,
                "backbone_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    try:
        RECEIPT.context_audit_evidence(
            audit_path,
            backbone_sha256="a" * 64,
            selected_adapter=adapter.resolve(),
            adapter_model_sha256=MODULE.sha256_file(adapter_model),
        )
    except RECEIPT.ResidualReleaseReceiptError as error:
        assert "did not pass" in str(error)
    else:
        raise AssertionError("failed pair audit was accepted for release")
