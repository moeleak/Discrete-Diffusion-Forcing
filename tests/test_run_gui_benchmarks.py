import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "D2F-eval" / "run_gui_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("run_gui_benchmarks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_dataset(root: Path, sample_ids: list[str]) -> None:
    root.mkdir(parents=True)
    records = root / "records.jsonl"
    records.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "target_action": "CLICK",
                    "target_bbox_1000": [0, 0, 10, 10],
                }
            )
            + "\n"
            for sample_id in sample_ids
        ),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "benchmarks": {
                    MODULE.BENCHMARK: {
                        "path": records.name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def summary(mean: float, *, maximum: float | None = None) -> dict:
    return {
        "count": 2,
        "mean": mean,
        "p50": mean,
        "p95": mean * 1.5,
        "min": mean / 2,
        "max": maximum if maximum is not None else mean * 2,
    }


def write_scores(
    output: Path,
    sample_ids: list[str],
    *,
    ssr: float,
    resident: float,
    dense: float,
) -> None:
    prediction_dir = output / MODULE.BENCHMARK
    prediction_dir.mkdir(parents=True)
    (prediction_dir / "part-00000.jsonl").write_text(
        "".join(
            json.dumps({"sample_id": sample_id}) + "\n" for sample_id in sample_ids
        ),
        encoding="utf-8",
    )
    scores = output / "scores"
    scores.mkdir()
    (scores / "results.json").write_text(
        json.dumps(
            {
                "benchmarks": {
                    MODULE.BENCHMARK: {
                        "num_samples": len(sample_ids),
                        "ssr_point_only": ssr,
                        "joint_step_success": ssr,
                        "action_f1_macro_present": 1.0,
                        "parse_rate": 1.0,
                        "convergence_steps": summary(14.0),
                        "latency_seconds": summary(4.2),
                    }
                },
                "runtime": {
                    MODULE.BENCHMARK: {
                        "model_elapsed_seconds": summary(4.0),
                        "total_tokens_per_second": summary(1000.0),
                        "dense_prefix_tokens": summary(dense),
                        "cached_prefix_tokens": summary(resident),
                        "max_prefill_position": summary(50.0, maximum=100.0),
                        "max_generation_position": summary(60.0, maximum=111.0),
                        "input_images": summary(4.0),
                        "peak_memory_allocated_gib": summary(10.0, maximum=12.0),
                        "errors": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def make_completed_run(tmp_path: Path) -> tuple[Path, dict]:
    dataset = tmp_path / "dataset"
    sample_ids = ["sample-a", "sample-b"]
    write_dataset(dataset, sample_ids)
    fingerprint = MODULE.dataset_fingerprint(dataset, len(sample_ids))
    run_dir = tmp_path / "run"
    logs = run_dir / "logs"
    result_dir = run_dir / "arms" / "long100--yarn128k-ocr"
    arm = MODULE.build_arm_record(
        benchmark_name="yarn128k-ocr",
        dataset_name="long100",
        benchmark_root=dataset,
        result_dir=result_dir,
        log_dir=logs,
        repo=Path(__file__).parents[1],
        root=tmp_path,
        lladao_repo=tmp_path / "lladao",
        limit=len(sample_ids),
        gpu="0",
        run_id="fixture",
        revision="abc1234",
        fingerprint=fingerprint,
    )
    arm["status"] = "completed"
    write_scores(
        result_dir / "model",
        sample_ids,
        ssr=0.25,
        resident=75.0,
        dense=100.0,
    )
    write_scores(
        result_dir / "fused",
        sample_ids,
        ssr=0.75,
        resident=75.0,
        dense=100.0,
    )
    run = {
        "schema_version": MODULE.SCHEMA_VERSION,
        "run_id": "fixture",
        "target": "yarn128k-ocr",
        "status": "completed",
        "limit": len(sample_ids),
        "gpu": "0",
        "revision": "abc1234",
        "groups": [
            {
                "name": "yarn128k-ocr",
                "arms": [arm["id"]],
            }
        ],
        "arms": [arm],
    }
    MODULE.atomic_write_json(run_dir / "run.json", run)
    return run_dir, arm


def test_catalog_resolves_all_nine_suites_without_duplicate_arms():
    selection = MODULE.resolve_selection("all")

    assert [name for name, _ in selection.groups] == [
        "deployment",
        "native16k-five-way",
        "yarn-isolation",
        "true-long-yarn",
        "tile-size-ablation",
        "kv-retrieval-scoring-ablation",
        "kv-retrieval-packing-ablation",
        "kv-retrieval-attention-ablation",
        "kv-retrieval-topk-ablation",
    ]
    assert len(selection.arms) == 19
    assert len(selection.arms) == len(set(selection.arms))


def test_single_benchmark_uses_its_default_dataset():
    selection = MODULE.resolve_selection("yarn128k-kv-top4-ocr")

    assert selection.arms == (("yarn128k-kv-top4-ocr", "long100"),)
    assert selection.groups == (
        (
            "yarn128k-kv-top4-ocr",
            (("yarn128k-kv-top4-ocr", "long100"),),
        ),
    )


def test_tile_size_ablation_changes_only_the_full_page_tile_size():
    selection = MODULE.resolve_selection("tile-size-ablation")
    specs = [MODULE.BENCHMARKS[name] for name, _ in selection.arms]

    assert [spec.full_page_tile_size for spec in specs] == [980, 686, 490]
    assert {spec.block_size for spec in specs} == {16}
    assert {spec.rope for spec in specs} == {"YaRN factor 8"}
    assert {spec.position_mode for spec in specs} == {"strided"}
    assert {spec.overview for spec in specs} == {"yes"}
    assert {spec.ocr for spec in specs} == {"prompt-only retrieval fusion"}
    assert {spec.kv_policy for spec in specs} == {"dense; compression off"}
    assert {spec.environment for spec in specs} == {
        MODULE.pairs(KV_CACHE_CAPACITY="65536")
    }


def test_kv_retrieval_scoring_ablation_changes_only_the_scorer():
    selection = MODULE.resolve_selection("kv-retrieval-scoring-ablation")

    assert selection.arms == (
        ("yarn128k-ocr", "long100"),
        ("yarn128k-kv-top4-causal-ocr", "long100"),
        ("yarn128k-kv-top4-ocr", "long100"),
    )
    causal = MODULE.BENCHMARKS["yarn128k-kv-top4-causal-ocr"]
    masked = MODULE.BENCHMARKS["yarn128k-kv-top4-ocr"]
    assert causal.launcher == masked.launcher
    assert causal.dataset == masked.dataset
    assert causal.input_processing == masked.input_processing
    assert causal.rope == masked.rope
    assert causal.max_context == masked.max_context
    assert causal.position_mode == masked.position_mode
    assert causal.crop == masked.crop
    assert causal.overview == masked.overview
    assert causal.truncation == masked.truncation
    assert causal.ocr == masked.ocr
    causal_environment = dict(causal.environment)
    masked_environment = dict(masked.environment)
    assert causal_environment.pop("KV_RETRIEVAL_SCORE_MODE") == (
        "causal_self_information"
    )
    assert masked_environment.pop("KV_RETRIEVAL_SCORE_MODE") == (
        "masked_self_information"
    )
    assert causal_environment == masked_environment


def test_kv_retrieval_packing_ablation_changes_only_batching():
    selection = MODULE.resolve_selection("kv-retrieval-packing-ablation")

    assert selection.arms == (
        ("yarn128k-kv-top4-sequential-ocr", "long100"),
        ("yarn128k-kv-top4-ocr", "long100"),
    )
    sequential = MODULE.BENCHMARKS[
        "yarn128k-kv-top4-sequential-ocr"
    ]
    packed = MODULE.BENCHMARKS["yarn128k-kv-top4-ocr"]
    assert sequential.launcher == packed.launcher
    assert sequential.dataset == packed.dataset
    assert sequential.input_processing == packed.input_processing
    assert sequential.rope == packed.rope
    assert sequential.max_context == packed.max_context
    assert sequential.position_mode == packed.position_mode
    assert sequential.crop == packed.crop
    assert sequential.overview == packed.overview
    assert sequential.truncation == packed.truncation
    assert sequential.ocr == packed.ocr
    sequential_environment = dict(sequential.environment)
    packed_environment = dict(packed.environment)
    assert sequential_environment.pop(
        "KV_RETRIEVAL_PACKED_SCORING"
    ) == "0"
    assert packed_environment.pop("KV_RETRIEVAL_PACKED_SCORING") == "1"
    assert sequential_environment == packed_environment


def test_kv_retrieval_attention_ablation_changes_only_attention_mode():
    selection = MODULE.resolve_selection("kv-retrieval-attention-ablation")

    assert selection.arms == (
        ("yarn128k-kv-top4-causal-masked-ocr", "long100"),
        ("yarn128k-kv-top4-sequential-ocr", "long100"),
    )
    causal = MODULE.BENCHMARKS[
        "yarn128k-kv-top4-causal-masked-ocr"
    ]
    bidirectional = MODULE.BENCHMARKS[
        "yarn128k-kv-top4-sequential-ocr"
    ]
    assert causal.launcher == bidirectional.launcher
    assert causal.dataset == bidirectional.dataset
    assert causal.input_processing == bidirectional.input_processing
    assert causal.rope == bidirectional.rope
    assert causal.max_context == bidirectional.max_context
    assert causal.position_mode == bidirectional.position_mode
    assert causal.crop == bidirectional.crop
    assert causal.overview == bidirectional.overview
    assert causal.truncation == bidirectional.truncation
    assert causal.ocr == bidirectional.ocr
    assert causal.retrieval_query == bidirectional.retrieval_query
    assert causal.block_size == bidirectional.block_size
    assert causal.full_page_tile_size == bidirectional.full_page_tile_size
    causal_environment = dict(causal.environment)
    bidirectional_environment = dict(bidirectional.environment)
    assert causal_environment.pop("KV_RETRIEVAL_SCORE_MODE") == (
        "causal_masked_self_information"
    )
    assert bidirectional_environment.pop("KV_RETRIEVAL_SCORE_MODE") == (
        "masked_self_information"
    )
    assert causal_environment == bidirectional_environment
    assert causal_environment["KV_RETRIEVAL_PACKED_SCORING"] == "0"


def test_kv_retrieval_topk_ablation_changes_only_retained_tile_count():
    selection = MODULE.resolve_selection("kv-retrieval-topk-ablation")

    assert selection.arms == (
        ("yarn128k-kv-top4-sequential-ocr", "long100"),
        ("yarn128k-kv-top8-sequential-ocr", "long100"),
    )
    top4 = MODULE.BENCHMARKS["yarn128k-kv-top4-sequential-ocr"]
    top8 = MODULE.BENCHMARKS["yarn128k-kv-top8-sequential-ocr"]
    assert top4.launcher == top8.launcher
    assert top4.dataset == top8.dataset
    assert top4.rope == top8.rope
    assert top4.max_context == top8.max_context
    assert top4.position_mode == top8.position_mode
    assert top4.crop == top8.crop
    assert top4.overview == top8.overview
    assert top4.truncation == top8.truncation
    assert top4.ocr == top8.ocr
    assert top4.retrieval_query == top8.retrieval_query
    assert top4.block_size == top8.block_size
    assert top4.full_page_tile_size == top8.full_page_tile_size
    top4_environment = dict(top4.environment)
    top8_environment = dict(top8.environment)
    assert top4_environment.pop("KV_RETRIEVAL_TOPK_IMAGES") == "4"
    assert top8_environment.pop("KV_RETRIEVAL_TOPK_IMAGES") == "8"
    assert top4_environment == top8_environment


def test_retrieval_target_tile_recall_uses_ground_truth_only_posthoc(
    tmp_path,
):
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "sample_id": "sample-a",
                "provenance": {
                    "source_bbox_xyxy": [120, 10, 180, 50],
                },
                "tile_layout": [
                    {"index": 0, "box_xyxy": [0, 0, 100, 100]},
                    {"index": 1, "box_xyxy": [100, 0, 200, 100]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    predictions = [
        {
            "sample_id": "sample-a",
            "kv_cache_retrieval_enabled": True,
            "kv_cache_retrieval_indices": [1, 2],
        }
    ]

    assert MODULE.retrieval_target_tile_recall(
        predictions,
        records,
        ["sample-a"],
    ) == 100.0


@pytest.mark.parametrize("limit", [0, 101])
def test_limit_rejects_values_outside_repository_policy(limit):
    with pytest.raises(ValueError, match=r"\[1, 100\]"):
        MODULE.validate_limit(limit)


@pytest.mark.parametrize("gpu", ["0,1", "-1", "cuda:0"])
def test_gpu_requires_one_device_index(gpu):
    with pytest.raises(ValueError, match="one non-negative"):
        MODULE.validate_gpu(gpu)


def test_run_id_rejects_paths():
    with pytest.raises(ValueError, match="run-id"):
        MODULE.validate_run_id("../../reuse-old-results")


def test_arm_record_uses_launcher_specific_output_variable(tmp_path):
    dataset = tmp_path / "dataset"
    write_dataset(dataset, ["sample-a"])
    fingerprint = MODULE.dataset_fingerprint(dataset, 1)
    common = {
        "dataset_name": "long100",
        "benchmark_root": dataset,
        "result_dir": tmp_path / "result",
        "log_dir": tmp_path / "logs",
        "repo": Path(__file__).parents[1],
        "root": tmp_path,
        "lladao_repo": tmp_path / "lladao",
        "limit": 1,
        "gpu": "3",
        "run_id": "fixture",
        "revision": "abc1234",
        "fingerprint": fingerprint,
    }

    direct = MODULE.build_arm_record(
        benchmark_name="original16k-native",
        **common,
    )
    wrapped = MODULE.build_arm_record(
        benchmark_name="yarn128k-ocr",
        **common,
    )

    assert direct["environment"]["OUTPUT_DIR"] == str(tmp_path / "result")
    assert "RESULT_ROOT" not in direct["environment"]
    assert wrapped["environment"]["RESULT_ROOT"] == str(tmp_path / "result")
    assert "OUTPUT_DIR" not in wrapped["environment"]
    assert direct["environment"]["GPU"] == "3"
    assert direct["environment"]["BLOCK_SIZE"] == "16"
    assert "FULL_PAGE_TILE_SIZE" not in direct["environment"]
    assert wrapped["environment"]["FULL_PAGE_TILE_SIZE"] == "980"


def test_report_writes_quality_performance_and_protocol_tables(tmp_path):
    run_dir, _ = make_completed_run(tmp_path)

    reports = MODULE.build_reports(run_dir)

    quality = reports["quality"]["yarn128k-ocr"][0]
    performance = reports["performance"]["yarn128k-ocr"][0]
    protocol = reports["protocol"]["yarn128k-ocr"][0]
    assert quality["Raw SSR (%)"] == 25.0
    assert quality["Final SSR (%)"] == 75.0
    assert quality["Final stage"] == "OCR/fused"
    assert performance["KV reduction (%)"] == 25.0
    assert performance["Max actual RoPE"] == 111.0
    assert performance["Mean convergence steps"] == 14.0
    assert performance["Mean end-to-end latency (s)"] == 4.2
    assert performance["Mean model latency (s)"] == 4.0
    assert len(protocol["Manifest SHA-256"]) == 64
    assert protocol["Worktree"] == "clean"
    for name in ("quality", "performance", "protocol"):
        for suffix in ("md", "csv", "json"):
            assert (run_dir / "tables" / f"{name}.{suffix}").is_file()
    markdown = (run_dir / "tables" / "quality.md").read_text(encoding="utf-8")
    assert "| yarn128k-ocr | 980 | 16 | long100 | 2 | 25.00 | 75.00 |" in markdown


def test_report_rejects_prediction_order_changes(tmp_path):
    run_dir, arm = make_completed_run(tmp_path)
    final_shard = (
        Path(arm["result_dir"]) / "fused" / MODULE.BENCHMARK / "part-00000.jsonl"
    )
    final_shard.write_text(
        json.dumps({"sample_id": "sample-b"})
        + "\n"
        + json.dumps({"sample_id": "sample-a"})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="prediction order differs"):
        MODULE.build_reports(run_dir)
