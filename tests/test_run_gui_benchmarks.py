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


def test_catalog_resolves_all_five_suites_without_duplicate_arms():
    selection = MODULE.resolve_selection("all")

    assert [name for name, _ in selection.groups] == [
        "deployment",
        "native16k-five-way",
        "yarn-isolation",
        "true-long-yarn",
        "block-size-ablation",
    ]
    assert len(selection.arms) == 14
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


def test_block_size_ablation_changes_only_the_diffusion_block_size():
    selection = MODULE.resolve_selection("block-size-ablation")
    specs = [MODULE.BENCHMARKS[name] for name, _ in selection.arms]

    assert [spec.block_size for spec in specs] == [16, 8, 4]
    assert {spec.input_processing for spec in specs} == {
        "checkpoint-native single-image resize"
    }
    assert {spec.rope for spec in specs} == {"none (checkpoint-native)"}
    assert {spec.kv_policy for spec in specs} == {"dense; compression off"}


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
    assert "| yarn128k-ocr | 16 | long100 | 2 | 25.00 | 75.00 |" in markdown


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
