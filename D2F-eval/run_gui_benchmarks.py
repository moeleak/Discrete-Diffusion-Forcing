#!/usr/bin/env python3
"""Run and report the supported LLaDA-o GUI-grounding benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

BENCHMARK = "mind2web_fullpage"
MAX_SAMPLES = 100
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    relative_path: str


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    description: str
    launcher: str
    dataset: str
    output_variable: str
    raw_output: str
    final_output: str | None
    environment: tuple[tuple[str, str], ...]
    input_processing: str
    rope: str
    max_context: int
    position_mode: str
    crop: str
    overview: str
    truncation: str
    ocr: str
    kv_policy: str
    retrieval_query: str
    block_size: int = 16
    full_page_tile_size: int | None = None


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    description: str
    arms: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Selection:
    target: str
    arms: tuple[tuple[str, str], ...]
    groups: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]


DATASETS = {
    "long100": DatasetSpec(
        name="long100",
        description="Mind2Web full-page 16K-64K ordered sample set",
        relative_path="data/mind2web-fullpage-16k-64k",
    ),
    "native16k100": DatasetSpec(
        name="native16k100",
        description="Mind2Web full-page native-16K seed42 sample set",
        relative_path="data/mind2web-fullpage-native16k-n100-seed42",
    ),
}


def pairs(**values: str) -> tuple[tuple[str, str], ...]:
    return tuple(values.items())


BENCHMARKS = {
    "native-ocr-crop": BenchmarkSpec(
        name="native-ocr-crop",
        description="Native D2F with full-page OCR retrieval and a 980px crop",
        launcher="d2f_vllm/mllm_lladao_gui_ocr_crop_pipeline.sh",
        dataset="long100",
        output_variable="RESULT_ROOT",
        raw_output="native-model",
        final_output="fused",
        environment=pairs(),
        input_processing="native resize + OCR-selected 980px crop",
        rope="none (checkpoint-native)",
        max_context=16_384,
        position_mode="native",
        crop="OCR retrieval crop",
        overview="no",
        truncation="no",
        ocr="retrieval + crop fusion",
        kv_policy="dense; compression off",
        retrieval_query="visible operation instruction",
    ),
    "yarn128k-ocr": BenchmarkSpec(
        name="yarn128k-ocr",
        description="YaRN 128K with all source tiles, overview, and OCR",
        launcher="d2f_vllm/mllm_lladao_gui_yarn_uncropped_ocr.sh",
        dataset="long100",
        output_variable="RESULT_ROOT",
        raw_output="model",
        final_output="fused",
        environment=pairs(KV_CACHE_CAPACITY="65536"),
        input_processing="all exact 980px tiles + whole-page overview",
        rope="YaRN factor 8",
        max_context=131_072,
        position_mode="strided",
        crop="no",
        overview="yes",
        truncation="no",
        ocr="prompt-only retrieval fusion",
        kv_policy="dense; compression off",
        retrieval_query="visible operation instruction",
        full_page_tile_size=980,
    ),
    "yarn128k-kv-top4-ocr": BenchmarkSpec(
        name="yarn128k-kv-top4-ocr",
        description=(
            "YaRN 128K with bidirectional masked-query Top-4 image KV retrieval"
        ),
        launcher=("d2f_vllm/mllm_lladao_gui_yarn_uncropped_kv_retrieval_ocr.sh"),
        dataset="long100",
        output_variable="RESULT_ROOT",
        raw_output="model",
        final_output="fused",
        environment=pairs(
            KV_CACHE_CAPACITY="65536",
            KV_RETRIEVAL_TOPK_IMAGES="4",
            KV_RETRIEVAL_SCORE_MODE="masked_self_information",
            KV_RETRIEVAL_MASK_ROUNDS="2",
        ),
        input_processing="Top-4 exact 980px tiles + forced overview",
        rope="YaRN factor 8",
        max_context=131_072,
        position_mode="strided",
        crop="no",
        overview="yes (force-kept)",
        truncation="whole-image retrieval",
        ocr="prompt-only retrieval fusion",
        kv_policy=(
            "whole-image Top-4 retrieval; bidirectional masked scoring; "
            "compression off"
        ),
        retrieval_query="operation instruction; complementary mask rounds=2",
        full_page_tile_size=980,
    ),
    "unscaled128k-ocr": BenchmarkSpec(
        name="unscaled128k-ocr",
        description="Unscaled 128K extrapolation with full-resolution tiles",
        launcher=("d2f_vllm/mllm_lladao_gui_unscaled_fullres_no_truncation_ocr.sh"),
        dataset="long100",
        output_variable="RESULT_ROOT",
        raw_output="model",
        final_output="fused",
        environment=pairs(KV_CACHE_CAPACITY="65536"),
        input_processing="all exact 980px source tiles",
        rope="none (unscaled extrapolation)",
        max_context=131_072,
        position_mode="strided",
        crop="no",
        overview="no",
        truncation="no",
        ocr="prompt-only retrieval fusion",
        kv_policy="dense; compression off",
        retrieval_query="visible operation instruction",
        full_page_tile_size=980,
    ),
    "yarn128k-fullres-ocr": BenchmarkSpec(
        name="yarn128k-fullres-ocr",
        description="YaRN 128K exact source tiles without overview",
        launcher=("d2f_vllm/mllm_lladao_gui_yarn_fullres_no_truncation_ocr.sh"),
        dataset="native16k100",
        output_variable="RESULT_ROOT",
        raw_output="model",
        final_output="fused",
        environment=pairs(KV_CACHE_CAPACITY="32768"),
        input_processing="all exact 980px source tiles",
        rope="YaRN factor 8",
        max_context=131_072,
        position_mode="strided",
        crop="no",
        overview="no",
        truncation="no",
        ocr="prompt-only retrieval fusion",
        kv_policy="dense; compression off",
        retrieval_query="visible operation instruction",
        full_page_tile_size=980,
    ),
    "native16k-fullres-ocr": BenchmarkSpec(
        name="native16k-fullres-ocr",
        description="Native 16K exact strided tiles without truncation",
        launcher=("d2f_vllm/mllm_lladao_gui_native_fullres_no_truncation_ocr.sh"),
        dataset="native16k100",
        output_variable="RESULT_ROOT",
        raw_output="model",
        final_output="fused",
        environment=pairs(),
        input_processing="all exact 980px source tiles",
        rope="none (checkpoint-native)",
        max_context=16_384,
        position_mode="strided",
        crop="no",
        overview="no",
        truncation="no; prefiltered to native capacity",
        ocr="prompt-only retrieval fusion",
        kv_policy="dense; compression off",
        retrieval_query="visible operation instruction",
        full_page_tile_size=980,
    ),
    "native16k-truncated-ocr": BenchmarkSpec(
        name="native16k-truncated-ocr",
        description="Native 16K full-resolution complete-tile truncation",
        launcher=("d2f_vllm/mllm_lladao_gui_native_fullres_truncated_ocr.sh"),
        dataset="native16k100",
        output_variable="RESULT_ROOT",
        raw_output="model",
        final_output="fused",
        environment=pairs(),
        input_processing="exact 980px source tiles within native capacity",
        rope="none (checkpoint-native)",
        max_context=16_384,
        position_mode="native",
        crop="no",
        overview="no",
        truncation="complete trailing tiles when required",
        ocr="prompt-only retrieval fusion",
        kv_policy="dense; compression off",
        retrieval_query="visible operation instruction",
        full_page_tile_size=980,
    ),
    "original16k-native": BenchmarkSpec(
        name="original16k-native",
        description="Controlled checkpoint-native resized-image arm",
        launcher="d2f_vllm/mllm_lladao_gui_long_context.sh",
        dataset="long100",
        output_variable="OUTPUT_DIR",
        raw_output=".",
        final_output=None,
        environment=pairs(
            MODE="original",
            INPUT_MODE="native_resize",
            KV_CACHE_CAPACITY="16384",
            KV_CACHE_COMPRESSION="0",
        ),
        input_processing="checkpoint-native single-image resize",
        rope="none (checkpoint-native)",
        max_context=16_384,
        position_mode="native",
        crop="no",
        overview="no",
        truncation="native resize",
        ocr="no",
        kv_policy="dense; compression off",
        retrieval_query="none",
    ),
    "yarn128k-native": BenchmarkSpec(
        name="yarn128k-native",
        description="Controlled YaRN arm on checkpoint-native resized images",
        launcher="d2f_vllm/mllm_lladao_gui_long_context.sh",
        dataset="long100",
        output_variable="OUTPUT_DIR",
        raw_output=".",
        final_output=None,
        environment=pairs(
            MODE="yarn",
            INPUT_MODE="native_resize",
            MAX_MODEL_LEN="131072",
            KV_CACHE_CAPACITY="16384",
            KV_CACHE_COMPRESSION="0",
        ),
        input_processing="checkpoint-native single-image resize",
        rope="YaRN factor 8",
        max_context=131_072,
        position_mode="native",
        crop="no",
        overview="no",
        truncation="native resize",
        ocr="no",
        kv_policy="dense 16K resident; compression off",
        retrieval_query="none",
    ),
    "unscaled128k-sequential": BenchmarkSpec(
        name="unscaled128k-sequential",
        description="True-long unscaled 128K controlled arm",
        launcher="d2f_vllm/mllm_lladao_gui_long_context.sh",
        dataset="long100",
        output_variable="OUTPUT_DIR",
        raw_output=".",
        final_output=None,
        environment=pairs(
            MODE="unscaled",
            INPUT_MODE="full_page",
            FULL_PAGE_POSITION_MODE="sequential",
            FULL_PAGE_OVERVIEW="0",
            FULL_PAGE_TRUNCATION="0",
            MAX_MODEL_LEN="131072",
            KV_CACHE_CAPACITY="65536",
            KV_CACHE_COMPRESSION="0",
        ),
        input_processing="all exact 980px source tiles",
        rope="none (unscaled extrapolation)",
        max_context=131_072,
        position_mode="sequential",
        crop="no",
        overview="no",
        truncation="no",
        ocr="no",
        kv_policy="dense 65,536 resident; compression off",
        retrieval_query="none",
        full_page_tile_size=980,
    ),
    "yarn128k-sequential": BenchmarkSpec(
        name="yarn128k-sequential",
        description="True-long YaRN 128K controlled arm",
        launcher="d2f_vllm/mllm_lladao_gui_long_context.sh",
        dataset="long100",
        output_variable="OUTPUT_DIR",
        raw_output=".",
        final_output=None,
        environment=pairs(
            MODE="yarn",
            INPUT_MODE="full_page",
            FULL_PAGE_POSITION_MODE="sequential",
            FULL_PAGE_OVERVIEW="0",
            FULL_PAGE_TRUNCATION="0",
            MAX_MODEL_LEN="131072",
            KV_CACHE_CAPACITY="65536",
            KV_CACHE_COMPRESSION="0",
        ),
        input_processing="all exact 980px source tiles",
        rope="YaRN factor 8",
        max_context=131_072,
        position_mode="sequential",
        crop="no",
        overview="no",
        truncation="no",
        ocr="no",
        kv_policy="dense 65,536 resident; compression off",
        retrieval_query="none",
        full_page_tile_size=980,
    ),
}

BENCHMARKS["yarn128k-kv-top4-causal-ocr"] = replace(
    BENCHMARKS["yarn128k-kv-top4-ocr"],
    name="yarn128k-kv-top4-causal-ocr",
    description=(
        "YaRN 128K with legacy causal next-token Top-4 image KV retrieval"
    ),
    environment=pairs(
        KV_CACHE_CAPACITY="65536",
        KV_RETRIEVAL_TOPK_IMAGES="4",
        KV_RETRIEVAL_SCORE_MODE="causal_self_information",
        KV_RETRIEVAL_MASK_ROUNDS="2",
    ),
    kv_policy=(
        "whole-image Top-4 retrieval; legacy causal next-token scoring; "
        "compression off"
    ),
    retrieval_query="operation instruction; clear causal next-token scoring",
)

for tile_size in (686, 490):
    name = f"yarn128k-ocr-tile{tile_size}"
    BENCHMARKS[name] = replace(
        BENCHMARKS["yarn128k-ocr"],
        name=name,
        description=(
            "YaRN 128K with all source tiles, overview, OCR, and "
            f"{tile_size}px full-page tiles"
        ),
        input_processing=(
            f"all exact {tile_size}px tiles + whole-page overview"
        ),
        full_page_tile_size=tile_size,
    )


SUITES = {
    "deployment": SuiteSpec(
        name="deployment",
        description="Three deployable long-page OCR configurations",
        arms=(
            ("native-ocr-crop", "long100"),
            ("yarn128k-ocr", "long100"),
            ("unscaled128k-ocr", "long100"),
        ),
    ),
    "native16k-five-way": SuiteSpec(
        name="native16k-five-way",
        description="Five configurations on the fixed native-16K seed42 set",
        arms=(
            ("native-ocr-crop", "native16k100"),
            ("yarn128k-ocr", "native16k100"),
            ("native16k-fullres-ocr", "native16k100"),
            ("yarn128k-fullres-ocr", "native16k100"),
            ("native16k-truncated-ocr", "native16k100"),
        ),
    ),
    "yarn-isolation": SuiteSpec(
        name="yarn-isolation",
        description="Controlled short-position original 16K versus YaRN",
        arms=(
            ("original16k-native", "long100"),
            ("yarn128k-native", "long100"),
        ),
    ),
    "true-long-yarn": SuiteSpec(
        name="true-long-yarn",
        description="Controlled true-long unscaled 128K versus YaRN 128K",
        arms=(
            ("unscaled128k-sequential", "long100"),
            ("yarn128k-sequential", "long100"),
        ),
    ),
    "tile-size-ablation": SuiteSpec(
        name="tile-size-ablation",
        description="Full-page image tiles of 980px, 686px, and 490px",
        arms=(
            ("yarn128k-ocr", "long100"),
            ("yarn128k-ocr-tile686", "long100"),
            ("yarn128k-ocr-tile490", "long100"),
        ),
    ),
    "kv-retrieval-scoring-ablation": SuiteSpec(
        name="kv-retrieval-scoring-ablation",
        description=(
            "Dense context versus causal and bidirectional whole-image "
            "retrieval scoring"
        ),
        arms=(
            ("yarn128k-ocr", "long100"),
            ("yarn128k-kv-top4-causal-ocr", "long100"),
            ("yarn128k-kv-top4-ocr", "long100"),
        ),
    ),
}


def arm_id(benchmark_name: str, dataset_name: str) -> str:
    return f"{dataset_name}--{benchmark_name}"


def resolve_selection(target: str) -> Selection:
    if target in BENCHMARKS:
        spec = BENCHMARKS[target]
        arm = (target, spec.dataset)
        return Selection(target=target, arms=(arm,), groups=((target, (arm,)),))
    if target in SUITES:
        suite = SUITES[target]
        return Selection(
            target=target,
            arms=suite.arms,
            groups=((suite.name, suite.arms),),
        )
    if target == "all":
        ordered: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        groups = []
        for suite in SUITES.values():
            groups.append((suite.name, suite.arms))
            for arm in suite.arms:
                if arm not in seen:
                    seen.add(arm)
                    ordered.append(arm)
        return Selection(target=target, arms=tuple(ordered), groups=tuple(groups))
    available = sorted((*BENCHMARKS, *SUITES, "all"))
    raise ValueError(
        f"unknown benchmark or suite {target!r}; choose one of: " + ", ".join(available)
    )


def validate_limit(limit: int) -> None:
    if not 1 <= limit <= MAX_SAMPLES:
        raise ValueError(f"--limit must be in [1, {MAX_SAMPLES}]")


def validate_gpu(gpu: str) -> None:
    if not gpu.isdigit():
        raise ValueError("--gpu must be one non-negative CUDA device index")


def validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError(
            "--run-id must contain only letters, numbers, dot, underscore, and hyphen"
        )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"expected JSON object at {path}:{line_number}")
            yield value


def dataset_fingerprint(
    benchmark_root: Path,
    limit: int,
) -> dict[str, Any]:
    manifest_path = benchmark_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing benchmark manifest: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    try:
        relative = manifest["benchmarks"][BENCHMARK]["path"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"{manifest_path} does not define {BENCHMARK}") from exc
    records_path = benchmark_root / relative
    if not records_path.is_file():
        raise FileNotFoundError(f"missing benchmark records: {records_path}")
    sample_ids: list[str] = []
    for index, row in enumerate(iter_jsonl(records_path)):
        if index >= limit:
            break
        if "sample_id" not in row:
            raise RuntimeError(f"sample without sample_id in {records_path}")
        sample_ids.append(str(row["sample_id"]))
    if len(sample_ids) != limit:
        raise RuntimeError(
            f"{records_path} contains only {len(sample_ids)} rows; {limit} requested"
        )
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(
            f"duplicate sample IDs in first {limit} rows of {records_path}"
        )
    encoded_ids = ("\n".join(sample_ids) + "\n").encode()
    return {
        "benchmark_root": str(benchmark_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "records": str(records_path.resolve()),
        "samples": len(sample_ids),
        "sample_ids_sha256": sha256_bytes(encoded_ids),
        "sample_ids": sample_ids,
    }


def prediction_records(predictions_dir: Path) -> list[dict[str, Any]]:
    shard_dir = predictions_dir / BENCHMARK
    shards = sorted(shard_dir.glob("part-*.jsonl"))
    if not shards:
        raise FileNotFoundError(
            f"no prediction shards for {BENCHMARK} below {predictions_dir}"
        )
    values: list[dict[str, Any]] = []
    for shard in shards:
        for row in iter_jsonl(shard):
            if "sample_id" not in row:
                raise RuntimeError(f"prediction without sample_id in {shard}")
            values.append(row)
    sample_ids = [str(row["sample_id"]) for row in values]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(f"duplicate prediction sample IDs below {predictions_dir}")
    return values


def prediction_ids(predictions_dir: Path) -> list[str]:
    return [
        str(row["sample_id"])
        for row in prediction_records(predictions_dir)
    ]


def validate_predictions(
    predictions_dir: Path,
    expected_ids: Sequence[str],
) -> None:
    actual = prediction_ids(predictions_dir)
    if actual == list(expected_ids):
        return
    missing = sorted(set(expected_ids) - set(actual))
    unexpected = sorted(set(actual) - set(expected_ids))
    if not missing and not unexpected:
        raise RuntimeError(
            f"prediction order differs from the benchmark below {predictions_dir}"
        )
    raise RuntimeError(
        f"prediction coverage mismatch below {predictions_dir}: "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def score_payload(predictions_dir: Path) -> dict[str, Any]:
    path = predictions_dir / "scores" / "results.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing score output: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        payload["benchmarks"][BENCHMARK]
        payload["runtime"][BENCHMARK]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"incomplete score payload: {path}") from exc
    return payload


def summary_value(
    runtime: dict[str, Any],
    field: str,
    aggregate: str,
) -> float | int | None:
    value = runtime.get(field)
    if not isinstance(value, dict):
        return None
    result = value.get(aggregate)
    return result if isinstance(result, (int, float)) else None


def pct(value: Any) -> float | None:
    return 100.0 * float(value) if isinstance(value, (int, float)) else None


def numeric_mean(rows: Sequence[dict[str, Any]], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if isinstance(row.get(field), (int, float))
    ]
    return sum(values) / len(values) if values else None


def target_tile_index(record: dict[str, Any]) -> int | None:
    layout = record.get("tile_layout")
    if not isinstance(layout, list) or not layout:
        return None
    provenance = record.get("provenance")
    source_box = (
        provenance.get("source_bbox_xyxy")
        if isinstance(provenance, dict)
        else None
    )
    if not (
        isinstance(source_box, list)
        and len(source_box) == 4
        and all(isinstance(value, (int, float)) for value in source_box)
    ):
        normalized_box = record.get("target_bbox_1000")
        width = record.get("image_width")
        height = record.get("image_height")
        if not (
            isinstance(normalized_box, list)
            and len(normalized_box) == 4
            and all(
                isinstance(value, (int, float))
                for value in normalized_box
            )
            and isinstance(width, (int, float))
            and isinstance(height, (int, float))
        ):
            return None
        source_box = [
            normalized_box[0] * width / 1000.0,
            normalized_box[1] * height / 1000.0,
            normalized_box[2] * width / 1000.0,
            normalized_box[3] * height / 1000.0,
        ]
    center_x = (float(source_box[0]) + float(source_box[2])) / 2.0
    center_y = (float(source_box[1]) + float(source_box[3])) / 2.0
    for ordinal, tile in enumerate(layout):
        if not isinstance(tile, dict):
            continue
        box = tile.get("box_xyxy")
        if not (
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(value, (int, float)) for value in box)
        ):
            continue
        if (
            float(box[0]) <= center_x < float(box[2])
            and float(box[1]) <= center_y < float(box[3])
        ):
            index = tile.get("index", ordinal)
            return int(index) if isinstance(index, (int, float)) else ordinal
    return None


def retrieval_target_tile_recall(
    predictions: Sequence[dict[str, Any]],
    records_path: Path,
    expected_ids: Sequence[str],
) -> float | None:
    retrieval_rows = [
        row
        for row in predictions
        if row.get("kv_cache_retrieval_enabled") is True
    ]
    if not retrieval_rows:
        return None
    expected = set(expected_ids)
    records = {
        str(row["sample_id"]): row
        for row in iter_jsonl(records_path)
        if str(row.get("sample_id")) in expected
    }
    hits = 0
    for prediction in retrieval_rows:
        sample_id = str(prediction["sample_id"])
        record = records.get(sample_id)
        target_index = target_tile_index(record) if record is not None else None
        selected = prediction.get("kv_cache_retrieval_indices")
        if target_index is None or not isinstance(selected, list):
            raise RuntimeError(
                "cannot audit retrieval target-tile recall for "
                f"{sample_id}"
            )
        hits += int(target_index in {int(index) for index in selected})
    return 100.0 * hits / len(retrieval_rows)


def resolved_output(result_dir: Path, relative: str) -> Path:
    return result_dir if relative == "." else result_dir / relative


def extract_arm_rows(
    arm: dict[str, Any],
    expected_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result_dir = Path(arm["result_dir"])
    raw_dir = resolved_output(result_dir, arm["raw_output"])
    final_dir = (
        resolved_output(result_dir, arm["final_output"])
        if arm.get("final_output")
        else raw_dir
    )
    validate_predictions(raw_dir, expected_ids)
    if final_dir != raw_dir:
        validate_predictions(final_dir, expected_ids)
    raw_payload = score_payload(raw_dir)
    final_payload = score_payload(final_dir) if final_dir != raw_dir else raw_payload
    raw_metrics = raw_payload["benchmarks"][BENCHMARK]
    final_metrics = final_payload["benchmarks"][BENCHMARK]
    runtime = raw_payload["runtime"][BENCHMARK]
    raw_predictions = prediction_records(raw_dir)
    retrieval_predictions = [
        row
        for row in raw_predictions
        if row.get("kv_cache_retrieval_enabled") is True
    ]
    target_recall = retrieval_target_tile_recall(
        raw_predictions,
        Path(arm["fingerprint"]["records"]),
        expected_ids,
    )
    dense = summary_value(runtime, "dense_prefix_tokens", "mean")
    resident = summary_value(runtime, "cached_prefix_tokens", "mean")
    reduction = None
    if (
        isinstance(dense, (int, float))
        and dense > 0
        and isinstance(resident, (int, float))
    ):
        reduction = 100.0 * (1.0 - float(resident) / float(dense))
    max_prefill = summary_value(runtime, "max_prefill_position", "max")
    max_generation = summary_value(runtime, "max_generation_position", "max")
    rope_candidates = [
        float(value)
        for value in (max_prefill, max_generation)
        if isinstance(value, (int, float))
    ]
    protocol = arm["protocol"]
    quality = {
        "Configuration": arm["benchmark"],
        "Tile size (px)": protocol["full_page_tile_size"],
        "Block size": protocol["block_size"],
        "Dataset": arm["dataset"],
        "Samples": final_metrics.get("num_samples"),
        "Raw SSR (%)": pct(raw_metrics.get("ssr_point_only")),
        "Final SSR (%)": pct(final_metrics.get("ssr_point_only")),
        "Joint SSR (%)": pct(final_metrics.get("joint_step_success")),
        "Action F1 (%)": pct(final_metrics.get("action_f1_macro_present")),
        "Parse rate (%)": pct(final_metrics.get("parse_rate")),
        "Target tile recall (%)": target_recall,
        "Final stage": "OCR/fused" if final_dir != raw_dir else "model",
    }
    performance = {
        "Configuration": arm["benchmark"],
        "Tile size (px)": protocol["full_page_tile_size"],
        "Block size": protocol["block_size"],
        "Dataset": arm["dataset"],
        "Mean convergence steps": summary_value(
            raw_metrics, "convergence_steps", "mean"
        ),
        "Mean end-to-end latency (s)": summary_value(
            raw_metrics, "latency_seconds", "mean"
        ),
        "P95 end-to-end latency (s)": summary_value(
            raw_metrics, "latency_seconds", "p95"
        ),
        "Mean model latency (s)": summary_value(
            runtime, "model_elapsed_seconds", "mean"
        ),
        "P95 model latency (s)": summary_value(runtime, "model_elapsed_seconds", "p95"),
        "Mean retrieval latency (s)": numeric_mean(
            raw_predictions,
            "kv_cache_retrieval_seconds",
        ),
        "Mean tokens/s": summary_value(runtime, "total_tokens_per_second", "mean"),
        "Mean retrieval candidates": numeric_mean(
            retrieval_predictions,
            "kv_cache_retrieval_candidates",
        ),
        "Mean selected images": numeric_mean(
            retrieval_predictions,
            "kv_cache_retrieval_selected",
        ),
        "Mean resident KV": resident,
        "Mean dense prefix": dense,
        "Max dense prefix": summary_value(runtime, "dense_prefix_tokens", "max"),
        "KV reduction (%)": reduction,
        "Max actual RoPE": max(rope_candidates) if rope_candidates else None,
        "Mean input images": summary_value(runtime, "input_images", "mean"),
        "Max input images": summary_value(runtime, "input_images", "max"),
        "Peak allocated (GiB)": summary_value(
            runtime, "peak_memory_allocated_gib", "max"
        ),
        "Errors": runtime.get("errors"),
    }
    configuration = {
        "Configuration": arm["benchmark"],
        "Dataset": arm["dataset"],
        "Benchmark root": arm["benchmark_root"],
        "Sample ID SHA-256": arm["fingerprint"]["sample_ids_sha256"],
        "Manifest SHA-256": arm["fingerprint"]["manifest_sha256"],
        "Revision": arm["revision"],
        "Worktree": (
            "dirty"
            if arm.get("worktree_dirty")
            else "clean"
            if arm.get("worktree_dirty") is False
            else "unknown"
        ),
        "Input processing": protocol["input_processing"],
        "Tile size (px)": protocol["full_page_tile_size"],
        "Block size": protocol["block_size"],
        "RoPE": protocol["rope"],
        "Max context": protocol["max_context"],
        "Position mode": protocol["position_mode"],
        "Crop": protocol["crop"],
        "Overview": protocol["overview"],
        "Truncation": protocol["truncation"],
        "OCR": protocol["ocr"],
        "KV policy": protocol["kv_policy"],
        "Retrieval query": protocol["retrieval_query"],
        "Retrieval score mode": protocol.get("retrieval_score_mode"),
        "Retrieval mask rounds": protocol.get("retrieval_mask_rounds"),
        "Retrieval Top-K": protocol.get("retrieval_topk_images"),
    }
    return quality, performance, configuration


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, groups: dict[str, list[dict[str, Any]]]) -> None:
    rows = [
        {"Group": group, **row}
        for group, group_rows in groups.items()
        for row in group_rows
    ]
    if not rows:
        raise RuntimeError(f"cannot write empty report: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_report(
    title: str,
    groups: dict[str, list[dict[str, Any]]],
) -> str:
    lines = [f"# {title}", ""]
    for group, rows in groups.items():
        if not rows:
            continue
        columns = list(rows[0])
        lines.extend(
            [
                f"## {group}",
                "",
                "| " + " | ".join(columns) + " |",
                "|" + "|".join("---" for _ in columns) + "|",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                + " | ".join(markdown_value(row.get(column)) for column in columns)
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def build_reports(run_dir: Path) -> dict[str, Any]:
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"missing unified run manifest: {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    arms_by_id = {arm["id"]: arm for arm in run.get("arms", [])}
    incomplete = [
        arm["id"] for arm in arms_by_id.values() if arm.get("status") != "completed"
    ]
    if incomplete:
        raise RuntimeError(
            "cannot report an incomplete run; unfinished arms: " + ", ".join(incomplete)
        )

    extracted: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for identifier, arm in arms_by_id.items():
        current = dataset_fingerprint(
            Path(arm["benchmark_root"]),
            int(run["limit"]),
        )
        saved = arm["fingerprint"]
        for key in ("manifest_sha256", "sample_ids_sha256", "samples"):
            if current[key] != saved[key]:
                raise RuntimeError(
                    f"dataset fingerprint changed for {identifier}: {key}"
                )
        extracted[identifier] = extract_arm_rows(arm, current["sample_ids"])

    quality_groups: dict[str, list[dict[str, Any]]] = {}
    performance_groups: dict[str, list[dict[str, Any]]] = {}
    protocol_groups: dict[str, list[dict[str, Any]]] = {}
    for group in run["groups"]:
        name = group["name"]
        identifiers = group["arms"]
        fingerprints = {
            (
                arms_by_id[identifier]["fingerprint"]["manifest_sha256"],
                arms_by_id[identifier]["fingerprint"]["sample_ids_sha256"],
            )
            for identifier in identifiers
        }
        if len(fingerprints) != 1:
            raise RuntimeError(
                f"group {name} mixes different sample IDs and cannot be compared"
            )
        quality_groups[name] = [extracted[value][0] for value in identifiers]
        performance_groups[name] = [extracted[value][1] for value in identifiers]
        protocol_groups[name] = [extracted[value][2] for value in identifiers]

    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    reports = {
        "quality": quality_groups,
        "performance": performance_groups,
        "protocol": protocol_groups,
    }
    for name, groups in reports.items():
        atomic_write_json(
            tables / f"{name}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run["run_id"],
                "target": run["target"],
                "groups": groups,
            },
        )
        write_csv(tables / f"{name}.csv", groups)
        (tables / f"{name}.md").write_text(
            markdown_report(
                f"{run['target']} GUI benchmark {name}",
                groups,
            ),
            encoding="utf-8",
        )
    (tables / "README.md").write_text(
        "\n".join(
            [
                f"# GUI benchmark report: {run['run_id']}",
                "",
                f"- Target: `{run['target']}`",
                f"- Samples per arm: `{run['limit']}`",
                f"- GPU: `{run['gpu']}`",
                f"- Revision: `{run['revision']}`",
                "",
                "- [Quality](quality.md)",
                "- [Performance](performance.md)",
                "- [Protocol and sample fingerprints](protocol.md)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return reports


def git_revision(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_worktree_dirty(repo: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout)


def default_run_id(target: str) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return f"{target}-{timestamp}"


def protocol_dict(spec: BenchmarkSpec) -> dict[str, Any]:
    environment = dict(spec.environment)
    score_mode = environment.get("KV_RETRIEVAL_SCORE_MODE")
    return {
        "input_processing": spec.input_processing,
        "rope": spec.rope,
        "max_context": spec.max_context,
        "position_mode": spec.position_mode,
        "crop": spec.crop,
        "overview": spec.overview,
        "truncation": spec.truncation,
        "ocr": spec.ocr,
        "kv_policy": spec.kv_policy,
        "retrieval_query": spec.retrieval_query,
        "retrieval_score_mode": score_mode or "disabled",
        "retrieval_mask_rounds": (
            int(environment["KV_RETRIEVAL_MASK_ROUNDS"])
            if score_mode == "masked_self_information"
            else 0
        ),
        "retrieval_topk_images": (
            int(environment["KV_RETRIEVAL_TOPK_IMAGES"])
            if score_mode is not None
            else 0
        ),
        "block_size": spec.block_size,
        "full_page_tile_size": spec.full_page_tile_size,
    }


def build_arm_record(
    *,
    benchmark_name: str,
    dataset_name: str,
    benchmark_root: Path,
    result_dir: Path,
    log_dir: Path,
    repo: Path,
    root: Path,
    lladao_repo: Path,
    limit: int,
    gpu: str,
    run_id: str,
    revision: str,
    fingerprint: dict[str, Any],
    worktree_dirty: bool = False,
) -> dict[str, Any]:
    spec = BENCHMARKS[benchmark_name]
    identifier = arm_id(benchmark_name, dataset_name)
    environment = {
        "ROOT": str(root),
        "REPO": str(repo),
        "LLADAO_REPO": str(lladao_repo),
        "BENCHMARK_ROOT": str(benchmark_root),
        "LIMIT": str(limit),
        "GPU": gpu,
        "BLOCK_SIZE": str(spec.block_size),
        "RUN_ID": run_id,
        "REVISION": revision,
        spec.output_variable: str(result_dir),
        "LOG": str(log_dir / f"{identifier}.pipeline.log"),
        "MODEL_LOG": str(log_dir / f"{identifier}.model.log"),
        "OCR_LOG": str(log_dir / f"{identifier}.ocr.log"),
    }
    if spec.full_page_tile_size is not None:
        environment["FULL_PAGE_TILE_SIZE"] = str(spec.full_page_tile_size)
    environment.update(dict(spec.environment))
    saved_fingerprint = {
        key: value for key, value in fingerprint.items() if key != "sample_ids"
    }
    return {
        "id": identifier,
        "benchmark": benchmark_name,
        "dataset": dataset_name,
        "description": spec.description,
        "launcher": str((repo / spec.launcher).resolve()),
        "command": ["bash", str((repo / spec.launcher).resolve())],
        "environment": environment,
        "result_dir": str(result_dir.resolve()),
        "log": str((log_dir / f"{identifier}.log").resolve()),
        "benchmark_root": str(benchmark_root.resolve()),
        "raw_output": spec.raw_output,
        "final_output": spec.final_output,
        "protocol": protocol_dict(spec),
        "fingerprint": saved_fingerprint,
        "revision": revision,
        "worktree_dirty": worktree_dirty,
        "status": "pending",
    }


def stream_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def dataset_root(root: Path, dataset_name: str) -> Path:
    return root / DATASETS[dataset_name].relative_path


def selection_dataset_names(selection: Selection) -> set[str]:
    return {dataset for _, dataset in selection.arms}


def run_selection(args: argparse.Namespace) -> Path | None:
    validate_limit(args.limit)
    validate_gpu(args.gpu)
    selection = resolve_selection(args.target)
    repo = args.repo.expanduser().resolve()
    root = args.root.expanduser().resolve()
    lladao_repo = (
        args.lladao_repo.expanduser().resolve()
        if args.lladao_repo
        else root / "src" / "LLaDA-o"
    )
    if not repo.is_dir():
        raise FileNotFoundError(f"missing D2F repository: {repo}")
    if not lladao_repo.is_dir():
        raise FileNotFoundError(f"missing LLaDA-o repository: {lladao_repo}")
    revision = git_revision(repo)
    worktree_dirty = git_worktree_dirty(repo)
    selected_datasets = selection_dataset_names(selection)
    if args.benchmark_root and len(selected_datasets) != 1:
        raise ValueError(
            "--benchmark-root is only valid for a benchmark or suite "
            "that uses one dataset"
        )
    override_root = (
        args.benchmark_root.expanduser().resolve() if args.benchmark_root else None
    )
    roots = {
        name: override_root or dataset_root(root, name) for name in selected_datasets
    }
    fingerprints = {
        name: dataset_fingerprint(path, args.limit) for name, path in roots.items()
    }
    run_id = args.run_id or default_run_id(selection.target)
    validate_run_id(run_id)
    output_parent = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else root / "results" / "gui-benchmarks"
    )
    run_dir = output_parent / run_id
    log_dir = run_dir / "logs"

    records = []
    for benchmark_name, dataset_name in selection.arms:
        spec = BENCHMARKS[benchmark_name]
        launcher = repo / spec.launcher
        if not launcher.is_file():
            raise FileNotFoundError(f"missing launcher: {launcher}")
        records.append(
            build_arm_record(
                benchmark_name=benchmark_name,
                dataset_name=dataset_name,
                benchmark_root=roots[dataset_name],
                result_dir=run_dir / "arms" / arm_id(benchmark_name, dataset_name),
                log_dir=log_dir,
                repo=repo,
                root=root,
                lladao_repo=lladao_repo,
                limit=args.limit,
                gpu=args.gpu,
                run_id=run_id,
                revision=revision,
                fingerprint=fingerprints[dataset_name],
                worktree_dirty=worktree_dirty,
            )
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "target": selection.target,
                    "run_dir": str(run_dir),
                    "revision": revision,
                    "worktree_dirty": worktree_dirty,
                    "limit": args.limit,
                    "gpu": args.gpu,
                    "arms": [
                        {
                            "id": record["id"],
                            "command": shlex.join(record["command"]),
                            "environment": record["environment"],
                            "sample_ids_sha256": record["fingerprint"][
                                "sample_ids_sha256"
                            ],
                        }
                        for record in records
                    ],
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return None

    if run_dir.exists():
        raise FileExistsError(
            f"run directory already exists; refusing to reuse results: {run_dir}"
        )
    log_dir.mkdir(parents=True)
    groups = [
        {
            "name": name,
            "description": (
                SUITES[name].description
                if name in SUITES
                else BENCHMARKS[name].description
            ),
            "arms": [
                arm_id(benchmark_name, dataset_name)
                for benchmark_name, dataset_name in arms
            ],
        }
        for name, arms in selection.groups
    ]
    run = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "target": selection.target,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "completed_at": None,
        "root": str(root),
        "repo": str(repo),
        "lladao_repo": str(lladao_repo),
        "revision": revision,
        "worktree_dirty": worktree_dirty,
        "limit": args.limit,
        "gpu": args.gpu,
        "groups": groups,
        "arms": records,
    }
    run_path = run_dir / "run.json"
    atomic_write_json(run_path, run)

    base_environment = os.environ.copy()
    base_environment["PYTHONUNBUFFERED"] = "1"
    for index, record in enumerate(run["arms"], start=1):
        print(
            f"[{index}/{len(run['arms'])}] running {record['id']} on GPU {args.gpu}",
            flush=True,
        )
        record["status"] = "running"
        record["started_at"] = datetime.now().astimezone().isoformat()
        atomic_write_json(run_path, run)
        environment = base_environment.copy()
        environment.update(record["environment"])
        try:
            status = stream_command(
                record["command"],
                cwd=repo,
                environment=environment,
                log_path=Path(record["log"]),
            )
            if status != 0:
                raise subprocess.CalledProcessError(status, record["command"])
            current = dataset_fingerprint(
                Path(record["benchmark_root"]),
                args.limit,
            )
            extract_arm_rows(record, current["sample_ids"])
        except BaseException as exc:
            record["status"] = "failed"
            record["completed_at"] = datetime.now().astimezone().isoformat()
            record["error"] = f"{type(exc).__name__}: {exc}"
            run["status"] = "failed"
            run["completed_at"] = datetime.now().astimezone().isoformat()
            atomic_write_json(run_path, run)
            raise
        record["status"] = "completed"
        record["completed_at"] = datetime.now().astimezone().isoformat()
        atomic_write_json(run_path, run)

    try:
        build_reports(run_dir)
    except BaseException as exc:
        run["status"] = "failed"
        run["completed_at"] = datetime.now().astimezone().isoformat()
        run["report_error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(run_path, run)
        raise
    run["status"] = "completed"
    run["completed_at"] = datetime.now().astimezone().isoformat()
    run["reports"] = {
        "quality": str((run_dir / "tables" / "quality.md").resolve()),
        "performance": str((run_dir / "tables" / "performance.md").resolve()),
        "protocol": str((run_dir / "tables" / "protocol.md").resolve()),
    }
    atomic_write_json(run_path, run)
    print(f"completed: {run_dir}", flush=True)
    print(f"tables: {run_dir / 'tables' / 'README.md'}", flush=True)
    return run_dir


def print_catalog(as_json: bool) -> None:
    payload = {
        "benchmarks": {
            name: {
                "description": spec.description,
                "default_dataset": spec.dataset,
            }
            for name, spec in BENCHMARKS.items()
        },
        "suites": {
            name: {
                "description": suite.description,
                "arms": [
                    {"benchmark": benchmark, "dataset": dataset}
                    for benchmark, dataset in suite.arms
                ],
            }
            for name, suite in SUITES.items()
        },
        "special": {"all": "run all five suites, deduplicating identical arms"},
    }
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return
    print("Benchmarks:")
    for name, spec in BENCHMARKS.items():
        print(f"  {name:<30} [{spec.dataset}] {spec.description}")
    print("\nSuites:")
    for name, suite in SUITES.items():
        values = ", ".join(benchmark for benchmark, _ in suite.arms)
        print(f"  {name:<30} {suite.description}")
        print(f"  {'':<30} {values}")
    print("\n  all                            run all five suites")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list", help="list benchmarks and suites")
    list_parser.add_argument("--json", action="store_true")

    repository = Path(__file__).resolve().parents[1]
    default_root = Path(os.environ.get("ROOT", "/home/ma-user/work/LLaDA-o"))
    run_parser = subparsers.add_parser(
        "run",
        help="run one benchmark or a predefined suite from scratch",
    )
    run_parser.add_argument("target")
    run_parser.add_argument("--limit", type=int, default=100)
    run_parser.add_argument("--gpu", default="0")
    run_parser.add_argument("--root", type=Path, default=default_root)
    run_parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("REPO", repository)),
    )
    run_parser.add_argument(
        "--lladao-repo",
        type=Path,
        default=(
            Path(os.environ["LLADAO_REPO"]) if "LLADAO_REPO" in os.environ else None
        ),
    )
    run_parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            Path(os.environ["GUI_BENCHMARK_OUTPUT_ROOT"])
            if "GUI_BENCHMARK_OUTPUT_ROOT" in os.environ
            else None
        ),
    )
    run_parser.add_argument(
        "--benchmark-root",
        type=Path,
        help="override the dataset root for a single-dataset target",
    )
    run_parser.add_argument("--run-id")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print commands without creating a run",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="validate and regenerate tables for an existing unified run",
    )
    report_parser.add_argument("run_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "list":
            print_catalog(args.json)
        elif args.command == "run":
            run_selection(args)
        elif args.command == "report":
            build_reports(args.run_dir.expanduser().resolve())
            print(f"tables: {args.run_dir.expanduser().resolve() / 'tables'}")
        else:
            raise AssertionError(f"unhandled command: {args.command}")
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
