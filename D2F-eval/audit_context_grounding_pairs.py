#!/usr/bin/env python3
"""Audit clean versus adversarial-hint grounding predictions sample by sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def benchmark_rows(root: Path, benchmark: str) -> tuple[Path, list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    details = (manifest.get("benchmarks") or {}).get(benchmark)
    if not isinstance(details, dict):
        raise ValueError(f"benchmark {benchmark!r} is missing from {manifest_path}")
    source = root / str(details.get("path") or "")
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(details.get("rows", -1)):
        raise ValueError(f"benchmark {benchmark!r} row count disagrees with its manifest")
    return source, rows


def predictions(root: Path, benchmark: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("benchmark") != benchmark:
                continue
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in result:
                raise ValueError(f"duplicate or empty prediction sample ID: {sample_id!r}")
            result[sample_id] = row
    return result


def valid_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) and 0.0 <= item <= 1000.0 for item in result):
        return None
    if result[0] > result[2] or result[1] > result[3]:
        return None
    return result


def center_in(box: list[float], target: Any) -> bool:
    target_box = valid_bbox(target)
    if target_box is None:
        raise ValueError(f"invalid benchmark box: {target!r}")
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    return target_box[0] <= x <= target_box[2] and target_box[1] <= y <= target_box[3]


def parsed(prediction: dict[str, Any]) -> bool:
    return prediction.get("error") is None and prediction.get("parse_error") is None and (
        valid_bbox(prediction.get("predicted_bbox_1000")) is not None
    )


def hit(prediction: dict[str, Any], target: dict[str, Any]) -> bool:
    box = valid_bbox(prediction.get("predicted_bbox_1000"))
    return bool(
        box is not None
        and prediction.get("error") is None
        and prediction.get("parse_error") is None
        and prediction.get("predicted_action") == target.get("target_action")
        and center_in(box, target.get("target_bbox_1000"))
    )


def audit(
    *,
    benchmark_root: Path,
    clean_benchmark: str,
    hard_benchmark: str,
    clean_predictions: Path,
    hard_predictions: Path,
    adapter: Path,
    backbone_sha256: str,
    min_clean_ssr: float,
    min_hard_ssr: float,
    max_hard_ssr_drop: float,
    max_hard_distractor_rate: float,
    min_parse_rate: float,
) -> dict[str, Any]:
    manifest_path = benchmark_root / "manifest.json"
    clean_source, clean_rows = benchmark_rows(benchmark_root, clean_benchmark)
    hard_source, hard_rows = benchmark_rows(benchmark_root, hard_benchmark)
    clean_by_id = {str(row["sample_id"]): row for row in clean_rows}
    hard_by_id = {str(row["sample_id"]): row for row in hard_rows}
    if list(clean_by_id) != list(hard_by_id):
        raise ValueError("clean and hard benchmarks must contain identical ordered sample IDs")
    if not clean_by_id:
        raise ValueError("paired context benchmark is empty")
    for sample_id in clean_by_id:
        clean = clean_by_id[sample_id]
        hard = hard_by_id[sample_id]
        if clean.get("hint_is_hard_negative") is not False:
            raise ValueError(f"clean row {sample_id!r} is not marked clean")
        if hard.get("hint_is_hard_negative") is not True:
            raise ValueError(f"hard row {sample_id!r} is not marked adversarial")
        if clean.get("target_bbox_1000") != hard.get("target_bbox_1000"):
            raise ValueError(f"paired row {sample_id!r} changed its target")
        if valid_bbox(hard.get("hard_negative_bbox_1000")) is None:
            raise ValueError(f"hard row {sample_id!r} has no valid distractor box")

    clean_outputs = predictions(clean_predictions, clean_benchmark)
    hard_outputs = predictions(hard_predictions, hard_benchmark)
    expected_ids = set(clean_by_id)
    for label, outputs in (("clean", clean_outputs), ("hard", hard_outputs)):
        if set(outputs) != expected_ids:
            missing = sorted(expected_ids - set(outputs))
            extra = sorted(set(outputs) - expected_ids)
            raise ValueError(
                f"{label} prediction coverage mismatch: missing={missing[:4]} extra={extra[:4]}"
            )

    clean_hit: dict[str, bool] = {}
    hard_hit: dict[str, bool] = {}
    distractor_hits = 0
    for sample_id in clean_by_id:
        clean_hit[sample_id] = hit(clean_outputs[sample_id], clean_by_id[sample_id])
        hard_hit[sample_id] = hit(hard_outputs[sample_id], hard_by_id[sample_id])
        hard_box = valid_bbox(hard_outputs[sample_id].get("predicted_bbox_1000"))
        if hard_box is not None and center_in(
            hard_box, hard_by_id[sample_id]["hard_negative_bbox_1000"]
        ):
            distractor_hits += 1

    count = len(clean_by_id)
    clean_ssr = sum(clean_hit.values()) / count
    hard_ssr = sum(hard_hit.values()) / count
    clean_parse = sum(parsed(clean_outputs[item]) for item in clean_by_id) / count
    hard_parse = sum(parsed(hard_outputs[item]) for item in hard_by_id) / count
    regressions = [item for item in clean_by_id if clean_hit[item] and not hard_hit[item]]
    recoveries = [item for item in clean_by_id if not clean_hit[item] and hard_hit[item]]
    distractor_rate = distractor_hits / count
    checks = {
        "clean_ssr": clean_ssr >= min_clean_ssr,
        "hard_ssr": hard_ssr >= min_hard_ssr,
        "hard_ssr_drop": clean_ssr - hard_ssr <= max_hard_ssr_drop + 1e-12,
        "hard_distractor_rate": distractor_rate <= max_hard_distractor_rate + 1e-12,
        "clean_parse_rate": clean_parse >= min_parse_rate,
        "hard_parse_rate": hard_parse >= min_parse_rate,
    }
    adapter_model = adapter / "adapter_model.safetensors"
    if not adapter_model.is_file():
        raise ValueError(f"selected adapter weights are missing: {adapter_model}")
    return {
        "schema_version": 1,
        "format": "lladao-context-grounding-pair-audit-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "release_eligible": all(checks.values()),
        "backbone_sha256": backbone_sha256,
        "adapter": {
            "path": str(adapter.resolve()),
            "adapter_model_sha256": sha256_file(adapter_model),
        },
        "benchmark": {
            "root": str(benchmark_root.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "clean": clean_benchmark,
            "clean_source_sha256": sha256_file(clean_source),
            "hard": hard_benchmark,
            "hard_source_sha256": sha256_file(hard_source),
            "samples": count,
        },
        "thresholds": {
            "min_clean_ssr": min_clean_ssr,
            "min_hard_ssr": min_hard_ssr,
            "max_hard_ssr_drop": max_hard_ssr_drop,
            "max_hard_distractor_rate": max_hard_distractor_rate,
            "min_parse_rate": min_parse_rate,
        },
        "checks": checks,
        "metrics": {
            "clean_ssr": clean_ssr,
            "hard_ssr": hard_ssr,
            "hard_ssr_drop": clean_ssr - hard_ssr,
            "clean_parse_rate": clean_parse,
            "hard_parse_rate": hard_parse,
            "hard_distractor_hits": distractor_hits,
            "hard_distractor_rate": distractor_rate,
            "clean_to_hard_regressions": len(regressions),
            "hard_to_clean_recoveries": len(recoveries),
            "paired_hit_consistency": sum(
                clean_hit[item] == hard_hit[item] for item in clean_by_id
            )
            / count,
        },
        "regression_sample_ids": regressions,
        "recovery_sample_ids": recoveries,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--clean-benchmark", required=True)
    parser.add_argument("--hard-benchmark", required=True)
    parser.add_argument("--clean-predictions", type=Path, required=True)
    parser.add_argument("--hard-predictions", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--backbone-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-clean-ssr", type=float, default=0.70)
    parser.add_argument("--min-hard-ssr", type=float, default=0.70)
    parser.add_argument("--max-hard-ssr-drop", type=float, default=0.05)
    parser.add_argument("--max-hard-distractor-rate", type=float, default=0.10)
    parser.add_argument("--min-parse-rate", type=float, default=0.98)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(
        benchmark_root=args.benchmark_root.expanduser().resolve(),
        clean_benchmark=args.clean_benchmark,
        hard_benchmark=args.hard_benchmark,
        clean_predictions=args.clean_predictions.expanduser().resolve(),
        hard_predictions=args.hard_predictions.expanduser().resolve(),
        adapter=args.adapter.expanduser().resolve(),
        backbone_sha256=args.backbone_sha256,
        min_clean_ssr=args.min_clean_ssr,
        min_hard_ssr=args.min_hard_ssr,
        max_hard_ssr_drop=args.max_hard_ssr_drop,
        max_hard_distractor_rate=args.max_hard_distractor_rate,
        min_parse_rate=args.min_parse_rate,
    )
    output = args.output.expanduser().resolve()
    write_json_atomic(output, result)
    print(output)
    if result["status"] != "passed":
        raise SystemExit("context grounding pair quality gate failed")


if __name__ == "__main__":
    main()
