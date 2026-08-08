#!/usr/bin/env python3
"""Build target-bearing mobile grounding shards from Planner JSONL data.

The source Planner dataset remains owned by LLaDA-Agent. This converter reads
it by explicit path and writes an independent D2F training/benchmark artifact;
it never changes the source checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


SPLITS = ("train", "validation", "test")
TARGET_ACTIONS = {"click", "long_press"}
CONVERSATION_TYPE = pa.list_(
    pa.struct([pa.field("from", pa.string()), pa.field("value", pa.string())])
)
IMAGE_TYPE = pa.struct(
    [pa.field("bytes", pa.binary()), pa.field("path", pa.string())]
)
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string()),
        pa.field("source", pa.string()),
        pa.field("image", IMAGE_TYPE),
        pa.field("conversations", CONVERSATION_TYPE),
        pa.field("metadata", pa.string()),
    ]
)


class MobileGroundingDataError(ValueError):
    """Raised when a source row cannot satisfy the grounding contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_key(seed: int, split: str, sample_id: str) -> tuple[str, str]:
    payload = f"{seed}\0{split}\0{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), sample_id


def normalized_target(value: Any) -> str:
    target = " ".join(str(value or "").replace("\x00", " ").split())
    target = target.rstrip(" .")
    if not target:
        raise MobileGroundingDataError("target-bearing action has no target text")
    return target[:2_000]


def normalized_bbox(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise MobileGroundingDataError("target_bbox_1000 must contain four values")
    try:
        bbox = [int(round(float(item))) for item in value]
    except (TypeError, ValueError) as error:
        raise MobileGroundingDataError("target_bbox_1000 is not numeric") from error
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise MobileGroundingDataError(f"invalid target_bbox_1000: {bbox}")
    return bbox


def safe_image_path(image_root: Path, value: Any) -> tuple[Path, str]:
    relative = Path(str(value or ""))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise MobileGroundingDataError(f"unsafe relative image path: {value!r}")
    resolved = (image_root / relative).resolve()
    if not resolved.is_relative_to(image_root) or not resolved.is_file():
        raise MobileGroundingDataError(f"source image is missing: {resolved}")
    return resolved, relative.as_posix()


@dataclass(frozen=True)
class GroundingRow:
    sample_id: str
    split: str
    trajectory_id: str
    source_action: str
    target: str
    bbox_1000: tuple[int, int, int, int]
    prompt: str
    answer: str
    image: Path
    source_image: str
    image_width: int
    image_height: int


def parse_grounding_row(
    row: dict[str, Any], *, split: str, image_root: Path
) -> GroundingRow | None:
    if row.get("split") != split:
        raise MobileGroundingDataError(
            f"row {row.get('id')} claims split {row.get('split')!r}, expected {split!r}"
        )
    action = row.get("planner_action")
    if not isinstance(action, dict):
        raise MobileGroundingDataError(f"row {row.get('id')} has no planner_action")
    source_action = str(action.get("action") or "").casefold()
    if source_action not in TARGET_ACTIONS:
        return None
    sample_id = str(row.get("id") or "")
    trajectory_id = str(row.get("trajectory_id") or "")
    if not sample_id or not trajectory_id:
        raise MobileGroundingDataError("target row has no stable ID or trajectory ID")
    target = normalized_target(action.get("target"))
    ground_truth = row.get("ground_truth")
    if not isinstance(ground_truth, dict):
        raise MobileGroundingDataError(f"row {sample_id} has no ground_truth")
    bbox = normalized_bbox(ground_truth.get("target_bbox_1000"))
    image, source_image = safe_image_path(image_root, row.get("image"))
    with Image.open(image) as opened:
        width, height = opened.size
        opened.verify()
    if width <= 0 or height <= 0:
        raise MobileGroundingDataError(f"row {sample_id} has an empty image")
    prompt = f"Click on {target}."
    answer = "lclick [" + ",".join(str(value) for value in bbox) + "]"
    return GroundingRow(
        sample_id=sample_id,
        split=split,
        trajectory_id=trajectory_id,
        source_action=source_action,
        target=target,
        bbox_1000=tuple(bbox),
        prompt=prompt,
        answer=answer,
        image=image,
        source_image=source_image,
        image_width=width,
        image_height=height,
    )


def parquet_record(row: GroundingRow) -> dict[str, Any]:
    return {
        "sample_id": f"mobile:{row.sample_id}",
        "source": "unigui_openmobile_target_grounding",
        "image": {"bytes": row.image.read_bytes(), "path": row.source_image},
        "conversations": [
            {"from": "human", "value": f"<image>\n{row.prompt}"},
            {"from": "gpt", "value": row.answer},
        ],
        "metadata": json.dumps(
            {
                "annotation": "planner-visible-target-v4",
                "bbox_1000": list(row.bbox_1000),
                "grounding_action": "lclick",
                "source_action": row.source_action,
                "source_image": row.source_image,
                "source_sample_id": row.sample_id,
                "split": row.split,
                "target": row.target,
                "trajectory_id": row.trajectory_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


class ShardedParquetWriter:
    def __init__(self, root: Path, shard_size: int) -> None:
        self.root = root
        self.shard_size = shard_size
        self.buffer: list[dict[str, Any]] = []
        self.paths: list[Path] = []
        self.count = 0
        root.mkdir(parents=True, exist_ok=True)

    def write(self, row: GroundingRow) -> None:
        self.buffer.append(parquet_record(row))
        self.count += 1
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        path = self.root / f"shard-{len(self.paths):05d}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(self.buffer, schema=OUTPUT_SCHEMA)
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=6,
            row_group_size=min(256, len(self.buffer)),
            use_dictionary=["source"],
        )
        os.replace(temporary, path)
        self.paths.append(path)
        self.buffer.clear()

    def close(self) -> dict[str, Any]:
        self.flush()
        return {
            "rows": self.count,
            "shards": [
                {
                    "path": path.relative_to(self.root.parent).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in self.paths
            ],
        }


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise MobileGroundingDataError(
                    f"invalid JSON at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise MobileGroundingDataError(
                    f"non-object JSON at {path}:{line_number}"
                )
            yield value


def write_benchmark(
    root: Path,
    *,
    split: str,
    rows: list[GroundingRow],
    seed: int,
    limit: int,
) -> dict[str, Any]:
    selected = sorted(
        rows, key=lambda row: stable_key(seed, split, row.sample_id)
    )[:limit]
    sample_root = root / "samples"
    image_root = root / "images"
    sample_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    name = f"mobile_{split}"
    records_path = sample_root / f"{name}.jsonl"
    digest = hashlib.sha256()
    with records_path.open("wb") as handle:
        for row in selected:
            image_digest = sha256_file(row.image)
            suffix = row.image.suffix.lower() or ".png"
            output_image = image_root / f"{image_digest}{suffix}"
            if not output_image.exists():
                shutil.copy2(row.image, output_image)
            record = {
                "sample_id": f"mobile:{row.sample_id}",
                "benchmark": name,
                "split": split,
                "image": output_image.relative_to(root).as_posix(),
                "image_width": row.image_width,
                "image_height": row.image_height,
                "prompt": row.prompt,
                "native_prompt": row.prompt,
                "target_action": "lclick",
                "target_bbox_1000": list(row.bbox_1000),
                "target_value": "",
                "source_action": row.source_action,
                "source_sample_id": row.sample_id,
                "trajectory_id": row.trajectory_id,
                "input_protocol": "native_resize",
            }
            payload = (
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
            handle.write(payload)
            digest.update(payload)
    return {
        "path": records_path.relative_to(root).as_posix(),
        "rows": len(selected),
        "sha256": digest.hexdigest(),
        "sample_ids_sha256": ordered_ids_sha256(
            f"mobile:{row.sample_id}" for row in selected
        ),
        "selection": "stable-sha256",
        "seed": seed,
        "prompt_protocol": "Click on <planner target>.",
        "paper_comparison_eligible": False,
    }


def prepare(
    *,
    prepared_root: Path,
    image_root: Path,
    output_root: Path,
    seed: int,
    eval_limit: int,
    shard_size: int,
    force: bool,
) -> dict[str, Any]:
    prepared_root = prepared_root.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not 1 <= eval_limit <= 100:
        raise ValueError("eval limit must be in [1, 100]")
    if shard_size <= 0:
        raise ValueError("shard size must be positive")
    manifest_path = prepared_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Planner manifest is missing: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("format") not in {
        "llada-agent-planner-v1",
        "llada-agent-planner-v2",
    }:
        raise MobileGroundingDataError("unsupported Planner dataset manifest")
    if not image_root.is_dir():
        raise FileNotFoundError(f"mobile image root is missing: {image_root}")
    if output_root.exists():
        if not force:
            raise FileExistsError(
                f"output already exists: {output_root}; pass --force to rebuild"
            )
        shutil.rmtree(output_root)
    temporary_root = output_root.with_name(
        output_root.name + f".building-{os.getpid()}"
    )
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    temporary_root.mkdir(parents=True)

    split_rows: dict[str, list[GroundingRow]] = {}
    split_reports: dict[str, Any] = {}
    seen_ids: set[str] = set()
    trajectories_by_split: dict[str, set[str]] = {}
    try:
        for split in SPLITS:
            input_path = prepared_root / f"{split}.jsonl"
            if not input_path.is_file():
                raise FileNotFoundError(f"Planner split is missing: {input_path}")
            writer = ShardedParquetWriter(temporary_root / split, shard_size)
            selected_rows: list[GroundingRow] = []
            source_rows = 0
            for raw in iter_jsonl(input_path):
                source_rows += 1
                parsed = parse_grounding_row(
                    raw, split=split, image_root=image_root
                )
                if parsed is None:
                    continue
                if parsed.sample_id in seen_ids:
                    raise MobileGroundingDataError(
                        f"duplicate source sample ID: {parsed.sample_id}"
                    )
                seen_ids.add(parsed.sample_id)
                selected_rows.append(parsed)
                writer.write(parsed)
            report = writer.close()
            if not selected_rows:
                raise MobileGroundingDataError(
                    f"{split} has no click/long_press rows with target boxes"
                )
            report.update(
                {
                    "source_rows": source_rows,
                    "source_jsonl": str(input_path),
                    "source_jsonl_sha256": sha256_file(input_path),
                    "selected_ids_sha256": ordered_ids_sha256(
                        row.sample_id for row in selected_rows
                    ),
                }
            )
            pinned = (source_manifest.get("prepared_files") or {}).get(split)
            if pinned is not None and (
                pinned.get("file") != input_path.name
                or int(pinned.get("rows", -1)) != source_rows
                or pinned.get("sha256") != report["source_jsonl_sha256"]
            ):
                raise MobileGroundingDataError(
                    f"{split} JSONL does not match its source manifest"
                )
            split_rows[split] = selected_rows
            split_reports[split] = report
            trajectories_by_split[split] = {
                row.trajectory_id for row in selected_rows
            }

        for index, left in enumerate(SPLITS):
            for right in SPLITS[index + 1 :]:
                overlap = trajectories_by_split[left] & trajectories_by_split[right]
                if overlap:
                    raise MobileGroundingDataError(
                        f"trajectory leakage between {left} and {right}: "
                        f"{sorted(overlap)[0]}"
                    )

        benchmark_root = temporary_root / "benchmark"
        benchmarks = {
            f"mobile_{split}": write_benchmark(
                benchmark_root,
                split=split,
                rows=split_rows[split],
                seed=seed,
                limit=eval_limit,
            )
            for split in ("validation", "test")
        }
        benchmark_manifest = {
            "schema_version": 1,
            "format": "lladao-gui-grounding-benchmark-v1",
            "benchmarks": benchmarks,
            "protocol_notes": [
                "Mobile examples use the Planner-visible target text as the grounding query.",
                "Click and long_press source actions are both spatial grounding examples and emit lclick boxes.",
                "At most 100 rows per held-out split are selected by stable SHA-256.",
            ],
        }
        (benchmark_root / "manifest.json").write_text(
            json.dumps(benchmark_manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        output_manifest = {
            "schema_version": 1,
            "format": "lladao-residual-mobile-grounding-v1",
            "seed": seed,
            "source": {
                "prepared_root": str(prepared_root),
                "manifest_sha256": sha256_file(manifest_path),
                "image_root": str(image_root),
                "dataset": source_manifest.get("source"),
            },
            "policy": {
                "actions": sorted(TARGET_ACTIONS),
                "prompt": "Click on <planner target>.",
                "answer": "lclick [x1,y1,x2,y2]",
                "eval_limit_per_split": eval_limit,
                "selection": "stable-sha256",
            },
            "splits": split_reports,
            "benchmark_manifest": "benchmark/manifest.json",
        }
        (temporary_root / "manifest.json").write_text(
            json.dumps(output_manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_root, output_root)
        return output_manifest
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare(
        prepared_root=args.prepared_root,
        image_root=args.image_root,
        output_root=args.output_root,
        seed=args.seed,
        eval_limit=args.eval_limit,
        shard_size=args.shard_size,
        force=args.force,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
