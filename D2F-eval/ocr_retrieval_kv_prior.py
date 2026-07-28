#!/usr/bin/env python3
"""Fuse full-page OCR with model and KV-retrieval spatial priors."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def bbox_center(bbox: Sequence[Any]) -> tuple[float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def model_reference_point(
    prediction: Mapping[str, Any],
    image_size: tuple[int, int],
) -> tuple[float, float] | None:
    bbox = prediction.get("predicted_bbox_1000")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    center_x, center_y = bbox_center(bbox)
    return (
        image_size[0] * center_x / 1000.0,
        image_size[1] * center_y / 1000.0,
    )


def ranked_retrieval_tiles(
    prediction: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Return selected source tiles in descending neural retrieval order."""

    layout = sample.get("tile_layout")
    selected = prediction.get("kv_cache_retrieval_indices")
    raw_scores = prediction.get("kv_cache_retrieval_scores")
    if (
        prediction.get("kv_cache_retrieval_enabled") is not True
        or not isinstance(layout, list)
        or not isinstance(selected, list)
        or not isinstance(raw_scores, dict)
    ):
        return []
    boxes: dict[int, tuple[float, float, float, float]] = {}
    for ordinal, tile in enumerate(layout):
        if not isinstance(tile, dict):
            continue
        raw_box = tile.get("box_xyxy")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            continue
        index = tile.get("index", ordinal)
        if not isinstance(index, (int, float)):
            continue
        boxes[int(index)] = tuple(float(value) for value in raw_box)
    scored = []
    for raw_index in selected:
        if not isinstance(raw_index, (int, float)):
            continue
        index = int(raw_index)
        if index not in boxes:
            continue
        raw_score = raw_scores.get(str(index), raw_scores.get(index))
        if not isinstance(raw_score, (int, float)):
            continue
        score = float(raw_score)
        if math.isfinite(score):
            scored.append((index, boxes[index], score))
    scored.sort(key=lambda item: (-item[2], item[0]))
    return [(index, box) for index, box, _ in scored]


def retrieval_reference_point(
    ranked_tiles: Sequence[
        tuple[int, tuple[float, float, float, float]]
    ],
) -> tuple[float, float] | None:
    return bbox_center(ranked_tiles[0][1]) if ranked_tiles else None


def _proximity(
    point: tuple[float, float],
    reference: tuple[float, float] | None,
    image_size: tuple[int, int],
) -> float:
    if reference is None:
        return 0.0
    diagonal = max(1.0, math.hypot(*image_size))
    return max(
        0.0,
        1.0 - math.dist(point, reference) / diagonal,
    )


def _retrieval_rank_prior(
    point: tuple[float, float],
    ranked_tiles: Sequence[
        tuple[int, tuple[float, float, float, float]]
    ],
) -> tuple[float, int | None]:
    count = len(ranked_tiles)
    for rank, (index, box) in enumerate(ranked_tiles):
        x1, y1, x2, y2 = box
        if x1 <= point[0] < x2 and y1 <= point[1] < y2:
            return (count - rank) / max(count, 1), index
    return 0.0, None


def select_detection(
    target: str,
    detections: Iterable[Mapping[str, Any]],
    *,
    text_similarity: Callable[[Any, Any], float],
    image_size: tuple[int, int],
    model_reference: tuple[float, float] | None,
    ranked_tiles: Sequence[
        tuple[int, tuple[float, float, float, float]]
    ],
    minimum_confidence: float,
    minimum_similarity: float,
    model_proximity_weight: float,
    retrieval_proximity_weight: float,
    retrieval_rank_weight: float,
) -> tuple[Mapping[str, Any] | None, float, dict[str, Any]]:
    """Rank OCR candidates without using target coordinates."""

    weights = (
        model_proximity_weight,
        retrieval_proximity_weight,
        retrieval_rank_weight,
    )
    if any(weight < 0.0 for weight in weights) or sum(weights) > 1.0:
        raise ValueError("spatial-prior weights must be non-negative and sum to <= 1")
    retrieval_reference = retrieval_reference_point(ranked_tiles)
    candidates = []
    for ordinal, detection in enumerate(detections):
        confidence = float(detection["confidence"])
        if confidence < minimum_confidence:
            continue
        similarity = float(text_similarity(target, detection["text"]))
        if similarity < minimum_similarity:
            continue
        center = bbox_center(detection["bbox_xyxy"])
        model_prior = _proximity(center, model_reference, image_size)
        retrieval_prior = _proximity(
            center,
            retrieval_reference,
            image_size,
        )
        rank_prior, tile_index = _retrieval_rank_prior(
            center,
            ranked_tiles,
        )
        base_score = 0.90 * similarity + 0.10 * min(
            1.0,
            max(0.0, confidence),
        )
        active_model_weight = (
            model_proximity_weight
            if model_reference is not None
            else 0.0
        )
        active_retrieval_weight = (
            retrieval_proximity_weight
            if retrieval_reference is not None
            else 0.0
        )
        active_rank_weight = (
            retrieval_rank_weight if ranked_tiles else 0.0
        )
        base_weight = 1.0 - (
            active_model_weight
            + active_retrieval_weight
            + active_rank_weight
        )
        score = (
            base_weight * base_score
            + active_model_weight * model_prior
            + active_retrieval_weight * retrieval_prior
            + active_rank_weight * rank_prior
        )
        x1, y1, _, _ = (
            float(value) for value in detection["bbox_xyxy"]
        )
        audit = {
            "base_score": base_score,
            "model_proximity": model_prior,
            "retrieval_proximity": retrieval_prior,
            "retrieval_rank_prior": rank_prior,
            "retrieval_tile_index": tile_index,
        }
        candidates.append(
            (
                score,
                similarity,
                -y1,
                -x1,
                -ordinal,
                detection,
                audit,
            )
        )
    if not candidates:
        return None, 0.0, {}
    selected = max(candidates, key=lambda item: item[:5])
    return selected[5], selected[0], selected[6]


def load_detection_cache(
    path: Path,
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            detections = row.get("detections")
            if not isinstance(sample_id, str) or not isinstance(
                detections,
                list,
            ):
                raise ValueError(
                    f"invalid detection cache row {line_number}"
                )
            if sample_id in rows:
                raise ValueError(
                    f"duplicate detection cache sample_id {sample_id!r}"
                )
            rows[sample_id] = [
                dict(detection)
                for detection in detections
                if isinstance(detection, dict)
            ]
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", default="mind2web_fullpage")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--detections-cache",
        type=Path,
        help=(
            "optional JSONL cache of full-page OCR detections; when set, "
            "EasyOCR is not run again"
        ),
    )
    parser.add_argument(
        "--write-detections-cache",
        type=Path,
        help="write the exact OCR detections used by this run to JSONL",
    )
    parser.add_argument("--languages", default="en")
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--minimum-confidence", type=float, default=0.20)
    parser.add_argument("--minimum-similarity", type=float, default=0.68)
    parser.add_argument("--model-proximity-weight", type=float, default=0.10)
    parser.add_argument(
        "--retrieval-proximity-weight",
        type=float,
        default=0.00,
    )
    parser.add_argument("--retrieval-rank-weight", type=float, default=0.02)
    parser.add_argument("--label-control-offset", type=float, default=40.0)
    args = parser.parse_args()
    if args.limit <= 0 or args.limit > 100:
        parser.error("--limit must be in [1, 100]")
    weights = (
        args.model_proximity_weight,
        args.retrieval_proximity_weight,
        args.retrieval_rank_weight,
    )
    if any(weight < 0.0 for weight in weights) or sum(weights) > 1.0:
        parser.error("spatial-prior weights must be non-negative and sum to <= 1")
    if args.label_control_offset < 0:
        parser.error("--label-control-offset must be non-negative")
    return args


def main() -> None:
    from eval.gui_grounding.ocr_fullpage_retrieval import (
        build_reader,
        detect_tiles,
        format_prediction,
        instruction_target,
        label_points_to_control,
        shift_detection,
        write_jsonl,
    )
    from eval.gui_grounding.score_benchmark import (
        load_predictions,
        load_targets,
    )
    from scripts.data.ocr_target_realignment import (
        OcrDetection,
        scale_bbox,
        text_similarity,
    )

    args = parse_args()
    root = args.benchmark_root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    targets = load_targets(
        root,
        manifest,
        args.benchmark,
        args.limit,
    )
    source_predictions = load_predictions(
        args.predictions_dir.expanduser().resolve(),
        args.benchmark,
    )
    predictions = {
        sample_id: source_predictions[sample_id]
        for sample_id in targets
        if sample_id in source_predictions
    }
    if set(predictions) != set(targets):
        raise RuntimeError(
            "prediction coverage mismatch: "
            f"missing={len(set(targets) - set(predictions))}"
        )
    cached_detections = (
        load_detection_cache(args.detections_cache)
        if args.detections_cache is not None
        else None
    )
    if cached_detections is not None:
        missing_detections = set(targets) - set(cached_detections)
        if missing_detections:
            raise RuntimeError(
                "detection cache coverage mismatch: "
                f"missing={len(missing_detections)}"
            )
        reader = None
    else:
        reader = build_reader(args)
    output_rows = []
    detection_cache_rows = []
    accepted = 0
    started = time.perf_counter()
    for index, (sample_id, sample) in enumerate(targets.items(), 1):
        row = dict(predictions[sample_id])
        action, target_text, value = instruction_target(str(sample["prompt"]))
        if cached_detections is not None:
            detections = [
                OcrDetection(
                    text=str(detection["text"]),
                    confidence=float(detection["confidence"]),
                    bbox_xyxy=tuple(
                        float(value)
                        for value in detection["bbox_xyxy"]
                    ),
                )
                for detection in cached_detections[sample_id]
            ]
        else:
            from PIL import Image

            with Image.open(root / sample["image"]) as source:
                detections = detect_tiles(
                    reader,
                    source.convert("RGB"),
                    sample,
                )
        image_size = (
            int(sample["image_width"]),
            int(sample["image_height"]),
        )
        model_reference = model_reference_point(row, image_size)
        ranked_tiles = ranked_retrieval_tiles(row, sample)
        detection_rows = [
            {
                "text": detection.text,
                "confidence": detection.confidence,
                "bbox_xyxy": list(detection.bbox_xyxy),
            }
            for detection in detections
        ]
        detection_cache_rows.append(
            {
                "sample_id": sample_id,
                "detections": detection_rows,
            }
        )
        selected, score, prior_audit = select_detection(
            target_text,
            detection_rows,
            text_similarity=text_similarity,
            image_size=image_size,
            model_reference=model_reference,
            ranked_tiles=ranked_tiles,
            minimum_confidence=args.minimum_confidence,
            minimum_similarity=args.minimum_similarity,
            model_proximity_weight=args.model_proximity_weight,
            retrieval_proximity_weight=args.retrieval_proximity_weight,
            retrieval_rank_weight=args.retrieval_rank_weight,
        )
        match = (
            OcrDetection(
                text=str(selected["text"]),
                confidence=float(selected["confidence"]),
                bbox_xyxy=tuple(
                    float(value) for value in selected["bbox_xyxy"]
                ),
            )
            if selected is not None
            else None
        )
        raw_match = match
        if match is not None:
            if label_points_to_control(action, target_text):
                match = shift_detection(
                    match,
                    offset_y=args.label_control_offset,
                )
            row["predicted_bbox_1000"] = scale_bbox(
                match.bbox_xyxy,
                *image_size,
            )
            accepted += 1
        bbox = row.get("predicted_bbox_1000")
        has_bbox = isinstance(bbox, (list, tuple)) and len(bbox) == 4
        row["predicted_action"] = action
        row["predicted_value"] = value
        row["parse_error"] = None if has_bbox else "no_bbox"
        row["prediction"] = format_prediction(
            action,
            bbox if has_bbox else None,
            value,
        )
        row["ocr_retrieval"] = {
            "accepted": match is not None,
            "target_text": target_text,
            "matched_text": match.text if match else "",
            "text_score": score,
            "ocr_confidence": match.confidence if match else 0.0,
            "bbox_xyxy": list(match.bbox_xyxy) if match else None,
            "raw_bbox_xyxy": (
                list(raw_match.bbox_xyxy)
                if raw_match is not None
                else None
            ),
            "model_reference_point_xy": (
                list(model_reference) if model_reference else None
            ),
            "retrieval_reference_point_xy": (
                list(retrieval_reference_point(ranked_tiles))
                if ranked_tiles
                else None
            ),
            "retrieval_ranked_tile_indices": [
                tile_index for tile_index, _ in ranked_tiles
            ],
            "model_proximity_weight": args.model_proximity_weight,
            "retrieval_proximity_weight": (
                args.retrieval_proximity_weight
            ),
            "retrieval_rank_weight": args.retrieval_rank_weight,
            "prior_components": prior_audit,
            "label_control_offset": (
                args.label_control_offset
                if match is not None
                and label_points_to_control(action, target_text)
                else 0.0
            ),
            "detections": len(detections),
            "uses_ground_truth_location": False,
        }
        output_rows.append(row)
        if index == 1 or index % 10 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"KV-prior OCR {index}/{len(targets)} "
                f"accepted={accepted} elapsed={elapsed:.1f}s",
                flush=True,
            )
    output = args.output_dir.expanduser().resolve()
    count = write_jsonl(
        output / args.benchmark / "part-00000.jsonl",
        output_rows,
    )
    if args.write_detections_cache is not None:
        write_jsonl(
            args.write_detections_cache.expanduser().resolve(),
            detection_cache_rows,
        )
    (output / "ocr-retrieval-config.json").write_text(
        json.dumps(
            {
                "benchmark_root": str(root),
                "predictions_dir": str(
                    args.predictions_dir.expanduser().resolve()
                ),
                "detections_cache": (
                    str(args.detections_cache.expanduser().resolve())
                    if args.detections_cache is not None
                    else None
                ),
                "write_detections_cache": (
                    str(
                        args.write_detections_cache.expanduser().resolve()
                    )
                    if args.write_detections_cache is not None
                    else None
                ),
                "samples": count,
                "accepted": accepted,
                "minimum_confidence": args.minimum_confidence,
                "minimum_similarity": args.minimum_similarity,
                "model_proximity_weight": args.model_proximity_weight,
                "retrieval_proximity_weight": (
                    args.retrieval_proximity_weight
                ),
                "retrieval_rank_weight": args.retrieval_rank_weight,
                "label_control_offset": args.label_control_offset,
                "uses_prompt_only": True,
                "uses_model_reference": (
                    args.model_proximity_weight > 0.0
                ),
                "uses_kv_retrieval_prior": (
                    args.retrieval_proximity_weight > 0.0
                    or args.retrieval_rank_weight > 0.0
                ),
                "uses_ground_truth_location": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {count} predictions to {output}", flush=True)


if __name__ == "__main__":
    main()
