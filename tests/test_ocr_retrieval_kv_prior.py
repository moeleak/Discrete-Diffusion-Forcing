import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "D2F-eval"
    / "ocr_retrieval_kv_prior.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ocr_retrieval_kv_prior",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def exact_similarity(target, detected):
    return float(str(target).casefold() == str(detected).casefold())


def retrieval_fixture():
    prediction = {
        "kv_cache_retrieval_enabled": True,
        "kv_cache_retrieval_indices": [0, 1, 2],
        "kv_cache_retrieval_scores": {
            "0": 0.1,
            "1": 0.9,
            "2": 100.0,
        },
    }
    sample = {
        "tile_layout": [
            {"index": 0, "box_xyxy": [0, 0, 100, 100]},
            {"index": 1, "box_xyxy": [100, 0, 200, 100]},
        ],
        "provenance": {
            "source_bbox_xyxy": [0, 0, 10, 10],
        },
    }
    return prediction, sample


def duplicate_detections():
    return [
        {
            "text": "Submit",
            "confidence": 0.99,
            "bbox_xyxy": [10, 10, 30, 30],
        },
        {
            "text": "Submit",
            "confidence": 0.99,
            "bbox_xyxy": [110, 10, 130, 30],
        },
    ]


def test_ranked_retrieval_tiles_uses_neural_score_and_ignores_overview():
    prediction, sample = retrieval_fixture()

    ranked = MODULE.ranked_retrieval_tiles(prediction, sample)

    assert [index for index, _ in ranked] == [1, 0]


def test_retrieval_prior_disambiguates_duplicate_text_in_top_tile():
    prediction, sample = retrieval_fixture()
    ranked = MODULE.ranked_retrieval_tiles(prediction, sample)

    selected, _, audit = MODULE.select_detection(
        "Submit",
        duplicate_detections(),
        text_similarity=exact_similarity,
        image_size=(200, 100),
        model_reference=None,
        ranked_tiles=ranked,
        minimum_confidence=0.2,
        minimum_similarity=0.68,
        model_proximity_weight=0.0,
        retrieval_proximity_weight=0.0,
        retrieval_rank_weight=0.1,
    )

    assert selected["bbox_xyxy"] == [110, 10, 130, 30]
    assert audit["retrieval_tile_index"] == 1


def test_no_retrieval_falls_back_to_prompt_text_ordering():
    selected, _, audit = MODULE.select_detection(
        "Submit",
        duplicate_detections(),
        text_similarity=exact_similarity,
        image_size=(200, 100),
        model_reference=None,
        ranked_tiles=[],
        minimum_confidence=0.2,
        minimum_similarity=0.68,
        model_proximity_weight=0.1,
        retrieval_proximity_weight=0.2,
        retrieval_rank_weight=0.3,
    )

    assert selected["bbox_xyxy"] == [10, 10, 30, 30]
    assert audit["retrieval_tile_index"] is None


def test_ground_truth_location_does_not_change_retrieval_ranking():
    prediction, sample = retrieval_fixture()
    expected = MODULE.ranked_retrieval_tiles(prediction, sample)
    sample["provenance"]["source_bbox_xyxy"] = [180, 80, 199, 99]
    sample["target_bbox_1000"] = [900, 900, 999, 999]

    assert MODULE.ranked_retrieval_tiles(prediction, sample) == expected


def test_detection_cache_loads_rows_and_rejects_duplicate_ids(tmp_path):
    cache = tmp_path / "detections.jsonl"
    row = {
        "sample_id": "sample-a",
        "detections": duplicate_detections(),
    }
    cache.write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = MODULE.load_detection_cache(cache)

    assert loaded == {"sample-a": duplicate_detections()}
    cache.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.load_detection_cache(cache)


@pytest.mark.parametrize(
    "weights",
    [
        (-0.1, 0.0, 0.0),
        (0.5, 0.5, 0.1),
    ],
)
def test_select_detection_rejects_invalid_prior_weights(weights):
    with pytest.raises(ValueError, match="spatial-prior weights"):
        MODULE.select_detection(
            "Submit",
            duplicate_detections(),
            text_similarity=exact_similarity,
            image_size=(200, 100),
            model_reference=None,
            ranked_tiles=[],
            minimum_confidence=0.2,
            minimum_similarity=0.68,
            model_proximity_weight=weights[0],
            retrieval_proximity_weight=weights[1],
            retrieval_rank_weight=weights[2],
        )
