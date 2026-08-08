from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "prepare_unigui_residual_grounding.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_unigui_residual_grounding", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def source_row(split: str, index: int, action: str) -> dict:
    planner_action = {"action": action}
    ground_truth = {"arguments": {"action": action}}
    if action in {"click", "long_press"}:
        planner_action["target"] = f"Button {index}"
        ground_truth["target_bbox_1000"] = [100, 200, 300, 400]
    return {
        "id": f"{split}:{index}",
        "split": split,
        "trajectory_id": f"trajectory-{split}-{index}",
        "image": f"{split}-{index}.png",
        "planner_action": planner_action,
        "ground_truth": ground_truth,
    }


def build_source(tmp_path: Path) -> tuple[Path, Path]:
    prepared = tmp_path / "prepared"
    images = tmp_path / "images"
    prepared.mkdir()
    images.mkdir()
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "format": "llada-agent-planner-v1",
                "source": {"dataset_id": "fixture"},
            }
        ),
        encoding="utf-8",
    )
    for split in MODULE.SPLITS:
        rows = [
            source_row(split, 0, "click"),
            source_row(split, 1, "long_press"),
            source_row(split, 2, "swipe"),
        ]
        for row in rows:
            Image.new("RGB", (32, 48), color=(10, 20, 30)).save(
                images / row["image"]
            )
        (prepared / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    return prepared, images


def test_prepare_builds_hard_label_shards_and_bounded_benchmarks(
    tmp_path: Path,
) -> None:
    prepared, images = build_source(tmp_path)
    output = tmp_path / "output"

    manifest = MODULE.prepare(
        prepared_root=prepared,
        image_root=images,
        output_root=output,
        seed=42,
        eval_limit=1,
        shard_size=1,
        force=False,
    )

    assert manifest["splits"]["train"]["rows"] == 2
    shards = sorted((output / "train").glob("*.parquet"))
    assert len(shards) == 2
    first = pq.read_table(shards[0]).to_pylist()[0]
    assert first["conversations"][0]["value"].startswith(
        "<image>\nClick on Button"
    )
    assert first["conversations"][1]["value"] == "lclick [100,200,300,400]"

    benchmark = json.loads(
        (output / "benchmark/manifest.json").read_text(encoding="utf-8")
    )
    assert benchmark["benchmarks"]["mobile_validation"]["rows"] == 1
    assert benchmark["benchmarks"]["mobile_test"]["rows"] == 1
    test_record = json.loads(
        (output / "benchmark/samples/mobile_test.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert test_record["target_action"] == "lclick"
    assert test_record["target_bbox_1000"] == [100, 200, 300, 400]
    assert (output / "benchmark" / test_record["image"]).is_file()
