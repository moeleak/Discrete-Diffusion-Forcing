import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "D2F-eval"
    / "summarize_kv_retrieval_results.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_kv_retrieval_results",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def quality_row(name, *, ssr, recall):
    return {
        "Configuration": name,
        "Final SSR (%)": ssr,
        "Target tile recall (%)": recall,
    }


def performance_row(
    name,
    *,
    latency,
    retrieval,
    resident=15_000,
):
    return {
        "Configuration": name,
        "Mean end-to-end latency (s)": latency,
        "Mean retrieval latency (s)": retrieval,
        "Mean resident KV": resident,
        "KV reduction (%)": 50.0,
        "Peak allocated (GiB)": 48.0,
    }


def write_run(
    root,
    run_id,
    revision,
    rows,
    *,
    fingerprint="fixed100",
    status="completed",
):
    tables = root / "tables"
    tables.mkdir(parents=True)
    quality = [quality_row(name, ssr=ssr, recall=recall) for (
        name,
        ssr,
        recall,
        _,
        _,
    ) in rows]
    performance = [
        performance_row(
            name,
            latency=latency,
            retrieval=retrieval,
        )
        for name, _, _, latency, retrieval in rows
    ]
    for name, values in (
        ("quality", quality),
        ("performance", performance),
    ):
        (tables / f"{name}.json").write_text(
            json.dumps({"groups": {run_id: values}}),
            encoding="utf-8",
        )
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "limit": 100,
                "revision": revision,
                "arms": [
                    {
                        "fingerprint": {
                            "samples": 100,
                            "sample_ids_sha256": fingerprint,
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def make_runs(tmp_path):
    attention = tmp_path / "attention"
    top8 = tmp_path / "top8"
    optimized = tmp_path / "optimized"
    write_run(
        attention,
        "attention",
        "aaa1111",
        [
            (MODULE.ATTENTION_CAUSAL, 70.0, 60.0, 5.0, 3.0),
            (
                MODULE.ATTENTION_BIDIRECTIONAL,
                74.0,
                90.0,
                9.0,
                6.0,
            ),
        ],
    )
    write_run(
        top8,
        "top8",
        "bbb2222",
        [
            (
                MODULE.TOP8_BIDIRECTIONAL,
                75.0,
                95.0,
                10.0,
                6.5,
            )
        ],
    )
    write_run(
        optimized,
        "optimized",
        "ccc3333",
        [
            (
                MODULE.ATTENTION_BIDIRECTIONAL,
                74.0,
                90.0,
                8.0,
                6.0,
            ),
            (
                MODULE.CACHED_BIDIRECTIONAL,
                74.0,
                89.0,
                6.0,
                3.0,
            ),
            (MODULE.OCR_CONTROL, 74.0, 89.0, 6.0, 3.0),
            (MODULE.OCR_PRIOR, 76.0, 89.0, 6.0, 3.0),
        ],
    )
    return attention, top8, optimized


def test_build_report_audits_fingerprint_and_computes_deltas(tmp_path):
    attention, top8, optimized = make_runs(tmp_path)

    report = MODULE.build_report(attention, top8, optimized)

    assert report["samples"] == 100
    assert report["sample_ids_sha256"] == "fixed100"
    assert len(report["rows"]) == 7
    assert report["comparisons"]["bidirectional_vs_causal_top4"] == {
        "latency_delta_pct": 80.0,
        "ssr_delta_points": 4.0,
        "target_tile_recall_delta_points": 30.0,
    }
    cached = report["comparisons"]["cached_vs_full_bidirectional"]
    assert cached["end_to_end_speedup"] == pytest.approx(8 / 6)
    assert cached["retrieval_speedup"] == pytest.approx(2.0)
    assert cached["ssr_delta_points"] == 0.0
    assert (
        report["comparisons"]["tile_rank_ocr_vs_shared_control"][
            "ssr_delta_points"
        ]
        == 2.0
    )
    markdown = MODULE.markdown_report(report)
    assert "Cached-visual + neural tile-rank OCR" in markdown
    assert "fixed100" in markdown


def test_build_report_rejects_cross_run_sample_mismatch(tmp_path):
    attention, top8, optimized = make_runs(tmp_path)
    run_path = top8 / "run.json"
    run = json.loads(run_path.read_text())
    run["arms"][0]["fingerprint"]["sample_ids_sha256"] = "different"
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(RuntimeError, match="different samples"):
        MODULE.build_report(attention, top8, optimized)


def test_build_report_rejects_incomplete_run(tmp_path):
    attention, top8, optimized = make_runs(tmp_path)
    run_path = optimized / "run.json"
    run = json.loads(run_path.read_text())
    run["status"] = "running"
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(RuntimeError, match="not completed"):
        MODULE.build_report(attention, top8, optimized)
