#!/usr/bin/env python3
"""Combine the controlled KV-retrieval runs into one audited report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ATTENTION_CAUSAL = "yarn128k-kv-top4-causal-masked-ocr"
ATTENTION_BIDIRECTIONAL = "yarn128k-kv-top4-sequential-ocr"
TOP8_BIDIRECTIONAL = "yarn128k-kv-top8-sequential-ocr"
CACHED_BIDIRECTIONAL = "yarn128k-kv-top4-cached-masked-ocr"
OCR_CONTROL = "yarn128k-kv-top4-cached-masked-prior-control-ocr"
OCR_PRIOR = "yarn128k-kv-top4-cached-masked-prior-ocr"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_completed_run(path: Path) -> tuple[Path, dict[str, Any]]:
    root = path.expanduser().resolve()
    run_path = root / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {run_path}")
    run = load_json(run_path)
    if run.get("status") != "completed":
        raise RuntimeError(
            f"run {run.get('run_id', root.name)!r} is not completed: "
            f"{run.get('status')!r}"
        )
    if int(run.get("limit", 0)) != 100:
        raise RuntimeError(
            f"run {run.get('run_id', root.name)!r} does not use 100 samples"
        )
    return root, run


def flattened_table(root: Path, name: str) -> dict[str, dict[str, Any]]:
    payload = load_json(root / "tables" / f"{name}.json")
    rows: dict[str, dict[str, Any]] = {}
    for group_rows in payload.get("groups", {}).values():
        for row in group_rows:
            configuration = str(row["Configuration"])
            if configuration in rows and rows[configuration] != row:
                raise RuntimeError(
                    f"conflicting {name} rows for {configuration}"
                )
            rows[configuration] = row
    return rows


def sample_fingerprint(run: dict[str, Any]) -> tuple[int, str]:
    fingerprints = {
        (
            int(arm["fingerprint"]["samples"]),
            str(arm["fingerprint"]["sample_ids_sha256"]),
        )
        for arm in run["arms"]
    }
    if len(fingerprints) != 1:
        raise RuntimeError(
            f"run {run.get('run_id')!r} contains mixed sample fingerprints"
        )
    return next(iter(fingerprints))


def require_configuration(
    quality: dict[str, dict[str, Any]],
    performance: dict[str, dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    if name not in quality or name not in performance:
        raise RuntimeError(f"missing required benchmark row {name!r}")
    return {
        **quality[name],
        **performance[name],
    }


def report_row(
    label: str,
    source_run: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "Configuration": label,
        "Benchmark key": row["Configuration"],
        "Revision": source_run["revision"],
        "Final SSR (%)": row["Final SSR (%)"],
        "Target tile recall (%)": row["Target tile recall (%)"],
        "Mean latency (s)": row["Mean end-to-end latency (s)"],
        "Mean retrieval (s)": row["Mean retrieval latency (s)"],
        "Mean resident KV": row["Mean resident KV"],
        "KV reduction (%)": row["KV reduction (%)"],
        "Peak allocated (GiB)": row["Peak allocated (GiB)"],
    }


def percent_delta(after: float, before: float) -> float:
    return 100.0 * (after - before) / before


def ratio(before: float, after: float) -> float:
    return before / after


def build_report(
    attention_root: Path,
    top8_root: Path,
    optimized_root: Path,
) -> dict[str, Any]:
    roots_and_runs = [
        load_completed_run(attention_root),
        load_completed_run(top8_root),
        load_completed_run(optimized_root),
    ]
    fingerprints = {sample_fingerprint(run) for _, run in roots_and_runs}
    if len(fingerprints) != 1:
        raise RuntimeError(
            "the attention, Top-K, and optimized runs use different samples"
        )
    fingerprint = next(iter(fingerprints))

    tables = {}
    for root, run in roots_and_runs:
        tables[run["run_id"]] = {
            "quality": flattened_table(root, "quality"),
            "performance": flattened_table(root, "performance"),
        }
    attention_run = roots_and_runs[0][1]
    top8_run = roots_and_runs[1][1]
    optimized_run = roots_and_runs[2][1]

    def get(run: dict[str, Any], name: str) -> dict[str, Any]:
        table = tables[run["run_id"]]
        return require_configuration(
            table["quality"],
            table["performance"],
            name,
        )

    causal = get(attention_run, ATTENTION_CAUSAL)
    bidirectional_attention = get(
        attention_run,
        ATTENTION_BIDIRECTIONAL,
    )
    top8 = get(top8_run, TOP8_BIDIRECTIONAL)
    bidirectional_optimized = get(
        optimized_run,
        ATTENTION_BIDIRECTIONAL,
    )
    cached = get(optimized_run, CACHED_BIDIRECTIONAL)
    ocr_control = get(optimized_run, OCR_CONTROL)
    ocr_prior = get(optimized_run, OCR_PRIOR)

    rows = [
        report_row(
            "Causal masked Top-4",
            attention_run,
            causal,
        ),
        report_row(
            "Full bidirectional Top-4 (attention control)",
            attention_run,
            bidirectional_attention,
        ),
        report_row(
            "Full bidirectional Top-8",
            top8_run,
            top8,
        ),
        report_row(
            "Full bidirectional Top-4 (optimization control)",
            optimized_run,
            bidirectional_optimized,
        ),
        report_row(
            "Cached-visual bidirectional Top-4",
            optimized_run,
            cached,
        ),
        report_row(
            "Cached-visual + shared OCR control",
            optimized_run,
            ocr_control,
        ),
        report_row(
            "Cached-visual + neural tile-rank OCR",
            optimized_run,
            ocr_prior,
        ),
    ]
    comparisons = {
        "bidirectional_vs_causal_top4": {
            "latency_delta_pct": percent_delta(
                bidirectional_attention["Mean end-to-end latency (s)"],
                causal["Mean end-to-end latency (s)"],
            ),
            "ssr_delta_points": (
                bidirectional_attention["Final SSR (%)"]
                - causal["Final SSR (%)"]
            ),
            "target_tile_recall_delta_points": (
                bidirectional_attention["Target tile recall (%)"]
                - causal["Target tile recall (%)"]
            ),
        },
        "top8_vs_top4_bidirectional": {
            "latency_delta_pct": percent_delta(
                top8["Mean end-to-end latency (s)"],
                bidirectional_attention["Mean end-to-end latency (s)"],
            ),
            "ssr_delta_points": (
                top8["Final SSR (%)"]
                - bidirectional_attention["Final SSR (%)"]
            ),
            "target_tile_recall_delta_points": (
                top8["Target tile recall (%)"]
                - bidirectional_attention["Target tile recall (%)"]
            ),
        },
        "cached_vs_full_bidirectional": {
            "end_to_end_speedup": ratio(
                bidirectional_optimized["Mean end-to-end latency (s)"],
                cached["Mean end-to-end latency (s)"],
            ),
            "retrieval_speedup": ratio(
                bidirectional_optimized["Mean retrieval latency (s)"],
                cached["Mean retrieval latency (s)"],
            ),
            "ssr_delta_points": (
                cached["Final SSR (%)"]
                - bidirectional_optimized["Final SSR (%)"]
            ),
            "target_tile_recall_delta_points": (
                cached["Target tile recall (%)"]
                - bidirectional_optimized["Target tile recall (%)"]
            ),
        },
        "tile_rank_ocr_vs_shared_control": {
            "ssr_delta_points": (
                ocr_prior["Final SSR (%)"]
                - ocr_control["Final SSR (%)"]
            ),
            "latency_delta_pct": percent_delta(
                ocr_prior["Mean end-to-end latency (s)"],
                ocr_control["Mean end-to-end latency (s)"],
            ),
        },
    }
    return {
        "schema_version": 1,
        "samples": fingerprint[0],
        "sample_ids_sha256": fingerprint[1],
        "runs": {
            "attention": attention_run["run_id"],
            "top8": top8_run["run_id"],
            "optimized": optimized_run["run_id"],
        },
        "rows": rows,
        "comparisons": comparisons,
    }


def format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def markdown_report(report: dict[str, Any]) -> str:
    columns = [
        "Configuration",
        "Final SSR (%)",
        "Target tile recall (%)",
        "Mean latency (s)",
        "Mean retrieval (s)",
        "Mean resident KV",
        "KV reduction (%)",
        "Peak allocated (GiB)",
        "Revision",
    ]
    lines = [
        "# KV-retrieval root-cause and optimization benchmark",
        "",
        f"- Samples: `{report['samples']}`",
        f"- Ordered sample-ID SHA-256: `{report['sample_ids_sha256']}`",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in report["rows"]:
        lines.append(
            "| "
            + " | ".join(format_value(row[column]) for column in columns)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Controlled deltas",
            "",
        ]
    )
    for name, values in report["comparisons"].items():
        rendered = ", ".join(
            f"{key}={format_value(value)}"
            for key, value in values.items()
        )
        lines.append(f"- `{name}`: {rendered}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-run", type=Path, required=True)
    parser.add_argument("--top8-run", type=Path, required=True)
    parser.add_argument("--optimized-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        args.attention_run,
        args.top8_run,
        args.optimized_run,
    )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "kv-retrieval-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "kv-retrieval-summary.md").write_text(
        markdown_report(report),
        encoding="utf-8",
    )
    print(output / "kv-retrieval-summary.md")


if __name__ == "__main__":
    main()
