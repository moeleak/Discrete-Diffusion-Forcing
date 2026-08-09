#!/usr/bin/env python3
"""Render one auditable Markdown table for residual-grounder release runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentage(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def summarize(index_path: Path) -> str:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected_sha = index["backbone_sha256"]
    rows = []
    for run in index["runs"]:
        config = json.loads(Path(run["run_config"]).read_text(encoding="utf-8"))
        scores = json.loads(Path(run["scores"]).read_text(encoding="utf-8"))
        benchmark = run["benchmark"]
        metrics = scores["benchmarks"][benchmark]
        if config.get("checkpoint_sha256") != expected_sha:
            raise ValueError(f"{benchmark} silently changed the Planner checkpoint")
        contract = config.get("residual_adapter_contract") or {}
        if (contract.get("backbone") or {}).get("sha256") != expected_sha:
            raise ValueError(f"{benchmark} did not load the bound residual adapter")
        count = int(metrics["num_samples"])
        if count != 100:
            raise ValueError(
                f"{benchmark} release evidence requires exactly 100 samples, found {count}"
            )
        rows.append(
            "| {label} | {count} | {ssr} | {joint} | {f1} | {parse} | {latency:.3f} |".format(
                label=run["label"],
                count=count,
                ssr=percentage(metrics["ssr_point_only"]),
                joint=percentage(metrics["joint_step_success"]),
                f1=percentage(metrics["action_f1_macro_present"]),
                parse=percentage(metrics["parse_rate"]),
                latency=float(metrics["latency_seconds"]["mean"]),
            )
        )
    lines = [
        "# Residual GUI-grounder benchmark",
        "",
        f"Planner checkpoint SHA-256: `{expected_sha}`",
        "",
        "| Domain / split | Samples | SSR | Joint SSR | Action F1 | Parse | Avg latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "Every row uses the same frozen final Planner and its SHA-bound residual LoRA. "
        "KV compression is disabled and each held-out selection is capped at 100 rows.",
        "",
        "> The historical 60-row old-LoRA-on-fused-Planner result (SSR/Joint 53.33%) "
        "is a double-delta compatibility diagnostic, not a release baseline.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summarize(args.index.expanduser().resolve()), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
