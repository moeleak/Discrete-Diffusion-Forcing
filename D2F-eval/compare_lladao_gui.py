#!/usr/bin/env python3
"""Compare paired baseline/D2F GUI results and enforce migration gates."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--d2f-predictions", type=Path, required=True)
    parser.add_argument("--baseline-scores", type=Path, required=True)
    parser.add_argument("--d2f-scores", type=Path, required=True)
    parser.add_argument("--benchmark", default="mind2web")
    parser.add_argument("--max-ssr-drop", type=float, default=1.0)
    parser.add_argument("--max-action-f1-drop", type=float, default=0.2)
    parser.add_argument("--min-generation-speedup", type=float, default=1.0)
    parser.add_argument("--min-end-to-end-speedup", type=float, default=0.95)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_shards(root: Path, benchmark: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((root / benchmark).glob("part-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row["sample_id"])
                if sample_id in rows:
                    raise RuntimeError(f"duplicate sample {sample_id} below {root}")
                rows[sample_id] = row
    if not rows:
        raise FileNotFoundError(f"no prediction shards below {root / benchmark}")
    return rows


def paired_ratio(
    baseline: dict[str, dict[str, Any]],
    d2f: dict[str, dict[str, Any]],
    field: str,
) -> tuple[float, int]:
    ratios = []
    for sample_id in sorted(set(baseline) & set(d2f)):
        left = baseline[sample_id].get(field)
        right = d2f[sample_id].get(field)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)) and right > 0:
            ratios.append(float(left) / float(right))
    if not ratios:
        raise RuntimeError(f"no paired finite values for {field}")
    return statistics.fmean(ratios), len(ratios)


def score(path: Path, benchmark: str) -> tuple[float, float]:
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result["benchmarks"][benchmark]
    return (
        100.0 * float(metrics["ssr_point_only"]),
        100.0 * float(metrics["action_f1_macro_present"]),
    )


def main() -> None:
    args = parse_args()
    baseline = load_shards(args.baseline_predictions, args.benchmark)
    d2f = load_shards(args.d2f_predictions, args.benchmark)
    if set(baseline) != set(d2f):
        raise RuntimeError(
            "baseline and D2F sample sets differ: "
            f"baseline={len(baseline)}, d2f={len(d2f)}, paired={len(set(baseline) & set(d2f))}"
        )
    baseline_ssr, baseline_action_f1 = score(args.baseline_scores, args.benchmark)
    d2f_ssr, d2f_action_f1 = score(args.d2f_scores, args.benchmark)
    generation_speedup, pairs = paired_ratio(baseline, d2f, "generation_seconds")
    end_to_end_speedup, _ = paired_ratio(baseline, d2f, "latency_seconds")
    ssr_drop = baseline_ssr - d2f_ssr
    action_f1_drop = baseline_action_f1 - d2f_action_f1
    baseline_errors = sum(bool(row.get("error")) for row in baseline.values())
    d2f_errors = sum(bool(row.get("error")) for row in d2f.values())
    gates = {
        "ssr": ssr_drop <= args.max_ssr_drop,
        "action_f1": action_f1_drop <= args.max_action_f1_drop,
        "generation_speed": generation_speedup >= args.min_generation_speedup,
        "end_to_end_speed": end_to_end_speedup >= args.min_end_to_end_speedup,
        "errors": args.allow_errors or (baseline_errors == 0 and d2f_errors == 0),
    }
    result = {
        "benchmark": args.benchmark,
        "paired_samples": pairs,
        "baseline_ssr_percent": baseline_ssr,
        "d2f_ssr_percent": d2f_ssr,
        "ssr_drop_points": ssr_drop,
        "baseline_action_f1_percent": baseline_action_f1,
        "d2f_action_f1_percent": d2f_action_f1,
        "action_f1_drop_points": action_f1_drop,
        "baseline_errors": baseline_errors,
        "d2f_errors": d2f_errors,
        "mean_paired_generation_speedup": generation_speedup,
        "mean_paired_end_to_end_speedup": end_to_end_speedup,
        "thresholds": {
            "max_ssr_drop_points": args.max_ssr_drop,
            "max_action_f1_drop_points": args.max_action_f1_drop,
            "min_generation_speedup": args.min_generation_speedup,
            "min_end_to_end_speedup": args.min_end_to_end_speedup,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
