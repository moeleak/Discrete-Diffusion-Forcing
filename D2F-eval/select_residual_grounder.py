#!/usr/bin/env python3
"""Select a residual grounding epoch by worst-domain validation SSR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--save-every", type=int, required=True)
    parser.add_argument("--mind2web-benchmark", default="mind2web_validation")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_ssr(path: Path, benchmark: str) -> tuple[float, float, int]:
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result["benchmarks"][benchmark]
    count = int(metrics["num_samples"])
    if count != 100:
        raise ValueError(
            f"checkpoint selection requires exactly 100 {benchmark} rows, found {count}"
        )
    return (
        float(metrics["ssr_point_only"]),
        float(metrics["joint_step_success"]),
        count,
    )


def select(
    run_root: Path,
    *,
    epochs: int,
    save_every: int,
    mind2web_benchmark: str,
) -> dict:
    if epochs <= 0 or save_every <= 0:
        raise ValueError("epochs and save-every must be positive")
    candidates = []
    for epoch in range(1, epochs + 1):
        epoch_root = run_root / "benchmark/validation" / f"epoch-{epoch:02d}"
        mind2web = read_ssr(
            epoch_root / "mind2web/scores/results.json", mind2web_benchmark
        )
        mobile = read_ssr(
            epoch_root / "mobile/scores/results.json", "mobile_validation"
        )
        step = epoch * save_every
        candidates.append(
            {
                "epoch": epoch,
                "step": step,
                "adapter": str(
                    run_root / f"step-{step:07d}" / "adapter"
                ),
                "mind2web_validation_ssr": mind2web[0],
                "mind2web_validation_joint_ssr": mind2web[1],
                "mobile_validation_ssr": mobile[0],
                "mobile_validation_joint_ssr": mobile[1],
                "worst_domain_ssr": min(mind2web[0], mobile[0]),
                "mean_domain_ssr": (mind2web[0] + mobile[0]) / 2.0,
                "worst_domain_joint_ssr": min(mind2web[1], mobile[1]),
            }
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["worst_domain_ssr"],
            item["mean_domain_ssr"],
            item["worst_domain_joint_ssr"],
            -item["epoch"],
        ),
    )
    return {
        "schema_version": 1,
        "selection_metric": "max(min(mind2web_validation_ssr,mobile_validation_ssr))",
        "tie_breakers": [
            "mean_domain_ssr",
            "worst_domain_joint_ssr",
            "earlier_epoch",
        ],
        "test_data_used_for_selection": False,
        "candidates": candidates,
        "selected": selected,
    }


def markdown(result: dict) -> str:
    lines = [
        "# Residual grounder checkpoint selection",
        "",
        "| Epoch | Mind2Web val SSR | Mobile val SSR | Worst-domain SSR | Selected |",
        "|---:|---:|---:|---:|:---:|",
    ]
    selected_epoch = result["selected"]["epoch"]
    for row in result["candidates"]:
        lines.append(
            "| {epoch} | {mind:.2f}% | {mobile:.2f}% | {worst:.2f}% | {selected} |".format(
                epoch=row["epoch"],
                mind=100 * row["mind2web_validation_ssr"],
                mobile=100 * row["mobile_validation_ssr"],
                worst=100 * row["worst_domain_ssr"],
                selected="yes" if row["epoch"] == selected_epoch else "",
            )
        )
    lines.extend(
        [
            "",
            "Selection maximizes the poorer domain's validation SSR. Test splits were not read.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = select(
        args.run_root.expanduser().resolve(),
        epochs=args.epochs,
        save_every=args.save_every,
        mind2web_benchmark=args.mind2web_benchmark,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(result["selected"]["adapter"])


if __name__ == "__main__":
    main()
