"""Training-loop invariants shared by the LLaDA-o D2F entrypoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch


@dataclass
class OptimizerStepMetricAccumulator:
    """Aggregate every microbatch instead of reporting only the final domain.

    Accelerate exposes the metrics from the microbatch that closes a gradient
    accumulation window.  Residual grounding deliberately ends each even-sized
    window with the mobile domain, where teacher distillation is disabled.  A
    naive logger would therefore report a misleading zero distillation loss
    even though every preceding Mind2Web microbatch ran the teacher objective.
    """

    sums: dict[str, torch.Tensor] = field(default_factory=dict)
    domain_sums: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    microbatches: int = 0
    domain_microbatches: dict[str, int] = field(default_factory=dict)

    def add(self, metrics: Mapping[str, torch.Tensor], *, domain: str) -> None:
        if not domain:
            raise ValueError("metric domain must be non-empty")
        domain_values = self.domain_sums.setdefault(domain, {})
        for name, value in metrics.items():
            detached = value.detach()
            if detached.numel() != 1:
                raise ValueError(f"metric {name!r} must be scalar")
            detached = detached.reshape(()).float()
            self.sums[name] = self.sums.get(name, torch.zeros_like(detached)) + detached
            domain_values[name] = (
                domain_values.get(name, torch.zeros_like(detached)) + detached
            )
        self.microbatches += 1
        self.domain_microbatches[domain] = self.domain_microbatches.get(domain, 0) + 1

    def take(
        self,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, dict[str, torch.Tensor]],
        dict[str, int],
    ]:
        """Return per-microbatch means for one completed optimizer step."""

        if self.microbatches <= 0:
            raise RuntimeError("cannot finalize an empty optimizer-step metric window")
        overall = {
            name: value / self.microbatches for name, value in self.sums.items()
        }
        by_domain = {
            domain: {
                name: value / self.domain_microbatches[domain]
                for name, value in values.items()
            }
            for domain, values in self.domain_sums.items()
        }
        counts = dict(self.domain_microbatches)
        self.clear()
        return overall, by_domain, counts

    def clear(self) -> None:
        self.sums.clear()
        self.domain_sums.clear()
        self.microbatches = 0
        self.domain_microbatches.clear()


def advance_scheduler_for_optimizer_update(
    scheduler,
    *,
    sync_gradients: bool,
    optimizer_step_was_skipped: bool,
) -> bool:
    """Advance ``scheduler`` exactly once after a real optimizer update."""
    optimizer_updated = sync_gradients and not optimizer_step_was_skipped
    if optimizer_updated:
        scheduler.step()
    return optimizer_updated


def validate_scheduler_global_step(scheduler, global_step: int) -> None:
    """Reject learning-rate state that has drifted from optimizer updates."""
    last_epoch = getattr(scheduler, "last_epoch", None)
    if last_epoch is None:
        raise RuntimeError("scheduler does not expose last_epoch")
    if int(last_epoch) != int(global_step):
        raise RuntimeError(
            "scheduler state is inconsistent with the optimizer update count: "
            f"last_epoch={last_epoch}, global_step={global_step}. Refusing to "
            "continue from a learning-rate schedule that advanced independently "
            "of optimizer updates."
        )
