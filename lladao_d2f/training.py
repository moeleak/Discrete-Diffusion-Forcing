"""Training-loop invariants shared by the LLaDA-o D2F entrypoint."""

from __future__ import annotations


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
