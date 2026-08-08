from __future__ import annotations

import pytest
import torch

from lladao_d2f.training import (
    OptimizerStepMetricAccumulator,
    advance_scheduler_for_optimizer_update,
    validate_scheduler_global_step,
)


class _Scheduler:
    def __init__(self, last_epoch: int = 0) -> None:
        self.last_epoch = last_epoch

    def step(self) -> None:
        self.last_epoch += 1


@pytest.mark.parametrize(
    ("sync_gradients", "optimizer_step_was_skipped"),
    [(False, False), (False, True), (True, True)],
)
def test_scheduler_does_not_advance_without_optimizer_update(
    sync_gradients: bool,
    optimizer_step_was_skipped: bool,
) -> None:
    scheduler = _Scheduler()

    updated = advance_scheduler_for_optimizer_update(
        scheduler,
        sync_gradients=sync_gradients,
        optimizer_step_was_skipped=optimizer_step_was_skipped,
    )

    assert updated is False
    assert scheduler.last_epoch == 0


def test_scheduler_advances_once_for_global_optimizer_update() -> None:
    scheduler = _Scheduler(last_epoch=17)

    updated = advance_scheduler_for_optimizer_update(
        scheduler,
        sync_gradients=True,
        optimizer_step_was_skipped=False,
    )

    assert updated is True
    assert scheduler.last_epoch == 18


def test_scheduler_state_must_match_global_step() -> None:
    scheduler = _Scheduler(last_epoch=40)

    with pytest.raises(
        RuntimeError,
        match=r"last_epoch=40, global_step=5",
    ):
        validate_scheduler_global_step(scheduler, global_step=5)


def test_scheduler_state_accepts_matching_global_step() -> None:
    validate_scheduler_global_step(_Scheduler(last_epoch=1377), global_step=1377)


def test_optimizer_step_metrics_include_both_residual_domains() -> None:
    accumulator = OptimizerStepMetricAccumulator()
    accumulator.add(
        {
            "loss": torch.tensor(4.0),
            "distill_loss": torch.tensor(3.0),
            "distill_tokens": torch.tensor(12),
        },
        domain="mind2web",
    )
    accumulator.add(
        {
            "loss": torch.tensor(2.0),
            "distill_loss": torch.tensor(0.0),
            "distill_tokens": torch.tensor(0),
        },
        domain="mobile",
    )

    overall, by_domain, counts = accumulator.take()

    assert overall["loss"].item() == pytest.approx(3.0)
    assert overall["distill_loss"].item() == pytest.approx(1.5)
    assert overall["distill_tokens"].item() == pytest.approx(6.0)
    assert by_domain["mind2web"]["distill_loss"].item() == pytest.approx(3.0)
    assert by_domain["mind2web"]["distill_tokens"].item() == pytest.approx(12.0)
    assert by_domain["mobile"]["distill_loss"].item() == 0.0
    assert by_domain["mobile"]["distill_tokens"].item() == 0.0
    assert counts == {"mind2web": 1, "mobile": 1}

    with pytest.raises(RuntimeError, match="empty"):
        accumulator.take()
