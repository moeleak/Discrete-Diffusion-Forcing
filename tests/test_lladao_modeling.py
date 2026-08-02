from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lladao_d2f.modeling import combine_d2f_and_action_losses, strip_unused_generation_experts


class _ToyUnderstandingModel(nn.Module):
    def __init__(self, *, visual_gen: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(visual_gen=visual_gen)
        self.keep = nn.Linear(4, 4, bias=False)
        self.block = nn.Module()
        self.block.q_proj_moe_gen = nn.Linear(4, 4, bias=False)
        self.block.mlp_moe_gen = nn.Sequential(
            nn.Linear(4, 8, bias=False),
            nn.SiLU(),
            nn.Linear(8, 4, bias=False),
        )


def test_strip_generation_experts_preserves_understanding_path() -> None:
    torch.manual_seed(7)
    model = _ToyUnderstandingModel()
    inputs = torch.randn(3, 4)
    expected = model.keep(inputs).detach().clone()
    removed_expected = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "_moe_gen" in name
    )

    removed = strip_unused_generation_experts(model)

    assert removed == removed_expected
    assert torch.equal(model.keep(inputs), expected)
    assert isinstance(model.block.q_proj_moe_gen, nn.Identity)
    assert isinstance(model.block.mlp_moe_gen, nn.Identity)
    assert not any("_moe_gen" in name for name, _ in model.named_parameters())


def test_strip_generation_experts_rejects_generation_model() -> None:
    with pytest.raises(ValueError, match="visual_gen=False"):
        strip_unused_generation_experts(_ToyUnderstandingModel(visual_gen=True))


def test_action_ce_is_balanced_per_action_not_added_to_d2f_estimator() -> None:
    metrics = combine_d2f_and_action_losses(
        distill=torch.tensor([2.0, 4.0, 8.0]),
        hard_ce=torch.tensor([1.0, 3.0, 5.0]),
        d2f_weights=torch.tensor([2.0, 0.0, 4.0]),
        action_mask=torch.tensor([False, True, True]),
        distill_weight=0.1,
        hard_ce_weight=1.0,
        action_ce_weight=1.0,
        action_class_weight=torch.tensor(3.0),
    )
    expected_d2f = ((1.0 + 0.2) * 2 + (5.0 + 0.8) * 4) / 6
    expected_action = (3.0 + 5.0) / 2
    assert metrics["d2f_loss"].item() == pytest.approx(expected_d2f)
    assert metrics["action_ce_loss"].item() == pytest.approx(expected_action)
    assert metrics["balanced_action_ce_loss"].item() == pytest.approx(3 * expected_action)
    assert metrics["loss"].item() == pytest.approx(expected_d2f + 3 * expected_action)


def test_action_ce_weight_requires_action_tokens() -> None:
    with pytest.raises(ValueError, match="no action tokens"):
        combine_d2f_and_action_losses(
            distill=torch.ones(2),
            hard_ce=torch.ones(2),
            d2f_weights=torch.ones(2),
            action_mask=torch.zeros(2, dtype=torch.bool),
            distill_weight=0.1,
            hard_ce_weight=1.0,
            action_ce_weight=1.0,
            action_class_weight=torch.tensor(1.0),
        )
