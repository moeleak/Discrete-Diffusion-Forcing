from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lladao_d2f.modeling import strip_unused_generation_experts


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
