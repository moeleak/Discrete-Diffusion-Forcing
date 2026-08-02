from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from accelerate import init_empty_weights
from safetensors.torch import save_file
from torch import nn

from lladao_d2f.modeling import (
    _convert_conv2d_to_linear_on_meta,
    combine_d2f_and_action_losses,
    load_strict_pruned_checkpoint,
    strip_unused_generation_experts,
)


class _ToyUnderstandingModel(nn.Module):
    def __init__(
        self,
        *,
        visual_gen: bool = False,
        meta_runtime_buffer: bool = False,
    ) -> None:
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
        buffer_device = "meta" if meta_runtime_buffer else "cpu"
        self.register_buffer(
            "runtime_scale",
            torch.arange(4, dtype=torch.float32, device=buffer_device),
            persistent=False,
        )


def _empty_toy(*, meta_runtime_buffer: bool = False) -> _ToyUnderstandingModel:
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with init_empty_weights(include_buffers=False):
            return _ToyUnderstandingModel(meta_runtime_buffer=meta_runtime_buffer)
    finally:
        torch.set_default_dtype(previous_dtype)


def _save_toy_checkpoint(path) -> _ToyUnderstandingModel:
    torch.manual_seed(7)
    reference = _ToyUnderstandingModel().to(torch.bfloat16)
    save_file(reference.state_dict(), str(path))
    return reference


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


def test_meta_checkpoint_load_is_strict_pruned_and_exact(tmp_path) -> None:
    checkpoint = tmp_path / "toy.safetensors"
    reference = _save_toy_checkpoint(checkpoint)
    model = _empty_toy()
    inputs = torch.randn(3, 4, dtype=torch.bfloat16)

    assert all(parameter.is_meta for parameter in model.parameters())
    assert not model.runtime_scale.is_meta
    removed = load_strict_pruned_checkpoint(model, checkpoint)

    assert removed == 80
    assert not any(parameter.is_meta for parameter in model.parameters())
    assert not any(buffer.is_meta for buffer in model.buffers())
    assert not any("_moe_gen" in name for name, _ in model.named_parameters())
    assert torch.equal(model.keep.weight, reference.keep.weight)
    assert torch.equal(model.keep(inputs), reference.keep(inputs))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing"),
        ("unexpected", "unexpected"),
        ("shape", "shape mismatches"),
        ("dtype", "dtype mismatches"),
    ],
)
def test_meta_checkpoint_load_rejects_mismatch(tmp_path, mutation, message) -> None:
    reference = _ToyUnderstandingModel().to(torch.bfloat16)
    state = dict(reference.state_dict())
    if mutation == "missing":
        del state["keep.weight"]
    elif mutation == "unexpected":
        state["unexpected.weight"] = torch.zeros(1, dtype=torch.bfloat16)
    elif mutation == "shape":
        state["keep.weight"] = torch.zeros(3, 4, dtype=torch.bfloat16)
    elif mutation == "dtype":
        state["keep.weight"] = state["keep.weight"].float()
    checkpoint = tmp_path / f"{mutation}.safetensors"
    save_file(state, str(checkpoint))

    with pytest.raises(RuntimeError, match=message):
        load_strict_pruned_checkpoint(_empty_toy(), checkpoint)


def test_meta_checkpoint_load_rejects_meta_runtime_buffer(tmp_path) -> None:
    checkpoint = tmp_path / "toy.safetensors"
    _save_toy_checkpoint(checkpoint)

    with pytest.raises(RuntimeError, match="buffer:runtime_scale"):
        load_strict_pruned_checkpoint(
            _empty_toy(meta_runtime_buffer=True),
            checkpoint,
        )


def test_vision_conversion_requires_explicit_meta_support() -> None:
    class Compatible:
        received_meta = None

        def convert_conv2d_to_linear(self, config, meta=False):
            self.received_meta = meta

    class Incompatible:
        def convert_conv2d_to_linear(self, config):
            raise AssertionError("must fail before calling an incompatible conversion")

    compatible = Compatible()
    _convert_conv2d_to_linear_on_meta(compatible, object())
    assert compatible.received_meta is True
    with pytest.raises(RuntimeError, match="explicitly support meta=True"):
        _convert_conv2d_to_linear_on_meta(Incompatible(), object())


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
