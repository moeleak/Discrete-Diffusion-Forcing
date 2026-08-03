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
    forward_masked_logits,
    full_response_reconstruction_metrics,
    load_strict_pruned_checkpoint,
    strip_unused_generation_experts,
    teacher_distillation_loss,
)


def test_masked_logits_use_packed_forward_while_model_is_in_eval_mode() -> None:
    class LanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = nn.Identity()
            self.forward_train_called = False

        def forward(self, *args, **kwargs):
            raise AssertionError("mode-dependent forward must not be used")

        def forward_train(self, **kwargs):
            self.forward_train_called = True
            assert kwargs["sample_lens"] == [3]
            assert kwargs["attention_mask"] is attention_mask
            return kwargs["packed_sequence"] + 1, None

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = LanguageModel()

    attention_mask = object()
    model = Model().eval()
    packed_sequence = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    batch = {
        "packed_text_indexes": torch.tensor([0, 2]),
        "packed_vit_token_indexes": torch.tensor([1]),
        "sample_lens": [3],
        "packed_position_ids": torch.tensor([0, 1, 2]),
        "ce_loss_indexes": torch.tensor([1, 2]),
    }

    logits = forward_masked_logits(model, packed_sequence, batch, attention_mask)

    assert model.language_model.forward_train_called
    assert torch.equal(logits, torch.tensor([[4.0, 5.0], [6.0, 7.0]]))


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
    assert metrics["content_ce_loss"].item() == 0
    assert metrics["balanced_content_ce_loss"].item() == 0


def test_content_ce_uses_action_class_weight_by_default_for_compatibility() -> None:
    metrics = combine_d2f_and_action_losses(
        distill=torch.tensor([2.0, 4.0, 8.0, 16.0]),
        hard_ce=torch.tensor([1.0, 3.0, 5.0, 7.0]),
        d2f_weights=torch.tensor([2.0, 0.0, 0.0, 4.0]),
        action_mask=torch.tensor([False, True, False, False]),
        content_mask=torch.tensor([False, False, True, False]),
        distill_weight=0.1,
        hard_ce_weight=1.0,
        action_ce_weight=1.0,
        content_ce_weight=0.5,
        action_class_weight=torch.tensor(3.0),
    )
    expected_d2f = ((1.0 + 0.2) * 2 + (7.0 + 1.6) * 4) / 6
    expected_action = 3.0
    expected_content = 5.0
    assert metrics["d2f_loss"].item() == pytest.approx(expected_d2f)
    assert metrics["action_ce_loss"].item() == pytest.approx(expected_action)
    assert metrics["balanced_action_ce_loss"].item() == pytest.approx(9.0)
    assert metrics["content_ce_loss"].item() == pytest.approx(expected_content)
    assert metrics["balanced_content_ce_loss"].item() == pytest.approx(15.0)
    assert metrics["loss"].item() == pytest.approx(expected_d2f + 9.0 + 0.5 * 15.0)


def test_content_ce_can_ignore_action_class_weight() -> None:
    metrics = combine_d2f_and_action_losses(
        distill=torch.tensor([2.0, 4.0, 8.0, 16.0]),
        hard_ce=torch.tensor([1.0, 3.0, 5.0, 7.0]),
        d2f_weights=torch.tensor([2.0, 0.0, 0.0, 4.0]),
        action_mask=torch.tensor([False, True, False, False]),
        content_mask=torch.tensor([False, False, True, False]),
        distill_weight=0.1,
        hard_ce_weight=1.0,
        action_ce_weight=1.0,
        content_ce_weight=0.5,
        action_class_weight=torch.tensor(3.0),
        content_ce_use_action_class_weight=False,
    )
    expected_d2f = ((1.0 + 0.2) * 2 + (7.0 + 1.6) * 4) / 6
    assert metrics["action_ce_loss"].item() == pytest.approx(3.0)
    assert metrics["balanced_action_ce_loss"].item() == pytest.approx(9.0)
    assert metrics["content_ce_loss"].item() == pytest.approx(5.0)
    assert metrics["balanced_content_ce_loss"].item() == pytest.approx(5.0)
    assert metrics["loss"].item() == pytest.approx(expected_d2f + 9.0 + 0.5 * 5.0)


def test_full_response_reconstruction_counts_tokens_and_exact_responses() -> None:
    logits = torch.tensor(
        [
            [9.0, 0.0, 0.0],
            [0.0, 9.0, 0.0],
            [0.0, 9.0, 0.0],
            [0.0, 0.0, 9.0],
            [9.0, 0.0, 0.0],
        ]
    )
    metrics = full_response_reconstruction_metrics(
        logits=logits,
        labels=torch.tensor([0, 1, 2, 2, 0]),
        token_mask=torch.tensor([False, True, True, True, True]),
        group_ids=torch.tensor([-1, 3, 3, 7, 7]),
    )

    assert metrics["full_response_token_correct"].item() == 3
    assert metrics["full_response_token_count"].item() == 4
    assert metrics["full_response_exact"].item() == 1
    assert metrics["full_response_count"].item() == 2


def test_zero_weight_distillation_does_not_construct_or_call_teacher(
    monkeypatch,
) -> None:
    import lladao_d2f.modeling as modeling

    def fail(*args, **kwargs):
        raise AssertionError("zero-weight distillation must not touch the teacher")

    monkeypatch.setattr(modeling, "create_full_document_mask", fail)
    monkeypatch.setattr(modeling, "forward_masked_logits", fail)
    student_log_probabilities = torch.randn(4, 7)

    loss = teacher_distillation_loss(
        object(),
        torch.randn(4, 3),
        {"sample_lens": [4]},
        student_log_probabilities,
        num_heads=2,
        distill_weight=0.0,
    )

    assert loss.shape == (4,)
    assert loss.dtype == student_log_probabilities.dtype
    assert torch.equal(loss, torch.zeros_like(loss))


def test_nonzero_weight_distillation_preserves_cross_entropy(monkeypatch) -> None:
    import lladao_d2f.modeling as modeling

    teacher_logits = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    student_logits = torch.tensor([[0.5, -0.5], [-0.25, 0.75]])
    marker = object()
    monkeypatch.setattr(
        modeling,
        "create_full_document_mask",
        lambda *args, **kwargs: marker,
    )

    def teacher_forward(model, packed_sequence, batch, attention_mask):
        assert attention_mask is marker
        return teacher_logits

    monkeypatch.setattr(modeling, "forward_masked_logits", teacher_forward)
    student_log_probabilities = torch.log_softmax(student_logits, dim=-1)

    loss = teacher_distillation_loss(
        object(),
        torch.randn(2, 3),
        {"sample_lens": [2]},
        student_log_probabilities,
        num_heads=2,
        distill_weight=0.1,
    )
    expected = -(
        torch.softmax(teacher_logits, dim=-1) * student_log_probabilities
    ).sum(dim=-1)

    assert torch.allclose(loss, expected)


def test_contentless_action_has_zero_content_ce_when_enabled() -> None:
    metrics = combine_d2f_and_action_losses(
        distill=torch.ones(2),
        hard_ce=torch.tensor([2.0, 3.0]),
        d2f_weights=torch.ones(2),
        action_mask=torch.tensor([True, False]),
        content_mask=torch.zeros(2, dtype=torch.bool),
        distill_weight=0.1,
        hard_ce_weight=1.0,
        action_ce_weight=1.0,
        content_ce_weight=1.0,
        action_class_weight=torch.tensor(2.0),
    )

    assert metrics["content_ce_loss"].item() == 0
    assert metrics["balanced_content_ce_loss"].item() == 0
    assert metrics["loss"].item() == pytest.approx(2.6 + 4.0)


def test_action_and_content_ce_masks_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        combine_d2f_and_action_losses(
            distill=torch.ones(2),
            hard_ce=torch.ones(2),
            d2f_weights=torch.ones(2),
            action_mask=torch.tensor([True, False]),
            content_mask=torch.tensor([True, False]),
            distill_weight=0.1,
            hard_ce_weight=1.0,
            action_ce_weight=1.0,
            content_ce_weight=1.0,
            action_class_weight=torch.tensor(1.0),
        )


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
