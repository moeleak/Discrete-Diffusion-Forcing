from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from accelerate import init_empty_weights
from safetensors.torch import save_file
from torch import nn

from lladao_d2f.modeling import (
    LLaDAOGuiD2FModel,
    _convert_conv2d_to_linear_on_meta,
    combine_d2f_and_action_losses,
    forward_masked_logits,
    full_response_reconstruction_metrics,
    load_strict_understanding_checkpoint,
    load_strict_understanding_model_checkpoint,
    load_strict_pruned_checkpoint,
    prepare_understanding_sequence,
    strip_unused_generation_experts,
    teacher_distillation_loss,
)


def test_training_wrapper_forwards_optional_prefix_segments(monkeypatch) -> None:
    import lladao_d2f.modeling as modeling

    prefix_segments = [[(0, 3, "image"), (3, 2, "prompt")]]
    rebuilt_batch = {
        "sample_lens": [7],
        "d2f_response_spans": [[(5, 2)]],
        "d2f_prefix_segments": prefix_segments,
        "ce_loss_indexes": torch.tensor([6]),
        "packed_label_ids": torch.tensor([1]),
        "ce_loss_weights": torch.ones(1),
        "action_ce_mask": torch.zeros(1, dtype=torch.bool),
        "content_ce_mask": torch.zeros(1, dtype=torch.bool),
        "full_response_ce_mask": torch.zeros(1, dtype=torch.bool),
        "full_response_group_ids": torch.full((1,), -1, dtype=torch.long),
        "d2f_response_count": torch.ones((), dtype=torch.long),
        "full_response_masked_count": torch.zeros((), dtype=torch.long),
    }
    captured = {}
    monkeypatch.setattr(
        modeling,
        "rebuild_and_corrupt_responses",
        lambda raw_batch, **kwargs: rebuilt_batch,
    )
    monkeypatch.setattr(
        modeling,
        "unwrap_lladao",
        lambda model: SimpleNamespace(num_heads=3),
    )
    monkeypatch.setattr(
        modeling,
        "prepare_understanding_sequence",
        lambda model, batch: torch.zeros(7, 2),
    )

    def capture_mask(sample_lens, response_spans, block_size, **kwargs):
        captured.update(
            sample_lens=sample_lens,
            response_spans=response_spans,
            block_size=block_size,
            **kwargs,
        )
        return object()

    monkeypatch.setattr(modeling, "create_training_block_mask", capture_mask)
    monkeypatch.setattr(
        modeling,
        "forward_masked_logits",
        lambda model, packed_sequence, batch, mask: torch.tensor([[0.0, 1.0]]),
    )
    monkeypatch.setattr(
        modeling,
        "teacher_distillation_loss",
        lambda *args, **kwargs: torch.zeros(1),
    )

    wrapper = LLaDAOGuiD2FModel(
        torch.nn.Identity(),
        block_size=16,
        distill_weight=0.0,
        hard_ce_weight=1.0,
    )
    metrics = wrapper({"raw": "batch"})

    assert captured["sample_lens"] == [7]
    assert captured["response_spans"] == [[(5, 2)]]
    assert captured["block_size"] == 16
    assert captured["prefix_segments"] is prefix_segments
    assert captured["num_heads"] == 3
    assert metrics["masked_tokens"].item() == 1


def test_training_wrapper_registers_a_generic_model_once() -> None:
    model = nn.Linear(3, 2)
    wrapper = LLaDAOGuiD2FModel(model)

    assert wrapper.model is model
    assert wrapper.peft_model is model
    assert list(dict(wrapper.named_children())) == ["model"]
    assert set(wrapper.state_dict()) == {"model.weight", "model.bias"}


def test_training_wrapper_keeps_frozen_teacher_out_of_student_state() -> None:
    student = nn.Linear(3, 2)
    teacher = nn.Linear(3, 2)
    wrapper = LLaDAOGuiD2FModel(student, teacher=teacher)

    assert wrapper.teacher is teacher
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    assert list(dict(wrapper.named_children())) == ["model"]
    assert all("teacher" not in key for key in wrapper.state_dict())


def test_understanding_sequence_preserves_full_model_gradients() -> None:
    class Vision(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(2.0))

        def forward(self, *, packed_pixel_values, **kwargs):
            return packed_pixel_values * self.scale

    class LanguageBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 2)

    class Language(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = LanguageBackbone()

    class FullModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_size = 2
            self.language_model = Language()
            self.vit_model = Vision()
            self.connector = nn.Linear(2, 2, bias=False)
            self.vit_pos_embed = nn.Embedding(4, 2)

    model = FullModel()
    batch = {
        "sequence_length": 4,
        "packed_text_ids": torch.tensor([1, 2]),
        "packed_text_indexes": torch.tensor([0, 3]),
        "packed_vit_tokens": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "packed_vit_token_indexes": torch.tensor([1, 2]),
        "packed_vit_position_ids": torch.tensor([0, 1]),
        "vit_token_seqlens": torch.tensor([2]),
    }

    sequence = prepare_understanding_sequence(model, batch)
    sequence.square().sum().backward()

    parameters = {
        "text embedding": model.language_model.model.embed_tokens.weight,
        "vision tower": model.vit_model.scale,
        "connector": model.connector.weight,
        "vision position": model.vit_pos_embed.weight,
    }
    for label, parameter in parameters.items():
        assert parameter.grad is not None, label
        assert bool(parameter.grad.abs().sum() > 0), label


def test_masked_logits_use_packed_forward_while_model_is_in_eval_mode() -> None:
    class NestedDispatcher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dropout = nn.Dropout(0.5)
            self.forward_train_called = False

        def forward(self, *args, **kwargs):
            if self.training:
                return self.forward_train(*args, **kwargs)
            raise AssertionError("nested dispatcher selected inference forward")

        def forward_train(self, **kwargs):
            self.forward_train_called = True
            assert self.dropout.training is False
            return kwargs["packed_sequence"] + 1

        def forward_inference(self, **kwargs):
            raise AssertionError("nested inference forward must not be used")

    class LanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = NestedDispatcher()
            self.lm_head = nn.Identity()
            self.forward_train_called = False

        def forward(self, *args, **kwargs):
            if self.training:
                return self.forward_train(*args, **kwargs)
            raise AssertionError("outer dispatcher selected inference forward")

        def forward_train(self, **kwargs):
            self.forward_train_called = True
            assert kwargs["sample_lens"] == [3]
            assert kwargs["attention_mask"] is attention_mask
            return self.model(**kwargs), None

        def forward_inference(self, **kwargs):
            raise AssertionError("outer inference forward must not be used")

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
    assert model.language_model.model.forward_train_called
    assert model.training is False
    assert model.language_model.training is False
    assert model.language_model.model.training is False
    assert model.language_model.model.dropout.training is False
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


def _save_toy_understanding_checkpoint(path) -> _ToyUnderstandingModel:
    torch.manual_seed(7)
    reference = _ToyUnderstandingModel().to(torch.bfloat16)
    state = {
        name: tensor
        for name, tensor in reference.state_dict().items()
        if "_moe_gen" not in name
    }
    save_file(state, str(path))
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


def test_understanding_only_checkpoint_load_is_strict_and_exact(tmp_path) -> None:
    checkpoint = tmp_path / "toy-understanding.safetensors"
    reference = _save_toy_understanding_checkpoint(checkpoint)
    model = _empty_toy()
    inputs = torch.randn(3, 4, dtype=torch.bfloat16)

    removed = load_strict_understanding_checkpoint(model, checkpoint)

    assert removed == 80
    assert not any(parameter.is_meta for parameter in model.parameters())
    assert not any("_moe_gen" in name for name, _ in model.named_parameters())
    assert torch.equal(model.keep.weight, reference.keep.weight)
    assert torch.equal(model.keep(inputs), reference.keep(inputs))


@pytest.mark.parametrize(
    ("understanding_only", "expected_format"),
    [(False, "complete"), (True, "understanding-only")],
)
def test_understanding_model_loader_dispatches_strict_formats(
    tmp_path, understanding_only, expected_format
) -> None:
    checkpoint = tmp_path / "toy.safetensors"
    if understanding_only:
        _save_toy_understanding_checkpoint(checkpoint)
    else:
        _save_toy_checkpoint(checkpoint)

    removed, checkpoint_format = load_strict_understanding_model_checkpoint(
        _empty_toy(), checkpoint
    )

    assert removed == 80
    assert checkpoint_format == expected_format


def test_complete_checkpoint_loader_rejects_understanding_only_file(tmp_path) -> None:
    checkpoint = tmp_path / "toy-understanding.safetensors"
    _save_toy_understanding_checkpoint(checkpoint)

    with pytest.raises(RuntimeError, match="missing"):
        load_strict_pruned_checkpoint(_empty_toy(), checkpoint)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing"),
        ("unexpected", "unexpected"),
        ("shape", "shape mismatches"),
        ("dtype", "dtype mismatches"),
    ],
)
def test_understanding_only_checkpoint_rejects_mismatch(
    tmp_path, mutation, message
) -> None:
    reference = _ToyUnderstandingModel().to(torch.bfloat16)
    state = {
        name: tensor
        for name, tensor in reference.state_dict().items()
        if "_moe_gen" not in name
    }
    if mutation == "missing":
        del state["keep.weight"]
    elif mutation == "unexpected":
        state["unexpected.weight"] = torch.zeros(1, dtype=torch.bfloat16)
    elif mutation == "shape":
        state["keep.weight"] = torch.zeros(3, 4, dtype=torch.bfloat16)
    elif mutation == "dtype":
        state["keep.weight"] = state["keep.weight"].float()
    checkpoint = tmp_path / f"understanding-{mutation}.safetensors"
    save_file(state, str(checkpoint))

    with pytest.raises(RuntimeError, match=message):
        load_strict_understanding_checkpoint(_empty_toy(), checkpoint)


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
