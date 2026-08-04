from __future__ import annotations

import pytest
import torch

from lladao_d2f.noise import rebuild_and_corrupt_responses


def _fully_supervised_batch() -> dict[str, object]:
    return {
        "packed_text_ids": torch.tensor([10, 999, 999, 999, 999, 999]),
        "packed_text_indexes": torch.arange(6),
        "ce_loss_indexes": torch.arange(1, 6),
        "packed_label_ids": torch.tensor([11, 12, 13, 14, 15]),
        "ce_loss_weights": torch.ones(5),
        "sample_lens": [6],
        "d2f_response_spans": [[(0, 6)]],
        "action_token_indexes": torch.tensor([2]),
        "content_token_indexes": torch.tensor([4]),
    }


def test_corruption_rebuilds_clean_response_and_supervises_only_new_masks() -> None:
    torch.manual_seed(7)
    mask_id = 999
    clean = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])
    batch = {
        "packed_text_ids": torch.tensor([10, 11, mask_id, 13, 14, mask_id, 16, 17]),
        "packed_text_indexes": torch.arange(8),
        "ce_loss_indexes": torch.tensor([2, 5]),
        "packed_label_ids": torch.tensor([12, 15]),
        "ce_loss_weights": torch.ones(2),
        "sample_lens": [8, 2],
        "d2f_response_spans": [[(0, 7)], []],
    }

    result = rebuild_and_corrupt_responses(batch, mask_id=mask_id, block_size=2)
    indexes = result["ce_loss_indexes"].long()
    labels = result["packed_label_ids"].long()
    assert bool(((indexes >= 1) & (indexes < 7)).all())
    assert torch.equal(labels, clean[indexes])
    assert bool((result["packed_text_ids"][indexes] == mask_id).all())
    assert bool((result["ce_loss_weights"] >= 1.0).all())
    assert not bool(result["action_ce_mask"].any())
    assert not bool(result["content_ce_mask"].any())


def test_zero_full_response_probability_preserves_legacy_rng_and_outputs() -> None:
    batch = _fully_supervised_batch()
    torch.manual_seed(17)
    legacy = rebuild_and_corrupt_responses(batch, mask_id=999, block_size=2)
    torch.manual_seed(17)
    explicit_zero = rebuild_and_corrupt_responses(
        batch,
        mask_id=999,
        block_size=2,
        full_response_mask_probability=0.0,
    )

    for key in (
        "packed_text_ids",
        "ce_loss_indexes",
        "packed_label_ids",
        "ce_loss_weights",
        "action_ce_mask",
        "content_ce_mask",
    ):
        assert torch.equal(legacy[key], explicit_zero[key])
    assert explicit_zero["full_response_masked_count"].item() == 0
    assert explicit_zero["d2f_response_count"].item() == 1
    assert not bool(explicit_zero["full_response_ce_mask"].any())


def test_full_response_corruption_masks_and_supervises_json_through_eos() -> None:
    batch = _fully_supervised_batch()

    result = rebuild_and_corrupt_responses(
        batch,
        mask_id=999,
        block_size=16,
        full_response_mask_probability=1.0,
    )

    # Position zero is the response BOS; position five represents its EOS.
    assert result["packed_text_ids"][0].item() == 10
    assert torch.equal(result["ce_loss_indexes"], torch.arange(1, 6))
    assert torch.equal(result["packed_label_ids"], torch.tensor([11, 12, 13, 14, 15]))
    assert bool((result["packed_text_ids"][1:] == 999).all())
    assert bool((result["ce_loss_weights"] == 1.0).all())
    assert bool(result["full_response_ce_mask"].all())
    assert torch.equal(result["full_response_group_ids"], torch.zeros(5, dtype=torch.long))
    assert result["full_response_masked_count"].item() == 1
    assert result["d2f_response_count"].item() == 1
    assert bool(result["action_ce_mask"][1])
    assert bool(result["content_ce_mask"][3])


def test_reserved_response_slots_remain_masked_and_outside_loss() -> None:
    batch = _fully_supervised_batch()
    batch["packed_text_ids"] = torch.cat(
        [batch["packed_text_ids"], torch.tensor([999, 999])]
    )
    batch["packed_text_indexes"] = torch.arange(8)
    batch["sample_lens"] = [8]
    batch["d2f_response_spans"] = [[(0, 8)]]
    batch["d2f_response_target_lengths"] = [[6]]

    result = rebuild_and_corrupt_responses(
        batch,
        mask_id=999,
        block_size=16,
        full_response_mask_probability=1.0,
    )

    assert torch.equal(result["ce_loss_indexes"], torch.arange(1, 6))
    assert torch.equal(result["packed_label_ids"], torch.tensor([11, 12, 13, 14, 15]))
    assert bool((result["packed_text_ids"][1:] == 999).all())
    assert not bool(torch.isin(torch.tensor([6, 7]), result["ce_loss_indexes"]).any())
    assert result["full_response_masked_count"].item() == 1


def test_reserved_response_slots_never_enter_random_corruption_loss() -> None:
    batch = _fully_supervised_batch()
    batch["packed_text_ids"] = torch.cat(
        [batch["packed_text_ids"], torch.tensor([999, 999])]
    )
    batch["packed_text_indexes"] = torch.arange(8)
    batch["sample_lens"] = [8]
    batch["d2f_response_spans"] = [[(0, 8)]]
    batch["d2f_response_target_lengths"] = [[6]]

    for seed in range(10):
        torch.manual_seed(seed)
        result = rebuild_and_corrupt_responses(
            batch,
            mask_id=999,
            block_size=2,
        )
        assert bool((result["ce_loss_indexes"] < 6).all())
        assert bool((result["packed_text_ids"][6:] == 999).all())


@pytest.mark.parametrize(
    ("target_lengths", "message"),
    [
        ([[9]], "cannot exceed"),
        ([[1]], "BOS and answer"),
        ([[]], "must match each sample"),
    ],
)
def test_response_target_lengths_are_validated(target_lengths, message) -> None:
    batch = _fully_supervised_batch()
    batch["d2f_response_target_lengths"] = target_lengths

    with pytest.raises(ValueError, match=message):
        rebuild_and_corrupt_responses(batch, mask_id=999, block_size=2)


def test_reserved_response_slots_must_be_unsupervised_masks() -> None:
    batch = _fully_supervised_batch()
    batch["packed_text_ids"] = torch.cat(
        [batch["packed_text_ids"], torch.tensor([42, 999])]
    )
    batch["packed_text_indexes"] = torch.arange(8)
    batch["sample_lens"] = [8]
    batch["d2f_response_spans"] = [[(0, 8)]]
    batch["d2f_response_target_lengths"] = [[6]]

    with pytest.raises(ValueError, match="must contain only MASK"):
        rebuild_and_corrupt_responses(batch, mask_id=999, block_size=2)

    batch["packed_text_ids"][6] = 999
    batch["ce_loss_indexes"] = torch.arange(1, 7)
    batch["packed_label_ids"] = torch.tensor([11, 12, 13, 14, 15, 16])
    with pytest.raises(ValueError, match="cannot be supervised"):
        rebuild_and_corrupt_responses(batch, mask_id=999, block_size=2)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_full_response_probability_must_be_valid(probability: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        rebuild_and_corrupt_responses(
            _fully_supervised_batch(),
            mask_id=999,
            block_size=2,
            full_response_mask_probability=probability,
        )


def test_full_response_corruption_requires_eos_supervision() -> None:
    batch = _fully_supervised_batch()
    batch["ce_loss_indexes"] = torch.arange(1, 5)
    batch["packed_label_ids"] = torch.tensor([11, 12, 13, 14])

    with pytest.raises(ValueError, match="including EOS"):
        rebuild_and_corrupt_responses(
            batch,
            mask_id=999,
            block_size=16,
            full_response_mask_probability=1.0,
        )


def test_corruption_rejects_batches_without_supervised_responses() -> None:
    batch = {
        "packed_text_ids": torch.tensor([1, 2]),
        "packed_text_indexes": torch.arange(2),
        "ce_loss_indexes": torch.empty(0, dtype=torch.long),
        "packed_label_ids": torch.empty(0, dtype=torch.long),
        "ce_loss_weights": torch.empty(0),
        "sample_lens": [2],
        "d2f_response_spans": [[]],
    }
    try:
        rebuild_and_corrupt_responses(batch, mask_id=999, block_size=2)
    except ValueError as exc:
        assert "no supervised" in str(exc)
    else:
        raise AssertionError("unsupervised batches should fail")


def test_action_tokens_are_always_masked_without_biasing_d2f_weights() -> None:
    mask_id = 999
    action_index = 2
    observed_forced_only = False
    for seed in range(20):
        torch.manual_seed(seed)
        batch = {
            "packed_text_ids": torch.tensor([10, 11, 12, 13, 14, 15]),
            "packed_text_indexes": torch.arange(6),
            "ce_loss_indexes": torch.arange(1, 6),
            "packed_label_ids": torch.tensor([11, 12, 13, 14, 15]),
            "ce_loss_weights": torch.ones(5),
            "sample_lens": [6],
            "d2f_response_spans": [[(0, 6)]],
            "action_token_indexes": torch.tensor([action_index]),
        }
        result = rebuild_and_corrupt_responses(batch, mask_id=mask_id, block_size=16)
        selected = result["ce_loss_indexes"].long()
        action_offset = int((selected == action_index).nonzero().item())
        assert result["packed_text_ids"][action_index].item() == mask_id
        assert bool(result["action_ce_mask"][action_offset])
        assert not bool(result["content_ce_mask"][action_offset])
        if result["ce_loss_weights"][action_offset].item() == 0:
            observed_forced_only = True
        else:
            assert result["ce_loss_weights"][action_offset].item() >= 1
    assert observed_forced_only


def test_content_tokens_are_always_masked_without_biasing_d2f_weights() -> None:
    mask_id = 999
    action_index = 1
    content_index = 4
    observed_forced_only = False
    for seed in range(20):
        torch.manual_seed(seed)
        batch = {
            "packed_text_ids": torch.tensor([10, 11, 12, 13, 14, 15]),
            "packed_text_indexes": torch.arange(6),
            "ce_loss_indexes": torch.arange(1, 6),
            "packed_label_ids": torch.tensor([11, 12, 13, 14, 15]),
            "ce_loss_weights": torch.ones(5),
            "sample_lens": [6],
            "d2f_response_spans": [[(0, 6)]],
            "action_token_indexes": torch.tensor([action_index]),
            "content_token_indexes": torch.tensor([content_index]),
        }
        result = rebuild_and_corrupt_responses(batch, mask_id=mask_id, block_size=16)
        selected = result["ce_loss_indexes"].long()
        content_offset = int((selected == content_index).nonzero().item())
        assert result["packed_text_ids"][content_index].item() == mask_id
        assert bool(result["content_ce_mask"][content_offset])
        assert not bool(result["action_ce_mask"][content_offset])
        assert not bool(
            (result["action_ce_mask"] & result["content_ce_mask"]).any()
        )
        if result["ce_loss_weights"][content_offset].item() == 0:
            observed_forced_only = True
        else:
            assert result["ce_loss_weights"][content_offset].item() >= 1
    assert observed_forced_only


def test_empty_content_indexes_produce_an_empty_content_ce_mask() -> None:
    batch = {
        "packed_text_ids": torch.tensor([10, 11, 12]),
        "packed_text_indexes": torch.arange(3),
        "ce_loss_indexes": torch.tensor([1, 2]),
        "packed_label_ids": torch.tensor([11, 12]),
        "ce_loss_weights": torch.ones(2),
        "sample_lens": [3],
        "d2f_response_spans": [[(0, 3)]],
        "action_token_indexes": torch.tensor([1]),
        "content_token_indexes": torch.empty(0, dtype=torch.long),
    }

    result = rebuild_and_corrupt_responses(batch, mask_id=999, block_size=2)

    assert not bool(result["content_ce_mask"].any())
    assert result["content_ce_mask"].shape == result["action_ce_mask"].shape


def test_action_and_content_indexes_must_be_disjoint() -> None:
    batch = {
        "packed_text_ids": torch.tensor([10, 11, 12]),
        "packed_text_indexes": torch.arange(3),
        "ce_loss_indexes": torch.tensor([1, 2]),
        "packed_label_ids": torch.tensor([11, 12]),
        "ce_loss_weights": torch.ones(2),
        "sample_lens": [3],
        "d2f_response_spans": [[(0, 3)]],
        "action_token_indexes": torch.tensor([1]),
        "content_token_indexes": torch.tensor([1]),
    }

    with pytest.raises(ValueError, match="must be disjoint"):
        rebuild_and_corrupt_responses(batch, mask_id=999, block_size=2)


def test_content_indexes_must_fall_inside_a_response_span() -> None:
    batch = {
        "packed_text_ids": torch.tensor([10, 11, 12, 13]),
        "packed_text_indexes": torch.arange(4),
        "ce_loss_indexes": torch.tensor([1, 2, 3]),
        "packed_label_ids": torch.tensor([11, 12, 13]),
        "ce_loss_weights": torch.ones(3),
        "sample_lens": [4],
        "d2f_response_spans": [[(0, 3)]],
        "action_token_indexes": torch.tensor([1]),
        "content_token_indexes": torch.tensor([3]),
    }

    with pytest.raises(ValueError, match="content token indexes must fall inside"):
        rebuild_and_corrupt_responses(batch, mask_id=999, block_size=2)


def test_action_indexes_must_be_supervised_answer_tokens() -> None:
    batch = {
        "packed_text_ids": torch.tensor([10, 11, 12]),
        "packed_text_indexes": torch.arange(3),
        "ce_loss_indexes": torch.tensor([1, 2]),
        "packed_label_ids": torch.tensor([11, 12]),
        "ce_loss_weights": torch.ones(2),
        "sample_lens": [3],
        "d2f_response_spans": [[(0, 3)]],
        "action_token_indexes": torch.tensor([0]),
    }
    try:
        rebuild_and_corrupt_responses(batch, mask_id=999, block_size=2)
    except ValueError as exc:
        assert "supervised answer" in str(exc)
    else:
        raise AssertionError("response BOS must not be accepted as an action token")
