from __future__ import annotations

import torch

from lladao_d2f.noise import rebuild_and_corrupt_responses


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
