from __future__ import annotations

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from lladao_d2f.masking import (
    block_attention_allowed,
    build_suffix_attention_bias,
    create_full_document_mask,
    create_training_block_mask,
)


def test_reference_block_visibility() -> None:
    prefix = 3
    block = 2
    assert block_attention_allowed(0, 2, prefix_length=prefix, block_size=block)
    assert not block_attention_allowed(0, 3, prefix_length=prefix, block_size=block)
    assert block_attention_allowed(3, 0, prefix_length=prefix, block_size=block)
    assert block_attention_allowed(3, 4, prefix_length=prefix, block_size=block)
    assert not block_attention_allowed(3, 5, prefix_length=prefix, block_size=block)
    assert block_attention_allowed(6, 3, prefix_length=prefix, block_size=block)


def test_suffix_bias_exposes_cache_and_only_prior_active_blocks() -> None:
    bias = build_suffix_attention_bias(
        cache_length=3,
        active_length=6,
        block_size=2,
        device="cpu",
        dtype=torch.float32,
    )[0, 0]
    allowed = bias == 0
    assert tuple(allowed.shape) == (6, 9)
    assert bool(allowed[:, :3].all())
    assert bool(allowed[0:2, 3:5].all())
    assert not bool(allowed[0:2, 5:].any())
    assert bool(allowed[2:4, 3:7].all())
    assert not bool(allowed[2:4, 7:].any())
    assert bool(allowed[4:6, 3:].all())


def test_suffix_bias_validates_lengths() -> None:
    try:
        build_suffix_attention_bias(
            cache_length=-1,
            active_length=4,
            block_size=2,
            device="cpu",
            dtype=torch.float32,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("negative cache length should fail")


def _assert_non_aligned_compiled_masks(device: str) -> None:
    training_mask = create_training_block_mask(
        [5],
        [[(3, 2)]],
        2,
        num_heads=1,
        device=device,
    )
    full_mask = create_full_document_mask([5], num_heads=1, device=device)

    # FlexAttention represents the partially occupied block as one dense block.
    assert tuple(training_mask.to_dense().shape) == (1, 1, 1, 1)
    assert tuple(full_mask.to_dense().shape) == (1, 1, 1, 1)
    assert bool(training_mask.to_dense().all())
    assert bool(full_mask.to_dense().all())


def test_compiled_masks_accept_non_aligned_length_on_cpu() -> None:
    _assert_non_aligned_compiled_masks("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compiled_masks_accept_non_aligned_length_on_cuda() -> None:
    _assert_non_aligned_compiled_masks("cuda")


def _assert_padded_token_mask_matches_oracle(device: str) -> None:
    sample_lens = [65, 64]
    response_spans = [[(61, 4)], []]
    total_length = sum(sample_lens)
    padded_length = 256
    num_heads = 2
    training_mask = create_training_block_mask(
        sample_lens,
        response_spans,
        2,
        num_heads=num_heads,
        device=device,
    )
    full_mask = create_full_document_mask(
        sample_lens,
        num_heads=num_heads,
        device=device,
    )
    actual_training = create_mask(
        training_mask.mask_mod,
        B=1,
        H=num_heads,
        Q_LEN=padded_length,
        KV_LEN=padded_length,
        device=device,
    )[0]
    actual_full = create_mask(
        full_mask.mask_mod,
        B=1,
        H=num_heads,
        Q_LEN=padded_length,
        KV_LEN=padded_length,
        device=device,
    )[0]

    expected_training = torch.zeros(
        (num_heads, padded_length, padded_length),
        dtype=torch.bool,
        device=device,
    )
    expected_full = torch.zeros_like(expected_training)
    for query in range(total_length):
        query_document = 0 if query < sample_lens[0] else 1
        for key in range(total_length):
            key_document = 0 if key < sample_lens[0] else 1
            if query_document != key_document:
                continue
            expected_full[:, query, key] = True
            if query_document == 1:
                # Samples without a response use full-document visibility.
                expected_training[:, query, key] = True
            elif query < 61:
                expected_training[:, query, key] = key < 61
            else:
                key_is_prefix = key < 61
                key_is_response = 61 <= key < 65
                prior_response_block = (key - 61) // 2 <= (query - 61) // 2
                expected_training[:, query, key] = key_is_prefix or (
                    key_is_response and prior_response_block
                )

    assert torch.equal(actual_training, expected_training)
    assert torch.equal(actual_full, expected_full)
    assert not bool(actual_training[:, total_length:, :].any())
    assert not bool(actual_training[:, :, total_length:].any())
    assert not bool(actual_full[:, total_length:, :].any())
    assert not bool(actual_full[:, :, total_length:].any())


def test_padded_token_masks_match_oracle_on_cpu() -> None:
    _assert_padded_token_mask_matches_oracle("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_padded_token_masks_match_oracle_on_cuda() -> None:
    _assert_padded_token_mask_matches_oracle("cuda")


@pytest.fixture
def fresh_dynamo_cache():
    import torch._dynamo

    torch._dynamo.reset()
    try:
        yield
    finally:
        torch._dynamo.reset()


def test_segmented_prefix_visibility_matches_training_contract(
    fresh_dynamo_cache,
) -> None:
    del fresh_dynamo_cache
    sample_len = 10
    padded_length = 128
    training_mask = create_training_block_mask(
        [sample_len],
        [[(6, 4)]],
        2,
        prefix_segments=[
            [
                (0, 3, "image"),
                (3, 2, "image"),
                (5, 1, "prompt"),
            ]
        ],
        num_heads=1,
        device="cpu",
    )
    actual = create_mask(
        training_mask.mask_mod,
        B=1,
        H=1,
        Q_LEN=padded_length,
        KV_LEN=padded_length,
        device="cpu",
    )[0, 0]
    expected = torch.zeros(
        (padded_length, padded_length),
        dtype=torch.bool,
    )
    expected[0:3, 0:3] = True
    expected[3:5, 0:5] = True
    expected[5, 0:6] = True
    expected[6:8, 0:8] = True
    expected[8:10, 0:10] = True

    assert torch.equal(actual, expected)


def test_segmented_prefix_applies_to_samples_without_a_response(
    fresh_dynamo_cache,
) -> None:
    del fresh_dynamo_cache
    training_mask = create_training_block_mask(
        [5],
        [[]],
        2,
        prefix_segments=[[(0, 3, "image"), (3, 2, "prompt")]],
        num_heads=1,
        device="cpu",
    )
    actual = create_mask(
        training_mask.mask_mod,
        B=1,
        H=1,
        Q_LEN=128,
        KV_LEN=128,
        device="cpu",
    )[0, 0]
    expected = torch.zeros((128, 128), dtype=torch.bool)
    expected[0:3, 0:3] = True
    expected[3:5, 0:5] = True

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ([(1, 2, "image"), (3, 3, "prompt")], "start at zero"),
        ([(0, 3, "image"), (4, 2, "prompt")], "contiguous"),
        ([(0, 3, "image"), (3, 2, "prompt")], "complete sample prefix"),
        ([(0, 3, "video"), (3, 3, "prompt")], "either 'image' or 'prompt'"),
        ([(0, 0, "image"), (0, 6, "prompt")], "must be positive"),
        ([(0, 3, "prompt"), (3, 3, "image")], "must precede prompt"),
    ],
)
def test_segmented_prefix_metadata_is_validated(segments, message) -> None:
    with pytest.raises(ValueError, match=message):
        create_training_block_mask(
            [10],
            [[(6, 4)]],
            2,
            prefix_segments=[segments],
            num_heads=1,
            device="cpu",
        )


def test_empty_prefix_segment_entry_keeps_legacy_visibility(fresh_dynamo_cache) -> None:
    del fresh_dynamo_cache
    legacy = create_training_block_mask(
        [10],
        [[(6, 4)]],
        2,
        num_heads=1,
        device="cpu",
    )
    explicit_empty = create_training_block_mask(
        [10],
        [[(6, 4)]],
        2,
        prefix_segments=[[]],
        num_heads=1,
        device="cpu",
    )

    assert torch.equal(legacy.to_dense(), explicit_empty.to_dense())


def test_compiled_masks_reuse_exact_lengths_within_one_bucket() -> None:
    """Changing valid lengths must not create one Dynamo graph per batch."""
    import torch._dynamo

    previous_cache_limit = torch._dynamo.config.cache_size_limit
    previous_accumulated_limit = torch._dynamo.config.accumulated_cache_size_limit
    torch._dynamo.reset()
    torch._dynamo.config.cache_size_limit = 8
    torch._dynamo.config.accumulated_cache_size_limit = 256
    try:
        # All ten exact lengths round to the same 256-token FlexAttention
        # bucket. Capturing the exact length as a Python integer used to exceed
        # this deliberately small cache limit before the loop completed.
        for total_length in range(129, 139):
            training_mask = create_training_block_mask(
                [total_length],
                [[(total_length - 2, 2)]],
                2,
                num_heads=1,
                device="cpu",
            )
            full_mask = create_full_document_mask(
                [total_length],
                num_heads=1,
                device="cpu",
            )
            assert tuple(training_mask.to_dense().shape) == (1, 1, 2, 2)
            assert tuple(full_mask.to_dense().shape) == (1, 1, 2, 2)
    finally:
        torch._dynamo.reset()
        torch._dynamo.config.cache_size_limit = previous_cache_limit
        torch._dynamo.config.accumulated_cache_size_limit = previous_accumulated_limit
