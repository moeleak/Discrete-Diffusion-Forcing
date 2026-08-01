from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn.attention.flex_attention import create_block_mask


FLEX_MASK_BLOCK_SIZE = 128


def _padded_mask_length(total_length: int) -> int:
    """Round metadata up to FlexAttention's block-mask evaluation grid."""
    return (
        (total_length + FLEX_MASK_BLOCK_SIZE - 1)
        // FLEX_MASK_BLOCK_SIZE
        * FLEX_MASK_BLOCK_SIZE
    )


def block_attention_allowed(
    query_position: int,
    key_position: int,
    *,
    prefix_length: int,
    block_size: int,
) -> bool:
    """Reference D2F visibility rule for tests and small masks."""
    if query_position < prefix_length:
        return key_position < prefix_length
    if key_position < prefix_length:
        return True
    query_block = (query_position - prefix_length) // block_size
    key_block = (key_position - prefix_length) // block_size
    return key_block <= query_block


def build_suffix_attention_bias(
    cache_length: int,
    active_length: int,
    block_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build [1, 1, Q, cache+Q] bias for pipelined active blocks.

    Cached image, prompt, and completed response blocks are visible to every
    active query.  Active blocks are bidirectional internally and causal
    between blocks.
    """
    if cache_length < 0 or active_length <= 0 or block_size <= 0:
        raise ValueError("cache_length must be non-negative and lengths must be positive")
    query_positions = torch.arange(active_length, device=device)
    key_positions = torch.arange(active_length, device=device)
    query_blocks = torch.div(query_positions, block_size, rounding_mode="floor")
    key_blocks = torch.div(key_positions, block_size, rounding_mode="floor")
    active_allowed = key_blocks.unsqueeze(0) <= query_blocks.unsqueeze(1)
    allowed = torch.ones(
        (active_length, cache_length + active_length),
        dtype=torch.bool,
        device=device,
    )
    allowed[:, cache_length:] = active_allowed
    bias = torch.zeros_like(allowed, dtype=dtype)
    bias.masked_fill_(~allowed, torch.finfo(dtype).min)
    return bias.unsqueeze(0).unsqueeze(0)


def create_training_block_mask(
    sample_lens: Sequence[int],
    response_spans: Sequence[Sequence[tuple[int, int]]],
    block_size: int,
    *,
    num_heads: int,
    device: torch.device | str,
):
    """Create a packed FlexAttention mask for multimodal D2F training."""
    if len(sample_lens) != len(response_spans):
        raise ValueError("sample_lens and response_spans must have equal length")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    document_ids: list[int] = []
    local_positions: list[int] = []
    response_starts: list[int] = []
    response_ends: list[int] = []
    for document_id, (sample_len, spans) in enumerate(zip(sample_lens, response_spans)):
        sample_len = int(sample_len)
        if sample_len < 0:
            raise ValueError("sample lengths must be non-negative")
        if len(spans) > 1:
            raise ValueError("lladao_gui D2F currently supports one response per sample")
        if spans:
            start, length = map(int, spans[0])
            if start < 0 or length <= 0 or start + length > sample_len:
                raise ValueError(f"invalid response span {(start, length)} for length {sample_len}")
            end = start + length
        else:
            start = end = -1
        document_ids.extend([document_id] * sample_len)
        local_positions.extend(range(sample_len))
        response_starts.extend([start] * sample_len)
        response_ends.extend([end] * sample_len)

    total_length = sum(map(int, sample_lens))
    padded_length = _padded_mask_length(total_length)
    metadata_padding = padded_length - total_length
    # create_block_mask evaluates mask_mod over complete 128-token blocks even
    # when Q_LEN/KV_LEN are not aligned.  Pad every indexed tensor so those
    # speculative q/k indices are safe, then reject them with explicit bounds.
    document_ids.extend([-1] * metadata_padding)
    local_positions.extend([-1] * metadata_padding)
    response_starts.extend([-1] * metadata_padding)
    response_ends.extend([-1] * metadata_padding)
    doc = torch.tensor(document_ids, device=device, dtype=torch.int32)
    local = torch.tensor(local_positions, device=device, dtype=torch.int32)
    starts = torch.tensor(response_starts, device=device, dtype=torch.int32)
    ends = torch.tensor(response_ends, device=device, dtype=torch.int32)
    # Keep the exact, unpadded length out of the Python closure guards used by
    # create_block_mask(_compile=True).  A Python integer here specializes the
    # compiled helper once per batch length and eventually exhausts Dynamo's
    # cache even when those lengths share the same 128-token compile bucket.
    valid_length = torch.scalar_tensor(total_length, device=device, dtype=torch.int32)

    def mask_mod(batch, head, query_index, key_index):
        del batch, head
        valid_query = query_index < valid_length
        valid_key = key_index < valid_length
        same_document = doc[query_index] == doc[key_index]
        start = starts[query_index]
        end = ends[query_index]
        no_response = start < 0
        query_local = local[query_index]
        key_local = local[key_index]
        query_prefix = query_local < start
        key_prefix = key_local < start
        query_response = (query_local >= start) & (query_local < end)
        key_response = (key_local >= start) & (key_local < end)
        query_block = torch.div(query_local - start, block_size, rounding_mode="floor")
        key_block = torch.div(key_local - start, block_size, rounding_mode="floor")
        response_allowed = query_response & (
            key_prefix | (key_response & (key_block <= query_block))
        )
        prefix_allowed = query_prefix & key_prefix
        return (
            valid_query
            & valid_key
            & same_document
            & (no_response | prefix_allowed | response_allowed)
        )

    return create_block_mask(
        mask_mod,
        B=1,
        H=num_heads,
        Q_LEN=total_length,
        KV_LEN=total_length,
        device=device,
        BLOCK_SIZE=FLEX_MASK_BLOCK_SIZE,
        _compile=True,
    )


def create_full_document_mask(
    sample_lens: Sequence[int],
    *,
    num_heads: int,
    device: torch.device | str,
):
    document_ids: list[int] = []
    for document_id, sample_len in enumerate(sample_lens):
        document_ids.extend([document_id] * int(sample_len))
    total_length = sum(map(int, sample_lens))
    padded_length = _padded_mask_length(total_length)
    document_ids.extend([-1] * (padded_length - total_length))
    doc = torch.tensor(document_ids, device=device, dtype=torch.int32)
    valid_length = torch.scalar_tensor(total_length, device=device, dtype=torch.int32)

    def mask_mod(batch, head, query_index, key_index):
        del batch, head
        valid_query = query_index < valid_length
        valid_key = key_index < valid_length
        return valid_query & valid_key & (doc[query_index] == doc[key_index])

    return create_block_mask(
        mask_mod,
        B=1,
        H=num_heads,
        Q_LEN=total_length,
        KV_LEN=total_length,
        device=device,
        BLOCK_SIZE=FLEX_MASK_BLOCK_SIZE,
        _compile=True,
    )
