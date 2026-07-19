from __future__ import annotations

from collections.abc import Sequence

import torch


def _clean_packed_text_ids(batch: dict[str, object]) -> torch.Tensor:
    packed_ids = batch["packed_text_ids"].clone()
    packed_indexes = batch["packed_text_indexes"]
    old_loss_indexes = batch["ce_loss_indexes"].long()
    old_labels = batch["packed_label_ids"].long()
    packed_offsets = torch.searchsorted(packed_indexes, old_loss_indexes)
    if not torch.equal(packed_indexes[packed_offsets], old_loss_indexes):
        raise ValueError("CE loss indexes must refer to packed text tokens")
    packed_ids[packed_offsets] = old_labels
    return packed_ids


def _monotonic_probabilities(num_blocks: int, device: torch.device) -> torch.Tensor:
    first = torch.rand((), device=device) * 0.5 + 0.2
    if num_blocks == 1:
        return first.unsqueeze(0)
    increments = torch.rand(num_blocks - 1, device=device) * (0.7 - first) / (num_blocks - 1)
    return torch.cat([first.unsqueeze(0), first + torch.cumsum(increments, dim=0)]).clamp(max=1.0)


def rebuild_and_corrupt_responses(
    batch: dict[str, object],
    *,
    mask_id: int,
    block_size: int,
) -> dict[str, object]:
    """Recover clean SFT responses, then apply official D2F block corruption."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    packed_indexes: torch.Tensor = batch["packed_text_indexes"]
    clean_ids = _clean_packed_text_ids(batch)
    noisy_ids = clean_ids.clone()
    sample_lens: Sequence[int] = batch["sample_lens"]
    response_spans: Sequence[Sequence[tuple[int, int]]] = batch["d2f_response_spans"]
    if len(sample_lens) != len(response_spans):
        raise ValueError("response span metadata does not match packed samples")

    new_indexes: list[torch.Tensor] = []
    new_labels: list[torch.Tensor] = []
    new_weights: list[torch.Tensor] = []
    document_offset = 0
    for sample_len, spans in zip(sample_lens, response_spans):
        if not spans:
            document_offset += int(sample_len)
            continue
        if len(spans) != 1:
            raise ValueError("lladao_gui D2F currently supports one response per sample")
        local_start, response_length = map(int, spans[0])
        if response_length <= 1:
            raise ValueError("D2F response span must contain BOS and answer tokens")
        absolute_positions = torch.arange(
            document_offset + local_start,
            document_offset + local_start + response_length,
            device=packed_indexes.device,
        )
        packed_offsets = torch.searchsorted(packed_indexes, absolute_positions)
        if not torch.equal(packed_indexes[packed_offsets], absolute_positions):
            raise ValueError("D2F response span must contain only text tokens")
        clean_response = clean_ids[packed_offsets]
        num_blocks = (response_length + block_size - 1) // block_size
        probabilities = _monotonic_probabilities(num_blocks, clean_response.device)
        probability_per_token = probabilities.repeat_interleave(block_size)[:response_length]
        masked = torch.rand(response_length, device=clean_response.device) < probability_per_token
        # LLaDA-o uses the response BOS as a clean anchor in the first block.
        # It participates in attention/block geometry but is never a target.
        masked[0] = False
        if not bool(masked.any()):
            masked[
                torch.randint(1, response_length, (), device=clean_response.device)
            ] = True
        noisy_ids[packed_offsets[masked]] = mask_id
        new_indexes.append(absolute_positions[masked])
        new_labels.append(clean_response[masked])
        new_weights.append(probability_per_token[masked].reciprocal())
        document_offset += int(sample_len)

    if not new_indexes:
        raise ValueError("packed batch has no supervised D2F responses")
    result = dict(batch)
    result["packed_text_ids"] = noisy_ids
    result["ce_loss_indexes"] = torch.cat(new_indexes)
    result["packed_label_ids"] = torch.cat(new_labels)
    result["ce_loss_weights"] = torch.cat(new_weights)
    return result
