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


def _validated_auxiliary_indexes(
    batch: dict[str, object],
    *,
    key: str,
    display_name: str,
    packed_indexes: torch.Tensor,
    allow_empty: bool,
) -> torch.Tensor:
    value = batch.get(key)
    if value is None:
        return torch.empty(0, dtype=torch.long, device=packed_indexes.device)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{display_name} token indexes must be a tensor")
    indexes = value.to(device=packed_indexes.device, dtype=torch.long)
    if indexes.ndim != 1 or (not allow_empty and len(indexes) == 0):
        qualifier = "one vector" if allow_empty else "one non-empty vector"
        raise ValueError(f"{display_name} token indexes must be {qualifier}")
    if len(torch.unique(indexes)) != len(indexes):
        raise ValueError(f"{display_name} token indexes must be unique")
    old_loss_indexes = batch["ce_loss_indexes"].to(
        device=packed_indexes.device,
        dtype=torch.long,
    )
    if not bool(torch.isin(indexes, old_loss_indexes).all()):
        raise ValueError(
            f"{display_name} token indexes must be supervised answer tokens"
        )
    return indexes


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
    action_indexes = _validated_auxiliary_indexes(
        batch,
        key="action_token_indexes",
        display_name="action",
        packed_indexes=packed_indexes,
        allow_empty=False,
    )
    content_indexes = _validated_auxiliary_indexes(
        batch,
        key="content_token_indexes",
        display_name="content",
        packed_indexes=packed_indexes,
        allow_empty=True,
    )
    if bool(torch.isin(action_indexes, content_indexes).any()):
        raise ValueError("action and content token indexes must be disjoint")
    clean_ids = _clean_packed_text_ids(batch)
    noisy_ids = clean_ids.clone()
    sample_lens: Sequence[int] = batch["sample_lens"]
    response_spans: Sequence[Sequence[tuple[int, int]]] = batch["d2f_response_spans"]
    if len(sample_lens) != len(response_spans):
        raise ValueError("response span metadata does not match packed samples")

    new_indexes: list[torch.Tensor] = []
    new_labels: list[torch.Tensor] = []
    new_weights: list[torch.Tensor] = []
    new_action_masks: list[torch.Tensor] = []
    new_content_masks: list[torch.Tensor] = []
    seen_action_indexes: list[torch.Tensor] = []
    seen_content_indexes: list[torch.Tensor] = []
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
        random_masked = (
            torch.rand(response_length, device=clean_response.device) < probability_per_token
        )
        # LLaDA-o uses the response BOS as a clean anchor in the first block.
        # It participates in attention/block geometry but is never a target.
        random_masked[0] = False
        if not bool(random_masked.any()):
            random_masked[
                torch.randint(1, response_length, (), device=clean_response.device)
            ] = True
        forced_action_masked = torch.isin(absolute_positions, action_indexes)
        forced_content_masked = torch.isin(absolute_positions, content_indexes)
        if bool(forced_action_masked[0]):
            raise ValueError("response BOS cannot be an action token")
        if bool(forced_content_masked[0]):
            raise ValueError("response BOS cannot be a content token")
        if bool((forced_action_masked & forced_content_masked).any()):
            raise ValueError("action and content CE masks must be disjoint")
        forced_masked = forced_action_masked | forced_content_masked
        selected = random_masked | forced_masked
        noisy_ids[packed_offsets[selected]] = mask_id
        new_indexes.append(absolute_positions[selected])
        new_labels.append(clean_response[selected])
        # Tokens selected only by either forced auxiliary mask are excluded
        # from the original D2F importance-sampling estimator.  They enter the
        # separate action/content CE terms instead.
        new_weights.append(
            torch.where(
                random_masked[selected],
                probability_per_token[selected].reciprocal(),
                torch.zeros_like(probability_per_token[selected]),
            )
        )
        new_action_masks.append(forced_action_masked[selected])
        new_content_masks.append(forced_content_masked[selected])
        if bool(forced_action_masked.any()):
            seen_action_indexes.append(absolute_positions[forced_action_masked])
        if bool(forced_content_masked.any()):
            seen_content_indexes.append(absolute_positions[forced_content_masked])
        document_offset += int(sample_len)

    if not new_indexes:
        raise ValueError("packed batch has no supervised D2F responses")
    if len(action_indexes):
        seen = torch.cat(seen_action_indexes) if seen_action_indexes else action_indexes[:0]
        if not torch.equal(torch.sort(seen).values, torch.sort(action_indexes).values):
            raise ValueError("action token indexes must fall inside one response span")
    if len(content_indexes):
        seen = (
            torch.cat(seen_content_indexes)
            if seen_content_indexes
            else content_indexes[:0]
        )
        if not torch.equal(torch.sort(seen).values, torch.sort(content_indexes).values):
            raise ValueError("content token indexes must fall inside one response span")
    result = dict(batch)
    result["packed_text_ids"] = noisy_ids
    result["ce_loss_indexes"] = torch.cat(new_indexes)
    result["packed_label_ids"] = torch.cat(new_labels)
    result["ce_loss_weights"] = torch.cat(new_weights)
    result["action_ce_mask"] = torch.cat(new_action_masks)
    result["content_ce_mask"] = torch.cat(new_content_masks)
    return result
