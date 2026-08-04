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
    full_response_mask_probability: float = 0.0,
) -> dict[str, object]:
    """Recover clean SFT responses, then apply hybrid D2F corruption.

    With probability ``full_response_mask_probability`` a response is trained
    from the inference-aligned state where every answer token is masked.  Its
    BOS remains a clean block anchor.  A zero probability intentionally avoids
    an extra RNG draw so existing recipes retain their exact corruption stream.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    full_response_mask_probability = float(full_response_mask_probability)
    if not 0.0 <= full_response_mask_probability <= 1.0:
        raise ValueError("full_response_mask_probability must be between 0 and 1")
    packed_indexes: torch.Tensor = batch["packed_text_indexes"]
    original_loss_indexes = batch["ce_loss_indexes"].to(
        device=packed_indexes.device,
        dtype=torch.long,
    )
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
    response_target_lengths = batch.get("d2f_response_target_lengths")
    if response_target_lengths is None:
        normalized_target_lengths = [
            [int(length) for _, length in spans]
            for spans in response_spans
        ]
    else:
        if len(response_target_lengths) != len(response_spans):
            raise ValueError(
                "response target length metadata does not match packed samples"
            )
        normalized_target_lengths = []
        for spans, target_lengths in zip(response_spans, response_target_lengths):
            if len(target_lengths) != len(spans):
                raise ValueError(
                    "response target lengths must match each sample's response spans"
                )
            normalized_target_lengths.append(
                [int(target_length) for target_length in target_lengths]
            )

    new_indexes: list[torch.Tensor] = []
    new_labels: list[torch.Tensor] = []
    new_weights: list[torch.Tensor] = []
    new_action_masks: list[torch.Tensor] = []
    new_content_masks: list[torch.Tensor] = []
    new_full_response_masks: list[torch.Tensor] = []
    new_full_response_group_ids: list[torch.Tensor] = []
    seen_action_indexes: list[torch.Tensor] = []
    seen_content_indexes: list[torch.Tensor] = []
    response_count = 0
    full_response_masked_count = 0
    document_offset = 0
    for sample_len, spans, target_lengths in zip(
        sample_lens,
        response_spans,
        normalized_target_lengths,
    ):
        if not spans:
            document_offset += int(sample_len)
            continue
        if len(spans) != 1:
            raise ValueError("lladao_gui D2F currently supports one response per sample")
        local_start, response_length = map(int, spans[0])
        target_length = target_lengths[0]
        if target_length <= 1:
            raise ValueError("D2F response span must contain BOS and answer tokens")
        if target_length > response_length:
            raise ValueError(
                "D2F response target length cannot exceed its response span"
            )
        full_absolute_positions = torch.arange(
            document_offset + local_start,
            document_offset + local_start + response_length,
            device=packed_indexes.device,
        )
        full_packed_offsets = torch.searchsorted(
            packed_indexes,
            full_absolute_positions,
        )
        if (
            len(full_packed_offsets)
            and int(full_packed_offsets[-1]) >= len(packed_indexes)
        ) or not torch.equal(
            packed_indexes[full_packed_offsets],
            full_absolute_positions,
        ):
            raise ValueError("D2F response span must contain only text tokens")
        tail_absolute_positions = full_absolute_positions[target_length:]
        tail_packed_offsets = full_packed_offsets[target_length:]
        if len(tail_absolute_positions):
            if bool(torch.isin(tail_absolute_positions, original_loss_indexes).any()):
                raise ValueError(
                    "reserved response MASK slots cannot be supervised targets"
                )
            if not bool((clean_ids[tail_packed_offsets] == mask_id).all()):
                raise ValueError(
                    "reserved response slots must contain only MASK tokens"
                )
        absolute_positions = torch.arange(
            document_offset + local_start,
            document_offset + local_start + target_length,
            device=packed_indexes.device,
        )
        packed_offsets = full_packed_offsets[:target_length]
        clean_response = clean_ids[packed_offsets]
        response_group_id = response_count
        response_count += 1
        if full_response_mask_probability == 0.0:
            full_response_masked = False
        elif full_response_mask_probability == 1.0:
            full_response_masked = True
        else:
            full_response_masked = bool(
                torch.rand((), device=clean_response.device)
                < full_response_mask_probability
            )
        if full_response_masked:
            if not bool(
                torch.isin(absolute_positions[1:], original_loss_indexes).all()
            ):
                raise ValueError(
                    "full-response corruption requires every answer token, "
                    "including EOS, to be supervised"
                )
            probability_per_token = torch.ones(
                target_length,
                device=clean_response.device,
                dtype=torch.float32,
            )
            random_masked = torch.ones(
                target_length,
                device=clean_response.device,
                dtype=torch.bool,
            )
            full_response_masked_count += 1
        else:
            num_blocks = (target_length + block_size - 1) // block_size
            probabilities = _monotonic_probabilities(num_blocks, clean_response.device)
            probability_per_token = probabilities.repeat_interleave(block_size)[
                :target_length
            ]
            random_masked = (
                torch.rand(target_length, device=clean_response.device)
                < probability_per_token
            )
        # LLaDA-o uses the response BOS as a clean anchor in the first block.
        # It participates in attention/block geometry but is never a target.
        random_masked[0] = False
        if not bool(random_masked.any()):
            random_masked[
                torch.randint(1, target_length, (), device=clean_response.device)
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
        full_response_ce_mask = torch.full_like(
            selected,
            full_response_masked,
            dtype=torch.bool,
        )
        full_response_ce_mask[0] = False
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
        new_full_response_masks.append(full_response_ce_mask[selected])
        new_full_response_group_ids.append(
            torch.where(
                full_response_ce_mask[selected],
                torch.full_like(
                    absolute_positions[selected],
                    response_group_id,
                    dtype=torch.long,
                ),
                torch.full_like(
                    absolute_positions[selected],
                    -1,
                    dtype=torch.long,
                ),
            )
        )
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
    result["full_response_ce_mask"] = torch.cat(new_full_response_masks)
    result["full_response_group_ids"] = torch.cat(new_full_response_group_ids)
    result["full_response_masked_count"] = torch.tensor(
        full_response_masked_count,
        device=packed_indexes.device,
        dtype=torch.long,
    )
    result["d2f_response_count"] = torch.tensor(
        response_count,
        device=packed_indexes.device,
        dtype=torch.long,
    )
    return result
