from __future__ import annotations

import torch

from lladao_d2f.masking import block_attention_allowed, build_suffix_attention_bias


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
