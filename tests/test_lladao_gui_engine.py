import pytest
import torch


def test_generation_attention_mask_is_block_causal():
    from d2f_vllm.lladao_gui_engine import build_generation_attention_mask

    mask = build_generation_attention_mask(
        3, 8, 4, device=torch.device("cpu")
    )
    assert mask.shape == (8, 11)
    assert mask[:, :3].all()
    assert mask[:4, 3:7].all()
    assert not mask[:4, 7:].any()
    assert mask[4:, 3:].all()


def test_generation_attention_mask_rejects_partial_blocks():
    from d2f_vllm.lladao_gui_engine import build_generation_attention_mask

    with pytest.raises(ValueError):
        build_generation_attention_mask(3, 6, 4, device=torch.device("cpu"))
