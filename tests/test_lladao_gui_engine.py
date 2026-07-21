import pytest
import torch


def test_lladao_residual_norm_rounds_before_normalizing():
    from d2f_vllm.layers.layernorm import RMSNorm

    norm = RMSNorm(4, eps=1e-5, residual_in_fp32=False)
    x = torch.tensor([[0.25, -0.5, 0.75, 1.0]], dtype=torch.bfloat16)
    residual = torch.tensor([[1.0, 0.25, -0.5, 0.125]], dtype=torch.bfloat16)
    output, updated_residual = norm(x, residual)
    expected_residual = x + residual
    working = expected_residual.float()
    expected = working * torch.rsqrt(working.pow(2).mean(-1, keepdim=True) + 1e-5)
    expected = expected.to(torch.bfloat16) * norm.weight
    assert torch.equal(updated_residual, expected_residual)
    assert torch.equal(output, expected)


def test_lladao_rope_can_match_bfloat16_reference_arithmetic():
    from d2f_vllm.layers.rotary_embedding import apply_rotary_emb

    x = torch.arange(16, dtype=torch.bfloat16).view(1, 2, 8) / 8
    cos = torch.linspace(0.25, 1.0, 8)
    sin = torch.linspace(-0.5, 0.5, 8)
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    expected = x * cos.to(torch.bfloat16).unsqueeze(-2)
    expected += rotated * sin.to(torch.bfloat16).unsqueeze(-2)
    actual = apply_rotary_emb(
        x, cos, sin, compute_in_float32=False
    )
    assert torch.equal(actual, expected)


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
