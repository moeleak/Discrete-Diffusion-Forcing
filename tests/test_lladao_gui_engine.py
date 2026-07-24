import pytest
import torch
import torch.nn.functional as F


def test_exact_runtime_lora_matches_peft_inference_arithmetic():
    from d2f_vllm.models.lladao_gui import _ExactLoRAMixin

    class ExactLoRAForTest(_ExactLoRAMixin, torch.nn.Module):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.tp_size = 1
            self._init_exact_lora(2, 2.0, 4, 3)

    module = ExactLoRAForTest()
    module.lora_A.data.copy_(
        torch.tensor([[0.5, -0.25, 0.125, 0.75], [-0.5, 0.25, 0.5, -0.125]])
    )
    module.lora_B.data.copy_(
        torch.tensor([[0.25, -0.5], [0.75, 0.125], [-0.25, 0.5]])
    )
    hidden = torch.tensor(
        [[0.5, -0.25, 1.0, 0.125]], dtype=torch.bfloat16
    )
    base_output = torch.tensor(
        [[0.25, -0.5, 0.75]], dtype=torch.bfloat16
    )
    delta = F.linear(F.linear(hidden.float(), module.lora_A), module.lora_B)
    expected = (base_output + delta).to(torch.bfloat16)
    actual = module._apply_exact_lora(hidden, base_output)
    assert torch.equal(actual, expected)

    shared_input = hidden.float()
    shared = module._apply_exact_lora(
        hidden, base_output, lora_input=shared_input
    )
    assert torch.equal(shared, expected)


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
    cos = torch.linspace(0.25, 1.0, 4)
    sin = torch.linspace(-0.5, 0.5, 4)
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    cos_full = torch.cat((cos, cos)).to(torch.bfloat16).unsqueeze(-2)
    sin_full = torch.cat((sin, sin)).to(torch.bfloat16).unsqueeze(-2)
    expected = x * cos_full
    expected += rotated * sin_full
    actual = apply_rotary_emb(
        x, cos, sin, compute_in_float32=False
    )
    assert torch.equal(actual, expected)


def test_yarn_matches_transformers_449_reference_formula():
    import math

    from d2f_vllm.layers.rotary_embedding import compute_yarn_parameters

    rotary_dim = 128
    base = 500_000.0
    original_max = 16_384
    factor = 8.0
    actual, scaling = compute_yarn_parameters(
        rotary_dim,
        base,
        original_max,
        factor,
    )

    def correction_dim(rotations):
        return (
            rotary_dim
            * math.log(original_max / (rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    low = max(math.floor(correction_dim(32)), 0)
    high = min(math.ceil(correction_dim(1)), rotary_dim - 1)
    frequencies = base ** (
        torch.arange(0, rotary_dim, 2).float() / rotary_dim
    )
    extrapolated = 1.0 / frequencies
    interpolated = 1.0 / (factor * frequencies)
    ramp = (
        (torch.arange(rotary_dim // 2).float() - low) / (high - low)
    ).clamp(0, 1)
    expected = interpolated * ramp + extrapolated * (1 - ramp)
    assert torch.equal(actual, expected)
    assert scaling == pytest.approx(1.0 + 0.1 * math.log(8.0))


def test_yarn_cache_supports_last_128k_position():
    from d2f_vllm.layers.rotary_embedding import get_rope

    rope = get_rope(
        128,
        128,
        131_072,
        500_000.0,
        {
            "rope_type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 16_384,
        },
    )
    query = torch.ones(1, 128)
    key = torch.ones(1, 128)
    rotated_query, rotated_key = rope(
        torch.tensor([131_071]), query, key
    )
    assert torch.isfinite(rotated_query).all()
    assert torch.isfinite(rotated_key).all()


def test_full_page_tiles_cover_source_in_row_major_order():
    from d2f_vllm.multimodal.lladao_gui import full_page_tile_boxes

    assert full_page_tile_boxes(1_318, 2_100) == [
        (0, 0, 980, 980),
        (980, 0, 1_318, 980),
        (0, 980, 980, 1_960),
        (980, 980, 1_318, 1_960),
        (0, 1_960, 980, 2_100),
        (980, 1_960, 1_318, 2_100),
    ]


def test_native_multimodal_positions_share_one_position_per_image():
    from d2f_vllm.multimodal.lladao_gui import (
        build_multimodal_position_ids,
    )

    image, prompt = build_multimodal_position_ids(
        [5, 3],
        4,
        mode="native",
    )
    assert image == [0] * 5 + [1] * 3
    assert prompt == [2, 3, 4, 5]


def test_sequential_multimodal_positions_reach_the_dense_token_length():
    from d2f_vllm.multimodal.lladao_gui import (
        build_multimodal_position_ids,
    )

    image, prompt = build_multimodal_position_ids(
        [4_902] * 13,
        100,
        mode="sequential",
    )
    assert image[0] == 0
    assert image[-1] == 63_725
    assert prompt[0] == 63_726
    assert prompt[-1] == 63_825
    assert image + prompt == list(range(63_826))


def test_multimodal_positions_reject_unknown_mode():
    from d2f_vllm.multimodal.lladao_gui import (
        build_multimodal_position_ids,
    )

    with pytest.raises(ValueError, match="native, sequential"):
        build_multimodal_position_ids([5], 2, mode="compressed")


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


def test_vision_tiles_preserve_two_dimensional_regions():
    from d2f_vllm.lladao_gui_engine import build_vision_tiles

    tiles = build_vision_tiles(3, 5, 2)
    assert tiles == [
        [0, 1, 5, 6],
        [2, 3, 7, 8],
        [4, 9],
        [10, 11],
        [12, 13],
        [14],
    ]


def test_vision_tile_selection_uses_peak_patch_attention():
    from d2f_vllm.lladao_gui_engine import select_top_vision_tiles

    scores = torch.tensor([0.1, 0.2, 0.3, 0.9, 0.4, 0.5])
    tiles = [[0, 1], [2, 3], [4, 5]]
    assert select_top_vision_tiles(scores, tiles, 1) == [1]
    assert select_top_vision_tiles(scores, tiles, 0) == [0, 1, 2]


def test_patch_eviction_selects_tokens_per_kv_head():
    from d2f_vllm.lladao_gui_engine import select_patch_tokens_per_head

    scores = torch.tensor(
        [
            [0.1, 0.8, 0.7, 0.2],
            [0.9, 0.1, 0.2, 0.8],
        ]
    )
    candidates = torch.tensor([0, 1, 2, 3])
    selected = select_patch_tokens_per_head(scores, candidates, 2)
    assert torch.equal(selected, torch.tensor([[1, 2], [0, 3]]))


def test_multi_image_kv_selection_keeps_each_boundary_and_prompt():
    from types import SimpleNamespace

    from d2f_vllm.lladao_gui_engine import (
        LLaDAOGuiD2FEngine,
        LLaDAOGuiKVCompressionConfig,
    )
    from d2f_vllm.multimodal.lladao_gui import (
        LLaDAOGuiImageSpan,
        LLaDAOGuiPrefix,
    )

    prefix = LLaDAOGuiPrefix(
        image_ids=[1, 0, 0, 0, 2, 1, 0, 0, 0, 2],
        image_positions=[0] * 5 + [1] * 5,
        image_embeddings=torch.empty(10, 4),
        image_spans=[
            LLaDAOGuiImageSpan(0, 1, 4, 5, 1, 3, (0, 0, 3, 1)),
            LLaDAOGuiImageSpan(5, 6, 9, 10, 1, 3, (3, 0, 6, 1)),
        ],
        source_width=6,
        source_height=1,
        prompt_ids=[3, 4],
        prompt_positions=[2, 3],
    )
    engine = object.__new__(LLaDAOGuiD2FEngine)
    engine.kv_compression = LLaDAOGuiKVCompressionConfig(
        enabled=True,
        vision_topk_tiles=0,
        vision_token_keep_ratio=0.5,
        vision_score_pool_kernel=1,
    )
    engine.model = SimpleNamespace(
        model=SimpleNamespace(layers=[object(), object()])
    )
    scores = {
        0: torch.tensor(
            [
                [0.1, 0.9, 0.2, 0.8, 0.1, 0.7],
                [0.9, 0.1, 0.2, 0.1, 0.8, 0.7],
            ]
        )
    }
    keep, stats = engine._build_vision_keep_indices(prefix, scores)
    assert stats["vision_kept_patches"] == 4
    assert stats["cached_prefix_tokens"] == 10
    assert len(keep) == 2
    assert keep[0].shape == (2, 10)
    for head in keep[0]:
        assert {0, 4, 5, 9, 10, 11}.issubset(set(head.tolist()))


def test_kv_compression_config_rejects_even_pool_kernel():
    from d2f_vllm.lladao_gui_engine import LLaDAOGuiKVCompressionConfig

    with pytest.raises(ValueError, match="positive odd"):
        LLaDAOGuiKVCompressionConfig(vision_score_pool_kernel=4)


def test_sdpa_mask_is_cached_on_the_decode_context():
    from d2f_vllm.layers.attention.attention_v4 import Attention
    from d2f_vllm.utils.context import ContextForDiffusionLM

    allowed = torch.tensor([[True, False], [True, True]])
    context = ContextForDiffusionLM(block_mask=allowed)
    reference = torch.empty(1, dtype=torch.bfloat16)
    first = Attention._cached_sdpa_mask(context, reference)
    second = Attention._cached_sdpa_mask(context, reference)
    assert first is second
    assert first.shape == (1, 1, 2, 2)
    assert first.dtype == torch.bfloat16
    assert first[0, 0, 0, 0] == 0
    assert first[0, 0, 0, 1] == torch.finfo(torch.bfloat16).min


def test_eager_silu_and_mul_matches_reference_expression():
    from d2f_vllm.layers.activation import SiluAndMul

    inputs = torch.linspace(-2, 2, 16, dtype=torch.bfloat16).view(2, 8)
    left, right = inputs.chunk(2, dim=-1)
    expected = F.silu(left) * right
    assert torch.equal(SiluAndMul()(inputs), expected)
