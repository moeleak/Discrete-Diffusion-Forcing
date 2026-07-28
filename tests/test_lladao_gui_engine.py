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


def test_full_page_truncation_keeps_complete_row_major_tiles():
    from d2f_vllm.multimodal.lladao_gui import (
        full_page_tile_token_length,
        truncate_full_page_tile_boxes,
    )

    boxes = [
        (0, 0, 980, 980),
        (980, 0, 1_318, 980),
        (0, 980, 980, 1_960),
    ]
    first = full_page_tile_token_length(boxes[0])
    second = full_page_tile_token_length(boxes[1])
    kept, used = truncate_full_page_tile_boxes(
        boxes,
        image_token_budget=first + second,
    )
    assert kept == boxes[:2]
    assert used == first + second
    assert full_page_tile_token_length((0, 0, 980, 980)) == 4_902


def test_full_page_truncation_requires_one_complete_tile():
    from d2f_vllm.multimodal.lladao_gui import (
        truncate_full_page_tile_boxes,
    )

    with pytest.raises(ValueError, match="cannot fit one complete"):
        truncate_full_page_tile_boxes(
            [(0, 0, 980, 980)],
            image_token_budget=4_901,
        )


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


def test_strided_multimodal_positions_preserve_native_image_invariant():
    from d2f_vllm.multimodal.lladao_gui import (
        build_multimodal_position_ids,
    )

    image, prompt = build_multimodal_position_ids(
        [4_902, 1_752, 4_902, 1_752],
        4,
        mode="strided",
    )
    assert image == (
        [0] * 4_902
        + [4_902] * 1_752
        + [6_654] * 4_902
        + [11_556] * 1_752
    )
    assert prompt == [13_308, 13_309, 13_310, 13_311]


def test_overview_grounding_prompt_uses_runtime_tile_count_and_global_anchor():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "D2F-eval"
        / "eval_lladao_gui.py"
    )
    spec = importlib.util.spec_from_file_location("eval_lladao_gui", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    prompt = module.overview_grounding_prompt(
        {
            "prompt": "Click on Submit.",
            "native_prompt": "Click on Submit.",
            "tile_layout": [{}, {}, {}],
            "image_width": 1_280,
            "image_height": 4_000,
        },
        tile_size=490,
    )
    assert "first 27 images are exact non-overlapping tiles" in prompt
    assert "final image is a resized overview" in prompt
    assert "complete original screenshot in [0,1000]" in prompt
    assert "Click on Submit." in prompt


def test_full_page_grounding_prompt_uses_runtime_tile_count():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "D2F-eval"
        / "eval_lladao_gui.py"
    )
    spec = importlib.util.spec_from_file_location("eval_lladao_gui", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    prompt = module.full_page_grounding_prompt(
        {
            "prompt": "Click on Quick Tools.",
            "native_prompt": "Click on Quick Tools.",
            "tile_layout": [{}] * 12,
            "image_width": 1_318,
            "image_height": 5_283,
        },
        tile_size=686,
    )

    assert "following 16 images are non-overlapping tiles" in prompt
    assert prompt.count("Click on Quick Tools.") == 1


def test_retrieval_query_strips_full_page_transport_wrapper():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "D2F-eval"
        / "eval_lladao_gui.py"
    )
    spec = importlib.util.spec_from_file_location("eval_lladao_gui", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sample = {
        "prompt": (
            "The following 12 images are non-overlapping tiles from one "
            "1318x5283 webpage screenshot, ordered left-to-right and then "
            "top-to-bottom. Treat them as one complete page. "
            "Click on Quick Tools. Return the action and bounding box with "
            "coordinates normalized to the complete original screenshot in "
            "[0,1000]."
        ),
        "tile_layout": [{}] * 12,
        "image_width": 1_318,
        "image_height": 5_283,
    }
    assert module.native_resize_prompt(sample) == "Click on Quick Tools."
    overview = module.overview_grounding_prompt(sample)
    assert overview.count("Click on Quick Tools.") == 1
    assert overview.count("Return the action and bounding box") == 1


def test_multimodal_positions_reject_unknown_mode():
    from d2f_vllm.multimodal.lladao_gui import (
        build_multimodal_position_ids,
    )

    with pytest.raises(ValueError, match="native, strided, sequential"):
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


def test_retrieval_masked_queries_are_complementary_and_keep_boundaries():
    from d2f_vllm.lladao_gui_engine import (
        build_complementary_masked_queries,
    )

    variants = build_complementary_masked_queries(
        [100, 10, 11, 12, 101],
        99,
        mask_rounds=2,
    )
    assert [variant.token_ids for variant in variants] == [
        (100, 99, 11, 99, 101),
        (100, 10, 99, 12, 101),
    ]
    assert [variant.target_indices for variant in variants] == [
        (1, 3),
        (2,),
    ]
    assert [variant.target_ids for variant in variants] == [
        (10, 12),
        (11,),
    ]
    scored = [
        index
        for variant in variants
        for index in variant.target_indices
    ]
    assert sorted(scored) == [1, 2, 3]
    assert len(scored) == len(set(scored))


def test_retrieval_masked_queries_require_positive_round_count():
    from d2f_vllm.lladao_gui_engine import (
        build_complementary_masked_queries,
    )

    with pytest.raises(ValueError, match="positive"):
        build_complementary_masked_queries(
            [100, 10, 101],
            99,
            mask_rounds=0,
        )


def test_retrieval_masked_scoring_batches_preserve_order_and_soft_cap():
    from d2f_vllm.lladao_gui_engine import (
        MaskedScoringRequest,
        build_masked_scoring_batches,
    )

    batches = build_masked_scoring_batches(
        [(0, 5), (2, 12)],
        query_length=3,
        variant_count=2,
        max_batch_tokens=15,
    )
    assert batches == [
        [MaskedScoringRequest(0, 0, 8)],
        [MaskedScoringRequest(0, 1, 8)],
        [MaskedScoringRequest(2, 0, 15)],
        [MaskedScoringRequest(2, 1, 15)],
    ]
    oversized = build_masked_scoring_batches(
        [(4, 20)],
        query_length=3,
        variant_count=1,
        max_batch_tokens=10,
    )
    assert oversized == [[MaskedScoringRequest(4, 0, 23)]]


def test_retrieval_legacy_scoring_mask_is_causal_after_stored_context():
    from d2f_vllm.lladao_gui_engine import (
        build_causal_append_attention_mask,
    )

    mask = build_causal_append_attention_mask(
        2,
        3,
        device=torch.device("cpu"),
    )
    assert mask.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, False],
        [True, True, True, True, True],
    ]


def test_retrieval_cached_scoring_mask_is_bidirectional_after_stored_context():
    from d2f_vllm.lladao_gui_engine import (
        build_bidirectional_append_attention_mask,
    )

    mask = build_bidirectional_append_attention_mask(
        2,
        3,
        device=torch.device("cpu"),
    )
    assert mask.tolist() == [
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True],
    ]


def test_joint_masked_query_selects_same_position_logits_under_full_prefill():
    from types import SimpleNamespace

    from d2f_vllm.lladao_gui_engine import (
        LLaDAOGuiD2FEngine,
        MaskedQueryVariant,
    )
    from d2f_vllm.multimodal.lladao_gui import (
        LLaDAOGuiImageSpan,
        LLaDAOGuiPrefix,
    )

    class FakeModel:
        def __init__(self):
            self.call = None

        def __call__(self, input_ids, positions, *, input_embeds):
            self.call = (input_ids, positions.clone(), input_embeds.clone())
            return input_embeds

        @staticmethod
        def compute_logits(hidden):
            return hidden

    engine = object.__new__(LLaDAOGuiD2FEngine)
    embedding = torch.nn.Embedding(128, 3)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.arange(128 * 3, dtype=torch.float32).reshape(128, 3)
        )
    engine.prefix_encoder = SimpleNamespace(token_embedding=embedding)
    engine.model = FakeModel()
    engine._ids_tensor = lambda ids: torch.tensor(ids, dtype=torch.long)
    engine._positions_tensor = lambda positions: torch.tensor(
        positions, dtype=torch.long
    )
    full_prefill_calls = []
    engine._set_full_prefill_context = (
        lambda length, slots, *, need_kv_cache_store: full_prefill_calls.append(
            (length, slots.clone(), need_kv_cache_store)
        )
    )
    prefix = LLaDAOGuiPrefix(
        image_ids=[1, 2],
        image_positions=[7, 7],
        image_embeddings=torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        ),
        image_spans=[
            LLaDAOGuiImageSpan(
                token_start=0,
                patch_start=1,
                patch_end=1,
                token_end=2,
                grid_height=0,
                grid_width=0,
                source_box=(0, 0, 1, 1),
            )
        ],
        source_width=1,
        source_height=1,
        prompt_ids=[],
        prompt_positions=[],
    )
    variant = MaskedQueryVariant(
        token_ids=(100, 99, 101),
        target_indices=(1,),
        target_ids=(10,),
    )
    logits = engine._forward_joint_masked_query(
        prefix,
        0,
        variant,
        [8, 9, 10],
    )
    assert torch.equal(logits, embedding(torch.tensor([99])))
    assert len(full_prefill_calls) == 1
    assert full_prefill_calls[0][0] == 5
    assert torch.equal(
        full_prefill_calls[0][1],
        torch.arange(5, dtype=torch.int32),
    )
    assert full_prefill_calls[0][2] is False
    assert engine.model.call[0] is None
    assert engine.model.call[1].tolist() == [7, 7, 8, 9, 10]


def test_packed_masked_queries_keep_documents_and_targets_separate():
    from types import SimpleNamespace

    from d2f_vllm.lladao_gui_engine import (
        LLaDAOGuiD2FEngine,
        MaskedQueryVariant,
        MaskedScoringRequest,
    )
    from d2f_vllm.multimodal.lladao_gui import (
        LLaDAOGuiImageSpan,
        LLaDAOGuiPrefix,
    )

    class FakeModel:
        def __init__(self):
            self.call = None

        def __call__(self, input_ids, positions, *, input_embeds):
            self.call = (input_ids, positions.clone(), input_embeds.clone())
            return input_embeds

        @staticmethod
        def compute_logits(hidden):
            return hidden

    engine = object.__new__(LLaDAOGuiD2FEngine)
    embedding = torch.nn.Embedding(128, 3)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.arange(128 * 3, dtype=torch.float32).reshape(128, 3)
        )
    engine.prefix_encoder = SimpleNamespace(token_embedding=embedding)
    engine.model = FakeModel()
    engine._ids_tensor = lambda ids: torch.tensor(ids, dtype=torch.long)
    engine._positions_tensor = lambda positions: torch.tensor(
        positions, dtype=torch.long
    )
    packed_calls = []
    engine._set_packed_full_prefill_context = (
        lambda lengths, slots: packed_calls.append(
            (list(lengths), slots.clone())
        )
    )
    prefix = LLaDAOGuiPrefix(
        image_ids=[1, 2, 3, 4],
        image_positions=[7, 7, 20, 20],
        image_embeddings=torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
            ]
        ),
        image_spans=[
            LLaDAOGuiImageSpan(0, 1, 1, 2, 0, 0, (0, 0, 1, 1)),
            LLaDAOGuiImageSpan(2, 3, 3, 4, 0, 0, (1, 0, 2, 1)),
        ],
        source_width=2,
        source_height=1,
        prompt_ids=[],
        prompt_positions=[],
    )
    variants = [
        MaskedQueryVariant((100, 99, 11, 101), (1,), (10,)),
        MaskedQueryVariant((100, 10, 99, 101), (2,), (11,)),
    ]
    requests = [
        MaskedScoringRequest(0, 0, 6),
        MaskedScoringRequest(1, 1, 6),
    ]
    logits, labels, target_ranges = engine._forward_packed_masked_queries(
        prefix,
        requests,
        variants,
        [30, 31, 32, 33],
    )
    assert torch.equal(logits, embedding(torch.tensor([99, 99])))
    assert labels.tolist() == [10, 11]
    assert target_ranges == [(0, 0, 1), (1, 1, 2)]
    assert packed_calls[0][0] == [6, 6]
    assert torch.equal(
        packed_calls[0][1],
        torch.arange(12, dtype=torch.int32),
    )
    assert engine.model.call[0] is None
    assert engine.model.call[1].tolist() == [
        7,
        7,
        30,
        31,
        32,
        33,
        20,
        20,
        30,
        31,
        32,
        33,
    ]


def test_packed_full_prefill_context_uses_varlen_boundaries_without_mask():
    from types import SimpleNamespace

    from d2f_vllm.lladao_gui_engine import LLaDAOGuiD2FEngine
    from d2f_vllm.utils.context import (
        get_context_diffusion_lm,
        reset_context_diffusion_lm,
    )

    engine = object.__new__(LLaDAOGuiD2FEngine)
    engine.config = SimpleNamespace(kv_cache_layout="unified")
    try:
        engine._set_packed_full_prefill_context(
            [3, 5],
            torch.arange(8, dtype=torch.int32),
        )
        context = get_context_diffusion_lm()
        assert context.cu_seqlens_q.tolist() == [0, 3, 8]
        assert context.cu_seqlens_k.tolist() == [0, 3, 8]
        assert context.max_seqlen_q == 5
        assert context.max_seqlen_k == 5
        assert context.seq_lens == [3, 5]
        assert context.seqs is None
        assert context.block_mask is None
        assert context.full_attention is True
        assert context.need_kv_cache_store is False
    finally:
        reset_context_diffusion_lm()


def test_packed_masked_scoring_aggregates_candidate_likelihoods(monkeypatch):
    from types import SimpleNamespace

    import d2f_vllm.lladao_gui_engine as engine_module
    from d2f_vllm.lladao_gui_engine import (
        LLaDAOGuiD2FEngine,
        LLaDAOGuiKVRetrievalConfig,
    )

    engine = object.__new__(LLaDAOGuiD2FEngine)
    engine.kv_retrieval = LLaDAOGuiKVRetrievalConfig(
        mask_rounds=2,
        packed_scoring=True,
        max_batch_tokens=100,
    )
    engine.mask_token_id = 99
    engine.kv_cache_capacity = 100
    prefix = SimpleNamespace(
        image_spans=[
            SimpleNamespace(token_start=0, token_end=2),
            SimpleNamespace(token_start=2, token_end=4),
        ],
        image_embeddings=torch.empty(4, 1),
    )

    def fake_forward(prefix, requests, variants, query_positions):
        del prefix, query_positions
        labels = []
        ranges = []
        offset = 0
        owners = []
        for request in requests:
            targets = variants[request.variant_index].target_ids
            labels.extend(targets)
            ranges.append(
                (
                    request.span_index,
                    offset,
                    offset + len(targets),
                )
            )
            owners.extend([request.span_index] * len(targets))
            offset += len(targets)
        logits = torch.zeros(len(labels), 128)
        for row, (label, owner) in enumerate(zip(labels, owners)):
            logits[row, label] = 5.0 if owner == 0 else -5.0
        return logits, torch.tensor(labels), ranges

    engine._forward_packed_masked_queries = fake_forward
    monkeypatch.setattr(engine_module, "flash_attn_varlen_func", object())
    scores, batches, packed = (
        engine._score_image_spans_masked_self_information(
            prefix,
            [0, 1],
            [100, 10, 11, 12, 101],
            [20, 21, 22, 23, 24],
        )
    )
    assert scores[0] > scores[1]
    assert batches == 1
    assert packed is True


def test_image_kv_retrieval_selects_whole_chunks_and_forced_overview():
    from d2f_vllm.lladao_gui_engine import select_top_image_spans

    selected = select_top_image_spans(
        [0.2, 0.9, 0.9, float("-inf")],
        [0, 1, 2],
        1,
        forced_indices=[3],
    )
    assert selected == [1, 3]


def test_image_kv_retrieval_config_rejects_token_eviction_modes():
    from d2f_vllm.lladao_gui_engine import LLaDAOGuiKVRetrievalConfig

    assert (
        LLaDAOGuiKVRetrievalConfig(
            score_mode="causal_self_information"
        ).score_mode
        == "causal_self_information"
    )
    assert (
        LLaDAOGuiKVRetrievalConfig(
            score_mode="causal_masked_self_information"
        ).score_mode
        == "causal_masked_self_information"
    )
    assert (
        LLaDAOGuiKVRetrievalConfig(
            score_mode="cached_masked_self_information"
        ).score_mode
        == "cached_masked_self_information"
    )
    with pytest.raises(ValueError, match="causal_self_information"):
        LLaDAOGuiKVRetrievalConfig(score_mode="per_head_eviction")
    with pytest.raises(ValueError, match="non-negative"):
        LLaDAOGuiKVRetrievalConfig(topk_images=-1)
    with pytest.raises(ValueError, match="positive"):
        LLaDAOGuiKVRetrievalConfig(mask_rounds=0)
    with pytest.raises(ValueError, match="positive"):
        LLaDAOGuiKVRetrievalConfig(max_batch_tokens=0)


def test_image_kv_retrieval_dispatches_legacy_causal_ablation():
    from types import SimpleNamespace

    from d2f_vllm.lladao_gui_engine import (
        LLaDAOGuiD2FEngine,
        LLaDAOGuiKVRetrievalConfig,
    )

    engine = object.__new__(LLaDAOGuiD2FEngine)
    engine.kv_retrieval = LLaDAOGuiKVRetrievalConfig(
        topk_images=1,
        score_mode="causal_self_information",
        keep_overview=False,
    )
    engine._score_image_span_masked_self_information = (
        lambda *args: pytest.fail("masked scorer was selected")
    )
    engine._score_image_span_causal_self_information = (
        lambda prefix, index, query_ids, query_positions: [0.1, 0.9][index]
    )
    selected, scores, candidates, score_batches, packed = (
        engine._select_retrieved_image_spans(
            SimpleNamespace(image_spans=[object(), object()]),
            has_overview=False,
            query_ids=[100, 10, 101],
            query_positions=[7, 8, 9],
        )
    )
    assert selected == [1]
    assert scores == {0: 0.1, 1: 0.9}
    assert candidates == 2
    assert score_batches == 4
    assert packed is False


def test_causal_masked_scorer_uses_same_position_targets():
    from types import SimpleNamespace

    from d2f_vllm.lladao_gui_engine import (
        LLaDAOGuiD2FEngine,
        LLaDAOGuiKVRetrievalConfig,
    )

    allocated = []
    released = []
    image_calls = []
    query_calls = []
    engine = object.__new__(LLaDAOGuiD2FEngine)
    engine.kv_retrieval = LLaDAOGuiKVRetrievalConfig(
        score_mode="causal_masked_self_information",
        mask_rounds=2,
    )
    engine.mask_token_id = 99
    engine.kv_cache_capacity = 100
    engine.page_size = 8
    engine._prefix_cache = SimpleNamespace(
        allocate_pages=lambda count: allocated.append(count) or [4],
        release_pages=lambda pages: released.append(list(pages)),
    )
    engine._ids_tensor = lambda ids: torch.tensor(ids, dtype=torch.long)

    def fake_image_forward(prefix, pages, indices):
        del prefix
        image_calls.append((list(pages), list(indices)))
        return 2

    def fake_query_forward(
        ids,
        positions,
        *,
        context_len,
        page_ids,
        start_token,
        need_kv_cache_store,
    ):
        query_calls.append(
            (
                tuple(ids),
                tuple(positions),
                context_len,
                list(page_ids),
                start_token,
                need_kv_cache_store,
            )
        )
        logits = torch.zeros(len(ids), 128)
        clean = (100, 10, 11, 12, 101)
        for index, token_id in enumerate(ids):
            if token_id == 99:
                logits[index, clean[index]] = 20.0
        return logits

    engine._forward_image_spans = fake_image_forward
    engine._forward_append_tokens_causal_paged = fake_query_forward
    prefix = SimpleNamespace(
        image_spans=[SimpleNamespace(token_start=0, token_end=2)]
    )
    score = engine._score_image_span_causal_masked_self_information(
        prefix,
        0,
        [100, 10, 11, 12, 101],
        [20, 21, 22, 23, 24],
    )

    assert score > -0.001
    assert allocated == [1]
    assert released == [[4]]
    assert image_calls == [([4], [0])]
    assert [call[0] for call in query_calls] == [
        (100, 99, 11, 99, 101),
        (100, 10, 99, 12, 101),
    ]
    assert all(call[1] == (20, 21, 22, 23, 24) for call in query_calls)
    assert all(call[2:] == (2, [4], 2, False) for call in query_calls)


def test_image_kv_retrieval_dispatches_controlled_causal_masked_ablation():
    from types import SimpleNamespace

    from d2f_vllm.lladao_gui_engine import (
        LLaDAOGuiD2FEngine,
        LLaDAOGuiKVRetrievalConfig,
    )

    engine = object.__new__(LLaDAOGuiD2FEngine)
    engine.mask_token_id = 99
    engine.kv_retrieval = LLaDAOGuiKVRetrievalConfig(
        topk_images=1,
        score_mode="causal_masked_self_information",
        mask_rounds=2,
        keep_overview=False,
    )
    engine._score_image_span_masked_self_information = (
        lambda *args: pytest.fail("bidirectional scorer was selected")
    )
    engine._score_image_span_causal_self_information = (
        lambda *args: pytest.fail("legacy scorer was selected")
    )
    engine._score_image_span_causal_masked_self_information = (
        lambda prefix, index, query_ids, query_positions: [0.1, 0.9][index]
    )
    selected, scores, candidates, score_batches, packed = (
        engine._select_retrieved_image_spans(
            SimpleNamespace(image_spans=[object(), object()]),
            has_overview=False,
            query_ids=[100, 10, 11, 12, 101],
            query_positions=[7, 8, 9, 10, 11],
        )
    )
    assert selected == [1]
    assert scores == {0: 0.1, 1: 0.9}
    assert candidates == 2
    assert score_batches == 6
    assert packed is False


def test_cached_bidirectional_scorer_reuses_one_visual_prefill():
    from types import SimpleNamespace

    from d2f_vllm.lladao_gui_engine import (
        LLaDAOGuiD2FEngine,
        LLaDAOGuiKVRetrievalConfig,
    )

    allocated = []
    released = []
    image_calls = []
    query_calls = []
    engine = object.__new__(LLaDAOGuiD2FEngine)
    engine.kv_retrieval = LLaDAOGuiKVRetrievalConfig(
        score_mode="cached_masked_self_information",
        mask_rounds=2,
    )
    engine.mask_token_id = 99
    engine.kv_cache_capacity = 100
    engine.page_size = 8
    engine._prefix_cache = SimpleNamespace(
        allocate_pages=lambda count: allocated.append(count) or [4],
        release_pages=lambda pages: released.append(list(pages)),
    )
    engine._ids_tensor = lambda ids: torch.tensor(ids, dtype=torch.long)

    def fake_image_forward(prefix, pages, indices):
        del prefix
        image_calls.append((list(pages), list(indices)))
        return 2

    def fake_query_forward(
        ids,
        positions,
        *,
        context_len,
        page_ids,
        start_token,
    ):
        query_calls.append(
            (
                tuple(ids),
                tuple(positions),
                context_len,
                list(page_ids),
                start_token,
            )
        )
        logits = torch.zeros(len(ids), 128)
        clean = (100, 10, 11, 12, 101)
        for index, token_id in enumerate(ids):
            if token_id == 99:
                logits[index, clean[index]] = 20.0
        return logits

    engine._forward_image_spans = fake_image_forward
    engine._forward_append_tokens_bidirectional_paged = fake_query_forward
    engine._forward_append_tokens_causal_paged = (
        lambda *args, **kwargs: pytest.fail("causal query forward was selected")
    )
    prefix = SimpleNamespace(
        image_spans=[SimpleNamespace(token_start=0, token_end=2)]
    )
    score = engine._score_image_span_bidirectional_cached_self_information(
        prefix,
        0,
        [100, 10, 11, 12, 101],
        [20, 21, 22, 23, 24],
    )

    assert score > -0.001
    assert allocated == [1]
    assert released == [[4]]
    assert image_calls == [([4], [0])]
    assert [call[0] for call in query_calls] == [
        (100, 99, 11, 99, 101),
        (100, 10, 99, 12, 101),
    ]
    assert all(call[1] == (20, 21, 22, 23, 24) for call in query_calls)
    assert all(call[2:] == (2, [4], 2) for call in query_calls)


def test_image_kv_retrieval_dispatches_cached_bidirectional_scorer():
    from types import SimpleNamespace

    from d2f_vllm.lladao_gui_engine import (
        LLaDAOGuiD2FEngine,
        LLaDAOGuiKVRetrievalConfig,
    )

    engine = object.__new__(LLaDAOGuiD2FEngine)
    engine.mask_token_id = 99
    engine.kv_retrieval = LLaDAOGuiKVRetrievalConfig(
        topk_images=1,
        score_mode="cached_masked_self_information",
        mask_rounds=2,
        keep_overview=False,
    )
    engine._score_image_span_bidirectional_cached_self_information = (
        lambda prefix, index, query_ids, query_positions: [0.1, 0.9][index]
    )
    engine._score_image_span_causal_masked_self_information = (
        lambda *args: pytest.fail("causal masked scorer was selected")
    )
    engine._score_image_span_causal_self_information = (
        lambda *args: pytest.fail("legacy scorer was selected")
    )
    selected, scores, candidates, score_batches, packed = (
        engine._select_retrieved_image_spans(
            SimpleNamespace(image_spans=[object(), object()]),
            has_overview=False,
            query_ids=[100, 10, 11, 12, 101],
            query_positions=[7, 8, 9, 10, 11],
        )
    )
    assert selected == [1]
    assert scores == {0: 0.1, 1: 0.9}
    assert candidates == 2
    assert score_batches == 6
    assert packed is False


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
