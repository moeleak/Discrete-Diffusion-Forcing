import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "convert_lladao_gui_d2f_vllm.py"
SPEC = importlib.util.spec_from_file_location("convert_lladao_gui_d2f_vllm", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_runtime_language_key_keeps_only_understanding_path():
    assert (
        MODULE.runtime_language_key(
            "language_model.model.layers.3.self_attn.q_proj.weight"
        )
        == "model.layers.3.self_attn.q_proj.weight"
    )
    assert (
        MODULE.runtime_language_key(
            "language_model.model.layers.3.self_attn.q_norm.weight"
        )
        == "model.layers.3.self_attn.q_norm.weight"
    )
    assert (
        MODULE.runtime_language_key(
            "language_model.model.layers.3.self_attn.q_proj_moe_gen.weight"
        )
        is None
    )
    assert MODULE.runtime_language_key("vit_model.encoder.weight") is None


def test_adapter_key_matches_peft_composite_namespace():
    assert MODULE.adapter_keys_for_source(
        "language_model.model.layers.31.self_attn.o_proj.weight"
    ) == (
        "base_model.model.language_model.model.layers.31.self_attn.o_proj.lora_A.weight",
        "base_model.model.language_model.model.layers.31.self_attn.o_proj.lora_B.weight",
    )
    assert (
        MODULE.adapter_keys_for_source(
            "language_model.model.layers.31.mlp.down_proj.weight"
        )
        is None
    )
