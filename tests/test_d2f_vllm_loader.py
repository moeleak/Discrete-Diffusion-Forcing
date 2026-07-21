import importlib.util
import json
from pathlib import Path


LOADER = (
    Path(__file__).parents[1]
    / "d2f_vllm"
    / "d2f_vllm"
    / "utils"
    / "loader.py"
)


def test_language_index_excludes_vision_sidecar(tmp_path, monkeypatch):
    # Import with lightweight dependency stubs so this path-resolution test can
    # run even on a development machine without the CUDA runtime installed.
    import sys
    import types

    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.Module = object
    torch.nn.Parameter = object
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.nn", torch.nn)
    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda value, **_: value
    sys.modules.setdefault("tqdm", tqdm)
    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = object
    sys.modules.setdefault("safetensors", safetensors)
    config_module = types.ModuleType("d2f_vllm.config")
    config_module.Config = object
    monkeypatch.setitem(sys.modules, "d2f_vllm.config", config_module)

    (tmp_path / "vision.safetensors").touch()
    (tmp_path / "model-00001-of-00002.safetensors").touch()
    (tmp_path / "model-00002-of-00002.safetensors").touch()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    spec = importlib.util.spec_from_file_location("loader_under_test", LOADER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module._base_weight_files(str(tmp_path)) == [
        str(tmp_path / "model-00001-of-00002.safetensors"),
        str(tmp_path / "model-00002-of-00002.safetensors"),
    ]
