import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).parents[1] / "d2f_vllm" / "d2f_vllm"


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_platform_falls_back_to_torch_without_vllm(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.setitem(sys.modules, "vllm.platforms", None)

    module = _load_file(
        "standalone_platform_under_test",
        ROOT / "utils" / "platform.py",
    )
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    assert isinstance(module.current_platform, module.TorchPlatform)
    assert module.current_platform.get_device_capability() == (0, 0)
    assert not module.current_platform.has_device_capability(80)


def test_flash_loader_falls_back_to_native_flash_attn(monkeypatch):
    sentinel = object()
    native_flash = types.ModuleType("flash_attn")
    native_flash.flash_attn_varlen_func = sentinel
    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", None)
    monkeypatch.setitem(sys.modules, "flash_attn", native_flash)

    module = _load_file(
        "standalone_flash_under_test",
        ROOT / "utils" / "vllm_flash.py",
    )

    assert module.flash_attn_varlen_func is sentinel
