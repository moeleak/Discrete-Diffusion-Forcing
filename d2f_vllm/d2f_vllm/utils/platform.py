from __future__ import annotations

import torch


class TorchPlatform:
    """Small vLLM-platform compatibility layer for standalone D2F runtimes."""

    @staticmethod
    def is_rocm() -> bool:
        return bool(getattr(torch.version, "hip", None))

    @staticmethod
    def get_device_capability() -> tuple[int, int]:
        if not torch.cuda.is_available():
            return (0, 0)
        try:
            major, minor = torch.cuda.get_device_capability()
        except (AssertionError, RuntimeError):
            return (0, 0)
        return int(major), int(minor)

    @classmethod
    def has_device_capability(
        cls,
        capability: int | tuple[int, int],
    ) -> bool:
        required = (
            capability
            if isinstance(capability, int)
            else int(capability[0]) * 10 + int(capability[1])
        )
        current = cls.get_device_capability()
        return current[0] * 10 + current[1] >= required

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        if cls.is_rocm() and hasattr(torch, "float8_e4m3fnuz"):
            return torch.float8_e4m3fnuz
        if hasattr(torch, "float8_e4m3fn"):
            return torch.float8_e4m3fn
        raise RuntimeError("this PyTorch build does not expose an FP8 dtype")


def _load_current_platform():
    try:
        from vllm.platforms import current_platform as vllm_platform
    except ImportError:
        return TorchPlatform()
    return vllm_platform


current_platform = _load_current_platform()
