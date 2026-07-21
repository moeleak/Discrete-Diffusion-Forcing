from __future__ import annotations

import logging


class _OptionalRocmProbeFilter(logging.Filter):
    _PREFIXES = (
        "Failed to import from amdsmi",
        "Failed to import from vllm._rocm_C",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(self._PREFIXES)


def _load_flash_attention():
    """Import vLLM FlashAttention without NVIDIA-irrelevant ROCm probe noise."""
    rocm_logger = logging.getLogger("vllm.platforms.rocm")
    probe_filter = _OptionalRocmProbeFilter()
    rocm_logger.addFilter(probe_filter)
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except ImportError:  # pragma: no cover - exercised by CPU-only unit tests
        return None
    finally:
        rocm_logger.removeFilter(probe_filter)
    return flash_attn_varlen_func


flash_attn_varlen_func = _load_flash_attention()
