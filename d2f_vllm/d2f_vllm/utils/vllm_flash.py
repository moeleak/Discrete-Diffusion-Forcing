from __future__ import annotations


def _load_flash_attention():
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except ImportError:  # pragma: no cover - exercised by CPU-only unit tests
        return None
    return flash_attn_varlen_func


flash_attn_varlen_func = _load_flash_attention()
