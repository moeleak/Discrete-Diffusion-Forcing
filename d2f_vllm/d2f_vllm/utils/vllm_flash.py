from __future__ import annotations

import logging


_OPTIONAL_ROCM_PREFIXES = (
    "Failed to import from amdsmi",
    "Failed to import from vllm._rocm_C",
)
_previous_log_record_factory = logging.getLogRecordFactory()


def _flash_log_record_factory(*args, **kwargs):
    record = _previous_log_record_factory(*args, **kwargs)
    if (
        record.name == "vllm.platforms.rocm"
        and record.getMessage().startswith(_OPTIONAL_ROCM_PREFIXES)
    ):
        record.levelno = logging.DEBUG
        record.levelname = "DEBUG"
    return record


# This module is imported before attention ops, including when the repository
# root shadows the installed package with a namespace package.
logging.setLogRecordFactory(_flash_log_record_factory)


def _load_flash_attention():
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except ImportError:  # pragma: no cover - exercised by CPU-only unit tests
        return None
    return flash_attn_varlen_func


flash_attn_varlen_func = _load_flash_attention()
