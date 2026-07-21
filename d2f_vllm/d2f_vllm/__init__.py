import logging


# vLLM imports its ROCm platform module while enumerating plugins, even on an
# NVIDIA-only host. vLLM reconfigures logging while importing plugins, which
# clears logger and handler filters, so narrowly demote only those two records
# at creation time. Other vLLM, CUDA, and NCCL warnings remain unchanged.
_OPTIONAL_ROCM_PREFIXES = (
    "Failed to import from amdsmi",
    "Failed to import from vllm._rocm_C",
)
_previous_log_record_factory = logging.getLogRecordFactory()


def _d2f_log_record_factory(*args, **kwargs):
    record = _previous_log_record_factory(*args, **kwargs)
    if (
        record.name == "vllm.platforms.rocm"
        and record.getMessage().startswith(_OPTIONAL_ROCM_PREFIXES)
    ):
        record.levelno = logging.DEBUG
        record.levelname = "DEBUG"
    return record


logging.setLogRecordFactory(_d2f_log_record_factory)

from d2f_vllm.llm import LLM
from d2f_vllm.sampling_params import SamplingParams
from d2f_vllm.fastdllm_engine import FastDLLMDreamEngine, FastDLLMEngineOutput
from d2f_vllm.lladao_gui_engine import LLaDAOGuiD2FEngine, LLaDAOGuiEngineOutput
