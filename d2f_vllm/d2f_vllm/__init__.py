import logging


class _OptionalRocmProbeFilter(logging.Filter):
    _PREFIXES = (
        "Failed to import from amdsmi",
        "Failed to import from vllm._rocm_C",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(self._PREFIXES)


# vLLM imports its ROCm platform module while enumerating plugins, even on an
# NVIDIA-only host. Install this narrow filter before importing any runtime
# modules; all other vLLM, CUDA, and NCCL warnings remain visible.
logging.getLogger("vllm.platforms.rocm").addFilter(_OptionalRocmProbeFilter())

from d2f_vllm.llm import LLM
from d2f_vllm.sampling_params import SamplingParams
from d2f_vllm.fastdllm_engine import FastDLLMDreamEngine, FastDLLMEngineOutput
from d2f_vllm.lladao_gui_engine import LLaDAOGuiD2FEngine, LLaDAOGuiEngineOutput
