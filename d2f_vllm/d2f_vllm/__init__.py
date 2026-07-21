import logging


class _OptionalRocmProbeFilter(logging.Filter):
    _PREFIXES = (
        "Failed to import from amdsmi",
        "Failed to import from vllm._rocm_C",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(self._PREFIXES)


# vLLM imports its ROCm platform module while enumerating plugins, even on an
# NVIDIA-only host. Let vLLM configure logging first because dictConfig clears
# filters installed on pre-existing child loggers, then install this narrow
# filter before importing any runtime modules. All other warnings remain.
try:
    import vllm.logger  # noqa: F401
except ImportError:
    pass
_rocm_probe_filter = _OptionalRocmProbeFilter()
logging.getLogger("vllm.platforms.rocm").addFilter(_rocm_probe_filter)
for _handler in logging.getLogger("vllm").handlers:
    _handler.addFilter(_rocm_probe_filter)

from d2f_vllm.llm import LLM
from d2f_vllm.sampling_params import SamplingParams
from d2f_vllm.fastdllm_engine import FastDLLMDreamEngine, FastDLLMEngineOutput
from d2f_vllm.lladao_gui_engine import LLaDAOGuiD2FEngine, LLaDAOGuiEngineOutput
