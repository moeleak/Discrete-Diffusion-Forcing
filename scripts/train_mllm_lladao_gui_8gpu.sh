#!/usr/bin/env bash
# Launch the LLaDA-o D2F adaptation on eight GPUs without changing the
# effective global batch size of the validated two-GPU recipe.

set -euo pipefail

ROOT="${LLADAO_WORK_ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${D2F_REPO:-${ROOT}/src/Discrete-Diffusion-Forcing}"
LLADAO="${LLADAO_REPO:-${ROOT}/src/LLaDA-o}"
ACCELERATE="${ACCELERATE:-${ROOT}/env/bin/accelerate}"
CONFIG="${CONFIG:-${REPO}/D2F-train/config/lladao_gui_8gpu.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/d2f-block16-r32-8gpu-scheduler-fixed}"
MAX_STEPS="${MAX_STEPS:-1377}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/d2f-8gpu-scheduler-fixed.log}"
PROGRESS_FILE="${PROGRESS_FILE:-${OUTPUT_ROOT}/progress.log}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}/diagnostics"
printf '\n[%s] launcher started; loading model before distributed initialization\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" >>"${PROGRESS_FILE}"

# Writing every tqdm refresh through a tee into the scheduler's logging pipe can
# block training when that downstream pipe applies backpressure.  Keep the
# complete stdout/stderr stream in a regular file; progress has its own file in
# OUTPUT_ROOT so neither path depends on the platform log collector.
echo "[$(date '+%Y-%m-%d %H:%M:%S')] stdout/stderr redirected to ${LOG_FILE}"
exec >>"${LOG_FILE}" 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log_exit() {
  local status=$?
  echo "[$(timestamp)] 8-GPU training launcher exited with status ${status}"
}
trap log_exit EXIT

export PYTHONPATH="${REPO}:${LLADAO}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${ROOT}/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/cache/torch}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-300}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-${OUTPUT_ROOT}/diagnostics/nccl_trace_}"
unset TORCH_NCCL_USE_COMM_NONBLOCKING

echo "[$(timestamp)] starting 8-GPU D2F training"
echo "[$(timestamp)] full stdout/stderr log: ${LOG_FILE}"
echo "[$(timestamp)] host kernel: $(uname -r)"

if [[ ! -x "${ACCELERATE}" ]]; then
  echo "accelerate executable not found: ${ACCELERATE}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "training config not found: ${CONFIG}" >&2
  exit 1
fi
visible_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]')"
if (( visible_gpus < NUM_PROCESSES )); then
  echo "requested ${NUM_PROCESSES} processes but only ${visible_gpus} GPUs are visible" >&2
  exit 1
fi

resume_args=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  resume_args=(--resume-from "${RESUME_FROM}")
elif [[ -d "${OUTPUT_ROOT}" ]]; then
  checkpoint="$(
    find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step-*' \
      -exec test -f '{}/training_state.pt' ';' -print 2>/dev/null \
      | sort | tail -1
  )"
  if [[ -n "${checkpoint}" ]]; then
    resume_args=(--resume-from "${checkpoint}")
  fi
fi

cd "${REPO}"
echo "[$(timestamp)] processes=${NUM_PROCESSES} config=${CONFIG} max_steps=${MAX_STEPS}"
if (( ${#resume_args[@]} )); then
  echo "[$(timestamp)] resuming from ${resume_args[1]}"
else
  echo "[$(timestamp)] starting without an adapter checkpoint"
fi
"${ACCELERATE}" launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  D2F-train/train_lladao_gui.py \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_ROOT}" \
  --max-steps "${MAX_STEPS}" \
  "${resume_args[@]}"
