#!/usr/bin/env bash
# Launch the LLaDA-o D2F adaptation on eight GPUs without changing the
# effective global batch size of the validated two-GPU recipe.

set -euo pipefail

ROOT="${LLADAO_WORK_ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${D2F_REPO:-${ROOT}/src/Discrete-Diffusion-Forcing}"
LLADAO="${LLADAO_REPO:-${ROOT}/src/LLaDA-o}"
ACCELERATE="${ACCELERATE:-${ROOT}/env/bin/accelerate}"
CONFIG="${CONFIG:-${REPO}/D2F-train/config/lladao_gui_8gpu.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/d2f-block16-r32-8gpu}"
MAX_STEPS="${MAX_STEPS:-1377}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

export PYTHONPATH="${REPO}:${LLADAO}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${ROOT}/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/cache/torch}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
exec "${ACCELERATE}" launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  D2F-train/train_lladao_gui.py \
  --config "${CONFIG}" \
  --max-steps "${MAX_STEPS}" \
  "${resume_args[@]}"
