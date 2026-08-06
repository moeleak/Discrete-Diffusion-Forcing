#!/usr/bin/env bash
# Direct, no-Slurm eight-GPU GUI-grounding LoRA retraining on a full planner
# checkpoint.  The caller supplies an allocation with exactly eight visible
# GPUs; this script does not set CUDA_VISIBLE_DEVICES or require cd.

set -Eeuo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [[ -h "${SOURCE}" ]]; do
  SOURCE_DIR="$(cd -P "$(dirname "${SOURCE}")" && pwd)"
  SOURCE="$(readlink "${SOURCE}")"
  [[ "${SOURCE}" = /* ]] || SOURCE="${SOURCE_DIR}/${SOURCE}"
done
SCRIPT_DIR="$(cd -P "$(dirname "${SOURCE}")" && pwd)"
REPO="${D2F_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORK_ROOT="${LLADAO_WORK_ROOT:-/home/ma-user/work/LLaDA-o}"

export D2F_REPO="${REPO}"
export LLADAO_REPO="${LLADAO_REPO:-${WORK_ROOT}/src/LLaDA-o}"
export PYTHON="${PYTHON:-${WORK_ROOT}/env/bin/python}"
export ACCELERATE="${ACCELERATE:-${WORK_ROOT}/env/bin/accelerate}"
export CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${REPO}/D2F-train/config/lladao_gui.yaml}"
export LLADAO_FULL_CHECKPOINT="${LLADAO_FULL_CHECKPOINT:-${WORK_ROOT}/models/planner-full-content-v2-131674/model.safetensors}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/runs/d2f-grounding-lora-full-content-v2-8gpu-$(date +%Y%m%d-%H%M%S)-$$}"
export LOG_FILE="${LOG_FILE:-${WORK_ROOT}/logs/d2f-grounding-lora-full-content-v2-8gpu.log}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export GPU_LAUNCHER="${GPU_LAUNCHER:-train_mllm_lladao_gui_8gpu.sh}"

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]')"
if [[ "${gpu_count}" != "8" ]]; then
  echo "full-checkpoint grounding launcher requires exactly 8 visible GPUs; mllm currently exposes ${gpu_count}" >&2
  exit 2
fi

[[ -x "${PYTHON}" ]] || { echo "Python is not executable: ${PYTHON}" >&2; exit 1; }
[[ -x "${ACCELERATE}" ]] || { echo "accelerate is not executable: ${ACCELERATE}" >&2; exit 1; }
[[ -s "${LLADAO_FULL_CHECKPOINT}" ]] || { echo "full checkpoint is missing: ${LLADAO_FULL_CHECKPOINT}" >&2; exit 1; }
[[ -d "${LLADAO_REPO}" ]] || { echo "LLaDA-o repository is missing: ${LLADAO_REPO}" >&2; exit 1; }

echo "full grounding repo=${REPO}"
echo "full grounding checkpoint=${LLADAO_FULL_CHECKPOINT}"
echo "full grounding output=${OUTPUT_ROOT}"
echo "full grounding log=${LOG_FILE}"
exec "${REPO}/scripts/train_mllm_lladao_gui_full_checkpoint_2gpu.sh"
