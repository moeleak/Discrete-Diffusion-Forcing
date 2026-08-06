#!/usr/bin/env bash
#SBATCH --account=guests
#SBATCH --qos=guest-dev
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-llada-grounding-%j.log
#SBATCH --error=slurm-llada-grounding-%j.log

# Full-checkpoint GUI-grounding LoRA retraining on one eight-GPU node.  The
# script is self-contained when submitted from a versioned repository archive;
# every machine-specific value remains overridable through the environment.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${D2F_REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORK_ROOT="${LLADAO_WORK_ROOT:-/home/guests/zhen/lladao-work}"
LLADAO_REPO="${LLADAO_REPO:-${WORK_ROOT}/LLaDA-o}"
PYTHON="${PYTHON:-/home/guests/zhen/jing/Discrete-Diffusion-Forcing/.venv/bin/python}"
ACCELERATE="${ACCELERATE:-/home/guests/zhen/jing/Discrete-Diffusion-Forcing/.venv/bin/accelerate}"
FULL_CHECKPOINT="${LLADAO_FULL_CHECKPOINT:-${WORK_ROOT}/runs/planner-full-content-v2-slurm-131674/training/checkpoints/epoch-02/model.safetensors}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/runs/d2f-grounding-lora-full-content-v2-8gpu-${SLURM_JOB_ID}}"
LOG_FILE="${LOG_FILE:-${WORK_ROOT}/logs/d2f-grounding-lora-full-content-v2-8gpu-${SLURM_JOB_ID}.log}"

export LLADAO_WORK_ROOT="${WORK_ROOT}"
export D2F_REPO="${REPO}"
export LLADAO_REPO
export PYTHON
export ACCELERATE
export LLADAO_FULL_CHECKPOINT="${FULL_CHECKPOINT}"
export OUTPUT_ROOT
export LOG_FILE
export NUM_PROCESSES=8
export GRADIENT_ACCUMULATION_STEPS=2
export GPU_LAUNCHER=train_mlladao_gui_8gpu.sh

mkdir -p "$(dirname "${LOG_FILE}")"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] full-checkpoint 8-GPU grounding job ${SLURM_JOB_ID}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] repo=${REPO} checkpoint=${FULL_CHECKPOINT}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] output=${OUTPUT_ROOT} log=${LOG_FILE}"

[[ -x "${PYTHON}" ]] || { echo "Python is not executable: ${PYTHON}" >&2; exit 1; }
[[ -x "${ACCELERATE}" ]] || { echo "accelerate is not executable: ${ACCELERATE}" >&2; exit 1; }
[[ -s "${FULL_CHECKPOINT}" ]] || { echo "full checkpoint is missing: ${FULL_CHECKPOINT}" >&2; exit 1; }
[[ -d "${LLADAO_REPO}" ]] || { echo "LLaDA-o repository is missing: ${LLADAO_REPO}" >&2; exit 1; }

exec srun --wait=60 --nodes=1 --ntasks=1 --gres=gpu:8 --exclusive \
  bash "${REPO}/scripts/train_mllm_lladao_gui_full_checkpoint_2gpu.sh"
