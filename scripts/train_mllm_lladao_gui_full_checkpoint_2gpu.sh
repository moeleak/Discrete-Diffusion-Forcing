#!/usr/bin/env bash
# Retrain the GUI-grounding LoRA on top of a standalone full-parameter
# planner checkpoint.  This launcher is intentionally no-Slurm and keeps the
# validated two-GPU/global-batch-16 recipe.

set -euo pipefail

ROOT="${LLADAO_WORK_ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${D2F_REPO:-${ROOT}/src/Discrete-Diffusion-Forcing}"
LLADAO="${LLADAO_REPO:-${ROOT}/src/LLaDA-o}"
PYTHON="${PYTHON:-${ROOT}/env/bin/python}"
ACCELERATE="${ACCELERATE:-${ROOT}/env/bin/accelerate}"
TEMPLATE="${CONFIG_TEMPLATE:-${REPO}/D2F-train/config/lladao_gui.yaml}"
FULL_CHECKPOINT="${LLADAO_FULL_CHECKPOINT:?set LLADAO_FULL_CHECKPOINT to the Full-parameter content-v2 model.safetensors}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
GPU_LAUNCHER="${GPU_LAUNCHER:-train_mllm_lladao_gui_2gpu.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/d2f-grounding-lora-full-content-v2-${NUM_PROCESSES}gpu}"
LOG_FILE="${LOG_FILE:-${ROOT}/logs/d2f-grounding-lora-full-content-v2-${NUM_PROCESSES}gpu.log}"
CONFIG="${OUTPUT_ROOT}/resolved-full-content-v2.yaml"

die() {
  echo "full-checkpoint grounding launcher error: $*" >&2
  exit 1
}

[[ -x "${PYTHON}" ]] || die "Python is not executable: ${PYTHON}"
[[ -x "${ACCELERATE}" ]] || die "accelerate is not executable: ${ACCELERATE}"
[[ -f "${TEMPLATE}" ]] || die "config template is missing: ${TEMPLATE}"
[[ -s "${FULL_CHECKPOINT}" ]] || die "full checkpoint is missing: ${FULL_CHECKPOINT}"
[[ ! -e "${OUTPUT_ROOT}" ]] || die "refusing to overwrite existing output: ${OUTPUT_ROOT}"

mkdir -p "${OUTPUT_ROOT}" "$(dirname "${LOG_FILE}")"

# Resolve all machine-specific paths into an auditable YAML.  The trainer then
# uses its normal strict checkpoint loader, which accepts the understanding-
# only full-parameter export and strips no additional trainable parameters.
"${PYTHON}" - "${TEMPLATE}" "${CONFIG}" "${FULL_CHECKPOINT}" "${LLADAO}" "${ROOT}" "${GRADIENT_ACCUMULATION_STEPS}" <<'PY'
import sys
from pathlib import Path

import yaml

template, output, checkpoint, lladao, root = map(Path, sys.argv[1:6])
gradient_accumulation_steps = int(sys.argv[6])
config = yaml.safe_load(template.read_text(encoding="utf-8"))
paths = config["paths"]
paths["lladao_repo"] = str(lladao.resolve())
paths["model_path"] = str((root / "models/lladao-gui-mind2web-step750").resolve())
paths["checkpoint"] = str(Path(checkpoint).resolve())
paths["train_data"] = str((root / "data/train_ocr/mind2web").resolve())
paths["dataset_config"] = str((lladao / "data/configs/gui_grounding_table1.yaml").resolve())
paths["output_dir"] = str(output.parent.resolve())
config["train"]["gradient_accumulation_steps"] = gradient_accumulation_steps
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

export D2F_REPO="${REPO}"
export LLADAO_REPO="${LLADAO}"
export ACCELERATE
export CONFIG
export OUTPUT_ROOT
export LOG_FILE
export NUM_PROCESSES
export GRADIENT_ACCUMULATION_STEPS
export PYTHONPATH="${REPO}:${LLADAO}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

echo "full grounding LoRA stdout/stderr log: ${LOG_FILE}"
exec "${REPO}/scripts/${GPU_LAUNCHER}"
