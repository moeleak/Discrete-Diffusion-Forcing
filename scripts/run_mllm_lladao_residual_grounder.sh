#!/usr/bin/env bash
# Prepare, train, and benchmark one residual GUI-grounding LoRA on the exact
# final Planner. The caller supplies a passed planner-result.json; there is no
# "latest checkpoint" or old-base fallback.

set -euo pipefail

ROOT="${LLADAO_WORK_ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${D2F_REPO:-${ROOT}/src/Discrete-Diffusion-Forcing}"
LLADAO="${LLADAO_REPO:-${ROOT}/src/LLaDA-o}"
LLADA_AGENT="${LLADA_AGENT_REPO:-${ROOT}/src/LLaDA-Agent}"
PYTHON="${PYTHON:-${ROOT}/env/bin/python}"
ACCELERATE="${ACCELERATE:-${ROOT}/env/bin/accelerate}"
TEMPLATE="${CONFIG_TEMPLATE:-${REPO}/D2F-train/config/lladao_gui_residual.yaml}"
SMOKE_ONLY="${RESIDUAL_SMOKE_ONLY:-0}"
PLANNER_RESULT="${FINAL_PLANNER_RESULT:-}"
MODEL_PATH="${LLADAO_MODEL_PATH:-${ROOT}/models/lladao-gui-mind2web-step750}"
MIND2WEB_TRAIN="${MIND2WEB_TRAIN:-${ROOT}/data/train_ocr/mind2web}"
MIND2WEB_BENCH="${MIND2WEB_BENCH:-${ROOT}/data/bench_ocr}"
MIND2WEB_VALIDATION_BENCH="${MIND2WEB_VALIDATION_BENCH:-${ROOT}/data/bench_ocr_validation}"
MIND2WEB_TEST_NAME="${MIND2WEB_TEST_NAME:-mind2web}"
MIND2WEB_VALIDATION_NAME="${MIND2WEB_VALIDATION_NAME:-mind2web_validation}"
PLANNER_DATA="${PLANNER_DATA_ROOT:-${ROOT}/data/unigui-openmobile-planner-v2-content-v4}"
MOBILE_IMAGES="${MOBILE_IMAGE_ROOT:-${LLADA_AGENT}/data/Uni-GUI-OpenMobile}"
MOBILE_DATA="${MOBILE_GROUNDING_ROOT:-${ROOT}/data/residual-grounding/mobile}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d-%H%M%S')}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/residual-grounder-${RUN_ID}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/residual-grounder.log}"
EPOCHS="${EPOCHS:-3}"

die() {
  echo "residual-grounder launcher error: $*" >&2
  exit 1
}

[[ -x "${PYTHON}" ]] || die "Python is not executable: ${PYTHON}"
[[ -x "${ACCELERATE}" ]] || die "accelerate is not executable: ${ACCELERATE}"
[[ -f "${TEMPLATE}" ]] || die "config template is missing: ${TEMPLATE}"
[[ -d "${MODEL_PATH}" ]] || die "model/tokenizer directory is missing: ${MODEL_PATH}"
[[ -d "${MIND2WEB_TRAIN}" ]] || die "Mind2Web training data is missing: ${MIND2WEB_TRAIN}"
[[ ! -e "${OUTPUT_ROOT}" ]] || die "refusing to overwrite output: ${OUTPUT_ROOT}"

if [[ "${SMOKE_ONLY}" == 1 ]]; then
  PLANNER_CHECKPOINT="${FINAL_PLANNER_CHECKPOINT:?smoke requires FINAL_PLANNER_CHECKPOINT}"
  PLANNER_SHA256="${FINAL_PLANNER_SHA256:?smoke requires FINAL_PLANNER_SHA256}"
  [[ -s "${PLANNER_CHECKPOINT}" ]] || die "smoke Planner checkpoint is missing"
  [[ "${PLANNER_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die "invalid smoke Planner SHA-256"
elif [[ "${SMOKE_ONLY}" == 0 ]]; then
  [[ -f "${PLANNER_RESULT}" ]] || die "set FINAL_PLANNER_RESULT to the passed final planner-result.json"
  [[ -f "${MIND2WEB_BENCH}/manifest.json" ]] || die "Mind2Web OCR test benchmark is missing"
  mapfile -t planner_contract < <(
    "${PYTHON}" - "${PLANNER_RESULT}" <<'PY'
import json
import re
import sys
from pathlib import Path

result_path = Path(sys.argv[1]).expanduser().resolve()
result = json.loads(result_path.read_text(encoding="utf-8"))
if result.get("status") != "passed":
    raise SystemExit(
        "final Planner quality gate has not passed; refusing residual training"
    )
benchmark = result.get("benchmark") or {}
if int(benchmark.get("sample_count", 0)) != 100:
    raise SystemExit("final Planner result is not the fixed 100-sample gate")
artifacts = result.get("artifacts") or {}
checkpoint = Path(str(artifacts.get("checkpoint") or ""))
if not checkpoint.is_absolute():
    checkpoint = (result_path.parent / checkpoint).resolve()
digest = str(artifacts.get("checkpoint_sha256") or "")
if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
    raise SystemExit(f"final Planner checkpoint is missing: {checkpoint}")
if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
    raise SystemExit("final Planner result has no valid checkpoint SHA-256")
print(checkpoint)
print(digest)
PY
  )
  (( ${#planner_contract[@]} == 2 )) || die "could not resolve final Planner contract"
  PLANNER_CHECKPOINT="${planner_contract[0]}"
  PLANNER_SHA256="${planner_contract[1]}"
  [[ -f "${MIND2WEB_VALIDATION_BENCH}/manifest.json" ]] || die \
    "independent Mind2Web validation benchmark is required for checkpoint selection: ${MIND2WEB_VALIDATION_BENCH}"
else
  die "RESIDUAL_SMOKE_ONLY must be 0 or 1"
fi

mkdir -p "${OUTPUT_ROOT}"
echo "residual-grounder stdout/stderr log: ${LOG_FILE}"
printf 'tail -F %q\n' "${LOG_FILE}"
exec >>"${LOG_FILE}" 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
trap 'status=$?; echo "[$(timestamp)] residual-grounder overall exit=${status}"' EXIT

export PYTHONPATH="${REPO}:${LLADAO}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${ROOT}/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/cache/torch}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"

if [[ "${SMOKE_ONLY}" == 1 ]]; then
  echo "[$(timestamp)] SMOKE ONLY: explicit non-release Planner checkpoint=${PLANNER_CHECKPOINT}"
  echo "[$(timestamp)] SMOKE ONLY: no benchmark or release conclusion will be written"
else
  echo "[$(timestamp)] final Planner result=${PLANNER_RESULT}"
fi
echo "[$(timestamp)] Planner checkpoint=${PLANNER_CHECKPOINT}"
echo "[$(timestamp)] Planner sha256=${PLANNER_SHA256}"

if [[ ! -f "${MOBILE_DATA}/manifest.json" ]]; then
  echo "[$(timestamp)] preparing target-bearing mobile grounding data"
  "${PYTHON}" "${REPO}/scripts/prepare_unigui_residual_grounding.py" \
    --prepared-root "${PLANNER_DATA}" \
    --image-root "${MOBILE_IMAGES}" \
    --output-root "${MOBILE_DATA}" \
    --seed 42 \
    --eval-limit 100
fi
[[ -f "${MOBILE_DATA}/manifest.json" ]] || die "mobile grounding manifest is missing"
[[ -f "${MOBILE_DATA}/benchmark/manifest.json" ]] || die "mobile benchmark is missing"
if [[ "${SMOKE_ONLY}" == 0 ]]; then
  "${PYTHON}" - \
    "${MIND2WEB_VALIDATION_BENCH}" "${MIND2WEB_VALIDATION_NAME}" \
    "${MIND2WEB_BENCH}" "${MIND2WEB_TEST_NAME}" \
    "${MOBILE_DATA}/benchmark" <<'PY'
import json
import sys
from pathlib import Path

def selected_ids(root: Path, benchmark: str) -> list[str]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    details = (manifest.get("benchmarks") or {}).get(benchmark)
    if not isinstance(details, dict):
        raise SystemExit(f"benchmark {benchmark!r} is missing below {root}")
    path = root / details["path"]
    rows = [
        json.loads(line)["sample_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < 100:
        raise SystemExit(f"benchmark {benchmark!r} has {len(rows)} rows, needs 100")
    return rows[:100]

validation_root, validation_name = Path(sys.argv[1]), sys.argv[2]
test_root, test_name = Path(sys.argv[3]), sys.argv[4]
mobile_root = Path(sys.argv[5])
validation_ids = selected_ids(validation_root, validation_name)
test_ids = selected_ids(test_root, test_name)
if set(validation_ids) & set(test_ids):
    raise SystemExit("Mind2Web validation-100 overlaps Mind2Web test-100")
selected_ids(mobile_root, "mobile_validation")
selected_ids(mobile_root, "mobile_test")
print("release benchmark preflight passed: independent validation/test sets, 100 rows each")
PY
fi

NUM_PROCESSES="$(${PYTHON} - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
case "${NUM_PROCESSES}" in
  2) GRADIENT_ACCUMULATION_STEPS=8 ;;
  8) GRADIENT_ACCUMULATION_STEPS=2 ;;
  *) die "expected exactly 2 or 8 visible GPUs, found ${NUM_PROCESSES}" ;;
esac

mapfile -t training_budget < <(
  "${PYTHON}" - "${MIND2WEB_TRAIN}" "${MOBILE_DATA}/train" "${EPOCHS}" <<'PY'
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

def rows(root: Path) -> int:
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no Parquet shards below {root}")
    return sum(pq.ParquetFile(path).metadata.num_rows for path in files)

mind2web = rows(Path(sys.argv[1]))
mobile = rows(Path(sys.argv[2]))
epochs = int(sys.argv[3])
if epochs <= 0:
    raise SystemExit("epochs must be positive")
# One optimizer update has eight global microbatches from each domain.
max_steps = epochs * math.ceil(max(mind2web, mobile) / 8)
save_every = math.ceil(max_steps / epochs)
print(mind2web)
print(mobile)
print(max_steps)
print(save_every)
PY
)
(( ${#training_budget[@]} == 4 )) || die "could not resolve training budget"
MIND2WEB_ROWS="${training_budget[0]}"
MOBILE_ROWS="${training_budget[1]}"
MAX_STEPS="${training_budget[2]}"
SAVE_EVERY="${training_budget[3]}"
if [[ "${SMOKE_ONLY}" == 1 ]]; then
  EPOCHS=1
  MAX_STEPS=1
  SAVE_EVERY=1
fi
CONFIG="${OUTPUT_ROOT}/resolved-config.yaml"

"${PYTHON}" - \
  "${TEMPLATE}" "${CONFIG}" "${ROOT}" "${LLADAO}" "${OUTPUT_ROOT}" \
  "${MODEL_PATH}" "${PLANNER_CHECKPOINT}" "${PLANNER_SHA256}" "${MIND2WEB_TRAIN}" \
  "${MOBILE_DATA}/train" "${GRADIENT_ACCUMULATION_STEPS}" \
  "${MAX_STEPS}" "${SAVE_EVERY}" "${EPOCHS}" "${SMOKE_ONLY}" <<'PY'
import sys
from pathlib import Path

import yaml

(
    template, output, root, lladao, output_root, model_path, checkpoint, checkpoint_sha,
    mind2web, mobile, accumulation, max_steps, save_every, epochs, smoke_only,
) = sys.argv[1:]
config = yaml.safe_load(Path(template).read_text(encoding="utf-8"))
config["paths"].update(
    lladao_repo=str(Path(lladao).resolve()),
    model_path=str(Path(model_path).resolve()),
    checkpoint=str(Path(checkpoint).resolve()),
    dataset_config=str((Path(lladao) / "data/configs/gui_grounding_table1.yaml").resolve()),
    output_dir=str(Path(output_root).resolve()),
)
config["model"]["expected_checkpoint_sha256"] = checkpoint_sha
config["data"]["domains"]["mind2web"]["path"] = str(Path(mind2web).resolve())
config["data"]["domains"]["mobile"]["path"] = str(Path(mobile).resolve())
config["train"].update(
    epochs=int(epochs),
    gradient_accumulation_steps=int(accumulation),
    max_steps=int(max_steps),
    save_every=int(save_every),
    release_eligible=smoke_only != "1",
)
Path(output).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

echo "[$(timestamp)] GPUs=${NUM_PROCESSES} global_batch=16 accumulation=${GRADIENT_ACCUMULATION_STEPS}"
echo "[$(timestamp)] rows mind2web=${MIND2WEB_ROWS} mobile=${MOBILE_ROWS} epochs=${EPOCHS} steps=${MAX_STEPS}"

resume_args=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  resume_args=(--resume-from "${RESUME_FROM}")
fi

echo "[$(timestamp)] starting residual grounding training"
"${ACCELERATE}" launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  "${REPO}/D2F-train/train_lladao_gui.py" \
  --config "${CONFIG}" \
  --max-steps "${MAX_STEPS}" \
  "${resume_args[@]}"

FINAL_ADAPTER="${OUTPUT_ROOT}/step-$(printf '%07d' "${MAX_STEPS}")/adapter"
[[ -s "${FINAL_ADAPTER}/adapter_model.safetensors" ]] || die "final adapter is missing"
[[ -s "${FINAL_ADAPTER}/training_contract.json" ]] || die "final adapter contract is missing"
if [[ "${SMOKE_ONLY}" == 1 ]]; then
  echo "[$(timestamp)] SMOKE ONLY completed one optimizer step and adapter save"
  echo "[$(timestamp)] SMOKE ONLY artifact is explicitly not release eligible; no benchmark was run"
  exit 0
fi

run_benchmark() {
  local label="$1"
  local benchmark_root="$2"
  local benchmark="$3"
  local adapter="$4"
  local output="${OUTPUT_ROOT}/benchmark/${label}"
  mkdir -p "${output}"
  echo "[$(timestamp)] benchmark ${label}: ${benchmark} (max 100)"
  "${PYTHON}" "${REPO}/D2F-eval/eval_lladao_gui.py" \
    --backend d2f \
    --lladao-repo "${LLADAO}" \
    --model-path "${MODEL_PATH}" \
    --checkpoint "${PLANNER_CHECKPOINT}" \
    --expected-checkpoint-sha256 "${PLANNER_SHA256}" \
    --adapter "${adapter}" \
    --require-residual-adapter-contract \
    --benchmark-root "${benchmark_root}" \
    --output-dir "${output}" \
    --benchmarks "${benchmark}" \
    --limit 100 \
    --seed 42 \
    --no-kv-cache-compression \
    --no-resume
  "${PYTHON}" "${LLADAO}/eval/gui_grounding/score_benchmark.py" \
    --benchmark-root "${benchmark_root}" \
    --predictions-dir "${output}" \
    --output-dir "${output}/scores" \
    --benchmarks "${benchmark}" \
    --limit 100
}

echo "[$(timestamp)] selecting one epoch on two independent validation-100 domains"
for ((epoch = 1; epoch <= EPOCHS; epoch++)); do
  step=$((epoch * SAVE_EVERY))
  adapter="${OUTPUT_ROOT}/step-$(printf '%07d' "${step}")/adapter"
  [[ -s "${adapter}/adapter_model.safetensors" ]] || die "epoch ${epoch} adapter is missing"
  run_benchmark \
    "validation/epoch-$(printf '%02d' "${epoch}")/mind2web" \
    "${MIND2WEB_VALIDATION_BENCH}" "${MIND2WEB_VALIDATION_NAME}" "${adapter}"
  run_benchmark \
    "validation/epoch-$(printf '%02d' "${epoch}")/mobile" \
    "${MOBILE_DATA}/benchmark" mobile_validation "${adapter}"
done

SELECTION="${OUTPUT_ROOT}/benchmark/checkpoint-selection.json"
SELECTED_ADAPTER="$("${PYTHON}" "${REPO}/D2F-eval/select_residual_grounder.py" \
  --run-root "${OUTPUT_ROOT}" \
  --epochs "${EPOCHS}" \
  --save-every "${SAVE_EVERY}" \
  --mind2web-benchmark "${MIND2WEB_VALIDATION_NAME}" \
  --output "${SELECTION}")"
[[ -s "${SELECTED_ADAPTER}/adapter_model.safetensors" ]] || die "selected adapter is missing"
echo "[$(timestamp)] selected adapter=${SELECTED_ADAPTER}"

# Test data is touched only after validation has selected one checkpoint, and
# each test split is evaluated exactly once.
run_benchmark mind2web-test "${MIND2WEB_BENCH}" "${MIND2WEB_TEST_NAME}" "${SELECTED_ADAPTER}"
run_benchmark mobile-test "${MOBILE_DATA}/benchmark" mobile_test "${SELECTED_ADAPTER}"

INDEX="${OUTPUT_ROOT}/benchmark/index.json"
"${PYTHON}" - "${INDEX}" "${OUTPUT_ROOT}" "${PLANNER_SHA256}" "${MIND2WEB_TEST_NAME}" <<'PY'
import json
import sys
from pathlib import Path

index, root, checkpoint_sha, mind2web_test = (
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
)
runs = []
for label, benchmark in (
    ("mind2web-test", mind2web_test),
    ("mobile-test", "mobile_test"),
):
    run = root / "benchmark" / label
    runs.append(
        {
            "label": label,
            "benchmark": benchmark,
            "run_config": str(run / "run-config-rank-00000.json"),
            "scores": str(run / "scores/results.json"),
        }
    )
index.write_text(
    json.dumps(
        {"schema_version": 1, "backbone_sha256": checkpoint_sha, "runs": runs},
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY
"${PYTHON}" "${REPO}/D2F-eval/summarize_residual_grounder.py" \
  --index "${INDEX}" \
  --output "${OUTPUT_ROOT}/benchmark/results.md"
cat "${OUTPUT_ROOT}/benchmark/results.md"
echo "[$(timestamp)] residual grounding training and fixed held-out benchmarks completed"
