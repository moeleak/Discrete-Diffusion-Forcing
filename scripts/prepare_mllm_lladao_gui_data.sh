#!/usr/bin/env bash
# Rebuild the paired Mind2Web training and evaluation data on mllm.
#
# The script intentionally uses CPU OCR workers so data preparation can proceed
# while the server GPUs are occupied. All paths remain below LLADAO_WORK_ROOT.

set -euo pipefail

ROOT="${LLADAO_WORK_ROOT:-/home/ma-user/work/LLaDA-o}"
PYTHON="${PYTHON:-${ROOT}/env/bin/python}"
LLADAO_REPO="${LLADAO_REPO:-${ROOT}/src/LLaDA-o}"
OCR_WORLD_SIZE="${OCR_WORLD_SIZE:-8}"
TRAIN_RAW="${ROOT}/data/train_raw"
TRAIN_WORK="${ROOT}/data/train_ocr_work"
TRAIN_OCR="${ROOT}/data/train_ocr"
BENCH_RAW="${ROOT}/data/bench_raw"
BENCH_WORK="${ROOT}/data/bench_ocr_work"
BENCH_OCR="${ROOT}/data/bench_ocr"
OCR_MODELS="${ROOT}/cache/easyocr"
LOG_DIR="${ROOT}/logs/data"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${ROOT}/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export EASYOCR_MODULE_PATH="${OCR_MODELS}"

mkdir -p "${LOG_DIR}" "${OCR_MODELS}"
cd "${LLADAO_REPO}"

if [[ -z "$(find "${TRAIN_RAW}/raw/mind2web" -name 'train-*.parquet' -print -quit 2>/dev/null)" ]]; then
  "${PYTHON}" scripts/data/prepare_gui_grounding.py download \
    --root "${TRAIN_RAW}" --sources mind2web --count 20000 --seed 42
fi
if [[ ! -f "${TRAIN_RAW}/parquet/manifest.json" ]]; then
  "${PYTHON}" scripts/data/prepare_gui_grounding.py build \
    --root "${TRAIN_RAW}" --sources mind2web --count 20000 --seed 42 \
    --mind2web-count 7341 --mind2web-crop-size 1280 \
    --mind2web-crop-mode random --mind2web-prompt-protocol target_grounding
fi

if [[ ! -f "${BENCH_RAW}/manifest.json" ]]; then
  "${PYTHON}" scripts/data/prepare_gui_grounding_benchmarks.py all \
    --root "${BENCH_RAW}" --mind2web-crop-size 1280
fi

# Download EasyOCR's English detector/recognizer once before parallel readers
# start, avoiding concurrent writes to the shared model directory.
"${PYTHON}" -c \
  'import easyocr, os; easyocr.Reader(["en"], gpu=False, model_storage_directory=os.environ["EASYOCR_MODULE_PATH"])'

pids=()
for ((rank = 0; rank < OCR_WORLD_SIZE; rank++)); do
  "${PYTHON}" scripts/data/realign_gui_grounding_training_ocr.py rewrite \
    --input-root "${TRAIN_RAW}/parquet" --work-dir "${TRAIN_WORK}" \
    --rank "${rank}" --world-size "${OCR_WORLD_SIZE}" \
    --model-dir "${OCR_MODELS}" --no-gpu --no-download \
    >"${LOG_DIR}/train-ocr-rank-${rank}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done
"${PYTHON}" scripts/data/realign_gui_grounding_training_ocr.py finalize \
  --input-root "${TRAIN_RAW}/parquet" --work-dir "${TRAIN_WORK}" \
  --output-root "${TRAIN_OCR}" --force

pids=()
for ((rank = 0; rank < OCR_WORLD_SIZE; rank++)); do
  "${PYTHON}" scripts/data/realign_gui_grounding_ocr.py detect \
    --benchmark-root "${BENCH_RAW}" --work-dir "${BENCH_WORK}" \
    --benchmark mind2web --rank "${rank}" --world-size "${OCR_WORLD_SIZE}" \
    --model-dir "${OCR_MODELS}" --no-gpu --no-download \
    >"${LOG_DIR}/bench-ocr-rank-${rank}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done
"${PYTHON}" scripts/data/realign_gui_grounding_ocr.py finalize \
  --benchmark-root "${BENCH_RAW}" --work-dir "${BENCH_WORK}" \
  --output-root "${BENCH_OCR}" --benchmark mind2web --force

"${PYTHON}" scripts/data/prepare_gui_grounding_benchmarks.py validate \
  --root "${BENCH_OCR}"

expected_mind2web_sha="011389659326fd7a08c5972cc872bf7573b130f46b0e6aedf2b350da377e87cc"
actual_mind2web_sha="$(sha256sum "${BENCH_OCR}/samples/mind2web.jsonl" | awk '{print $1}')"
if [[ "${actual_mind2web_sha}" == "${expected_mind2web_sha}" ]]; then
  echo "Mind2Web OCR benchmark matches the source artifact: ${actual_mind2web_sha}"
else
  echo "Mind2Web OCR checksum differs from the source artifact." >&2
  echo "expected=${expected_mind2web_sha}" >&2
  echo "actual=${actual_mind2web_sha}" >&2
  echo "Use a paired baseline rebuilt on this same benchmark root." >&2
fi
