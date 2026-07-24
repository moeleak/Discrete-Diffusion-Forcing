#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${REPO:-$ROOT/src/Discrete-Diffusion-Forcing}"
LLADAO_REPO="${LLADAO_REPO:-$ROOT/src/LLaDA-o}"
D2F_PYTHON="${D2F_PYTHON:-$ROOT/env-d2f-vllm/bin/python}"
OCR_PYTHON="${OCR_PYTHON:-$ROOT/env/bin/python}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$ROOT/data/mind2web-fullpage-16k-64k}"
LIMIT="${LIMIT:-100}"
GPU="${GPU:-0}"
RUN_MODEL="${RUN_MODEL:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REVISION="${REVISION:-$(git -C "$REPO" rev-parse --short HEAD)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/d2f-vllm-fullpage-ocr-crop-${REVISION}-n${LIMIT}-${RUN_ID}}"
MODEL_OUTPUT="${MODEL_OUTPUT:-$RESULT_ROOT/native-model}"
OCR_OUTPUT="${OCR_OUTPUT:-$RESULT_ROOT/ocr-retrieval}"
CROP_BENCHMARK_ROOT="${CROP_BENCHMARK_ROOT:-$RESULT_ROOT/retrieval-crop-benchmark}"
CROP_MODEL_OUTPUT="${CROP_MODEL_OUTPUT:-$RESULT_ROOT/retrieval-crop-model}"
FUSED_OUTPUT="${FUSED_OUTPUT:-$RESULT_ROOT/fused}"
LOG="${LOG:-$ROOT/logs/d2f-vllm-fullpage-ocr-crop-${REVISION}-n${LIMIT}-${RUN_ID}.log}"

if ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || (( LIMIT > 100 )); then
  echo "OCR crop pipeline LIMIT must be an integer in [1, 100]" >&2
  exit 2
fi
if [[ "$RUN_MODEL" != "0" && "$RUN_MODEL" != "1" ]]; then
  echo "RUN_MODEL must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG")" "$RESULT_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] starting D2F full-page OCR crop pipeline"
echo "[$(date '+%F %T')] revision=$REVISION samples=$LIMIT gpu=$GPU"
echo "[$(date '+%F %T')] result_root=$RESULT_ROOT"

RUN_MODEL="$RUN_MODEL" \
MODE=original \
LIMIT="$LIMIT" \
GPU="$GPU" \
MASTER_PORT=32743 \
RUN_ID="$RUN_ID" \
REVISION="$REVISION" \
MODEL_OUTPUT="$MODEL_OUTPUT" \
OUTPUT_DIR="$OCR_OUTPUT" \
RESULT_ROOT="$RESULT_ROOT/native-ocr-stage" \
LOG="$ROOT/logs/d2f-vllm-fullpage-ocr-stage-${REVISION}-n${LIMIT}-${RUN_ID}.log" \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
D2F_PYTHON="$D2F_PYTHON" \
OCR_PYTHON="$OCR_PYTHON" \
bash "$REPO/d2f_vllm/mllm_lladao_gui_ocr_retrieval.sh"

(
  cd "$LLADAO_REPO"
  "$OCR_PYTHON" -m eval.gui_grounding.prepare_ocr_retrieval_crops \
    --benchmark-root "$BENCHMARK_ROOT" \
    --retrieval-dir "$OCR_OUTPUT" \
    --output-root "$CROP_BENCHMARK_ROOT" \
    --limit "$LIMIT"
)

MODE=original \
INPUT_MODE=native_resize \
LIMIT="$LIMIT" \
GPU="$GPU" \
MASTER_PORT=32753 \
RUN_ID="$RUN_ID" \
KV_CACHE_COMPRESSION=0 \
BENCHMARK_ROOT="$CROP_BENCHMARK_ROOT" \
OUTPUT_DIR="$CROP_MODEL_OUTPUT" \
LOG="$ROOT/logs/d2f-vllm-retrieval-crop-model-${REVISION}-n${LIMIT}-${RUN_ID}.log" \
PYTHON="$D2F_PYTHON" \
bash "$REPO/d2f_vllm/mllm_lladao_gui_long_context.sh"

(
  cd "$LLADAO_REPO"
  "$OCR_PYTHON" -m eval.gui_grounding.fuse_ocr_crop_predictions \
    --benchmark-root "$BENCHMARK_ROOT" \
    --ocr-predictions-dir "$OCR_OUTPUT" \
    --crop-benchmark-root "$CROP_BENCHMARK_ROOT" \
    --crop-predictions-dir "$CROP_MODEL_OUTPUT" \
    --output-dir "$FUSED_OUTPUT" \
    --limit "$LIMIT"
  "$OCR_PYTHON" -m eval.gui_grounding.score_benchmark \
    --benchmark-root "$BENCHMARK_ROOT" \
    --predictions-dir "$FUSED_OUTPUT" \
    --output-dir "$FUSED_OUTPUT/scores" \
    --benchmarks mind2web_fullpage \
    --limit "$LIMIT"
)

echo "[$(date '+%F %T')] OCR_CROP_PIPELINE_DONE output=$FUSED_OUTPUT"
