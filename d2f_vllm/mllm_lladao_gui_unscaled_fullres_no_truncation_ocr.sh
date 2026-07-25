#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${REPO:-$ROOT/src/Discrete-Diffusion-Forcing}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$ROOT/data/mind2web-fullpage-16k-64k}"
LIMIT="${LIMIT:-100}"
GPU="${GPU:-0}"
KV_CACHE_CAPACITY="${KV_CACHE_CAPACITY:-65536}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REVISION="${REVISION:-$(git -C "$REPO" rev-parse --short HEAD)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/d2f-vllm-unscaled128k-fullres-no-truncation-ocr-${REVISION}-n${LIMIT}-${RUN_ID}}"
MODEL_OUTPUT="${MODEL_OUTPUT:-$RESULT_ROOT/model}"
FUSED_OUTPUT="${FUSED_OUTPUT:-$RESULT_ROOT/fused}"
MODEL_LOG="${MODEL_LOG:-$ROOT/logs/d2f-vllm-unscaled128k-fullres-no-truncation-model-${REVISION}-n${LIMIT}-${RUN_ID}.log}"
OCR_LOG="${OCR_LOG:-$ROOT/logs/d2f-vllm-unscaled128k-fullres-no-truncation-ocr-${REVISION}-n${LIMIT}-${RUN_ID}.log}"

if ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || (( LIMIT > 100 )); then
  echo "unscaled full-resolution OCR LIMIT must be an integer in [1, 100]" >&2
  exit 2
fi

mkdir -p "$RESULT_ROOT"

MODE=unscaled \
INPUT_MODE=full_page \
FULL_PAGE_POSITION_MODE=strided \
FULL_PAGE_OVERVIEW=0 \
FULL_PAGE_TRUNCATION=0 \
KV_CACHE_COMPRESSION=0 \
MAX_MODEL_LEN=131072 \
KV_CACHE_CAPACITY="$KV_CACHE_CAPACITY" \
LIMIT="$LIMIT" \
GPU="$GPU" \
MASTER_PORT=33243 \
RUN_ID="$RUN_ID" \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
OUTPUT_DIR="$MODEL_OUTPUT" \
LOG="$MODEL_LOG" \
bash "$REPO/d2f_vllm/mllm_lladao_gui_long_context.sh"

# OCR retrieval reuses the completed predictions, so MODE only satisfies the
# retrieval launcher's validation and does not alter model inference.
RUN_MODEL=0 \
MODE=original \
LIMIT="$LIMIT" \
GPU="$GPU" \
RUN_ID="$RUN_ID" \
REVISION="$REVISION" \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
MODEL_OUTPUT="$MODEL_OUTPUT" \
OUTPUT_DIR="$FUSED_OUTPUT" \
RESULT_ROOT="$RESULT_ROOT/ocr-stage" \
LOG="$OCR_LOG" \
MODEL_PROXIMITY_WEIGHT=0.10 \
bash "$REPO/d2f_vllm/mllm_lladao_gui_ocr_retrieval.sh"

echo "[$(date '+%F %T')] UNSCALED_FULLRES_NO_TRUNCATION_OCR_DONE output=$FUSED_OUTPUT"
