#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${REPO:-$ROOT/src/Discrete-Diffusion-Forcing}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$ROOT/data/mind2web-fullpage-16k-64k}"
LIMIT="${LIMIT:-100}"
GPU="${GPU:-0}"
RUN_MODEL="${RUN_MODEL:-1}"
KV_CACHE_CAPACITY="${KV_CACHE_CAPACITY:-65536}"
KV_RETRIEVAL_TOPK_IMAGES="${KV_RETRIEVAL_TOPK_IMAGES:-4}"
KV_RETRIEVAL_SCORE_MODE="${KV_RETRIEVAL_SCORE_MODE:-masked_self_information}"
KV_RETRIEVAL_MASK_ROUNDS="${KV_RETRIEVAL_MASK_ROUNDS:-2}"
KV_RETRIEVAL_PACKED_SCORING="${KV_RETRIEVAL_PACKED_SCORING:-1}"
KV_RETRIEVAL_MAX_BATCH_TOKENS="${KV_RETRIEVAL_MAX_BATCH_TOKENS:-65536}"
KV_RETRIEVAL_OCR_PRIOR="${KV_RETRIEVAL_OCR_PRIOR:-0}"
MODEL_PROXIMITY_WEIGHT="${MODEL_PROXIMITY_WEIGHT:-0.10}"
RETRIEVAL_PROXIMITY_WEIGHT="${RETRIEVAL_PROXIMITY_WEIGHT:-0.00}"
RETRIEVAL_RANK_WEIGHT="${RETRIEVAL_RANK_WEIGHT:-0.02}"
OCR_DETECTIONS_CACHE="${OCR_DETECTIONS_CACHE:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REVISION="${REVISION:-$(git -C "$REPO" rev-parse --short HEAD)}"
if [[ "$KV_RETRIEVAL_PACKED_SCORING" == "1" ]]; then
  SCORING_BATCH_TAG="packed${KV_RETRIEVAL_MAX_BATCH_TOKENS}"
elif [[ "$KV_RETRIEVAL_PACKED_SCORING" == "0" ]]; then
  SCORING_BATCH_TAG="sequential"
else
  echo "KV_RETRIEVAL_PACKED_SCORING must be 0 or 1" >&2
  exit 2
fi
case "$KV_RETRIEVAL_SCORE_MODE" in
  masked_self_information)
    RETRIEVAL_TAG="kvretrieve${KV_RETRIEVAL_TOPK_IMAGES}-masked${KV_RETRIEVAL_MASK_ROUNDS}-${SCORING_BATCH_TAG}"
    ;;
  cached_masked_self_information)
    RETRIEVAL_TAG="kvretrieve${KV_RETRIEVAL_TOPK_IMAGES}-cachedmasked${KV_RETRIEVAL_MASK_ROUNDS}-${SCORING_BATCH_TAG}"
    ;;
  causal_masked_self_information)
    RETRIEVAL_TAG="kvretrieve${KV_RETRIEVAL_TOPK_IMAGES}-causalmasked${KV_RETRIEVAL_MASK_ROUNDS}"
    ;;
  causal_self_information)
    RETRIEVAL_TAG="kvretrieve${KV_RETRIEVAL_TOPK_IMAGES}-causal"
    ;;
  *)
    echo "KV_RETRIEVAL_SCORE_MODE must be masked_self_information, cached_masked_self_information, causal_masked_self_information, or causal_self_information" >&2
    exit 2
    ;;
esac
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/d2f-vllm-yarn128k-uncropped-${RETRIEVAL_TAG}-ocr-${REVISION}-n${LIMIT}-${RUN_ID}}"
MODEL_OUTPUT="${MODEL_OUTPUT:-$RESULT_ROOT/model}"
FUSED_OUTPUT="${FUSED_OUTPUT:-$RESULT_ROOT/fused}"
MODEL_LOG="${MODEL_LOG:-$ROOT/logs/d2f-vllm-yarn128k-uncropped-${RETRIEVAL_TAG}-model-${REVISION}-n${LIMIT}-${RUN_ID}.log}"
OCR_LOG="${OCR_LOG:-$ROOT/logs/d2f-vllm-yarn128k-uncropped-${RETRIEVAL_TAG}-ocr-${REVISION}-n${LIMIT}-${RUN_ID}.log}"

if ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || (( LIMIT > 100 )); then
  echo "uncropped YaRN KV-retrieval OCR LIMIT must be in [1, 100]" >&2
  exit 2
fi
if ! [[ "$KV_RETRIEVAL_TOPK_IMAGES" =~ ^[0-9]+$ ]]; then
  echo "KV_RETRIEVAL_TOPK_IMAGES must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$KV_RETRIEVAL_MASK_ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "KV_RETRIEVAL_MASK_ROUNDS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$KV_RETRIEVAL_MAX_BATCH_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "KV_RETRIEVAL_MAX_BATCH_TOKENS must be a positive integer" >&2
  exit 2
fi
if [[ "$RUN_MODEL" != "0" && "$RUN_MODEL" != "1" ]]; then
  echo "RUN_MODEL must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$RESULT_ROOT"

if [[ "$RUN_MODEL" == "1" ]]; then
  MODE=yarn \
  INPUT_MODE=full_page \
  FULL_PAGE_POSITION_MODE=strided \
  FULL_PAGE_OVERVIEW=1 \
  FULL_PAGE_TRUNCATION=0 \
  KV_CACHE_COMPRESSION=0 \
  KV_CACHE_RETRIEVAL=1 \
  KV_RETRIEVAL_TOPK_IMAGES="$KV_RETRIEVAL_TOPK_IMAGES" \
  KV_RETRIEVAL_SCORE_MODE="$KV_RETRIEVAL_SCORE_MODE" \
  KV_RETRIEVAL_MASK_ROUNDS="$KV_RETRIEVAL_MASK_ROUNDS" \
  KV_RETRIEVAL_PACKED_SCORING="$KV_RETRIEVAL_PACKED_SCORING" \
  KV_RETRIEVAL_MAX_BATCH_TOKENS="$KV_RETRIEVAL_MAX_BATCH_TOKENS" \
  KV_RETRIEVAL_KEEP_OVERVIEW=1 \
  MAX_MODEL_LEN=131072 \
  KV_CACHE_CAPACITY="$KV_CACHE_CAPACITY" \
  LIMIT="$LIMIT" \
  GPU="$GPU" \
  MASTER_PORT=33943 \
  RUN_ID="$RUN_ID" \
  BENCHMARK_ROOT="$BENCHMARK_ROOT" \
  OUTPUT_DIR="$MODEL_OUTPUT" \
  LOG="$MODEL_LOG" \
  bash "$REPO/d2f_vllm/mllm_lladao_gui_long_context.sh"
elif ! compgen -G "$MODEL_OUTPUT/mind2web_fullpage/*.jsonl" >/dev/null; then
  echo "no reusable model predictions below $MODEL_OUTPUT" >&2
  exit 2
fi

RUN_MODEL=0 \
MODE=yarn \
LIMIT="$LIMIT" \
GPU="$GPU" \
RUN_ID="$RUN_ID" \
REVISION="$REVISION" \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
MODEL_OUTPUT="$MODEL_OUTPUT" \
OUTPUT_DIR="$FUSED_OUTPUT" \
RESULT_ROOT="$RESULT_ROOT/ocr-stage" \
LOG="$OCR_LOG" \
KV_RETRIEVAL_OCR_PRIOR="$KV_RETRIEVAL_OCR_PRIOR" \
MODEL_PROXIMITY_WEIGHT="$MODEL_PROXIMITY_WEIGHT" \
RETRIEVAL_PROXIMITY_WEIGHT="$RETRIEVAL_PROXIMITY_WEIGHT" \
RETRIEVAL_RANK_WEIGHT="$RETRIEVAL_RANK_WEIGHT" \
OCR_DETECTIONS_CACHE="$OCR_DETECTIONS_CACHE" \
bash "$REPO/d2f_vllm/mllm_lladao_gui_ocr_retrieval.sh"

EXPECTED_MODEL_OUTPUT="$RESULT_ROOT/model"
if [[ "$MODEL_OUTPUT" != "$EXPECTED_MODEL_OUTPUT" \
  && ! -e "$EXPECTED_MODEL_OUTPUT" \
  && ! -L "$EXPECTED_MODEL_OUTPUT" ]]; then
  ln -s "$MODEL_OUTPUT" "$EXPECTED_MODEL_OUTPUT"
fi

echo "[$(date '+%F %T')] YARN_UNCROPPED_KV_RETRIEVAL_OCR_DONE output=$FUSED_OUTPUT"
