#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${REPO:-$ROOT/src/Discrete-Diffusion-Forcing}"
LLADAO_REPO="${LLADAO_REPO:-$ROOT/src/LLaDA-o}"
D2F_PYTHON="${D2F_PYTHON:-$ROOT/env-d2f-vllm/bin/python}"
OCR_PYTHON="${OCR_PYTHON:-$ROOT/env/bin/python}"
OCR_MODEL_DIR="${OCR_MODEL_DIR:-$ROOT/cache/easyocr}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$ROOT/data/mind2web-fullpage-16k-64k}"
MODE="${MODE:-original}"
LIMIT="${LIMIT:-100}"
GPU="${GPU:-0}"
MASTER_PORT="${MASTER_PORT:-32543}"
RUN_MODEL="${RUN_MODEL:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REVISION="${REVISION:-$(git -C "$REPO" rev-parse --short HEAD)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/d2f-vllm-ocr-retrieval-${MODE}-${REVISION}-n${LIMIT}-${RUN_ID}}"
MODEL_OUTPUT="${MODEL_OUTPUT:-$RESULT_ROOT/model}"
OUTPUT_DIR="${OUTPUT_DIR:-$RESULT_ROOT/fused}"
LOG="${LOG:-$ROOT/logs/d2f-vllm-ocr-retrieval-${MODE}-${REVISION}-n${LIMIT}-${RUN_ID}.log}"
MODEL_PROXIMITY_WEIGHT="${MODEL_PROXIMITY_WEIGHT:-0.10}"
KV_RETRIEVAL_OCR_PRIOR="${KV_RETRIEVAL_OCR_PRIOR:-0}"
RETRIEVAL_PROXIMITY_WEIGHT="${RETRIEVAL_PROXIMITY_WEIGHT:-0.00}"
RETRIEVAL_RANK_WEIGHT="${RETRIEVAL_RANK_WEIGHT:-0.02}"
OCR_DETECTIONS_CACHE="${OCR_DETECTIONS_CACHE:-}"
LABEL_CONTROL_OFFSET="${LABEL_CONTROL_OFFSET:-40}"

if ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || (( LIMIT > 100 )); then
  echo "OCR retrieval LIMIT must be an integer in [1, 100]" >&2
  exit 2
fi
if [[ "$MODE" != "original" && "$MODE" != "yarn" ]]; then
  echo "MODE must be original or yarn" >&2
  exit 2
fi
if [[ "$RUN_MODEL" != "0" && "$RUN_MODEL" != "1" ]]; then
  echo "RUN_MODEL must be 0 or 1" >&2
  exit 2
fi
if [[ "$KV_RETRIEVAL_OCR_PRIOR" != "0" \
  && "$KV_RETRIEVAL_OCR_PRIOR" != "1" ]]; then
  echo "KV_RETRIEVAL_OCR_PRIOR must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG")" "$RESULT_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] starting native-resize D2F + full-page OCR retrieval"
echo "[$(date '+%F %T')] mode=$MODE revision=$REVISION samples=$LIMIT gpu=$GPU"
echo "[$(date '+%F %T')] model_output=$MODEL_OUTPUT fused_output=$OUTPUT_DIR"
echo "[$(date '+%F %T')] kv_retrieval_ocr_prior=$KV_RETRIEVAL_OCR_PRIOR"
echo "[$(date '+%F %T')] model_weight=$MODEL_PROXIMITY_WEIGHT retrieval_proximity_weight=$RETRIEVAL_PROXIMITY_WEIGHT retrieval_rank_weight=$RETRIEVAL_RANK_WEIGHT"
if [[ -n "$OCR_DETECTIONS_CACHE" ]]; then
  echo "[$(date '+%F %T')] detections_cache=$OCR_DETECTIONS_CACHE"
fi

if [[ "$RUN_MODEL" == "1" ]]; then
  MODE="$MODE" \
  INPUT_MODE=native_resize \
  LIMIT="$LIMIT" \
  GPU="$GPU" \
  MASTER_PORT="$MASTER_PORT" \
  RUN_ID="$RUN_ID" \
  MAX_MODEL_LEN=131072 \
  KV_CACHE_CAPACITY=16384 \
  KV_CACHE_COMPRESSION=0 \
  OUTPUT_DIR="$MODEL_OUTPUT" \
  BENCHMARK_ROOT="$BENCHMARK_ROOT" \
  PYTHON="$D2F_PYTHON" \
  bash "$REPO/d2f_vllm/mllm_lladao_gui_long_context.sh"
else
  if ! compgen -G "$MODEL_OUTPUT/mind2web_fullpage/*.jsonl" >/dev/null; then
    echo "no reusable model predictions below $MODEL_OUTPUT" >&2
    exit 2
  fi
  echo "[$(date '+%F %T')] reusing model predictions"
fi

if [[ "$KV_RETRIEVAL_OCR_PRIOR" == "1" ]]; then
  OCR_ARGS=(
    --benchmark-root "$BENCHMARK_ROOT"
    --predictions-dir "$MODEL_OUTPUT"
    --output-dir "$OUTPUT_DIR"
    --benchmark mind2web_fullpage
    --limit "$LIMIT"
    --model-dir "$OCR_MODEL_DIR"
    --model-proximity-weight "$MODEL_PROXIMITY_WEIGHT"
    --retrieval-proximity-weight "$RETRIEVAL_PROXIMITY_WEIGHT"
    --retrieval-rank-weight "$RETRIEVAL_RANK_WEIGHT"
    --label-control-offset "$LABEL_CONTROL_OFFSET"
    --gpu
  )
  if [[ -n "$OCR_DETECTIONS_CACHE" ]]; then
    OCR_ARGS+=(--detections-cache "$OCR_DETECTIONS_CACHE")
  else
    OCR_ARGS+=(
      --write-detections-cache "$OUTPUT_DIR/ocr-detections.jsonl"
    )
  fi
  (
    cd "$LLADAO_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
      PYTHONPATH="$LLADAO_REPO${PYTHONPATH:+:$PYTHONPATH}" \
      "$OCR_PYTHON" "$REPO/D2F-eval/ocr_retrieval_kv_prior.py" \
        "${OCR_ARGS[@]}"
  )
else
  (
    cd "$LLADAO_REPO"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
      "$OCR_PYTHON" -m eval.gui_grounding.ocr_fullpage_retrieval \
      --benchmark-root "$BENCHMARK_ROOT" \
      --predictions-dir "$MODEL_OUTPUT" \
      --output-dir "$OUTPUT_DIR" \
      --benchmark mind2web_fullpage \
      --limit "$LIMIT" \
      --model-dir "$OCR_MODEL_DIR" \
      --model-proximity-weight "$MODEL_PROXIMITY_WEIGHT" \
      --label-control-offset "$LABEL_CONTROL_OFFSET" \
      --gpu
  )
fi

while IFS= read -r config; do
  cp "$config" "$OUTPUT_DIR/$(basename "$config")"
done < <(find "$MODEL_OUTPUT" -maxdepth 1 -name 'run-config-rank-*.json' -type f | sort)

(
  cd "$LLADAO_REPO"
  "$OCR_PYTHON" -m eval.gui_grounding.score_benchmark \
    --benchmark-root "$BENCHMARK_ROOT" \
    --predictions-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR/scores" \
    --benchmarks mind2web_fullpage \
    --limit "$LIMIT"
)

echo "[$(date '+%F %T')] OCR_RETRIEVAL_DONE output=$OUTPUT_DIR"
