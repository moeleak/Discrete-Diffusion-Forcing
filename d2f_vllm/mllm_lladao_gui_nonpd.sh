#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${REPO:-$ROOT/src/Discrete-Diffusion-Forcing}"
PYTHON="${PYTHON:-$ROOT/env-d2f-vllm/bin/python}"
RUNTIME_MODEL="${RUNTIME_MODEL:-$ROOT/models/lladao-gui-d2f-vllm-step1377-exact}"
SOURCE_MODEL="${SOURCE_MODEL:-$ROOT/models/lladao-gui-mind2web-step750}"
LLADAO_REPO="${LLADAO_REPO:-$ROOT/src/LLaDA-o}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$ROOT/data/bench_ocr}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/d2f-vllm-nonpd-100}"
LIMIT="${LIMIT:-100}"
GPU="${GPU:-0}"
MASTER_PORT="${MASTER_PORT:-32333}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG="${LOG:-$ROOT/logs/d2f-vllm-nonpd-${RUN_ID}.log}"
KV_CACHE_COMPRESSION="${KV_CACHE_COMPRESSION:-1}"
VISION_TILE_SIZE="${VISION_TILE_SIZE:-16}"
VISION_TOPK_TILES="${VISION_TOPK_TILES:-0}"
VISION_TOKEN_KEEP_RATIO="${VISION_TOKEN_KEEP_RATIO:-0.75}"
VISION_SCORE_QUERY_WINDOW="${VISION_SCORE_QUERY_WINDOW:-32}"
VISION_SCORE_LAYERS="${VISION_SCORE_LAYERS:-0}"
VISION_SCORE_LAYER_MODE="${VISION_SCORE_LAYER_MODE:-all}"
VISION_SCORE_POOL_KERNEL="${VISION_SCORE_POOL_KERNEL:-7}"

mkdir -p "$(dirname "$LOG")" "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export D2F_VLLM_ATTENTION_BACKEND="${D2F_VLLM_ATTENTION_BACKEND:-sdpa}"
export D2F_VLLM_RMS_NORM_BACKEND="${D2F_VLLM_RMS_NORM_BACKEND:-vllm}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [[ "$KV_CACHE_COMPRESSION" == "1" ]]; then
  COMPRESSION_FLAG="--kv-cache-compression"
else
  COMPRESSION_FLAG="--no-kv-cache-compression"
fi

echo "[$(date '+%F %T')] LLaDA-o GUI Non-PD: gpu=$GPU limit=$LIMIT model=$RUNTIME_MODEL" | tee -a "$LOG"
echo "[$(date '+%F %T')] KV compression: enabled=$KV_CACHE_COMPRESSION tile=$VISION_TILE_SIZE topk=$VISION_TOPK_TILES keep=$VISION_TOKEN_KEEP_RATIO query_window=$VISION_SCORE_QUERY_WINDOW layers=$VISION_SCORE_LAYER_MODE:$VISION_SCORE_LAYERS pool=$VISION_SCORE_POOL_KERNEL" | tee -a "$LOG"
"$PYTHON" "$REPO/D2F-eval/eval_lladao_gui.py" \
  --backend d2f_vllm \
  --lladao-repo "$LLADAO_REPO" \
  --model-path "$SOURCE_MODEL" \
  --checkpoint "$SOURCE_MODEL/ema.safetensors" \
  --runtime-model "$RUNTIME_MODEL" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --benchmarks mind2web \
  --limit "$LIMIT" \
  --warmup 1 \
  --max-new-tokens 64 \
  --block-size 16 \
  --block-add-threshold 0.1 \
  --decoded-token-threshold 0.95 \
  --skip-threshold 0.9 \
  --max-model-len 16384 \
  --master-port "$MASTER_PORT" \
  --attention-backend "$D2F_VLLM_ATTENTION_BACKEND" \
  --rms-norm-backend "$D2F_VLLM_RMS_NORM_BACKEND" \
  "$COMPRESSION_FLAG" \
  --vision-tile-size "$VISION_TILE_SIZE" \
  --vision-topk-tiles "$VISION_TOPK_TILES" \
  --vision-token-keep-ratio "$VISION_TOKEN_KEEP_RATIO" \
  --vision-score-query-window "$VISION_SCORE_QUERY_WINDOW" \
  --vision-score-layers "$VISION_SCORE_LAYERS" \
  --vision-score-layer-mode "$VISION_SCORE_LAYER_MODE" \
  --vision-score-pool-kernel "$VISION_SCORE_POOL_KERNEL" \
  --fail-fast \
  2>&1 | tee -a "$LOG"

(
  cd "$LLADAO_REPO"
  "$PYTHON" -m eval.gui_grounding.score_benchmark \
    --benchmark-root "$BENCHMARK_ROOT" \
    --predictions-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR/scores" \
    --benchmarks mind2web \
    --limit "$LIMIT"
) 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] LLADAO_GUI_NONPD_DONE output=$OUTPUT_DIR log=$LOG" | tee -a "$LOG"
