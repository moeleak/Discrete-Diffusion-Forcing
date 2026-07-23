#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${REPO:-$ROOT/src/Discrete-Diffusion-Forcing}"
LLADAO_REPO="${LLADAO_REPO:-$ROOT/src/LLaDA-o}"
PYTHON="${PYTHON:-$ROOT/env-d2f-vllm/bin/python}"
RUNTIME_MODEL="${RUNTIME_MODEL:-$ROOT/models/lladao-gui-d2f-vllm-step1377-exact}"
SOURCE_MODEL="${SOURCE_MODEL:-$ROOT/models/lladao-gui-mind2web-step750}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$ROOT/data/mind2web-fullpage-16k-64k}"
MODE="${MODE:-yarn}"
GPU="${GPU:-0}"
MASTER_PORT="${MASTER_PORT:-32343}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/d2f-vllm-fullpage-${MODE}}"
LOG="${LOG:-$ROOT/logs/d2f-vllm-fullpage-${MODE}-${RUN_ID}.log}"
KV_CACHE_COMPRESSION="${KV_CACHE_COMPRESSION:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
KV_CACHE_CAPACITY="${KV_CACHE_CAPACITY:-65536}"

mkdir -p "$(dirname "$LOG")" "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export D2F_VLLM_ATTENTION_BACKEND="${D2F_VLLM_ATTENTION_BACKEND:-sdpa}"
export D2F_VLLM_RMS_NORM_BACKEND="${D2F_VLLM_RMS_NORM_BACKEND:-vllm}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

case "$MODE" in
  yarn)
    ROPE_ARGS=(--rope-scaling yarn --rope-factor 8)
    ;;
  unscaled)
    ROPE_ARGS=(--rope-scaling none --allow-unscaled-max-model-len)
    ;;
  original)
    MAX_MODEL_LEN=16384
    KV_CACHE_CAPACITY=16384
    ROPE_ARGS=(--rope-scaling none)
    ;;
  *)
    echo "MODE must be one of: original, unscaled, yarn" >&2
    exit 2
    ;;
esac

if [[ "$KV_CACHE_COMPRESSION" == "1" ]]; then
  COMPRESSION_FLAG="--kv-cache-compression"
else
  COMPRESSION_FLAG="--no-kv-cache-compression"
fi

{
  echo "[$(date '+%F %T')] mode=$MODE gpu=$GPU"
  echo "[$(date '+%F %T')] max_model_len=$MAX_MODEL_LEN kv_cache_capacity=$KV_CACHE_CAPACITY"
  echo "[$(date '+%F %T')] benchmark=$BENCHMARK_ROOT output=$OUTPUT_DIR"
} | tee -a "$LOG"

"$PYTHON" "$REPO/D2F-eval/eval_lladao_gui.py" \
  --backend d2f_vllm \
  --lladao-repo "$LLADAO_REPO" \
  --model-path "$SOURCE_MODEL" \
  --checkpoint "$SOURCE_MODEL/ema.safetensors" \
  --runtime-model "$RUNTIME_MODEL" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --benchmarks mind2web_fullpage \
  --warmup 0 \
  --max-new-tokens 64 \
  --block-size 16 \
  --block-add-threshold 0.1 \
  --decoded-token-threshold 0.95 \
  --skip-threshold 0.9 \
  --max-model-len "$MAX_MODEL_LEN" \
  --kv-cache-capacity "$KV_CACHE_CAPACITY" \
  --original-max-position-embeddings 16384 \
  --full-page-tiles \
  --full-page-tile-size 980 \
  --master-port "$MASTER_PORT" \
  --attention-backend "$D2F_VLLM_ATTENTION_BACKEND" \
  --rms-norm-backend "$D2F_VLLM_RMS_NORM_BACKEND" \
  "$COMPRESSION_FLAG" \
  --vision-tile-size 16 \
  --vision-topk-tiles 20 \
  --vision-token-keep-ratio 0.75 \
  --vision-score-query-window 32 \
  --vision-score-layers 4 \
  --vision-score-layer-mode last \
  --vision-score-pool-kernel 7 \
  "${ROPE_ARGS[@]}" \
  2>&1 | tee -a "$LOG"

(
  cd "$LLADAO_REPO"
  "$PYTHON" -m eval.gui_grounding.score_benchmark \
    --benchmark-root "$BENCHMARK_ROOT" \
    --predictions-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR/scores" \
    --benchmarks mind2web_fullpage
) 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] LONG_CONTEXT_DONE mode=$MODE output=$OUTPUT_DIR" |
  tee -a "$LOG"
