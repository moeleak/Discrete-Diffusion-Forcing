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
INPUT_MODE="${INPUT_MODE:-}"
LIMIT="${LIMIT:-}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
FULL_PAGE_TILE_SIZE="${FULL_PAGE_TILE_SIZE:-980}"
FULL_PAGE_POSITION_MODE="${FULL_PAGE_POSITION_MODE:-sequential}"
FULL_PAGE_OVERVIEW="${FULL_PAGE_OVERVIEW:-0}"
FULL_PAGE_TRUNCATION="${FULL_PAGE_TRUNCATION:-0}"
ALLOW_UNTRUNCATED_ORIGINAL_FULL_PAGE="${ALLOW_UNTRUNCATED_ORIGINAL_FULL_PAGE:-0}"
KV_CACHE_COMPRESSION="${KV_CACHE_COMPRESSION:-0}"
KV_CACHE_RETRIEVAL="${KV_CACHE_RETRIEVAL:-0}"
KV_RETRIEVAL_TOPK_IMAGES="${KV_RETRIEVAL_TOPK_IMAGES:-4}"
KV_RETRIEVAL_SCORE_MODE="${KV_RETRIEVAL_SCORE_MODE:-masked_self_information}"
KV_RETRIEVAL_MASK_ROUNDS="${KV_RETRIEVAL_MASK_ROUNDS:-2}"
KV_RETRIEVAL_PACKED_SCORING="${KV_RETRIEVAL_PACKED_SCORING:-1}"
KV_RETRIEVAL_MAX_BATCH_TOKENS="${KV_RETRIEVAL_MAX_BATCH_TOKENS:-65536}"
KV_RETRIEVAL_KEEP_OVERVIEW="${KV_RETRIEVAL_KEEP_OVERVIEW:-1}"
if [[ "$KV_RETRIEVAL_PACKED_SCORING" == "1" ]]; then
  SCORING_BATCH_TAG="packed${KV_RETRIEVAL_MAX_BATCH_TOKENS}"
elif [[ "$KV_RETRIEVAL_PACKED_SCORING" == "0" ]]; then
  SCORING_BATCH_TAG="sequential"
else
  echo "KV_RETRIEVAL_PACKED_SCORING must be 0 or 1" >&2
  exit 2
fi
if [[ "$KV_CACHE_COMPRESSION" == "1" ]]; then
  CACHE_TAG="kvcompress"
elif [[ "$KV_CACHE_COMPRESSION" == "0" ]]; then
  CACHE_TAG="nocompress"
else
  echo "KV_CACHE_COMPRESSION must be 0 or 1" >&2
  exit 2
fi
if [[ "$KV_CACHE_RETRIEVAL" == "1" ]]; then
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
elif [[ "$KV_CACHE_RETRIEVAL" == "0" ]]; then
  RETRIEVAL_TAG="noretrieve"
else
  echo "KV_CACHE_RETRIEVAL must be 0 or 1" >&2
  exit 2
fi
if [[ "$KV_CACHE_COMPRESSION" == "1" && "$KV_CACHE_RETRIEVAL" == "1" ]]; then
  echo "KV cache retrieval and compression are mutually exclusive" >&2
  exit 2
fi
if ! [[ "$KV_RETRIEVAL_TOPK_IMAGES" =~ ^[0-9]+$ ]]; then
  echo "KV_RETRIEVAL_TOPK_IMAGES must be a non-negative integer" >&2
  exit 2
fi
if [[
  "$KV_RETRIEVAL_SCORE_MODE" != "masked_self_information"
  && "$KV_RETRIEVAL_SCORE_MODE" != "cached_masked_self_information"
  && "$KV_RETRIEVAL_SCORE_MODE" != "causal_masked_self_information"
  && "$KV_RETRIEVAL_SCORE_MODE" != "causal_self_information"
]]; then
  echo "KV_RETRIEVAL_SCORE_MODE must be masked_self_information, cached_masked_self_information, causal_masked_self_information, or causal_self_information" >&2
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
if [[
  "$KV_RETRIEVAL_KEEP_OVERVIEW" != "0"
  && "$KV_RETRIEVAL_KEEP_OVERVIEW" != "1"
]]; then
  echo "KV_RETRIEVAL_KEEP_OVERVIEW must be 0 or 1" >&2
  exit 2
fi
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/d2f-vllm-fullpage-${FULL_PAGE_POSITION_MODE}-${CACHE_TAG}-${RETRIEVAL_TAG}-${MODE}}"
LOG="${LOG:-$ROOT/logs/d2f-vllm-fullpage-${FULL_PAGE_POSITION_MODE}-${CACHE_TAG}-${RETRIEVAL_TAG}-${MODE}-${RUN_ID}.log}"
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

if [[ -z "$INPUT_MODE" ]]; then
  if [[ "$MODE" == "original" ]]; then
    INPUT_MODE="native_resize"
  else
    INPUT_MODE="full_page"
  fi
fi
case "$INPUT_MODE" in
  native_resize)
    FULL_PAGE_POSITION_MODE=native
    FULL_PAGE_ARGS=(--no-full-page-tiles)
    ;;
  full_page)
    FULL_PAGE_ARGS=(--full-page-tiles)
    ;;
  *)
    echo "INPUT_MODE must be one of: native_resize, full_page" >&2
    exit 2
    ;;
esac
if [[ "$FULL_PAGE_OVERVIEW" == "1" ]]; then
  if [[ "$INPUT_MODE" != "full_page" ]]; then
    echo "FULL_PAGE_OVERVIEW=1 requires INPUT_MODE=full_page" >&2
    exit 2
  fi
  OVERVIEW_FLAG="--full-page-overview"
elif [[ "$FULL_PAGE_OVERVIEW" == "0" ]]; then
  OVERVIEW_FLAG="--no-full-page-overview"
else
  echo "FULL_PAGE_OVERVIEW must be 0 or 1" >&2
  exit 2
fi
if [[ "$FULL_PAGE_TRUNCATION" == "1" ]]; then
  if [[ "$INPUT_MODE" != "full_page" ]]; then
    echo "FULL_PAGE_TRUNCATION=1 requires INPUT_MODE=full_page" >&2
    exit 2
  fi
  TRUNCATION_FLAG="--truncate-full-page-tiles"
elif [[ "$FULL_PAGE_TRUNCATION" == "0" ]]; then
  TRUNCATION_FLAG="--no-truncate-full-page-tiles"
else
  echo "FULL_PAGE_TRUNCATION must be 0 or 1" >&2
  exit 2
fi
if [[ "$FULL_PAGE_OVERVIEW" == "1" && "$FULL_PAGE_TRUNCATION" == "1" ]]; then
  echo "full-page overview and truncation are mutually exclusive" >&2
  exit 2
fi
if [[
  "$ALLOW_UNTRUNCATED_ORIGINAL_FULL_PAGE" != "0"
  && "$ALLOW_UNTRUNCATED_ORIGINAL_FULL_PAGE" != "1"
]]; then
  echo "ALLOW_UNTRUNCATED_ORIGINAL_FULL_PAGE must be 0 or 1" >&2
  exit 2
fi
if [[
  "$MODE" == "original"
  && "$INPUT_MODE" == "full_page"
  && "$FULL_PAGE_TRUNCATION" != "1"
  && "$ALLOW_UNTRUNCATED_ORIGINAL_FULL_PAGE" != "1"
]]; then
  echo "MODE=original with full-page input requires FULL_PAGE_TRUNCATION=1 or an explicitly prefiltered benchmark" >&2
  exit 2
fi

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  if ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "LIMIT must be a positive integer" >&2
    exit 2
  fi
  LIMIT_ARGS=(--limit "$LIMIT")
fi
if ! [[ "$BLOCK_SIZE" =~ ^[1-9][0-9]*$ ]] || (( 64 % BLOCK_SIZE != 0 )); then
  echo "BLOCK_SIZE must be a positive divisor of 64" >&2
  exit 2
fi
if ! [[ "$FULL_PAGE_TILE_SIZE" =~ ^[1-9][0-9]*$ ]] || (( FULL_PAGE_TILE_SIZE > 980 )); then
  echo "FULL_PAGE_TILE_SIZE must be an integer in [1, 980]" >&2
  exit 2
fi

if [[ "$KV_CACHE_COMPRESSION" == "1" ]]; then
  COMPRESSION_FLAG="--kv-cache-compression"
else
  COMPRESSION_FLAG="--no-kv-cache-compression"
fi
if [[ "$KV_CACHE_RETRIEVAL" == "1" ]]; then
  RETRIEVAL_FLAG="--kv-cache-retrieval"
else
  RETRIEVAL_FLAG="--no-kv-cache-retrieval"
fi
if [[ "$KV_RETRIEVAL_KEEP_OVERVIEW" == "1" ]]; then
  RETRIEVAL_OVERVIEW_FLAG="--kv-retrieval-keep-overview"
else
  RETRIEVAL_OVERVIEW_FLAG="--no-kv-retrieval-keep-overview"
fi
if [[ "$KV_RETRIEVAL_PACKED_SCORING" == "1" ]]; then
  RETRIEVAL_PACKED_FLAG="--kv-retrieval-packed-scoring"
else
  RETRIEVAL_PACKED_FLAG="--no-kv-retrieval-packed-scoring"
fi

{
  echo "[$(date '+%F %T')] mode=$MODE gpu=$GPU"
  echo "[$(date '+%F %T')] max_model_len=$MAX_MODEL_LEN kv_cache_capacity=$KV_CACHE_CAPACITY"
  echo "[$(date '+%F %T')] input_mode=$INPUT_MODE full_page_tile_size=$FULL_PAGE_TILE_SIZE full_page_position_mode=$FULL_PAGE_POSITION_MODE full_page_overview=$FULL_PAGE_OVERVIEW full_page_truncation=$FULL_PAGE_TRUNCATION allow_untruncated_original_full_page=$ALLOW_UNTRUNCATED_ORIGINAL_FULL_PAGE kv_cache_compression=$KV_CACHE_COMPRESSION kv_cache_retrieval=$KV_CACHE_RETRIEVAL kv_retrieval_topk_images=$KV_RETRIEVAL_TOPK_IMAGES kv_retrieval_score_mode=$KV_RETRIEVAL_SCORE_MODE kv_retrieval_mask_rounds=$KV_RETRIEVAL_MASK_ROUNDS kv_retrieval_packed_scoring=$KV_RETRIEVAL_PACKED_SCORING kv_retrieval_max_batch_tokens=$KV_RETRIEVAL_MAX_BATCH_TOKENS kv_retrieval_keep_overview=$KV_RETRIEVAL_KEEP_OVERVIEW block_size=$BLOCK_SIZE limit=${LIMIT:-all}"
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
  "${LIMIT_ARGS[@]}" \
  --warmup 0 \
  --max-new-tokens 64 \
  --block-size "$BLOCK_SIZE" \
  --block-add-threshold 0.1 \
  --decoded-token-threshold 0.95 \
  --skip-threshold 0.9 \
  --max-model-len "$MAX_MODEL_LEN" \
  --kv-cache-capacity "$KV_CACHE_CAPACITY" \
  --original-max-position-embeddings 16384 \
  "${FULL_PAGE_ARGS[@]}" \
  --full-page-tile-size "$FULL_PAGE_TILE_SIZE" \
  --full-page-position-mode "$FULL_PAGE_POSITION_MODE" \
  "$OVERVIEW_FLAG" \
  "$TRUNCATION_FLAG" \
  --master-port "$MASTER_PORT" \
  --attention-backend "$D2F_VLLM_ATTENTION_BACKEND" \
  --rms-norm-backend "$D2F_VLLM_RMS_NORM_BACKEND" \
  "$COMPRESSION_FLAG" \
  "$RETRIEVAL_FLAG" \
  --kv-retrieval-topk-images "$KV_RETRIEVAL_TOPK_IMAGES" \
  --kv-retrieval-score-mode "$KV_RETRIEVAL_SCORE_MODE" \
  --kv-retrieval-mask-rounds "$KV_RETRIEVAL_MASK_ROUNDS" \
  "$RETRIEVAL_PACKED_FLAG" \
  --kv-retrieval-max-batch-tokens "$KV_RETRIEVAL_MAX_BATCH_TOKENS" \
  "$RETRIEVAL_OVERVIEW_FLAG" \
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
    --benchmarks mind2web_fullpage \
    "${LIMIT_ARGS[@]}"
) 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] LONG_CONTEXT_DONE mode=$MODE output=$OUTPUT_DIR" |
  tee -a "$LOG"
