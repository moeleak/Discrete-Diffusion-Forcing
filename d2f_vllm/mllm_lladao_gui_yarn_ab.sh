#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${REPO:-$ROOT/src/Discrete-Diffusion-Forcing}"
LLADAO_REPO="${LLADAO_REPO:-$ROOT/src/LLaDA-o}"
PYTHON="${PYTHON:-$ROOT/env-d2f-vllm/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
FULL_PAGE_POSITION_MODE="${FULL_PAGE_POSITION_MODE:-sequential}"
KV_CACHE_COMPRESSION="${KV_CACHE_COMPRESSION:-0}"
if [[ "$KV_CACHE_COMPRESSION" != "0" ]]; then
  echo "the original-16K/YaRN-128K comparison requires KV_CACHE_COMPRESSION=0" >&2
  exit 2
fi
CACHE_TAG="nocompress"
AB_LOG="${AB_LOG:-$ROOT/logs/d2f-vllm-yarn-ab-${FULL_PAGE_POSITION_MODE}-${CACHE_TAG}-${RUN_ID}.log}"
ORIGINAL_OUTPUT="${ORIGINAL_OUTPUT:-$ROOT/results/d2f-vllm-fullpage-native-resize-nocompress-original16k}"
YARN_OUTPUT="${YARN_OUTPUT:-$ROOT/results/d2f-vllm-fullpage-${FULL_PAGE_POSITION_MODE}-${CACHE_TAG}-yarn}"
COMPARISON_OUTPUT="${COMPARISON_OUTPUT:-$ROOT/results/d2f-vllm-fullpage-original16k-vs-yarn128k-comparison}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$ROOT/data/mind2web-fullpage-16k-64k}"

mkdir -p "$(dirname "$AB_LOG")"
exec > >(tee -a "$AB_LOG") 2>&1

echo "[$(date '+%F %T')] starting original-16K/YaRN-128K A/B"
echo "[$(date '+%F %T')] benchmark=$BENCHMARK_ROOT"
echo "[$(date '+%F %T')] full_page_position_mode=$FULL_PAGE_POSITION_MODE kv_cache_compression=$KV_CACHE_COMPRESSION"

MODE=original \
GPU=0 \
MASTER_PORT=32343 \
RUN_ID="$RUN_ID" \
FULL_PAGE_POSITION_MODE=native \
KV_CACHE_COMPRESSION=0 \
OUTPUT_DIR="$ORIGINAL_OUTPUT" \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
bash "$REPO/d2f_vllm/mllm_lladao_gui_long_context.sh" &
ORIGINAL_PID=$!

MODE=yarn \
GPU=1 \
MASTER_PORT=32353 \
RUN_ID="$RUN_ID" \
FULL_PAGE_POSITION_MODE="$FULL_PAGE_POSITION_MODE" \
KV_CACHE_COMPRESSION="$KV_CACHE_COMPRESSION" \
OUTPUT_DIR="$YARN_OUTPUT" \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
bash "$REPO/d2f_vllm/mllm_lladao_gui_long_context.sh" &
YARN_PID=$!

echo "$ORIGINAL_PID" > "$ROOT/logs/d2f-vllm-original16k.pid"
echo "$YARN_PID" > "$ROOT/logs/d2f-vllm-yarn.pid"
echo "[$(date '+%F %T')] pids original16k=$ORIGINAL_PID yarn128k=$YARN_PID"

original_status=0
yarn_status=0
wait "$ORIGINAL_PID" || original_status=$?
wait "$YARN_PID" || yarn_status=$?
echo "[$(date '+%F %T')] workers exited original16k=$original_status yarn128k=$yarn_status"
if [[ "$original_status" -ne 0 || "$yarn_status" -ne 0 ]]; then
  exit 1
fi

(
  cd "$LLADAO_REPO"
  "$PYTHON" -m eval.gui_grounding.compare_long_context \
    --benchmark-root "$BENCHMARK_ROOT" \
    --original-dir "$ORIGINAL_OUTPUT" \
    --yarn-dir "$YARN_OUTPUT" \
    --output-dir "$COMPARISON_OUTPUT" \
    --require-original-vs-yarn
)

echo "[$(date '+%F %T')] ORIGINAL16K_YARN128K_AB_DONE comparison=$COMPARISON_OUTPUT"
