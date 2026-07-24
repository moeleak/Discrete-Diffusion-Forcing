#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${REPO:-$ROOT/src/Discrete-Diffusion-Forcing}"
LLADAO_REPO="${LLADAO_REPO:-$ROOT/src/LLaDA-o}"
PYTHON="${PYTHON:-$ROOT/env-d2f-vllm/bin/python}"
LIMIT="${LIMIT:-100}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REVISION="${REVISION:-$(git -C "$REPO" rev-parse --short HEAD)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/d2f-vllm-true-long-yarn-isolation-${REVISION}-n${LIMIT}}"
UNSCALED_OUTPUT="${UNSCALED_OUTPUT:-$RESULT_ROOT/unscaled128k}"
YARN_OUTPUT="${YARN_OUTPUT:-$RESULT_ROOT/yarn128k}"
COMPARISON_OUTPUT="${COMPARISON_OUTPUT:-$RESULT_ROOT/comparison}"
AB_LOG="${AB_LOG:-$ROOT/logs/d2f-vllm-true-long-yarn-isolation-${REVISION}-n${LIMIT}-${RUN_ID}.log}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$ROOT/data/mind2web-fullpage-16k-64k}"

if ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || (( LIMIT > 100 )); then
  echo "true-long YaRN isolation LIMIT must be an integer in [1, 100]" >&2
  exit 2
fi

mkdir -p "$(dirname "$AB_LOG")" "$RESULT_ROOT"
exec > >(tee -a "$AB_LOG") 2>&1

echo "[$(date '+%F %T')] starting true-long unscaled-128K/YaRN-128K isolation"
echo "[$(date '+%F %T')] revision=$REVISION samples=$LIMIT benchmark=$BENCHMARK_ROOT"
echo "[$(date '+%F %T')] invariant=full_page_tiles,sequential_positions,kv_capacity_65536,no_kv_compression"

MODE=unscaled \
INPUT_MODE=full_page \
LIMIT="$LIMIT" \
GPU=0 \
MASTER_PORT=32343 \
RUN_ID="$RUN_ID" \
MAX_MODEL_LEN=131072 \
KV_CACHE_CAPACITY=65536 \
KV_CACHE_COMPRESSION=0 \
FULL_PAGE_POSITION_MODE=sequential \
OUTPUT_DIR="$UNSCALED_OUTPUT" \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
bash "$REPO/d2f_vllm/mllm_lladao_gui_long_context.sh" &
UNSCALED_PID=$!

MODE=yarn \
INPUT_MODE=full_page \
LIMIT="$LIMIT" \
GPU=1 \
MASTER_PORT=32353 \
RUN_ID="$RUN_ID" \
MAX_MODEL_LEN=131072 \
KV_CACHE_CAPACITY=65536 \
KV_CACHE_COMPRESSION=0 \
FULL_PAGE_POSITION_MODE=sequential \
OUTPUT_DIR="$YARN_OUTPUT" \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
bash "$REPO/d2f_vllm/mllm_lladao_gui_long_context.sh" &
YARN_PID=$!

echo "$UNSCALED_PID" > "$ROOT/logs/d2f-vllm-unscaled128k.pid"
echo "$YARN_PID" > "$ROOT/logs/d2f-vllm-yarn.pid"
echo "[$(date '+%F %T')] pids unscaled128k=$UNSCALED_PID yarn128k=$YARN_PID"

unscaled_status=0
yarn_status=0
wait "$UNSCALED_PID" || unscaled_status=$?
wait "$YARN_PID" || yarn_status=$?
echo "[$(date '+%F %T')] workers exited unscaled128k=$unscaled_status yarn128k=$yarn_status"
if [[ "$unscaled_status" -ne 0 || "$yarn_status" -ne 0 ]]; then
  exit 1
fi

(
  cd "$LLADAO_REPO"
  "$PYTHON" -m eval.gui_grounding.compare_long_context \
    --benchmark-root "$BENCHMARK_ROOT" \
    --unscaled-dir "$UNSCALED_OUTPUT" \
    --yarn-dir "$YARN_OUTPUT" \
    --output-dir "$COMPARISON_OUTPUT" \
    --limit "$LIMIT" \
    --require-long-yarn-isolation
)

echo "[$(date '+%F %T')] TRUE_LONG_YARN_ISOLATION_DONE comparison=$COMPARISON_OUTPUT"
