#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${REPO:-$ROOT/src/Discrete-Diffusion-Forcing}"
LLADAO_REPO="${LLADAO_REPO:-$ROOT/src/LLaDA-o}"
PYTHON="${PYTHON:-$ROOT/env-d2f-vllm/bin/python}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$ROOT/data/mind2web-fullpage-16k-64k}"
LIMIT="${LIMIT:-100}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REVISION="${REVISION:-$(git -C "$REPO" rev-parse --short HEAD)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/d2f-vllm-yarn-isolation-${REVISION}-n${LIMIT}}"
ORIGINAL_OUTPUT="${ORIGINAL_OUTPUT:-$RESULT_ROOT/original16k}"
YARN_OUTPUT="${YARN_OUTPUT:-$RESULT_ROOT/yarn128k}"
COMPARISON_OUTPUT="${COMPARISON_OUTPUT:-$RESULT_ROOT/comparison}"
LOG="${LOG:-$ROOT/logs/d2f-vllm-yarn-isolation-${REVISION}-n${LIMIT}-${RUN_ID}.log}"

if ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || (( LIMIT > 100 )); then
  echo "YaRN isolation LIMIT must be an integer in [1, 100]" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG")" "$RESULT_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] starting controlled YaRN isolation A/B"
echo "[$(date '+%F %T')] revision=$REVISION samples=$LIMIT"
echo "[$(date '+%F %T')] invariant=native_resize,native_positions,kv_capacity_16k,no_kv_compression"

MODE=original \
INPUT_MODE=native_resize \
LIMIT="$LIMIT" \
GPU=0 \
MASTER_PORT=32443 \
RUN_ID="$RUN_ID" \
KV_CACHE_COMPRESSION=0 \
OUTPUT_DIR="$ORIGINAL_OUTPUT" \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
bash "$REPO/d2f_vllm/mllm_lladao_gui_long_context.sh" &
ORIGINAL_PID=$!

MODE=yarn \
INPUT_MODE=native_resize \
LIMIT="$LIMIT" \
GPU=1 \
MASTER_PORT=32453 \
RUN_ID="$RUN_ID" \
MAX_MODEL_LEN=131072 \
KV_CACHE_CAPACITY=16384 \
KV_CACHE_COMPRESSION=0 \
OUTPUT_DIR="$YARN_OUTPUT" \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
bash "$REPO/d2f_vllm/mllm_lladao_gui_long_context.sh" &
YARN_PID=$!

echo "$ORIGINAL_PID" > "$ROOT/logs/d2f-vllm-yarn-isolation-original.pid"
echo "$YARN_PID" > "$ROOT/logs/d2f-vllm-yarn-isolation-yarn.pid"
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
    --limit "$LIMIT" \
    --require-yarn-isolation
)

echo "[$(date '+%F %T')] YARN_ISOLATION_DONE comparison=$COMPARISON_OUTPUT"
