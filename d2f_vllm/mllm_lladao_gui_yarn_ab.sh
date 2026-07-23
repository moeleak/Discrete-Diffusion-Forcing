#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${REPO:-$ROOT/src/Discrete-Diffusion-Forcing}"
LLADAO_REPO="${LLADAO_REPO:-$ROOT/src/LLaDA-o}"
PYTHON="${PYTHON:-$ROOT/env-d2f-vllm/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
AB_LOG="${AB_LOG:-$ROOT/logs/d2f-vllm-yarn-ab-${RUN_ID}.log}"
UNSCALED_OUTPUT="${UNSCALED_OUTPUT:-$ROOT/results/d2f-vllm-fullpage-unscaled}"
YARN_OUTPUT="${YARN_OUTPUT:-$ROOT/results/d2f-vllm-fullpage-yarn}"
COMPARISON_OUTPUT="${COMPARISON_OUTPUT:-$ROOT/results/d2f-vllm-fullpage-comparison}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$ROOT/data/mind2web-fullpage-16k-64k}"

mkdir -p "$(dirname "$AB_LOG")"
exec > >(tee -a "$AB_LOG") 2>&1

echo "[$(date '+%F %T')] starting unscaled/YARN A/B"
echo "[$(date '+%F %T')] benchmark=$BENCHMARK_ROOT"

MODE=unscaled \
GPU=0 \
MASTER_PORT=32343 \
RUN_ID="$RUN_ID" \
OUTPUT_DIR="$UNSCALED_OUTPUT" \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
bash "$REPO/d2f_vllm/mllm_lladao_gui_long_context.sh" &
UNSCALED_PID=$!

MODE=yarn \
GPU=1 \
MASTER_PORT=32353 \
RUN_ID="$RUN_ID" \
OUTPUT_DIR="$YARN_OUTPUT" \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
bash "$REPO/d2f_vllm/mllm_lladao_gui_long_context.sh" &
YARN_PID=$!

echo "$UNSCALED_PID" > "$ROOT/logs/d2f-vllm-unscaled.pid"
echo "$YARN_PID" > "$ROOT/logs/d2f-vllm-yarn.pid"
echo "[$(date '+%F %T')] pids unscaled=$UNSCALED_PID yarn=$YARN_PID"

unscaled_status=0
yarn_status=0
wait "$UNSCALED_PID" || unscaled_status=$?
wait "$YARN_PID" || yarn_status=$?
echo "[$(date '+%F %T')] workers exited unscaled=$unscaled_status yarn=$yarn_status"
if [[ "$unscaled_status" -ne 0 || "$yarn_status" -ne 0 ]]; then
  exit 1
fi

(
  cd "$LLADAO_REPO"
  "$PYTHON" -m eval.gui_grounding.compare_long_context \
    --benchmark-root "$BENCHMARK_ROOT" \
    --unscaled-dir "$UNSCALED_OUTPUT" \
    --yarn-dir "$YARN_OUTPUT" \
    --output-dir "$COMPARISON_OUTPUT"
)

echo "[$(date '+%F %T')] YARN_AB_DONE comparison=$COMPARISON_OUTPUT"
