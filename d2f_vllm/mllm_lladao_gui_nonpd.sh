#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${REPO:-$ROOT/src/Discrete-Diffusion-Forcing}"
PYTHON="${PYTHON:-$ROOT/env-d2f-vllm/bin/python}"
RUNTIME_MODEL="${RUNTIME_MODEL:-$ROOT/models/lladao-gui-d2f-vllm-step1377}"
SOURCE_MODEL="${SOURCE_MODEL:-$ROOT/models/lladao-gui-mind2web-step750}"
LLADAO_REPO="${LLADAO_REPO:-$ROOT/src/LLaDA-o}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$ROOT/data/bench_ocr}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/d2f-vllm-nonpd-100}"
LIMIT="${LIMIT:-100}"
GPU="${GPU:-0}"
MASTER_PORT="${MASTER_PORT:-32333}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG="${LOG:-$ROOT/logs/d2f-vllm-nonpd-${RUN_ID}.log}"

mkdir -p "$(dirname "$LOG")" "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export D2F_VLLM_ATTENTION_BACKEND="${D2F_VLLM_ATTENTION_BACKEND:-sdpa}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[$(date '+%F %T')] LLaDA-o GUI Non-PD: gpu=$GPU limit=$LIMIT model=$RUNTIME_MODEL" | tee -a "$LOG"
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
