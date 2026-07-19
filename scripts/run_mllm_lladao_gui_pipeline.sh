#!/usr/bin/env bash
# Wait for prepared data and idle GPUs, then train and gate the LLaDA-o D2F adapter.

set -euo pipefail

ROOT="${LLADAO_WORK_ROOT:-/home/ma-user/work/LLaDA-o}"
REPO="${D2F_REPO:-${ROOT}/src/Discrete-Diffusion-Forcing}"
LLADAO="${LLADAO_REPO:-${ROOT}/src/LLaDA-o}"
PYTHON="${PYTHON:-${ROOT}/env/bin/python}"
ACCELERATE="${ACCELERATE:-${ROOT}/env/bin/accelerate}"
TORCHRUN="${TORCHRUN:-${ROOT}/env/bin/torchrun}"
CONFIG="${CONFIG:-${REPO}/D2F-train/config/lladao_gui.yaml}"
MODEL="${MODEL:-${ROOT}/models/lladao-gui-mind2web-step750}"
TRAIN_ROOT="${TRAIN_ROOT:-${ROOT}/data/train_ocr}"
BENCH_ROOT="${BENCH_ROOT:-${ROOT}/data/bench_ocr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/d2f-block16-r32}"
SMOKE_LIMIT="${SMOKE_LIMIT:-100}"
FULL_BENCHMARKS="${FULL_BENCHMARKS:-mind2web,screenspot_web_text,screenspot_web_icon}"
MAX_STEPS="${MAX_STEPS:-1377}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-4096}"
GPU_UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-10}"
GPU_STABLE_CHECKS="${GPU_STABLE_CHECKS:-5}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"

export PYTHONPATH="${REPO}:${LLADAO}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${ROOT}/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/cache/torch}"
export TOKENIZERS_PARALLELISM=false

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

wait_for_file() {
  local path="$1"
  while [[ ! -f "${path}" ]]; do
    echo "[$(timestamp)] waiting for ${path}"
    sleep 60
  done
}

wait_for_idle_gpus() {
  local stable=0
  while (( stable < GPU_STABLE_CHECKS )); do
    mapfile -t rows < <(
      nvidia-smi \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits
    )
    local idle=1
    if (( ${#rows[@]} < 2 )); then
      idle=0
    else
      local row memory utilization
      for row in "${rows[@]:0:2}"; do
        IFS=',' read -r memory utilization <<<"${row}"
        memory="${memory//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        if (( memory >= GPU_MEMORY_LIMIT_MIB || utilization >= GPU_UTIL_LIMIT_PERCENT )); then
          idle=0
        fi
      done
    fi
    if (( idle )); then
      ((stable += 1))
    else
      stable=0
    fi
    echo "[$(timestamp)] GPU idle stability ${stable}/${GPU_STABLE_CHECKS}: ${rows[*]:-unavailable}"
    (( stable < GPU_STABLE_CHECKS )) && sleep "${GPU_POLL_SECONDS}"
  done
}

latest_checkpoint() {
  find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step-*' \
    -exec test -f '{}/training_state.pt' ';' -print 2>/dev/null | sort | tail -1
}

run_eval() {
  local backend="$1"
  local output="$2"
  local limit="$3"
  local benchmarks="$4"
  local reset="$5"
  shift 5
  local adapter_args=()
  local inference_args=("$@")
  local limit_args=()
  local resume_args=()
  if [[ "${backend}" == d2f ]]; then
    adapter_args=(--adapter "${OUTPUT_ROOT}/step-$(printf '%07d' "${MAX_STEPS}")/adapter")
  fi
  if [[ -n "${limit}" ]]; then
    limit_args=(--limit "${limit}")
  fi
  if [[ "${reset}" == true ]]; then
    resume_args=(--no-resume)
  fi
  "${TORCHRUN}" --standalone --nproc-per-node=2 \
    "${REPO}/D2F-eval/eval_lladao_gui.py" \
    --backend "${backend}" \
    --lladao-repo "${LLADAO}" \
    --model-path "${MODEL}" \
    --checkpoint "${MODEL}/ema.safetensors" \
    "${adapter_args[@]}" \
    --benchmark-root "${BENCH_ROOT}" \
    --output-dir "${output}" \
    --benchmarks "${benchmarks}" \
    "${limit_args[@]}" \
    "${resume_args[@]}" \
    "${inference_args[@]}"
  "${PYTHON}" "${LLADAO}/eval/gui_grounding/score_benchmark.py" \
    --benchmark-root "${BENCH_ROOT}" \
    --predictions-dir "${output}" \
    --output-dir "${output}/scores" \
    --benchmarks "${benchmarks}" \
    "${limit_args[@]}"
}

run_gate() {
  local baseline="$1"
  local d2f="$2"
  local benchmark="$3"
  local output="$4"
  "${PYTHON}" D2F-eval/compare_lladao_gui.py \
    --baseline-predictions "${baseline}" \
    --d2f-predictions "${d2f}" \
    --baseline-scores "${baseline}/scores/results.json" \
    --d2f-scores "${d2f}/scores/results.json" \
    --benchmark "${benchmark}" \
    --output "${output}"
}

validate_prepared_data() {
  "${PYTHON}" - "${TRAIN_ROOT}/manifest.json" "${BENCH_ROOT}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

train = json.loads(Path(sys.argv[1]).read_text())
benchmark = json.loads(Path(sys.argv[2]).read_text())
train_ocr = train["mind2web_ocr_realignment"]
benchmark_ocr = benchmark["ocr_target_realignment"]
checks = {
    "training rows": (int(train_ocr["rows"]), 7341),
    "training OCR errors": (int(train_ocr["processing_errors"]), 0),
    "benchmark rows": (int(benchmark_ocr["samples"]), 6055),
    "benchmark OCR errors": (
        int(benchmark_ocr.get("counters", {}).get("mind2web:ocr_error", 0)),
        0,
    ),
    "benchmark manifest rows": (
        int(benchmark["benchmarks"]["mind2web"]["rows"]),
        6055,
    ),
}
failed = {name: value for name, value in checks.items() if value[0] != value[1]}
if failed:
    raise SystemExit(f"prepared data validation failed: {failed}")
print("prepared data validation passed: " + ", ".join(
    f"{name}={actual}" for name, (actual, _) in checks.items()
))
PY
}

wait_for_file "${TRAIN_ROOT}/manifest.json"
wait_for_file "${BENCH_ROOT}/manifest.json"
validate_prepared_data
echo "7d7796a27cfc81b85c3711873799aefd99c16951b15fbfcc3f56954bef9bbb23  ${MODEL}/ema.safetensors" \
  | sha256sum --check --status
wait_for_idle_gpus

cd "${REPO}"
mkdir -p "${OUTPUT_ROOT}"

checkpoint="$(latest_checkpoint)"
if [[ -z "${checkpoint}" ]]; then
  echo "[$(timestamp)] running one-step distributed training smoke test"
  "${ACCELERATE}" launch --num_processes 2 \
    D2F-train/train_lladao_gui.py --config "${CONFIG}" \
    --max-steps "${MAX_STEPS}" --stop-after-step 1
  checkpoint="$(latest_checkpoint)"
fi

final_checkpoint="${OUTPUT_ROOT}/step-$(printf '%07d' "${MAX_STEPS}")"
if [[ ! -f "${final_checkpoint}/adapter/adapter_model.safetensors" ]]; then
  echo "[$(timestamp)] resuming D2F training from ${checkpoint} to step ${MAX_STEPS}"
  "${ACCELERATE}" launch --num_processes 2 \
    D2F-train/train_lladao_gui.py --config "${CONFIG}" \
    --max-steps "${MAX_STEPS}" --resume-from "${checkpoint}"
fi

smoke_root="${ROOT}/runs/paired-smoke-${SMOKE_LIMIT}"
echo "[$(timestamp)] running paired ${SMOKE_LIMIT}-sample benchmark"
smoke_baseline="${smoke_root}/baseline"
run_eval baseline "${smoke_baseline}" "${SMOKE_LIMIT}" mind2web true
candidate_specs=(
  "fast 0.10 0.95 0.90"
  "balanced 0.25 0.95 0.95"
  "official 0.50 0.90 1.00"
  "quality 0.50 1.00 1.00"
)
selected_name=""
selected_args=()
for spec in "${candidate_specs[@]}"; do
  read -r name block_add decoded skip <<<"${spec}"
  candidate_output="${smoke_root}/d2f-${name}"
  echo "[$(timestamp)] trying D2F decode candidate ${name}: ${block_add}/${decoded}/${skip}"
  if run_eval d2f "${candidate_output}" "${SMOKE_LIMIT}" mind2web true \
      --block-add-threshold "${block_add}" \
      --decoded-token-threshold "${decoded}" \
      --skip-threshold "${skip}" \
    && run_gate \
      "${smoke_baseline}" "${candidate_output}" mind2web \
      "${smoke_root}/gate-${name}.json"; then
    selected_name="${name}"
    selected_args=(
      --block-add-threshold "${block_add}"
      --decoded-token-threshold "${decoded}"
      --skip-threshold "${skip}"
    )
    break
  fi
done
if [[ -z "${selected_name}" ]]; then
  echo "[$(timestamp)] no D2F decode candidate passed the paired smoke gate" >&2
  exit 1
fi
"${PYTHON}" - \
  "${smoke_root}/selected-decode.json" "${selected_name}" \
  "${selected_args[1]}" "${selected_args[3]}" "${selected_args[5]}" <<'PY'
import json
import sys
from pathlib import Path

path, name, block_add, decoded, skip = sys.argv[1:]
value = {
    "name": name,
    "block_add_threshold": float(block_add),
    "decoded_token_threshold": float(decoded),
    "skip_threshold": float(skip),
}
Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
echo "[$(timestamp)] paired smoke gate passed with ${selected_name}"

full_root="${ROOT}/runs/paired-full"
echo "[$(timestamp)] running resumable full benchmark: ${FULL_BENCHMARKS}"
run_eval baseline "${full_root}/baseline" "" "${FULL_BENCHMARKS}" false
full_d2f="${full_root}/d2f-${selected_name}"
run_eval d2f "${full_d2f}" "" "${FULL_BENCHMARKS}" false "${selected_args[@]}"
IFS=',' read -ra full_benchmarks <<<"${FULL_BENCHMARKS}"
for benchmark in "${full_benchmarks[@]}"; do
  benchmark="${benchmark//[[:space:]]/}"
  [[ -n "${benchmark}" ]] && run_gate \
    "${full_root}/baseline" "${full_d2f}" "${benchmark}" \
    "${full_root}/gate-${benchmark}-${selected_name}.json"
done
echo "[$(timestamp)] all full paired gates passed"
