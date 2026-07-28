# D2F-vLLM

This directory vendors the minimal runtime sources from
[`menik1126/Discrete-Diffusion-Forcing`](https://github.com/menik1126/Discrete-Diffusion-Forcing)
at commit `082c8c0`. Local LLaDA-o GUI-grounding support is maintained on top of
that baseline in this repository.

vLLM implementation for Diffusion LLMs, D2F is integrated as the core inference strategy, while also support training-free strategies like Fast-dLLM.

## Foundation of Our vLLM Implementation

Based on [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm).

## How We Implement



## Easy Install D2F-vLLM

```shell
pip install d2f_vllm
```

## Configure the Project from Source (for Developers)

We use [UV](https://github.com/astral-sh/uv) to manage the whole project. 

### Install UV

[UV Installation](https://docs.astral.sh/uv/getting-started/installation/)

### Initialize the Project

```shell
uv sync
source .venv/bin/activate
uv pip install -e .
```

For easy-activation:

```shell
echo "alias uvon=source .venv/bin/activate" >> ~/.zshrc # If using bash, change to .bashrc
source ~/.zshrc
```

Then, use `uvon` under the project root path to activate.

### Download vLLM

```shell
uv pip install vllm
```

`D2F-vLLM` still depends on some modules of `vLLM`, however, there are some problems lies in UV venv management, thus we have to install `vLLM` independently.

### Download Flash Attention (NO NEED RIGHT NOW)

```shell
uv pip install flash-attn --no-build-isolation
```

If not working, build `flash-attn` from scratch. This may take some while (most of the time is cost on compiling `cutlass`).

```shell
git submodule update --init --recursive
cd third_party/flash-attn
MAX_JOBS=$(nproc) python setup.py install --verbose
```

## User Guideline

### Setting Generation Mode

Setting `add_new_block_threshold<1.0`, together with our `D2F` training strategy, enables support for the D2F-specific decoding paradigm.

In contrast, setting `add_new_block_threshold=1.0` allows compatibility with Fast-dLLM inference, which is Training-free.

## TODO List

- [x] Implement KV Cache loading kernel
- [x] Tensor Parallel
- [x] Data Parallel
- [ ] Implement Async Engine and Streaming Generation
- [ ] Faster Flash Attention Kernel
- [ ] Diffusion LM CUDA Graph Capturing

## LLaDA-o GUI long-context benchmark

### Unified benchmark runner

Use the unified runner instead of invoking the individual launchers when a
machine-readable run manifest and all result tables are required:

```bash
cd /home/ma-user/work/LLaDA-o/src/Discrete-Diffusion-Forcing

# Show every benchmark and suite.
python D2F-eval/run_gui_benchmarks.py list

# Run one configuration from scratch on GPU 0.
python D2F-eval/run_gui_benchmarks.py run yarn128k-ocr \
  --gpu 0 --limit 100

# Run one comparison suite serially on a single GPU.
python D2F-eval/run_gui_benchmarks.py run deployment \
  --gpu 0 --limit 100

# Run all comparison suites. Identical arms within this invocation are
# executed once, but a later invocation always creates a fresh run.
python D2F-eval/run_gui_benchmarks.py run all \
  --gpu 0 --limit 100

# Resume after an interrupted model or OCR stage. Complete model predictions
# are validated and reused instead of being generated again.
python D2F-eval/run_gui_benchmarks.py resume \
  /home/ma-user/work/LLaDA-o/results/gui-benchmarks/<run-id>
```

The predefined suites are:

| Suite | Arms | Default sample set |
|---|---|---|
| `deployment` | native OCR crop, YaRN 128K + OCR, unscaled 128K + OCR | ordered long-page 16K–64K set |
| `native16k-five-way` | the five full-page configurations documented below | fixed native-16K seed42 set |
| `yarn-isolation` | original 16K versus YaRN on native-resized input | identical ordered long-page IDs |
| `true-long-yarn` | unscaled 128K versus YaRN with sequential positions | identical ordered long-page IDs |
| `tile-size-ablation` | full-page image tiles of 980px, 686px, and 490px with every other setting fixed | identical ordered long-page IDs |
| `kv-retrieval-tile-size-ablation` | packed cached-visual bidirectional Top-4 retrieval with 980px, 686px, and 490px tiles | identical ordered long-page IDs |
| `kv-retrieval-scoring-ablation` | dense context, legacy causal Top-4 retrieval, and bidirectional masked Top-4 retrieval | identical ordered long-page IDs |
| `kv-retrieval-packing-ablation` | sequential versus packed bidirectional masked Top-4 retrieval | identical ordered long-page IDs |
| `kv-retrieval-attention-ablation` | causal versus bidirectional attention over identical masked queries | identical ordered long-page IDs |
| `kv-retrieval-topk-ablation` | bidirectional Top-4 versus Top-8 retained source tiles | identical ordered long-page IDs |
| `kv-retrieval-feedback-ablation` | full joint scoring versus cached-visual bidirectional scoring | identical ordered long-page IDs |
| `kv-retrieval-ocr-prior-ablation` | cached-visual output plus identical shared-path OCR detections, without versus with neural tile-rank fusion | identical ordered long-page IDs |
| `kv-retrieval-optimized-ablation` | full joint versus cached-visual scoring, followed by shared-output OCR tile-rank controls | identical ordered long-page IDs |
| `kv-retrieval-final-ablation` | causal Top-4, bidirectional Top-4/Top-8, packed cached-visual Top-4, and shared-output OCR tile-rank controls without duplicate model arms | identical ordered long-page IDs |

All suite arms run serially on the selected GPU so latency and peak-memory
measurements are not polluted by a competing arm. `--limit` is restricted to
1–100 by repository policy. Use `--benchmark-root` to override the dataset for
a single benchmark or single-dataset suite, and use `--dry-run` to validate
the paths, sample fingerprint, and resolved commands without starting
inference:

```bash
python D2F-eval/run_gui_benchmarks.py run native16k-five-way \
  --gpu 0 --limit 100 --dry-run
```

Every real invocation creates a new directory below
`$ROOT/results/gui-benchmarks/<run-id>` and refuses to reuse it. The directory
contains `run.json`, one log and result directory per arm, and three reports
in Markdown, CSV, and JSON:

```text
tables/quality.{md,csv,json}
tables/performance.{md,csv,json}
tables/protocol.{md,csv,json}
```

The quality table reports raw/final SSR, joint SSR, action F1, parse rate,
and post-hoc target-tile recall for retrieval arms.
The performance table reports full-page tile size, the fixed D2F block size,
convergence steps, synchronized end-to-end latency, model-only latency,
retrieval latency, throughput, resident/dense KV, selected-image count,
input-image count, actual maximum RoPE position, peak memory, and errors. The
protocol table freezes the input mode,
RoPE/KV/OCR settings, Git revision, manifest hash, and ordered sample-ID
SHA-256. It also labels the runtime checkout as `clean` or `dirty` so a
revision is never mistaken for an exact clean checkout. A comparison table is
rejected if its arms do not have the same sample fingerprint.
Regenerate and revalidate the tables of an existing unified run without
performing inference with:

```bash
python D2F-eval/run_gui_benchmarks.py report \
  /home/ma-user/work/LLaDA-o/results/gui-benchmarks/<run-id>
```

Run the controlled full-page tile-size ablation with:

```bash
python D2F-eval/run_gui_benchmarks.py run tile-size-ablation \
  --gpu 0 --limit 100
```

All three arms use the same 100 ordered long-page samples, YaRN weights and
positions, whole-page overview, OCR fusion, prompt instruction, seed, D2F
block size 16, decoding thresholds, dense 65,536-token resident KV, and
disabled KV compression. Only `FULL_PAGE_TILE_SIZE` changes from 980 to 686
to 490 pixels. All three values are exact multiples of the 14px vision patch,
so the experiment does not introduce interior-tile padding. The runtime
prompt derives its image count from the selected tile size instead of reusing
the prepared 980px layout.

To measure tile size on the optimized retrieval path instead of dense context,
run:

```bash
python D2F-eval/run_gui_benchmarks.py run \
  kv-retrieval-tile-size-ablation --gpu 0 --limit 100
```

This suite keeps cached-visual packed bidirectional scoring, Top-4 retention,
the forced overview, YaRN positions, neural tile-rank OCR fusion, D2F block
size 16, the 65,536-token resident-KV capacity, and disabled KV compression
fixed. Only the 14px-aligned full-page tile edge changes from 980 to 686 to
490 pixels.

The validated 100-sample run used ordered sample-ID SHA-256
`8d54d1912ae7ab966bd341df46488c843e54a0f4c16c6a898d8a5bec7d89bc4f`.
All arms completed without inference errors:

| Tile size | Raw SSR | OCR SSR | Δ OCR SSR | Mean latency | Speed vs 980 | P95 latency | Mean / max input images | Mean / max dense prefix | Peak allocated |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 980px | 7% | **75%** | — | 5.861 s | 1.000× | 9.010 s | 11.86 / 21 | 33,483 / 63,057 | 50.907 GiB |
| 686px | **8%** | 74% | -1 pp | **5.837 s** | **1.004×** | 9.436 s | 16.22 / 29 | 33,492 / 63,073 | 50.912 GiB |
| 490px | 4% | 74% | -1 pp | 6.537 s | 0.897× | 10.367 s | 31.96 / 61 | 33,524 / 63,137 | 50.913 GiB |

The 686px arm is only 0.41% faster in mean latency, while its P95 is 4.72%
slower and final SSR is one point lower. Its raw-SSR change is not stable: it
gains five samples and loses four relative to 980px. After OCR fusion it gains
two and loses three. The 490px arm increases mean latency by 11.53%, nearly
2.7× the number of input images, and does not recover the quality point.
Because every source pixel remains present, smaller tiles do not materially
reduce dense KV tokens or peak memory; they mainly add image-boundary and
scheduling overhead. All three fused arms have 100% action F1 and parse rate.
Keep 980px as the default for this configuration.

Run the controlled KV-retrieval scorer ablation with:

```bash
python D2F-eval/run_gui_benchmarks.py run \
  kv-retrieval-scoring-ablation --gpu 0 --limit 100
```

The three arms are executed serially on one GPU and share the same first 100
ordered long-page samples, exact 980px tiles, whole-page overview, operation
instruction query, checkpoint, YaRN factor 8, strided positions, 128K model
limit, 65,536-token resident capacity, D2F decoding parameters, and
prompt-only OCR fusion. Token/head KV compression is disabled. The two
retrieval arms differ only in `KV_RETRIEVAL_SCORE_MODE`: the legacy arm uses
clear-query causal next-token likelihood, while the default arm uses two
complementary masked queries with full bidirectional attention. Ground-truth
boxes are read only by the report step for target-tile recall.

The clean, controlled run on revision `34593f2` used sequential masked
scoring and ordered sample-ID SHA-256
`8d54d1912ae7ab966bd341df46488c843e54a0f4c16c6a898d8a5bec7d89bc4f`.
All three arms completed 100 samples without errors:

| Configuration | Score mode | Target-tile recall | Raw SSR | OCR SSR | Parse | Mean resident / dense KV | KV reduction | Retrieval latency | Mean model latency | Peak allocated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense YaRN + OCR | disabled; all tiles resident | 100% by construction | 7% | **75%** | 100% | 33,483 / 33,483 | 0% | 0.00 s | **5.93 s** | 50.91 GiB |
| Top-4 + overview | legacy causal next-token | 63% | 0% | 71% | 93% | 13,399 / 33,483 | **59.98%** | 3.49 s | 5.91 s | **49.74 GiB** |
| Top-4 + overview | bidirectional masked, 2 rounds | **90%** | **7%** | 74% | **100%** | 15,594 / 33,483 | 53.43% | 5.85 s | 9.03 s | 49.78 GiB |

Action F1 and joint SSR equal 100%/75%, 100%/71%, and 100%/74% for
dense, causal, and masked respectively. Compared with causal scoring, masked
scoring gains 27 target-tile-recall points, 7 raw-SSR points, 3 final-SSR
points, and 7 parse-rate points. The paired audit contains 30 samples found
only by masked retrieval and 3 found only by causal retrieval; final SSR gains
five samples and loses two. Compared with dense context, masked retrieval
loses one final-SSR sample and gains none.

Masked retrieval therefore fixes the causal scorer's relevance failure and
keeps quality within one point of dense context, while reducing mean resident
KV by 53.43%. This table predates packed scoring: the sequential implementation
made mean latency 52.13% higher than dense and 52.63% higher than causal. Peak
allocated memory falls by only
1.13 GiB versus dense because the model and preallocated cache dominate that
measurement. The lower causal convergence-step count is caused by malformed
or early-terminated outputs and must not be interpreted as better decoding.

The complete run manifest, predictions, logs, and Markdown/CSV/JSON tables
are under:

```text
/home/ma-user/work/LLaDA-o/results/gui-benchmarks/kv-retrieval-scoring-ablation-n100-34593f2
```

The complete Markdown, CSV, JSON, logs, manifests, and per-sample predictions
are under:

```text
/home/ma-user/work/LLaDA-o/results/gui-benchmarks/tile-size-ablation-n100-9a34a5f
```

The runtime worktree could not be reset because it contains the other
in-progress runtime changes, so `run.json` correctly labels it `dirty` at
`9e544d1`. The evaluator, unified runner, and launcher used by the run were
byte-identical to pushed commit `9a34a5f`.

The native GUI Non-PD path supports a reproducible 128K YaRN configuration
without forcing the KV cache to reserve the full positional limit:

```bash
MODE=yarn \
MAX_MODEL_LEN=131072 \
KV_CACHE_CAPACITY=65536 \
FULL_PAGE_POSITION_MODE=sequential \
KV_CACHE_COMPRESSION=0 \
bash d2f_vllm/mllm_lladao_gui_long_context.sh
```

`MAX_MODEL_LEN` is the positional limit. `KV_CACHE_CAPACITY` is the maximum
resident sequence length for the current process. The 16K-to-64K benchmark
uses bf16 KV on one A800 and retains the checkpoint's original
`rope_theta=500000`. YaRN uses factor 8 from the original 16,384-position
window. The long-context launcher disables vision KV compression so every
dense prefix token remains resident throughout decoding.

LLaDA-o natively assigns one shared global LLM RoPE position to all tokens
from an image. That native mode does not exercise long RoPE positions even
when the visual KV sequence is tens of thousands of tokens. The benchmark
therefore opts into `FULL_PAGE_POSITION_MODE=sequential`: every visual
boundary and patch token receives a continuous absolute position, the prompt
starts after the complete visual prefix, and generation continues from there.
This mode deliberately differs from the checkpoint's native position packing
and must be reported as an extrapolation experiment.

Full-page Mind2Web screenshots are not resized or target-cropped. They are
split into non-overlapping, row-major 980-pixel image tiles. Each tile receives
independent bidirectional visual prefill, and the grounding prompt attends all
tile KV in a single request. This avoids quadratic attention across unrelated
tiles while preserving the complete page context.

Run the causal true-long YaRN comparison concurrently on two GPUs. Both arms
keep the original full-page pixels as tiles, use sequential positions above
16K, advertise 128K, reserve 65,536 resident KV tokens, and disable KV
compression. The only model change is unscaled RoPE versus YaRN:

```bash
LIMIT=100 nohup bash d2f_vllm/mllm_lladao_gui_yarn_ab.sh \
  > /home/ma-user/work/LLaDA-o/logs/yarn-ab-nohup.log 2>&1 &
```

The default output is isolated below
`results/d2f-vllm-true-long-yarn-isolation-<revision>-n100`. Every prediction
records the dense runtime token count, `position_mode`,
`max_prefill_position`, and `max_generation_position`. The comparison rejects
native positions, samples at or below 16K, compressed KV, mismatched seeds, or
mismatched runtime token counts.

For the validated first 100 samples, every actual generation position was
between 16,619 and 62,464. Unscaled 128K scored 1% SSR and YaRN 128K scored 0%
SSR. This does not isolate a YaRN implementation failure: both arms collapse
after changing the checkpoint-native shared visual position into untrained
sequential positions and asking a single-image GUI checkpoint to localize over
multiple full-page tiles. The YaRN inverse frequencies, attention scale, and
cos/sin cache were separately checked against Transformers 4.57.6 through
position 131,071 and matched exactly.

### Unscaled 128K full-page OCR

To extrapolate the checkpoint's original RoPE directly, without YaRN scaling,
retain only the exact source tiles and run:

```bash
LIMIT=100 GPU=0 KV_CACHE_CAPACITY=65536 \
  bash d2f_vllm/mllm_lladao_gui_unscaled_fullres_no_truncation_ocr.sh
```

This deployment diagnostic sets `rope_scaling=none`, advertises 131,072
positions with the explicit unscaled override, uses `strided` full-page
positions, and disables overview, crop, tile truncation, and KV compression.
On the same first 100 long-page sample IDs as the YaRN overview result, every
source tile was retained and actual generation positions ranged from 16,619
to 62,464. The raw model scored 2% SSR. Prompt-only OCR scored 74% SSR, 74%
joint SSR, 100% action F1, and a 100% parse rate. OCR uses no ground-truth
target location.

### Controlled YaRN isolation

To measure whether static YaRN perturbs inputs that remain inside the
checkpoint-native regime, compare original 16K and YaRN 128K with every
non-RoPE variable held constant:

```bash
LIMIT=100 bash d2f_vllm/mllm_lladao_gui_yarn_isolation_ab.sh
```

Both arms use the same 100 samples, checkpoint-native image resize and
multimodal positions, 16K resident KV capacity, and disabled KV compression.
Only RoPE scaling and the advertised maximum position differ. On revision
`9e544d1`, the controlled result was 26% SSR for original 16K and 22% for YaRN
128K. Both arms had a maximum generation position of 86, so this is a
short-position sensitivity measurement rather than a long-position
extrapolation result. It must not be used to claim that YaRN does or does not
improve positions above 16K.

### Full-page OCR retrieval

Native resize discards small full-page text and reduced the same 100-sample
raw-page benchmark to 26% SSR. The deployable retrieval launcher keeps that
D2F prediction as a fallback and weak location prior, then locates the target
phrase in the original screenshot with tile-wise OCR:

```bash
LIMIT=100 MODE=original \
  bash d2f_vllm/mllm_lladao_gui_ocr_retrieval.sh
```

The retrieval stage reads the visible instruction and screenshot only; it
does not read target boxes or DOM locations. The validated defaults restored
SSR to 79% on those 100 samples while retaining 100% action F1. Use
`MODE=yarn` to apply the identical retrieval stage to the controlled YaRN
arm. To reuse an existing model prediction directory:

```bash
RUN_MODEL=0 \
MODEL_OUTPUT=/path/to/model/predictions \
LIMIT=100 MODE=original \
  bash d2f_vllm/mllm_lladao_gui_ocr_retrieval.sh
```

### Uncropped YaRN 128K with a full-page coordinate anchor

The raw true-long run above has no global coordinate reference: it presents
independent full-resolution tiles to a checkpoint trained on one image and
asks it to return a box normalized to the complete page. The deployable
uncropped launcher keeps every exact tile, appends a checkpoint-native resized
overview of the same complete screenshot as a global coordinate anchor, and
then applies prompt-only OCR retrieval:

```bash
cd /home/ma-user/work/LLaDA-o/src/Discrete-Diffusion-Forcing

LIMIT=100 \
GPU=0 \
RESULT_ROOT=/home/ma-user/work/LLaDA-o/results/yarn128k-uncropped-ocr-n100 \
MODEL_LOG=/home/ma-user/work/LLaDA-o/logs/yarn128k-uncropped-model.log \
OCR_LOG=/home/ma-user/work/LLaDA-o/logs/yarn128k-uncropped-ocr.log \
nohup bash d2f_vllm/mllm_lladao_gui_yarn_uncropped_ocr.sh \
  > /home/ma-user/work/LLaDA-o/logs/yarn128k-uncropped-launcher.log 2>&1 &
```

Monitor the model and OCR stages separately:

```bash
tail -F /home/ma-user/work/LLaDA-o/logs/yarn128k-uncropped-model.log
tail -F /home/ma-user/work/LLaDA-o/logs/yarn128k-uncropped-ocr.log
```

The final predictions and score table are written to:

```text
/home/ma-user/work/LLaDA-o/results/yarn128k-uncropped-ocr-n100/fused/
/home/ma-user/work/LLaDA-o/results/yarn128k-uncropped-ocr-n100/fused/scores/results.csv
```

This protocol does not target-crop or discard any source region. All original
pixels remain in exact, non-overlapping 980-pixel tiles; the additional
overview is a resized duplicate used only to establish the page-wide
coordinate system. The runtime uses YaRN factor 8, a 131,072-position model
limit, `strided` multimodal positions, 65,536 resident KV tokens, and no KV
compression. Do not set resident KV capacity to 131,072 merely to match the
positional limit: the runtime preallocates that cache and requires about
64.5 GiB for KV alone.

On the fixed first 100 Mind2Web full-page samples, actual sequences ranged
from 19,166 to 63,121 tokens and the maximum generation RoPE position was
63,120. The raw model scored 7% SSR. Prompt-only OCR, which reads only the
visible instruction and screenshot and never reads the target box or DOM
location, produced 75% SSR, 75% joint SSR, 100% action F1, and a 100% parse
rate. The native D2F plus OCR retrieval-crop pipeline below scored 80% SSR on
the identical sample IDs. These are full-page deployment diagnostics, not
paper-comparable cropped Mind2Web results.

### Whole-image KV retrieval without KV compression

The retrieval variant starts from the uncropped YaRN protocol above. It strips
the deterministic full-page transport wrapper with `native_resize_prompt()`,
then scores each exact source tile independently with native dLLM masked
self-information over only the public operation instruction (for example,
`Click on Quick Tools.`). The default two complementary corruptions mask every
non-boundary query token exactly once. Each candidate image and corrupted
query are evaluated together with full bidirectional attention, and CE is
read only from the original positions of the masked tokens. This avoids both
next-token causal scoring and target leakage from a visible answer token. The
runtime packs independent candidate/round documents into FlashAttention
varlen batches by default. `cu_seqlens` keeps their attention domains
disjoint, so batching changes the number of model calls without changing the
bidirectional scoring formula. The runtime retains the top four complete
image spans and force-keeps the resized whole-page overview. Selection never
drops individual patch tokens, layers, or KV heads, so this mode is mutually
exclusive with `KV_CACHE_COMPRESSION=1`.

`KV_RETRIEVAL_PACKED_SCORING=1` and
`KV_RETRIEVAL_MAX_BATCH_TOKENS=65536` are the defaults. The token budget is a
soft cap over the independent documents in one forward; a single larger
document is still admitted after the normal per-document context check. Lower
the budget to reduce transient activation memory, or set
`KV_RETRIEVAL_PACKED_SCORING=0` to reproduce the sequential implementation
for score/selection equivalence tests. If the FlashAttention varlen operator
is unavailable, the runtime safely falls back to sequential scoring and
reports `kv_cache_retrieval_packed_scoring=false`.

Run the controlled latency/equivalence comparison with:

```bash
python D2F-eval/run_gui_benchmarks.py run \
  kv-retrieval-packing-ablation --gpu 0 --limit 100
```

Both arms use the same revision and ordered sample fingerprint. The runner
changes only `KV_RETRIEVAL_PACKED_SCORING`, executes the arms serially, and
writes quality, latency, scoring-batch, memory, and protocol tables.

For controlled regression experiments only,
`KV_RETRIEVAL_SCORE_MODE=causal_self_information` restores the retired
clear-query causal next-token proxy. It is not the default and should not be
used as the dLLM retrieval implementation. Use the unified
`kv-retrieval-scoring-ablation` suite so both scoring arms receive identical
runtime inputs.

The model input and two-round scoring input are conceptually:

```text
generation: [image tiles] [overview] [BOS] wrapper + question + format [EOS]
retrieval 1: [one candidate image] [BOS] [MASK] on [MASK] Tools [MASK] [EOS]
retrieval 2: [one candidate image] [BOS] Click [MASK] Quick [MASK] . [EOS]
```

The exact subword masks depend on tokenization. BOS and EOS remain visible,
all other query tokens are scored once, and no OCR text or target annotation
is inserted into the retrieval query.

```bash
cd /home/ma-user/work/LLaDA-o/src/Discrete-Diffusion-Forcing

LIMIT=100 \
GPU=0 \
KV_RETRIEVAL_TOPK_IMAGES=4 \
KV_RETRIEVAL_MASK_ROUNDS=2 \
KV_RETRIEVAL_PACKED_SCORING=1 \
KV_RETRIEVAL_MAX_BATCH_TOKENS=65536 \
RESULT_ROOT=/home/ma-user/work/LLaDA-o/results/yarn128k-uncropped-kvretrieve4-masked2-packed65536-ocr-n100 \
MODEL_LOG=/home/ma-user/work/LLaDA-o/logs/yarn128k-uncropped-kvretrieve4-masked2-packed65536-model.log \
OCR_LOG=/home/ma-user/work/LLaDA-o/logs/yarn128k-uncropped-kvretrieve4-masked2-packed65536-ocr.log \
nohup bash d2f_vllm/mllm_lladao_gui_yarn_uncropped_kv_retrieval_ocr.sh \
  > /home/ma-user/work/LLaDA-o/logs/yarn128k-uncropped-kvretrieve4-masked2-packed65536-launcher.log 2>&1 &
```

Monitor and inspect it with:

```bash
tail -F /home/ma-user/work/LLaDA-o/logs/yarn128k-uncropped-kvretrieve4-masked2-packed65536-model.log

python - <<'PY'
import json
from pathlib import Path

path = Path(
    "/home/ma-user/work/LLaDA-o/results/"
    "yarn128k-uncropped-kvretrieve4-masked2-packed65536-ocr-n100/model/"
    "mind2web_fullpage/part-00000.jsonl"
)
record = json.loads(path.read_text().splitlines()[0])
for key in (
    "kv_cache_retrieval_query",
    "kv_cache_retrieval_query_tokens",
    "kv_cache_retrieval_score_mode",
    "kv_cache_retrieval_mask_rounds",
    "kv_cache_retrieval_packed_scoring",
    "kv_cache_retrieval_score_batches",
    "kv_cache_retrieval_max_batch_tokens",
    "kv_cache_retrieval_indices",
    "kv_cache_retrieval_ratio",
    "kv_cache_retrieval_seconds",
    "kv_cache_compression_ratio",
):
    print(key, record[key])
PY
```

The retrieval ratio measures selected whole-span KV relative to the dense
full-page prefix. It is not the compression ratio. A valid run must report
`kv_cache_retrieval_enabled=true`, the requested source-tile Top-K plus the
forced overview,
`kv_cache_retrieval_score_mode="masked_self_information"`,
`kv_cache_retrieval_mask_rounds=2`,
`kv_cache_retrieval_packed_scoring=true`, a positive
`kv_cache_retrieval_score_batches`, `kv_cache_compression_ratio=1.0`, and
`kv_cache_compression_seconds=0.0`.

The following table is retained as a historical causal-scoring baseline. The
run on revision `b877dda` predates bidirectional masked scoring; its quality
and latency numbers must not be reported as results of the current method.
It used the same ordered 100 sample IDs, dense prefix lengths, input-image
counts, and generation positions as the uncropped no-retrieval YaRN run:

| Configuration | Retrieval query | Target-tile hit | Raw SSR | Final SSR | Joint SSR | Action F1 | Parse | Mean resident / dense prefix | Max RoPE | Mean model latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YaRN 128K + OCR, all original tiles + overview | None; all tiles resident | 100% | 7% | 75% | 75% | 100% | 100% | 33,483 / 33,483 (100%) | 63,120 | 13.36 s |
| Legacy image Top-4 + overview | Complete grounding prompt | 36% | 0% | 71% | 71% | 100% | 93% | 11,253 / 33,483 (33.61%) | 63,120 | 5.50 s |
| Legacy causal image Top-4 + overview | Operation instruction only | 63% | 0% | 71% | 71% | 100% | 93% | 13,399 / 33,483 (40.02%) | 63,120 | 5.89 s |

The legacy operation-only row reduced the token-weighted resident
visual/prompt KV working set by 59.98%, while losing four final SSR points.
Its 5.89-second number covers model preprocessing, causal retrieval, cache
construction, and generation; it does not include the subsequent OCR fusion
stage. Across all 100 historical records, exactly four source tiles plus one
overview were retained, compression ratio stayed at 1.0, compression time
stayed at zero, and mean causal retrieval scoring time was 3.45 seconds. The
target-tile hit rate is a post-hoc diagnostic computed from ground-truth boxes
after inference; no target box, DOM field, or provenance description enters
the query.

### Five-way comparison on 100 native-16K full-page samples

The long-page results above use a set selected specifically above 16K. For a
comparison that starts with complete original-resolution pages inside the
checkpoint's native limit, rebuild the source set with an inclusive 16,384
token ceiling and freeze a deterministic random subset:

```bash
ROOT=/home/ma-user/work/LLaDA-o
LLADAO=$ROOT/src/LLaDA-o
PYTHON=$ROOT/env/bin/python

cd "$LLADAO"
"$PYTHON" scripts/data/prepare_gui_grounding_benchmarks.py build-fullpage \
  --root "$ROOT/data/mind2web-fullpage-native16k" \
  --raw-root "$ROOT/data/bench_raw/raw/mind2web" \
  --tokenizer "$ROOT/models/lladao-gui-d2f-vllm-step1377-exact" \
  --min-total-tokens 0 \
  --max-total-tokens 16384

"$PYTHON" scripts/data/select_gui_grounding_subset.py \
  --source-root "$ROOT/data/mind2web-fullpage-native16k" \
  --output-root "$ROOT/data/mind2web-fullpage-native16k-n100-seed42" \
  --count 100 \
  --seed 42
```

The selected full-resolution sequences contain 7,319–16,252 tokens, with 67
domain, 18 task, and 15 website samples. The selected-ID SHA-256 is
`273ce580bbb036230088d4aab84dd34fdc01f0c3e312ed7c7bfa47a2a55e8e9f`.
Run all five arms against that exact benchmark root:

```bash
ROOT=/home/ma-user/work/LLaDA-o
REPO=$ROOT/src/Discrete-Diffusion-Forcing
BENCHMARK_ROOT=$ROOT/data/mind2web-fullpage-native16k-n100-seed42
COMPARE=$ROOT/results/compare-native16k-seed42

cd "$REPO"

# Native D2F, full-page OCR retrieval, and OCR-selected 980px crop.
LIMIT=100 GPU=0 \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
RESULT_ROOT="$COMPARE/native-ocr-crop" \
  bash d2f_vllm/mllm_lladao_gui_ocr_crop_pipeline.sh

# YaRN 128K, all exact tiles plus a resized full-page coordinate overview.
LIMIT=100 GPU=0 KV_CACHE_CAPACITY=32768 \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
RESULT_ROOT="$COMPARE/yarn128k-uncropped" \
  bash d2f_vllm/mllm_lladao_gui_yarn_uncropped_ocr.sh

# YaRN 128K: exact original tiles, no overview, crop, or truncation.
LIMIT=100 GPU=0 KV_CACHE_CAPACITY=32768 \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
RESULT_ROOT="$COMPARE/yarn128k-fullres-no-truncation" \
  bash d2f_vllm/mllm_lladao_gui_yarn_fullres_no_truncation_ocr.sh

# No YaRN: the same strided original tiles, no overview, crop, or truncation.
LIMIT=100 GPU=0 \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
RESULT_ROOT="$COMPARE/native16k-fullres-no-truncation" \
  bash d2f_vllm/mllm_lladao_gui_native_fullres_no_truncation_ocr.sh

# No YaRN: exact full-resolution tiles, native 16K positions, tail truncation.
LIMIT=100 GPU=0 \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
RESULT_ROOT="$COMPARE/native16k-fullres-truncated" \
  bash d2f_vllm/mllm_lladao_gui_native_fullres_truncated_ocr.sh
```

All arms disable KV compression and use the same 100 sample IDs:

| Configuration | Raw SSR | Final SSR | Joint SSR | Action F1 | Parse | Runtime tokens | Max RoPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native D2F + OCR retrieval crop | 70% | **72%** | 72% | 100% | 100% | 2,663–4,972 | 83 |
| YaRN 128K + OCR, no target crop | 44% | **66%** | 66% | 100% | 99% | 11,475–18,868 | 18,867 |
| Native 16K + OCR, exact strided tiles, no truncation | 0% | **65%** | 65% | 100% | 100% | 7,319–16,252 | 16,251 |
| YaRN 128K + OCR, exact original tiles only | 1% | **64%** | 64% | 100% | 98% | 7,319–16,252 | 16,251 |
| Native 16K + OCR, full-resolution tiles | 2% | **63%** | 63% | 100% | 100% | 7,319–16,252 | 152 |

The native-16K full-resolution launcher reserves prompt and 64 generation
tokens, then drops only complete trailing tiles if an input exceeds capacity;
it never cuts through one image's visual tokens. No selected sample triggered
that fallback (`truncated_images=0` for all 100). A separate overlength smoke
test reduced 12 source tiles to four complete tiles, produced a 13,445-token
sequence, and kept the maximum generation position at 140 with no RoPE
scaling.

The final columns include prompt-only OCR. The first YaRN arm adds a resized
whole-page overview. The two exact-strided arms both use only the exact source
tiles, `full_page_overview=false`, `truncate_full_page_tiles=false`, and a
dense KV ratio of 1.0. They consumed all four to six source tiles for every
sample, with zero truncated tiles and no errors. The corrected native arm uses
`rope_scaling=none`, a 16,384 model limit, and a 16,384-token KV capacity; the
earlier YaRN arm uses factor-8 YaRN, a 131,072 model limit, and a 32,768-token
KV capacity. Native and YaRN scored 65% and 64% after OCR, respectively.
Their maximum position was 16,251, still inside the original 16,384-position
window, so neither arm is a long-RoPE extrapolation result. Because their KV
capacities also differ, this table records deployment configurations and does
not attribute the one-point difference to YaRN. Use the controlled YaRN
isolation launcher above for a causal static-YaRN comparison. OCR and crop
selection use no ground-truth target location.

For controls whose visible label is adjacent to rather than inside the
clickable box, run the complete two-stage retrieval-crop pipeline:

```bash
LIMIT=100 \
  bash d2f_vllm/mllm_lladao_gui_ocr_crop_pipeline.sh
```

The pipeline runs native D2F once to provide a weak duplicate-text prior,
performs tile-wise OCR on the original screenshot, runs native D2F again on a
980-pixel retrieval crop, and maps the selected local box back into the
original full-page coordinate system. Crop selection uses only the visible
instruction and OCR output; ground-truth locations are used only by the
scorer. On the controlled 100-sample full-page set this produced 80% SSR,
80% joint SSR, 100% action F1, and a 100% parse rate, compared with 0% SSR for
the unadapted true-long YaRN arm.

All intermediate artifacts are retained below the result root:
`native-model`, `ocr-retrieval`, `retrieval-crop-benchmark`,
`retrieval-crop-model`, and `fused`. Set `RUN_MODEL=0` and `MODEL_OUTPUT` to
reuse an existing native prediction directory.
