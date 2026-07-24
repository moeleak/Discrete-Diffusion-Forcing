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
