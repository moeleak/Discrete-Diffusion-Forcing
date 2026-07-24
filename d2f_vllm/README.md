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

Run the unscaled/YaRN comparison concurrently on two GPUs:

```bash
nohup bash d2f_vllm/mllm_lladao_gui_yarn_ab.sh \
  > /home/ma-user/work/LLaDA-o/logs/yarn-ab-nohup.log 2>&1 &
```

The launcher is append-only and resumable. Its default output is isolated from
native/compressed runs below
`results/d2f-vllm-fullpage-sequential-nocompress-*`. Every prediction records
`position_mode`, `max_prefill_position`, and `max_generation_position` so the
actual RoPE range can be audited.
