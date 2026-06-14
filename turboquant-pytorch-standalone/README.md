# TurboQuant — standalone PyTorch library (the one that actually works here)

This documents running **[tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)**
on the RTX 3080 (sm86) / WSL2 box — a clean-room PyTorch implementation of the TurboQuant
KV-cache compression paper (ICLR 2026).

## Why this is separate from the rest of the repo

There are **two different things** both called "TurboQuant":

1. **`turboquant-plus-vllm` 0.13.7** — a vLLM *integration* (3-bit weights + 4-bit KV in a
   running vLLM server). This is what the top-level repo fights with. It *serves* but at
   **~0.1 tok/s**, because only its per-token PyTorch reference path runs (the fused CUDA kernel
   is blocked by a `bit_width` constraint + vLLM-version mismatch). See top-level README.

2. **`turboquant-pytorch` (this dir)** — a standalone *library/research* tool. Not a server, no
   OpenAI API. It just implements the compression and tests its quality. **It works cleanly and
   fast on this exact hardware** — no UVA hacks, no CUDA-graph errors.

> The 0.1 tok/s disaster was the **vLLM glue, not the TurboQuant algorithm.**

## Setup (fresh env)

```bash
git clone https://github.com/tonbistudio/turboquant-pytorch
python3 -m venv ~/tq_pytorch_env
~/tq_pytorch_env/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
~/tq_pytorch_env/bin/pip install -e ./turboquant-pytorch
```

Pinned env: [`requirements-turboquant-pytorch.txt`](requirements-turboquant-pytorch.txt)
(torch 2.12.0+cu126, transformers 5.12, bitsandbytes 0.49.2). Upstream commit in `UPSTREAM.txt`.

## Run

```bash
# synthetic validation (no model download) -- output saved here
python -m turboquant.test_turboquant

# real-model needle-in-haystack (downloads Qwen2.5-3B-Instruct ~6GB)
python -m turboquant.generation_test
python -m turboquant.validate_v3
```

## Results on RTX 3080 (sm86), synthetic test

Full output in [`test_turboquant_output.txt`](test_turboquant_output.txt). All 7 tests pass:

| Test | Result |
|---|---|
| Lloyd-Max codebook | PASSED (symmetric, distortion within bounds) |
| MSE distortion vs theory | within bound at 1–4 bit |
| Inner-product unbiasedness (QJL) | corr 0.80 → 0.97 (2→4 bit) |
| KV-cache compression | **7.76× (2-bit), 5.22× (3-bit), 3.94× (4-bit)** |
| Needle-in-haystack | **EXACT at every bit-width, 512/2048/8192 ctx** |
| GPU benchmark | quantize 8192 keys 24.7 ms; **5.3× compression** |

Note: compressed inner-product (~14.5 ms) is far slower than a raw fp16 matmul (~0.03 ms) — the
algorithm trades **compute for memory**. That's why production use needs a *fused* kernel, and why
the naive per-token vLLM integration is slow. The library itself is correct and the win is VRAM.
