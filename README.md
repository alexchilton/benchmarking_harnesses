# LLM Inference Engine Benchmark — vLLM vs SGLang vs TurboQuant

Benchmarking **Qwen3-4B** across three inference engines on an **NVIDIA RTX 3080 Laptop
GPU (Ampere, sm86, 16 GB)** running inside **WSL2 Ubuntu**. All engines serve an
OpenAI-compatible API on port `8010`; `benchmark.py` drives whichever one is up.

> The hard part here was not the benchmark — it was getting each engine to run on
> this specific hardware/OS combo (Ampere sm86 + WSL2 + CUDA 12.6). This repo records
> every fix so the setup is reproducible.

## TL;DR status

| Engine | Runs? | What it took |
|---|---|---|
| **vLLM 0.23.0** | ✅ | `/usr/local/cuda` → 12.6, `ninja`+`nvcc` on system PATH, `--served-model-name default` |
| **SGLang 0.5.2** | ✅ | Rolled back from 0.5.13: 0.5.13's `sglang-kernel 0.4.3` ships sm90/sm100 only (no Ampere). 0.5.2 + `sgl-kernel 0.3.9.post2` (has sm_80, runs on sm86) + torch 2.8.0 |
| **TurboQuant (turboquant-plus-vllm 0.13.7)** | ⚠️ serves, but ~0.1 tok/s | torch 2.11 + torchvision 0.26 ABI match, launcher fixes, **WSL UVA-gate override**, KV-layout patch (`unbind(0)`→`unbind(1)`), `--enforce-eager`, tiny warmup batch. Only the slow **PyTorch** compression path runs; CUDA fast-path blocked (see below). |

## Results (Qwen3-4B, RTX 3080 sm86, port 8010)

Throughput (cool-GPU / least-throttled runs) and steady-state VRAM:

| Metric | vLLM 0.23 | SGLang 0.5.2 | TurboQuant |
|---|---|---|---|
| Steady-state VRAM | 15,560 MiB | 14,397 MiB | 14,507 MiB |
| basic_long (300 tok) tok/s | 42.2 | 40.0 | ~0.1 |
| rag_style (100 tok) tok/s | 41.4 | 38.9 | ~0.1 |
| tool_call | needs `--tool-call-parser`; ~36 | native, 36.5 | (not run) |
| basic_short median latency | 0.25 s | 0.27 s | very slow |

- **vLLM vs SGLang:** throughput is ~tied when cool (~40 tok/s); SGLang uses ~1.2 GB less VRAM
  and handles tool-calling natively. Latency degrades badly mid-run on both from **thermal
  throttling** (see caveat) — the VRAM column is the trustworthy axis.
- **TurboQuant:** runs but at **~0.1 tok/s** (≈400× slower) because only the PyTorch reference
  compression path is available here. This benchmark measures the *cost* of compression, not its
  *benefit* (memory) — for that, see the `--kv-capacity` probe. Raw JSON in [`results/`](results/).

## How TurboQuant is meant to be deployed

Getting it to *run* here proves the integration, but this is **not** its intended deployment:

1. **Native Linux, not WSL** — so UVA is genuinely available (no gate override needed).
2. **Its fused CUDA compression kernel, not the PyTorch reference path** — the Python per-token
   path runs the compressor on every decode step (~0.1 tok/s). The CUDA kernel does it on-device
   with negligible overhead. Engage it with a kernel build matching the configured bit-widths
   (`norm_correction=False` + ≤4-bit everywhere; default 8-bit boundary layers hit `bit_width must be 1-4`).
3. **The vLLM version TurboQuant was validated against** — 0.13.7 assumes the old KV-cache layout,
   so on vLLM 0.23 it needs the `unbind` patch here.

TurboQuant is a **memory** play (3-bit weights + 4-bit KV → bigger models / longer context / more
concurrency in fixed VRAM). On the fused CUDA kernel + native Linux, throughput is near-native and
the win shows up as capacity, not speed.

## Environment

- Host: Windows 11 + WSL2 Ubuntu 24.04, GPU RTX 3080 Laptop (sm86, 16 GB, ~115 W cap).
- Three separate venvs under `~`: `vllm_env`, `sglang2_env`, `turboquant_env` (Python 3.12).
- Model: `Qwen/Qwen3-4B` (bf16, ~7.5 GB weights). Served as `default` on port 8010.
- **CUDA 12.6** throughout. The whole stack is `+cu126`; do **not** use CUDA 13 (see below).

Pinned package sets are in [`requirements/`](requirements/) (full `pip freeze` of each env).
Key versions: vLLM `0.23.0` / torch `2.11.0+cu126`; SGLang `0.5.2` / `sgl-kernel 0.3.9.post2`
/ torch `2.8.0+cu126`; TurboQuant `turboquant-plus-vllm 0.13.7` on vLLM `0.23.0` / torch `2.11.0+cu126`.

## Fixes, by engine

### Common
- **CUDA toolkit:** a prior CUDA-13 install had repointed `/usr/local/cuda` at a runtime-only
  cuda-13.0 (no `nvcc`), breaking flashinfer JIT. Restore: `sudo ln -sfn /usr/local/cuda-12.6 /etc/alternatives/cuda`.
- **`ninja` not found at warmup:** vLLM's spawned EngineCore subprocess doesn't inherit a venv
  PATH. Put both tools on the system PATH: `sudo ln -sf <env>/bin/ninja /usr/local/bin/ninja`
  (and `nvcc` via the cuda symlink above).
- **404 on every request:** `benchmark.py` sends `"model": "default"`, so each server must be
  launched with `--served-model-name default`.
- The `libnvrtc.so.13: cannot open shared object file` lines are **harmless warnings** (optional
  cu13 probes); the stack is cu126.

### SGLang
SGLang 0.5.13 is a bleeding-edge CUDA-13 build whose `sglang-kernel 0.4.3` ships `common_ops`
for **sm90/sm100 only** — it dropped Ampere, so it can't load on sm86. Even PyPI `sgl-kernel 0.3.21`
is sm90/sm100-only. **`sgl-kernel 0.3.9.post2`** (single fat `common_ops.abi3.so`) contains
**sm_80/89/90/100/120** SASS, and sm_80 cubins are binary-forward-compatible to sm86. So:
install **`sglang[srt]==0.5.2`** (which pins `sgl-kernel 0.3.9.post2` + torch 2.8.0) in a fresh env.
Fallback if FlashInfer misbehaves: `--attention-backend triton --sampling-backend pytorch`.

### TurboQuant (the deep one)
`turboquant-plus-vllm` patches vLLM to do 3-bit weight quant + 4-bit KV-cache compression.
Layered fixes (each a distinct root cause):
1. **torch ABI:** env had torch 2.12 but vLLM 0.23's `_C.abi3.so` needs torch 2.11 → `pip install torch==2.11.0+cu126`.
2. **torchvision::nms missing:** align torchvision to torch 2.11 → `torchvision==0.26.0+cu126`.
3. **Launcher import:** `FlexibleArgumentParser` moved to `vllm.utils.argparse_utils`.
4. **multiprocessing spawn guard:** wrap server start in `if __name__ == "__main__":`, but keep
   the TurboQuant + UVA patches at module top level so the spawned EngineCore worker also applies them.
5. **UVA gate (the blocker):** TurboQuant needs vLLM's V2 model runner, which requires UVA.
   vLLM's `is_uva_available()` returns False **only because it blanket-disables pinned memory on WSL**
   — but UVA/mapped-pinned memory actually works here (verified: host write visible on device view).
   `tq_launcher.py` overrides `is_uva_available -> True` at import (in both processes).
6. **KV-cache layout:** TurboQuant assumed `(2, num_blocks, …)`; vLLM 0.23 uses
   `(num_blocks, 2, block_size, …)`. Patch `unbind(0)` → `unbind(1)` — see
   [`patches/turboquant_vllm_patch.diff`](patches/turboquant_vllm_patch.diff).
7. **CUDA graphs:** TurboQuant's per-token Python compression does host-side `.item()`/dict ops,
   illegal during CUDA-graph capture → run with **`--enforce-eager`**.

Note: TurboQuant runs the **PyTorch compression path** (slow) because `norm_correction=True`;
its CUDA kernel path needs `norm_correction=False`. Eager + per-token Python compression makes
warmup slow.

## Reproduce

```bash
# CUDA + build tools (once)
sudo ln -sfn /usr/local/cuda-12.6 /etc/alternatives/cuda
sudo ln -sf ~/vllm_env/bin/ninja /usr/local/bin/ninja

# pick an engine, start it, then benchmark
bash scripts/start_vllm.sh        # or start_sglang2.sh / start_turboquant.sh
python3 benchmark.py --label vllm # adds peak-VRAM per test; --kv-capacity for OOM probe
```

Only one server may hold the GPU at a time. SGLang spawns a tree of inductor compile-workers
that survive a naive `pkill`; kill by the renamed procs (`sglang::scheduler` etc.) to free VRAM.

## Benchmark tool

`benchmark.py` measures latency, tokens/sec, **and GPU memory** (steady-state + peak VRAM per
test), with an optional `--kv-capacity` probe (largest `max_tokens` before OOM) — the test where
a KV/weight compressor like TurboQuant should actually win, since the speed tests only show its
*cost* (dequant overhead), not its memory *benefit*. Raw results in [`results/`](results/).

## Caveat: thermal throttling

This is a power-limited laptop GPU (~115 W, ~70 °C idle). `nvidia-smi` shows
`SW Thermal Slowdown: Active` under sustained load; tok/s collapses run-over-run (e.g. multi_turn
40 → 5 tok/s). Cross-engine *speed* comparisons need cooldowns and are noisy; the VRAM figures are
the trustworthy axis.
