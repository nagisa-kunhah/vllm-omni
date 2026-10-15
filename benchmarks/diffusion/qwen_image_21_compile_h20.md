# Qwen-Image-2.1 GEMM autotuning on H20

Measured on 2026-10-14 for issue #8586, item A2. This is an opt-in compilation
recipe and reproducible benchmark; serving defaults are unchanged.

Implementation follow-up: the runner now exposes
`diffusion_compile_gemm_autotune` / `--diffusion-compile-gemm-autotune` and the
benchmark's tuned mode uses that scoped option. The measurements below predate
this integration and used environment-based tuning. A GPU rerun of the new
entry point was attempted, but the current host reports no GPU devices;
these historical measurements must not be presented as validation of the new
configuration entry point.

## Controls

- Source: `76f52f1ae59078cecf47759a56d73e8bcd2ba311`.
- Checkpoint: `Qwen/Qwen-Image-2.1`, revision
  `d26bb61231c349cf6b7896fa83353113880e1ba3`; all seven weight shards SHA-256 verified.
- One NVIDIA H20, 97871 MiB reported VRAM, driver 580.76.05.
- Torch 2.13.0+cu130, vLLM 0.31.0, Transformers 5.14.1, Diffusers 0.40.0,
  fa3-fwd 0.0.3. The installed Omni version string is stale; source SHA above is authoritative.
- BF16, batch 1, 50 steps, seed 42, `true_cfg_scale=1.0`, native FA3,
  regional compilation with dynamic shapes, explicit decode CUDA Graphs.
- Candidate changes only GEMM autotuning, retaining ATen and Triton candidates.
  No step reduction, quantization, approximate attention, or weight change.
- Each mode/case: one first request and three steady requests. One local
  in-process worker; text encoding, denoising and VAE decoding included.
  GPU synchronization brackets timing. CPU output transfer, PNG encoding and
  HTTP serving are excluded. Stage-boundary profiling is common to both modes.
- Short1024 runs first in its own process; long1024, edit1024, short2048 then
  run sequentially in a second process, accumulating graph entries. Same case
  order in both modes. Editing uses the baseline short1024 PNG converted to RGB.

## Full-model results

All values are seconds per image; reduction is relative to the default median.

| Workload | Default median (range) | Tuned median (range) | Reduction | Peak allocated GiB, both |
|---|---:|---:|---:|---:|
|1024 short prompt|25.3403 (25.3110–25.5184)|24.8605 (24.8565–24.9055)|1.893%|36.891|
|1024 long prompt|26.2549 (26.2512–26.2557)|25.6158 (25.6037–25.6433)|2.434%|37.240|
|1024 editing|29.6274 (29.5831–29.6314)|29.0448 (29.0048–29.0538)|1.966%|39.293|
|2048 short prompt|136.4768 (136.4758–136.4780)|134.7924 (134.7765–134.8021)|1.234%|59.016|

Raw decoded tensors and RGBA PNGs are bitwise identical across modes for all
four inputs. Repeated requests are bitwise identical within each mode. All
outputs are finite. This exceeds the planned PSNR >=40 dB and SSIM >=0.99 gates,
including each individual RGBA channel. It does not establish equivalence for
all inputs. The editing baseline itself is visibly oversaturated; this result
only establishes absence of an additional change from this optimization.

Every request records 49 successful graph decode calls and zero fallbacks.
Each workload captures once, with no steady recapture. Peak allocated memory
is unchanged; maximum reserved memory over the workload sequence is 67.643 GiB
in both modes. Memory values include previously retained graph entries.

After all tuned measurements, the reusable repository benchmark reran default
short1024: median 25.4611 s (25.4512–25.4624), identical output. Tuned short1024
is 2.359% faster than this reversed-order baseline. The primary table retains
its original matched baseline. Three samples are descriptive, not a confidence
interval, and the small variation argues against claiming more precision.

## Startup tradeoff

| Workload | Default first request | Tuned first request |
|---|---:|---:|
|1024 short|38.0915|49.2779|
|1024 long|35.2644|34.4451|
|1024 editing|35.1504|33.9686|
|2048 short|141.9336|140.1755|

Short1024 has 11.666 s extra first-minus-steady overhead, corresponding to about
25 subsequent requests at the measured saving. These runs share a persistent
compiler cache with component experiments; they are **not cold-cache compile
measurements**. Later workloads reuse compiled code and autotuning decisions.
Their faster first requests do not mean autotuning has no compilation cost.
Graph capture wrapper times include warmup and lazy compilation.

## Why this candidate

Native transformer-block CUDA Graph microbenchmarks show 2.61%, 2.51%, 2.16%
and 1.25% reductions for the four actual shapes, using shared random parameters.
A short1024 trace has 17 kernels in both modes. FA3 remains approximately
2.12 ms; compiled GEMM/template fusion accounts for the improvement. Random
block outputs have small rounding differences at1024, so full-model output
checks were retained even though the measured full-model outputs match exactly.

Attention-only trials did not justify a backend change. Default FA3 medians
were 2.211/2.558/4.226/31.502 ms. Splits2 and4 regressed. cuDNN was roughly tied
and inconsistent across shapes; PyTorch FLASH was about50–58% slower. Changing
cuBLAS versus cuBLASLt preference did not provide a meaningful gain.

Nsight Compute failed with `ERR_NVGPUCTRPERM`; no occupancy or hardware
bandwidth claim is made. Latencies use CUDA events and full-model wall time;
Torch CUDA traces provide kernel attribution.

## Reproduce

Use the same local checkpoint or pass `--revision` when loading from the Hub.
Choose a persistent artifact/cache location with sufficient free space. Run
both modes in fresh processes on an otherwise idle GPU:

```bash
python benchmarks/diffusion/qwen_image_21_compile.py \
  --model /path/to/Qwen-Image-2.1 --output-dir /path/to/results --mode baseline
python benchmarks/diffusion/qwen_image_21_compile.py \
  --model /path/to/Qwen-Image-2.1 --output-dir /path/to/results --mode tune
```

The benchmark records exact prompts and reference hashes. `--cases` can split
the run into short1024 first, followed by long1024/edit1024/short2048, matching
the process boundaries above. The full measured matrix used the precursor
artifact harness with the same inference/timing operations; the final portable
script was also exercised on the reversed-order short1024 baseline.

JSONL request records, per-case summaries, graph counters and raw/PNG outputs
are written under the output directory. Compare outputs across modes; hashes
make bitwise equality inspectable. See the regional compilation guide for the
opt-in environment variables and stage `env` configuration for Ray workers.

The local experiment bundle is `profile/qwen21-a2-perf-20261014/`, with traces,
raw samples, environment records and `analysis/E2E_COMPARISON.md`. This report
contains the reviewable measurements even when that machine-local bundle is
not present in another checkout. No issue comment or PR was published.
