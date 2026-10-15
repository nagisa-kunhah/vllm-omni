# Regional Compilation

Regional compilation applies `torch.compile` to the repeated transformer blocks
declared by a diffusion model. It is the default compilation scope when
diffusion inference runs without `--enforce-eager`.

## Configuration

Dynamic compilation is enabled by default so the compiled regions can handle
mixed resolutions. For a fixed-shape workload, disable it explicitly:

```bash
vllm serve <model> --omni --no-diffusion-compile-dynamic
```

The equivalent per-stage deploy configuration is:

```yaml
stages:
  - stage_id: 0
    diffusion_compile_granularity: regional
    diffusion_compile_dynamic: false
```

For an experimental whole-transformer compile scope, set
`--diffusion-compile-granularity full` or use
`diffusion_compile_granularity: full` in the deploy configuration. Full scope may
still contain graph breaks; it does not force one graph. It is rejected when
HSDP, sequence parallelism, CPU offload, or layerwise offload is enabled. Use
regional scope with those features.

These settings control the generic model-runner compilation path. Pipelines
that provide their own `setup_compile()` implementation manage their compilation
policy independently. Compilation is lazy, so backend or graph errors can first
surface on the initial request.

## GEMM autotuning

Enable GEMM autotuning for the generic diffusion compilation path with:

```bash
vllm serve Qwen/Qwen-Image-2.1 --omni --diffusion-compile-gemm-autotune
```

The equivalent per-stage deploy configuration is:

```yaml
stages:
  - stage_id: 0
    diffusion_compile_gemm_autotune: true
```

Python callers can pass `diffusion_compile_gemm_autotune=True` to `Omni` or
`OmniDiffusionConfig`. The option defaults to false. Disable it explicitly with
`--no-diffusion-compile-gemm-autotune` when overriding a stage configuration.

When enabled, the model runner passes `max_autotune_gemm=True` and
`max_autotune_gemm_backends="ATEN,TRITON"` to the transformer's `torch.compile`.
These options apply to that compilation without changing process-wide Inductor
settings. Regional and full compilation scopes both support the option. Dynamic
shapes remain controlled independently. Existing explicit decode CUDA Graph
settings are preserved, including disabling Inductor's own graph layer.

Autotuning benchmarks GEMM candidates during compilation, so measure first-request
and steady latency separately. Compare decoded outputs because changing GEMM
implementations or fusion can change floating-point rounding. The option follows
the generic compile lifecycle: `--enforce-eager` bypasses it, and pipelines with
their own `setup_compile()` implementation manage compilation independently.
When disabled, the runner leaves GEMM options unset and existing external
Inductor environment settings still apply.

For a repeatable Qwen-Image-2.1 comparison, use
[`qwen_image_21_compile.py`](../../../benchmarks/diffusion/qwen_image_21_compile.py).
Its `tune` mode uses this configuration field, with environment-based autotuning
disabled in both modes to isolate the change. It records first-request and steady
timings, graph captures and fallbacks, memory peaks, and decoded outputs. See the
[benchmark instructions](../../../benchmarks/diffusion/README.md#qwen-image-21-compilation-comparison)
for the baseline/tuned workflow.

## Pinning one packed shape (MiniMax-H3)

Compiled regions are keyed by input shape. MiniMax-H3 packs a request into a
row count that follows the request -- a longer prompt, more frames, a larger
frame -- and rounds it up to the next 64-row boundary, so two requests can land
on two different packed lengths.

Under the default `diffusion_compile_dynamic: true` a new packed length is
absorbed by the dynamic shape, so it is not by itself a recompilation. It still
hands the backend a shape it has not autotuned, which is the risk this knob
exists to remove: a fixed-shape deployment (`--no-diffusion-compile-dynamic`)
recompiles the transformer blocks per shape, and a first-seen large shape can
push autotuning into its worst-case memory use.

A request pins the length with `extra_args["pad_seq_len"]`:

```python
sampling_params = SamplingParams(extra_args={"pad_seq_len": 54080})
```

The value must be a positive multiple of 64 and must cover the rows the request
actually uses; the packed sequence is then padded to it instead of to the next
64-row boundary. Pick a bucket that covers every request dimension the
deployment accepts -- prompt length, frame count, frame size and reference
blocks all change the used row count. The server logs the effective length as
`MiniMax H3 packed sequence: ... pad_seq_len=... used=... seq_len=...` whenever
a request pins it. The padding rows are masked out, so their cost is the
attention and feed-forward work on those rows.

Use `--enforce-eager` to disable the model runner's generic compile setup.
Pipelines that compile internally define their own eager-mode behavior.
