# MOSS-TTS Local Serving Defaults

This note documents the default high-throughput path for
MOSS-TTS-Local-Transformer-v1.5 in vLLM-Omni.

## Recommended Path

Use the HF-compatible Stage0 backbone path by default. It is the hand-written
Qwen3-compatible implementation under
`vllm_omni/model_executor/models/moss_tts_local/hf_compatible_qwen3.py`.

The native vLLM Qwen3 path remains available for experiments by setting:

```bash
MOSS_TTS_LOCAL_NATIVE=1
```

The default is intentionally not native. For this TTS workload, the
HF-compatible path keeps the Stage0 decode/output routing behavior closer to
the reference implementation and avoids the native outer CUDA graph padding
interactions observed during throughput testing.

## Recommended Deploy Config

Use:

```bash
--deploy-config vllm_omni/deploy/moss_tts_local_fullseq.yaml
```

The deploy config uses:

- Stage0 `max_num_seqs: 64`
- Stage1 `max_num_seqs: 8`
- `async_chunk: true`
- connector `codec_streaming: false`

`codec_streaming: false` is important for throughput. Stage0 still uses the
async chunk adapter for transfer, but it sends one terminal full-sequence codec
payload per finished request. Stage1 can then batch completed utterances and run
full-sequence vocoder decode.

## Default Performance Features

The following MOSS Local performance features are enabled by default:

- Stage0 batched decode preprocess
- Stage0 fast pure-decode preprocess
- Stage0 local frame CUDA graph
- Stage0 delayed stop synchronization
- HF-compatible backbone fused QKV projection
- HF-compatible backbone fused RMSNorm
- HF-compatible backbone fused RoPE when Triton and CUDA are available
- Stage1 full-sequence admission coalescing for `codec_streaming: false`
- Stage1 opportunistic ready-payload drain
- Stage1 vocoder CUDA graph
- Stage1 vocoder graph-bucket grouping

These were selected because they are lossless or default-safe for the
full-sequence high-throughput path.

## Explicit Opt-Outs

The defaults can be disabled independently for debugging:

```bash
MOSS_TTS_LOCAL_DISABLE_BATCH_PREPROCESS=1
MOSS_TTS_LOCAL_DISABLE_FAST_BATCH_PREPROCESS=1
MOSS_TTS_LOCAL_DISABLE_FRAME_GRAPH=1
MOSS_TTS_LOCAL_DISABLE_DELAY_STOP_SYNC=1
MOSS_TTS_LOCAL_DISABLE_FUSED_QKV=1
MOSS_TTS_LOCAL_DISABLE_FUSED_RMSNORM=1
MOSS_TTS_LOCAL_DISABLE_FUSED_ROPE=1
MOSS_TTS_LOCAL_VOCODER_DISABLE_CUDA_GRAPH=1
MOSS_TTS_LOCAL_STAGE1_SYNC_DRAIN=0
```

The legacy positive enable flags are not required for the recommended path.

## Experimental Features

The following features remain opt-in because they did not become part of the
validated default path:

- native vLLM Qwen3 backbone (`MOSS_TTS_LOCAL_NATIVE=1`)
- local transformer `torch.compile`
- MLP `torch.compile`
- vocoder `torch.compile`
- vocoder TensorRT export/runtime
- vocoder multi-stream overlap
- FlashInfer single/batched decode experiments in the HF-compatible backbone

These should be tested with end-to-end serving benchmarks and audio-quality
checks before being promoted to defaults.

## Benchmark Reference

The latest H20 p500 validation used warmup before the formal run and the
recommended HF-compatible full-sequence configuration. Across three p500 runs,
the observed throughput was approximately:

- 41.77 audio seconds/s
- 37.40 audio seconds/s
- 45.47 audio seconds/s

The SGLang-Omni comparison baseline provided for the same input shape was
34.39 audio seconds/s. The median validated run is above that baseline by about
20 percent, while one run was lower due to run-to-run variance. Use repeated
warmup plus formal p500 runs when comparing future changes.
