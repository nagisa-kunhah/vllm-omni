# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare default versus GEMM-autotuned Qwen-Image-2.1 compilation.

Launch each mode in a fresh process, preserving the same device and cache settings::

    python benchmarks/diffusion/qwen_image_21_compile.py \
        --model Qwen/Qwen-Image-2.1 --output-dir ./qwen-compile --mode baseline
    python benchmarks/diffusion/qwen_image_21_compile.py \
        --model Qwen/Qwen-Image-2.1 --output-dir ./qwen-compile --mode tune

Both modes retain regional dynamic compilation, BF16 and explicit decode graphs.
The tuned mode sets diffusion_compile_gemm_autotune=True; environment-based
autotuning is disabled in both modes to isolate the scoped configuration.
Each case has one first request and three steady requests by default. First-request
cost is not guaranteed cold-cache compilation. Timed intervals exclude CPU image
transfer and saving; stage-boundary instrumentation is common to both modes.
Final decoded tensors and PNGs are saved; every request has a raw tensor hash.
Editing uses the same baseline-short1024-final.png for both modes unless an
explicit --reference is supplied. Run the baseline short case first.

This script uses one local in-process diffusion worker. CUDA_VISIBLE_DEVICES and
all cache/device environment settings are preserved. GEMM and broad autotune
settings are explicitly controlled in this benchmark process before importing
Torch. These settings do not modify the calling shell or installed packages.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import time
import traceback
import uuid
from pathlib import Path

SHORT = "A ceramic teapot on a wooden table"
LONG = (
    "A ceramic teapot on a wooden table in a quiet sunlit pottery studio. "
    + (
        "The scene contains blue and white porcelain, warm natural wood, delicate leaves, "
        "soft window light, realistic shadows, and carefully arranged handcrafted objects. "
    )
    * 24
)


def collect_provenance(args, torch):
    """Record inspectable source/runtime provenance without network calls."""
    repo = Path(__file__).resolve().parents[2]

    def git(*arguments):
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo), *arguments], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    packages = {}
    for name in ("torch", "vllm", "vllm-omni", "diffusers", "transformers", "fa3-fwd"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    local = Path(args.model)
    resolved = local.resolve() if local.exists() else None
    snapshot = None
    if resolved is not None and "snapshots" in resolved.parts:
        index = resolved.parts.index("snapshots")
        if len(resolved.parts) > index + 1:
            snapshot = resolved.parts[index + 1]
    config_commit = None
    if resolved is not None:
        for filename in ("model_index.json", "config.json"):
            config_path = resolved / filename
            if config_path.exists():
                try:
                    config_commit = json.loads(config_path.read_text()).get("_commit_hash") or config_commit
                except (ValueError, OSError):
                    pass
    try:
        nvidia_smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        nvidia_smi = None
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "nvidia_smi_all_devices": nvidia_smi,
        "nvidia_smi_note": "Physical inventory; CUDA_VISIBLE_DEVICES may remap the logical device index.",
        "gpu": {"name": gpu.name, "total_memory": gpu.total_memory, "capability": [gpu.major, gpu.minor]},
        "model_requested": args.model,
        "revision_requested": args.revision,
        "local_snapshot_revision": snapshot,
        "config_commit_hash": config_commit,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cache_environment": {
            key: os.environ.get(key)
            for key in ("HF_HOME", "HF_HUB_CACHE", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
        },
        "autotune_environment": {
            key: os.environ.get(key)
            for key in (
                "TORCHINDUCTOR_MAX_AUTOTUNE",
                "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE",
                "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM",
                "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS",
            )
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["baseline", "tune"], required=True)
    ap.add_argument("--model", required=True, help="Local model directory or Hugging Face repository ID")
    ap.add_argument("--revision", default=None, help="Optional model revision passed to the loader")
    ap.add_argument("--output-dir", type=Path, required=True, help="Shared artifact directory for baseline and tune")
    ap.add_argument(
        "--cases",
        nargs="+",
        choices=["short1024", "long1024", "edit1024", "short2048"],
        default=["short1024", "long1024", "edit1024", "short2048"],
    )
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument(
        "--reference", type=Path, help="Same input image for both modes; defaults to baseline short1024 final PNG"
    )
    a = ap.parse_args()
    if a.repetitions < 1:
        ap.error("--repetitions must be positive")
    if len(a.cases) != len(set(a.cases)):
        ap.error("--cases must not contain duplicates")
    root = a.output_dir.resolve()
    a.reference = (a.reference or root / "outputs/baseline-short1024-final.png").resolve()
    if "edit1024" in a.cases and not a.reference.exists():
        creates_reference = (
            a.mode == "baseline"
            and "short1024" in a.cases
            and a.cases.index("short1024") < a.cases.index("edit1024")
            and a.reference == root / "outputs/baseline-short1024-final.png"
        )
        if not creates_reference:
            ap.error("Editing reference is missing; run baseline short1024 first or pass --reference")
    # Explicitly remove inherited broad tuning; isolate GEMM autotuning only.
    tune = a.mode == "tune"
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE"] = "0"
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "ATEN,TRITON"
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "FLASH_ATTN"
    import numpy as np
    import torch
    from PIL import Image

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.models.qwen_image_21.decode_graph import (
        QwenImage21DecodeGraphEntry,
        QwenImage21DecodeGraphManager,
    )
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    output = root / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    records = root / "analysis" / f"e2e-{a.mode}.jsonl"
    records.parent.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex

    def emit(event):
        event["run_id"] = run_id
        event["utc_unix"] = time.time()
        event.setdefault("mode", a.mode)
        line = json.dumps(event, default=str)
        print(line, flush=True)
        with records.open("a") as f:
            f.write(line + "\n")
            f.flush()

    counters = {}

    def reset():
        counters.clear()
        counters.update(
            decode_calls=0,
            graph_success=0,
            graph_fallback=0,
            capture_attempts=0,
            capture_success=0,
            capture_wall_seconds=0.0,
            capture_allocated_delta=0,
            capture_reserved_delta=0,
        )

    reset()
    old_try = QwenImage21DecodeGraphManager.try_decode
    old_capture = QwenImage21DecodeGraphEntry.capture

    def tracked_try(self, *args, **kwargs):
        counters["decode_calls"] += 1
        out = old_try(self, *args, **kwargs)
        counters["graph_success" if out is not None else "graph_fallback"] += 1
        return out

    def tracked_capture(self, body):
        before_a = torch.cuda.memory_allocated()
        before_r = torch.cuda.memory_reserved()
        start = time.perf_counter()
        counters["capture_attempts"] += 1
        try:
            result = old_capture(self, body)
            counters["capture_success"] += int(self.graph is not None)
            return result
        finally:
            counters["capture_wall_seconds"] += time.perf_counter() - start
            counters["capture_allocated_delta"] += torch.cuda.memory_allocated() - before_a
            counters["capture_reserved_delta"] += torch.cuda.memory_reserved() - before_r

    QwenImage21DecodeGraphManager.try_decode = tracked_try
    QwenImage21DecodeGraphEntry.capture = tracked_capture
    dynamic = True
    cfg = OmniDiffusionConfig(
        model=a.model,
        revision=a.revision,
        enforce_eager=False,
        enable_cuda_graph_decode=True,
        diffusion_compile_granularity="regional",
        diffusion_compile_dynamic=dynamic,
        diffusion_compile_gemm_autotune=tune,
        enable_diffusion_pipeline_profiler=True,
        dtype=torch.bfloat16,
    )
    engine = None
    try:
        cfg.enrich_config()
        provenance = collect_provenance(a, torch)
        start = time.perf_counter()
        engine = DiffusionEngine(cfg)
        torch.accelerator.synchronize()
        emit(
            {
                "event": "init",
                "seconds": time.perf_counter() - start,
                "pid": os.getpid(),
                "config": str(cfg),
                "dynamic": dynamic,
                "gemm_autotune": tune,
                "cases": a.cases,
                "repetitions": a.repetitions,
                "model_resolved": str(Path(a.model).resolve()) if Path(a.model).exists() else a.model,
                "cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                "provenance": provenance,
                "methodology": (
                    "First request is process/workload cold, NOT a guaranteed cold compiler cache. "
                    "Common stage-boundary profiler; time excludes D2H/postprocessing/save. "
                    "Graph capture wall includes warmup and lazy compile; graph entries accumulate across cases. "
                    "Output GPU references released before next request."
                ),
            }
        )
        transformer = engine.executor.driver_worker.worker.model_runner.pipeline.transformer
        case_summaries = []
        for case in a.cases:
            size = 2048 if case == "short2048" else 1024
            prompt = LONG if case == "long1024" else SHORT
            base = {"prompt": prompt}
            if case == "edit1024":
                prompt = "Make the ceramic teapot blue and white, keeping the wooden table and lighting unchanged."
                with Image.open(a.reference) as im:
                    reference = im.convert("RGB")
                base = {"prompt": prompt, "multi_modal_data": {"image": reference}}
            emit(
                {
                    "event": "case",
                    "case": case,
                    "prompt": prompt,
                    "size": size,
                    "seed": 42,
                    "steps": 50,
                    "true_cfg_scale": 1.0,
                    "reference": str(a.reference) if case == "edit1024" else None,
                    "reference_sha256": hashlib.sha256(a.reference.read_bytes()).hexdigest()
                    if case == "edit1024"
                    else None,
                }
            )
            steady_hashes = []
            steady_records = []
            cold_seconds = None
            for index in range(a.repetitions + 1):
                label = "cold" if index == 0 else f"steady{index}"
                req = OmniDiffusionRequest(
                    prompt=dict(base),
                    sampling_params=OmniDiffusionSamplingParams(
                        height=size, width=size, num_inference_steps=50, true_cfg_scale=1.0, seed=42
                    ),
                    request_id=f"{run_id}-{a.mode}-{case}-{label}",
                )
                reset()
                torch.accelerator.synchronize()
                torch.accelerator.reset_peak_memory_stats()
                start = time.perf_counter()
                out = engine.add_req_and_wait_for_response(req)
                torch.accelerator.synchronize()
                elapsed = time.perf_counter() - start
                manager = transformer._decode_graph_manager
                event = {
                    "event": "request",
                    "case": case,
                    "label": label,
                    "seconds": elapsed,
                    "stage_durations": out.stage_durations,
                    "allocated_peak": torch.accelerator.max_memory_allocated(),
                    "reserved_peak": torch.cuda.max_memory_reserved(),
                    "allocated_after": torch.cuda.memory_allocated(),
                    "reserved_after": torch.cuda.memory_reserved(),
                    "graph": dict(counters),
                    "graph_entries": [
                        {
                            "key": str(k),
                            "captured": v is not None and v.graph is not None,
                            "captures": v.captures if v else 0,
                        }
                        for k, v in manager.entries.items()
                    ]
                    if manager
                    else [],
                    "error": out.error,
                }
                if out.error:
                    emit(event)
                    raise RuntimeError(out.error)
                raw = out.output
                if isinstance(raw, dict):
                    raw = raw["payload"]["image"]
                array = raw.detach().float().cpu().numpy()
                finite = bool(np.isfinite(array).all())
                digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
                event.update(
                    raw_finite=finite,
                    raw_shape=list(array.shape),
                    raw_dtype=str(array.dtype),
                    raw_min=float(array.min()) if finite else None,
                    raw_max=float(array.max()) if finite else None,
                    raw_sha256=digest,
                )
                if index:
                    steady_hashes.append(digest)
                    steady_records.append(event)
                else:
                    cold_seconds = elapsed
                if index == a.repetitions:
                    np.save(output / f"{a.mode}-{case}-final.npy", array)
                    images = engine.post_process_func(out.output)
                    if isinstance(images, dict):
                        images = images["payload"]["image"]
                    images[0].save(output / f"{a.mode}-{case}-final.png")
                    del images
                    event["saved_raw"] = str(output / f"{a.mode}-{case}-final.npy")
                    event["saved_image"] = str(output / f"{a.mode}-{case}-final.png")
                    event["png_sha256"] = hashlib.sha256(Path(event["saved_image"]).read_bytes()).hexdigest()
                emit(event)
                del array, raw, out, req
                if not finite:
                    raise RuntimeError(f"Nonfinite output: {case}/{label}")
            steady_seconds = [record["seconds"] for record in steady_records]
            summary = {
                "event": "case_complete",
                "case": case,
                "steady_hashes": steady_hashes,
                "steady_bitexact": len(set(steady_hashes)) == 1,
                "steady_count": len(steady_seconds),
                "steady_median_seconds": statistics.median(steady_seconds),
                "steady_min_seconds": min(steady_seconds),
                "steady_max_seconds": max(steady_seconds),
                "cold_seconds": cold_seconds,
                "steady_peak_allocated_bytes": max(record["allocated_peak"] for record in steady_records),
                "steady_peak_reserved_bytes": max(record["reserved_peak"] for record in steady_records),
                "steady_max_graph_fallbacks": max(record["graph"]["graph_fallback"] for record in steady_records),
                "steady_all_finite": all(record["raw_finite"] for record in steady_records),
            }
            emit(summary)
            case_summaries.append(summary)
        emit({"event": "complete", "case_summaries": case_summaries})
    except BaseException as exc:
        emit({"event": "failure", "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        try:
            if engine is not None:
                engine.close()
        finally:
            QwenImage21DecodeGraphManager.try_decode = old_try
            QwenImage21DecodeGraphEntry.capture = old_capture


if __name__ == "__main__":
    main()
