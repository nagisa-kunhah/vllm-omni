# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download and assemble a local NAVA directory for vLLM-Omni."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODEL_INDEX = {
    "_class_name": "NAVAPipeline",
    "nava_ckpt": "NAVA.safetensors",
    "fp8_ckpt": "NAVA_fp8.safetensors",
    "config": "configs/nava.yaml",
    "wan_dir": "Wan2.2-TI2V-5B",
    "audio_vae_dir": "params",
}

REQUIRED_PATHS = ("Wan2.2-TI2V-5B", "params")
CONFIG_PATHS = ("configs/nava.yaml", "nava.yaml")
CHECKPOINT_PATHS = ("NAVA.safetensors", "NAVA.ckpt", "NAVA_fp8.safetensors", "NAVA_fp8.ckpt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download baidu/NAVA and write a vLLM-Omni model_index.json.")
    parser.add_argument("--repo-id", default="baidu/NAVA", help="NAVA Hugging Face repository.")
    parser.add_argument("--local-dir", required=True, help="Target local model directory.")
    parser.add_argument(
        "--bf16-only",
        action="store_true",
        help="Exclude NAVA_fp8.safetensors and download the bf16 checkpoint only.",
    )
    parser.add_argument(
        "--fp8-only",
        action="store_true",
        help="Exclude NAVA.safetensors and download the fp8 checkpoint only.",
    )
    parser.add_argument(
        "--install-upstream",
        action="store_true",
        help="Run `pip install -e <upstream-dir>` after download. Pass --upstream-dir when the code repo is separate.",
    )
    parser.add_argument(
        "--upstream-dir",
        default=None,
        help="Local ernie-research/NAVA source checkout. Required only with --install-upstream if not equal to --local-dir.",
    )
    parser.add_argument(
        "--prepare-redimnet",
        action="store_true",
        help="Preload the upstream ReDimNet speaker model into the torch hub cache for timbre control.",
    )
    parser.add_argument(
        "--torch-home",
        default=None,
        help="Optional TORCH_HOME used by --prepare-redimnet. Set the same value when running inference.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that --local-dir has the files required by NAVAPipeline; do not download.",
    )
    return parser.parse_args()


def prepare_redimnet(torch_home: str | None) -> None:
    if torch_home:
        torch_home_path = Path(torch_home).expanduser().resolve()
        torch_home_path.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_home_path)

    import torch

    torch.hub.load(
        "IDRnD/ReDimNet",
        "ReDimNet",
        model_name="M",
        train_type="ft_mix",
        dataset="vb2+vox2+cnc",
    ).eval()


def verify_model_dir(local_dir: Path) -> None:
    missing = [name for name in REQUIRED_PATHS if not (local_dir / name).exists()]
    if not any((local_dir / name).exists() for name in CONFIG_PATHS):
        missing.append("configs/nava.yaml or nava.yaml")
    if not any((local_dir / name).exists() for name in CHECKPOINT_PATHS):
        missing.append("NAVA.safetensors or NAVA.ckpt")
    if missing:
        formatted = ", ".join(missing)
        raise SystemExit(
            f"NAVA model directory is incomplete: missing {formatted}. "
            "Run this script without --verify-only or pass a prepared local directory."
        )


def build_model_index(local_dir: Path) -> dict[str, str]:
    model_index = dict(MODEL_INDEX)
    if not (local_dir / model_index["config"]).exists() and (local_dir / "nava.yaml").exists():
        model_index["config"] = "nava.yaml"
    return model_index


def main() -> None:
    args = parse_args()
    if args.bf16_only and args.fp8_only:
        raise SystemExit("--bf16-only and --fp8-only are mutually exclusive.")

    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        verify_model_dir(local_dir)
        print(f"Verified {local_dir}")
    else:
        cmd = ["huggingface-cli", "download", args.repo_id, "--local-dir", str(local_dir)]
        if args.bf16_only:
            cmd.extend(["--exclude", "NAVA_fp8.safetensors"])
        if args.fp8_only:
            cmd.extend(["--exclude", "NAVA.safetensors"])
        subprocess.run(cmd, check=True)

        model_index_path = local_dir / "model_index.json"
        model_index_path.write_text(json.dumps(build_model_index(local_dir), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {model_index_path}")
    verify_model_dir(local_dir)

    if args.install_upstream:
        upstream_dir = Path(args.upstream_dir or local_dir).expanduser().resolve()
        if not (upstream_dir / "nava_src").exists():
            raise SystemExit(
                f"Could not find nava_src under {upstream_dir}. "
                "Clone https://github.com/ernie-research/NAVA and pass --upstream-dir."
            )
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(upstream_dir)], check=True)

    if args.prepare_redimnet:
        prepare_redimnet(args.torch_home)
        if args.torch_home:
            print(f"Prepared ReDimNet with TORCH_HOME={Path(args.torch_home).expanduser().resolve()}")


if __name__ == "__main__":
    main()
