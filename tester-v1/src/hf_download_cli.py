"""Download GGUF from Hugging Face using model.yaml hf-download metadata."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from config import MODEL_YAML
from hf_download_config import (
    build_hf_download_argv,
    format_path_for_yaml,
    load_hf_download_required,
    resolve_hf_cache_path,
    update_model_path_if_changed,
)


def run_hf_download(model_dir: Path, *, dry_run: bool = False) -> int:
    if not model_dir.is_dir():
        print(f"model directory not found: {model_dir}", file=sys.stderr)
        return 1

    model_yaml_path = model_dir / MODEL_YAML
    if not model_yaml_path.is_file():
        print(f"model config not found: {model_yaml_path}", file=sys.stderr)
        return 1

    try:
        config = load_hf_download_required(model_yaml_path)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    argv = build_hf_download_argv(config.repo, config.file)

    if dry_run:
        print(shlex.join(argv))
        return 0

    if shutil.which("hf") is None:
        print(
            "hf CLI not found on PATH (install with: pip install huggingface_hub[cli])",
            file=sys.stderr,
        )
        return 1

    proc = subprocess.run(argv, check=False)
    if proc.returncode != 0:
        return proc.returncode

    try:
        resolved_path = resolve_hf_cache_path(config.repo, config.file)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if update_model_path_if_changed(model_yaml_path, resolved_path):
        print(f"Updated model: {format_path_for_yaml(resolved_path)}")
    else:
        print("model: unchanged")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download GGUF from Hugging Face using model.yaml hf-download metadata",
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="model directory containing model.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print final command line and exit without running",
    )
    args = parser.parse_args(argv)

    return run_hf_download(args.model_dir, dry_run=args.dry_run)
