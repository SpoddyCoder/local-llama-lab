"""Run llama-bench from a model config directory."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from llama_bench_config import build_llama_bench_argv, resolve_llama_bench


def run_llama_bench(
    model_dir: Path,
    *,
    n_cpu_moe: int | None = None,
    dry_run: bool = False,
) -> int:
    try:
        config = resolve_llama_bench(model_dir, n_cpu_moe=n_cpu_moe)
    except (ValueError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1

    argv = build_llama_bench_argv(config)

    if dry_run:
        print(shlex.join(argv))
        return 0

    proc = subprocess.run(argv, check=False)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run llama-bench from a model config directory",
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="model directory containing model.yaml and llama_bench.yaml",
    )
    parser.add_argument(
        "--n-cpu-moe",
        type=int,
        default=None,
        help="offload N MoE layers to CPU (appends --n-cpu-moe and --n-gpu-layers)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print final command line and exit without running",
    )
    args = parser.parse_args(argv)

    return run_llama_bench(
        args.model_dir,
        n_cpu_moe=args.n_cpu_moe,
        dry_run=args.dry_run,
    )
