"""Launch llama-server from a model config and hold until Ctrl+C."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from model_layout import resolve_model_server
from server import managed_server, resolve_base_url


def run_launch_server(
    model_dir: Path,
    *,
    n_cpu_moe: int | None = None,
) -> int:
    try:
        server, _model_yaml_path = resolve_model_server(model_dir, n_cpu_moe=n_cpu_moe)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    base_url = resolve_base_url(server, None)
    model_display = model_dir.name

    print(f"Starting {server.binary} (model: {model_display})...")
    print(f"Health URL: {base_url}")

    try:
        with managed_server(server, base_url, inherit_stdio=True) as proc:
            endpoint = proc.ready_endpoint or "unknown"
            print(f"Server ready at {base_url} (health: {endpoint})")
            print("Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    except (TimeoutError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch llama-server from a model config and hold until Ctrl+C",
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="model directory containing model.yaml and server.yaml",
    )
    parser.add_argument(
        "--n-cpu-moe",
        type=int,
        default=None,
        help="offload N MoE layers to CPU (appends --n-cpu-moe and --n-gpu-layers)",
    )
    args = parser.parse_args(argv)

    try:
        return run_launch_server(args.model_dir, n_cpu_moe=args.n_cpu_moe)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
