"""Launch llama-server from a model config and hold until Ctrl+C."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from config import (
    MODEL_YAML,
    apply_n_cpu_moe,
    config_dir_metadata,
    load_client_config,
    load_variant_server,
)
from server import managed_server, resolve_base_url

_TESTER_ROOT = Path(__file__).resolve().parent.parent
_BENCH_VARIANT = "bench"


def resolve_launch_config_paths(model_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve server.yaml, client.yaml, and model.yaml for launch_server."""
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {model_dir}")

    if (model_dir / "server.yaml").is_file():
        variant_dir = model_dir
        model_yaml = model_dir.parent / MODEL_YAML
    elif (model_dir / _BENCH_VARIANT / "server.yaml").is_file():
        variant_dir = model_dir / _BENCH_VARIANT
        model_yaml = model_dir / MODEL_YAML
    else:
        raise FileNotFoundError(
            f"No server.yaml in {model_dir} or {model_dir / _BENCH_VARIANT}"
        )

    server = variant_dir / "server.yaml"
    client = variant_dir / "client.yaml"

    missing: list[str] = []
    if not client.is_file():
        missing.append(client.name)
    if not model_yaml.is_file():
        missing.append(MODEL_YAML)
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Config files missing for {model_dir}: {names}")

    return server, client, model_yaml


def run_launch_server(
    model_dir: Path,
    *,
    n_cpu_moe: int | None = None,
) -> int:
    server_path, client_path, model_yaml_path = resolve_launch_config_paths(model_dir)
    server = load_variant_server(server_path, model_yaml_path)
    if n_cpu_moe is not None:
        try:
            server = apply_n_cpu_moe(server, n_cpu_moe)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1

    client = load_client_config(client_path)
    base_url = resolve_base_url(server, client.base_url)

    meta = config_dir_metadata(server_path.parent, _TESTER_ROOT)
    model_display = meta.get("model") or server.model_slug

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
        help=(
            "model directory (uses bench/server.yaml) or variant directory "
            "containing server.yaml"
        ),
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
