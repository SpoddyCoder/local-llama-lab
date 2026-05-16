#!/usr/bin/env python3
"""Tester v1 entrypoint — single-run loop and test modes."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from client import run_chat_completion
from config import (
    ServerConfig,
    load_client_config,
    load_server_config,
    redact_model_path,
    slug_from_config_dir,
)
from metadata import collect_run_metadata
from metrics import build_metrics_dict, format_metrics_summary
from results import (
    build_result_document,
    format_error_summary,
    format_run_summary,
    utc_now,
    write_result,
)
from server import build_argv, managed_server, resolve_base_url
from vram import VramPoller, sample_vram_mb

_TESTER_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SERVER = _TESTER_ROOT / "server.yaml"
_DEFAULT_CLIENT = _TESTER_ROOT / "client.yaml"
_RESULTS_DIR = _TESTER_ROOT / "results"


def _load_server(server_path: Path, config_dir: Path | None) -> ServerConfig:
    server = load_server_config(server_path)
    if config_dir is not None:
        server = replace(
            server,
            run_slug=slug_from_config_dir(config_dir, _TESTER_ROOT),
        )
    return server


def _result_path_display(result_path: Path) -> str:
    try:
        return str(result_path.relative_to(_TESTER_ROOT))
    except ValueError:
        return str(result_path)


def _run_default(
    server_path: Path,
    client_path: Path,
    config_dir: Path | None = None,
    *,
    quiet: bool = False,
) -> int:
    server = _load_server(server_path, config_dir)
    client = load_client_config(client_path)
    base_url = resolve_base_url(server, client.base_url)
    client = replace(client, base_url=base_url)

    started_at = utc_now()
    status = "ok"
    error: str | None = None
    metrics_dict: dict[str, float | int | None] | None = None
    result_path: Path | None = None
    run_id: str | None = None

    try:
        with managed_server(server, base_url) as proc:
            idle_vram_mb = sample_vram_mb()
            poller = VramPoller()
            poller.start()
            try:
                completion = run_chat_completion(client, base_url=base_url)
            finally:
                peak_vram_mb = poller.stop()
            metrics_dict = build_metrics_dict(completion)
            metrics_dict["server_ready_s"] = proc.server_ready_s
            metrics_dict["idle_vram_mb"] = idle_vram_mb
            metrics_dict["peak_vram_mb"] = peak_vram_mb
    except (TimeoutError, RuntimeError, ValueError, KeyboardInterrupt) as exc:
        status = "error"
        error = str(exc)
        if isinstance(exc, KeyboardInterrupt):
            error = "interrupted"

    finished_at = utc_now()

    try:
        document = build_result_document(
            server_config=server,
            client_config=client,
            metrics_dict=metrics_dict,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            error=error,
            metadata=collect_run_metadata(
                server,
                config_dir=config_dir,
                tester_root=_TESTER_ROOT,
            ),
        )
        run_id = document["run_id"]
        result_path = write_result(_RESULTS_DIR, document)
    except OSError as exc:
        print(f"Failed to write result JSON: {exc}", file=sys.stderr)
        if status == "ok":
            print(format_metrics_summary(metrics_dict or {}))
        elif error:
            print(f"Error: {error}", file=sys.stderr)
        return 1

    if status == "ok" and metrics_dict is not None and run_id is not None:
        if quiet:
            print(f"Wrote {_result_path_display(result_path)}", file=sys.stderr)
        else:
            print(
                format_run_summary(
                    run_id,
                    result_path,
                    server,
                    metrics_dict,
                    tester_root=_TESTER_ROOT,
                )
            )
        return 0

    print(
        format_error_summary(error or "unknown error", result_path, tester_root=_TESTER_ROOT),
        file=sys.stderr,
    )
    return 1


def _run_test_server(
    server_path: Path,
    client_path: Path,
    config_dir: Path | None = None,
) -> int:
    server = _load_server(server_path, config_dir)
    client = load_client_config(client_path)
    base_url = resolve_base_url(server, client.base_url)

    print(f"Starting {server.binary} (model: {server.model_slug})...")
    print(f"Health URL: {base_url}")

    try:
        with managed_server(server, base_url) as proc:
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


def resolve_config_paths(
    config_dir: Path | None,
    server_override: Path | None,
    client_override: Path | None,
    tester_root: Path,
) -> tuple[Path, Path]:
    if config_dir is None:
        server = server_override or tester_root / "server.yaml"
        client = client_override or tester_root / "client.yaml"
        return server, client

    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    server = server_override or config_dir / "server.yaml"
    client = client_override or config_dir / "client.yaml"

    missing: list[str] = []
    if not server.is_file():
        missing.append(server.name)
    if not client.is_file():
        missing.append(client.name)
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Config files missing in {config_dir}: {names}")

    return server, client


def _run_test_client(
    server_path: Path,
    client_path: Path,
    config_dir: Path | None = None,
) -> int:
    server = _load_server(server_path, config_dir)
    client = load_client_config(client_path)
    base_url = resolve_base_url(server, client.base_url)
    client = replace(client, base_url=base_url)

    print(f"Starting {server.binary} (model: {server.model_slug})...")
    print(f"API URL: {base_url}")

    try:
        with managed_server(server, base_url) as proc:
            idle_vram_mb = sample_vram_mb()
            poller = VramPoller()
            poller.start()
            try:
                print("Running streaming chat completion...")
                result = run_chat_completion(client, base_url=base_url)
            finally:
                peak_vram_mb = poller.stop()
            metrics = build_metrics_dict(result)
            metrics["server_ready_s"] = proc.server_ready_s
            metrics["idle_vram_mb"] = idle_vram_mb
            metrics["peak_vram_mb"] = peak_vram_mb
            print(f"\nModel: {redact_model_path(server.model)}\n")
            print(format_metrics_summary(metrics))
            if result.completion_text:
                snippet = result.completion_text[:120].replace("\n", " ")
                suffix = "..." if len(result.completion_text) > 120 else ""
                print(f"\n(completion preview: {snippet}{suffix})")
        return 0
    except (TimeoutError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local model tester v1")
    parser.add_argument(
        "config_dir",
        nargs="?",
        type=Path,
        help="directory containing server.yaml and client.yaml",
    )
    parser.add_argument(
        "--server",
        type=Path,
        default=None,
        help=f"server.yaml path (default: {_DEFAULT_SERVER}, or config_dir/server.yaml)",
    )
    parser.add_argument(
        "--client",
        type=Path,
        default=None,
        help=f"client.yaml path (default: {_DEFAULT_CLIENT}, or config_dir/client.yaml)",
    )
    parser.add_argument(
        "--test-server",
        action="store_true",
        help="start llama-server, wait for health, hold until Ctrl+C",
    )
    parser.add_argument(
        "--test-client",
        action="store_true",
        help="start server, run one streaming completion, print metrics, teardown",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress stdout run summary; write one line to stderr on success",
    )
    args = parser.parse_args(argv)

    try:
        server_path, client_path = resolve_config_paths(
            args.config_dir,
            args.server,
            args.client,
            _TESTER_ROOT,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    config_dir = args.config_dir
    if args.test_client:
        return _run_test_client(server_path, client_path, config_dir)
    if args.test_server:
        return _run_test_server(server_path, client_path, config_dir)
    return _run_default(server_path, client_path, config_dir, quiet=args.quiet)


if __name__ == "__main__":
    sys.exit(main())
