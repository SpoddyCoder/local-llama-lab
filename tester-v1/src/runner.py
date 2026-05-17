"""Single-run orchestration: load configs, server, client, results, teardown."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from client import run_chat_completion
from config import (
    ServerConfig,
    config_dir_metadata,
    load_client_config,
    load_server_config,
    redact_model_path,
)
from result_layout import completion_output_path
from metadata import collect_run_metadata
from metrics import build_metrics_dict, format_probe_stdout
from results import (
    build_result_document,
    format_error_summary,
    format_run_summary,
    utc_now,
    write_result,
)
from server import managed_server, resolve_base_url
from vram import VramPoller, sample_vram_mb

_TESTER_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _TESTER_ROOT / "results"


def _load_server(server_path: Path) -> ServerConfig:
    return load_server_config(server_path)


def _result_path_display(result_path: Path) -> str:
    try:
        return str(result_path.relative_to(_TESTER_ROOT))
    except ValueError:
        return str(result_path)


def _run(
    server_path: Path,
    client_path: Path,
    config_dir: Path | None = None,
    *,
    save_result: bool,
    quiet: bool,
    full_output: bool = False,
    session_id: str | None = None,
) -> int:
    if save_result and config_dir is None:
        print("Error: --save-result requires config_dir", file=sys.stderr)
        return 1

    server = _load_server(server_path)
    client = load_client_config(client_path)
    base_url = resolve_base_url(server, client.base_url)
    client = replace(client, base_url=base_url)

    if not save_result:
        meta = config_dir_metadata(config_dir, _TESTER_ROOT)
        model_display = meta.get("model") or server.model_slug
        print(f"Starting {server.binary} (model: {model_display})...")
        print(f"API URL: {base_url}")

    started_at = utc_now() if save_result else None
    status = "ok"
    error: str | None = None
    metrics_dict: dict[str, float | int | None] | None = None
    completion = None

    try:
        with managed_server(server, base_url) as proc:
            idle_vram_mb = sample_vram_mb()
            poller = VramPoller()
            poller.start()
            try:
                if not save_result:
                    print("Running streaming chat completion...")
                completion = run_chat_completion(client, base_url=base_url)
            finally:
                peak_vram_mb = poller.stop()
            metrics_dict = build_metrics_dict(completion)
            metrics_dict["server_ready_s"] = proc.server_ready_s
            metrics_dict["idle_vram_mb"] = idle_vram_mb
            metrics_dict["peak_vram_mb"] = peak_vram_mb
    except (TimeoutError, RuntimeError, ValueError, KeyboardInterrupt) as exc:
        if save_result:
            status = "error"
            error = str(exc)
            if isinstance(exc, KeyboardInterrupt):
                error = "interrupted"
        else:
            if isinstance(exc, KeyboardInterrupt):
                raise
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if not save_result:
        print(f"\nModel: {redact_model_path(server.model)}\n")
        print(format_probe_stdout(metrics_dict or {}))
        return 0

    finished_at = utc_now()
    result_path: Path | None = None
    run_id: str | None = None

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
            config_dir=config_dir,
            tester_root=_TESTER_ROOT,
            session_id=session_id,
        )
        run_id = document["run_id"]
        result_path = write_result(
            _RESULTS_DIR,
            document,
            config_dir=config_dir,
            tester_root=_TESTER_ROOT,
            started_at=started_at,
        )
    except OSError as exc:
        print(f"Failed to write result JSON: {exc}", file=sys.stderr)
        if status == "ok":
            print(format_probe_stdout(metrics_dict or {}))
        elif error:
            print(f"Error: {error}", file=sys.stderr)
        return 1

    if status == "ok" and metrics_dict is not None and run_id is not None:
        if full_output:
            try:
                output_path = completion_output_path(result_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    completion.completion_text if completion else "",
                    encoding="utf-8",
                )
            except OSError as exc:
                print(f"Failed to write completion text: {exc}", file=sys.stderr)
                return 1
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
    server = _load_server(server_path)
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
        if server_override is None or client_override is None:
            missing = []
            if server_override is None:
                missing.append("--server")
            if client_override is None:
                missing.append("--client")
            raise FileNotFoundError(
                "config_dir is required, or pass both "
                + " and ".join(missing)
            )
        return server_override, client_override

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
        help="server.yaml path (default: config_dir/server.yaml)",
    )
    parser.add_argument(
        "--client",
        type=Path,
        default=None,
        help="client.yaml path (default: config_dir/client.yaml)",
    )
    parser.add_argument(
        "--test-server",
        action="store_true",
        help="start llama-server, wait for health, hold until Ctrl+C",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="write result JSON to results/ and print run summary",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only applies with --save-result; suppress stdout run summary; write one line to stderr on success",
    )
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="with --save-result, write completion text beside result JSON",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="optional session id for result JSON (e.g. calibration)",
    )
    args = parser.parse_args(argv)

    if args.config_dir is None and args.server is None and args.client is None:
        parser.print_help()
        return 0

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
    if args.test_server:
        return _run_test_server(server_path, client_path, config_dir)

    if args.full_output and not args.save_result:
        print("Error: --full-output requires --save-result", file=sys.stderr)
        return 1

    return _run(
        server_path,
        client_path,
        config_dir,
        save_result=args.save_result,
        quiet=args.quiet if args.save_result else False,
        full_output=args.full_output,
        session_id=args.session_id,
    )
