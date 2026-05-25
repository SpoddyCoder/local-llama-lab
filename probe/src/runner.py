"""Single-run orchestration: load configs, server, client, results, teardown."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import httpx

from client import run_chat_completion
from config import (
    load_client_config,
    parse_n_cpu_moe_from_args,
    redact_model_path,
)
from model_layout import RunConfig, resolve_run_config
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
from host_ram import sample_process_rss_mb
from paths import PROBE_ROOT, RESULTS_DIR
from server import fetch_model_max_context, managed_server, resolve_base_url
from vram import VramPoller, sample_vram_mb

_PROBE_ROOT = PROBE_ROOT


def _result_path_display(result_path: Path) -> str:
    try:
        return str(result_path.relative_to(PROBE_ROOT))
    except ValueError:
        return str(result_path)


def _run(
    run_config: RunConfig,
    *,
    save_result: bool,
    quiet: bool,
    include_output: bool = False,
    session_id: str | None = None,
) -> int:
    if save_result and (not run_config.model or not run_config.variant):
        print("Error: --save-result requires model and variant", file=sys.stderr)
        return 1

    server = run_config.server
    client = load_client_config(run_config.client_path)
    base_url = resolve_base_url(server, client.base_url)
    client = replace(client, base_url=base_url)

    if not save_result:
        model_display = run_config.model or server.model_slug
        print(f"Starting {server.binary} (model: {model_display})...")
        print(f"API URL: {base_url}")

    started_at = utc_now() if save_result else None
    status = "ok"
    error: str | None = None
    metrics_dict: dict[str, float | int | None] | None = None
    completion = None

    try:
        with managed_server(server, base_url) as proc:
            with httpx.Client(timeout=2.0) as http_client:
                model_max_context = fetch_model_max_context(http_client, base_url)
            idle_vram_mb = sample_vram_mb()
            idle_system_ram_mb: int | None = None
            n_cpu_moe_from_args = parse_n_cpu_moe_from_args(server.args)
            if n_cpu_moe_from_args is not None:
                idle_system_ram_mb = sample_process_rss_mb(proc.pid)
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
            metrics_dict["model_max_context"] = model_max_context
            if n_cpu_moe_from_args is not None:
                metrics_dict["idle_system_ram_mb"] = idle_system_ram_mb
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
        if include_output:
            print()
            print(completion.completion_text if completion else "")
        return 0

    finished_at = utc_now()
    result_path: Path | None = None
    run_id: str | None = None
    model_name = run_config.model
    variant_name = run_config.variant

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
                model=model_name,
                variant=variant_name,
            ),
            model=model_name,
            variant=variant_name,
            session_id=session_id,
        )
        run_id = document["run_id"]
        result_path = write_result(
            RESULTS_DIR,
            document,
            model=model_name,
            variant=variant_name,
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
        if include_output:
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
                    probe_root=PROBE_ROOT,
                )
            )
        return 0

    print(
        format_error_summary(error or "unknown error", result_path, probe_root=PROBE_ROOT),
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local model probe run")
    parser.add_argument(
        "model_dir",
        type=Path,
        help="model directory containing model.yaml and server.yaml",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="hello-world-baseline",
        help="variant name (default: hello-world-baseline)",
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
        "--include-output",
        action="store_true",
        help="print completion text on stdout; with --save-result, also write beside result JSON",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="optional session id for result JSON (e.g. calibration)",
    )
    parser.add_argument(
        "--n-cpu-moe",
        type=int,
        default=None,
        help="offload N MoE layers to CPU (replace --n-cpu-moe and --n-gpu-layers)",
    )
    args = parser.parse_args(argv)

    try:
        run_config = resolve_run_config(
            args.model_dir,
            args.variant,
            n_cpu_moe=args.n_cpu_moe,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    return _run(
        run_config,
        save_result=args.save_result,
        quiet=args.quiet if args.save_result else False,
        include_output=args.include_output,
        session_id=args.session_id,
    )
