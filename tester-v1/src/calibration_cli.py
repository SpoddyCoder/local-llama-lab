"""VRAM calibration orchestration: probe runs and summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from calibration import (
    VARIANT_CTX_PROBE,
    VARIANT_FOOTPRINT,
    compute_summary,
    find_latest_result,
    format_summary_lines,
    gguf_size_gb,
    read_idle_vram_from_result,
    run_variant_subprocess,
)
from config import load_server_config, parse_context_from_args, slug_from_config_dir
from vram import query_gpu_total_mb

_DEFAULT_TESTER_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIRNAME = "results"
_OUTPUT_TAIL_LINES = 40


def _variant_dir(model_dir: Path, name: str) -> Path:
    return model_dir / name


def _validate_variant_dir(variant_dir: Path) -> None:
    missing: list[str] = []
    for name in ("server.yaml", "client.yaml"):
        if not (variant_dir / name).is_file():
            missing.append(name)
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Config files missing in {variant_dir}: {names}")


def _validate_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    for variant in (VARIANT_FOOTPRINT, VARIANT_CTX_PROBE):
        variant_dir = _variant_dir(model_dir, variant)
        if not variant_dir.is_dir():
            raise FileNotFoundError(
                f"variant directory not found: {variant_dir} "
                f"(expected {variant}/ under {model_dir})"
            )
        _validate_variant_dir(variant_dir)


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _output_tail(text: str, *, max_lines: int = _OUTPUT_TAIL_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run VRAM calibration probes and print context estimate",
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="model directory containing calibration-footprint/ and calibration-ctx-probe/ variants",
    )
    parser.add_argument(
        "--margin-mib",
        type=int,
        default=1536,
        help="VRAM margin reserved for non-KV use (default: 1536)",
    )
    parser.add_argument(
        "--tester-root",
        type=Path,
        default=_DEFAULT_TESTER_ROOT,
        help="tester-v1 root (default: directory containing this package)",
    )
    args = parser.parse_args(argv)

    tester_root = args.tester_root.resolve()
    model_dir = args.model_dir.resolve()
    results_dir = tester_root / _RESULTS_DIRNAME

    try:
        _validate_model_dir(model_dir)
    except FileNotFoundError as exc:
        return _fail(str(exc))

    footprint_dir = _variant_dir(model_dir, VARIANT_FOOTPRINT)
    ctx_probe_dir = _variant_dir(model_dir, VARIANT_CTX_PROBE)

    variants = [
        (VARIANT_FOOTPRINT, footprint_dir, "Running calibration-footprint..."),
        (VARIANT_CTX_PROBE, ctx_probe_dir, "Running calibration-ctx-probe..."),
    ]
    for _name, variant_dir, progress in variants:
        print(progress, file=sys.stderr)
        returncode, output = run_variant_subprocess(tester_root, variant_dir)
        if returncode != 0:
            tail = _output_tail(output)
            return _fail(
                f"{progress.rstrip('.')} failed (exit {returncode}) for {variant_dir}\n"
                f"{tail}"
            )

    try:
        footprint_slug = slug_from_config_dir(footprint_dir, tester_root)
        ctx_probe_slug = slug_from_config_dir(ctx_probe_dir, tester_root)
        footprint_result = find_latest_result(results_dir, footprint_slug)
        ctx_probe_result = find_latest_result(results_dir, ctx_probe_slug)
    except (FileNotFoundError, ValueError) as exc:
        return _fail(f"{exc}\n(results dir: {results_dir})")

    try:
        footprint_idle = read_idle_vram_from_result(footprint_result)
        ctx_probe_idle = read_idle_vram_from_result(ctx_probe_result)
    except ValueError as exc:
        return _fail(str(exc))

    footprint_server_path = footprint_dir / "server.yaml"
    ctx_probe_server_path = ctx_probe_dir / "server.yaml"
    try:
        footprint_server = load_server_config(footprint_server_path)
        ctx_probe_server = load_server_config(ctx_probe_server_path)
        footprint_c = parse_context_from_args(footprint_server.args)
        ctx_c = parse_context_from_args(ctx_probe_server.args)
        gguf_gb = gguf_size_gb(footprint_server.model)
    except ValueError as exc:
        return _fail(
            f"{exc}\n"
            f"footprint server: {footprint_server_path}\n"
            f"ctx-probe server: {ctx_probe_server_path}"
        )

    gpu_total_mb = query_gpu_total_mb()
    if gpu_total_mb is None:
        return _fail(
            "GPU total VRAM unavailable (nvidia-smi missing or failed); "
            "install drivers and ensure nvidia-smi works"
        )

    try:
        summary = compute_summary(
            footprint_idle,
            ctx_probe_idle,
            footprint_c,
            ctx_c,
            gpu_total_mb,
            args.margin_mib,
            gguf_gb,
        )
    except ValueError as exc:
        return _fail(
            f"{exc}\n"
            f"footprint result: {footprint_result}\n"
            f"ctx-probe result: {ctx_probe_result}\n"
            f"footprint config: {footprint_dir}\n"
            f"ctx-probe config: {ctx_probe_dir}"
        )

    print("", file=sys.stdout)
    for line in format_summary_lines(summary):
        print(line)
    return 0
