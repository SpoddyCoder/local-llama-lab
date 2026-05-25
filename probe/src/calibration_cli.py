"""VRAM calibration orchestration: probe runs and summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from calibration import (
    DEFAULT_CALIBRATION_MARGIN_MIB,
    VARIANT_CTX_PROBE,
    VARIANT_FOOTPRINT,
    VARIANT_HELLO_WORLD,
    compute_summary,
    format_summary_lines,
    gguf_size_gb,
    parse_decode_tok_s_from_metrics_stdout,
    parse_idle_system_ram_from_metrics_stdout,
    parse_idle_vram_from_metrics_stdout,
    parse_model_max_context_from_metrics_stdout,
    read_decode_tok_s_from_result,
    read_idle_system_ram_from_result,
    read_idle_vram_from_result,
    read_model_max_context_from_result,
    run_calibration_variant,
    write_calibration_session_summary,
)
from config import (
    MODEL_YAML,
    load_model_config,
    parse_context_from_args,
    parse_n_cpu_moe_from_args,
)
from model_layout import resolve_run_config
from result_layout import find_latest_result
from paths import CALIBRATION_PROBES_ROOT, PROBE_ROOT, RESULTS_DIR
from results import format_compact_utc, utc_now
from vram import query_gpu_total_mb

_PROBE_ROOT = PROBE_ROOT

_OUTPUT_TAIL_LINES = 40


def _validate_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    model_yaml = model_dir / MODEL_YAML
    if not model_yaml.is_file():
        raise FileNotFoundError(f"model config not found: {model_yaml}")
    server_yaml = model_dir / "server.yaml"
    if not server_yaml.is_file():
        raise FileNotFoundError(f"server config not found: {server_yaml}")
    for variant in (VARIANT_FOOTPRINT, VARIANT_CTX_PROBE, VARIANT_HELLO_WORLD):
        probe_dir = CALIBRATION_PROBES_ROOT / variant
        missing: list[str] = []
        for name in ("client.yaml", "server.yaml"):
            if not (probe_dir / name).is_file():
                missing.append(f"calibration-probes/{variant}/{name}")
        if missing:
            names = ", ".join(missing)
            raise FileNotFoundError(
                f"calibration probe config missing for {variant}: {names}"
            )


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _output_tail(text: str, *, max_lines: int = _OUTPUT_TAIL_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def _idle_vram_from_probe_output(output: str, *, variant: str) -> int:
    try:
        return parse_idle_vram_from_metrics_stdout(output)
    except ValueError as exc:
        raise ValueError(f"{exc} (variant {variant})") from exc


def _print_probe_output(output: str) -> None:
    if not output:
        return
    end = "" if output.endswith("\n") else "\n"
    print(output, end=end, file=sys.stdout)


def _run_calibration(
    model_dir: Path,
    margin_mib: int,
    *,
    save_result: bool,
    quiet: bool,
    n_cpu_moe: int | None = None,
) -> int:
    results_dir = RESULTS_DIR
    session_id: str | None = None
    if save_result:
        session_id = format_compact_utc(utc_now())

    variants = [
        (VARIANT_FOOTPRINT, "Running calibration-footprint..."),
        (VARIANT_CTX_PROBE, "Running calibration-ctx-probe..."),
        (VARIANT_HELLO_WORLD, "Running hello-world-baseline..."),
    ]
    probe_outputs: list[tuple[str, str]] = []
    for variant_name, progress in variants:
        print(progress, file=sys.stderr)
        returncode, output = run_calibration_variant(
            model_dir,
            variant_name,
            save_result=save_result,
            quiet=save_result and quiet,
            session_id=session_id,
            n_cpu_moe=n_cpu_moe,
        )
        if returncode != 0:
            tail = _output_tail(output)
            return _fail(
                f"{progress.rstrip('.')} failed (exit {returncode}) for {variant_name}\n"
                f"{tail}"
            )
        if not save_result:
            _print_probe_output(output)
            probe_outputs.append((variant_name, output))

    try:
        model_path = load_model_config(model_dir / MODEL_YAML)
        footprint_cfg = resolve_run_config(
            model_dir, VARIANT_FOOTPRINT, n_cpu_moe=n_cpu_moe
        )
        ctx_probe_cfg = resolve_run_config(
            model_dir, VARIANT_CTX_PROBE, n_cpu_moe=n_cpu_moe
        )
        effective_n_cpu_moe = parse_n_cpu_moe_from_args(footprint_cfg.server.args)
        footprint_c = parse_context_from_args(footprint_cfg.server.args)
        ctx_c = parse_context_from_args(ctx_probe_cfg.server.args)
        gguf_gb = gguf_size_gb(model_path)
    except ValueError as exc:
        return _fail(
            f"{exc}\n"
            f"merged server config for variants "
            f"{VARIANT_FOOTPRINT!r} and {VARIANT_CTX_PROBE!r}"
        )

    if save_result:
        try:
            model = model_dir.name
            footprint_result = find_latest_result(
                results_dir, model, VARIANT_FOOTPRINT
            )
            ctx_probe_result = find_latest_result(
                results_dir, model, VARIANT_CTX_PROBE
            )
            hello_world_result = find_latest_result(
                results_dir, model, VARIANT_HELLO_WORLD
            )
        except (FileNotFoundError, ValueError) as exc:
            return _fail(f"{exc}\n(results dir: {results_dir})")

        try:
            footprint_idle = read_idle_vram_from_result(footprint_result)
            ctx_probe_idle = read_idle_vram_from_result(ctx_probe_result)
            model_max_context = read_model_max_context_from_result(footprint_result)
            generation_throughput_tok_s = read_decode_tok_s_from_result(
                hello_world_result
            )
            if effective_n_cpu_moe is not None:
                footprint_model_system_ram_mb = read_idle_system_ram_from_result(
                    footprint_result
                )
            else:
                footprint_model_system_ram_mb = None
        except ValueError as exc:
            return _fail(str(exc))
    else:
        try:
            footprint_idle = _idle_vram_from_probe_output(
                probe_outputs[0][1],
                variant=probe_outputs[0][0],
            )
            ctx_probe_idle = _idle_vram_from_probe_output(
                probe_outputs[1][1],
                variant=probe_outputs[1][0],
            )
            model_max_context = parse_model_max_context_from_metrics_stdout(
                probe_outputs[0][1]
            )
            generation_throughput_tok_s = parse_decode_tok_s_from_metrics_stdout(
                probe_outputs[2][1]
            )
            if effective_n_cpu_moe is not None:
                footprint_model_system_ram_mb = (
                    parse_idle_system_ram_from_metrics_stdout(probe_outputs[0][1])
                )
            else:
                footprint_model_system_ram_mb = None
        except ValueError as exc:
            return _fail(str(exc))

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
            margin_mib,
            gguf_gb,
        )
        summary["model_max_context"] = model_max_context
        summary["generation_throughput_tok_s"] = generation_throughput_tok_s
        if effective_n_cpu_moe is not None:
            summary["model_system_ram_mb"] = footprint_model_system_ram_mb
    except ValueError as exc:
        if save_result:
            return _fail(
                f"{exc}\n"
                f"footprint result: {footprint_result}\n"
                f"ctx-probe result: {ctx_probe_result}\n"
                f"variants: {VARIANT_FOOTPRINT!r}, {VARIANT_CTX_PROBE!r}"
            )
        return _fail(
            f"{exc}\n"
            f"variants: {VARIANT_FOOTPRINT!r}, {VARIANT_CTX_PROBE!r}"
        )

    if save_result:
        write_calibration_session_summary(
            PROBE_ROOT,
            model_dir.name,
            session_id,
            summary,
            footprint_result,
            ctx_probe_result,
            hello_world_result,
        )

    if not (save_result and quiet):
        print("", file=sys.stdout)
        for line in format_summary_lines(summary):
            print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run VRAM calibration probes and print context estimate",
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="model directory containing model.yaml",
    )
    parser.add_argument(
        "--margin-mib",
        type=int,
        default=DEFAULT_CALIBRATION_MARGIN_MIB,
        help=(
            "VRAM safety buffer subtracted from the KV budget "
            f"(default: {DEFAULT_CALIBRATION_MARGIN_MIB} MiB)"
        ),
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="save calibration tests JSON to the results directory",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only applies with --save-result; suppress calibration summary on stdout",
    )
    parser.add_argument(
        "--n-cpu-moe",
        type=int,
        default=None,
        help="offload N MoE layers to CPU (replace --n-cpu-moe and --n-gpu-layers)",
    )
    effective = sys.argv[1:] if argv is None else argv
    if not effective:
        parser.print_help()
        return 0
    args = parser.parse_args(effective)

    model_dir = args.model_dir.resolve()

    try:
        _validate_model_dir(model_dir)
    except FileNotFoundError as exc:
        return _fail(str(exc))

    return _run_calibration(
        model_dir,
        args.margin_mib,
        save_result=args.save_result,
        quiet=args.quiet if args.save_result else False,
        n_cpu_moe=args.n_cpu_moe,
    )
