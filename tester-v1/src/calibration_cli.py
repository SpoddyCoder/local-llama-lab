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
    parse_idle_vram_from_metrics_stdout,
    parse_model_max_context_from_metrics_stdout,
    read_decode_tok_s_from_result,
    read_idle_vram_from_result,
    read_model_max_context_from_result,
    run_variant_subprocess,
    write_calibration_session_summary,
)
from config import (
    MODEL_YAML,
    config_dir_metadata,
    load_model_config,
    load_server_config,
    parse_context_from_args,
)
from result_layout import find_latest_result
from results import format_compact_utc, utc_now
from vram import query_gpu_total_mb

_DEFAULT_TESTER_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIRNAME = "results"
_CONFIGS_DIRNAME = "configs"
_REFERENCE_DIR = "reference"
_OUTPUT_TAIL_LINES = 40


def _reference_root(tester_root: Path) -> Path:
    return tester_root / _CONFIGS_DIRNAME / _REFERENCE_DIR


def _reference_variant_dir(tester_root: Path, name: str) -> Path:
    return _reference_root(tester_root) / name


def _result_config_dir(model_dir: Path, name: str) -> Path:
    return model_dir / name


def _validate_variant_dir(variant_dir: Path) -> None:
    missing: list[str] = []
    for name in ("server.yaml", "client.yaml"):
        if not (variant_dir / name).is_file():
            missing.append(name)
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Config files missing in {variant_dir}: {names}")


def _validate_model_dir(model_dir: Path, tester_root: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    model_yaml = model_dir / MODEL_YAML
    if not model_yaml.is_file():
        raise FileNotFoundError(f"model config not found: {model_yaml}")
    for variant in (VARIANT_FOOTPRINT, VARIANT_CTX_PROBE, VARIANT_HELLO_WORLD):
        ref_dir = _reference_variant_dir(tester_root, variant)
        if not ref_dir.is_dir():
            raise FileNotFoundError(
                f"reference variant directory not found: {ref_dir} "
                f"(expected configs/reference/{variant}/)"
            )
        _validate_variant_dir(ref_dir)


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _output_tail(text: str, *, max_lines: int = _OUTPUT_TAIL_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def _idle_vram_from_probe_output(output: str, *, variant_dir: Path) -> int:
    try:
        return parse_idle_vram_from_metrics_stdout(output)
    except ValueError as exc:
        raise ValueError(f"{exc} (variant {variant_dir})") from exc


def _print_probe_output(output: str) -> None:
    if not output:
        return
    end = "" if output.endswith("\n") else "\n"
    print(output, end=end, file=sys.stdout)


def _run_calibration(
    model_dir: Path,
    tester_root: Path,
    margin_mib: int,
    *,
    save_result: bool,
    quiet: bool,
) -> int:
    results_dir = tester_root / _RESULTS_DIRNAME
    footprint_result_dir = _result_config_dir(model_dir, VARIANT_FOOTPRINT)
    ctx_probe_result_dir = _result_config_dir(model_dir, VARIANT_CTX_PROBE)
    hello_world_result_dir = _result_config_dir(model_dir, VARIANT_HELLO_WORLD)
    session_id: str | None = None
    if save_result:
        session_id = format_compact_utc(utc_now())

    variants = [
        (
            VARIANT_FOOTPRINT,
            footprint_result_dir,
            _reference_variant_dir(tester_root, VARIANT_FOOTPRINT),
            "Running calibration-footprint...",
        ),
        (
            VARIANT_CTX_PROBE,
            ctx_probe_result_dir,
            _reference_variant_dir(tester_root, VARIANT_CTX_PROBE),
            "Running calibration-ctx-probe...",
        ),
        (
            VARIANT_HELLO_WORLD,
            hello_world_result_dir,
            _reference_variant_dir(tester_root, VARIANT_HELLO_WORLD),
            "Running hello-world-baseline...",
        ),
    ]
    probe_outputs: list[tuple[Path, str]] = []
    for _name, result_dir, reference_dir, progress in variants:
        print(progress, file=sys.stderr)
        returncode, output = run_variant_subprocess(
            tester_root,
            result_dir,
            reference_dir,
            save_result=save_result,
            quiet=save_result and quiet,
            session_id=session_id,
        )
        if returncode != 0:
            tail = _output_tail(output)
            return _fail(
                f"{progress.rstrip('.')} failed (exit {returncode}) for {result_dir}\n"
                f"{tail}"
            )
        if not save_result:
            _print_probe_output(output)
            probe_outputs.append((result_dir, output))

    if save_result:
        try:
            footprint_result = find_latest_result(
                results_dir, footprint_result_dir, tester_root
            )
            ctx_probe_result = find_latest_result(
                results_dir, ctx_probe_result_dir, tester_root
            )
            hello_world_result = find_latest_result(
                results_dir, hello_world_result_dir, tester_root
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
        except ValueError as exc:
            return _fail(str(exc))
    else:
        try:
            footprint_idle = _idle_vram_from_probe_output(
                probe_outputs[0][1], variant_dir=probe_outputs[0][0]
            )
            ctx_probe_idle = _idle_vram_from_probe_output(
                probe_outputs[1][1], variant_dir=probe_outputs[1][0]
            )
            model_max_context = parse_model_max_context_from_metrics_stdout(
                probe_outputs[0][1]
            )
            generation_throughput_tok_s = parse_decode_tok_s_from_metrics_stdout(
                probe_outputs[2][1]
            )
        except ValueError as exc:
            return _fail(str(exc))

    footprint_server_path = (
        _reference_variant_dir(tester_root, VARIANT_FOOTPRINT) / "server.yaml"
    )
    ctx_probe_server_path = (
        _reference_variant_dir(tester_root, VARIANT_CTX_PROBE) / "server.yaml"
    )
    try:
        model_path = load_model_config(model_dir / MODEL_YAML)
        footprint_server = load_server_config(
            footprint_server_path, model=model_path
        )
        ctx_probe_server = load_server_config(ctx_probe_server_path, model=model_path)
        footprint_c = parse_context_from_args(footprint_server.args)
        ctx_c = parse_context_from_args(ctx_probe_server.args)
        gguf_gb = gguf_size_gb(model_path)
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
            margin_mib,
            gguf_gb,
        )
        summary["model_max_context"] = model_max_context
        summary["generation_throughput_tok_s"] = generation_throughput_tok_s
    except ValueError as exc:
        if save_result:
            return _fail(
                f"{exc}\n"
                f"footprint result: {footprint_result}\n"
                f"ctx-probe result: {ctx_probe_result}\n"
                f"footprint reference: {footprint_server_path}\n"
                f"ctx-probe reference: {ctx_probe_server_path}"
            )
        return _fail(
            f"{exc}\n"
            f"footprint reference: {footprint_server_path}\n"
            f"ctx-probe reference: {ctx_probe_server_path}"
        )

    if save_result:
        meta = config_dir_metadata(model_dir, tester_root)
        model = meta["model"] or model_dir.name
        write_calibration_session_summary(
            tester_root,
            model,
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
        help=(
            "model directory containing model.yaml only; calibration probes use "
            "server/client YAML from configs/reference/{variant}/"
        ),
    )
    parser.add_argument(
        "--margin-mib",
        type=int,
        default=DEFAULT_CALIBRATION_MARGIN_MIB,
        help=(
            "VRAM margin reserved for non-KV use before the model loads "
            f"(default: {DEFAULT_CALIBRATION_MARGIN_MIB})"
        ),
    )
    parser.add_argument(
        "--tester-root",
        type=Path,
        default=_DEFAULT_TESTER_ROOT,
        help="tester-v1 root (default: directory containing this package)",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="run probes with --save-result and read idle_vram_mb from results/ JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only applies with --save-result; suppress calibration summary on stdout",
    )
    args = parser.parse_args(argv)

    tester_root = args.tester_root.resolve()
    model_dir = args.model_dir.resolve()

    try:
        _validate_model_dir(model_dir, tester_root)
    except FileNotFoundError as exc:
        return _fail(str(exc))

    return _run_calibration(
        model_dir,
        tester_root,
        args.margin_mib,
        save_result=args.save_result,
        quiet=args.quiet if args.save_result else False,
    )
