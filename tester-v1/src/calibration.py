"""VRAM calibration from footprint and ctx-probe variant results."""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
from pathlib import Path

from model_layout import resolve_run_config
from results import format_iso_utc, utc_now
from runner import _run

VARIANT_FOOTPRINT = "calibration-footprint"
VARIANT_CTX_PROBE = "calibration-ctx-probe"
VARIANT_HELLO_WORLD = "hello-world-baseline"

# Small VRAM safety buffer subtracted from the KV budget (see --margin-mib).
DEFAULT_CALIBRATION_MARGIN_MIB = 100


_IDLE_VRAM_METRIC_KEY = "idle_vram_mb"
_IDLE_SYSTEM_RAM_METRIC_KEY = "idle_system_ram_mb"
_MODEL_MAX_CONTEXT_METRIC_KEY = "model_max_context"
_DECODE_TOK_S_METRIC_KEY = "decode_tok_s"


def parse_idle_vram_from_metrics_stdout(text: str) -> int:
    """Parse idle_vram_mb from a default probe stdout block."""
    for line in text.splitlines():
        if not line.startswith(_IDLE_VRAM_METRIC_KEY):
            continue
        value_part = line[len(_IDLE_VRAM_METRIC_KEY) :].strip()
        if not value_part or value_part == "n/a":
            raise ValueError("idle_vram_mb unavailable in probe stdout")
        return int(float(value_part))
    raise ValueError("idle_vram_mb not found in probe stdout")


def parse_idle_system_ram_from_metrics_stdout(text: str) -> int | None:
    """Parse idle_system_ram_mb from probe stdout; None if missing or n/a."""
    for line in text.splitlines():
        if not line.startswith(_IDLE_SYSTEM_RAM_METRIC_KEY):
            continue
        value_part = line[len(_IDLE_SYSTEM_RAM_METRIC_KEY) :].strip()
        if not value_part or value_part == "n/a":
            return None
        return int(float(value_part))
    return None


def parse_model_max_context_from_metrics_stdout(text: str) -> int | None:
    """Parse model_max_context from probe stdout; None if missing or n/a."""
    for line in text.splitlines():
        if not line.startswith(_MODEL_MAX_CONTEXT_METRIC_KEY):
            continue
        value_part = line[len(_MODEL_MAX_CONTEXT_METRIC_KEY) :].strip()
        if not value_part or value_part == "n/a":
            return None
        return int(float(value_part))
    return None


def parse_decode_tok_s_from_metrics_stdout(text: str) -> float | None:
    """Parse decode_tok_s from probe stdout; None if missing or n/a."""
    for line in text.splitlines():
        if not line.startswith(_DECODE_TOK_S_METRIC_KEY):
            continue
        value_part = line[len(_DECODE_TOK_S_METRIC_KEY) :].strip()
        if not value_part or value_part == "n/a":
            return None
        return float(value_part)
    return None


def read_model_max_context_from_result(path: Path) -> int | None:
    """Load model_max_context from a result JSON metrics; None if absent."""
    with path.open(encoding="utf-8") as f:
        document = json.load(f)
    if document.get("status") != "ok":
        return None
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("model_max_context")
    if value is None:
        return None
    return int(value)


def read_decode_tok_s_from_result(path: Path) -> float | None:
    """Load decode_tok_s from a result JSON metrics; None if absent."""
    with path.open(encoding="utf-8") as f:
        document = json.load(f)
    if document.get("status") != "ok":
        return None
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("decode_tok_s")
    if value is None:
        return None
    return float(value)


def read_idle_vram_from_result(path: Path) -> int:
    """Load idle_vram_mb from a result JSON; require ok status."""
    with path.open(encoding="utf-8") as f:
        document = json.load(f)
    if document.get("status") != "ok":
        raise ValueError(f"result status is not ok: {path}")
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"result metrics missing: {path}")
    idle = metrics.get("idle_vram_mb")
    if idle is None:
        raise ValueError(f"idle_vram_mb missing in result: {path}")
    return int(idle)


def read_idle_system_ram_from_result(path: Path) -> int | None:
    """Load idle_system_ram_mb from result JSON metrics; None if absent or null."""
    with path.open(encoding="utf-8") as f:
        document = json.load(f)
    if document.get("status") != "ok":
        return None
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("idle_system_ram_mb")
    if value is None:
        return None
    return int(value)


def _relative_result_path(path: Path, tester_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(tester_root.resolve()))
    except ValueError:
        return str(path)


def _probe_ref(path: Path, tester_root: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        document = json.load(f)
    run_id = document.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"run_id missing in result: {path}")
    return {"run_id": run_id, "path": _relative_result_path(path, tester_root)}


def write_calibration_session_summary(
    tester_root: Path,
    model: str,
    session_id: str,
    summary: dict[str, float | int | None],
    footprint_result: Path,
    ctx_probe_result: Path,
    hello_world_result: Path,
) -> Path:
    """Write calibration session summary JSON under results/{model}/calibration-sessions/."""
    tester_root = tester_root.resolve()
    out_dir = tester_root / "results" / model / "calibration-sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}.json"
    document = {
        "schema_version": "1",
        "session_id": session_id,
        "model": model,
        "created_at": format_iso_utc(utc_now()),
        "summary": {
            "gguf_gb": summary["gguf_gb"],
            "model_vram_mb": summary["model_vram_mb"],
            "kv_vram_mb": summary["kv_vram_mb"],
            "estimated_context_max": summary["estimated_context_max"],
            "model_max_context": summary.get("model_max_context"),
            "generation_throughput_tok_s": summary.get("generation_throughput_tok_s"),
            **(
                {"model_system_ram_mb": summary["model_system_ram_mb"]}
                if "model_system_ram_mb" in summary
                else {}
            ),
        },
        "probes": {
            "footprint": _probe_ref(footprint_result, tester_root),
            "ctx_probe": _probe_ref(ctx_probe_result, tester_root),
            "hello_world": _probe_ref(hello_world_result, tester_root),
        },
    }
    out_path.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_path


def gguf_size_gb(model_path: str) -> float:
    """Return GGUF file size in gibibytes (GB label, bytes / 1e9)."""
    expanded = os.path.expanduser(model_path)
    size_bytes = os.path.getsize(expanded)
    return size_bytes / 1_000_000_000


def compute_summary(
    footprint_idle: int,
    ctx_probe_idle: int,
    footprint_c: int,
    ctx_c: int,
    gpu_total_mb: int,
    margin_mb: int,
    gguf_gb: float,
) -> dict[str, float | int]:
    """Derive calibration summary fields from probe idle VRAM and contexts."""
    if ctx_c <= footprint_c:
        raise ValueError(
            f"ctx-probe context ({ctx_c}) must be greater than footprint context "
            f"({footprint_c})"
        )

    delta_idle = ctx_probe_idle - footprint_idle
    delta_ctx = ctx_c - footprint_c
    if delta_idle <= 0:
        raise ValueError(
            f"KV VRAM slope is non-positive: ctx-probe idle {ctx_probe_idle} MiB "
            f"<= footprint idle {footprint_idle} MiB"
        )

    kv_mib_per_1k = delta_idle / delta_ctx * 1024
    if kv_mib_per_1k <= 0:
        raise ValueError(
            f"KV VRAM slope is non-positive: {kv_mib_per_1k} MiB per 1k tokens"
        )

    kv_vram_mb = gpu_total_mb - footprint_idle - margin_mb
    if kv_vram_mb < 0:
        raise ValueError(
            f"KV VRAM budget is negative ({kv_vram_mb} MiB): gpu_total "
            f"{gpu_total_mb} - footprint_idle {footprint_idle} - margin {margin_mb}"
        )

    extra_tokens = kv_vram_mb / kv_mib_per_1k * 1024
    estimated_context_max = footprint_c + math.floor(extra_tokens)

    return {
        "gguf_gb": gguf_gb,
        "model_vram_mb": footprint_idle,
        "kv_vram_mb": kv_vram_mb,
        "estimated_context_max": estimated_context_max,
    }


def format_mib_as_gb(mib: int | float) -> str:
    """Format MiB as XX.YY GB for README-style calibration output."""
    return f"{mib / 1024:.2f} GB"


def format_generation_throughput_line(decode_tok_s: float | None) -> str:
    """Format README-style generation throughput line."""
    if decode_tok_s is None:
        return "* Generation throughput: n/a"
    return f"* Generation throughput: ~{round(decode_tok_s)} tok/s"


def format_summary_lines(summary: dict[str, float | int | None]) -> list[str]:
    """Format calibration summary lines for stdout (README Performance block)."""
    model_max = summary.get("model_max_context")
    if model_max is None:
        model_max_line = "* Model Max Context: n/a"
    else:
        model_max_line = f"* Model Max Context: {model_max} tokens"
    decode = summary.get("generation_throughput_tok_s")
    if decode is not None and not isinstance(decode, (int, float)):
        decode = None
    elif decode is not None:
        decode = float(decode)
    lines = [
        format_generation_throughput_line(decode),
        f"* Estimated Max Context: {summary['estimated_context_max']} tokens",
        model_max_line,
        f"* Model VRAM: {format_mib_as_gb(summary['model_vram_mb'])}",
    ]
    if "model_system_ram_mb" in summary:
        ram = summary["model_system_ram_mb"]
        if ram is None:
            lines.append("* Model System RAM: unavailable")
        else:
            lines.append(f"* Model System RAM: {format_mib_as_gb(int(ram))}")
    lines.extend(
        [
            f"* KV VRAM: {format_mib_as_gb(summary['kv_vram_mb'])}",
            f"* GGUF on disk: {summary['gguf_gb']:.2f} GB",
        ]
    )
    return lines


def run_calibration_variant(
    model_dir: Path,
    variant: str,
    *,
    save_result: bool,
    quiet: bool,
    session_id: str | None = None,
    n_cpu_moe: int | None = None,
) -> tuple[int, str]:
    """Run a calibration probe in-process via resolve_run_config; return (returncode, captured output)."""
    run_config = resolve_run_config(model_dir, variant, n_cpu_moe=n_cpu_moe)
    captured = io.StringIO()
    if save_result and quiet:
        with contextlib.redirect_stdout(captured):
            returncode = _run(
                run_config,
                save_result=save_result,
                quiet=quiet,
                session_id=session_id,
            )
    elif not save_result:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            returncode = _run(
                run_config,
                save_result=save_result,
                quiet=False,
                session_id=session_id,
            )
    else:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            returncode = _run(
                run_config,
                save_result=save_result,
                quiet=quiet,
                session_id=session_id,
            )
    return returncode, captured.getvalue()
