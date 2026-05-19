"""VRAM calibration from footprint and ctx-probe variant results."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from results import format_iso_utc, utc_now

VARIANT_FOOTPRINT = "calibration-footprint"
VARIANT_CTX_PROBE = "calibration-ctx-probe"


_IDLE_VRAM_METRIC_KEY = "idle_vram_mb"
_MODEL_MAX_CONTEXT_METRIC_KEY = "model_max_context"


def parse_idle_vram_from_metrics_stdout(text: str) -> int:
    """Parse idle_vram_mb from a default single_test_runner probe stdout block."""
    for line in text.splitlines():
        if not line.startswith(_IDLE_VRAM_METRIC_KEY):
            continue
        value_part = line[len(_IDLE_VRAM_METRIC_KEY) :].strip()
        if not value_part or value_part == "n/a":
            raise ValueError("idle_vram_mb unavailable in probe stdout")
        return int(float(value_part))
    raise ValueError("idle_vram_mb not found in probe stdout")


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
        },
        "probes": {
            "footprint": _probe_ref(footprint_result, tester_root),
            "ctx_probe": _probe_ref(ctx_probe_result, tester_root),
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


def format_summary_lines(summary: dict[str, float | int | None]) -> list[str]:
    """Format calibration summary lines for stdout (README Performance block)."""
    model_max = summary.get("model_max_context")
    if model_max is None:
        model_max_line = "* Model Max Context: n/a"
    else:
        model_max_line = f"* Model Max Context: {model_max} tokens"
    return [
        f"* Estimated Max Context: {summary['estimated_context_max']} tokens",
        model_max_line,
        f"* Model VRAM: {summary['model_vram_mb']} MiB",
        f"* KV VRAM: {summary['kv_vram_mb']} MiB",
        f"* GGUF on disk: {summary['gguf_gb']:.2f} GB",
    ]


def run_variant_subprocess(
    tester_root: Path,
    variant_dir: Path,
    *,
    save_result: bool,
    quiet: bool,
    session_id: str | None = None,
) -> tuple[int, str]:
    """Run single_test_runner for a variant directory; return (returncode, captured output)."""
    runner = tester_root / "single_test_runner.py"
    cmd = [sys.executable, str(runner), str(variant_dir)]
    if save_result:
        cmd.append("--save-result")
    if save_result and quiet:
        cmd.append("--quiet")
    if session_id is not None:
        cmd.extend(["--session-id", session_id])
    proc = subprocess.run(
        cmd,
        cwd=tester_root,
        capture_output=True,
        text=True,
        check=False,
    )
    parts: list[str] = []
    if proc.stdout:
        parts.append(proc.stdout)
    if proc.stderr:
        parts.append(proc.stderr)
    return proc.returncode, "".join(parts)
