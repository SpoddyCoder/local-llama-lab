"""Build result JSON documents, write files, and format stdout summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import ClientConfig, ServerConfig
from metrics import format_metrics_summary


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_compact_utc(dt: datetime) -> str:
    """Compact UTC timestamp for run_id filenames: YYYYMMDDTHHMMSSZ."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def format_iso_utc(dt: datetime) -> str:
    """ISO 8601 UTC timestamp for JSON fields: YYYY-MM-DDTHH:MM:SSZ."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(timestamp_utc: datetime, model_slug: str) -> str:
    """e.g. 20260516T134500Z_Qwen_Qwen3.5-9B-Q8_0"""
    return f"{format_compact_utc(timestamp_utc)}_{model_slug}"


def empty_metrics() -> dict[str, float | int | None]:
    return {
        "wall_time_s": None,
        "ttft_s": None,
        "completion_time_s": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "prefill_tok_s": None,
        "decode_tok_s": None,
        "tokens_per_second": None,
        "server_ready_s": None,
        "idle_vram_mb": None,
        "peak_vram_mb": None,
    }


def build_result_document(
    *,
    server_config: ServerConfig,
    client_config: ClientConfig,
    metrics_dict: dict[str, float | int | None] | None,
    status: str,
    started_at: datetime,
    finished_at: datetime,
    error: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    run_id = make_run_id(started_at, server_config.model_slug)
    metrics = dict(metrics_dict) if metrics_dict is not None else empty_metrics()
    if "server_ready_s" not in metrics:
        metrics["server_ready_s"] = None
    if "idle_vram_mb" not in metrics:
        metrics["idle_vram_mb"] = None
    if "peak_vram_mb" not in metrics:
        metrics["peak_vram_mb"] = None

    return {
        "run_id": run_id,
        "started_at": format_iso_utc(started_at),
        "finished_at": format_iso_utc(finished_at),
        "server_config": server_config.to_dict(),
        "client_config": client_config.to_dict(),
        "metrics": metrics,
        "status": status,
        "error": error,
        "notes": notes,
    }


def write_result(results_dir: str | Path, document: dict[str, Any]) -> Path:
    """Write document to results/{run_id}.json; create directory if needed."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = document["run_id"]
    path = results_dir / f"{run_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
        f.write("\n")
    return path


def format_run_summary(
    run_id: str,
    result_path: Path,
    server_config: ServerConfig,
    metrics_dict: dict[str, float | int | None],
    *,
    tester_root: Path | None = None,
) -> str:
    """Human-readable stdout block after a successful run."""
    if tester_root is not None:
        try:
            rel = result_path.relative_to(tester_root)
            result_display = str(rel)
        except ValueError:
            result_display = str(result_path)
    else:
        result_display = str(result_path)

    lines = [
        f"Run: {run_id}",
        f"Model: {server_config.model}",
        f"Result: {result_display}",
        "",
        format_metrics_summary(metrics_dict),
    ]
    return "\n".join(lines)


def format_error_summary(
    error: str,
    result_path: Path | None,
    *,
    tester_root: Path | None = None,
) -> str:
    lines = [f"Error: {error}"]
    if result_path is not None:
        if tester_root is not None:
            try:
                rel = result_path.relative_to(tester_root)
                result_display = str(rel)
            except ValueError:
                result_display = str(result_path)
        else:
            result_display = str(result_path)
        lines.append(f"Result: {result_display} (partial)")
    return "\n".join(lines)
