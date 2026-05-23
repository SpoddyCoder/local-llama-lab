"""Build result JSON documents, write files, and format stdout summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import ClientConfig, ServerConfig, redact_model_path
from metadata import empty_metadata
from metrics import format_probe_stdout


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


def make_run_id(timestamp_utc: datetime, run_id_suffix: str) -> str:
    """e.g. 20260516T134500Z_Qwen_Qwen3.5-9B-Q8_0"""
    return f"{format_compact_utc(timestamp_utc)}_{run_id_suffix}"


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
        "idle_system_ram_mb": None,
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
    metadata: dict[str, str | None] | None = None,
    config_dir: Path | None = None,
    tester_root: Path | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    meta = dict(metadata) if metadata is not None else empty_metadata()
    for key in empty_metadata():
        meta.setdefault(key, None)

    if config_dir is not None and tester_root is not None:
        from result_layout import (
            resolve_result_target,
            run_id_from_started_at,
            suite_from_variant,
        )

        target = resolve_result_target(config_dir, tester_root, started_at)
        run_id = run_id_from_started_at(started_at, target.run_id_suffix)
        suite = suite_from_variant(
            meta.get("variant") if isinstance(meta.get("variant"), str) else None
        )
    else:
        run_id = make_run_id(started_at, server_config.model_slug)
        suite = "probe"

    metrics = dict(metrics_dict) if metrics_dict is not None else empty_metrics()
    if "server_ready_s" not in metrics:
        metrics["server_ready_s"] = None
    if "idle_vram_mb" not in metrics:
        metrics["idle_vram_mb"] = None
    if "idle_system_ram_mb" not in metrics:
        metrics["idle_system_ram_mb"] = None
    if "peak_vram_mb" not in metrics:
        metrics["peak_vram_mb"] = None

    return {
        "schema_version": "1",
        "run_id": run_id,
        "suite": suite,
        "session_id": session_id,
        "started_at": format_iso_utc(started_at),
        "finished_at": format_iso_utc(finished_at),
        "server_config": server_config.to_dict(),
        "client_config": client_config.to_dict(),
        "metrics": metrics,
        "metadata": meta,
        "status": status,
        "error": error,
        "notes": notes,
    }


def write_result(
    results_dir: str | Path,
    document: dict[str, Any],
    *,
    config_dir: Path,
    tester_root: Path,
    started_at: datetime,
) -> Path:
    """Write document to hierarchical result path."""
    from result_layout import resolve_result_target

    target = resolve_result_target(config_dir, tester_root, started_at)
    json_path = target.json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
        f.write("\n")
    return json_path


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
        f"Model: {redact_model_path(server_config.model)}",
        f"Result: {result_display}",
        "",
        format_probe_stdout(metrics_dict),
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
