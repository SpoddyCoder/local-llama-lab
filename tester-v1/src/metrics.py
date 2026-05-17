"""Assemble metrics dict from a streaming completion result."""

from __future__ import annotations

from client import CompletionResult


def build_metrics_dict(result: CompletionResult) -> dict[str, float | int | None]:
    wall = result.wall_time_s
    ttft = result.ttft_s
    prompt = result.prompt_tokens
    completion = result.completion_tokens
    total = result.total_tokens

    completion_time_s: float | None = None
    if ttft is not None:
        completion_time_s = wall - ttft

    prefill_tok_s: float | None = None
    if ttft is not None and ttft > 0 and prompt is not None:
        prefill_tok_s = prompt / ttft

    decode_tok_s: float | None = None
    if (
        completion_time_s is not None
        and completion_time_s > 0
        and completion is not None
    ):
        decode_tok_s = completion / completion_time_s

    tokens_per_second: float | None = None
    if wall > 0 and completion is not None:
        tokens_per_second = completion / wall

    return {
        "wall_time_s": wall,
        "ttft_s": ttft,
        "completion_time_s": completion_time_s,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "tokens_per_second": tokens_per_second,
        "server_ready_s": None,
        "idle_vram_mb": None,
        "peak_vram_mb": None,
    }


def format_metrics_summary(metrics: dict[str, float | int | None]) -> str:
    """Human-readable metrics block matching the Phase 1 stdout plan."""

    def _fmt(key: str, width: int = 18) -> str:
        value = metrics.get(key)
        if value is None:
            text = "n/a"
        elif isinstance(value, float):
            text = f"{value:.2f}"
        else:
            text = str(value)
        return f"{key:<{width}}{text}"

    lines = [
        _fmt("server_ready_s"),
        _fmt("wall_time_s"),
        _fmt("ttft_s"),
        _fmt("completion_time_s"),
        _fmt("prompt_tokens"),
        _fmt("completion_tokens"),
        _fmt("total_tokens"),
        _fmt("prefill_tok_s"),
        _fmt("decode_tok_s"),
        _fmt("tokens_per_second"),
        _fmt("idle_vram_mb"),
        _fmt("peak_vram_mb"),
    ]
    return "\n".join(lines)


def format_headline_summary(metrics: dict[str, float | int | None]) -> str:
    """Short human-oriented summary (wall time, peak VRAM, decode throughput)."""

    def _headline(label: str, text: str, width: int = 24) -> str:
        return f"{label:<{width}}{text}"

    wall = metrics.get("wall_time_s")
    if wall is None:
        e2e = "n/a"
    else:
        e2e = f"{wall:.2f} s"

    peak_mb = metrics.get("peak_vram_mb")
    if peak_mb is None:
        peak = "n/a"
    else:
        peak = f"{peak_mb / 1024:.2f} GiB"

    decode = metrics.get("decode_tok_s")
    if decode is None:
        throughput = "n/a"
    else:
        throughput = f"{decode:.2f} tok/s"

    lines = [
        _headline("End-to-end time", e2e),
        _headline("Peak VRAM", peak),
        _headline("Generation throughput", throughput),
    ]
    return "\n".join(lines)


def format_probe_stdout(metrics: dict[str, float | int | None]) -> str:
    """Metrics block plus headline summary for probe stdout."""
    return format_metrics_summary(metrics) + "\n\n" + format_headline_summary(metrics)
