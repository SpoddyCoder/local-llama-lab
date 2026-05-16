"""Streaming OpenAI-compatible chat client and SSE parsing."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterator, Literal

import httpx

from config import ClientConfig

_SSE_DONE = "[DONE]"
# OpenAI content plus llama.cpp variants (e.g. Qwen reasoning streams).
_CONTENT_DELTA_KEYS = ("content", "reasoning_content", "text")


@dataclass(frozen=True)
class CompletionResult:
    wall_time_s: float
    ttft_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    completion_text: str = ""


def run_chat_completion(
    client_config: ClientConfig,
    *,
    base_url: str | None = None,
) -> CompletionResult:
    """POST streaming chat completion; return timings and token usage."""
    url = f"{(base_url or client_config.base_url).rstrip('/')}/v1/chat/completions"
    body: dict[str, Any] = {
        "messages": client_config.messages,
        **client_config.params,
        "stream": True,
    }
    timeout = httpx.Timeout(client_config.timeout_s)

    start = time.monotonic()
    ttft_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    completion_parts: list[str] = []

    try:
        with httpx.Client(timeout=timeout) as http:
            with http.stream("POST", url, json=body) as response:
                if response.is_error:
                    detail = _read_error_body(response)
                    raise RuntimeError(
                        f"chat completion failed: HTTP {response.status_code}"
                        + (f" — {detail}" if detail else "")
                    )

                for event in parse_sse_stream(response.iter_lines()):
                    if event is _SSE_DONE:
                        break
                    chunk = event

                    content = extract_content_delta(chunk)
                    if content and ttft_s is None:
                        ttft_s = time.monotonic() - start
                    if content:
                        completion_parts.append(content)

                    usage = extract_usage(chunk)
                    if usage is not None:
                        prompt_tokens, completion_tokens, total_tokens = usage

    except httpx.TimeoutException as exc:
        raise TimeoutError(
            f"chat completion timed out after {client_config.timeout_s}s"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"chat completion request failed: {exc}") from exc

    wall_time_s = time.monotonic() - start
    return CompletionResult(
        wall_time_s=wall_time_s,
        ttft_s=ttft_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        completion_text="".join(completion_parts),
    )


def parse_sse_stream(lines: Iterator[str]) -> Iterator[dict[str, Any] | Literal["[DONE]"]]:
    """Yield parsed JSON chunks from SSE lines; yield '[DONE]' once when seen."""
    for line in lines:
        parsed = parse_sse_line(line)
        if parsed is None:
            continue
        if parsed is _SSE_DONE:
            yield _SSE_DONE
            return
        yield parsed


def parse_sse_line(line: str) -> dict[str, Any] | Literal["[DONE]"] | None:
    """Parse one SSE line. Returns None for ignorable lines."""
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if not stripped.startswith("data:"):
        raise ValueError(f"malformed SSE line (expected data:): {line!r}")

    payload = stripped[5:].lstrip()
    if not payload:
        return None
    if payload == _SSE_DONE:
        return _SSE_DONE

    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed SSE JSON: {payload[:200]!r}") from exc

    if not isinstance(chunk, dict):
        raise ValueError(f"malformed SSE chunk (expected object): {type(chunk).__name__}")
    return chunk


def extract_content_delta(chunk: dict[str, Any]) -> str | None:
    """Return first non-empty text from choices[0].delta (content or equivalent)."""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return None
    for key in _CONTENT_DELTA_KEYS:
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def extract_usage(
    chunk: dict[str, Any],
) -> tuple[int | None, int | None, int | None] | None:
    """Return token counts from usage or llama-server timings on the chunk."""
    usage = chunk.get("usage")
    if isinstance(usage, dict):
        prompt = _coerce_int(usage.get("prompt_tokens"))
        completion = _coerce_int(usage.get("completion_tokens"))
        total = _coerce_int(usage.get("total_tokens"))
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        return prompt, completion, total

    timings = chunk.get("timings")
    if isinstance(timings, dict):
        prompt = _coerce_int(timings.get("prompt_n"))
        completion = _coerce_int(timings.get("predicted_n"))
        total = None
        if prompt is not None and completion is not None:
            total = prompt + completion
        return prompt, completion, total

    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_error_body(response: httpx.Response) -> str:
    try:
        response.read()
        text = response.text.strip()
        if len(text) > 500:
            return text[:500] + "..."
        return text
    except httpx.HTTPError:
        return ""
