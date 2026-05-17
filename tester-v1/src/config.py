"""YAML config loaders and validation for tester-v1."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_MODEL_FLAGS = frozenset({"-m", "--model"})
_DEFAULT_BINARY = "llama-server"
_DEFAULT_READY_TIMEOUT_S = 120.0
_DEFAULT_READY_POLL_INTERVAL_S = 0.5
_DEFAULT_BASE_URL = "http://127.0.0.1:8080"
_DEFAULT_TIMEOUT_S = 600.0
_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_HF_HUB_DIR_MARKER = "/hub/"
_HF_MODELS_DIR_PREFIX = "models--"


def redact_model_path(path: str) -> str:
    """Strip home/cache prefixes from a model path for logs and result JSON."""
    normalized = path.replace("\\", "/")
    if _HF_HUB_DIR_MARKER in normalized:
        idx = normalized.index(_HF_HUB_DIR_MARKER) + len(_HF_HUB_DIR_MARKER)
        return normalized[idx:]
    if _HF_MODELS_DIR_PREFIX in normalized:
        idx = normalized.index(_HF_MODELS_DIR_PREFIX)
        return normalized[idx:]
    return Path(path).name


@dataclass(frozen=True)
class ServerConfig:
    model: str
    args: list[str]
    binary: str = _DEFAULT_BINARY
    ready_timeout_s: float = _DEFAULT_READY_TIMEOUT_S
    ready_poll_interval_s: float = _DEFAULT_READY_POLL_INTERVAL_S

    @property
    def model_slug(self) -> str:
        return _sanitize_slug(Path(self.model).stem)

    @property
    def port(self) -> int | None:
        return parse_port_from_args(self.args)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": redact_model_path(self.model),
            "args": list(self.args),
            "binary": self.binary,
            "ready_timeout_s": self.ready_timeout_s,
            "ready_poll_interval_s": self.ready_poll_interval_s,
        }


def slug_from_config_dir(config_dir: Path, tester_root: Path) -> str:
    """Derive a result filename slug from a config directory path."""
    config_dir = config_dir.resolve()
    configs_root = (tester_root / "configs").resolve()
    try:
        parts = config_dir.relative_to(configs_root).parts
        joined = "-".join(parts)
    except ValueError:
        parts = [p for p in config_dir.parts if p]
        joined = "-".join(parts)
    return _sanitize_slug(joined)


def config_dir_metadata(
    config_dir: Path | None,
    tester_root: Path,
) -> dict[str, str | None]:
    """Derive config path fields for result JSON metadata."""
    empty: dict[str, str | None] = {
        "config_path": None,
        "model": None,
        "variant": None,
    }
    if config_dir is None:
        return empty

    config_dir = config_dir.resolve()
    configs_root = (tester_root / "configs").resolve()
    try:
        parts = config_dir.relative_to(configs_root).parts
    except ValueError:
        return empty

    if len(parts) != 2:
        return empty

    model = None if parts[0] == "reference" else parts[0]
    return {
        "config_path": f"{parts[0]}/{parts[1]}/",
        "model": model,
        "variant": parts[1],
    }


@dataclass(frozen=True)
class ClientConfig:
    messages: list[dict[str, Any]]
    base_url: str = _DEFAULT_BASE_URL
    params: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = _DEFAULT_TIMEOUT_S

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "messages": self.messages,
            "params": dict(self.params),
            "timeout_s": self.timeout_s,
        }


def parse_args_block(text: str, path: str | Path | None = None) -> list[str]:
    """Parse server args from a YAML block scalar.

    One flag per line; space-separated flag and value (e.g. ``--host 127.0.0.1``).
    Flag-only lines like ``--no-mmap`` are OK.
    """
    if not text or not text.strip():
        return []
    result: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            msg = (
                f"server config args: use space-separated flags, not flag=value: "
                f"{stripped!r}"
            )
            if path is not None:
                msg += f" ({path})"
            raise ValueError(msg)
        result.extend(shlex.split(stripped))
    return result


def load_server_config(path: str | Path) -> ServerConfig:
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"server config must be a YAML mapping: {path}")

    model_raw = raw.get("model")
    if not model_raw or not isinstance(model_raw, str):
        raise ValueError(f"server config 'model' is required and must be a string: {path}")

    model = os.path.expanduser(model_raw)
    if not os.path.isfile(model):
        raise ValueError(f"server config model file does not exist: {model}")

    args_raw = raw.get("args")
    if args_raw is None:
        args = []
    elif isinstance(args_raw, list):
        raise ValueError(
            f"server config 'args' must be a multiline string (YAML block scalar): {path}"
        )
    elif not isinstance(args_raw, str):
        raise ValueError(
            f"server config 'args' must be a multiline string (YAML block scalar): {path}"
        )
    else:
        args = parse_args_block(args_raw, path)
    _validate_args_no_model_flag(args, path)

    if "label" in raw:
        raise ValueError(
            f"'label' is removed; use configs/... directory layout "
            f"for result filenames: {path}"
        )

    binary = raw.get("binary", _DEFAULT_BINARY)
    if not isinstance(binary, str) or not binary.strip():
        raise ValueError(f"server config 'binary' must be a non-empty string: {path}")

    ready_timeout_s = _coerce_positive_float(
        raw.get("ready_timeout_s", _DEFAULT_READY_TIMEOUT_S),
        "ready_timeout_s",
        path,
        "server",
    )
    ready_poll_interval_s = _coerce_positive_float(
        raw.get("ready_poll_interval_s", _DEFAULT_READY_POLL_INTERVAL_S),
        "ready_poll_interval_s",
        path,
        "server",
    )

    return ServerConfig(
        model=model,
        args=args,
        binary=binary.strip(),
        ready_timeout_s=ready_timeout_s,
        ready_poll_interval_s=ready_poll_interval_s,
    )


def load_client_config(path: str | Path) -> ClientConfig:
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"client config must be a YAML mapping: {path}")

    messages = raw.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError(f"client config 'messages' must be a non-empty list: {path}")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"client config messages[{i}] must be a mapping: {path}")

    base_url = raw.get("base_url", _DEFAULT_BASE_URL)
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError(f"client config 'base_url' must be a non-empty string: {path}")

    params = raw.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(f"client config 'params' must be a mapping: {path}")

    timeout_s = _coerce_positive_float(
        raw.get("timeout_s", _DEFAULT_TIMEOUT_S),
        "timeout_s",
        path,
        "client",
    )

    return ClientConfig(
        messages=messages,
        base_url=base_url.rstrip("/"),
        params=dict(params),
        timeout_s=timeout_s,
    )


def parse_context_from_args(args: list[str]) -> int:
    """Return context size from -c / --ctx-size flags in llama-server args."""
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-c", "--ctx-size"):
            if i + 1 >= len(args):
                raise ValueError("server config args: -c / --ctx-size requires a value")
            try:
                ctx = int(args[i + 1])
            except ValueError as exc:
                raise ValueError(
                    f"server config args: invalid context value {args[i + 1]!r}"
                ) from exc
            if ctx < 1:
                raise ValueError(
                    f"server config args: context size must be positive: {ctx}"
                )
            return ctx
        if arg.startswith("-c="):
            try:
                ctx = int(arg.split("=", 1)[1])
            except ValueError as exc:
                raise ValueError(f"server config args: invalid context in {arg!r}") from exc
            if ctx < 1:
                raise ValueError(
                    f"server config args: context size must be positive: {ctx}"
                )
            return ctx
        if arg.startswith("--ctx-size="):
            try:
                ctx = int(arg.split("=", 1)[1])
            except ValueError as exc:
                raise ValueError(
                    f"server config args: invalid context in {arg!r}"
                ) from exc
            if ctx < 1:
                raise ValueError(
                    f"server config args: context size must be positive: {ctx}"
                )
            return ctx
        i += 1
    raise ValueError(
        "server config args: context size (-c / --ctx-size) is required but not found"
    )


def parse_port_from_args(args: list[str]) -> int | None:
    """Return port from --port / -p flags in llama-server args, or None."""
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--port", "-p"):
            if i + 1 >= len(args):
                raise ValueError("server config args: --port / -p requires a value")
            try:
                port = int(args[i + 1])
            except ValueError as exc:
                raise ValueError(
                    f"server config args: invalid port value {args[i + 1]!r}"
                ) from exc
            if port < 1 or port > 65535:
                raise ValueError(f"server config args: port out of range: {port}")
            return port
        if arg.startswith("--port="):
            try:
                port = int(arg.split("=", 1)[1])
            except ValueError as exc:
                raise ValueError(f"server config args: invalid port in {arg!r}") from exc
            if port < 1 or port > 65535:
                raise ValueError(f"server config args: port out of range: {port}")
            return port
        i += 1
    return None


def _load_yaml(path: str | Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_args_no_model_flag(args: list[str], path: str | Path) -> None:
    for arg in args:
        if arg in _MODEL_FLAGS or arg.startswith("--model="):
            raise ValueError(
                f"server config 'args' must not contain -m or --model "
                f"(runner injects model): {path}"
            )


def _sanitize_slug(value: str) -> str:
    slug = _SLUG_RE.sub("_", value.strip())
    slug = slug.strip("._-")
    return slug or "model"


def _coerce_positive_float(
    value: Any,
    field_name: str,
    path: str | Path,
    kind: str,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{kind} config '{field_name}' must be a positive number: {path}"
        ) from exc
    if result <= 0:
        raise ValueError(
            f"{kind} config '{field_name}' must be a positive number: {path}"
        )
    return result
