"""YAML config loaders and validation for probe."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

MODEL_YAML = "model.yaml"
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


def run_metadata(model: str, variant: str) -> dict[str, str | None]:
    """Derive config path fields for result JSON metadata."""
    return {
        "config_path": f"models/{model}/",
        "model": model,
        "variant": variant,
    }


def slug_from_run(model: str, variant: str) -> str:
    """Derive a sanitized result filename slug from model and variant names.

    Nested variants (e.g. ``foo/bar``) normalize ``/`` to ``-`` before
    sanitization so the slug matches ``run_id_suffix`` style.
    """
    suffix = f"{model}-{variant.replace('/', '-')}"
    return _sanitize_slug(suffix)


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


def load_model_config(path: str | Path) -> str:
    """Load and validate the GGUF path from model.yaml."""
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"model config must be a YAML mapping: {path}")

    model_raw = raw.get("model")
    if not model_raw or not isinstance(model_raw, str):
        raise ValueError(f"model config 'model' is required and must be a string: {path}")

    model = os.path.expanduser(model_raw)
    if not os.path.isfile(model):
        raise ValueError(f"model config model file does not exist: {model} ({path})")
    return model


def load_server_config(path: str | Path, *, model: str) -> ServerConfig:
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"server config must be a YAML mapping: {path}")

    if "model" in raw:
        raise ValueError(
            f"server config must not contain 'model'; use {MODEL_YAML} "
            f"in the model directory: {path}"
        )

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
            f"'label' is removed; use models/... directory layout "
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


def load_variant_server(server_path: Path, model_yaml_path: Path) -> ServerConfig:
    """Load server config with model path from model.yaml."""
    model = load_model_config(model_yaml_path)
    return load_server_config(server_path, model=model)


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


_FLAG_ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"-c", "--ctx-size"}),
    frozenset({"-p", "--port"}),
)
_VALUE_FLAGS_WITH_EQUALS = (
    "-c",
    "--ctx-size",
    "--n-cpu-moe",
    "--n-gpu-layers",
    "--host",
    "--port",
    "-p",
    "--parallel",
    "--fit",
)
_BOOLEAN_SERVER_FLAGS = frozenset({"--no-mmap"})


def _aliases_for_flag(flag: str) -> frozenset[str]:
    for group in _FLAG_ALIAS_GROUPS:
        if flag in group:
            return group
    return frozenset({flag})


def replace_flag_args(
    args: list[str],
    replacements: dict[str, str | None],
) -> list[str]:
    """Return args with selected flags removed and optionally re-appended.

    Keys in *replacements* are flag names (e.g. ``--ctx-size``, ``-c``).
    Value ``None`` removes the flag; a string value appends ``[flag, value]``
    at the end (or just ``flag`` for boolean flags when value is ``""``).
    ``-c`` / ``--ctx-size`` and ``-p`` / ``--port`` are treated as aliases.
    """
    flags_to_remove: set[str] = set()
    to_append: list[tuple[str, str | None]] = []
    for flag, value in replacements.items():
        flags_to_remove.update(_aliases_for_flag(flag))
        to_append.append((flag, value))

    result: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        removed = False
        for flag in flags_to_remove:
            if arg == flag:
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    i += 2
                else:
                    i += 1
                removed = True
                break
            prefix = f"{flag}="
            if arg.startswith(prefix):
                i += 1
                removed = True
                break
        if not removed:
            result.append(arg)
            i += 1

    for flag, value in to_append:
        if value is None:
            continue
        if value == "":
            result.append(flag)
        else:
            result.extend([flag, value])
    return result


def _extract_flag_replacements_from_args(args: list[str]) -> dict[str, str | None]:
    """Parse known llama-server flags from args into a replacements dict."""
    replacements: dict[str, str | None] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-c", "--ctx-size", "--port", "-p", "--n-cpu-moe", "--n-gpu-layers",
                   "--host", "--parallel", "--fit"):
            if i + 1 >= len(args):
                raise ValueError(f"server config args: {arg} requires a value")
            replacements[arg] = args[i + 1]
            i += 2
            continue
        matched_equals = False
        for flag in _VALUE_FLAGS_WITH_EQUALS:
            prefix = f"{flag}="
            if arg.startswith(prefix):
                replacements[flag] = arg[len(prefix):]
                i += 1
                matched_equals = True
                break
        if matched_equals:
            continue
        if arg in _BOOLEAN_SERVER_FLAGS:
            replacements[arg] = ""
            i += 1
            continue
        i += 1
    return replacements


def merge_server_configs(
    base: ServerConfig,
    override: ServerConfig | None,
    *,
    n_cpu_moe: int | None = None,
) -> ServerConfig:
    """Merge server args: override YAML flags on base; CLI *n_cpu_moe* wins last."""
    merged_args = list(base.args)
    if override is not None:
        merged_args = replace_flag_args(
            merged_args,
            _extract_flag_replacements_from_args(override.args),
        )
    if n_cpu_moe is not None:
        if n_cpu_moe <= 0:
            raise ValueError(
                f"n_cpu_moe must be a positive integer, got {n_cpu_moe}"
            )
        merged_args = replace_flag_args(
            merged_args,
            {
                "--n-cpu-moe": str(n_cpu_moe),
                "--n-gpu-layers": "999",
            },
        )
    return replace(base, args=merged_args)


def parse_n_cpu_moe_from_args(args: list[str]) -> int | None:
    """Return MoE CPU offload count from --n-cpu-moe in llama-server args, or None."""
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--n-cpu-moe":
            if i + 1 >= len(args):
                raise ValueError("server config args: --n-cpu-moe requires a value")
            try:
                n = int(args[i + 1])
            except ValueError as exc:
                raise ValueError(
                    f"server config args: invalid --n-cpu-moe value {args[i + 1]!r}"
                ) from exc
            if n < 1:
                raise ValueError(
                    f"server config args: --n-cpu-moe must be positive: {n}"
                )
            return n
        if arg.startswith("--n-cpu-moe="):
            try:
                n = int(arg.split("=", 1)[1])
            except ValueError as exc:
                raise ValueError(
                    f"server config args: invalid --n-cpu-moe in {arg!r}"
                ) from exc
            if n < 1:
                raise ValueError(
                    f"server config args: --n-cpu-moe must be positive: {n}"
                )
            return n
        i += 1
    return None


def apply_n_cpu_moe(server: ServerConfig, n: int) -> ServerConfig:
    """Set MoE CPU offload flags on server args (replace if already present)."""
    if n <= 0:
        raise ValueError(f"n_cpu_moe must be a positive integer, got {n}")
    return replace(
        server,
        args=replace_flag_args(
            server.args,
            {
                "--n-cpu-moe": str(n),
                "--n-gpu-layers": "999",
            },
        ),
    )


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
