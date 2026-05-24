"""YAML config loaders for llama-bench."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from config import MODEL_YAML, load_model_config, parse_args_block, replace_flag_args

LLAMA_BENCH_YAML = "llama_bench.yaml"
_DEFAULT_BINARY = "llama-bench"
_MODEL_FLAGS = frozenset({"-m", "--model"})


@dataclass(frozen=True)
class LlamaBenchConfig:
    model: str
    args: list[str]
    binary: str = _DEFAULT_BINARY


def load_llama_bench_config(path: str | Path, *, model: str) -> LlamaBenchConfig:
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"llama bench config must be a YAML mapping: {path}")

    if "model" in raw:
        raise ValueError(
            f"llama bench config must not contain 'model'; use {MODEL_YAML} "
            f"in the model directory: {path}"
        )

    args_raw = raw.get("args")
    if args_raw is None:
        args = []
    elif isinstance(args_raw, list):
        raise ValueError(
            f"llama bench config 'args' must be a multiline string (YAML block scalar): {path}"
        )
    elif not isinstance(args_raw, str):
        raise ValueError(
            f"llama bench config 'args' must be a multiline string (YAML block scalar): {path}"
        )
    else:
        args = parse_args_block(args_raw, path)
    _validate_args_no_model_flag(args, path)

    binary = raw.get("binary", _DEFAULT_BINARY)
    if not isinstance(binary, str) or not binary.strip():
        raise ValueError(f"llama bench config 'binary' must be a non-empty string: {path}")

    return LlamaBenchConfig(
        model=model,
        args=args,
        binary=binary.strip(),
    )


def resolve_llama_bench(
    model_dir: Path,
    *,
    n_cpu_moe: int | None = None,
) -> LlamaBenchConfig:
    """Load llama_bench.yaml for a model directory with model path from model.yaml."""
    _require_model_dir(model_dir)
    model_yaml_path, bench_yaml_path = _require_base_configs(model_dir)

    model = load_model_config(model_yaml_path)
    config = load_llama_bench_config(bench_yaml_path, model=model)

    if n_cpu_moe is not None:
        if n_cpu_moe <= 0:
            raise ValueError(
                f"n_cpu_moe must be a positive integer, got {n_cpu_moe}"
            )
        config = replace(
            config,
            args=replace_flag_args(
                config.args,
                {
                    "--n-cpu-moe": str(n_cpu_moe),
                    "--n-gpu-layers": "999",
                },
            ),
        )

    return config


def build_llama_bench_argv(config: LlamaBenchConfig) -> list[str]:
    return [config.binary, *config.args, "-m", config.model]


def _require_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")


def _require_base_configs(model_dir: Path) -> tuple[Path, Path]:
    model_yaml_path = model_dir / MODEL_YAML
    bench_yaml_path = model_dir / LLAMA_BENCH_YAML
    missing: list[Path] = []
    if not model_yaml_path.is_file():
        missing.append(model_yaml_path)
    if not bench_yaml_path.is_file():
        missing.append(bench_yaml_path)
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Required model config file(s) missing: {names}")
    return model_yaml_path, bench_yaml_path


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
                f"llama bench config 'args' must not contain -m or --model "
                f"(wrapper injects model path): {path}"
            )
