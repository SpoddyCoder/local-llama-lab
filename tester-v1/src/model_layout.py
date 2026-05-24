"""Resolve model server and variant run configs on disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import MODEL_YAML, ServerConfig, load_variant_server, merge_server_configs
from paths import CALIBRATION_TESTS_ROOT


@dataclass(frozen=True)
class RunConfig:
    server: ServerConfig
    client_path: Path
    model_yaml_path: Path
    model: str
    variant: str


def _require_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")


def _require_base_configs(model_dir: Path) -> tuple[Path, Path]:
    model_yaml_path = model_dir / MODEL_YAML
    server_yaml_path = model_dir / "server.yaml"
    missing: list[Path] = []
    if not model_yaml_path.is_file():
        missing.append(model_yaml_path)
    if not server_yaml_path.is_file():
        missing.append(server_yaml_path)
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Required model config file(s) missing: {names}")
    return model_yaml_path, server_yaml_path


def _resolve_variant_paths(
    model_dir: Path,
    variant: str,
) -> tuple[Path, Path | None]:
    calib_client = CALIBRATION_TESTS_ROOT / variant / "client.yaml"
    local_client = model_dir / variant / "client.yaml"

    if calib_client.is_file():
        client_path = calib_client
        override_path = CALIBRATION_TESTS_ROOT / variant / "server.yaml"
    elif local_client.is_file():
        client_path = local_client
        override_path = model_dir / variant / "server.yaml"
    else:
        raise FileNotFoundError(
            f"No client.yaml for variant {variant!r}; looked in:\n"
            f"  {calib_client}\n"
            f"  {local_client}"
        )

    server_override_path = override_path if override_path.is_file() else None
    return client_path, server_override_path


def resolve_run_config(
    model_dir: Path,
    variant: str,
    *,
    n_cpu_moe: int | None = None,
) -> RunConfig:
    """Resolve merged server config and client path for a model + variant."""
    _require_model_dir(model_dir)
    model_yaml_path, server_yaml_path = _require_base_configs(model_dir)

    base = load_variant_server(server_yaml_path, model_yaml_path)
    client_path, override_path = _resolve_variant_paths(model_dir, variant)

    override: ServerConfig | None = None
    if override_path is not None:
        override = load_variant_server(override_path, model_yaml_path)

    server = merge_server_configs(base, override, n_cpu_moe=n_cpu_moe)
    return RunConfig(
        server=server,
        client_path=client_path,
        model_yaml_path=model_yaml_path,
        model=model_dir.name,
        variant=variant,
    )


def resolve_model_server(
    model_dir: Path,
    *,
    n_cpu_moe: int | None = None,
) -> tuple[ServerConfig, Path]:
    """Load base server.yaml for a model directory (no variant client/override)."""
    _require_model_dir(model_dir)
    model_yaml_path, server_yaml_path = _require_base_configs(model_dir)

    base = load_variant_server(server_yaml_path, model_yaml_path)
    server = merge_server_configs(base, None, n_cpu_moe=n_cpu_moe)
    return server, model_yaml_path
