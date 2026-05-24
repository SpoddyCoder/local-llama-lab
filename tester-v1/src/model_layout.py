"""Resolve model and variant config paths on disk."""

from __future__ import annotations

from pathlib import Path

from config import MODEL_YAML


def resolve_model_variant(
    model_dir: Path,
    variant: str = "bench",
) -> tuple[Path, Path, Path]:
    """Return (server.yaml, client.yaml, model.yaml) for a model or variant directory."""
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {model_dir}")

    if (model_dir / "server.yaml").is_file():
        variant_dir = model_dir
        model_yaml = model_dir.parent / MODEL_YAML
    elif (model_dir / variant / "server.yaml").is_file():
        variant_dir = model_dir / variant
        model_yaml = model_dir / MODEL_YAML
    else:
        raise FileNotFoundError(
            f"No server.yaml in {model_dir} or {model_dir / variant}"
        )

    server = variant_dir / "server.yaml"
    client = variant_dir / "client.yaml"

    missing: list[str] = []
    if not client.is_file():
        missing.append(client.name)
    if not model_yaml.is_file():
        missing.append(MODEL_YAML)
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Config files missing for {model_dir}: {names}")

    return server, client, model_yaml
