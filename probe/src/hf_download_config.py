"""Hugging Face download metadata and cache path helpers for model.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_HF_DOWNLOAD_KEY = "hf-download"
_DEFAULT_REF_NAMES = ("main", "master")


@dataclass(frozen=True)
class HfDownloadConfig:
    repo: str
    file: str


def load_hf_download(model_yaml_path: str | Path) -> HfDownloadConfig | None:
    """Return hf-download metadata from model.yaml, or None if the key is absent."""
    raw = _load_yaml(model_yaml_path)
    if not isinstance(raw, dict):
        raise ValueError(f"model config must be a YAML mapping: {model_yaml_path}")

    block = raw.get(_HF_DOWNLOAD_KEY)
    if block is None:
        return None
    return _parse_hf_download_block(block, model_yaml_path)


def load_hf_download_required(model_yaml_path: str | Path) -> HfDownloadConfig:
    """Return hf-download metadata; raise ValueError if missing or malformed."""
    config = load_hf_download(model_yaml_path)
    if config is None:
        raise ValueError(
            f"model config {_HF_DOWNLOAD_KEY!r} block is required: {model_yaml_path}"
        )
    return config


def build_hf_download_argv(repo: str, file: str) -> list[str]:
    return ["hf", "download", repo, file]


def get_current_model_path(model_yaml_path: str | Path) -> str | None:
    """Return expanded model path from model.yaml, or None if the key is absent."""
    raw = _load_yaml(model_yaml_path)
    if not isinstance(raw, dict):
        raise ValueError(f"model config must be a YAML mapping: {model_yaml_path}")

    model_raw = raw.get("model")
    if model_raw is None:
        return None
    if not isinstance(model_raw, str):
        raise ValueError(
            f"model config 'model' must be a string when present: {model_yaml_path}"
        )
    return os.path.expanduser(model_raw)


def resolve_hf_cache_path(repo: str, file: str) -> Path:
    """Locate a downloaded GGUF under the local Hugging Face hub cache."""
    cache_dir = _hf_hub_dir() / _repo_to_cache_dir_name(repo)
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Hugging Face cache directory not found for {repo!r}: {cache_dir}"
        )

    candidate = _path_from_refs(cache_dir, file)
    if candidate is not None:
        return candidate

    candidate = _newest_snapshot_path(cache_dir, file)
    if candidate is not None:
        return candidate

    raise FileNotFoundError(
        f"GGUF not found in Hugging Face cache for {repo!r} file {file!r} under {cache_dir}"
    )


def update_model_path_if_changed(model_yaml_path: str | Path, new_path: str | Path) -> bool:
    """Update model.yaml when the resolved model path changed; return True if written."""
    path = Path(model_yaml_path)
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"model config must be a YAML mapping: {path}")

    formatted = format_path_for_yaml(new_path)
    current = raw.get("model")
    if isinstance(current, str) and _paths_equal(current, formatted):
        return False

    raw["model"] = formatted
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False, default_flow_style=False)
    return True


def format_path_for_yaml(path: str | Path) -> str:
    """Format a filesystem path for storage in model.yaml."""
    absolute = Path(path).expanduser().absolute()
    try:
        relative = absolute.relative_to(Path.home())
    except ValueError:
        return str(absolute)
    return f"~/{relative.as_posix()}"


def _parse_hf_download_block(block: Any, model_yaml_path: str | Path) -> HfDownloadConfig:
    if not isinstance(block, dict):
        raise ValueError(
            f"model config {_HF_DOWNLOAD_KEY!r} must be a mapping: {model_yaml_path}"
        )

    repo = block.get("repo")
    if not isinstance(repo, str) or not repo.strip():
        raise ValueError(
            f"model config {_HF_DOWNLOAD_KEY!r}.repo is required and must be a "
            f"non-empty string: {model_yaml_path}"
        )

    file_name = block.get("file")
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError(
            f"model config {_HF_DOWNLOAD_KEY!r}.file is required and must be a "
            f"non-empty string: {model_yaml_path}"
        )

    return HfDownloadConfig(repo=repo.strip(), file=file_name.strip())


def _load_yaml(path: str | Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _hf_hub_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_to_cache_dir_name(repo: str) -> str:
    if "/" not in repo:
        raise ValueError(f"invalid Hugging Face repo id (expected org/name): {repo!r}")
    org, name = repo.split("/", 1)
    if not org or not name:
        raise ValueError(f"invalid Hugging Face repo id (expected org/name): {repo!r}")
    return f"models--{org}--{name}"


def _path_from_refs(cache_dir: Path, file_name: str) -> Path | None:
    refs_dir = cache_dir / "refs"
    if not refs_dir.is_dir():
        return None

    for ref_name in _DEFAULT_REF_NAMES:
        revision = _read_ref_revision(refs_dir / ref_name)
        if revision is None:
            continue
        candidate = cache_dir / "snapshots" / revision / file_name
        if candidate.is_file():
            return candidate
    return None


def _read_ref_revision(ref_path: Path) -> str | None:
    if not ref_path.exists():
        return None

    if ref_path.is_symlink():
        target = ref_path.resolve()
        if target.parent.name == "snapshots" and target.is_dir():
            return target.name
        if target.is_file():
            content = target.read_text(encoding="utf-8").strip()
            return content or None
        return None

    content = ref_path.read_text(encoding="utf-8").strip()
    return content or None


def _newest_snapshot_path(cache_dir: Path, file_name: str) -> Path | None:
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    newest: tuple[float, Path] | None = None
    for snapshot_dir in snapshots_dir.iterdir():
        if not snapshot_dir.is_dir():
            continue
        candidate = snapshot_dir / file_name
        if not candidate.is_file():
            continue
        mtime = candidate.stat().st_mtime
        if newest is None or mtime > newest[0]:
            newest = (mtime, candidate)
    return newest[1] if newest is not None else None


def _paths_equal(left: str | Path, right: str | Path) -> bool:
    left_abs = Path(os.path.expanduser(str(left))).absolute()
    right_abs = Path(os.path.expanduser(str(right))).absolute()
    if left_abs == right_abs:
        return True
    try:
        return left_abs.is_file() and right_abs.is_file() and os.path.samefile(left_abs, right_abs)
    except OSError:
        return False
