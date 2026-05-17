"""Result file paths and run IDs for tester-v1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import config_dir_metadata, slug_from_config_dir
from results import format_compact_utc


@dataclass(frozen=True)
class ResultTarget:
    json_path: Path
    run_id_suffix: str
    model: str | None
    variant: str | None
    layout: str


def _result_variant_dir(config_dir: Path, tester_root: Path) -> Path:
    """Directory that holds result JSON files for a config directory."""
    config_dir = config_dir.resolve()
    tester_root = tester_root.resolve()
    configs_root = (tester_root / "configs").resolve()
    try:
        parts = config_dir.relative_to(configs_root).parts
    except ValueError:
        parts = None

    if parts is not None and len(parts) == 2:
        return tester_root / "results" / parts[0] / parts[1]

    slug = slug_from_config_dir(config_dir, tester_root)
    return tester_root / "results" / "_other" / slug


def resolve_result_target(
    config_dir: Path,
    tester_root: Path,
    started_at: datetime,
) -> ResultTarget:
    """Resolve result JSON path and run_id suffix for a config directory."""
    config_dir = config_dir.resolve()
    tester_root = tester_root.resolve()
    compact = format_compact_utc(started_at)
    meta = config_dir_metadata(config_dir, tester_root)

    configs_root = (tester_root / "configs").resolve()
    try:
        parts = config_dir.relative_to(configs_root).parts
    except ValueError:
        parts = None

    if parts is not None and len(parts) == 2:
        model_dir, variant_name = parts[0], parts[1]
        json_path = (
            tester_root
            / "results"
            / model_dir
            / variant_name
            / f"{compact}.json"
        )
        return ResultTarget(
            json_path=json_path,
            run_id_suffix=f"{model_dir}-{variant_name}",
            model=meta["model"],
            variant=meta["variant"],
            layout="standard",
        )

    slug = slug_from_config_dir(config_dir, tester_root)
    json_path = tester_root / "results" / "_other" / slug / f"{compact}.json"
    return ResultTarget(
        json_path=json_path,
        run_id_suffix=slug,
        model=None,
        variant=None,
        layout="fallback",
    )


def run_id_from_started_at(started_at: datetime, run_id_suffix: str) -> str:
    """Build run_id as {compact_utc}_{run_id_suffix}."""
    return f"{format_compact_utc(started_at)}_{run_id_suffix}"


def find_latest_result(
    results_dir: Path,
    config_dir: Path,
    tester_root: Path,
) -> Path:
    """Return the newest result JSON for a config directory (by filename stem)."""
    variant_dir = _result_variant_dir(config_dir, tester_root)
    if not variant_dir.is_dir():
        raise FileNotFoundError(f"no result directory: {variant_dir}")

    candidates = sorted(variant_dir.glob("*.json"), key=lambda p: p.stem)
    if not candidates:
        raise FileNotFoundError(f"no result JSON in {variant_dir}")

    return candidates[-1]


def completion_output_path(json_path: Path) -> Path:
    """Sibling path for completion text: {stem}-output.txt."""
    return json_path.with_name(f"{json_path.stem}-output.txt")


def suite_from_variant(variant: str | None) -> str:
    """Classify suite from variant name."""
    if variant is not None and variant.startswith("calibration-"):
        return "calibration"
    return "probe"
