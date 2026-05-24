"""Result file paths and run IDs for tester-v1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import _models_parts, config_dir_metadata
from paths import RESULTS_DIR
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
    model, variant = _models_parts(config_dir)
    return RESULTS_DIR / model / variant


def resolve_result_target(
    config_dir: Path,
    tester_root: Path,
    started_at: datetime,
) -> ResultTarget:
    """Resolve result JSON path and run_id suffix for a config directory."""
    compact = format_compact_utc(started_at)
    meta = config_dir_metadata(config_dir, tester_root)
    model, variant = _models_parts(config_dir)
    json_path = RESULTS_DIR / model / variant / f"{compact}.json"
    return ResultTarget(
        json_path=json_path,
        run_id_suffix=f"{model}-{variant}",
        model=meta["model"],
        variant=meta["variant"],
        layout="standard",
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
