"""Shared filesystem roots for the probe runner."""

from __future__ import annotations

from pathlib import Path

PROBE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROBE_ROOT.parent
MODELS_ROOT = REPO_ROOT / "models"
CALIBRATION_PROBES_ROOT = PROBE_ROOT / "calibration-probes"
RESULTS_DIR = PROBE_ROOT / "results"
