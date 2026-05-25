"""Shared filesystem roots for the tester runner."""

from __future__ import annotations

from pathlib import Path

TESTER_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = TESTER_ROOT.parent
MODELS_ROOT = REPO_ROOT / "models"
CALIBRATION_TESTS_ROOT = TESTER_ROOT / "calibration-tests"
RESULTS_DIR = TESTER_ROOT / "results"
