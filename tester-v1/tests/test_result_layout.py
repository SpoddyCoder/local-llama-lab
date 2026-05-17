"""Unit tests for result path layout helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from result_layout import (  # noqa: E402
    completion_output_path,
    find_latest_result,
    resolve_result_target,
    run_id_from_started_at,
    suite_from_variant,
)
from runner import _TESTER_ROOT  # noqa: E402


class TestResolveResultTarget(unittest.TestCase):
    def setUp(self) -> None:
        self.started = datetime(2026, 5, 17, 17, 2, 40, tzinfo=timezone.utc)

    def test_standard_layout(self) -> None:
        config_dir = (
            _TESTER_ROOT / "configs" / "qwen3.5-9b-q8" / "hello-world-baseline"
        )
        target = resolve_result_target(config_dir, _TESTER_ROOT, self.started)
        self.assertEqual(target.layout, "standard")
        self.assertEqual(target.model, "qwen3.5-9b-q8")
        self.assertEqual(target.variant, "hello-world-baseline")
        self.assertEqual(target.run_id_suffix, "qwen3.5-9b-q8-hello-world-baseline")
        self.assertEqual(
            target.json_path,
            _TESTER_ROOT
            / "results"
            / "qwen3.5-9b-q8"
            / "hello-world-baseline"
            / "20260517T170240Z.json",
        )

    def test_fallback_layout(self) -> None:
        config_dir = Path("/tmp/my-run")
        target = resolve_result_target(config_dir, _TESTER_ROOT, self.started)
        self.assertEqual(target.layout, "fallback")
        self.assertIsNone(target.model)
        self.assertIsNone(target.variant)
        self.assertEqual(target.run_id_suffix, "tmp-my-run")
        self.assertEqual(
            target.json_path,
            _TESTER_ROOT / "results" / "_other" / "tmp-my-run" / "20260517T170240Z.json",
        )


class TestRunIdFromStartedAt(unittest.TestCase):
    def test_format(self) -> None:
        started = datetime(2026, 5, 17, 17, 2, 40, tzinfo=timezone.utc)
        self.assertEqual(
            run_id_from_started_at(started, "qwen3.5-9b-q8-hello-world-baseline"),
            "20260517T170240Z_qwen3.5-9b-q8-hello-world-baseline",
        )


class TestFindLatestResult(unittest.TestCase):
    def test_picks_newest_by_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tester_root = Path(tmp)
            config_dir = tester_root / "configs" / "m" / "v"
            variant_dir = tester_root / "results" / "m" / "v"
            variant_dir.mkdir(parents=True)
            (variant_dir / "20260101T000000Z.json").write_text("{}", encoding="utf-8")
            (variant_dir / "20261231T235959Z.json").write_text("{}", encoding="utf-8")
            (variant_dir / "20260615T120000Z.json").write_text("{}", encoding="utf-8")

            latest = find_latest_result(
                tester_root / "results",
                config_dir,
                tester_root,
            )
            self.assertEqual(latest.name, "20261231T235959Z.json")

    def test_missing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tester_root = Path(tmp)
            config_dir = tester_root / "configs" / "m" / "v"
            with self.assertRaises(FileNotFoundError):
                find_latest_result(
                    tester_root / "results",
                    config_dir,
                    tester_root,
                )


class TestCompletionOutputPath(unittest.TestCase):
    def test_sibling_output_txt(self) -> None:
        json_path = Path("/data/results/m/v/20260517T170240Z.json")
        self.assertEqual(
            completion_output_path(json_path),
            Path("/data/results/m/v/20260517T170240Z-output.txt"),
        )


class TestSuiteFromVariant(unittest.TestCase):
    def test_calibration_prefix(self) -> None:
        self.assertEqual(suite_from_variant("calibration-footprint"), "calibration")

    def test_probe_default(self) -> None:
        self.assertEqual(suite_from_variant("hello-world-baseline"), "probe")
        self.assertEqual(suite_from_variant(None), "probe")


if __name__ == "__main__":
    unittest.main()
