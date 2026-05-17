"""Unit tests for VRAM calibration helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from calibration import (  # noqa: E402
    compute_summary,
    find_latest_result,
    format_summary_lines,
    parse_idle_vram_from_metrics_stdout,
    read_idle_vram_from_result,
)
from metrics import format_metrics_summary  # noqa: E402
from config import slug_from_config_dir  # noqa: E402
from runner import _TESTER_ROOT  # noqa: E402


def _write_result(path: Path, *, idle_vram_mb: int, status: str = "ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "run_id": path.stem,
        "status": status,
        "metrics": {"idle_vram_mb": idle_vram_mb},
    }
    path.write_text(json.dumps(document), encoding="utf-8")


class TestComputeSummary(unittest.TestCase):
    def test_happy_path_known_numbers(self) -> None:
        summary = compute_summary(
            footprint_idle=10353,
            ctx_probe_idle=10429,
            footprint_c=4096,
            ctx_c=16384,
            gpu_total_mb=16303,
            margin_mb=1536,
            gguf_gb=9.55,
        )
        self.assertEqual(summary["model_vram_mb"], 10353)
        self.assertEqual(summary["kv_vram_mb"], 4414)
        self.assertEqual(summary["gguf_gb"], 9.55)
        self.assertEqual(summary["estimated_context_max"], 717770)

    def test_ctx_c_not_greater_than_footprint_c(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            compute_summary(
                footprint_idle=1000,
                ctx_probe_idle=1100,
                footprint_c=8192,
                ctx_c=4096,
                gpu_total_mb=16000,
                margin_mb=1536,
                gguf_gb=1.0,
            )
        self.assertIn("must be greater than footprint", str(ctx.exception))

    def test_non_positive_slope_equal_idle(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            compute_summary(
                footprint_idle=10000,
                ctx_probe_idle=10000,
                footprint_c=4096,
                ctx_c=16384,
                gpu_total_mb=16000,
                margin_mb=1536,
                gguf_gb=1.0,
            )
        self.assertIn("slope is non-positive", str(ctx.exception))

    def test_non_positive_slope_ctx_probe_lower_idle(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            compute_summary(
                footprint_idle=10000,
                ctx_probe_idle=9990,
                footprint_c=4096,
                ctx_c=16384,
                gpu_total_mb=16000,
                margin_mb=1536,
                gguf_gb=1.0,
            )
        self.assertIn("slope is non-positive", str(ctx.exception))

    def test_negative_kv_budget(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            compute_summary(
                footprint_idle=15000,
                ctx_probe_idle=15100,
                footprint_c=4096,
                ctx_c=16384,
                gpu_total_mb=16000,
                margin_mb=1536,
                gguf_gb=1.0,
            )
        self.assertIn("budget is negative", str(ctx.exception))


class TestFindLatestResult(unittest.TestCase):
    def test_picks_newest_by_run_id_timestamp(self) -> None:
        config_dir = (
            _TESTER_ROOT
            / "configs"
            / "qwen3.5-9b-q8"
            / "calibration-footprint"
        )
        slug = slug_from_config_dir(config_dir, _TESTER_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            older = results_dir / f"20260101T000000Z_{slug}.json"
            newer = results_dir / f"20260215T120000Z_{slug}.json"
            _write_result(older, idle_vram_mb=1)
            _write_result(newer, idle_vram_mb=2)
            self.assertEqual(find_latest_result(results_dir, slug), newer)

    def test_missing_slug_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                find_latest_result(results_dir, "missing-slug")


class TestReadIdleVramFromResult(unittest.TestCase):
    def test_reads_ok_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            _write_result(path, idle_vram_mb=12345)
            self.assertEqual(read_idle_vram_from_result(path), 12345)

    def test_rejects_non_ok_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            _write_result(path, idle_vram_mb=100, status="error")
            with self.assertRaises(ValueError) as ctx:
                read_idle_vram_from_result(path)
            self.assertIn("status is not ok", str(ctx.exception))

    def test_rejects_missing_idle_vram(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(
                json.dumps({"status": "ok", "metrics": {}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                read_idle_vram_from_result(path)
            self.assertIn("idle_vram_mb missing", str(ctx.exception))


class TestParseIdleVramFromMetricsStdout(unittest.TestCase):
    def test_reads_idle_vram_from_probe_block(self) -> None:
        metrics = {
            "server_ready_s": 1.0,
            "wall_time_s": 2.0,
            "idle_vram_mb": 10353,
            "peak_vram_mb": 10400,
        }
        text = f"Model: example\n\n{format_metrics_summary(metrics)}\n"
        self.assertEqual(parse_idle_vram_from_metrics_stdout(text), 10353)

    def test_rejects_missing_idle_vram(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_idle_vram_from_metrics_stdout("wall_time_s          1.00\n")
        self.assertIn("not found", str(ctx.exception))

    def test_rejects_na_idle_vram(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_idle_vram_from_metrics_stdout("idle_vram_mb         n/a\n")
        self.assertIn("unavailable", str(ctx.exception))


class TestFormatSummaryLines(unittest.TestCase):
    def test_output_shape(self) -> None:
        summary = {
            "gguf_gb": 9.55,
            "model_vram_mb": 10353,
            "kv_vram_mb": 4414,
            "estimated_context_max": 128000,
        }
        lines = format_summary_lines(summary)
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], "* GGUF on disk: 9.55 GB")
        self.assertEqual(lines[1], "* Model VRAM: 10353 MiB")
        self.assertEqual(lines[2], "* KV VRAM: 4414 MiB")
        self.assertEqual(lines[3], "* Estimated Max Context: 128000 tokens")


if __name__ == "__main__":
    unittest.main()
