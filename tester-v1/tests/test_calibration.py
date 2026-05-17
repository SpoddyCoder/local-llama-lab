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
    format_summary_lines,
    parse_idle_vram_from_metrics_stdout,
    read_idle_vram_from_result,
    write_calibration_session_summary,
)
from metrics import format_metrics_summary  # noqa: E402
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


class TestWriteCalibrationSessionSummary(unittest.TestCase):
    def test_writes_expected_structure(self) -> None:
        summary = {
            "gguf_gb": 9.55,
            "model_vram_mb": 10353,
            "kv_vram_mb": 4414,
            "estimated_context_max": 128000,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tester_root = Path(tmp)
            footprint = (
                tester_root
                / "results"
                / "qwen3.5-9b-q8"
                / "calibration-footprint"
                / "20260101T000000Z.json"
            )
            ctx_probe = (
                tester_root
                / "results"
                / "qwen3.5-9b-q8"
                / "calibration-ctx-probe"
                / "20260101T000001Z.json"
            )
            _write_result(
                footprint,
                idle_vram_mb=10353,
            )
            _write_result(ctx_probe, idle_vram_mb=10429)
            with footprint.open(encoding="utf-8") as f:
                footprint_doc = json.load(f)
            with ctx_probe.open(encoding="utf-8") as f:
                ctx_probe_doc = json.load(f)

            out_path = write_calibration_session_summary(
                tester_root,
                "qwen3.5-9b-q8",
                "20260517T120000Z",
                summary,
                footprint,
                ctx_probe,
            )
            self.assertEqual(
                out_path,
                tester_root
                / "results"
                / "qwen3.5-9b-q8"
                / "calibration-sessions"
                / "20260517T120000Z.json",
            )
            with out_path.open(encoding="utf-8") as f:
                document = json.load(f)
            self.assertEqual(document["schema_version"], "1")
            self.assertEqual(document["session_id"], "20260517T120000Z")
            self.assertEqual(document["model"], "qwen3.5-9b-q8")
            self.assertIn("created_at", document)
            self.assertEqual(document["summary"], summary)
            self.assertEqual(
                document["probes"]["footprint"],
                {
                    "run_id": footprint_doc["run_id"],
                    "path": "results/qwen3.5-9b-q8/calibration-footprint/20260101T000000Z.json",
                },
            )
            self.assertEqual(
                document["probes"]["ctx_probe"],
                {
                    "run_id": ctx_probe_doc["run_id"],
                    "path": "results/qwen3.5-9b-q8/calibration-ctx-probe/20260101T000001Z.json",
                },
            )


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
