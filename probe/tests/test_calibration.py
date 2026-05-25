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
    DEFAULT_CALIBRATION_MARGIN_MIB,
    compute_summary,
    format_generation_throughput_line,
    format_mib_as_gb,
    format_summary_lines,
    parse_decode_tok_s_from_metrics_stdout,
    parse_idle_system_ram_from_metrics_stdout,
    parse_idle_vram_from_metrics_stdout,
    parse_model_max_context_from_metrics_stdout,
    read_decode_tok_s_from_result,
    read_idle_system_ram_from_result,
    read_idle_vram_from_result,
    read_model_max_context_from_result,
    write_calibration_session_summary,
)
from metrics import format_metrics_summary  # noqa: E402

def _write_result(
    path: Path,
    *,
    idle_vram_mb: int | None = None,
    status: str = "ok",
    model_max_context: int | None = None,
    decode_tok_s: float | None = None,
    include_idle_system_ram: bool = False,
    idle_system_ram_mb: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, int | float | None] = {}
    if idle_vram_mb is not None:
        metrics["idle_vram_mb"] = idle_vram_mb
    if model_max_context is not None:
        metrics["model_max_context"] = model_max_context
    if decode_tok_s is not None:
        metrics["decode_tok_s"] = decode_tok_s
    if include_idle_system_ram:
        metrics["idle_system_ram_mb"] = idle_system_ram_mb
    document = {
        "run_id": path.stem,
        "status": status,
        "metrics": metrics,
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
            margin_mb=0,
            gguf_gb=9.55,
        )
        self.assertEqual(summary["model_vram_mb"], 10353)
        self.assertEqual(summary["kv_vram_mb"], 5950)
        self.assertEqual(summary["gguf_gb"], 9.55)
        self.assertEqual(summary["estimated_context_max"], 966117)

    def test_margin_reduces_kv_budget(self) -> None:
        without_margin = compute_summary(
            footprint_idle=10353,
            ctx_probe_idle=10429,
            footprint_c=4096,
            ctx_c=16384,
            gpu_total_mb=16303,
            margin_mb=0,
            gguf_gb=9.55,
        )
        with_default_margin = compute_summary(
            footprint_idle=10353,
            ctx_probe_idle=10429,
            footprint_c=4096,
            ctx_c=16384,
            gpu_total_mb=16303,
            margin_mb=DEFAULT_CALIBRATION_MARGIN_MIB,
            gguf_gb=9.55,
        )
        self.assertEqual(
            with_default_margin["kv_vram_mb"],
            without_margin["kv_vram_mb"] - DEFAULT_CALIBRATION_MARGIN_MIB,
        )
        self.assertLess(
            with_default_margin["estimated_context_max"],
            without_margin["estimated_context_max"],
        )

    def test_ctx_c_not_greater_than_footprint_c(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            compute_summary(
                footprint_idle=1000,
                ctx_probe_idle=1100,
                footprint_c=8192,
                ctx_c=4096,
                gpu_total_mb=16000,
                margin_mb=0,
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
                margin_mb=0,
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
                margin_mb=0,
                gguf_gb=1.0,
            )
        self.assertIn("slope is non-positive", str(ctx.exception))

    def test_negative_kv_budget(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            compute_summary(
                footprint_idle=16100,
                ctx_probe_idle=16200,
                footprint_c=4096,
                ctx_c=16384,
                gpu_total_mb=16000,
                margin_mb=0,
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
            "model_max_context": 128000,
            "generation_throughput_tok_s": 91.2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            probe_root = Path(tmp)
            footprint = (
                probe_root
                / "results"
                / "qwen3.5-9b-q8"
                / "calibration-footprint"
                / "20260101T000000Z.json"
            )
            ctx_probe = (
                probe_root
                / "results"
                / "qwen3.5-9b-q8"
                / "calibration-ctx-probe"
                / "20260101T000001Z.json"
            )
            hello_world = (
                probe_root
                / "results"
                / "qwen3.5-9b-q8"
                / "hello-world-baseline"
                / "20260101T000002Z.json"
            )
            _write_result(
                footprint,
                idle_vram_mb=10353,
            )
            _write_result(ctx_probe, idle_vram_mb=10429)
            _write_result(hello_world, idle_vram_mb=0, decode_tok_s=91.2)
            with footprint.open(encoding="utf-8") as f:
                footprint_doc = json.load(f)
            with ctx_probe.open(encoding="utf-8") as f:
                ctx_probe_doc = json.load(f)
            with hello_world.open(encoding="utf-8") as f:
                hello_world_doc = json.load(f)

            out_path = write_calibration_session_summary(
                probe_root,
                "qwen3.5-9b-q8",
                "20260517T120000Z",
                summary,
                footprint,
                ctx_probe,
                hello_world,
            )
            self.assertEqual(
                out_path,
                probe_root
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
            self.assertEqual(
                document["summary"],
                {
                    "gguf_gb": 9.55,
                    "model_vram_mb": 10353,
                    "kv_vram_mb": 4414,
                    "estimated_context_max": 128000,
                    "model_max_context": 128000,
                    "generation_throughput_tok_s": 91.2,
                },
            )
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
            self.assertEqual(
                document["probes"]["hello_world"],
                {
                    "run_id": hello_world_doc["run_id"],
                    "path": "results/qwen3.5-9b-q8/hello-world-baseline/20260101T000002Z.json",
                },
            )

    def test_writes_model_system_ram_mb_when_present(self) -> None:
        summary = {
            "gguf_gb": 9.55,
            "model_vram_mb": 10353,
            "kv_vram_mb": 4414,
            "estimated_context_max": 128000,
            "model_max_context": 128000,
            "generation_throughput_tok_s": 91.2,
            "model_system_ram_mb": 24000,
        }
        with tempfile.TemporaryDirectory() as tmp:
            probe_root = Path(tmp)
            footprint = (
                probe_root / "results" / "x" / "calibration-footprint" / "a.json"
            )
            ctx_probe = probe_root / "results" / "x" / "calibration-ctx-probe" / "b.json"
            hello_world = (
                probe_root / "results" / "x" / "hello-world-baseline" / "c.json"
            )
            for p in (footprint, ctx_probe, hello_world):
                _write_result(p, idle_vram_mb=1)
            out_path = write_calibration_session_summary(
                probe_root,
                "x",
                "sess",
                summary,
                footprint,
                ctx_probe,
                hello_world,
            )
            doc = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(doc["summary"]["model_system_ram_mb"], 24000)


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


class TestParseModelMaxContextFromMetricsStdout(unittest.TestCase):
    def test_reads_model_max_context(self) -> None:
        text = "model_max_context    128000\n"
        self.assertEqual(parse_model_max_context_from_metrics_stdout(text), 128000)

    def test_returns_none_when_missing(self) -> None:
        self.assertIsNone(parse_model_max_context_from_metrics_stdout("idle_vram_mb 100\n"))

    def test_returns_none_for_na(self) -> None:
        self.assertIsNone(
            parse_model_max_context_from_metrics_stdout("model_max_context n/a\n")
        )


class TestReadModelMaxContextFromResult(unittest.TestCase):
    def test_reads_ok_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            _write_result(path, idle_vram_mb=100, model_max_context=65536)
            self.assertEqual(read_model_max_context_from_result(path), 65536)

    def test_returns_none_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            _write_result(path, idle_vram_mb=100)
            self.assertIsNone(read_model_max_context_from_result(path))

    def test_returns_none_for_non_ok_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            _write_result(path, idle_vram_mb=100, status="error", model_max_context=1)
            self.assertIsNone(read_model_max_context_from_result(path))


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


class TestParseIdleSystemRamFromMetricsStdout(unittest.TestCase):
    def test_reads_from_probe_block(self) -> None:
        metrics = {
            "server_ready_s": 1.0,
            "wall_time_s": 2.0,
            "idle_vram_mb": 10353,
            "idle_system_ram_mb": 24000,
        }
        text = f"Model: example\n\n{format_metrics_summary(metrics)}\n"
        self.assertEqual(parse_idle_system_ram_from_metrics_stdout(text), 24000)

    def test_returns_none_when_missing(self) -> None:
        self.assertIsNone(
            parse_idle_system_ram_from_metrics_stdout("idle_vram_mb 100\n")
        )

    def test_returns_none_for_na(self) -> None:
        self.assertIsNone(
            parse_idle_system_ram_from_metrics_stdout("idle_system_ram_mb n/a\n")
        )


class TestReadIdleSystemRamFromResult(unittest.TestCase):
    def test_reads_ok_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            _write_result(
                path,
                idle_vram_mb=100,
                include_idle_system_ram=True,
                idle_system_ram_mb=32000,
            )
            self.assertEqual(read_idle_system_ram_from_result(path), 32000)

    def test_returns_none_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            _write_result(path, idle_vram_mb=100)
            self.assertIsNone(read_idle_system_ram_from_result(path))

    def test_returns_none_when_json_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            _write_result(
                path,
                idle_vram_mb=100,
                include_idle_system_ram=True,
                idle_system_ram_mb=None,
            )
            self.assertIsNone(read_idle_system_ram_from_result(path))

    def test_returns_none_for_non_ok_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            _write_result(
                path,
                idle_vram_mb=100,
                status="error",
                include_idle_system_ram=True,
                idle_system_ram_mb=1,
            )
            self.assertIsNone(read_idle_system_ram_from_result(path))


class TestParseDecodeTokSFromMetricsStdout(unittest.TestCase):
    def test_reads_decode_tok_s(self) -> None:
        text = "decode_tok_s          91.24\n"
        self.assertEqual(parse_decode_tok_s_from_metrics_stdout(text), 91.24)

    def test_returns_none_when_missing(self) -> None:
        self.assertIsNone(parse_decode_tok_s_from_metrics_stdout("wall_time_s          1.00\n"))

    def test_returns_none_for_na(self) -> None:
        self.assertIsNone(parse_decode_tok_s_from_metrics_stdout("decode_tok_s         n/a\n"))


class TestReadDecodeTokSFromResult(unittest.TestCase):
    def test_reads_ok_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            _write_result(path, idle_vram_mb=0, decode_tok_s=114.6)
            self.assertEqual(read_decode_tok_s_from_result(path), 114.6)

    def test_returns_none_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            _write_result(path, idle_vram_mb=100)
            self.assertIsNone(read_decode_tok_s_from_result(path))

    def test_returns_none_for_non_ok_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            _write_result(path, idle_vram_mb=0, status="error", decode_tok_s=50.0)
            self.assertIsNone(read_decode_tok_s_from_result(path))


class TestFormatMibAsGb(unittest.TestCase):
    def test_converts_mib_to_two_decimal_gb(self) -> None:
        self.assertEqual(format_mib_as_gb(10353), "10.11 GB")
        self.assertEqual(format_mib_as_gb(5781), "5.65 GB")


class TestFormatGenerationThroughputLine(unittest.TestCase):
    def test_rounds_to_readme_style(self) -> None:
        self.assertEqual(
            format_generation_throughput_line(91.24),
            "* Generation throughput: ~91 tok/s",
        )

    def test_na_when_missing(self) -> None:
        self.assertEqual(
            format_generation_throughput_line(None),
            "* Generation throughput: n/a",
        )


class TestFormatSummaryLines(unittest.TestCase):
    def test_output_shape(self) -> None:
        summary = {
            "gguf_gb": 9.55,
            "model_vram_mb": 10353,
            "kv_vram_mb": 4414,
            "estimated_context_max": 128000,
            "model_max_context": 128000,
            "generation_throughput_tok_s": 91.24,
        }
        lines = format_summary_lines(summary)
        self.assertEqual(len(lines), 6)
        self.assertEqual(lines[0], "* Generation throughput: ~91 tok/s")
        self.assertEqual(lines[1], "* Estimated Max Context: 128000 tokens")
        self.assertEqual(lines[2], "* Model Max Context: 128000 tokens")
        self.assertEqual(lines[3], "* Model VRAM: 10.11 GB")
        self.assertEqual(lines[4], "* KV VRAM: 4.31 GB")
        self.assertEqual(lines[5], "* GGUF on disk: 9.55 GB")

    def test_model_max_context_na(self) -> None:
        summary = {
            "gguf_gb": 1.0,
            "model_vram_mb": 1000,
            "kv_vram_mb": 2000,
            "estimated_context_max": 4096,
        }
        lines = format_summary_lines(summary)
        self.assertEqual(lines[0], "* Generation throughput: n/a")
        self.assertEqual(lines[2], "* Model Max Context: n/a")

    def test_includes_model_system_ram_after_model_vram(self) -> None:
        summary = {
            "gguf_gb": 9.55,
            "model_vram_mb": 10353,
            "kv_vram_mb": 4414,
            "estimated_context_max": 128000,
            "model_max_context": 128000,
            "generation_throughput_tok_s": 91.24,
            "model_system_ram_mb": 24000,
        }
        lines = format_summary_lines(summary)
        self.assertEqual(len(lines), 7)
        self.assertEqual(lines[3], "* Model VRAM: 10.11 GB")
        self.assertEqual(lines[4], "* Model System RAM: 23.44 GB")
        self.assertEqual(lines[5], "* KV VRAM: 4.31 GB")

    def test_model_system_ram_unavailable(self) -> None:
        summary = {
            "gguf_gb": 1.0,
            "model_vram_mb": 1000,
            "kv_vram_mb": 2000,
            "estimated_context_max": 4096,
            "model_system_ram_mb": None,
        }
        lines = format_summary_lines(summary)
        self.assertEqual(len(lines), 7)
        self.assertEqual(lines[3], "* Model VRAM: 0.98 GB")
        self.assertEqual(lines[4], "* Model System RAM: unavailable")
        self.assertEqual(lines[5], "* KV VRAM: 1.95 GB")


if __name__ == "__main__":
    unittest.main()
