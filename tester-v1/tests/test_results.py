"""Unit tests for result document and run_id generation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
_TESTER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

from config import ClientConfig, ServerConfig, redact_model_path  # noqa: E402
from results import (  # noqa: E402
    build_result_document,
    empty_metrics,
    format_compact_utc,
    format_run_summary,
    make_run_id,
    write_result,
)


class TestRedactModelPath(unittest.TestCase):
    def test_huggingface_hub_path(self) -> None:
        full = (
            "/home/fernpa/.cache/huggingface/hub/"
            "models--bartowski--Qwen_Qwen3.5-9B-GGUF/snapshots/"
            "ff13963796ee209598509a81340172bb1c3869fe/Qwen_Qwen3.5-9B-Q8_0.gguf"
        )
        self.assertEqual(
            redact_model_path(full),
            "models--bartowski--Qwen_Qwen3.5-9B-GGUF/snapshots/"
            "ff13963796ee209598509a81340172bb1c3869fe/Qwen_Qwen3.5-9B-Q8_0.gguf",
        )

    def test_non_hub_path_uses_basename(self) -> None:
        self.assertEqual(
            redact_model_path("/home/user/models/Qwen_Qwen3.5-9B-Q8_0.gguf"),
            "Qwen_Qwen3.5-9B-Q8_0.gguf",
        )


class TestMakeRunId(unittest.TestCase):
    def test_compact_timestamp_and_slug(self) -> None:
        ts = datetime(2026, 5, 16, 13, 45, 0, tzinfo=timezone.utc)
        self.assertEqual(
            make_run_id(ts, "Qwen_Qwen3.5-9B-Q8_0"),
            "20260516T134500Z_Qwen_Qwen3.5-9B-Q8_0",
        )

    def test_format_compact_utc(self) -> None:
        ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        self.assertEqual(format_compact_utc(ts), "20260102T030405Z")


class TestBuildResultDocument(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ServerConfig(
            model="/home/user/models/Qwen_Qwen3.5-9B-Q8_0.gguf",
            args=["--host", "0.0.0.0", "--port", "8080"],
        )
        self.client = ClientConfig(
            messages=[{"role": "user", "content": "hi"}],
            base_url="http://127.0.0.1:8080",
            params={"max_tokens": 1024, "temperature": 0},
            timeout_s=600.0,
        )
        self.started = datetime(2026, 5, 16, 13, 45, 0, tzinfo=timezone.utc)
        self.finished = datetime(2026, 5, 16, 13, 47, 12, tzinfo=timezone.utc)
        self.metrics = {
            "wall_time_s": 132.4,
            "ttft_s": 0.85,
            "completion_time_s": 131.55,
            "prompt_tokens": 42,
            "completion_tokens": 980,
            "total_tokens": 1022,
            "prefill_tok_s": 49.4,
            "decode_tok_s": 7.5,
            "tokens_per_second": 7.4,
            "server_ready_s": 45.2,
            "idle_vram_mb": None,
            "idle_system_ram_mb": None,
            "peak_vram_mb": None,
        }
        self.config_dir = (
            _TESTER_ROOT / "configs" / "qwen3.5-9b-q8" / "hello-world-baseline"
        )

    def test_document_shape_ok(self) -> None:
        doc = build_result_document(
            server_config=self.server,
            client_config=self.client,
            metrics_dict=self.metrics,
            status="ok",
            started_at=self.started,
            finished_at=self.finished,
        )
        self.assertEqual(doc["schema_version"], "1")
        self.assertEqual(doc["suite"], "probe")
        self.assertIsNone(doc["session_id"])
        self.assertEqual(doc["run_id"], "20260516T134500Z_Qwen_Qwen3.5-9B-Q8_0")
        self.assertEqual(doc["started_at"], "2026-05-16T13:45:00Z")
        self.assertEqual(doc["finished_at"], "2026-05-16T13:47:12Z")
        self.assertEqual(doc["status"], "ok")
        self.assertIsNone(doc["error"])
        self.assertEqual(
            doc["server_config"]["model"],
            redact_model_path(self.server.model),
        )
        self.assertEqual(doc["client_config"]["params"]["max_tokens"], 1024)
        self.assertEqual(doc["metrics"]["server_ready_s"], 45.2)
        self.assertIsNone(doc["metrics"]["idle_vram_mb"])
        self.assertIsNone(doc["metrics"]["idle_system_ram_mb"])
        self.assertIsNone(doc["metrics"]["peak_vram_mb"])
        self.assertEqual(
            doc["metadata"],
            {
                "server_version": None,
                "gpu_name": None,
                "driver_version": None,
                "config_path": None,
                "model": None,
                "variant": None,
            },
        )

    def test_metadata_passed_through(self) -> None:
        meta = {
            "server_version": "version: 9158 (abc)",
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "driver_version": "550.54.15",
            "config_path": "qwen3.5-9b-q8/hello-world-baseline/",
            "model": "qwen3.5-9b-q8",
            "variant": "hello-world-baseline",
        }
        doc = build_result_document(
            server_config=self.server,
            client_config=self.client,
            metrics_dict=self.metrics,
            status="ok",
            started_at=self.started,
            finished_at=self.finished,
            metadata=meta,
        )
        self.assertEqual(doc["metadata"], meta)

    def test_error_status_with_empty_metrics(self) -> None:
        doc = build_result_document(
            server_config=self.server,
            client_config=self.client,
            metrics_dict=None,
            status="error",
            started_at=self.started,
            finished_at=self.finished,
            error="server not ready",
        )
        self.assertEqual(doc["status"], "error")
        self.assertEqual(doc["error"], "server not ready")
        self.assertIsNone(doc["metrics"]["wall_time_s"])
        self.assertIsNone(doc["metrics"]["server_ready_s"])
        self.assertIsNone(doc["metrics"]["idle_vram_mb"])
        self.assertIsNone(doc["metrics"]["idle_system_ram_mb"])
        self.assertIsNone(doc["metrics"]["peak_vram_mb"])

    def test_empty_metrics_includes_idle_system_ram_mb(self) -> None:
        m = empty_metrics()
        self.assertIn("idle_system_ram_mb", m)
        self.assertIsNone(m["idle_system_ram_mb"])

    def test_build_result_document_defaults_missing_idle_system_ram_mb(self) -> None:
        metrics_no_sys_ram = dict(self.metrics)
        del metrics_no_sys_ram["idle_system_ram_mb"]
        doc = build_result_document(
            server_config=self.server,
            client_config=self.client,
            metrics_dict=metrics_no_sys_ram,
            status="ok",
            started_at=self.started,
            finished_at=self.finished,
        )
        self.assertIn("idle_system_ram_mb", doc["metrics"])
        self.assertIsNone(doc["metrics"]["idle_system_ram_mb"])

    def test_config_dir_drives_run_id_and_suite(self) -> None:
        meta = {
            "server_version": None,
            "gpu_name": None,
            "driver_version": None,
            "config_path": "qwen3.5-9b-q8/hello-world-baseline/",
            "model": "qwen3.5-9b-q8",
            "variant": "hello-world-baseline",
        }
        doc = build_result_document(
            server_config=self.server,
            client_config=self.client,
            metrics_dict=self.metrics,
            status="ok",
            started_at=self.started,
            finished_at=self.finished,
            metadata=meta,
            config_dir=self.config_dir,
            tester_root=_TESTER_ROOT,
            session_id="sess-abc",
        )
        self.assertEqual(
            doc["run_id"],
            "20260516T134500Z_qwen3.5-9b-q8-hello-world-baseline",
        )
        self.assertEqual(doc["suite"], "probe")
        self.assertEqual(doc["session_id"], "sess-abc")

    def test_fallback_run_id_uses_model_slug_without_config_dir(self) -> None:
        doc = build_result_document(
            server_config=self.server,
            client_config=self.client,
            metrics_dict=self.metrics,
            status="ok",
            started_at=self.started,
            finished_at=self.finished,
            config_dir=None,
            tester_root=None,
        )
        self.assertEqual(
            doc["run_id"],
            "20260516T134500Z_Qwen_Qwen3.5-9B-Q8_0",
        )
        self.assertEqual(doc["suite"], "probe")
        self.assertIsNone(doc["session_id"])

    def test_calibration_variant_suite(self) -> None:
        meta = {
            "server_version": None,
            "gpu_name": None,
            "driver_version": None,
            "config_path": "qwen3.5-9b-q8/calibration-footprint/",
            "model": "qwen3.5-9b-q8",
            "variant": "calibration-footprint",
        }
        config_dir = _TESTER_ROOT / "configs" / "qwen3.5-9b-q8" / "calibration-footprint"
        doc = build_result_document(
            server_config=self.server,
            client_config=self.client,
            metrics_dict=self.metrics,
            status="ok",
            started_at=self.started,
            finished_at=self.finished,
            metadata=meta,
            config_dir=config_dir,
            tester_root=_TESTER_ROOT,
        )
        self.assertEqual(doc["suite"], "calibration")

    def test_json_serializable(self) -> None:
        doc = build_result_document(
            server_config=self.server,
            client_config=self.client,
            metrics_dict=self.metrics,
            status="ok",
            started_at=self.started,
            finished_at=self.finished,
        )
        text = json.dumps(doc)
        parsed = json.loads(text)
        self.assertEqual(parsed["run_id"], doc["run_id"])


class TestWriteResult(unittest.TestCase):
    def test_writes_hierarchical_file(self) -> None:
        started = datetime(2026, 5, 16, 13, 45, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = (
                root / "configs" / "qwen3.5-9b-q8" / "hello-world-baseline"
            )
            config_dir.mkdir(parents=True)
            doc = build_result_document(
                server_config=ServerConfig(
                    model="/home/user/models/Qwen_Qwen3.5-9B-Q8_0.gguf",
                    args=[],
                ),
                client_config=ClientConfig(messages=[{"role": "user", "content": "hi"}]),
                metrics_dict=None,
                status="ok",
                started_at=started,
                finished_at=started,
                metadata={
                    "server_version": None,
                    "gpu_name": None,
                    "driver_version": None,
                    "config_path": "qwen3.5-9b-q8/hello-world-baseline/",
                    "model": "qwen3.5-9b-q8",
                    "variant": "hello-world-baseline",
                },
                config_dir=config_dir,
                tester_root=root,
            )
            results_dir = root / "results"
            path = write_result(
                results_dir,
                doc,
                config_dir=config_dir,
                tester_root=root,
                started_at=started,
            )
            self.assertEqual(
                path,
                root
                / "results"
                / "qwen3.5-9b-q8"
                / "hello-world-baseline"
                / "20260516T134500Z.json",
            )
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["run_id"], doc["run_id"])

    def test_writes_other_layout_for_outside_configs(self) -> None:
        started = datetime(2026, 5, 17, 17, 2, 40, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = Path("/tmp/my-run")
            doc = build_result_document(
                server_config=ServerConfig(
                    model="/home/user/models/Qwen_Qwen3.5-9B-Q8_0.gguf",
                    args=[],
                ),
                client_config=ClientConfig(messages=[{"role": "user", "content": "hi"}]),
                metrics_dict=None,
                status="ok",
                started_at=started,
                finished_at=started,
                metadata={
                    "server_version": None,
                    "gpu_name": None,
                    "driver_version": None,
                    "config_path": None,
                    "model": None,
                    "variant": None,
                },
                config_dir=config_dir,
                tester_root=root,
            )
            results_dir = root / "results"
            path = write_result(
                results_dir,
                doc,
                config_dir=config_dir,
                tester_root=root,
                started_at=started,
            )
            self.assertEqual(
                path,
                root / "results" / "_other" / "tmp-my-run" / "20260517T170240Z.json",
            )
            self.assertEqual(
                doc["run_id"],
                "20260517T170240Z_tmp-my-run",
            )


class TestFormatRunSummary(unittest.TestCase):
    def test_includes_run_model_result_and_metrics(self) -> None:
        server = ServerConfig(
            model="/path/Qwen_Qwen3.5-9B-Q8_0.gguf",
            args=[],
        )
        metrics = {
            "wall_time_s": 132.4,
            "ttft_s": 0.85,
            "prompt_tokens": 42,
            "completion_tokens": 980,
            "prefill_tok_s": 49.4,
            "decode_tok_s": 7.5,
            "peak_vram_mb": 8192,
        }
        root = Path("/tmp/tester-v1")
        result_path = (
            root
            / "results"
            / "qwen3.5-9b-q8"
            / "hello-world-baseline"
            / "20260516T134500Z.json"
        )
        text = format_run_summary(
            "20260516T134500Z_qwen3.5-9b-q8-hello-world-baseline",
            result_path,
            server,
            metrics,
            tester_root=root,
        )
        self.assertIn(
            "Run: 20260516T134500Z_qwen3.5-9b-q8-hello-world-baseline", text
        )
        self.assertIn("Model: Qwen_Qwen3.5-9B-Q8_0.gguf", text)
        self.assertIn(
            "Result: results/qwen3.5-9b-q8/hello-world-baseline/20260516T134500Z.json",
            text,
        )
        self.assertIn("wall_time_s", text)
        self.assertIn("132.40", text)
        self.assertIn("End-to-end time", text)
        self.assertIn("Peak VRAM", text)
        self.assertIn("Generation throughput", text)
        self.assertIn("132.40 s", text)
        self.assertIn("8.00 GiB", text)
        self.assertIn("7.50 tok/s", text)


if __name__ == "__main__":
    unittest.main()
