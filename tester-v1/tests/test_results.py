"""Unit tests for result document and run_id generation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import ClientConfig, ServerConfig  # noqa: E402
from results import (  # noqa: E402
    build_result_document,
    format_compact_utc,
    format_run_summary,
    make_run_id,
    write_result,
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
            label=None,
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
            "peak_vram_mb": None,
        }

    def test_document_shape_ok(self) -> None:
        doc = build_result_document(
            server_config=self.server,
            client_config=self.client,
            metrics_dict=self.metrics,
            status="ok",
            started_at=self.started,
            finished_at=self.finished,
        )
        self.assertEqual(doc["run_id"], "20260516T134500Z_Qwen_Qwen3.5-9B-Q8_0")
        self.assertEqual(doc["started_at"], "2026-05-16T13:45:00Z")
        self.assertEqual(doc["finished_at"], "2026-05-16T13:47:12Z")
        self.assertEqual(doc["status"], "ok")
        self.assertIsNone(doc["error"])
        self.assertEqual(doc["server_config"]["model"], self.server.model)
        self.assertEqual(doc["client_config"]["params"]["max_tokens"], 1024)
        self.assertEqual(doc["metrics"]["server_ready_s"], 45.2)
        self.assertIsNone(doc["metrics"]["idle_vram_mb"])
        self.assertIsNone(doc["metrics"]["peak_vram_mb"])

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
        self.assertIsNone(doc["metrics"]["peak_vram_mb"])

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
    def test_writes_named_file(self) -> None:
        doc = {
            "run_id": "20260516T134500Z_test-model",
            "status": "ok",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = write_result(tmp, doc)
            self.assertEqual(path.name, "20260516T134500Z_test-model.json")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["run_id"], doc["run_id"])


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
        }
        root = Path("/tmp/tester-v1")
        result_path = root / "results" / "20260516T134500Z_Qwen_Qwen3.5-9B-Q8_0.json"
        text = format_run_summary(
            "20260516T134500Z_Qwen_Qwen3.5-9B-Q8_0",
            result_path,
            server,
            metrics,
            tester_root=root,
        )
        self.assertIn("Run: 20260516T134500Z_Qwen_Qwen3.5-9B-Q8_0", text)
        self.assertIn("Model: /path/Qwen_Qwen3.5-9B-Q8_0.gguf", text)
        self.assertIn("Result: results/20260516T134500Z_Qwen_Qwen3.5-9B-Q8_0.json", text)
        self.assertIn("wall_time_s", text)
        self.assertIn("132.40", text)


if __name__ == "__main__":
    unittest.main()
