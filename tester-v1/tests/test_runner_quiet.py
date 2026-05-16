"""Unit tests for --quiet behavior in python_runner."""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import ClientConfig, ServerConfig  # noqa: E402
from python_runner import _TESTER_ROOT, _run_default, main  # noqa: E402


@contextmanager
def _capture_output():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
        yield stdout, stderr


class TestQuietArgparse(unittest.TestCase):
    def test_main_parser_accepts_quiet(self) -> None:
        with patch("python_runner.resolve_config_paths") as resolve:
            resolve.return_value = (_TESTER_ROOT / "server.yaml", _TESTER_ROOT / "client.yaml")
            with patch("python_runner._run_default", return_value=0) as run_default:
                main(["--quiet"])
        run_default.assert_called_once()
        self.assertTrue(run_default.call_args.kwargs.get("quiet"))


class TestRunDefaultQuiet(unittest.TestCase):
    def _fake_server(self, tmp: str) -> ServerConfig:
        model_path = Path(tmp) / "model.gguf"
        model_path.write_bytes(b"gguf")
        return ServerConfig(model=str(model_path), args=["-c", "4096"])

    def test_quiet_suppresses_stdout_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = self._fake_server(tmp)
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])
            result_path = _TESTER_ROOT / "results" / "20260101T000000Z_test.json"

            @contextmanager
            def fake_managed_server(*_args, **_kwargs):
                proc = MagicMock()
                proc.server_ready_s = 0.1
                yield proc

            completion = MagicMock()
            with (
                patch("python_runner._load_server", return_value=server),
                patch("python_runner.load_client_config", return_value=client),
                patch("python_runner.resolve_base_url", return_value="http://127.0.0.1:8080"),
                patch("python_runner.managed_server", fake_managed_server),
                patch("python_runner.run_chat_completion", return_value=completion),
                patch("python_runner.sample_vram_mb", return_value=None),
                patch("python_runner.VramPoller") as poller_cls,
                patch(
                    "python_runner.build_metrics_dict",
                    return_value={"wall_time_s": 1.0},
                ),
                patch("python_runner.collect_run_metadata", return_value={}),
                patch("python_runner.write_result", return_value=result_path),
                _capture_output() as (stdout, stderr),
            ):
                poller = poller_cls.return_value
                poller.stop.return_value = None
                code = _run_default(
                    Path(tmp) / "server.yaml",
                    Path(tmp) / "client.yaml",
                    quiet=True,
                )

            self.assertEqual(code, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("Wrote results/", stderr.getvalue())
            self.assertIn("20260101T000000Z_test.json", stderr.getvalue())

    def test_non_quiet_prints_stdout_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = self._fake_server(tmp)
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])
            result_path = _TESTER_ROOT / "results" / "20260101T000000Z_test.json"

            @contextmanager
            def fake_managed_server(*_args, **_kwargs):
                proc = MagicMock()
                proc.server_ready_s = 0.1
                yield proc

            completion = MagicMock()
            with (
                patch("python_runner._load_server", return_value=server),
                patch("python_runner.load_client_config", return_value=client),
                patch("python_runner.resolve_base_url", return_value="http://127.0.0.1:8080"),
                patch("python_runner.managed_server", fake_managed_server),
                patch("python_runner.run_chat_completion", return_value=completion),
                patch("python_runner.sample_vram_mb", return_value=None),
                patch("python_runner.VramPoller") as poller_cls,
                patch(
                    "python_runner.build_metrics_dict",
                    return_value={"wall_time_s": 1.0},
                ),
                patch("python_runner.collect_run_metadata", return_value={}),
                patch("python_runner.write_result", return_value=result_path),
                _capture_output() as (stdout, stderr),
            ):
                poller = poller_cls.return_value
                poller.stop.return_value = None
                code = _run_default(
                    Path(tmp) / "server.yaml",
                    Path(tmp) / "client.yaml",
                    quiet=False,
                )

            self.assertEqual(code, 0)
            self.assertIn("Run:", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
