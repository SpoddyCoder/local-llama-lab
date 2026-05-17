"""Unit tests for single_test_runner CLI (_run, main)."""

from __future__ import annotations

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
from runner import _TESTER_ROOT, _run, main  # noqa: E402


@contextmanager
def _capture_output():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
        yield stdout, stderr


class TestRunSaveResult(unittest.TestCase):
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
                patch("runner._load_server", return_value=server),
                patch("runner.load_client_config", return_value=client),
                patch("runner.resolve_base_url", return_value="http://127.0.0.1:8080"),
                patch("runner.managed_server", fake_managed_server),
                patch("runner.run_chat_completion", return_value=completion),
                patch("runner.sample_vram_mb", return_value=None),
                patch("runner.VramPoller") as poller_cls,
                patch(
                    "runner.build_metrics_dict",
                    return_value={"wall_time_s": 1.0},
                ),
                patch("runner.collect_run_metadata", return_value={}),
                patch("runner.write_result", return_value=result_path) as write_result,
                _capture_output() as (stdout, stderr),
            ):
                poller = poller_cls.return_value
                poller.stop.return_value = None
                code = _run(
                    Path(tmp) / "server.yaml",
                    Path(tmp) / "client.yaml",
                    save_result=True,
                    quiet=True,
                )

            self.assertEqual(code, 0)
            write_result.assert_called_once()
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
                patch("runner._load_server", return_value=server),
                patch("runner.load_client_config", return_value=client),
                patch("runner.resolve_base_url", return_value="http://127.0.0.1:8080"),
                patch("runner.managed_server", fake_managed_server),
                patch("runner.run_chat_completion", return_value=completion),
                patch("runner.sample_vram_mb", return_value=None),
                patch("runner.VramPoller") as poller_cls,
                patch(
                    "runner.build_metrics_dict",
                    return_value={"wall_time_s": 1.0},
                ),
                patch("runner.collect_run_metadata", return_value={}),
                patch("runner.write_result", return_value=result_path),
                _capture_output() as (stdout, stderr),
            ):
                poller = poller_cls.return_value
                poller.stop.return_value = None
                code = _run(
                    Path(tmp) / "server.yaml",
                    Path(tmp) / "client.yaml",
                    save_result=True,
                    quiet=False,
                )

            self.assertEqual(code, 0)
            self.assertIn("Run:", stdout.getvalue())
            self.assertIn("wall_time_s", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")


class TestRunStdoutDefault(unittest.TestCase):
    def _fake_server(self, tmp: str) -> ServerConfig:
        model_path = Path(tmp) / "model.gguf"
        model_path.write_bytes(b"gguf")
        return ServerConfig(model=str(model_path), args=["-c", "4096"])

    def test_save_result_false_prints_metrics_not_run_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = self._fake_server(tmp)
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])

            @contextmanager
            def fake_managed_server(*_args, **_kwargs):
                proc = MagicMock()
                proc.server_ready_s = 0.1
                yield proc

            completion = MagicMock()
            completion.completion_text = ""
            with (
                patch("runner._load_server", return_value=server),
                patch("runner.load_client_config", return_value=client),
                patch("runner.resolve_base_url", return_value="http://127.0.0.1:8080"),
                patch("runner.managed_server", fake_managed_server),
                patch("runner.run_chat_completion", return_value=completion),
                patch("runner.sample_vram_mb", return_value=None),
                patch("runner.VramPoller") as poller_cls,
                patch(
                    "runner.build_metrics_dict",
                    return_value={"wall_time_s": 1.0},
                ),
                patch("runner.write_result") as write_result,
                _capture_output() as (stdout, stderr),
            ):
                poller = poller_cls.return_value
                poller.stop.return_value = None
                code = _run(
                    Path(tmp) / "server.yaml",
                    Path(tmp) / "client.yaml",
                    save_result=False,
                    quiet=False,
                )

            self.assertEqual(code, 0)
            write_result.assert_not_called()
            out = stdout.getvalue()
            self.assertIn("wall_time_s", out)
            self.assertNotIn("Run:", out)
            self.assertEqual(stderr.getvalue(), "")


class TestMainArgparse(unittest.TestCase):
    def test_main_no_args_prints_help(self) -> None:
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            code = main([])
        self.assertEqual(code, 0)
        self.assertIn("usage:", stdout.getvalue())
        self.assertIn("config_dir", stdout.getvalue())

    def test_main_save_result_quiet_passes_flags_to_run(self) -> None:
        config = "configs/qwen3.5-9b-q8/hello-world-baseline"
        with patch("runner.resolve_config_paths") as resolve:
            resolve.return_value = (_TESTER_ROOT / "server.yaml", _TESTER_ROOT / "client.yaml")
            with patch("runner._run", return_value=0) as run:
                main([config, "--save-result", "--quiet"])
        run.assert_called_once()
        kwargs = run.call_args.kwargs
        self.assertTrue(kwargs["save_result"])
        self.assertTrue(kwargs["quiet"])

    def test_main_quiet_without_save_result_is_noop(self) -> None:
        config = "configs/qwen3.5-9b-q8/hello-world-baseline"
        with patch("runner.resolve_config_paths") as resolve:
            resolve.return_value = (_TESTER_ROOT / "server.yaml", _TESTER_ROOT / "client.yaml")
            with patch("runner._run", return_value=0) as run:
                main([config, "--quiet"])
        run.assert_called_once()
        kwargs = run.call_args.kwargs
        self.assertFalse(kwargs["save_result"])
        self.assertFalse(kwargs["quiet"])

    def test_main_rejects_removed_test_client_flag(self) -> None:
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            with self.assertRaises(SystemExit):
                main(["--test-client"])
        self.assertIn("unrecognized arguments", stderr.getvalue())
        self.assertIn("--test-client", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
