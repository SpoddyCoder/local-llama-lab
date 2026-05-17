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

_CONFIG_DIR = _TESTER_ROOT / "configs" / "qwen3.5-9b-q8" / "hello-world-baseline"


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

    def test_save_result_requires_config_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = self._fake_server(tmp)
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])
            with (
                patch("runner._load_server", return_value=server),
                patch("runner.load_client_config", return_value=client),
                _capture_output() as (_stdout, stderr),
            ):
                code = _run(
                    Path(tmp) / "server.yaml",
                    Path(tmp) / "client.yaml",
                    save_result=True,
                    quiet=False,
                    full_output=False,
                )
            self.assertEqual(code, 1)
            self.assertIn("--save-result requires config_dir", stderr.getvalue())

    def test_quiet_suppresses_stdout_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = self._fake_server(tmp)
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])
            result_path = (
                _TESTER_ROOT
                / "results"
                / "qwen3.5-9b-q8"
                / "hello-world-baseline"
                / "20260101T000000Z.json"
            )

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
                    _CONFIG_DIR,
                    save_result=True,
                    quiet=True,
                    full_output=False,
                )

            self.assertEqual(code, 0)
            write_result.assert_called_once()
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("Wrote results/", stderr.getvalue())
            self.assertIn("20260101T000000Z.json", stderr.getvalue())

    def test_non_quiet_prints_stdout_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = self._fake_server(tmp)
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])
            result_path = (
                _TESTER_ROOT
                / "results"
                / "qwen3.5-9b-q8"
                / "hello-world-baseline"
                / "20260101T000000Z.json"
            )

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
                    _CONFIG_DIR,
                    save_result=True,
                    quiet=False,
                    full_output=False,
                )

            self.assertEqual(code, 0)
            self.assertIn("Run:", stdout.getvalue())
            self.assertIn("wall_time_s", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")


class TestRunStdoutDefault(unittest.TestCase):
    _PROBE_METRICS = {
        "wall_time_s": 1.0,
        "peak_vram_mb": 2048.0,
        "decode_tok_s": 42.0,
    }

    def _fake_server(self, tmp: str) -> ServerConfig:
        model_path = Path(tmp) / "model.gguf"
        model_path.write_bytes(b"gguf")
        return ServerConfig(model=str(model_path), args=["-c", "4096"])

    def test_save_result_false_prints_probe_stdout_not_run_line(self) -> None:
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
                    return_value=self._PROBE_METRICS.copy(),
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
                    full_output=False,
                )

            self.assertEqual(code, 0)
            write_result.assert_not_called()
            out = stdout.getvalue()
            self.assertIn("wall_time_s", out)
            self.assertIn("End-to-end time", out)
            self.assertIn("Generation throughput", out)
            self.assertNotIn("Run:", out)
            self.assertNotIn("completion preview", out)
            self.assertEqual(stderr.getvalue(), "")

    def test_full_output_with_save_result_writes_output_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = self._fake_server(tmp)
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])
            result_path = (
                Path(tmp)
                / "results"
                / "qwen3.5-9b-q8"
                / "hello-world-baseline"
                / "20260101T000000Z.json"
            )
            output_path = result_path.with_name("20260101T000000Z-output.txt")

            @contextmanager
            def fake_managed_server(*_args, **_kwargs):
                proc = MagicMock()
                proc.server_ready_s = 0.1
                yield proc

            completion = MagicMock()
            completion.completion_text = "hello from model"
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
                _capture_output(),
            ):
                poller = poller_cls.return_value
                poller.stop.return_value = None
                code = _run(
                    Path(tmp) / "server.yaml",
                    Path(tmp) / "client.yaml",
                    _CONFIG_DIR,
                    save_result=True,
                    quiet=True,
                    full_output=True,
                )

            self.assertEqual(code, 0)
            self.assertTrue(output_path.is_file())
            self.assertEqual(
                output_path.read_text(encoding="utf-8"), "hello from model"
            )


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
        self.assertFalse(kwargs["full_output"])
        self.assertIsNone(kwargs["session_id"])

    def test_main_full_output_requires_save_result(self) -> None:
        config = "configs/qwen3.5-9b-q8/hello-world-baseline"
        with patch("runner.resolve_config_paths") as resolve:
            resolve.return_value = (_TESTER_ROOT / "server.yaml", _TESTER_ROOT / "client.yaml")
            with _capture_output() as (_stdout, stderr):
                code = main([config, "--full-output"])
        self.assertEqual(code, 1)
        self.assertIn("--full-output requires --save-result", stderr.getvalue())

    def test_main_full_output_with_save_result_passes_bool(self) -> None:
        config = "configs/qwen3.5-9b-q8/hello-world-baseline"
        with patch("runner.resolve_config_paths") as resolve:
            resolve.return_value = (_TESTER_ROOT / "server.yaml", _TESTER_ROOT / "client.yaml")
            with patch("runner._run", return_value=0) as run:
                main([config, "--save-result", "--full-output"])
        run.assert_called_once()
        kwargs = run.call_args.kwargs
        self.assertTrue(kwargs["save_result"])
        self.assertTrue(kwargs["full_output"])

    def test_main_session_id_passed_through(self) -> None:
        config = "configs/qwen3.5-9b-q8/hello-world-baseline"
        with patch("runner.resolve_config_paths") as resolve:
            resolve.return_value = (_TESTER_ROOT / "server.yaml", _TESTER_ROOT / "client.yaml")
            with patch("runner._run", return_value=0) as run:
                main([config, "--save-result", "--session-id", "cal-1"])
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["session_id"], "cal-1")

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
        self.assertFalse(kwargs["full_output"])

    def test_main_rejects_removed_test_client_flag(self) -> None:
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            with self.assertRaises(SystemExit):
                main(["--test-client"])
        self.assertIn("unrecognized arguments", stderr.getvalue())
        self.assertIn("--test-client", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
