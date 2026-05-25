"""Unit tests for probe runner CLI (_run, main)."""

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

from calibration import parse_idle_system_ram_from_metrics_stdout  # noqa: E402
from config import ClientConfig, ServerConfig  # noqa: E402
from model_layout import RunConfig  # noqa: E402
from paths import RESULTS_DIR  # noqa: E402
from runner import _run, main  # noqa: E402


def _make_run_config(
    tmp: str,
    *,
    model: str = "test-model",
    variant: str = "hello-world-baseline",
    server_args: list[str] | None = None,
) -> RunConfig:
    model_path = Path(tmp) / "model.gguf"
    model_path.write_bytes(b"gguf")
    return RunConfig(
        server=ServerConfig(
            model=str(model_path),
            args=server_args or ["-c", "4096"],
        ),
        client_path=Path(tmp) / "client.yaml",
        model_yaml_path=Path(tmp) / "model.yaml",
        model=model,
        variant=variant,
    )


@contextmanager
def _capture_output():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
        yield stdout, stderr


class TestRunSaveResult(unittest.TestCase):
    def test_save_result_requires_model_and_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_config = _make_run_config(tmp, model="", variant="")
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])
            with (
                patch("runner.load_client_config", return_value=client),
                _capture_output() as (_stdout, stderr),
            ):
                code = _run(
                    run_config,
                    save_result=True,
                    quiet=False,
                    include_output=False,
                )
            self.assertEqual(code, 1)
            self.assertIn("--save-result requires model and variant", stderr.getvalue())

    def test_quiet_suppresses_stdout_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_config = _make_run_config(tmp, model="qwen3.5-9b-q8")
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])
            result_path = (
                RESULTS_DIR
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
                    run_config,
                    save_result=True,
                    quiet=True,
                    include_output=False,
                )

            self.assertEqual(code, 0)
            write_result.assert_called_once()
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("Wrote results/", stderr.getvalue())
            self.assertIn("20260101T000000Z.json", stderr.getvalue())

    def test_non_quiet_prints_stdout_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_config = _make_run_config(tmp, model="qwen3.5-9b-q8")
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])
            result_path = (
                RESULTS_DIR
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
                    run_config,
                    save_result=True,
                    quiet=False,
                    include_output=False,
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

    def test_save_result_false_prints_probe_stdout_not_run_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_config = _make_run_config(tmp)
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])

            @contextmanager
            def fake_managed_server(*_args, **_kwargs):
                proc = MagicMock()
                proc.server_ready_s = 0.1
                yield proc

            completion = MagicMock()
            completion.completion_text = ""
            with (
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
                    run_config,
                    save_result=False,
                    quiet=False,
                    include_output=False,
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

    def test_include_output_with_save_result_writes_output_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_config = _make_run_config(tmp, model="qwen3.5-9b-q8")
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
                    run_config,
                    save_result=True,
                    quiet=True,
                    include_output=True,
                )

            self.assertEqual(code, 0)
            self.assertTrue(output_path.is_file())
            self.assertEqual(
                output_path.read_text(encoding="utf-8"), "hello from model"
            )

    def test_include_output_without_save_result_prints_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_config = _make_run_config(tmp)
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])

            @contextmanager
            def fake_managed_server(*_args, **_kwargs):
                proc = MagicMock()
                proc.server_ready_s = 0.1
                yield proc

            completion = MagicMock()
            completion.completion_text = "hello from model"
            with (
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
                    run_config,
                    save_result=False,
                    quiet=False,
                    include_output=True,
                )

            self.assertEqual(code, 0)
            write_result.assert_not_called()
            out = stdout.getvalue()
            self.assertIn("wall_time_s", out)
            self.assertIn("hello from model", out)
            self.assertEqual(stderr.getvalue(), "")


class TestRunNCpuMoe(unittest.TestCase):
    def test_run_uses_server_from_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_config = _make_run_config(
                tmp,
                server_args=["-c", "4096", "--n-cpu-moe", "22", "--n-gpu-layers", "999"],
            )
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])
            captured_servers: list[ServerConfig] = []

            @contextmanager
            def fake_managed_server(server_arg, *_args, **_kwargs):
                captured_servers.append(server_arg)
                proc = MagicMock()
                proc.server_ready_s = 0.1
                yield proc

            completion = MagicMock()
            completion.completion_text = ""
            with (
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
                _capture_output(),
            ):
                poller = poller_cls.return_value
                poller.stop.return_value = None
                code = _run(
                    run_config,
                    save_result=False,
                    quiet=False,
                    include_output=False,
                )

            self.assertEqual(code, 0)
            self.assertEqual(len(captured_servers), 1)
            self.assertEqual(
                captured_servers[0].args,
                ["-c", "4096", "--n-cpu-moe", "22", "--n-gpu-layers", "999"],
            )

    def test_save_result_sets_idle_system_ram_mb_when_n_cpu_moe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_config = _make_run_config(
                tmp,
                model="qwen3.5-9b-q8",
                server_args=["-c", "4096", "--n-cpu-moe", "22", "--n-gpu-layers", "999"],
            )
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])
            result_path = (
                RESULTS_DIR
                / "qwen3.5-9b-q8"
                / "hello-world-baseline"
                / "20260101T000000Z.json"
            )

            @contextmanager
            def fake_managed_server(*_args, **_kwargs):
                proc = MagicMock()
                proc.pid = 12345
                proc.server_ready_s = 0.1
                yield proc

            completion = MagicMock()
            with (
                patch("runner.load_client_config", return_value=client),
                patch("runner.resolve_base_url", return_value="http://127.0.0.1:8080"),
                patch("runner.managed_server", fake_managed_server),
                patch("runner.run_chat_completion", return_value=completion),
                patch("runner.sample_vram_mb", return_value=None),
                patch("runner.sample_process_rss_mb", return_value=8888) as sample_rss,
                patch("runner.VramPoller") as poller_cls,
                patch(
                    "runner.build_metrics_dict",
                    return_value={"wall_time_s": 1.0},
                ),
                patch("runner.collect_run_metadata", return_value={}),
                patch("runner.write_result", return_value=result_path) as write_result,
                _capture_output(),
            ):
                poller = poller_cls.return_value
                poller.stop.return_value = None
                code = _run(
                    run_config,
                    save_result=True,
                    quiet=True,
                    include_output=False,
                )

            self.assertEqual(code, 0)
            sample_rss.assert_called_once_with(12345)
            document = write_result.call_args.args[1]
            self.assertEqual(document["metrics"]["idle_system_ram_mb"], 8888)

    def test_probe_stdout_includes_idle_system_ram_parsed_by_calibration_when_n_cpu_moe(
        self,
    ) -> None:
        """MoE run prints idle_system_ram_mb; calibration helpers parse it from captured stdout."""
        with tempfile.TemporaryDirectory() as tmp:
            run_config = _make_run_config(
                tmp,
                server_args=["-c", "4096", "--n-cpu-moe", "22", "--n-gpu-layers", "999"],
            )
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])

            @contextmanager
            def fake_managed_server(*_args, **_kwargs):
                proc = MagicMock()
                proc.pid = 44444
                proc.server_ready_s = 0.1
                yield proc

            completion = MagicMock()
            completion.completion_text = ""
            with (
                patch("runner.load_client_config", return_value=client),
                patch("runner.resolve_base_url", return_value="http://127.0.0.1:8080"),
                patch("runner.managed_server", fake_managed_server),
                patch("runner.run_chat_completion", return_value=completion),
                patch("runner.sample_vram_mb", return_value=None),
                patch("runner.sample_process_rss_mb", return_value=6611) as sample_rss,
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
                    run_config,
                    save_result=False,
                    quiet=False,
                    include_output=False,
                )

            self.assertEqual(code, 0)
            write_result.assert_not_called()
            sample_rss.assert_called_once_with(44444)
            out = stdout.getvalue()
            self.assertEqual(parse_idle_system_ram_from_metrics_stdout(out), 6611)
            self.assertEqual(stderr.getvalue(), "")

    def test_save_result_defaults_idle_system_ram_mb_null_without_n_cpu_moe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_config = _make_run_config(tmp, model="qwen3.5-9b-q8")
            client = ClientConfig(messages=[{"role": "user", "content": "hi"}])
            result_path = (
                RESULTS_DIR
                / "qwen3.5-9b-q8"
                / "hello-world-baseline"
                / "20260101T000000Z.json"
            )

            @contextmanager
            def fake_managed_server(*_args, **_kwargs):
                proc = MagicMock()
                proc.pid = 99999
                proc.server_ready_s = 0.1
                yield proc

            completion = MagicMock()
            with (
                patch("runner.load_client_config", return_value=client),
                patch("runner.resolve_base_url", return_value="http://127.0.0.1:8080"),
                patch("runner.managed_server", fake_managed_server),
                patch("runner.run_chat_completion", return_value=completion),
                patch("runner.sample_vram_mb", return_value=None),
                patch("runner.sample_process_rss_mb") as sample_rss,
                patch("runner.VramPoller") as poller_cls,
                patch(
                    "runner.build_metrics_dict",
                    return_value={"wall_time_s": 1.0},
                ),
                patch("runner.collect_run_metadata", return_value={}),
                patch("runner.write_result", return_value=result_path) as write_result,
                _capture_output(),
            ):
                poller = poller_cls.return_value
                poller.stop.return_value = None
                code = _run(
                    run_config,
                    save_result=True,
                    quiet=True,
                    include_output=False,
                )

            self.assertEqual(code, 0)
            sample_rss.assert_not_called()
            document = write_result.call_args.args[1]
            self.assertIn("idle_system_ram_mb", document["metrics"])
            self.assertIsNone(document["metrics"]["idle_system_ram_mb"])


class TestMainArgparse(unittest.TestCase):
    def _fake_run_config(self) -> RunConfig:
        return RunConfig(
            server=ServerConfig(model="/tmp/model.gguf", args=["-c", "4096"]),
            client_path=Path("/tmp/client.yaml"),
            model_yaml_path=Path("/tmp/model.yaml"),
            model="test-model",
            variant="hello-world-baseline",
        )

    def test_main_save_result_quiet_passes_flags_to_run(self) -> None:
        fake_config = self._fake_run_config()
        with patch("runner.resolve_run_config", return_value=fake_config):
            with patch("runner._run", return_value=0) as run:
                main(["/tmp/model", "--variant", "hello-world-baseline", "--save-result", "--quiet"])
        run.assert_called_once()
        self.assertIs(run.call_args.args[0], fake_config)
        kwargs = run.call_args.kwargs
        self.assertTrue(kwargs["save_result"])
        self.assertTrue(kwargs["quiet"])
        self.assertFalse(kwargs["include_output"])
        self.assertIsNone(kwargs["session_id"])

    def test_main_include_output_without_save_result_passes_bool(self) -> None:
        fake_config = self._fake_run_config()
        with patch("runner.resolve_run_config", return_value=fake_config):
            with patch("runner._run", return_value=0) as run:
                main(["/tmp/model", "--include-output"])
        run.assert_called_once()
        kwargs = run.call_args.kwargs
        self.assertFalse(kwargs["save_result"])
        self.assertTrue(kwargs["include_output"])

    def test_main_include_output_with_save_result_passes_bool(self) -> None:
        fake_config = self._fake_run_config()
        with patch("runner.resolve_run_config", return_value=fake_config):
            with patch("runner._run", return_value=0) as run:
                main(["/tmp/model", "--save-result", "--include-output"])
        run.assert_called_once()
        kwargs = run.call_args.kwargs
        self.assertTrue(kwargs["save_result"])
        self.assertTrue(kwargs["include_output"])

    def test_main_session_id_passed_through(self) -> None:
        fake_config = self._fake_run_config()
        with patch("runner.resolve_run_config", return_value=fake_config):
            with patch("runner._run", return_value=0) as run:
                main(["/tmp/model", "--save-result", "--session-id", "cal-1"])
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["session_id"], "cal-1")

    def test_main_n_cpu_moe_passes_to_resolve_run_config(self) -> None:
        fake_config = self._fake_run_config()
        with patch("runner.resolve_run_config", return_value=fake_config) as resolve:
            with patch("runner._run", return_value=0):
                main(["/tmp/model", "--n-cpu-moe", "22"])
        resolve.assert_called_once()
        self.assertEqual(resolve.call_args.kwargs["n_cpu_moe"], 22)

    def test_main_default_variant_is_hello_world_baseline(self) -> None:
        fake_config = self._fake_run_config()
        with patch("runner.resolve_run_config", return_value=fake_config) as resolve:
            with patch("runner._run", return_value=0):
                main(["/tmp/model"])
        resolve.assert_called_once()
        self.assertEqual(resolve.call_args.args[1], "hello-world-baseline")

    def test_main_sandbox_variant_passed_to_resolve_run_config(self) -> None:
        fake_config = self._fake_run_config()
        with patch("runner.resolve_run_config", return_value=fake_config) as resolve:
            with patch("runner._run", return_value=0):
                main(["/tmp/model", "--variant", "sandbox"])
        resolve.assert_called_once()
        self.assertEqual(resolve.call_args.args[1], "sandbox")

    def test_main_quiet_without_save_result_is_noop(self) -> None:
        fake_config = self._fake_run_config()
        with patch("runner.resolve_run_config", return_value=fake_config):
            with patch("runner._run", return_value=0) as run:
                main(["/tmp/model", "--quiet"])
        run.assert_called_once()
        kwargs = run.call_args.kwargs
        self.assertFalse(kwargs["save_result"])
        self.assertFalse(kwargs["quiet"])
        self.assertFalse(kwargs["include_output"])

    def test_main_rejects_removed_test_server_flag(self) -> None:
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            with self.assertRaises(SystemExit):
                main(["/tmp/model", "--test-server"])
        self.assertIn("unrecognized arguments", stderr.getvalue())
        self.assertIn("--test-server", stderr.getvalue())

    def test_main_rejects_removed_test_client_flag(self) -> None:
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            with self.assertRaises(SystemExit):
                main(["/tmp/model", "--test-client"])
        self.assertIn("unrecognized arguments", stderr.getvalue())
        self.assertIn("--test-client", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
