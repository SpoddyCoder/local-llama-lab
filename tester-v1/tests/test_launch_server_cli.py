"""Unit tests for launch_server CLI."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from launch_server import (  # noqa: E402
    main,
    resolve_launch_config_paths,
    run_launch_server,
)
from paths import MODELS_ROOT  # noqa: E402

_MODEL_DIR = MODELS_ROOT / "qwen3.5-9b-q8"
_BENCH_DIR = _MODEL_DIR / "bench"


@contextmanager
def _capture_output():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
        yield stdout, stderr


class TestResolveLaunchConfigPaths(unittest.TestCase):
    def test_model_dir_resolves_bench(self) -> None:
        server, client, model_yaml = resolve_launch_config_paths(_MODEL_DIR)
        self.assertEqual(server, _BENCH_DIR / "server.yaml")
        self.assertEqual(client, _BENCH_DIR / "client.yaml")
        self.assertEqual(model_yaml, _MODEL_DIR / "model.yaml")

    def test_variant_dir_resolves_directly(self) -> None:
        server, client, model_yaml = resolve_launch_config_paths(_BENCH_DIR)
        self.assertEqual(server, _BENCH_DIR / "server.yaml")
        self.assertEqual(client, _BENCH_DIR / "client.yaml")
        self.assertEqual(model_yaml, _MODEL_DIR / "model.yaml")

    def test_missing_dir_raises(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_launch_config_paths(MODELS_ROOT / "missing-model")
        self.assertIn("Config directory not found", str(ctx.exception))


class TestRunLaunchServer(unittest.TestCase):
    @contextmanager
    def _fake_managed_server(self, captured: dict[str, object] | None = None):
        @contextmanager
        def fake_managed(_server, _base_url, *, inherit_stdio=False):
            if captured is not None:
                captured["inherit_stdio"] = inherit_stdio
            proc = MagicMock()
            proc.ready_endpoint = "/health"
            yield proc

        with patch("launch_server.managed_server", fake_managed):
            with patch("launch_server.time.sleep", side_effect=KeyboardInterrupt):
                yield

    def test_uses_inherit_stdio_managed_server(self) -> None:
        captured: dict[str, object] = {}
        with self._fake_managed_server(captured):
            code = run_launch_server(_MODEL_DIR)
        self.assertEqual(code, 0)
        self.assertTrue(captured["inherit_stdio"])

    def test_n_cpu_moe_passes_augmented_server(self) -> None:
        captured: dict[str, object] = {}

        @contextmanager
        def fake_managed(server, _base_url, *, inherit_stdio=False):
            captured["server"] = server
            captured["inherit_stdio"] = inherit_stdio
            proc = MagicMock()
            proc.ready_endpoint = "/health"
            yield proc

        with patch("launch_server.managed_server", fake_managed):
            with patch("launch_server.time.sleep", side_effect=KeyboardInterrupt):
                code = run_launch_server(_MODEL_DIR, n_cpu_moe=22)
        self.assertEqual(code, 0)
        server = captured["server"]
        self.assertIn("--n-cpu-moe", server.args)
        self.assertIn("22", server.args)
        self.assertTrue(captured["inherit_stdio"])


class TestLaunchServerMain(unittest.TestCase):
    def test_main_model_dir(self) -> None:
        with patch("launch_server.run_launch_server", return_value=0) as run:
            code = main([str(_MODEL_DIR)])
        self.assertEqual(code, 0)
        run.assert_called_once_with(_MODEL_DIR, n_cpu_moe=None)

    def test_main_n_cpu_moe(self) -> None:
        with patch("launch_server.run_launch_server", return_value=0) as run:
            main([str(_MODEL_DIR), "--n-cpu-moe", "22"])
        run.assert_called_once_with(_MODEL_DIR, n_cpu_moe=22)

    def test_main_missing_dir(self) -> None:
        with _capture_output() as (_stdout, stderr):
            code = main(["models/missing-model"])
        self.assertEqual(code, 1)
        self.assertIn("Config directory not found", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
