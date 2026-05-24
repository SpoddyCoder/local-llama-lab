"""Unit tests for launch_server CLI."""

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

from config import MODEL_YAML  # noqa: E402
from launch_server import main, run_launch_server  # noqa: E402


def _write_launch_model_dir(root: Path, *, slug: str = "launch-model") -> Path:
    model_dir = root / slug
    model_dir.mkdir(parents=True)
    gguf = model_dir / "model.gguf"
    gguf.write_bytes(b"gguf")
    (model_dir / MODEL_YAML).write_text(f"model: {gguf}\n", encoding="utf-8")
    (model_dir / "server.yaml").write_text(
        "args: |\n"
        "  --host 127.0.0.1\n"
        "  --port 8080\n"
        "  -c 4096\n",
        encoding="utf-8",
    )
    return model_dir


@contextmanager
def _capture_output():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
        yield stdout, stderr


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
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_launch_model_dir(Path(tmp))
            captured: dict[str, object] = {}
            with self._fake_managed_server(captured):
                code = run_launch_server(model_dir)
            self.assertEqual(code, 0)
            self.assertTrue(captured["inherit_stdio"])

    def test_n_cpu_moe_passes_augmented_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_launch_model_dir(Path(tmp))
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
                    code = run_launch_server(model_dir, n_cpu_moe=22)
            self.assertEqual(code, 0)
            server = captured["server"]
            self.assertIn("--n-cpu-moe", server.args)
            self.assertIn("22", server.args)
            self.assertTrue(captured["inherit_stdio"])

    def test_missing_server_yaml_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_launch_model_dir(Path(tmp))
            (model_dir / "server.yaml").unlink()
            with _capture_output() as (_stdout, stderr):
                code = main([str(model_dir)])
            self.assertEqual(code, 1)
            self.assertIn("server.yaml", stderr.getvalue())


class TestLaunchServerMain(unittest.TestCase):
    def test_main_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_launch_model_dir(Path(tmp))
            with patch("launch_server.run_launch_server", return_value=0) as run:
                code = main([str(model_dir)])
            self.assertEqual(code, 0)
            run.assert_called_once_with(model_dir, n_cpu_moe=None)

    def test_main_n_cpu_moe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_launch_model_dir(Path(tmp))
            with patch("launch_server.run_launch_server", return_value=0) as run:
                main([str(model_dir), "--n-cpu-moe", "22"])
            run.assert_called_once_with(model_dir, n_cpu_moe=22)

    def test_main_missing_dir(self) -> None:
        with _capture_output() as (_stdout, stderr):
            code = main(["models/missing-model"])
        self.assertEqual(code, 1)
        self.assertIn("Model directory not found", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
