"""Unit tests for llama_bench CLI."""

from __future__ import annotations

import io
import shlex
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import MODEL_YAML  # noqa: E402
from llama_bench_cli import main, run_llama_bench  # noqa: E402
from llama_bench_config import build_llama_bench_argv, resolve_llama_bench  # noqa: E402


def _write_llama_bench_model_dir(root: Path, *, slug: str = "bench-model") -> Path:
    model_dir = root / slug
    model_dir.mkdir(parents=True)
    gguf = model_dir / "model.gguf"
    gguf.write_bytes(b"gguf")
    (model_dir / MODEL_YAML).write_text(f"model: {gguf}\n", encoding="utf-8")
    (model_dir / "llama_bench.yaml").write_text(
        "binary: llama-bench\n"
        "args: |\n"
        "  -mmp 0\n"
        "  -p 512\n"
        "  -n 128\n",
        encoding="utf-8",
    )
    return model_dir


@contextmanager
def _capture_output():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
        yield stdout, stderr


class TestLlamaBenchCli(unittest.TestCase):
    def test_resolves_argv_injects_model_from_model_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_llama_bench_model_dir(Path(tmp))
            config = resolve_llama_bench(model_dir)
            argv = build_llama_bench_argv(config)
            self.assertEqual(argv[0], "llama-bench")
            self.assertEqual(argv[-2], "-m")
            self.assertEqual(argv[-1], str(model_dir / "model.gguf"))

    def test_missing_llama_bench_yaml_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_llama_bench_model_dir(Path(tmp))
            (model_dir / "llama_bench.yaml").unlink()
            with _capture_output() as (_stdout, stderr):
                code = main([str(model_dir)])
            self.assertEqual(code, 1)
            self.assertIn("llama_bench.yaml", stderr.getvalue())

    def test_n_cpu_moe_augmented_argv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_llama_bench_model_dir(Path(tmp))
            with _capture_output() as (stdout, _stderr):
                with patch("llama_bench_cli.subprocess.run") as run:
                    code = main([str(model_dir), "--n-cpu-moe", "22", "--dry-run"])
            self.assertEqual(code, 0)
            run.assert_not_called()
            argv = shlex.split(stdout.getvalue().strip())
            self.assertIn("--n-cpu-moe", argv)
            self.assertIn("22", argv)
            self.assertIn("--n-gpu-layers", argv)
            self.assertIn("999", argv)

    def test_dry_run_prints_command_without_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_llama_bench_model_dir(Path(tmp))
            with _capture_output() as (stdout, _stderr):
                with patch("llama_bench_cli.subprocess.run") as run:
                    code = run_llama_bench(model_dir, dry_run=True)
            self.assertEqual(code, 0)
            run.assert_not_called()
            config = resolve_llama_bench(model_dir)
            expected = shlex.join(build_llama_bench_argv(config))
            self.assertEqual(stdout.getvalue().strip(), expected)

    def test_forwards_subprocess_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_llama_bench_model_dir(Path(tmp))
            with patch("llama_bench_cli.subprocess.run") as run:
                run.return_value = MagicMock(returncode=7)
                code = run_llama_bench(model_dir)
            self.assertEqual(code, 7)
            run.assert_called_once()
            argv = run.call_args[0][0]
            self.assertEqual(argv[-2], "-m")
            self.assertEqual(argv[-1], str(model_dir / "model.gguf"))


if __name__ == "__main__":
    unittest.main()
