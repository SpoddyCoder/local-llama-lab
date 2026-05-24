"""Unit tests for model_layout run config resolution."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import (  # noqa: E402
    MODEL_YAML,
    load_model_config,
    parse_context_from_args,
    parse_n_cpu_moe_from_args,
)
from model_layout import RunConfig, resolve_model_server, resolve_run_config  # noqa: E402
from paths import MODELS_ROOT  # noqa: E402

_BASE_SERVER_ARGS = (
    "  --host 127.0.0.1\n"
    "  --port 8080\n"
    "  --parallel 1\n"
    "  --no-mmap\n"
    "  --fit off\n"
    "  --ctx-size 4096\n"
)


def _write_model_dir(
    root: Path,
    *,
    slug: str = "my-model",
    server_args: str = _BASE_SERVER_ARGS,
    include_server: bool = True,
    include_model_yaml: bool = True,
) -> Path:
    model_dir = root / slug
    model_dir.mkdir(parents=True)
    gguf = model_dir / "model.gguf"
    gguf.write_bytes(b"gguf")
    if include_model_yaml:
        (model_dir / MODEL_YAML).write_text(f"model: {gguf}\n", encoding="utf-8")
    if include_server:
        (model_dir / "server.yaml").write_text(
            f"args: |\n{server_args}",
            encoding="utf-8",
        )
    return model_dir


def _write_client(path: Path, *, messages: str = "- role: user\n  content: hi\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"messages:\n  {messages}", encoding="utf-8")


class TestResolveRunConfigCalibration(unittest.TestCase):
    def test_calibration_variant_uses_shared_probe_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            calib_root = tmp_path / "calibration-tests"
            model_dir = _write_model_dir(tmp_path)
            variant = "hello-world-baseline"
            _write_client(calib_root / variant / "client.yaml")

            with patch("model_layout.CALIBRATION_TESTS_ROOT", calib_root):
                cfg = resolve_run_config(model_dir, variant)

            self.assertIsInstance(cfg, RunConfig)
            self.assertEqual(cfg.model, "my-model")
            self.assertEqual(cfg.variant, variant)
            self.assertEqual(cfg.client_path, calib_root / variant / "client.yaml")
            self.assertEqual(cfg.model_yaml_path, model_dir / MODEL_YAML)
            self.assertEqual(parse_context_from_args(cfg.server.args), 4096)


class TestResolveRunConfigServerOverride(unittest.TestCase):
    def test_thin_override_merges_ctx_size_onto_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            calib_root = tmp_path / "calibration-tests"
            model_dir = _write_model_dir(tmp_path)
            variant = "calibration-ctx-probe"
            _write_client(calib_root / variant / "client.yaml")
            (calib_root / variant / "server.yaml").write_text(
                "args: |\n  --ctx-size 8192\n",
                encoding="utf-8",
            )

            with patch("model_layout.CALIBRATION_TESTS_ROOT", calib_root):
                cfg = resolve_run_config(model_dir, variant)

            self.assertEqual(parse_context_from_args(cfg.server.args), 8192)
            self.assertIn("--host", cfg.server.args)
            self.assertIn("127.0.0.1", cfg.server.args)
            self.assertIn("--port", cfg.server.args)
            self.assertIn("8080", cfg.server.args)


class TestResolveRunConfigModelLocal(unittest.TestCase):
    def test_sandbox_variant_resolves_from_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_dir = _write_model_dir(tmp_path)
            variant = "sandbox"
            _write_client(model_dir / variant / "client.yaml")

            cfg = resolve_run_config(model_dir, variant)

            self.assertEqual(cfg.client_path, model_dir / variant / "client.yaml")
            self.assertEqual(cfg.model, "my-model")
            self.assertEqual(cfg.variant, variant)

    def test_sandbox_server_yaml_stub_with_empty_args_uses_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_dir = _write_model_dir(tmp_path)
            variant = "sandbox"
            _write_client(model_dir / variant / "client.yaml")
            (model_dir / variant / "server.yaml").write_text(
                "# optional override\nargs: |\n  # --ctx-size 4096\n",
                encoding="utf-8",
            )

            cfg = resolve_run_config(model_dir, variant)

            self.assertEqual(parse_context_from_args(cfg.server.args), 4096)

    def test_sandbox_server_yaml_merges_ctx_onto_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_dir = _write_model_dir(tmp_path)
            variant = "sandbox"
            _write_client(model_dir / variant / "client.yaml")
            (model_dir / variant / "server.yaml").write_text(
                "args: |\n  --ctx-size 8192\n",
                encoding="utf-8",
            )

            cfg = resolve_run_config(model_dir, variant)

            self.assertEqual(parse_context_from_args(cfg.server.args), 8192)
            self.assertIn("--host", cfg.server.args)
            self.assertIn("127.0.0.1", cfg.server.args)
            self.assertIn("--port", cfg.server.args)
            self.assertIn("8080", cfg.server.args)


class TestResolveRunConfigNCpuMoe(unittest.TestCase):
    def test_cli_n_cpu_moe_wins_over_base_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            calib_root = tmp_path / "calibration-tests"
            server_args = (
                "  --host 127.0.0.1\n"
                "  -c 4096\n"
                "  --n-cpu-moe 24\n"
                "  --n-gpu-layers 999\n"
            )
            model_dir = _write_model_dir(tmp_path, server_args=server_args)
            variant = "hello-world-baseline"
            _write_client(calib_root / variant / "client.yaml")

            with patch("model_layout.CALIBRATION_TESTS_ROOT", calib_root):
                cfg = resolve_run_config(model_dir, variant, n_cpu_moe=22)

            self.assertEqual(parse_n_cpu_moe_from_args(cfg.server.args), 22)
            self.assertEqual(cfg.server.args.count("--n-cpu-moe"), 1)


class TestResolveRunConfigErrors(unittest.TestCase):
    def test_missing_variant_lists_search_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            calib_root = tmp_path / "calibration-tests"
            model_dir = _write_model_dir(tmp_path)
            variant = "missing-variant"

            with patch("model_layout.CALIBRATION_TESTS_ROOT", calib_root):
                with self.assertRaises(FileNotFoundError) as ctx:
                    resolve_run_config(model_dir, variant)

            msg = str(ctx.exception)
            self.assertIn(variant, msg)
            self.assertIn(str(calib_root / variant / "client.yaml"), msg)
            self.assertIn(str(model_dir / variant / "client.yaml"), msg)

    def test_missing_model_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_model_dir(Path(tmp), include_model_yaml=False)
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_run_config(model_dir, "hello-world-baseline")
            self.assertIn(MODEL_YAML, str(ctx.exception))

    def test_missing_server_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_model_dir(Path(tmp), include_server=False)
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_run_config(model_dir, "hello-world-baseline")
            self.assertIn("server.yaml", str(ctx.exception))


class TestLiveModelLayout(unittest.TestCase):
    _LIVE_SLUG = "qwen3.5-9b-q8"

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_dir = MODELS_ROOT / cls._LIVE_SLUG
        cls._skip_reason: str | None = None
        if not cls.model_dir.is_dir():
            cls._skip_reason = f"model directory not found: {cls.model_dir}"
            return
        model_yaml = cls.model_dir / MODEL_YAML
        server_yaml = cls.model_dir / "server.yaml"
        if not model_yaml.is_file() or not server_yaml.is_file():
            cls._skip_reason = "model.yaml or server.yaml missing at model root"
            return
        try:
            load_model_config(model_yaml)
        except ValueError as exc:
            cls._skip_reason = str(exc)

    def setUp(self) -> None:
        if self._skip_reason is not None:
            self.skipTest(self._skip_reason)

    def test_resolve_run_config_hello_world_baseline(self) -> None:
        cfg = resolve_run_config(self.model_dir, "hello-world-baseline")

        self.assertIsInstance(cfg, RunConfig)
        self.assertEqual(cfg.model, self._LIVE_SLUG)
        self.assertEqual(cfg.variant, "hello-world-baseline")
        self.assertEqual(cfg.model_yaml_path, self.model_dir / MODEL_YAML)
        self.assertEqual(parse_context_from_args(cfg.server.args), 4096)

    def test_resolve_model_server(self) -> None:
        server, model_yaml_path = resolve_model_server(self.model_dir)

        self.assertEqual(model_yaml_path, self.model_dir / MODEL_YAML)
        self.assertEqual(parse_context_from_args(server.args), 4096)


class TestResolveModelServer(unittest.TestCase):
    def test_loads_model_server_yaml_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_model_dir(Path(tmp))
            server, model_yaml_path = resolve_model_server(model_dir)

            self.assertEqual(model_yaml_path, model_dir / MODEL_YAML)
            self.assertEqual(parse_context_from_args(server.args), 4096)
            self.assertEqual(server.model, str(model_dir / "model.gguf"))

    def test_applies_n_cpu_moe_from_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server_args = (
                "  --host 127.0.0.1\n"
                "  -c 4096\n"
                "  --n-cpu-moe 24\n"
                "  --n-gpu-layers 999\n"
            )
            model_dir = _write_model_dir(Path(tmp), server_args=server_args)
            server, _ = resolve_model_server(model_dir, n_cpu_moe=22)
            self.assertEqual(parse_n_cpu_moe_from_args(server.args), 22)


if __name__ == "__main__":
    unittest.main()
