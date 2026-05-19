"""Unit tests for model.yaml loading."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import (  # noqa: E402
    MODEL_YAML,
    load_model_config,
    load_variant_server,
    model_yaml_path_for_variant,
)


class TestModelYamlPathForVariant(unittest.TestCase):
    def test_returns_parent_model_yaml(self) -> None:
        variant_dir = Path("/tmp/configs/my-model/hello-world-baseline")
        self.assertEqual(
            model_yaml_path_for_variant(variant_dir),
            Path("/tmp/configs/my-model") / MODEL_YAML,
        )


class TestLoadModelConfig(unittest.TestCase):
    def test_loads_and_expands_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.gguf"
            model_path.write_bytes(b"gguf")
            yaml_path = Path(tmp) / MODEL_YAML
            yaml_path.write_text(f"model: {model_path}\n", encoding="utf-8")
            self.assertEqual(load_model_config(yaml_path), str(model_path))

    def test_missing_model_key_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / MODEL_YAML
            yaml_path.write_text("other: value\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_model_config(yaml_path)
            self.assertIn("'model' is required", str(ctx.exception))

    def test_missing_yaml_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / MODEL_YAML
            with self.assertRaises(ValueError) as ctx:
                load_model_config(yaml_path)
            self.assertIn("config file not found", str(ctx.exception))

    def test_non_mapping_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / MODEL_YAML
            yaml_path.write_text("- not a mapping\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_model_config(yaml_path)
            self.assertIn("must be a YAML mapping", str(ctx.exception))

    def test_non_string_model_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / MODEL_YAML
            yaml_path.write_text("model: 123\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_model_config(yaml_path)
            self.assertIn("'model' is required", str(ctx.exception))

    def test_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / MODEL_YAML
            yaml_path.write_text(
                f"model: {Path(tmp) / 'missing.gguf'}\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_model_config(yaml_path)
            self.assertIn("does not exist", str(ctx.exception))


class TestLoadVariantServer(unittest.TestCase):
    def test_loads_server_with_model_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.gguf"
            model_path.write_bytes(b"gguf")
            (root / MODEL_YAML).write_text(f"model: {model_path}\n", encoding="utf-8")
            server_path = root / "hello-world-baseline" / "server.yaml"
            server_path.parent.mkdir()
            server_path.write_text(
                "args: |\n  --host 127.0.0.1\n  -c 4096\n",
                encoding="utf-8",
            )
            cfg = load_variant_server(server_path, root / MODEL_YAML)
            self.assertEqual(cfg.model, str(model_path))
            self.assertEqual(cfg.args, ["--host", "127.0.0.1", "-c", "4096"])


if __name__ == "__main__":
    unittest.main()
