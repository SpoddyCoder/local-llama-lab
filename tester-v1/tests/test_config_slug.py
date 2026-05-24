"""Unit tests for path-derived config slugs and model_slug."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import (  # noqa: E402
    ServerConfig,
    config_dir_metadata,
    load_server_config,
    slug_from_config_dir,
)
from paths import MODELS_ROOT, TESTER_ROOT  # noqa: E402


class TestSlugFromConfigDir(unittest.TestCase):
    def test_under_models(self) -> None:
        config_dir = MODELS_ROOT / "qwen3.5-9b-q8" / "hello-world-baseline"
        self.assertEqual(
            slug_from_config_dir(config_dir, TESTER_ROOT),
            "qwen3.5-9b-q8-hello-world-baseline",
        )

    def test_outside_models_raises(self) -> None:
        config_dir = Path("/tmp/my-run")
        with self.assertRaises(ValueError) as ctx:
            slug_from_config_dir(config_dir, TESTER_ROOT)
        self.assertIn("must be under", str(ctx.exception))


class TestConfigDirMetadata(unittest.TestCase):
    def test_under_models(self) -> None:
        config_dir = MODELS_ROOT / "qwen3.5-9b-q8" / "hello-world-baseline"
        self.assertEqual(
            config_dir_metadata(config_dir, TESTER_ROOT),
            {
                "config_path": "models/qwen3.5-9b-q8/hello-world-baseline/",
                "model": "qwen3.5-9b-q8",
                "variant": "hello-world-baseline",
            },
        )

    def test_none_config_dir(self) -> None:
        self.assertEqual(
            config_dir_metadata(None, TESTER_ROOT),
            {
                "config_path": None,
                "model": None,
                "variant": None,
            },
        )

    def test_outside_models_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            config_dir_metadata(Path("/tmp/my-run"), TESTER_ROOT)
        self.assertIn("must be under", str(ctx.exception))


class TestServerConfigSlug(unittest.TestCase):
    def test_model_slug_uses_gguf_stem(self) -> None:
        server = ServerConfig(
            model="/home/user/models/Qwen_Qwen3.5-9B-Q8_0.gguf",
            args=[],
        )
        self.assertEqual(server.model_slug, "Qwen_Qwen3.5-9B-Q8_0")


class TestLoadServerConfigRejectsLabel(unittest.TestCase):
    def test_label_in_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.gguf"
            model_path.write_bytes(b"gguf")
            yaml_path = Path(tmp) / "server.yaml"
            yaml_path.write_text(
                "label: old-label\n"
                "args: |\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_server_config(yaml_path, model=str(model_path))
            self.assertIn("'label' is removed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
