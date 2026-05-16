"""Unit tests for path-derived config slugs and ServerConfig.run_slug."""

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
from runner import _TESTER_ROOT  # noqa: E402


class TestSlugFromConfigDir(unittest.TestCase):
    def test_under_test_configs(self) -> None:
        config_dir = (
            _TESTER_ROOT / "test-configs" / "test1" / "qwen3.5-9b-q8" / "baseline"
        )
        self.assertEqual(
            slug_from_config_dir(config_dir, _TESTER_ROOT),
            "test1-qwen3.5-9b-q8-baseline",
        )

    def test_outside_test_configs(self) -> None:
        config_dir = Path("/tmp/my-run")
        self.assertEqual(
            slug_from_config_dir(config_dir, _TESTER_ROOT),
            "tmp-my-run",
        )


class TestConfigDirMetadata(unittest.TestCase):
    def test_under_test_configs(self) -> None:
        config_dir = (
            _TESTER_ROOT / "test-configs" / "test1" / "qwen3.5-9b-q8" / "baseline"
        )
        self.assertEqual(
            config_dir_metadata(config_dir, _TESTER_ROOT),
            {
                "test_config_path": "test1/qwen3.5-9b-q8/baseline/",
                "test_name": "test1",
                "model": "qwen3.5-9b-q8",
                "test_config": "baseline",
            },
        )

    def test_none_config_dir(self) -> None:
        self.assertEqual(
            config_dir_metadata(None, _TESTER_ROOT),
            {
                "test_config_path": None,
                "test_name": None,
                "model": None,
                "test_config": None,
            },
        )

    def test_outside_test_configs(self) -> None:
        self.assertEqual(
            config_dir_metadata(Path("/tmp/my-run"), _TESTER_ROOT),
            {
                "test_config_path": None,
                "test_name": None,
                "model": None,
                "test_config": None,
            },
        )


class TestServerConfigSlug(unittest.TestCase):
    def test_without_run_slug_uses_model_stem(self) -> None:
        server = ServerConfig(
            model="/home/user/models/Qwen_Qwen3.5-9B-Q8_0.gguf",
            args=[],
        )
        self.assertEqual(server.model_slug, "Qwen_Qwen3.5-9B-Q8_0")

    def test_run_slug_drives_model_slug(self) -> None:
        server = ServerConfig(
            model="/home/user/models/Qwen_Qwen3.5-9B-Q8_0.gguf",
            args=[],
            run_slug="test1-qwen3.5-9b-q8-baseline",
        )
        self.assertEqual(server.model_slug, "test1-qwen3.5-9b-q8-baseline")


class TestLoadServerConfigRejectsLabel(unittest.TestCase):
    def test_label_in_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.gguf"
            model_path.write_bytes(b"gguf")
            yaml_path = Path(tmp) / "server.yaml"
            yaml_path.write_text(
                "label: old-label\n"
                f"model: {model_path}\n"
                "args: []\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_server_config(yaml_path)
            self.assertIn("'label' is removed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
