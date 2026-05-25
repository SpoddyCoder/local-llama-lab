"""Unit tests for run metadata slugs and model_slug."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import (  # noqa: E402
    ServerConfig,
    load_server_config,
    run_metadata,
    slug_from_run,
)


class TestSlugFromRun(unittest.TestCase):
    def test_flat_variant(self) -> None:
        self.assertEqual(
            slug_from_run("qwen3.5-9b-q8", "hello-world-baseline"),
            "qwen3.5-9b-q8-hello-world-baseline",
        )

    def test_sandbox_variant(self) -> None:
        self.assertEqual(
            slug_from_run("qwen3.5-9b-q8", "sandbox"),
            "qwen3.5-9b-q8-sandbox",
        )

    def test_nested_variant_normalizes_slashes(self) -> None:
        self.assertEqual(
            slug_from_run("qwen3.5-9b-q8", "foo/bar"),
            "qwen3.5-9b-q8-foo-bar",
        )


class TestRunMetadata(unittest.TestCase):
    def test_flat_variant(self) -> None:
        self.assertEqual(
            run_metadata("qwen3.5-9b-q8", "hello-world-baseline"),
            {
                "config_path": "models/qwen3.5-9b-q8/",
                "model": "qwen3.5-9b-q8",
                "variant": "hello-world-baseline",
            },
        )

    def test_sandbox_variant(self) -> None:
        self.assertEqual(
            run_metadata("qwen3.5-9b-q8", "sandbox"),
            {
                "config_path": "models/qwen3.5-9b-q8/",
                "model": "qwen3.5-9b-q8",
                "variant": "sandbox",
            },
        )

    def test_nested_variant(self) -> None:
        self.assertEqual(
            run_metadata("qwen3.5-9b-q8", "foo/bar"),
            {
                "config_path": "models/qwen3.5-9b-q8/",
                "model": "qwen3.5-9b-q8",
                "variant": "foo/bar",
            },
        )


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
