"""Unit tests for config path resolution in runner."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import MODEL_YAML  # noqa: E402
from runner import resolve_config_paths  # noqa: E402


def _variant_layout(tmp: str, *, with_model_yaml: bool = True) -> tuple[Path, Path]:
    """Create model_dir/variant/{server,client}.yaml and optional model.yaml."""
    model_dir = Path(tmp)
    variant_dir = model_dir / "hello-world-baseline"
    variant_dir.mkdir(parents=True)
    (variant_dir / "server.yaml").write_text("server: {}\n")
    (variant_dir / "client.yaml").write_text("client: {}\n")
    if with_model_yaml:
        (model_dir / MODEL_YAML).write_text("model: /tmp/fake.gguf\n")
    return model_dir, variant_dir


class TestResolveConfigPaths(unittest.TestCase):
    def test_valid_config_dir_resolves_yamls_and_model_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, variant_dir = _variant_layout(tmp)

            server, client, model_yaml = resolve_config_paths(variant_dir)

            self.assertEqual(server, variant_dir / "server.yaml")
            self.assertEqual(client, variant_dir / "client.yaml")
            self.assertEqual(model_yaml, model_dir / MODEL_YAML)

    def test_missing_config_dir_raises(self) -> None:
        missing = Path("/nonexistent/config/dir/for/tests")
        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_config_paths(missing)
        self.assertIn("Config directory not found", str(ctx.exception))
        self.assertIn(str(missing), str(ctx.exception))

    def test_missing_server_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _model_dir, variant_dir = _variant_layout(tmp)
            (variant_dir / "server.yaml").unlink()

            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_config_paths(variant_dir)
            self.assertIn("server.yaml", str(ctx.exception))
            self.assertIn(str(variant_dir), str(ctx.exception))

    def test_missing_client_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _model_dir, variant_dir = _variant_layout(tmp)
            (variant_dir / "client.yaml").unlink()

            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_config_paths(variant_dir)
            self.assertIn("client.yaml", str(ctx.exception))
            self.assertIn(str(variant_dir), str(ctx.exception))

    def test_missing_model_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _model_dir, variant_dir = _variant_layout(tmp, with_model_yaml=False)

            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_config_paths(variant_dir)
            self.assertIn(MODEL_YAML, str(ctx.exception))
            self.assertIn(str(variant_dir), str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
