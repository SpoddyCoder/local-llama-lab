"""Unit tests for config path resolution in runner."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import MODEL_YAML  # noqa: E402
from runner import (  # noqa: E402
    _TESTER_ROOT,
    resolve_config_paths,
)


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

            server, client, model_yaml = resolve_config_paths(
                variant_dir, None, None, None, _TESTER_ROOT
            )

            self.assertEqual(server, variant_dir / "server.yaml")
            self.assertEqual(client, variant_dir / "client.yaml")
            self.assertEqual(model_yaml, model_dir / MODEL_YAML)

    def test_missing_config_dir_raises(self) -> None:
        missing = Path("/nonexistent/config/dir/for/tests")
        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_config_paths(missing, None, None, None, _TESTER_ROOT)
        self.assertIn("Config directory not found", str(ctx.exception))
        self.assertIn(str(missing), str(ctx.exception))

    def test_missing_server_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, variant_dir = _variant_layout(tmp)
            (variant_dir / "server.yaml").unlink()

            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_config_paths(variant_dir, None, None, None, _TESTER_ROOT)
            self.assertIn("server.yaml", str(ctx.exception))
            self.assertIn(str(variant_dir), str(ctx.exception))

    def test_missing_client_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, variant_dir = _variant_layout(tmp)
            (variant_dir / "client.yaml").unlink()

            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_config_paths(variant_dir, None, None, None, _TESTER_ROOT)
            self.assertIn("client.yaml", str(ctx.exception))
            self.assertIn(str(variant_dir), str(ctx.exception))

    def test_missing_model_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, variant_dir = _variant_layout(tmp, with_model_yaml=False)

            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_config_paths(variant_dir, None, None, None, _TESTER_ROOT)
            self.assertIn(MODEL_YAML, str(ctx.exception))
            self.assertIn(str(variant_dir), str(ctx.exception))

    def test_overrides_with_config_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir, variant_dir = _variant_layout(tmp)

            with tempfile.TemporaryDirectory() as override_tmp:
                override_dir = Path(override_tmp)
                custom_server = override_dir / "custom-server.yaml"
                custom_client = override_dir / "custom-client.yaml"
                custom_server.write_text("server: {}\n")
                custom_client.write_text("client: {}\n")

                server, client, model_yaml = resolve_config_paths(
                    variant_dir,
                    custom_server,
                    custom_client,
                    None,
                    _TESTER_ROOT,
                )

                self.assertEqual(server, custom_server)
                self.assertEqual(client, custom_client)
                self.assertEqual(model_yaml, model_dir / MODEL_YAML)

    def test_no_config_dir_without_overrides_raises(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_config_paths(None, None, None, None, _TESTER_ROOT)
        self.assertIn("config_dir is required", str(ctx.exception))

    def test_no_config_dir_partial_override_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            custom_server = Path(tmp) / "s.yaml"
            custom_server.write_text("server: {}\n")
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_config_paths(None, custom_server, None, None, _TESTER_ROOT)
            self.assertIn("--client", str(ctx.exception))

    def test_no_config_dir_without_model_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            override_dir = Path(tmp)
            custom_server = override_dir / "s.yaml"
            custom_client = override_dir / "c.yaml"
            custom_server.write_text("server: {}\n")
            custom_client.write_text("client: {}\n")

            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_config_paths(
                    None, custom_server, custom_client, None, _TESTER_ROOT
                )
            self.assertIn("--model-yaml", str(ctx.exception))

    def test_no_config_dir_respects_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            override_dir = Path(tmp)
            custom_server = override_dir / "s.yaml"
            custom_client = override_dir / "c.yaml"
            custom_model_yaml = override_dir / MODEL_YAML
            custom_server.write_text("server: {}\n")
            custom_client.write_text("client: {}\n")
            custom_model_yaml.write_text("model: /tmp/fake.gguf\n")

            server, client, model_yaml = resolve_config_paths(
                None,
                custom_server,
                custom_client,
                custom_model_yaml,
                _TESTER_ROOT,
            )

            self.assertEqual(server, custom_server)
            self.assertEqual(client, custom_client)
            self.assertEqual(model_yaml, custom_model_yaml)


if __name__ == "__main__":
    unittest.main()
