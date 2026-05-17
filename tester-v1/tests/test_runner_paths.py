"""Unit tests for config path resolution in runner."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from runner import (  # noqa: E402
    _TESTER_ROOT,
    resolve_config_paths,
)


class TestResolveConfigPaths(unittest.TestCase):
    def test_valid_config_dir_resolves_both_yamls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "server.yaml").write_text("server: {}\n")
            (config_dir / "client.yaml").write_text("client: {}\n")

            server, client = resolve_config_paths(config_dir, None, None, _TESTER_ROOT)

            self.assertEqual(server, config_dir / "server.yaml")
            self.assertEqual(client, config_dir / "client.yaml")

    def test_missing_config_dir_raises(self) -> None:
        missing = Path("/nonexistent/config/dir/for/tests")
        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_config_paths(missing, None, None, _TESTER_ROOT)
        self.assertIn("Config directory not found", str(ctx.exception))
        self.assertIn(str(missing), str(ctx.exception))

    def test_missing_server_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "client.yaml").write_text("client: {}\n")

            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_config_paths(config_dir, None, None, _TESTER_ROOT)
            self.assertIn("server.yaml", str(ctx.exception))
            self.assertIn(str(config_dir), str(ctx.exception))

    def test_missing_client_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "server.yaml").write_text("server: {}\n")

            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_config_paths(config_dir, None, None, _TESTER_ROOT)
            self.assertIn("client.yaml", str(ctx.exception))
            self.assertIn(str(config_dir), str(ctx.exception))

    def test_overrides_with_config_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "server.yaml").write_text("server: {}\n")
            (config_dir / "client.yaml").write_text("client: {}\n")

            with tempfile.TemporaryDirectory() as override_tmp:
                override_dir = Path(override_tmp)
                custom_server = override_dir / "custom-server.yaml"
                custom_client = override_dir / "custom-client.yaml"
                custom_server.write_text("server: {}\n")
                custom_client.write_text("client: {}\n")

                server, client = resolve_config_paths(
                    config_dir,
                    custom_server,
                    custom_client,
                    _TESTER_ROOT,
                )

                self.assertEqual(server, custom_server)
                self.assertEqual(client, custom_client)

    def test_no_config_dir_without_overrides_raises(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            resolve_config_paths(None, None, None, _TESTER_ROOT)
        self.assertIn("config_dir is required", str(ctx.exception))

    def test_no_config_dir_partial_override_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            custom_server = Path(tmp) / "s.yaml"
            custom_server.write_text("server: {}\n")
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_config_paths(None, custom_server, None, _TESTER_ROOT)
            self.assertIn("--client", str(ctx.exception))

    def test_no_config_dir_respects_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            override_dir = Path(tmp)
            custom_server = override_dir / "s.yaml"
            custom_client = override_dir / "c.yaml"
            custom_server.write_text("server: {}\n")
            custom_client.write_text("client: {}\n")

            server, client = resolve_config_paths(
                None,
                custom_server,
                custom_client,
                _TESTER_ROOT,
            )

            self.assertEqual(server, custom_server)
            self.assertEqual(client, custom_client)


if __name__ == "__main__":
    unittest.main()
