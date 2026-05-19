"""Unit tests for server config args block scalar parsing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import (  # noqa: E402
    load_server_config,
    parse_args_block,
    parse_context_from_args,
    parse_port_from_args,
)


class TestParseArgsBlock(unittest.TestCase):
    def test_space_separated_lines(self) -> None:
        text = (
            "--host 127.0.0.1\n"
            "--port 8080\n"
            "-c 16384\n"
        )
        self.assertEqual(
            parse_args_block(text),
            ["--host", "127.0.0.1", "--port", "8080", "-c", "16384"],
        )

    def test_empty_block(self) -> None:
        self.assertEqual(parse_args_block(""), [])
        self.assertEqual(parse_args_block("   \n  \n"), [])

    def test_comments_and_blank_lines_ignored(self) -> None:
        text = (
            "# llama-server flags\n"
            "\n"
            "--host 127.0.0.1\n"
            "  \n"
            "# context\n"
            "-c 4096\n"
        )
        self.assertEqual(
            parse_args_block(text),
            ["--host", "127.0.0.1", "-c", "4096"],
        )

    def test_flag_without_value_stays_one_token(self) -> None:
        text = "--host 127.0.0.1\n--no-mmap\n"
        self.assertEqual(
            parse_args_block(text),
            ["--host", "127.0.0.1", "--no-mmap"],
        )

    def test_equals_form_raises(self) -> None:
        for line in ("--host=127.0.0.1", "-c=4096", "--port=8080"):
            with self.subTest(line=line):
                with self.assertRaises(ValueError) as ctx:
                    parse_args_block(line)
                msg = str(ctx.exception)
                self.assertTrue(
                    "space-separated" in msg or "flag=value" in msg,
                    msg,
                )


class TestLoadServerConfigArgs(unittest.TestCase):
    def _write_server_yaml(self, tmp: str, body: str) -> tuple[Path, str]:
        model_path = Path(tmp) / "model.gguf"
        model_path.write_bytes(b"gguf")
        yaml_path = Path(tmp) / "server.yaml"
        yaml_path.write_text(body, encoding="utf-8")
        return yaml_path, str(model_path)

    def test_block_scalar_parses_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path, model = self._write_server_yaml(
                tmp,
                "args: |\n"
                "  --host 127.0.0.1\n"
                "  --port 9090\n"
                "  -c 8192\n",
            )
            cfg = load_server_config(yaml_path, model=model)
            self.assertEqual(
                cfg.args,
                ["--host", "127.0.0.1", "--port", "9090", "-c", "8192"],
            )

    def test_missing_args_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path, model = self._write_server_yaml(tmp, "{}\n")
            cfg = load_server_config(yaml_path, model=model)
            self.assertEqual(cfg.args, [])

    def test_empty_block_scalar_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path, model = self._write_server_yaml(tmp, "args: |\n")
            cfg = load_server_config(yaml_path, model=model)
            self.assertEqual(cfg.args, [])

    def test_yaml_list_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path, model = self._write_server_yaml(
                tmp,
                "args:\n  - --host 127.0.0.1\n",
            )
            with self.assertRaises(ValueError) as ctx:
                load_server_config(yaml_path, model=model)
            self.assertIn("multiline string (YAML block scalar)", str(ctx.exception))

    def test_equals_form_in_block_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path, model = self._write_server_yaml(
                tmp,
                "args: |\n  --host=127.0.0.1\n",
            )
            with self.assertRaises(ValueError) as ctx:
                load_server_config(yaml_path, model=model)
            msg = str(ctx.exception)
            self.assertTrue(
                "space-separated" in msg or "flag=value" in msg,
                msg,
            )

    def test_model_flag_in_block_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path, model = self._write_server_yaml(
                tmp,
                "args: |\n  -m\n  /other/model.gguf\n",
            )
            with self.assertRaises(ValueError) as ctx:
                load_server_config(yaml_path, model=model)
            self.assertIn("must not contain -m or --model", str(ctx.exception))

    def test_double_dash_model_in_block_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path, model = self._write_server_yaml(
                tmp,
                "args: |\n  --model /other/model.gguf\n",
            )
            with self.assertRaises(ValueError) as ctx:
                load_server_config(yaml_path, model=model)
            self.assertIn("must not contain -m or --model", str(ctx.exception))

    def test_model_key_in_server_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.gguf"
            model_path.write_bytes(b"gguf")
            yaml_path = Path(tmp) / "server.yaml"
            yaml_path.write_text(
                f"model: {model_path}\nargs: |\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_server_config(yaml_path, model=str(model_path))
            self.assertIn("must not contain 'model'", str(ctx.exception))
            self.assertIn("model.yaml", str(ctx.exception))

    def test_parsed_args_work_with_context_and_port_parsers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path, model = self._write_server_yaml(
                tmp,
                "args: |\n"
                "  --host 127.0.0.1\n"
                "  --port 7070\n"
                "  -c 4096\n",
            )
            cfg = load_server_config(yaml_path, model=model)
            self.assertEqual(parse_context_from_args(cfg.args), 4096)
            self.assertEqual(parse_port_from_args(cfg.args), 7070)


if __name__ == "__main__":
    unittest.main()
