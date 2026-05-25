"""Unit tests for parse_context_from_args."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import parse_context_from_args  # noqa: E402


class TestParseContextFromArgs(unittest.TestCase):
    def test_c_space_value(self) -> None:
        self.assertEqual(parse_context_from_args(["-c", "4096"]), 4096)

    def test_c_equals_value(self) -> None:
        self.assertEqual(parse_context_from_args(["-c=8192"]), 8192)

    def test_ctx_size_space_value(self) -> None:
        self.assertEqual(parse_context_from_args(["--ctx-size", "2048"]), 2048)

    def test_ctx_size_equals_value(self) -> None:
        self.assertEqual(parse_context_from_args(["--ctx-size=16384"]), 16384)

    def test_missing_value_after_c_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_context_from_args(["-c"])
        self.assertIn("requires a value", str(ctx.exception))

    def test_invalid_value_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_context_from_args(["-c", "nope"])
        self.assertIn("invalid context value", str(ctx.exception))

    def test_non_positive_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_context_from_args(["-c", "0"])
        self.assertIn("must be positive", str(ctx.exception))

    def test_missing_c_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_context_from_args(["--host", "127.0.0.1"])
        self.assertIn("required but not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
