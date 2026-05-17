#!/usr/bin/env python3
"""CLI: run tester-v1 Python unit tests (unittest discover)."""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run tester-v1 unit tests (unittest discover -s tests).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less output (omit unittest -v).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args, extra = _build_parser().parse_known_args(argv)
    os.chdir(_ROOT)

    discover_argv = ["unittest", "discover", "-s", "tests"]
    if not args.quiet:
        discover_argv.append("-v")
    discover_argv.extend(extra)

    program = unittest.main(module=None, argv=discover_argv, exit=False)
    return 0 if program.result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
