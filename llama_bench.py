#!/usr/bin/env python3
"""CLI: run llama-bench from a model config directory."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "probe" / "src"
sys.path.insert(0, str(_SRC))

from llama_bench_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
