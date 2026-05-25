#!/usr/bin/env python3
"""CLI: single llama-server run — start, complete once, optional result JSON."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "probe" / "src"
sys.path.insert(0, str(_SRC))

from runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
