#!/usr/bin/env python3
"""CLI: run footprint, ctx-probe, and hello-world-baseline; print calibration summary.

Default: probe stdout only (no JSON). Pass --save-result to write probe JSON under results/.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(_SRC))

from calibration_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
