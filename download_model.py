#!/usr/bin/env python3
"""download_model CLI: download GGUF from Hugging Face using model.yaml hf-download metadata."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "tester-v1" / "src"
sys.path.insert(0, str(_SRC))

from hf_download_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
