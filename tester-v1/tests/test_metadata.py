"""Unit tests for run metadata collection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import ServerConfig  # noqa: E402
from metadata import (  # noqa: E402
    collect_run_metadata,
    query_gpu_info,
    query_server_version,
)


class TestQueryServerVersion(unittest.TestCase):
    def test_parses_version_line(self) -> None:
        with patch("metadata.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = ""
            run.return_value.stderr = (
                "ggml_cuda_init: failed\n"
                "version: 9158 (3e037f313)\n"
                "built with GNU 13.3.0\n"
            )
            self.assertEqual(
                query_server_version("llama-server"),
                "version: 9158 (3e037f313)",
            )

    def test_falls_back_to_first_line(self) -> None:
        with patch("metadata.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "llama.cpp server 1.2.3\n"
            run.return_value.stderr = ""
            self.assertEqual(query_server_version("llama-server"), "llama.cpp server 1.2.3")

    def test_returns_none_on_failure(self) -> None:
        with patch("metadata.subprocess.run", side_effect=FileNotFoundError):
            self.assertIsNone(query_server_version("missing-binary"))


class TestQueryGpuInfo(unittest.TestCase):
    def test_parses_single_gpu(self) -> None:
        with patch("metadata.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "NVIDIA GeForce RTX 4090, 550.54.15\n"
            self.assertEqual(
                query_gpu_info(),
                ("NVIDIA GeForce RTX 4090", "550.54.15"),
            )

    def test_joins_multiple_gpu_names(self) -> None:
        with patch("metadata.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = (
                "NVIDIA GeForce RTX 4090, 550.54.15\n"
                "NVIDIA GeForce RTX 4090, 550.54.15\n"
            )
            self.assertEqual(
                query_gpu_info(),
                ("NVIDIA GeForce RTX 4090; NVIDIA GeForce RTX 4090", "550.54.15"),
            )

    def test_returns_none_on_failure(self) -> None:
        with patch("metadata.subprocess.run") as run:
            run.return_value.returncode = 1
            run.return_value.stdout = ""
            self.assertEqual(query_gpu_info(), (None, None))


class TestCollectRunMetadata(unittest.TestCase):
    def test_combines_server_and_gpu(self) -> None:
        server = ServerConfig(model="/path/model.gguf", args=[], binary="llama-server")
        with (
            patch("metadata.query_server_version", return_value="version: 1"),
            patch(
                "metadata.query_gpu_info",
                return_value=("GPU A", "535.00"),
            ),
        ):
            self.assertEqual(
                collect_run_metadata(server),
                {
                    "server_version": "version: 1",
                    "gpu_name": "GPU A",
                    "driver_version": "535.00",
                },
            )


if __name__ == "__main__":
    unittest.main()
