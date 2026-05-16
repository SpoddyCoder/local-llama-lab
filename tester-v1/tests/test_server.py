"""Unit tests for server ready timing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import ServerConfig  # noqa: E402
from server import ServerProcess  # noqa: E402


class TestServerReadyS(unittest.TestCase):
    def test_records_time_until_health_ok(self) -> None:
        server = ServerConfig(
            model="/tmp/model.gguf",
            args=[],
            ready_timeout_s=10.0,
            ready_poll_interval_s=0.01,
        )
        proc = ServerProcess(server, "http://127.0.0.1:8080")
        mock_popen = MagicMock()
        mock_popen.return_value.stdout = MagicMock()
        mock_popen.return_value.stderr = MagicMock()
        mock_popen.return_value.poll.return_value = None

        with (
            patch("server.subprocess.Popen", mock_popen),
            patch("server._stream_lines"),
            patch("server._probe_health", side_effect=[None, "/health"]),
        ):
            proc.start()
            proc.wait_ready()

        self.assertIsNotNone(proc.server_ready_s)
        self.assertGreaterEqual(proc.server_ready_s, 0.0)
        self.assertEqual(proc.ready_endpoint, "/health")


if __name__ == "__main__":
    unittest.main()
