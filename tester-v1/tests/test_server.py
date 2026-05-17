"""Unit tests for server ready timing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import ServerConfig  # noqa: E402
from server import ServerProcess, fetch_model_max_context  # noqa: E402


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


class TestFetchModelMaxContext(unittest.TestCase):
    def test_parses_n_ctx_train(self) -> None:
        client = MagicMock()
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "data": [{"meta": {"n_ctx_train": 128000}}],
        }
        client.get.return_value = response
        self.assertEqual(
            fetch_model_max_context(client, "http://127.0.0.1:8080"),
            128000,
        )
        client.get.assert_called_once_with("http://127.0.0.1:8080/v1/models")

    def test_returns_none_when_meta_missing(self) -> None:
        client = MagicMock()
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"data": [{"meta": {}}]}
        client.get.return_value = response
        self.assertIsNone(fetch_model_max_context(client, "http://127.0.0.1:8080"))

    def test_returns_none_on_http_error(self) -> None:
        import httpx

        client = MagicMock()
        client.get.side_effect = httpx.HTTPError("down")
        self.assertIsNone(fetch_model_max_context(client, "http://127.0.0.1:8080"))

    def test_returns_none_on_non_200(self) -> None:
        client = MagicMock()
        response = MagicMock()
        response.status_code = 503
        client.get.return_value = response
        self.assertIsNone(fetch_model_max_context(client, "http://127.0.0.1:8080"))


if __name__ == "__main__":
    unittest.main()
