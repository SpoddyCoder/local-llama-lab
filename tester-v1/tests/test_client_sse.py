"""Unit tests for SSE parsing and chunk extraction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from client import (  # noqa: E402
    _SSE_DONE,
    extract_content_delta,
    extract_usage,
    parse_sse_line,
    parse_sse_stream,
)


class TestParseSseLine(unittest.TestCase):
    def test_content_chunk(self) -> None:
        line = 'data: {"choices":[{"delta":{"content":"Hi"}}]}'
        chunk = parse_sse_line(line)
        assert chunk is not None
        assert chunk != _SSE_DONE
        self.assertEqual(extract_content_delta(chunk), "Hi")

    def test_done_sentinel(self) -> None:
        self.assertEqual(parse_sse_line("data: [DONE]"), _SSE_DONE)

    def test_empty_and_comments_ignored(self) -> None:
        self.assertIsNone(parse_sse_line(""))
        self.assertIsNone(parse_sse_line(": keep-alive"))

    def test_usage_chunk(self) -> None:
        line = (
            'data: {"choices":[],"usage":'
            '{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}'
        )
        chunk = parse_sse_line(line)
        assert chunk is not None
        self.assertEqual(extract_usage(chunk), (10, 5, 15))

    def test_reasoning_content_delta(self) -> None:
        line = 'data: {"choices":[{"delta":{"reasoning_content":"Hi"}}]}'
        chunk = parse_sse_line(line)
        assert chunk is not None
        self.assertEqual(extract_content_delta(chunk), "Hi")

    def test_timings_chunk(self) -> None:
        line = (
            'data: {"choices":[{"finish_reason":"length","delta":{}}],'
            '"timings":{"prompt_n":24,"predicted_n":16}}'
        )
        chunk = parse_sse_line(line)
        assert chunk is not None
        self.assertEqual(extract_usage(chunk), (24, 16, 40))

    def test_malformed_json_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_sse_line("data: {not-json")

    def test_non_data_line_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_sse_line("event: message")


class TestParseSseStream(unittest.TestCase):
    def test_multiple_chunks_then_done(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"content":"A"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"B"}}]}',
            "data: [DONE]",
        ]
        events = list(parse_sse_stream(iter(lines)))
        self.assertEqual(len(events), 3)
        self.assertEqual(extract_content_delta(events[0]), "A")
        self.assertEqual(extract_content_delta(events[1]), "B")
        self.assertEqual(events[2], _SSE_DONE)


if __name__ == "__main__":
    unittest.main()
