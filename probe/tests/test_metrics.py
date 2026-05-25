"""Unit tests for metrics formatting helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from metrics import (  # noqa: E402
    format_headline_summary,
    format_metrics_summary,
    format_probe_stdout,
)


def _sample_metrics() -> dict[str, float | int | None]:
    return {
        "server_ready_s": 1.5,
        "wall_time_s": 9.14,
        "ttft_s": 0.42,
        "completion_time_s": 8.72,
        "prompt_tokens": 128,
        "completion_tokens": 256,
        "total_tokens": 384,
        "prefill_tok_s": 304.76,
        "decode_tok_s": 116.32,
        "tokens_per_second": 28.01,
        "idle_vram_mb": 8000,
        "idle_system_ram_mb": 12000,
        "peak_vram_mb": 8961,
    }


class TestFormatMetricsSummary(unittest.TestCase):
    def test_includes_completion_time_total_tokens_tokens_per_second(self) -> None:
        text = format_metrics_summary(_sample_metrics())
        lines = text.splitlines()
        keys = [line.split()[0] for line in lines if line.strip()]
        self.assertIn("completion_time_s", keys)
        self.assertIn("total_tokens", keys)
        self.assertIn("tokens_per_second", keys)

        ttft_idx = keys.index("ttft_s")
        self.assertEqual(keys[ttft_idx + 1], "completion_time_s")

        completion_idx = keys.index("completion_tokens")
        self.assertEqual(keys[completion_idx + 1], "total_tokens")

        decode_idx = keys.index("decode_tok_s")
        self.assertEqual(keys[decode_idx + 1], "tokens_per_second")
        self.assertEqual(keys[decode_idx + 2], "idle_vram_mb")
        self.assertEqual(keys[decode_idx + 3], "idle_system_ram_mb")
        self.assertEqual(keys[decode_idx + 4], "peak_vram_mb")
        self.assertEqual(keys[decode_idx + 5], "model_max_context")

    def test_missing_values_render_as_na(self) -> None:
        text = format_metrics_summary({"wall_time_s": 1.0})
        self.assertRegex(text, r"completion_time_s\s+n/a")
        self.assertRegex(text, r"total_tokens\s+n/a")
        self.assertRegex(text, r"tokens_per_second\s+n/a")
        self.assertRegex(text, r"idle_system_ram_mb\s+n/a")


class TestFormatHeadlineSummary(unittest.TestCase):
    def test_formats_known_inputs(self) -> None:
        text = format_headline_summary(_sample_metrics())
        self.assertIn("End-to-end time         9.14 s", text)
        self.assertIn("Peak VRAM               8.75 GiB", text)
        self.assertIn("Generation throughput   116.32 tok/s", text)

    def test_missing_values_render_as_na(self) -> None:
        text = format_headline_summary({})
        self.assertIn("End-to-end time         n/a", text)
        self.assertIn("Peak VRAM               n/a", text)
        self.assertIn("Generation throughput   n/a", text)


class TestFormatProbeStdout(unittest.TestCase):
    def test_joins_blocks_with_blank_line(self) -> None:
        text = format_probe_stdout(_sample_metrics())
        metrics_block = format_metrics_summary(_sample_metrics())
        headline_block = format_headline_summary(_sample_metrics())
        self.assertEqual(text, metrics_block + "\n\n" + headline_block)


if __name__ == "__main__":
    unittest.main()
