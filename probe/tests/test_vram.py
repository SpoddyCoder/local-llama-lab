"""Unit tests for VRAM sampling and polling."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from vram import VramPoller, query_gpu_total_mb, sample_vram_mb  # noqa: E402


class TestSampleVramMb(unittest.TestCase):
    def test_parses_single_gpu(self) -> None:
        with patch("vram.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "  8192\n"
            self.assertEqual(sample_vram_mb(), 8192)

    def test_parses_max_across_gpus(self) -> None:
        with patch("vram.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "4096\n12288\n"
            self.assertEqual(sample_vram_mb(), 12288)

    def test_returns_none_on_failure(self) -> None:
        with patch("vram.subprocess.run") as run:
            run.return_value.returncode = 1
            run.return_value.stdout = ""
            self.assertIsNone(sample_vram_mb())

    def test_returns_none_on_missing_binary(self) -> None:
        with patch("vram.subprocess.run", side_effect=FileNotFoundError):
            self.assertIsNone(sample_vram_mb())


class TestQueryGpuTotalMb(unittest.TestCase):
    def test_parses_single_gpu(self) -> None:
        with patch("vram.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "  24576\n"
            self.assertEqual(query_gpu_total_mb(), 24576)
            run.assert_called_once()
            query = run.call_args.args[0]
            self.assertTrue(any("memory.total" in part for part in query))

    def test_parses_max_across_gpus(self) -> None:
        with patch("vram.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "8192\n16384\n"
            self.assertEqual(query_gpu_total_mb(), 16384)

    def test_returns_none_on_failure(self) -> None:
        with patch("vram.subprocess.run") as run:
            run.return_value.returncode = 1
            run.return_value.stdout = ""
            self.assertIsNone(query_gpu_total_mb())

    def test_returns_none_on_missing_binary(self) -> None:
        with patch("vram.subprocess.run", side_effect=FileNotFoundError):
            self.assertIsNone(query_gpu_total_mb())


class TestVramPoller(unittest.TestCase):
    def test_tracks_peak_during_poll(self) -> None:
        samples = iter([1000, 5000, 3000, 2000])

        def sampler() -> int | None:
            try:
                return next(samples)
            except StopIteration:
                return 2000

        poller = VramPoller(interval_s=0.01, sampler=sampler)
        poller.start()
        time.sleep(0.05)
        peak = poller.stop()
        self.assertEqual(peak, 5000)

    def test_note_updates_peak(self) -> None:
        poller = VramPoller(sampler=lambda: None)
        poller.note(2048)
        poller.note(1024)
        self.assertEqual(poller.peak_mb, 2048)


if __name__ == "__main__":
    unittest.main()
