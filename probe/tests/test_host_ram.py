"""Unit tests for host process RSS sampling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from host_ram import sample_process_rss_mb  # noqa: E402


class TestSampleProcessRssMb(unittest.TestCase):
    def test_converts_kb_to_rounded_mib(self) -> None:
        pid = 42
        # 2048 kB equals 2 MiB exactly.
        status = (
            "Name:\tpython\n"
            "VmRSS:\t2048 kB\n"
        )
        with patch("host_ram.open", mock_open(read_data=status)):
            self.assertEqual(sample_process_rss_mb(pid), 2)

    def test_rounds_fractional_mib(self) -> None:
        pid = 7
        # 1536 kB -> 1536 / 1024 = 1.5 -> rounds to 2 MiB.
        status = "VmRSS:\t 1536\tkB\n"
        with patch("host_ram.open", mock_open(read_data=status)):
            self.assertEqual(sample_process_rss_mb(pid), 2)

    def test_rounds_down_near_one_mib(self) -> None:
        pid = 9
        # 1535 kB -> rounds to 1 MiB.
        status = "VmRSS:\t1535 kB\n"
        with patch("host_ram.open", mock_open(read_data=status)):
            self.assertEqual(sample_process_rss_mb(pid), 1)

    def test_opens_correct_proc_status_path(self) -> None:
        pid = 12345
        status = "VmRSS:\t1024 kB\n"
        m_open = mock_open(read_data=status)
        with patch("host_ram.open", m_open):
            self.assertEqual(sample_process_rss_mb(pid), 1)
        m_open.assert_called_once_with(f"/proc/{pid}/status", encoding="utf-8")

    def test_returns_none_when_proc_missing(self) -> None:
        with patch("host_ram.open", side_effect=FileNotFoundError):
            self.assertIsNone(sample_process_rss_mb(999999))

    def test_returns_none_when_no_vmrss(self) -> None:
        status = "Name:\tbash\nPid:\t1\nPPid:\t0\n"
        with patch("host_ram.open", mock_open(read_data=status)):
            self.assertIsNone(sample_process_rss_mb(1))

    def test_returns_none_when_vmrss_malformed(self) -> None:
        status = "VmRSS:\tnot-a-number kB\n"
        with patch("host_ram.open", mock_open(read_data=status)):
            self.assertIsNone(sample_process_rss_mb(2))

    def test_returns_none_when_vmrss_empty_value(self) -> None:
        status = "VmRSS:\n"
        with patch("host_ram.open", mock_open(read_data=status)):
            self.assertIsNone(sample_process_rss_mb(3))

    def test_returns_none_on_permission_denied(self) -> None:
        with patch("host_ram.open", side_effect=PermissionError):
            self.assertIsNone(sample_process_rss_mb(4))

    def test_returns_none_on_unicode_decode_when_read_fails(self) -> None:
        inner = MagicMock()
        inner.read.side_effect = UnicodeDecodeError(
            "utf-8",
            b"\xff",
            0,
            1,
            "invalid start byte",
        )
        cm = MagicMock()
        cm.__enter__.return_value = inner
        cm.__exit__.return_value = None
        with patch("host_ram.open", return_value=cm):
            self.assertIsNone(sample_process_rss_mb(5))


if __name__ == "__main__":
    unittest.main()
