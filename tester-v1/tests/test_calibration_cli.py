"""Unit tests for model_calibration CLI (calibration_cli)."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from calibration_cli import _run_calibration, main  # noqa: E402


class TestMainArgparse(unittest.TestCase):
    def test_main_save_result_quiet_passes_flags_to_run(self) -> None:
        model_dir = Path("/tmp/model")
        with patch("calibration_cli._validate_model_dir"):
            with patch("calibration_cli._run_calibration", return_value=0) as run:
                main(
                    [
                        str(model_dir),
                        "--save-result",
                        "--quiet",
                    ]
                )
        run.assert_called_once()
        kwargs = run.call_args.kwargs
        self.assertTrue(kwargs["save_result"])
        self.assertTrue(kwargs["quiet"])

    def test_main_quiet_without_save_result_is_noop(self) -> None:
        model_dir = Path("/tmp/model")
        with patch("calibration_cli._validate_model_dir"):
            with patch("calibration_cli._run_calibration", return_value=0) as run:
                main([str(model_dir), "--quiet"])
        kwargs = run.call_args.kwargs
        self.assertFalse(kwargs["save_result"])
        self.assertFalse(kwargs["quiet"])


class TestRunCalibrationStdout(unittest.TestCase):
    def test_default_prints_summary_not_suppressed(self) -> None:
        model_dir = Path("/tmp/model")
        tester_root = Path("/tmp/tester")
        stdout = io.StringIO()
        stderr = io.StringIO()
        summary = {
            "gguf_gb": 1.0,
            "model_vram_mb": 1000,
            "kv_vram_mb": 2000,
            "estimated_context_max": 4096,
        }
        with (
            patch("calibration_cli.run_variant_subprocess") as run_variant,
            patch("calibration_cli._idle_vram_from_probe_output", side_effect=[1000, 1100]),
            patch("calibration_cli.load_server_config") as load_server,
            patch("calibration_cli.parse_context_from_args", side_effect=[4096, 16384]),
            patch("calibration_cli.gguf_size_gb", return_value=1.0),
            patch("calibration_cli.query_gpu_total_mb", return_value=16000),
            patch("calibration_cli.compute_summary", return_value=summary),
            patch("sys.stdout", stdout),
            patch("sys.stderr", stderr),
        ):
            load_server.return_value.model = "/tmp/model.gguf"
            load_server.return_value.args = []
            run_variant.return_value = (0, "probe\n")
            code = _run_calibration(
                model_dir,
                tester_root,
                1536,
                save_result=False,
                quiet=False,
            )
        self.assertEqual(code, 0)
        self.assertIn("* GGUF on disk:", stdout.getvalue())
        run_variant.assert_called()
        self.assertFalse(run_variant.call_args.kwargs["save_result"])

    def test_save_result_quiet_suppresses_summary(self) -> None:
        model_dir = Path("/tmp/model")
        tester_root = Path("/tmp/tester")
        stdout = io.StringIO()
        with (
            patch("calibration_cli.run_variant_subprocess", return_value=(0, "")),
            patch("calibration_cli.slug_from_config_dir", side_effect=["fp", "cp"]),
            patch(
                "calibration_cli.find_latest_result",
                side_effect=[Path("/a.json"), Path("/b.json")],
            ),
            patch("calibration_cli.read_idle_vram_from_result", side_effect=[1000, 1100]),
            patch("calibration_cli.load_server_config") as load_server,
            patch("calibration_cli.parse_context_from_args", side_effect=[4096, 16384]),
            patch("calibration_cli.gguf_size_gb", return_value=1.0),
            patch("calibration_cli.query_gpu_total_mb", return_value=16000),
            patch(
                "calibration_cli.compute_summary",
                return_value={
                    "gguf_gb": 1.0,
                    "model_vram_mb": 1000,
                    "kv_vram_mb": 2000,
                    "estimated_context_max": 4096,
                },
            ),
            patch("sys.stdout", stdout),
            patch("sys.stderr", io.StringIO()),
        ):
            load_server.return_value.model = "/tmp/model.gguf"
            load_server.return_value.args = []
            code = _run_calibration(
                model_dir,
                tester_root,
                1536,
                save_result=True,
                quiet=True,
            )
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
