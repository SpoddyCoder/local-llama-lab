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
            "model_max_context": 128000,
        }
        with (
            patch("calibration_cli.run_variant_subprocess") as run_variant,
            patch("calibration_cli._idle_vram_from_probe_output", side_effect=[1000, 1100]),
            patch(
                "calibration_cli.parse_model_max_context_from_metrics_stdout",
                return_value=128000,
            ),
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
                0,
                save_result=False,
                quiet=False,
            )
        self.assertEqual(code, 0)
        self.assertIn("* GGUF on disk:", stdout.getvalue())
        self.assertIn("* Model Max Context:", stdout.getvalue())
        run_variant.assert_called()
        self.assertFalse(run_variant.call_args.kwargs["save_result"])

    def test_save_result_quiet_suppresses_summary(self) -> None:
        model_dir = Path("/tmp/model")
        tester_root = Path("/tmp/tester")
        stdout = io.StringIO()
        footprint_result = Path("/footprint.json")
        ctx_probe_result = Path("/ctx_probe.json")

        def fake_find_latest(
            results_dir: Path, config_dir: Path, root: Path
        ) -> Path:
            if "footprint" in str(config_dir):
                return footprint_result
            return ctx_probe_result

        with (
            patch("calibration_cli.run_variant_subprocess", return_value=(0, "")),
            patch("calibration_cli.format_compact_utc", return_value="sess123"),
            patch("calibration_cli.find_latest_result", side_effect=fake_find_latest),
            patch("calibration_cli.read_idle_vram_from_result", side_effect=[1000, 1100]),
            patch(
                "calibration_cli.read_model_max_context_from_result",
                return_value=128000,
            ),
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
            patch("calibration_cli.config_dir_metadata") as meta,
            patch("calibration_cli.write_calibration_session_summary") as write_summary,
            patch("sys.stdout", stdout),
            patch("sys.stderr", io.StringIO()),
        ):
            load_server.return_value.model = "/tmp/model.gguf"
            load_server.return_value.args = []
            meta.return_value = {"model": "my-model", "variant": None, "config_path": None}
            code = _run_calibration(
                model_dir,
                tester_root,
                0,
                save_result=True,
                quiet=True,
            )
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "")
        write_summary.assert_called_once_with(
            tester_root,
            "my-model",
            "sess123",
            {
                "gguf_gb": 1.0,
                "model_vram_mb": 1000,
                "kv_vram_mb": 2000,
                "estimated_context_max": 4096,
                "model_max_context": 128000,
            },
            footprint_result,
            ctx_probe_result,
        )

    def test_save_result_passes_session_id_to_subprocess(self) -> None:
        model_dir = Path("/tmp/model")
        tester_root = Path("/tmp/tester")
        summary = {
            "gguf_gb": 1.0,
            "model_vram_mb": 1000,
            "kv_vram_mb": 2000,
            "estimated_context_max": 4096,
            "model_max_context": None,
        }

        def fake_find_latest(
            results_dir: Path, config_dir: Path, root: Path
        ) -> Path:
            return Path("/result.json")

        with (
            patch("calibration_cli.run_variant_subprocess", return_value=(0, "")) as run,
            patch("calibration_cli.format_compact_utc", return_value="sess456"),
            patch("calibration_cli.find_latest_result", side_effect=fake_find_latest),
            patch("calibration_cli.read_idle_vram_from_result", side_effect=[1000, 1100]),
            patch(
                "calibration_cli.read_model_max_context_from_result",
                return_value=None,
            ),
            patch("calibration_cli.load_server_config") as load_server,
            patch("calibration_cli.parse_context_from_args", side_effect=[4096, 16384]),
            patch("calibration_cli.gguf_size_gb", return_value=1.0),
            patch("calibration_cli.query_gpu_total_mb", return_value=16000),
            patch("calibration_cli.compute_summary", return_value=summary),
            patch("calibration_cli.config_dir_metadata") as meta,
            patch("calibration_cli.write_calibration_session_summary"),
            patch("sys.stdout", io.StringIO()),
            patch("sys.stderr", io.StringIO()),
        ):
            load_server.return_value.model = "/tmp/model.gguf"
            load_server.return_value.args = []
            meta.return_value = {"model": None, "variant": None, "config_path": None}
            code = _run_calibration(
                model_dir,
                tester_root,
                0,
                save_result=True,
                quiet=False,
            )
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["session_id"], "sess456")

    def test_save_result_find_latest_failure_returns_error(self) -> None:
        model_dir = Path("/tmp/model")
        tester_root = Path("/tmp/tester")

        with (
            patch("calibration_cli.run_variant_subprocess", return_value=(0, "")),
            patch("calibration_cli.format_compact_utc", return_value="sess789"),
            patch(
                "calibration_cli.find_latest_result",
                side_effect=FileNotFoundError("no result JSON in /tmp/results"),
            ),
            patch("calibration_cli.load_server_config") as load_server,
            patch("sys.stdout", io.StringIO()),
            patch("sys.stderr", io.StringIO()) as stderr,
        ):
            load_server.return_value.model = "/tmp/model.gguf"
            load_server.return_value.args = []
            code = _run_calibration(
                model_dir,
                tester_root,
                0,
                save_result=True,
                quiet=True,
            )
        self.assertEqual(code, 1)
        err = stderr.getvalue()
        self.assertIn("no result JSON", err)
        self.assertIn("results dir:", err)


if __name__ == "__main__":
    unittest.main()
