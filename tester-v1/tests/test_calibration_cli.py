"""Unit tests for model_calibration CLI (calibration_cli)."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from calibration import VARIANT_FOOTPRINT  # noqa: E402
from calibration import (  # noqa: E402
    DEFAULT_CALIBRATION_MARGIN_MIB,
    VARIANT_CTX_PROBE,
    VARIANT_HELLO_WORLD,
)
from calibration_cli import (  # noqa: E402
    _run_calibration,
    _validate_model_dir,
    main,
)
from config import MODEL_YAML  # noqa: E402
from paths import TESTER_ROOT  # noqa: E402


from paths import TESTER_ROOT  # noqa: E402

_CALIBRATION_VARIANTS = (
    VARIANT_FOOTPRINT,
    VARIANT_CTX_PROBE,
    VARIANT_HELLO_WORLD,
)


def _write_probe_variants(cal_root: Path) -> None:
    for variant in _CALIBRATION_VARIANTS:
        probe_dir = cal_root / variant
        probe_dir.mkdir(parents=True, exist_ok=True)
        (probe_dir / "server.yaml").write_text("args: |\n", encoding="utf-8")
        (probe_dir / "client.yaml").write_text("messages: []\n", encoding="utf-8")


def _resolve_run_config_side_effect(
    *,
    yaml_n_cpu_moe: int | None = None,
):
    def side_effect(model_dir, variant, *, n_cpu_moe=None):
        footprint_cfg = MagicMock()
        ctx_cfg = MagicMock()
        effective = n_cpu_moe if n_cpu_moe is not None else yaml_n_cpu_moe
        if variant == VARIANT_FOOTPRINT and effective is not None:
            footprint_cfg.server.args = [
                "--n-cpu-moe",
                str(effective),
                "--n-gpu-layers",
                "999",
            ]
        else:
            footprint_cfg.server.args = ["-c", "4096"]
        ctx_cfg.server.args = ["-c", "16384"]
        return footprint_cfg if variant == VARIANT_FOOTPRINT else ctx_cfg

    return side_effect


class TestValidateModelDir(unittest.TestCase):
    def test_accepts_model_and_server_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = Path(tmp) / "tester"
            cal_root = harness / "calibration-tests"
            _write_probe_variants(cal_root)
            model_dir = harness / "models" / "my-model"
            model_dir.mkdir(parents=True)
            (model_dir / MODEL_YAML).write_text("model: /tmp/x.gguf\n", encoding="utf-8")
            (model_dir / "server.yaml").write_text("args: |\n", encoding="utf-8")
            with patch("calibration_cli.CALIBRATION_TESTS_ROOT", cal_root):
                _validate_model_dir(model_dir)

    def test_missing_model_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "my-model"
            model_dir.mkdir()
            (model_dir / "server.yaml").write_text("args: |\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError) as ctx:
                _validate_model_dir(model_dir)
            self.assertIn("model config not found", str(ctx.exception))

    def test_missing_server_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "my-model"
            model_dir.mkdir()
            (model_dir / MODEL_YAML).write_text("model: /tmp/x.gguf\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError) as ctx:
                _validate_model_dir(model_dir)
            self.assertIn("server config not found", str(ctx.exception))

    def test_missing_calibration_variant_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = Path(tmp) / "tester"
            cal_root = harness / "calibration-tests"
            footprint = cal_root / VARIANT_FOOTPRINT
            footprint.mkdir(parents=True)
            (footprint / "server.yaml").write_text("args: |\n", encoding="utf-8")
            (footprint / "client.yaml").write_text("messages: []\n", encoding="utf-8")
            model_dir = harness / "models" / "my-model"
            model_dir.mkdir(parents=True)
            (model_dir / MODEL_YAML).write_text("model: /tmp/x.gguf\n", encoding="utf-8")
            (model_dir / "server.yaml").write_text("args: |\n", encoding="utf-8")
            with patch("calibration_cli.CALIBRATION_TESTS_ROOT", cal_root):
                with self.assertRaises(FileNotFoundError) as ctx:
                    _validate_model_dir(model_dir)
            self.assertIn("calibration probe config missing", str(ctx.exception))
            self.assertIn("calibration-tests/", str(ctx.exception))


class TestRunCalibrationVariant(unittest.TestCase):
    def test_resolves_run_config_and_calls_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = Path(tmp) / "tester"
            cal_root = harness / "calibration-tests"
            ref_dir = cal_root / VARIANT_FOOTPRINT
            ref_dir.mkdir(parents=True)
            (ref_dir / "server.yaml").write_text("args: |\n", encoding="utf-8")
            (ref_dir / "client.yaml").write_text("messages: []\n", encoding="utf-8")
            model_dir = harness / "models" / "my-model"
            model_dir.mkdir(parents=True)
            (model_dir / MODEL_YAML).write_text("model: /tmp/x.gguf\n", encoding="utf-8")
            (model_dir / "server.yaml").write_text("args: |\n", encoding="utf-8")

            from calibration import run_calibration_variant  # noqa: E402
            from model_layout import RunConfig  # noqa: E402

            fake_config = RunConfig(
                server=MagicMock(),
                client_path=ref_dir / "client.yaml",
                model_yaml_path=model_dir / MODEL_YAML,
                model="my-model",
                variant=VARIANT_FOOTPRINT,
            )

            with (
                patch("calibration.resolve_run_config", return_value=fake_config) as resolve,
                patch("calibration._run", return_value=0) as run,
            ):
                returncode, _output = run_calibration_variant(
                    model_dir,
                    VARIANT_FOOTPRINT,
                    save_result=True,
                    quiet=True,
                    session_id="sess1",
                )
            self.assertEqual(returncode, 0)
            resolve.assert_called_once_with(
                model_dir, VARIANT_FOOTPRINT, n_cpu_moe=None
            )
            run.assert_called_once()
            kwargs = run.call_args.kwargs
            self.assertIs(run.call_args.args[0], fake_config)
            self.assertTrue(kwargs["save_result"])
            self.assertTrue(kwargs["quiet"])
            self.assertEqual(kwargs["session_id"], "sess1")

    def test_passes_n_cpu_moe_to_resolve_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = Path(tmp) / "tester"
            cal_root = harness / "calibration-tests"
            ref_dir = cal_root / VARIANT_FOOTPRINT
            ref_dir.mkdir(parents=True)
            (ref_dir / "server.yaml").write_text("args: |\n", encoding="utf-8")
            (ref_dir / "client.yaml").write_text("messages: []\n", encoding="utf-8")
            model_dir = harness / "models" / "my-model"
            model_dir.mkdir(parents=True)
            (model_dir / MODEL_YAML).write_text("model: /tmp/x.gguf\n", encoding="utf-8")
            (model_dir / "server.yaml").write_text("args: |\n", encoding="utf-8")

            from calibration import run_calibration_variant  # noqa: E402
            from model_layout import RunConfig  # noqa: E402

            fake_config = RunConfig(
                server=MagicMock(),
                client_path=ref_dir / "client.yaml",
                model_yaml_path=model_dir / MODEL_YAML,
                model="my-model",
                variant=VARIANT_FOOTPRINT,
            )

            with (
                patch("calibration.resolve_run_config", return_value=fake_config) as resolve,
                patch("calibration._run", return_value=0),
            ):
                run_calibration_variant(
                    model_dir,
                    VARIANT_FOOTPRINT,
                    save_result=False,
                    quiet=False,
                    n_cpu_moe=22,
                )
            self.assertEqual(resolve.call_args.kwargs["n_cpu_moe"], 22)


class TestMainArgparse(unittest.TestCase):
    def test_main_no_args_prints_help(self) -> None:
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            code = main([])
        self.assertEqual(code, 0)
        help_text = stdout.getvalue()
        self.assertIn("usage:", help_text)
        self.assertIn("model_dir", help_text)

    def test_main_save_result_quiet_passes_flags_to_run(self) -> None:
        model_dir = Path("/tmp/model")
        with patch("calibration_cli._validate_model_dir", return_value=None):
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
        with patch("calibration_cli._validate_model_dir", return_value=None):
            with patch("calibration_cli._run_calibration", return_value=0) as run:
                main([str(model_dir), "--quiet"])
        kwargs = run.call_args.kwargs
        self.assertFalse(kwargs["save_result"])
        self.assertFalse(kwargs["quiet"])

    def test_main_default_margin_mib(self) -> None:
        model_dir = Path("/tmp/model")
        with patch("calibration_cli._validate_model_dir", return_value=None):
            with patch("calibration_cli._run_calibration", return_value=0) as run:
                main([str(model_dir)])
        self.assertEqual(
            run.call_args.args[1],
            DEFAULT_CALIBRATION_MARGIN_MIB,
        )

    def test_main_margin_mib_override(self) -> None:
        model_dir = Path("/tmp/model")
        with patch("calibration_cli._validate_model_dir", return_value=None):
            with patch("calibration_cli._run_calibration", return_value=0) as run:
                main([str(model_dir), "--margin-mib", "0"])
        self.assertEqual(run.call_args.args[1], 0)

    def test_main_n_cpu_moe_passes_to_run_calibration(self) -> None:
        model_dir = Path("/tmp/model")
        with patch("calibration_cli._validate_model_dir", return_value=None):
            with patch("calibration_cli._run_calibration", return_value=0) as run:
                main([str(model_dir), "--n-cpu-moe", "22"])
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["n_cpu_moe"], 22)


class TestRunCalibrationStdout(unittest.TestCase):
    def test_loads_model_from_model_yaml(self) -> None:
        model_dir = Path("/tmp/model")
        summary = {
            "gguf_gb": 1.0,
            "model_vram_mb": 1000,
            "kv_vram_mb": 2000,
            "estimated_context_max": 4096,
            "model_max_context": 128000,
        }
        with (
            patch("calibration_cli.run_calibration_variant", return_value=(0, "probe\n")),
            patch("calibration_cli._idle_vram_from_probe_output", side_effect=[1000, 1100]),
            patch(
                "calibration_cli.parse_model_max_context_from_metrics_stdout",
                return_value=128000,
            ),
            patch(
                "calibration_cli.parse_decode_tok_s_from_metrics_stdout",
                return_value=91.2,
            ),
            patch(
                "calibration_cli.load_model_config",
                return_value="/tmp/model.gguf",
            ) as load_model,
            patch(
                "calibration_cli.resolve_run_config",
                side_effect=_resolve_run_config_side_effect(),
            ) as resolve,
            patch("calibration_cli.parse_context_from_args", side_effect=[4096, 16384]),
            patch("calibration_cli.gguf_size_gb", return_value=1.0),
            patch("calibration_cli.query_gpu_total_mb", return_value=16000),
            patch("calibration_cli.compute_summary", return_value=summary),
            patch("sys.stdout", io.StringIO()),
            patch("sys.stderr", io.StringIO()),
        ):
            code = _run_calibration(
                model_dir,
                0,
                save_result=False,
                quiet=False,
            )
        self.assertEqual(code, 0)
        load_model.assert_called_once_with(model_dir / MODEL_YAML)
        self.assertEqual(resolve.call_count, 2)
        resolve.assert_any_call(model_dir, VARIANT_FOOTPRINT, n_cpu_moe=None)
        resolve.assert_any_call(model_dir, VARIANT_CTX_PROBE, n_cpu_moe=None)

    def test_default_prints_summary_not_suppressed(self) -> None:
        model_dir = Path("/tmp/model")
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
            patch("calibration_cli.run_calibration_variant") as run_variant,
            patch("calibration_cli._idle_vram_from_probe_output", side_effect=[1000, 1100]),
            patch(
                "calibration_cli.parse_model_max_context_from_metrics_stdout",
                return_value=128000,
            ),
            patch(
                "calibration_cli.parse_decode_tok_s_from_metrics_stdout",
                return_value=91.2,
            ),
            patch(
                "calibration_cli.load_model_config",
                return_value="/tmp/model.gguf",
            ),
            patch(
                "calibration_cli.resolve_run_config",
                side_effect=_resolve_run_config_side_effect(),
            ),
            patch("calibration_cli.parse_context_from_args", side_effect=[4096, 16384]),
            patch("calibration_cli.gguf_size_gb", return_value=1.0),
            patch("calibration_cli.query_gpu_total_mb", return_value=16000),
            patch("calibration_cli.compute_summary", return_value=summary),
            patch("sys.stdout", stdout),
            patch("sys.stderr", stderr),
        ):
            run_variant.return_value = (0, "probe\n")
            code = _run_calibration(
                model_dir,
                0,
                save_result=False,
                quiet=False,
            )
        self.assertEqual(code, 0)
        out = stdout.getvalue()
        self.assertIn("* Generation throughput: ~91 tok/s", out)
        self.assertIn("* GGUF on disk:", out)
        self.assertIn("* Model Max Context:", out)
        self.assertEqual(run_variant.call_count, 3)
        self.assertFalse(run_variant.call_args.kwargs["save_result"])

    def test_save_result_quiet_suppresses_summary(self) -> None:
        model_dir = Path("/tmp/model")
        stdout = io.StringIO()
        footprint_result = Path("/footprint.json")
        ctx_probe_result = Path("/ctx_probe.json")
        hello_world_result = Path("/hello_world.json")

        def fake_find_latest(
            results_dir: Path, model: str, variant: str
        ) -> Path:
            if variant == "calibration-footprint":
                return footprint_result
            if variant == "calibration-ctx-probe":
                return ctx_probe_result
            return hello_world_result

        with (
            patch("calibration_cli.run_calibration_variant", return_value=(0, "")),
            patch("calibration_cli.format_compact_utc", return_value="sess123"),
            patch("calibration_cli.find_latest_result", side_effect=fake_find_latest),
            patch("calibration_cli.read_idle_vram_from_result", side_effect=[1000, 1100]),
            patch(
                "calibration_cli.read_model_max_context_from_result",
                return_value=128000,
            ),
            patch(
                "calibration_cli.read_decode_tok_s_from_result",
                return_value=91.2,
            ),
            patch(
                "calibration_cli.load_model_config",
                return_value="/tmp/model.gguf",
            ),
            patch(
                "calibration_cli.resolve_run_config",
                side_effect=_resolve_run_config_side_effect(),
            ),
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
            patch("calibration_cli.write_calibration_session_summary") as write_summary,
            patch("sys.stdout", stdout),
            patch("sys.stderr", io.StringIO()),
        ):
            code = _run_calibration(
                model_dir,
                0,
                save_result=True,
                quiet=True,
            )
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "")
        write_summary.assert_called_once_with(
            TESTER_ROOT,
            "model",
            "sess123",
            {
                "gguf_gb": 1.0,
                "model_vram_mb": 1000,
                "kv_vram_mb": 2000,
                "estimated_context_max": 4096,
                "model_max_context": 128000,
                "generation_throughput_tok_s": 91.2,
            },
            footprint_result,
            ctx_probe_result,
            hello_world_result,
        )

    def test_save_result_passes_session_id_to_subprocess(self) -> None:
        model_dir = Path("/tmp/model")
        summary = {
            "gguf_gb": 1.0,
            "model_vram_mb": 1000,
            "kv_vram_mb": 2000,
            "estimated_context_max": 4096,
            "model_max_context": None,
        }

        def fake_find_latest(
            results_dir: Path, model: str, variant: str
        ) -> Path:
            return Path("/result.json")

        with (
            patch("calibration_cli.run_calibration_variant", return_value=(0, "")) as run,
            patch("calibration_cli.format_compact_utc", return_value="sess456"),
            patch("calibration_cli.find_latest_result", side_effect=fake_find_latest),
            patch("calibration_cli.read_idle_vram_from_result", side_effect=[1000, 1100]),
            patch(
                "calibration_cli.read_model_max_context_from_result",
                return_value=None,
            ),
            patch(
                "calibration_cli.read_decode_tok_s_from_result",
                return_value=None,
            ),
            patch(
                "calibration_cli.load_model_config",
                return_value="/tmp/model.gguf",
            ),
            patch(
                "calibration_cli.resolve_run_config",
                side_effect=_resolve_run_config_side_effect(),
            ),
            patch("calibration_cli.parse_context_from_args", side_effect=[4096, 16384]),
            patch("calibration_cli.gguf_size_gb", return_value=1.0),
            patch("calibration_cli.query_gpu_total_mb", return_value=16000),
            patch("calibration_cli.compute_summary", return_value=summary),
            patch("calibration_cli.write_calibration_session_summary"),
            patch("sys.stdout", io.StringIO()),
            patch("sys.stderr", io.StringIO()),
        ):
            code = _run_calibration(
                model_dir,
                0,
                save_result=True,
                quiet=False,
            )
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 3)
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["session_id"], "sess456")

    def test_n_cpu_moe_stdout_includes_model_system_ram_from_footprint_parse(self) -> None:
        model_dir = Path("/tmp/model")
        stdout = io.StringIO()
        with (
            patch("calibration_cli.run_calibration_variant", return_value=(0, "foot\n")),
            patch("calibration_cli._idle_vram_from_probe_output", side_effect=[1000, 1100]),
            patch(
                "calibration_cli.parse_model_max_context_from_metrics_stdout",
                return_value=128000,
            ),
            patch(
                "calibration_cli.parse_decode_tok_s_from_metrics_stdout",
                return_value=91.2,
            ),
            patch(
                "calibration_cli.parse_idle_system_ram_from_metrics_stdout",
                return_value=5000,
            ) as parse_ram,
            patch(
                "calibration_cli.load_model_config",
                return_value="/tmp/model.gguf",
            ),
            patch(
                "calibration_cli.resolve_run_config",
                side_effect=_resolve_run_config_side_effect(),
            ) as resolve,
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
            code = _run_calibration(
                model_dir,
                0,
                save_result=False,
                quiet=False,
                n_cpu_moe=4,
            )
        self.assertEqual(code, 0)
        parse_ram.assert_called_once_with("foot\n")
        self.assertIn("* Model System RAM: 4.88 GB", stdout.getvalue())
        self.assertEqual(resolve.call_count, 2)
        for call in resolve.call_args_list:
            self.assertEqual(call.kwargs["n_cpu_moe"], 4)

    def test_server_yaml_n_cpu_moe_includes_model_system_ram_without_cli_flag(
        self,
    ) -> None:
        model_dir = Path("/tmp/model")
        stdout = io.StringIO()

        def resolve_with_yaml_moe(model_dir_arg, variant, *, n_cpu_moe=None):
            footprint_cfg = MagicMock()
            ctx_cfg = MagicMock()
            footprint_cfg.server.args = [
                "--n-cpu-moe",
                "24",
                "--n-gpu-layers",
                "999",
            ]
            ctx_cfg.server.args = []
            if variant == VARIANT_FOOTPRINT:
                return footprint_cfg
            return ctx_cfg

        with (
            patch("calibration_cli.run_calibration_variant", return_value=(0, "foot\n")),
            patch("calibration_cli._idle_vram_from_probe_output", side_effect=[1000, 1100]),
            patch(
                "calibration_cli.parse_model_max_context_from_metrics_stdout",
                return_value=128000,
            ),
            patch(
                "calibration_cli.parse_decode_tok_s_from_metrics_stdout",
                return_value=91.2,
            ),
            patch(
                "calibration_cli.parse_idle_system_ram_from_metrics_stdout",
                return_value=10368,
            ) as parse_ram,
            patch(
                "calibration_cli.load_model_config",
                return_value="/tmp/model.gguf",
            ),
            patch(
                "calibration_cli.resolve_run_config",
                side_effect=resolve_with_yaml_moe,
            ) as resolve,
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
            code = _run_calibration(
                model_dir,
                0,
                save_result=False,
                quiet=False,
                n_cpu_moe=None,
            )
        self.assertEqual(code, 0)
        parse_ram.assert_called_once_with("foot\n")
        self.assertIn("* Model System RAM: 10.12 GB", stdout.getvalue())
        self.assertEqual(resolve.call_count, 2)
        for call in resolve.call_args_list:
            self.assertIsNone(call.kwargs["n_cpu_moe"])

    def test_save_result_n_cpu_moe_includes_model_system_ram_in_session_summary(
        self,
    ) -> None:
        model_dir = Path("/tmp/model")
        stdout = io.StringIO()
        footprint_result = Path("/footprint.json")
        ctx_probe_result = Path("/ctx_probe.json")
        hello_world_result = Path("/hello_world.json")

        def fake_find_latest(
            results_dir: Path, model: str, variant: str
        ) -> Path:
            if variant == "calibration-footprint":
                return footprint_result
            if variant == "calibration-ctx-probe":
                return ctx_probe_result
            return hello_world_result

        with (
            patch("calibration_cli.run_calibration_variant", return_value=(0, "")),
            patch("calibration_cli.format_compact_utc", return_value="sessmoe"),
            patch("calibration_cli.find_latest_result", side_effect=fake_find_latest),
            patch(
                "calibration_cli.read_idle_vram_from_result",
                side_effect=[1000, 1100],
            ),
            patch(
                "calibration_cli.read_idle_system_ram_from_result",
                return_value=8765,
            ) as read_ram,
            patch(
                "calibration_cli.read_model_max_context_from_result",
                return_value=None,
            ),
            patch(
                "calibration_cli.read_decode_tok_s_from_result",
                return_value=None,
            ),
            patch(
                "calibration_cli.load_model_config",
                return_value="/tmp/model.gguf",
            ),
            patch(
                "calibration_cli.resolve_run_config",
                side_effect=_resolve_run_config_side_effect(),
            ),
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
            patch("calibration_cli.write_calibration_session_summary") as write_summary,
            patch("sys.stdout", stdout),
            patch("sys.stderr", io.StringIO()),
        ):
            code = _run_calibration(
                model_dir,
                0,
                save_result=True,
                quiet=True,
                n_cpu_moe=2,
            )
        self.assertEqual(code, 0)
        read_ram.assert_called_once_with(footprint_result)
        write_summary.assert_called_once()
        passed_summary = write_summary.call_args.args[3]
        self.assertEqual(passed_summary["model_system_ram_mb"], 8765)

    def test_save_result_find_latest_failure_returns_error(self) -> None:
        model_dir = Path("/tmp/model")

        with (
            patch("calibration_cli.run_calibration_variant", return_value=(0, "")),
            patch("calibration_cli.format_compact_utc", return_value="sess789"),
            patch(
                "calibration_cli.find_latest_result",
                side_effect=FileNotFoundError("no result JSON in /tmp/results"),
            ),
            patch(
                "calibration_cli.load_model_config",
                return_value="/tmp/model.gguf",
            ),
            patch(
                "calibration_cli.resolve_run_config",
                side_effect=_resolve_run_config_side_effect(),
            ),
            patch("calibration_cli.gguf_size_gb", return_value=1.0),
            patch("sys.stdout", io.StringIO()),
            patch("sys.stderr", io.StringIO()) as stderr,
        ):
            code = _run_calibration(
                model_dir,
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
