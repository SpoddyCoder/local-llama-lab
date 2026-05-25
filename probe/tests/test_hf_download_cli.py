"""Unit tests for hf_download CLI and config helpers."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import MODEL_YAML  # noqa: E402
from hf_download_cli import main, run_hf_download  # noqa: E402
from hf_download_config import (  # noqa: E402
    build_hf_download_argv,
    format_path_for_yaml,
    load_hf_download,
    load_hf_download_required,
    resolve_hf_cache_path,
    update_model_path_if_changed,
)


def _write_hf_model_dir(
    root: Path,
    *,
    slug: str = "hf-model",
    model_path: str = "/path/to/old.gguf",
    repo: str = "org/repo",
    file: str = "model.gguf",
    include_hf_download: bool = True,
) -> Path:
    model_dir = root / slug
    model_dir.mkdir(parents=True)
    lines = [f"model: {model_path}"]
    if include_hf_download:
        lines.extend(
            [
                "hf-download:",
                f"  repo: {repo}",
                f"  file: {file}",
            ]
        )
    (model_dir / MODEL_YAML).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return model_dir


def _write_hf_hub_layout(
    hf_home: Path,
    *,
    repo: str = "org/repo",
    file_name: str = "model.gguf",
    revision: str = "abc123rev",
    ref_name: str = "main",
    extra_snapshots: list[tuple[str, float]] | None = None,
) -> Path:
    org, name = repo.split("/", 1)
    cache_dir = hf_home / "hub" / f"models--{org}--{name}"
    refs_dir = cache_dir / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / ref_name).write_text(revision, encoding="utf-8")

    snapshot_path = cache_dir / "snapshots" / revision / file_name
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(b"gguf")

    if extra_snapshots:
        for rev, mtime in extra_snapshots:
            candidate = cache_dir / "snapshots" / rev / file_name
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(b"gguf")
            os.utime(candidate, (mtime, mtime))

    return snapshot_path


@contextmanager
def _capture_output():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
        yield stdout, stderr


class TestHfDownloadConfig(unittest.TestCase):
    def test_load_hf_download_returns_none_when_key_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_hf_model_dir(Path(tmp), include_hf_download=False)
            model_yaml = model_dir / MODEL_YAML
            self.assertIsNone(load_hf_download(model_yaml))

    def test_load_hf_download_required_raises_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_hf_model_dir(Path(tmp), include_hf_download=False)
            model_yaml = model_dir / MODEL_YAML
            with self.assertRaises(ValueError) as ctx:
                load_hf_download_required(model_yaml)
            self.assertIn("hf-download", str(ctx.exception))

    def test_invalid_hf_download_block_not_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "bad-block"
            model_dir.mkdir()
            model_yaml = model_dir / MODEL_YAML
            model_yaml.write_text("model: /x.gguf\nhf-download: not-a-map\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_hf_download(model_yaml)
            self.assertIn("must be a mapping", str(ctx.exception))

    def test_invalid_hf_download_empty_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "empty-repo"
            model_dir.mkdir()
            model_yaml = model_dir / MODEL_YAML
            model_yaml.write_text(
                "model: /x.gguf\nhf-download:\n  repo: '  '\n  file: model.gguf\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_hf_download(model_yaml)
            self.assertIn(".repo", str(ctx.exception))

    def test_invalid_hf_download_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "empty-file"
            model_dir.mkdir()
            model_yaml = model_dir / MODEL_YAML
            model_yaml.write_text(
                "model: /x.gguf\nhf-download:\n  repo: org/repo\n  file: ''\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_hf_download(model_yaml)
            self.assertIn(".file", str(ctx.exception))

    def test_invalid_hf_download_wrong_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "wrong-types"
            model_dir.mkdir()
            model_yaml = model_dir / MODEL_YAML
            model_yaml.write_text(
                "model: /x.gguf\nhf-download:\n  repo: 123\n  file: 456\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_hf_download(model_yaml)

    def test_build_hf_download_argv(self) -> None:
        self.assertEqual(
            build_hf_download_argv("org/repo", "model.gguf"),
            ["hf", "download", "org/repo", "model.gguf"],
        )

    def test_resolve_hf_cache_path_via_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hf_home = Path(tmp)
            expected = _write_hf_hub_layout(hf_home, revision="rev-from-refs")
            with patch.dict(os.environ, {"HF_HOME": str(hf_home)}, clear=False):
                resolved = resolve_hf_cache_path("org/repo", "model.gguf")
            self.assertEqual(resolved, expected)

    def test_resolve_hf_cache_path_fallback_newest_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hf_home = Path(tmp)
            org, name = "org", "repo"
            cache_dir = hf_home / "hub" / f"models--{org}--{name}"
            cache_dir.mkdir(parents=True)

            older_mtime = time.time() - 100
            newer_mtime = time.time()
            older = cache_dir / "snapshots" / "older-rev" / "model.gguf"
            newer = cache_dir / "snapshots" / "newer-rev" / "model.gguf"
            older.parent.mkdir(parents=True)
            newer.parent.mkdir(parents=True)
            older.write_bytes(b"old")
            newer.write_bytes(b"new")
            os.utime(older, (older_mtime, older_mtime))
            os.utime(newer, (newer_mtime, newer_mtime))

            with patch.dict(os.environ, {"HF_HOME": str(hf_home)}, clear=False):
                resolved = resolve_hf_cache_path("org/repo", "model.gguf")
            self.assertEqual(resolved, newer)

    def test_resolve_hf_cache_path_raises_when_cache_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hf_home = Path(tmp)
            with patch.dict(os.environ, {"HF_HOME": str(hf_home)}, clear=False):
                with self.assertRaises(FileNotFoundError):
                    resolve_hf_cache_path("org/missing", "model.gguf")

    def test_update_model_path_if_changed_writes_when_path_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_hf_model_dir(Path(tmp), model_path="/old/path/model.gguf")
            model_yaml = model_dir / MODEL_YAML
            new_path = Path(tmp) / "new" / "model.gguf"
            new_path.parent.mkdir(parents=True)
            new_path.write_bytes(b"gguf")

            changed = update_model_path_if_changed(model_yaml, new_path)
            self.assertTrue(changed)

            raw = yaml.safe_load(model_yaml.read_text(encoding="utf-8"))
            self.assertEqual(raw["model"], format_path_for_yaml(new_path))

    def test_update_model_path_if_changed_noop_when_same(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gguf = root / "models" / "model.gguf"
            gguf.parent.mkdir(parents=True)
            gguf.write_bytes(b"gguf")
            home_path = format_path_for_yaml(gguf)

            model_dir = _write_hf_model_dir(root, model_path=home_path)
            model_yaml = model_dir / MODEL_YAML
            before = model_yaml.read_text(encoding="utf-8")

            changed = update_model_path_if_changed(model_yaml, gguf)
            self.assertFalse(changed)
            self.assertEqual(model_yaml.read_text(encoding="utf-8"), before)

            link = root / "linked-model.gguf"
            link.symlink_to(gguf)
            changed_via_link = update_model_path_if_changed(model_yaml, link)
            self.assertFalse(changed_via_link)
            self.assertEqual(model_yaml.read_text(encoding="utf-8"), before)

    def test_format_path_for_yaml_uses_home_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_home = Path(tmp) / "home"
            fake_home.mkdir()
            under_home = fake_home / "cache" / "model.gguf"
            under_home.parent.mkdir(parents=True)
            under_home.write_bytes(b"x")

            with patch.object(Path, "home", return_value=fake_home):
                formatted = format_path_for_yaml(under_home)
            self.assertEqual(formatted, "~/cache/model.gguf")

            outside = Path(tmp) / "outside.gguf"
            outside.write_bytes(b"x")
            self.assertEqual(format_path_for_yaml(outside), str(outside.resolve()))


class TestRunHfDownload(unittest.TestCase):
    def test_dry_run_prints_command_without_subprocess_or_yaml_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_hf_model_dir(Path(tmp))
            model_yaml = model_dir / MODEL_YAML
            before = model_yaml.read_text(encoding="utf-8")

            with _capture_output() as (stdout, _stderr):
                with patch("hf_download_cli.subprocess.run") as run:
                    code = run_hf_download(model_dir, dry_run=True)
            self.assertEqual(code, 0)
            self.assertEqual(stdout.getvalue().strip(), "hf download org/repo model.gguf")
            run.assert_not_called()
            self.assertEqual(model_yaml.read_text(encoding="utf-8"), before)

    def test_missing_model_dir_returns_error(self) -> None:
        with _capture_output() as (_stdout, stderr):
            code = run_hf_download(Path("models/missing-hf-model"))
        self.assertEqual(code, 1)
        self.assertIn("model directory not found", stderr.getvalue())

    def test_missing_model_yaml_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "empty-model"
            model_dir.mkdir()
            with _capture_output() as (_stdout, stderr):
                code = run_hf_download(model_dir)
            self.assertEqual(code, 1)
            self.assertIn("model config not found", stderr.getvalue())

    def test_missing_hf_download_block_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_hf_model_dir(Path(tmp), include_hf_download=False)
            with _capture_output() as (_stdout, stderr):
                code = run_hf_download(model_dir)
            self.assertEqual(code, 1)
            self.assertIn("hf-download", stderr.getvalue())

    def test_successful_download_updates_model_when_path_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hf_home = Path(tmp) / "hf"
            model_dir = _write_hf_model_dir(Path(tmp) / "models", model_path="/old/path/model.gguf")
            model_yaml = model_dir / MODEL_YAML
            resolved = _write_hf_hub_layout(hf_home)

            proc = MagicMock(returncode=0)
            with _capture_output() as (stdout, _stderr):
                with patch("hf_download_cli.shutil.which", return_value="/usr/bin/hf"):
                    with patch("hf_download_cli.subprocess.run", return_value=proc) as run:
                        with patch.dict(os.environ, {"HF_HOME": str(hf_home)}, clear=False):
                            code = run_hf_download(model_dir)
            self.assertEqual(code, 0)
            run.assert_called_once_with(["hf", "download", "org/repo", "model.gguf"], check=False)
            self.assertIn("Updated model:", stdout.getvalue())

            raw = yaml.safe_load(model_yaml.read_text(encoding="utf-8"))
            self.assertEqual(raw["model"], format_path_for_yaml(resolved))

    def test_successful_download_prints_unchanged_when_path_same(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hf_home = Path(tmp) / "hf"
            resolved = _write_hf_hub_layout(hf_home)
            model_dir = _write_hf_model_dir(
                Path(tmp) / "models",
                model_path=format_path_for_yaml(resolved),
            )

            proc = MagicMock(returncode=0)
            with _capture_output() as (stdout, _stderr):
                with patch("hf_download_cli.shutil.which", return_value="/usr/bin/hf"):
                    with patch("hf_download_cli.subprocess.run", return_value=proc):
                        with patch.dict(os.environ, {"HF_HOME": str(hf_home)}, clear=False):
                            code = run_hf_download(model_dir)
            self.assertEqual(code, 0)
            self.assertIn("model: unchanged", stdout.getvalue())

    def test_propagates_subprocess_nonzero_returncode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_hf_model_dir(Path(tmp))
            proc = MagicMock(returncode=42)
            with patch("hf_download_cli.shutil.which", return_value="/usr/bin/hf"):
                with patch("hf_download_cli.subprocess.run", return_value=proc):
                    code = run_hf_download(model_dir)
            self.assertEqual(code, 42)

    def test_hf_not_on_path_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_hf_model_dir(Path(tmp))
            with _capture_output() as (_stdout, stderr):
                with patch("hf_download_cli.shutil.which", return_value=None):
                    with patch("hf_download_cli.subprocess.run") as run:
                        code = run_hf_download(model_dir)
            self.assertEqual(code, 1)
            self.assertIn("hf CLI not found", stderr.getvalue())
            run.assert_not_called()


class TestHfDownloadMain(unittest.TestCase):
    def test_main_delegates_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_hf_model_dir(Path(tmp))
            with patch("hf_download_cli.run_hf_download", return_value=0) as run:
                code = main([str(model_dir), "--dry-run"])
            self.assertEqual(code, 0)
            run.assert_called_once_with(model_dir, dry_run=True)

    def test_main_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _write_hf_model_dir(Path(tmp))
            with patch("hf_download_cli.run_hf_download", return_value=0) as run:
                code = main([str(model_dir)])
            self.assertEqual(code, 0)
            run.assert_called_once_with(model_dir, dry_run=False)


if __name__ == "__main__":
    unittest.main()
