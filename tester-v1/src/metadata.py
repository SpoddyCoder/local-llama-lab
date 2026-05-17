"""Run metadata: llama-server version and GPU info from nvidia-smi."""

from __future__ import annotations

import subprocess

from pathlib import Path

from config import ServerConfig, config_dir_metadata

_NVIDIA_SMI_GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,driver_version",
    "--format=csv,noheader",
]
_VERSION_TIMEOUT_S = 10.0
_NVIDIA_SMI_TIMEOUT_S = 5.0


def empty_metadata() -> dict[str, str | None]:
    return {
        "server_version": None,
        "gpu_name": None,
        "driver_version": None,
        "config_path": None,
        "model": None,
        "variant": None,
    }


def query_server_version(binary: str) -> str | None:
    """Run ``binary --version``; return the version line or compact output."""
    try:
        proc = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=_VERSION_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    text = _combine_output(proc.stdout, proc.stderr)
    if not text:
        return None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("version:"):
            return stripped

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else None


def query_gpu_info() -> tuple[str | None, str | None]:
    """Return (gpu_name, driver_version) from nvidia-smi, or (None, None)."""
    try:
        proc = subprocess.run(
            _NVIDIA_SMI_GPU_QUERY,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None

    if proc.returncode != 0 or not proc.stdout.strip():
        return None, None

    names: list[str] = []
    drivers: list[str] = []
    for line in proc.stdout.splitlines():
        part = line.strip()
        if not part:
            continue
        fields = [f.strip() for f in part.split(",", 1)]
        if len(fields) != 2:
            continue
        name, driver = fields
        if name:
            names.append(name)
        if driver:
            drivers.append(driver)

    if not names and not drivers:
        return None, None

    gpu_name = "; ".join(names) if names else None
    driver_version = drivers[0] if drivers else None
    return gpu_name, driver_version


def collect_run_metadata(
    server: ServerConfig,
    *,
    config_dir: Path | None = None,
    tester_root: Path | None = None,
) -> dict[str, str | None]:
    """Gather optional metadata fields for a result JSON document."""
    gpu_name, driver_version = query_gpu_info()
    meta = empty_metadata()
    meta["server_version"] = query_server_version(server.binary)
    meta["gpu_name"] = gpu_name
    meta["driver_version"] = driver_version
    if config_dir is not None and tester_root is not None:
        meta.update(config_dir_metadata(config_dir, tester_root))
    return meta


def _combine_output(stdout: str, stderr: str) -> str:
    parts = [stdout.strip(), stderr.strip()]
    return "\n".join(p for p in parts if p)
