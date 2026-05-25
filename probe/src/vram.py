"""GPU VRAM sampling via nvidia-smi."""

from __future__ import annotations

import subprocess
import threading
from typing import Callable

_POLL_INTERVAL_S = 0.5
_NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=memory.used",
    "--format=csv,noheader,nounits",
]
_NVIDIA_SMI_QUERY_TOTAL = [
    "nvidia-smi",
    "--query-gpu=memory.total",
    "--format=csv,noheader,nounits",
]


def _parse_nvidia_smi_csv_mb(stdout: str) -> int | None:
    values: list[int] = []
    for line in stdout.splitlines():
        part = line.strip()
        if not part:
            continue
        try:
            values.append(int(float(part)))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


def _run_nvidia_smi_query(query: list[str]) -> int | None:
    try:
        proc = subprocess.run(
            query,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return _parse_nvidia_smi_csv_mb(proc.stdout)


def sample_vram_mb() -> int | None:
    """Return max memory.used (MiB) across GPUs, or None if unavailable."""
    return _run_nvidia_smi_query(_NVIDIA_SMI_QUERY)


def query_gpu_total_mb() -> int | None:
    """Return max memory.total (MiB) across GPUs, or None if unavailable."""
    return _run_nvidia_smi_query(_NVIDIA_SMI_QUERY_TOTAL)


def _update_peak(current: int | None, sample: int | None) -> int | None:
    if sample is None:
        return current
    if current is None or sample > current:
        return sample
    return current


class VramPoller:
    """Background nvidia-smi polling; tracks peak MiB until stopped."""

    def __init__(
        self,
        *,
        interval_s: float = _POLL_INTERVAL_S,
        sampler: Callable[[], int | None] | None = None,
    ) -> None:
        self._interval_s = interval_s
        self._sampler = sampler or sample_vram_mb
        self._peak_mb: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def peak_mb(self) -> int | None:
        return self._peak_mb

    def note(self, mb: int | None) -> None:
        """Include a sample in the running peak (e.g. idle reading before prompt)."""
        self._peak_mb = _update_peak(self._peak_mb, mb)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("VRAM poller already started")
        self._stop.clear()
        self._peak_mb = _update_peak(self._peak_mb, self._sampler())
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s + 5)
            self._thread = None
        sample = self._sampler()
        self._peak_mb = _update_peak(self._peak_mb, sample)
        return self._peak_mb

    def _poll_loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._peak_mb = _update_peak(self._peak_mb, self._sampler())
