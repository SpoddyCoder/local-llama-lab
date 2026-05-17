"""llama-server subprocess lifecycle: start, health poll, teardown."""

from __future__ import annotations

import collections
import os
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import IO, Iterator

import httpx

from config import ServerConfig

_TAIL_LINES = 50
_TEARDOWN_TERM_WAIT_S = 10.0
_HEALTH_REQUEST_TIMEOUT_S = 2.0
_HEALTH_PATHS = ("/health", "/v1/models")


def build_argv(server: ServerConfig) -> list[str]:
    """Build llama-server argv: [binary, *args, '-m', model_path]."""
    return [server.binary, *server.args, "-m", server.model]


def resolve_base_url(server: ServerConfig, base_url: str | None = None) -> str:
    """Use explicit base_url or derive from server port (default 8080)."""
    if base_url is not None:
        return base_url.rstrip("/")
    port = server.port
    if port is not None:
        return f"http://127.0.0.1:{port}"
    return "http://127.0.0.1:8080"


def _stream_lines(pipe: IO[str], tail: collections.deque[str]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            tail.append(line.rstrip("\n"))
    finally:
        pipe.close()


class ServerProcess:
    """Managed llama-server subprocess with health polling and teardown."""

    def __init__(self, server: ServerConfig, base_url: str) -> None:
        self.server = server
        self.base_url = base_url.rstrip("/")
        self._proc: subprocess.Popen[str] | None = None
        self._stdout_tail: collections.deque[str] = collections.deque(maxlen=_TAIL_LINES)
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=_TAIL_LINES)
        self._reader_threads: list[threading.Thread] = []
        self._stopped = False
        self.ready_endpoint: str | None = None
        self.server_ready_s: float | None = None
        self._start_monotonic: float | None = None

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("server already started")

        argv = build_argv(self.server)
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._reader_threads = [
            threading.Thread(
                target=_stream_lines,
                args=(self._proc.stdout, self._stdout_tail),
                daemon=True,
            ),
            threading.Thread(
                target=_stream_lines,
                args=(self._proc.stderr, self._stderr_tail),
                daemon=True,
            ),
        ]
        for t in self._reader_threads:
            t.start()
        self._start_monotonic = time.monotonic()

    def wait_ready(self) -> None:
        if self._proc is None:
            raise RuntimeError("server not started")

        deadline = time.monotonic() + self.server.ready_timeout_s
        last_error: str | None = None

        with httpx.Client(timeout=_HEALTH_REQUEST_TIMEOUT_S) as client:
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        _ready_failure_message(
                            self,
                            f"llama-server exited early with code {self._proc.returncode}",
                        )
                    )

                endpoint = _probe_health(client, self.base_url)
                if endpoint is not None:
                    self.ready_endpoint = endpoint
                    if self._start_monotonic is not None:
                        self.server_ready_s = time.monotonic() - self._start_monotonic
                    return

                last_error = "health endpoints not ready"
                time.sleep(self.server.ready_poll_interval_s)

        raise TimeoutError(
            _ready_failure_message(
                self,
                f"server not ready within {self.server.ready_timeout_s}s ({last_error})",
            )
        )

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        proc = self._proc
        if proc is None or proc.poll() is not None:
            return

        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return

        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return

        try:
            proc.wait(timeout=_TEARDOWN_TERM_WAIT_S)
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def __enter__(self) -> ServerProcess:
        self.start()
        self.wait_ready()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


@contextmanager
def managed_server(
    server_config: ServerConfig,
    base_url: str,
) -> Iterator[ServerProcess]:
    """Start server, poll until healthy, always tear down on exit."""
    proc = ServerProcess(server_config, base_url)
    old_sigint = signal.getsignal(signal.SIGINT)

    def _on_sigint(signum: int, frame: object | None) -> None:
        proc.stop()
        if callable(old_sigint) and old_sigint not in (
            signal.SIG_DFL,
            signal.SIG_IGN,
        ):
            old_sigint(signum, frame)
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)
    try:
        proc.start()
        proc.wait_ready()
        yield proc
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        proc.stop()


def fetch_model_max_context(client: httpx.Client, base_url: str) -> int | None:
    """Return native context limit from GET /v1/models ``data[0].meta.n_ctx_train``."""
    try:
        resp = client.get(f"{base_url.rstrip('/')}/v1/models")
        if resp.status_code != 200:
            return None
        payload = resp.json()
    except (httpx.HTTPError, ValueError, TypeError):
        return None

    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    meta = first.get("meta")
    if not isinstance(meta, dict):
        return None
    n_ctx_train = meta.get("n_ctx_train")
    if n_ctx_train is None:
        return None
    try:
        return int(n_ctx_train)
    except (TypeError, ValueError):
        return None


def _probe_health(client: httpx.Client, base_url: str) -> str | None:
    for path in _HEALTH_PATHS:
        try:
            resp = client.get(f"{base_url}{path}")
            if resp.status_code == 200:
                return path
        except httpx.HTTPError:
            continue
    return None


def _ready_failure_message(proc: ServerProcess, reason: str) -> str:
    parts = [reason]
    if proc._stderr_tail:
        parts.append("stderr (last lines):")
        parts.extend(proc._stderr_tail)
    return "\n".join(parts)

