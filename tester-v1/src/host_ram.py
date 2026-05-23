"""Host process RAM (RSS) sampling via /proc/<pid>/status."""

from __future__ import annotations


def sample_process_rss_mb(pid: int) -> int | None:
    """Return VmRSS for ``pid`` converted to MiB, or ``None`` if unavailable.

    Reads ``/proc/<pid>/status``, parses the ``VmRSS`` field (kB reported by the
    kernel), and converts with ``round(kb / 1024)``.
    """
    path = f"/proc/{pid}/status"
    try:
        with open(path, encoding="utf-8") as f:
            contents = f.read()
    except (OSError, UnicodeDecodeError):
        return None

    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line.startswith("VmRSS:"):
            continue
        remainder = line.removeprefix("VmRSS:").strip()
        tokens = remainder.split()
        if not tokens:
            return None
        try:
            kb = int(tokens[0])
        except ValueError:
            return None
        return round(kb / 1024)

    return None
