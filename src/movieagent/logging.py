"""Logging setup and the JSON-lines trace sink (ADR-0013).

The same typed `Trace` object that the UI renders is serialized here, so there is one
source of truth for what the agent did rather than a log format the UI has to parse.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_CONFIGURED = False


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Idempotent logging setup. Safe to call from Streamlit, which re-runs the script
    on every interaction."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("movieagent")
    root.setLevel(level.upper())
    root.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_file is not None:
        # A misconfigured LOG_FILE must degrade to stderr, never take the app down.
        # Opening a directory raises PermissionError on Windows and IsADirectoryError
        # on POSIX, so both are caught.
        try:
            if log_file.is_dir():
                raise IsADirectoryError(
                    f"LOG_FILE points at a directory ({log_file}); it must name a file"
                )
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(fmt)
            root.addHandler(file_handler)
        except (OSError, ValueError) as exc:
            root.warning("file logging disabled (%s); logging to stderr only", exc)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"movieagent.{name}")


class TraceSink:
    """Append-only JSONL sink for `Trace` objects.

    Deliberately best-effort: an observability failure must never break a user's turn,
    so write errors are logged and swallowed.
    """

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._log = get_logger("trace")

    def write(self, payload: dict[str, Any]) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str, ensure_ascii=False) + "\n")
        except OSError as exc:  # pragma: no cover - environment dependent
            self._log.warning("could not write trace line: %s", exc)
