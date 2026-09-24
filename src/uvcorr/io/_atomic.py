"""Atomic text-file replacement for the exporters."""

from __future__ import annotations

import contextlib
import os
import secrets
from collections.abc import Sequence
from pathlib import Path


def atomic_write_texts(items: Sequence[tuple[str | Path, str, str]]) -> list[Path]:
    """Replace several text files, each through a temporary file renamed over it.

    Every temporary file (``.<name>.<random>.tmp`` next to its destination) is
    written and ``fsync``-ed first; only then are they renamed over their
    destinations, back to back. So a failure while writing (disk full, an
    unwritable directory, Ctrl-C) leaves every destination untouched, and a
    reader never sees a half-written file. Line endings are written exactly as
    given (no newline translation).

    Args:
        items: ``(path, text, encoding)`` per file; each directory must exist.

    Returns:
        The destination paths.

    Raises:
        OSError: If a file cannot be written or renamed.
        UnicodeEncodeError: If a text does not fit its encoding.
    """
    paths = [Path(path) for path, _, _ in items]
    temps: list[Path] = []
    try:
        for path, (_, text, encoding) in zip(paths, items):
            tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
            with open(tmp, "x", encoding=encoding, newline="") as f:
                temps.append(tmp)
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
        for tmp, path in zip(temps, paths):
            os.replace(tmp, path)
    except BaseException:
        for tmp in temps:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
        raise
    for directory in {path.parent for path in paths}:
        _fsync_directory(directory)
    return paths


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    """Replace one text file atomically (see :func:`atomic_write_texts`).

    Args:
        path: Destination file; its directory must exist.
        text: The complete file content.
        encoding: Text encoding.

    Returns:
        The destination path.
    """
    return atomic_write_texts([(path, text, encoding)])[0]


def _fsync_directory(directory: Path) -> None:
    """Persist renames in ``directory`` (best effort; not supported everywhere)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
