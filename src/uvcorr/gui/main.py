"""Entry point of the ``uvcorr-gui`` console script: ``uvcorr-gui [file]``.

This module is deliberately light: it imports only the standard library at
module level. Fit All runs the analysis in a ``spawn`` process pool, and
every spawned worker re-runs the parent's main script, which for the console
script is ``from uvcorr.gui.main import main``. Qt, pyqtgraph and the window
(:mod:`uvcorr.gui.window`) are therefore imported inside :func:`main`, so the
workers never pay for them. ``MainWindow`` is still reachable as
``uvcorr.gui.main.MainWindow`` (imported on first access).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Any

__all__ = ["main"]

_SIGNAL_POLL_MS = 250  # how often Python gets to run a pending SIGINT handler


def __getattr__(name: str) -> Any:
    """Lazy access to the window module's public names (``MainWindow``, ...)."""
    if name in ("MainWindow", "PHASE5_TABS"):
        from uvcorr.gui import window

        return getattr(window, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="uvcorr-gui",
        description="View and re-fit RENA-3 U/V ellipse corrections.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="raw .dat file (its UV cache is built or reused) or a .uv.h5 cache",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker processes for Fit All (default: min(8, CPU count))",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="log progress to stderr (-v: info, -vv: debug)",
    )
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``uvcorr-gui [file]``. Returns a process exit code."""
    args = _parse_args(argv)
    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    from uvcorr.gui.window import MainWindow

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    app.setOrganizationName("uvcorr")
    app.setApplicationName("uvcorr-gui")
    window = MainWindow(workers=args.workers)
    window.show()
    if args.file is not None:
        path: Path = args.file
        QTimer.singleShot(0, lambda: window.open_path(path))

    # Ctrl+C in the terminal: Qt's event loop never returns to Python on its own, so
    # a timer wakes the interpreter to run the handler, which closes the window
    # properly (stopping running work and saving the layout).
    previous = signal.signal(
        signal.SIGINT, lambda _signum, _frame: QTimer.singleShot(0, window.request_quit)
    )
    wake = QTimer()
    wake.timeout.connect(lambda: None)
    wake.start(_SIGNAL_POLL_MS)
    try:
        return int(app.exec())
    finally:
        wake.stop()
        signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    sys.exit(main())
