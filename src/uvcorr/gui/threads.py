"""``QThread`` workers of the GUI (plan section 9).

Follows adc2kev 2.2.7 ``gui/main.py`` (``ProcessingThread`` through
``CacheBuildThread``): each long operation runs in its own ``QThread`` whose
``run`` calls a blocking, Qt-free function of :mod:`uvcorr.gui.session`, and
reports back through signals, which Qt queues to the GUI thread:

- ``progress``: see each class;
- ``finished(object)``: the operation's result (it shadows ``QThread.finished``
  as adc2kev's threads do; the thread is still finishing ``run`` when the slot
  runs, so call ``wait()`` before dropping the last reference);
- ``error(str)``: a message for the user;
- ``stopped()``: :meth:`stop` was honoured and nothing was changed.

:meth:`stop` sets a ``threading.Event`` that the cache build and the analysis
poll (``stop_flag``), so stopping takes effect within one parser batch or
about 0.1 s of the analysis.

:class:`ChannelDetailThread` is the scatter's short-lived loader (specview's
``ScatterWorker`` pattern): it carries a generation number so the main window
can drop results that a newer selection has superseded.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from uvcorr.analysis import AnalysisCancelled, AnalysisError, WorkerCrashedError
from uvcorr.cache import BuildSettings, CacheBuildCancelled, UVCacheError
from uvcorr.gui.session import (
    BatchOutcome,
    DetailRequest,
    OpenedFile,
    SessionError,
    UVSession,
    load_cache,
    load_raw,
)
from uvcorr.options import FitOptions

logger = logging.getLogger(__name__)

__all__ = [
    "WORKER_CRASH_MESSAGE",
    "CacheBuildThread",
    "CacheOpenThread",
    "ChannelDetailThread",
    "FitAllThread",
    "error_text",
]

# Errors whose message is written for the user; anything else gets its type.
_USER_ERRORS = (UVCacheError, AnalysisError, SessionError, OSError, ValueError, KeyError)


WORKER_CRASH_MESSAGE = (
    "A Fit All worker process stopped abruptly (for example it ran out of memory), so "
    "nothing was stored. Run Fit All again with fewer workers: start uvcorr-gui with "
    "--workers 2, or --workers 1 to fit in-process and name the board that fails."
)
"""The whole message shown for a :class:`~uvcorr.analysis.WorkerCrashedError`."""


def error_text(exc: BaseException) -> str:
    """A user-facing message for an exception raised by a worker.

    A :class:`~uvcorr.analysis.WorkerCrashedError` gets
    :data:`WORKER_CRASH_MESSAGE` (the analysis's own text gives CLI advice).
    """
    if isinstance(exc, WorkerCrashedError):
        return WORKER_CRASH_MESSAGE
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    text = str(exc)
    if isinstance(exc, _USER_ERRORS):
        return text or type(exc).__name__
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


class _WorkerThread(QThread):
    """Base: ``finished`` / ``error`` / ``stopped`` signals and a stop event."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    stopped = pyqtSignal()

    _CANCELLED: tuple[type[BaseException], ...] = (CacheBuildCancelled, AnalysisCancelled)
    _LABEL = "Operation"

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._stop_event = threading.Event()

    @property
    def stop_requested(self) -> bool:
        """Whether :meth:`stop` was called."""
        return self._stop_event.is_set()

    def stop(self) -> None:
        """Ask the operation to stop at its next check."""
        self._stop_event.set()
        logger.info(f"{self._LABEL}: stop requested")

    def run(self) -> None:
        """Run :meth:`_work` and report its outcome through the signals."""
        try:
            value = self._work()
        except self._CANCELLED:
            logger.info(f"{self._LABEL} stopped by the user")
            self.stopped.emit()
            return
        except Exception as exc:
            logger.exception(f"{self._LABEL} failed")
            self.error.emit(error_text(exc))
            return
        self.finished.emit(value)

    def _work(self) -> Any:
        raise NotImplementedError


class CacheBuildThread(_WorkerThread):
    """Open a raw ``.dat`` file: reuse or build its UV cache, then read the index.

    Wraps :func:`uvcorr.gui.session.load_raw` (``open_or_build``).

    Signals:
        progress(float): Build progress in ``[0, 1]`` (1.0 at once when a
            valid cache is reused).
        finished(OpenedFile): The opened file, ready for
            :meth:`UVSession.install`.
        error(str): The open failed (nothing was changed).
        stopped(): The build was stopped (no cache was written).
    """

    progress = pyqtSignal(float)
    _LABEL = "UV cache build"

    def __init__(
        self,
        dat_path: str | Path,
        cache_path: str | Path | None = None,
        *,
        force: bool = False,
        settings: BuildSettings | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.dat_path = Path(dat_path)
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.force = force
        self.settings = settings

    def _work(self) -> OpenedFile:
        return load_raw(
            self.dat_path,
            self.cache_path,
            force=self.force,
            progress_cb=self.progress.emit,
            stop_flag=self._stop_event,
            settings=self.settings,
        )


class CacheOpenThread(_WorkerThread):
    """Open a UV cache directly and read its index (:func:`uvcorr.gui.session.load_cache`).

    Reading the index takes ~0.2 s with stored results and up to a few
    seconds without (every board is scanned for its channels). Not
    stoppable (``stop`` is accepted and ignored).

    Signals:
        finished(OpenedFile), error(str): As :class:`CacheBuildThread`.
    """

    _LABEL = "Opening the UV cache"

    def __init__(self, cache_path: str | Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.cache_path = Path(cache_path)

    def _work(self) -> OpenedFile:
        return load_cache(self.cache_path)


class FitAllThread(_WorkerThread):
    """Fit every channel and store the results (:meth:`UVSession.run_batch`).

    The analysis runs in a ``spawn`` process pool (safe from a ``QThread``);
    ``workers=1`` runs it in this thread instead (tests, debugging).

    Signals:
        progress(object, object): ``(done, total)`` exactly as
            :func:`~uvcorr.analysis.analyze_all` reports them (Python ints of
            any size: event counts can exceed a 32-bit int, so the signal
            does not use C++ ints). Only their ratio is meant for display.
        finished(BatchOutcome): The stored results, for
            :meth:`UVSession.apply_batch`.
        error(str): The batch failed or could not be stored.
        stopped(): The batch was stopped; nothing was stored.
    """

    progress = pyqtSignal(object, object)
    _LABEL = "Fit All"

    def __init__(
        self,
        session: UVSession,
        options: FitOptions,
        *,
        workers: int | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.options = options
        self.workers = workers

    def _work(self) -> BatchOutcome:
        return self.session.run_batch(
            self.options,
            workers=self.workers,
            progress_cb=self.progress.emit,
            stop_flag=self._stop_event,
        )


class ChannelDetailThread(QThread):
    """Load one channel and recompute its fit mask for the scatter (:meth:`UVSession.compute_detail`).

    Signals:
        done(int, ChannelDetail): ``(generation, detail)``.
        failed(int, str): ``(generation, message)``.
    """

    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        session: UVSession,
        request: DetailRequest,
        generation: int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.request = request
        self.generation = generation

    def run(self) -> None:
        """Compute the detail and emit ``done`` or ``failed``."""
        try:
            detail = self.session.compute_detail(self.request)
        except Exception as exc:
            logger.exception(f"Loading {self.request.key} for the scatter failed")
            self.failed.emit(self.generation, error_text(exc))
            return
        self.done.emit(self.generation, detail)
