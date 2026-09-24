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

:class:`RefitThread` re-fits one channel or board in-process (stoppable
between channels) and stores the outcome, as :class:`FitAllThread` stores a
batch; :class:`RevertThread` deletes overrides (a write that can wait up to
a second for a busy file, so it does not run on the GUI thread either).

:class:`ChannelDetailThread` is the channel tabs' short-lived loader
(specview's ``ScatterWorker`` pattern): it carries a generation number so the
main window can drop results that a newer selection has superseded. It works
in two stages, so the scatter is drawn as soon as its data are ready: first
the channel detail (the Scatter tab), then the Radial and Radius vs angle
tabs' data computed from it (:class:`DetailViews`, up to ~0.3 s more for the
largest channel). :class:`BoardGridThread` computes the Board grid tab's
data for one board, with its own generation number.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from uvcorr.analysis import (
    AnalysisCancelled,
    AnalysisError,
    ChannelKey,
    ChannelResult,
    WorkerCrashedError,
)
from uvcorr.cache import BuildSettings, CacheBuildCancelled, UVCache, UVCacheError
from uvcorr.gui.angle import AngleViewData, compute_angle_view
from uvcorr.gui.board_grid import compute_board_grid
from uvcorr.gui.radial import RadialViewData, compute_radial_view
from uvcorr.gui.session import (
    BatchOutcome,
    ChannelDetail,
    DetailRequest,
    OpenedFile,
    RefitOutcome,
    RefitRequest,
    RevertOutcome,
    RevertRequest,
    SessionError,
    UVSession,
    load_cache,
    load_raw,
)
from uvcorr.options import FitOptions

logger = logging.getLogger(__name__)

__all__ = [
    "WORKER_CRASH_MESSAGE",
    "BoardGridThread",
    "CacheBuildThread",
    "CacheOpenThread",
    "ChannelDetailThread",
    "DetailViews",
    "FitAllThread",
    "RefitThread",
    "RevertThread",
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
        keep_overrides: bool = True,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.options = options
        self.workers = workers
        self.keep_overrides = keep_overrides

    def _work(self) -> BatchOutcome:
        return self.session.run_batch(
            self.options,
            keep_overrides=self.keep_overrides,
            workers=self.workers,
            progress_cb=self.progress.emit,
            stop_flag=self._stop_event,
        )


class RefitThread(_WorkerThread):
    """Re-fit a channel or board and store the outcome (:meth:`UVSession.run_refit`).

    Runs in-process (no process pool): a board is at most ~3 s of fitting.
    :meth:`stop` takes effect before the next channel, so it stops a board
    re-fit, not a single channel's fit.

    Signals:
        progress(int, int): ``(channels_done, channels_total)``.
        finished(RefitOutcome): The stored outcome, for
            :meth:`UVSession.apply_refit`.
        error(str): The re-fit failed or could not be stored (nothing changed).
        stopped(): The re-fit was stopped; nothing was stored.
    """

    progress = pyqtSignal(int, int)
    _LABEL = "Re-fit"

    def __init__(
        self, session: UVSession, request: RefitRequest, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.request = request

    def _work(self) -> RefitOutcome:
        return self.session.run_refit(
            self.request, progress_cb=self.progress.emit, stop_flag=self._stop_event
        )


class RevertThread(_WorkerThread):
    """Delete overrides in one cache write (:meth:`UVSession.run_revert`).

    Not stoppable (``stop`` is accepted and ignored): the write is short.

    Signals:
        finished(RevertOutcome): The outcome, for :meth:`UVSession.apply_revert`.
        error(str): The write failed (nothing was deleted).
    """

    _LABEL = "Revert to batch"

    def __init__(
        self, session: UVSession, request: RevertRequest, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.request = request

    def _work(self) -> RevertOutcome:
        return self.session.run_revert(self.request)


@dataclass(frozen=True, eq=False)
class DetailViews:
    """The Radial and Radius vs angle tabs' data of one channel (second stage).

    Attributes:
        radial: :func:`~uvcorr.gui.radial.compute_radial_view` of the detail,
            or None if it failed.
        angle: :func:`~uvcorr.gui.angle.compute_angle_view` of the detail,
            or None if it failed.
        error: Why a view could not be computed ("" if both were).
        seconds: Time spent computing both.
    """

    radial: RadialViewData | None
    angle: AngleViewData | None
    error: str
    seconds: float


class ChannelDetailThread(QThread):
    """Load one channel for the channel tabs, in two stages.

    1. :meth:`UVSession.compute_detail` (points, kept mask, correction):
       ``done``.
    2. The Radial and Radius vs angle tabs' data (:class:`DetailViews`):
       ``views``; its value is None when :meth:`skip_views` was called (a
       newer selection superseded this one), so no time is spent on a
       channel nobody will see.

    The last signal is always exactly one of ``views`` and ``failed`` (also
    after ``done``, if the second stage raises), so the main window's queue
    of loads never stalls.

    Signals:
        done(int, ChannelDetail): ``(generation, detail)``.
        views(int, DetailViews | None): ``(generation, views)``.
        failed(int, str): ``(generation, message)``.
    """

    done = pyqtSignal(int, object)
    views = pyqtSignal(int, object)
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
        self._skip_views = threading.Event()

    def skip_views(self) -> None:
        """Do not compute the second stage (emit ``views`` with None instead)."""
        self._skip_views.set()

    def run(self) -> None:
        """Compute the detail, then the views; see the class docstring for the signals.

        Exactly one of ``failed`` and ``views`` is always the last signal,
        even if something unexpected escapes (the main window starts the
        next load only when it arrives).
        """
        ended = False
        try:
            try:
                detail = self.session.compute_detail(self.request)
                self.done.emit(self.generation, detail)
                views = None if self._skip_views.is_set() else self._compute_views(detail)
            except Exception as exc:
                logger.exception(f"Loading {self.request.key} failed")
                self.failed.emit(self.generation, error_text(exc))
                ended = True
                return
            self.views.emit(self.generation, views)
            ended = True
        finally:
            if not ended:  # something beyond Exception: still release the window's queue
                logger.error(f"Loading {self.request.key} stopped unexpectedly")
                self.failed.emit(self.generation, "the channel load stopped unexpectedly")

    def _compute_views(self, detail: ChannelDetail) -> DetailViews:
        """The Radial and Radius vs angle tabs' data (a failed view is reported, not raised)."""
        t_start = time.perf_counter()
        radial: RadialViewData | None = None
        angle: AngleViewData | None = None
        errors: list[str] = []
        try:
            radial = compute_radial_view(detail)
        except Exception as exc:
            logger.exception(f"Computing the Radial tab of {self.request.key} failed")
            errors.append(f"radial histograms: {error_text(exc)}")
        try:
            angle = compute_angle_view(detail)
        except Exception as exc:
            logger.exception(f"Computing the Radius vs angle tab of {self.request.key} failed")
            errors.append(f"radius vs angle: {error_text(exc)}")
        return DetailViews(radial, angle, "; ".join(errors), time.perf_counter() - t_start)


class BoardGridThread(QThread):
    """Compute the Board grid tab's data of one board.

    Runs :func:`~uvcorr.gui.board_grid.compute_board_grid` on the board's
    events, read from ``cache`` through the session's board cache; the cache
    and the results are a snapshot taken on the GUI thread.

    Signals:
        done(int, BoardGridData): ``(generation, data)``.
        failed(int, str): ``(generation, message)``.
    """

    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        session: UVSession,
        cache: UVCache,
        node: int,
        board: int,
        results: Mapping[ChannelKey, ChannelResult | None],
        generation: int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.cache = cache
        self.node = node
        self.board = board
        self.results = results
        self.generation = generation

    def run(self) -> None:
        """Compute the grid and emit ``done`` or ``failed``."""
        try:
            board_uv = self.session.board_data(self.node, self.board, self.cache)
            data = compute_board_grid(board_uv, self.results)
        except Exception as exc:
            logger.exception(f"Computing the board grid of node {self.node} board {self.board}")
            self.failed.emit(self.generation, error_text(exc))
            return
        self.done.emit(self.generation, data)
