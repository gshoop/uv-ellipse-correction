"""Data/session layer of the GUI (plan section 9).

The widgets never touch the cache or the analysis modules directly: they go
through :class:`UVSession` (the pattern of specview's ``session.py``). The
session owns

- the opened file: the raw ``.dat`` path (or the one recorded in the cache),
  the :class:`~uvcorr.cache.UVCache`, its boards and the channels with events
  (computed once per open, :class:`OpenedFile`);
- the stored results (``/results/current``: batch rows plus overrides) and
  their merged view, one :class:`~uvcorr.analysis.ChannelResult` per channel;
- the fit options of the control band and the current channel selection;
- a small LRU of loaded boards (:data:`DEFAULT_BOARD_CACHE_SIZE` boards;
  a board is ~8 MB and up to ~45 MB).

Everything here is Qt-free apart from :class:`ChannelView` (a map record
whose module uses ``QColor``, a value type that needs no ``QApplication``),
so the session is unit tested without widgets.

Threads and cache access. Opening a file (:func:`load_raw`,
:func:`load_cache`), the Fit All batch (:meth:`UVSession.run_batch`) and the
scatter's channel detail (:meth:`UVSession.compute_detail`) are blocking and
run in the worker threads of :mod:`uvcorr.gui.threads`; everything that
changes the session's state (:meth:`UVSession.install`,
:meth:`UVSession.apply_batch`, :meth:`UVSession.store_override`) runs on the
GUI thread. No HDF5 handle is held between calls (the cache API opens the
file per call), but in-process read and write handles on one file conflict,
so every cache access of the session holds one lock (``_io_lock``). The one
exception is the read-only board loading inside
:func:`~uvcorr.analysis.analyze_all` (with ``workers=1`` it reads in-process
without the lock): concurrent reads are safe, and no session write can run
while a batch is running (:class:`SessionBusyError`). A cache that another
process holds open for writing raises
:class:`~uvcorr.cache.CacheBusyError` with a message meant for the user.

Channel detail (the scatter). ``ChannelResult`` does not store which points
the robust fit kept, so :meth:`UVSession.compute_detail` recomputes the mask
by re-running :func:`~uvcorr.ellipse.fit_ellipse` with the options the row
was fitted with (the batch options, or the override's own). The fit is
deterministic, so it reproduces the stored ellipse; the check allows for
last-bit BLAS differences and logs a warning if the parameters or the kept
count disagree. The drawn ellipse and correction always use the *stored*
parameters (what the exports contain). The refit takes ~0.2 s for the
largest real channel (946k events), which is why it runs in a thread.

Phase 5 hooks: :meth:`UVSession.refit_channel` (compute a channel with given
options), :meth:`UVSession.store_override` (persist a re-fit; the cache
requires stored batch results first) and :meth:`UVSession.merged_results`
(what the exports write).
"""

from __future__ import annotations

import bisect
import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from uvcorr.analysis import (
    OPTIONS_BATCH,
    OPTIONS_OVERRIDE,
    ChannelKey,
    ChannelResult,
    analyze_all,
    analyze_channel,
    default_workers,
)
from uvcorr.cache import (
    CACHE_SUFFIX,
    UV_CACHE_VERSION,
    BoardUV,
    BuildSettings,
    ProgressCallback,
    ResultsError,
    StopFlag,
    StoredOverride,
    StoredResults,
    UVCache,
    UVCacheError,
    default_cache_path,
    open_or_build,
)
from uvcorr.channels import electrode_label, is_active_channel, polarity_name
from uvcorr.ellipse import EllipseParams, correct, fit_ellipse
from uvcorr.gui._system_map_model import ChannelView
from uvcorr.options import (
    FLAG_GAUSS_FIT_FAILED_PRE,
    STATUS_FIT_FAILED,
    STATUS_OK,
    STATUS_TOO_FEW_EVENTS,
    FitOptions,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BOARD_CACHE_SIZE",
    "INFORMATIONAL_FLAGS",
    "MAP_METRIC_FIELDS",
    "BatchOutcome",
    "ChannelDetail",
    "DetailRequest",
    "OpenedFile",
    "RawFileCheck",
    "NoEventsError",
    "SessionBusyError",
    "SessionError",
    "UVSession",
    "channel_view",
    "channel_title",
    "inspect_raw",
    "is_cache_file",
    "load_cache",
    "load_raw",
]

DEFAULT_BOARD_CACHE_SIZE = 4
"""Boards kept in memory by the session's LRU (a board is ~8 MB, at most ~45 MB)."""

INFORMATIONAL_FLAGS: frozenset[str] = frozenset({FLAG_GAUSS_FIT_FAILED_PRE})
"""Flags that do not make an ``ok`` channel "flagged" in the GUI's status colours.

The radii of an *uncorrected* ellipse about its centre are not Gaussian (they
spread between b and a), so a failed pre-correction Gaussian fit says nothing
about the correction: on the full test acquisition 1,093 of 6,115 ok channels
carry it. The flag is still listed in the tooltips and the Fit Inspector."""

MAP_METRIC_FIELDS: tuple[str, ...] = (
    "n_events",
    "post_sigma",
    "timing_jitter_ns",
    "phase_ks",
    "axis_ratio",
    "n_rejected",
)
"""Result fields copied into a :class:`ChannelView`'s metrics (the map derives
``rejected_fraction`` from ``n_rejected / n_events``)."""

# RENA/channel number above any real one (anchors "after the last channel of a board")
_AFTER_ANY = 1 << 16

# Tolerances of the display refit check (the refit runs the same code on the
# same events; only BLAS round-off can differ).
_PARAM_RTOL = 1e-6
_PHI_ATOL = 1e-6

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

BatchProgressCallback = Callable[[int, int], None]
"""``progress_cb(done, total)`` of :meth:`UVSession.run_batch`, passed through to
:func:`~uvcorr.analysis.analyze_all` (only the ratio is meaningful to the GUI)."""


class SessionError(Exception):
    """A session operation cannot run in the session's current state."""


class SessionBusyError(SessionError):
    """A Fit All batch is running; the operation would conflict with it."""


class NoEventsError(SessionError):
    """The opened file holds no events on active channels (e.g. it is not raw data)."""


# ---------------------------------------------------------------------------
# Opening files
# ---------------------------------------------------------------------------


def is_cache_file(path: str | Path) -> bool:
    """Whether ``path`` names a UV cache (``*.uv.h5`` / ``*.h5`` / ``*.hdf5``) rather than raw data."""
    name = Path(path).name.lower()
    return name.endswith(CACHE_SUFFIX) or name.endswith((".h5", ".hdf5"))


@dataclass(frozen=True)
class RawFileCheck:
    """What opening a raw ``.dat`` file will do (:func:`inspect_raw`).

    Attributes:
        dat_path: The raw file.
        cache_path: The cache it opens or builds.
        valid: The existing cache is valid for the raw file (reused).
        cache_exists: A file exists at ``cache_path``.
        has_results: The existing cache stores ``/results``; a rebuild
            (``not valid``) discards them.
    """

    dat_path: Path
    cache_path: Path
    valid: bool
    cache_exists: bool
    has_results: bool

    @property
    def needs_build(self) -> bool:
        """Opening builds (or rebuilds) the cache."""
        return not self.valid

    @property
    def discards_results(self) -> bool:
        """Opening rebuilds a cache that stores results (they are lost)."""
        return not self.valid and self.cache_exists and self.has_results


def inspect_raw(dat_path: str | Path, cache_path: str | Path | None = None) -> RawFileCheck:
    """Check whether opening ``dat_path`` reuses or (re)builds its cache.

    Fast (a stat and a hash of the first MiB): meant for the GUI thread,
    before a :class:`~uvcorr.gui.threads.CacheBuildThread` is started.

    Raises:
        FileNotFoundError: If ``dat_path`` does not exist.
        CacheBusyError: If another process has the cache open for writing.
    """
    dat = Path(dat_path)
    if not dat.is_file():
        raise FileNotFoundError(f"Raw data file not found: {dat}")
    cache = UVCache(cache_path if cache_path is not None else default_cache_path(dat))
    valid = cache.is_valid_for(dat)
    exists = cache.exists()
    has_results = cache.has_results() if exists and not valid else False
    return RawFileCheck(dat, cache.path, valid, exists, has_results)


@dataclass(frozen=True, eq=False)
class OpenedFile:
    """Everything the session reads when a file is opened (built in a worker thread).

    Attributes:
        cache: The UV cache.
        dat_path: The raw file: the one opened, or the ``source_path``
            recorded in the cache when a cache was opened directly.
        board_counts: ``{(node, board): n_events}`` of every cached board.
        channel_counts: ``{ChannelKey: n_events}`` of every channel with
            events (from the stored results where they cover a board, else
            from a scan of the board).
        stored: The stored results and overrides, or None.
        built: The cache was (re)built by this open.
        stale: When a cache was opened directly: True if its source file
            exists but has changed since the build, False if it matches,
            None if the source file is missing. Always False for a raw open.
        metadata: The cache's ``/metadata`` attributes.
        seconds: Wall time of the open (build included).
        results_error: Why the stored results could not be read (they are
            then ignored: ``stored`` is None and a new Fit All replaces
            them), or None.
    """

    cache: UVCache
    dat_path: Path | None
    board_counts: Mapping[tuple[int, int], int]
    channel_counts: Mapping[ChannelKey, int]
    stored: StoredResults | None
    built: bool
    stale: bool | None
    metadata: Mapping[str, Any]
    seconds: float
    results_error: str | None = None


def load_raw(
    dat_path: str | Path,
    cache_path: str | Path | None = None,
    *,
    force: bool = False,
    progress_cb: ProgressCallback | None = None,
    stop_flag: StopFlag | None = None,
    settings: BuildSettings | None = None,
) -> OpenedFile:
    """Open a raw ``.dat`` file: reuse its valid cache or build it, then read the index.

    Blocking; run it in a worker thread. See
    :func:`~uvcorr.cache.open_or_build` for the build. A file without any
    event on an active channel (not raw data, an empty acquisition) is
    refused, and a cache this call built for it is removed again, so no
    empty "valid" cache is left behind.

    Raises:
        FileNotFoundError: If ``dat_path`` does not exist.
        NoEventsError: If the file yields no events.
        CacheBuildCancelled: If ``stop_flag`` stopped a build.
        CacheBusyError, CacheBuildError, ResultsError: See ``open_or_build``
            and :meth:`~uvcorr.cache.UVCache.load_results`.
    """
    t_start = time.perf_counter()
    dat = Path(dat_path)
    cache = open_or_build(
        dat,
        cache_path,
        force=force,
        progress_cb=progress_cb,
        stop_flag=stop_flag,
        settings=settings,
    )
    built = cache.last_build is not None
    _require_events(cache, dat, remove_if_empty=built)
    return _read_index(cache, dat, built, False, t_start)


def load_cache(cache_path: str | Path) -> OpenedFile:
    """Open a UV cache directly (no source-file validation or rebuild) and read its index.

    Blocking; run it in a worker thread.

    Raises:
        FileNotFoundError: If the file does not exist.
        UVCacheError: If it is not a UV cache of this layout version.
        NoEventsError: If the cache holds no events.
        CacheBusyError: If another process holds the cache's lock.
    """
    t_start = time.perf_counter()
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"UV cache not found: {path}")
    cache = UVCache(path)
    try:
        metadata = cache.metadata()
    except (OSError, KeyError) as exc:
        raise UVCacheError(f"{path} is not a uvcorr UV cache: {exc}") from exc
    version = metadata.get("uv_cache_version")
    if version != UV_CACHE_VERSION:
        raise UVCacheError(
            f"{path} has UV cache layout {version!r}, this uvcorr reads {UV_CACHE_VERSION!r}; "
            "open the raw .dat file to rebuild it"
        )
    source = metadata.get("source_path")
    dat = Path(source) if isinstance(source, str) and source else None
    _require_events(cache, dat, remove_if_empty=False)
    stale: bool | None = None
    if dat is not None and dat.is_file():
        stale = not cache.is_valid_for(dat)
    return _read_index(cache, dat, False, stale, t_start)


def _require_events(cache: UVCache, source: Path | None, *, remove_if_empty: bool) -> None:
    """Raise :class:`NoEventsError` if the cache holds no events (optionally deleting it)."""
    if cache.board_event_counts():
        return
    metadata = cache.metadata()
    name = source.name if source is not None else cache.path.name
    if metadata.get("parser_frames") == 0:
        message = f"No valid frames in {name}: is this a raw .dat file?"
    else:
        message = f"{name} holds no events on active channels"
    if remove_if_empty:
        try:
            cache.path.unlink()
            logger.info(f"Removed the empty UV cache {cache.path}")
        except OSError as exc:
            logger.warning(f"Could not remove the empty UV cache {cache.path}: {exc}")
    raise NoEventsError(message)


def _read_index(
    cache: UVCache, dat: Path | None, built: bool, stale: bool | None, t_start: float
) -> OpenedFile:
    metadata = cache.metadata()
    board_counts = cache.board_event_counts()
    results_error: str | None = None
    try:
        stored = cache.load_results()
    except ResultsError as exc:
        # The events are still usable: open without results, Fit All replaces them
        logger.warning(f"Ignoring the stored results of {cache.path}: {exc}")
        stored, results_error = None, str(exc)
    channel_counts = _index_channels(cache, board_counts, stored)
    return OpenedFile(
        cache=cache,
        dat_path=dat,
        board_counts=board_counts,
        channel_counts=channel_counts,
        stored=stored,
        built=built,
        stale=stale,
        metadata=metadata,
        seconds=time.perf_counter() - t_start,
        results_error=results_error,
    )


def _index_channels(
    cache: UVCache,
    board_counts: Mapping[tuple[int, int], int],
    stored: StoredResults | None,
) -> dict[ChannelKey, int]:
    """Channels with events and their counts.

    Stored results hold one row per active channel with events (and the
    cache holds only active channels), so where a board's rows add up to its
    event count they are the index; any other board is scanned (~15 ms per
    board; ~2 s for the 155 boards of the full test file).
    """
    counts: dict[ChannelKey, int] = {}
    per_board: dict[tuple[int, int], dict[ChannelKey, int]] = {}
    if stored is not None:
        for result in stored.merged():
            per_board.setdefault((result.node, result.board), {})[result.key] = result.n_events
    n_scanned = 0
    for (node, board), n_events in board_counts.items():
        rows = per_board.get((node, board), {})
        if sum(rows.values()) == n_events:
            counts.update(rows)
            continue
        n_scanned += 1
        for rena, channel, n in cache.channels(node, board):
            counts[ChannelKey(node, board, rena, channel)] = n
    if n_scanned:
        logger.info(f"Scanned {n_scanned} board(s) of {cache.path} for their channels")
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Results as map views
# ---------------------------------------------------------------------------


def _compact_count(n: int) -> str:
    """``208267950`` -> ``"208.3M"``, ``12345`` -> ``"12,345"``."""
    if n >= 10_000_000:
        return f"{n / 1e6:.1f}M"
    return f"{n:,}"


def channel_view(result: ChannelResult) -> ChannelView:
    """The System Map record of one result (status, flags, source, metrics)."""
    metrics: dict[str, float] = {}
    for name in MAP_METRIC_FIELDS:
        value = getattr(result, name)
        if value is not None:
            metrics[name] = float(value)
    return ChannelView(
        status=result.status,
        flags=result.flags,
        options_source=result.options_source,
        metrics=metrics,
    )


def channel_title(key: ChannelKey | tuple[int, int, int, int]) -> str:
    """Short channel name, e.g. ``"N2 B16 R0 Ch27 (C04)"`` (electrode when active)."""
    node, board, rena, channel = (int(k) for k in key)
    text = f"N{node} B{board} R{rena} Ch{channel:02d}"
    if is_active_channel(rena, channel):
        text += f" ({electrode_label(board, rena, channel)})"
    return text


# ---------------------------------------------------------------------------
# Channel detail (scatter data)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetailRequest:
    """A snapshot of what :meth:`UVSession.compute_detail` needs (taken on the GUI thread).

    Attributes:
        key: The channel.
        cache: The cache to read it from.
        result: Its merged result, or None.
        options: The options the result was fitted with, or None.
        has_results: Whether the session has stored results at all.
    """

    key: ChannelKey
    cache: UVCache
    result: ChannelResult | None
    options: FitOptions | None
    has_results: bool


@dataclass(frozen=True, eq=False)
class ChannelDetail:
    """One channel's points and fit, ready to draw.

    Attributes:
        key: The channel.
        u, v: The raw points (float64, file order).
        result: The merged result, or None.
        options: The options the result was fitted with, or None.
        params: The stored ellipse (None unless the status is ``ok``).
        kept: The points the robust fit used (recomputed; None when there is
            no ellipse or the refit did not succeed).
        u_corr, v_corr: The corrected points (None without an ellipse).
        consistent: The refit reproduced the stored ellipse and kept count
            (True when there was nothing to check).
        message: A note for the user (why there is no ellipse, or a
            refit/count mismatch); empty when all is well.
        seconds: Time spent computing the detail.
    """

    key: ChannelKey
    u: FloatArray
    v: FloatArray
    result: ChannelResult | None
    options: FitOptions | None
    params: EllipseParams | None
    kept: BoolArray | None
    u_corr: FloatArray | None
    v_corr: FloatArray | None
    consistent: bool
    message: str
    seconds: float

    @property
    def n_events(self) -> int:
        """Number of points."""
        return int(self.u.shape[0])

    @property
    def n_rejected(self) -> int:
        """Points the fit rejected (0 without a mask)."""
        if self.kept is None:
            return 0
        return self.n_events - int(np.count_nonzero(self.kept))


def _params_agree(refit: EllipseParams, stored: EllipseParams) -> bool:
    scale = max(stored.a, 1.0)
    for got, want in (
        (refit.cx, stored.cx),
        (refit.cy, stored.cy),
        (refit.a, stored.a),
        (refit.b, stored.b),
    ):
        if not abs(got - want) <= _PARAM_RTOL * scale:
            return False
    if stored.a - stored.b <= _PARAM_RTOL * scale:
        return True  # a circle: phi is undefined
    diff = math.remainder(refit.phi - stored.phi, math.pi)
    return abs(diff) <= _PHI_ATOL


def _no_fit_message(request: DetailRequest, n_points: int) -> str:
    result = request.result
    if n_points == 0 and (result is None or result.n_events == 0):
        return "No events on this channel (no data)."
    if result is None:
        if request.has_results:
            return "No stored result for this channel: raw points only."
        return "Not fitted yet: raw points only. Run Fit All to fit every channel."
    if result.status == STATUS_TOO_FEW_EVENTS:
        min_events = request.options.min_events if request.options is not None else None
        limit = f" < min events {min_events}" if min_events is not None else ""
        return f"Too few events ({n_points:,}{limit}): no ellipse was fitted, raw points only."
    if result.status == STATUS_FIT_FAILED:
        return "The ellipse fit failed (no ellipse fits these points): raw points only."
    return "No ellipse parameters stored for this channel: raw points only."


def _build_detail(
    request: DetailRequest, u16: npt.NDArray[Any], v16: npt.NDArray[Any], t_start: float
) -> ChannelDetail:
    key = request.key
    u = np.asarray(u16, dtype=np.float64)
    v = np.asarray(v16, dtype=np.float64)
    result = request.result
    params = result.params if result is not None and result.status == STATUS_OK else None
    notes: list[str] = []
    if result is not None and result.n_events != u.shape[0]:
        notes.append(
            f"The cache holds {u.shape[0]:,} events but the result counts {result.n_events:,}."
        )
        logger.warning(f"{key}: {notes[-1]}")
    if params is None or result is None:
        notes.insert(0, _no_fit_message(request, int(u.shape[0])))
        return ChannelDetail(
            key=key,
            u=u,
            v=v,
            result=result,
            options=request.options,
            params=None,
            kept=None,
            u_corr=None,
            v_corr=None,
            consistent=True,
            message=" ".join(notes),
            seconds=time.perf_counter() - t_start,
        )

    kept: BoolArray | None = None
    consistent = True
    fit = fit_ellipse(u, v, request.options)
    if fit.ok and fit.params is not None:
        kept = fit.mask_used
        if not _params_agree(fit.params, params) or fit.n_used != result.n_used:
            consistent = False
            logger.warning(
                f"{key}: re-running the fit for the scatter gave {fit.params} with "
                f"{fit.n_used:,} points used, the stored result has {params} with "
                f"{result.n_used} used; the stored ellipse is drawn"
            )
    else:
        consistent = False
        logger.warning(
            f"{key}: re-running the fit for the scatter gave status {fit.status!r} "
            f"(stored: {result.status!r}); rejected points cannot be shown"
        )
    if not consistent:
        notes.append(
            "Re-running the fit to find the rejected points did not reproduce the stored "
            "ellipse (see the log); the stored ellipse is drawn."
        )
    u_corr, v_corr = correct(u, v, params)
    return ChannelDetail(
        key=key,
        u=u,
        v=v,
        result=result,
        options=request.options,
        params=params,
        kept=kept,
        u_corr=u_corr,
        v_corr=v_corr,
        consistent=consistent,
        message=" ".join(notes),
        seconds=time.perf_counter() - t_start,
    )


# ---------------------------------------------------------------------------
# Batch outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class BatchOutcome:
    """Result of :meth:`UVSession.run_batch`.

    Attributes:
        cache_path: The cache the batch ran on.
        stored: The results as stored after the batch (overrides kept).
        options: The batch options.
        workers: Worker processes used.
        seconds: Wall time (analysis and storing).
    """

    cache_path: Path
    stored: StoredResults
    options: FitOptions
    workers: int
    seconds: float


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class UVSession:
    """The GUI's data layer: opened cache, results, options, selection and board LRU.

    See the module docstring for the threading rules. Methods documented as
    blocking may run in a worker thread; all others belong to the GUI thread.
    """

    def __init__(self, board_cache_size: int = DEFAULT_BOARD_CACHE_SIZE) -> None:
        """Args:
        board_cache_size: Boards kept in the LRU (at least 1).
        """
        if board_cache_size < 1:
            raise ValueError(f"board_cache_size must be >= 1, got {board_cache_size}")
        self._board_cache_size = board_cache_size
        self._io_lock = threading.RLock()
        self._batch_lock = threading.Lock()
        self._boards_lru: OrderedDict[tuple[str, int, int], BoardUV] = OrderedDict()
        self._opened: OpenedFile | None = None
        self._stored: StoredResults | None = None
        self._merged: dict[ChannelKey, ChannelResult] = {}
        self._channels: tuple[ChannelKey, ...] = ()
        self._options = FitOptions()
        self._selection: ChannelKey | None = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """Whether a file is open."""
        return self._opened is not None

    @property
    def opened(self) -> OpenedFile | None:
        """What was read when the current file was opened."""
        return self._opened

    @property
    def cache(self) -> UVCache | None:
        """The open cache."""
        return self._opened.cache if self._opened is not None else None

    @property
    def dat_path(self) -> Path | None:
        """The raw ``.dat`` file (as opened, or recorded in the cache)."""
        return self._opened.dat_path if self._opened is not None else None

    @property
    def stored(self) -> StoredResults | None:
        """The stored results and overrides, or None."""
        return self._stored

    @property
    def has_results(self) -> bool:
        """Whether results are stored."""
        return self._stored is not None

    @property
    def batch_options(self) -> FitOptions | None:
        """Options of the stored batch run."""
        return self._stored.options if self._stored is not None else None

    @property
    def options(self) -> FitOptions:
        """Fit options for the next fit (the control band's)."""
        return self._options

    @options.setter
    def options(self, options: FitOptions) -> None:
        self._options = options

    @property
    def selection(self) -> ChannelKey | None:
        """The selected channel."""
        return self._selection

    def select(self, key: ChannelKey | tuple[int, int, int, int] | None) -> ChannelKey | None:
        """Select a channel (or clear the selection); returns the normalised key."""
        self._selection = None if key is None else ChannelKey(*(int(k) for k in key))
        return self._selection

    @property
    def batch_running(self) -> bool:
        """Whether :meth:`run_batch` is running."""
        return self._batch_lock.locked()

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------

    def install(self, opened: OpenedFile) -> None:
        """Make ``opened`` the session's file (GUI thread): results, index, empty LRU.

        The fit options become the stored batch options (the defaults
        without results) and the selection is cleared.

        Raises:
            SessionBusyError: If a batch is running on the current file.
        """
        if self.batch_running:
            raise SessionBusyError("Fit All is running; stop it before opening another file")
        self._opened = opened
        self._channels = tuple(opened.channel_counts)
        self._set_stored(opened.stored)
        self._options = opened.stored.options if opened.stored is not None else FitOptions()
        self._selection = None
        with self._io_lock:
            self._boards_lru.clear()

    def close(self) -> None:
        """Forget the open file, results, selection and loaded boards."""
        if self.batch_running:
            raise SessionBusyError("Fit All is running; stop it before closing the file")
        self._opened = None
        self._channels = ()
        self._set_stored(None)
        self._selection = None
        with self._io_lock:
            self._boards_lru.clear()

    def _set_stored(self, stored: StoredResults | None) -> None:
        self._stored = stored
        merged = stored.merged() if stored is not None else []
        self._merged = {result.key: result for result in merged}

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    @property
    def boards(self) -> list[tuple[int, int]]:
        """``(node, board)`` of every cached board, sorted."""
        return list(self._opened.board_counts) if self._opened is not None else []

    @property
    def data_channels(self) -> tuple[ChannelKey, ...]:
        """Every channel with events, sorted."""
        return self._channels

    def channel_count(self, key: ChannelKey | tuple[int, int, int, int]) -> int:
        """Events of a channel (0 if it has none)."""
        if self._opened is None:
            return 0
        return int(self._opened.channel_counts.get(ChannelKey(*key), 0))

    def nodes(self) -> list[int]:
        """Nodes with data, sorted."""
        return sorted({node for node, _ in self.boards})

    def boards_on_node(self, node: int) -> list[int]:
        """Boards of ``node`` with data, sorted."""
        return [board for n, board in self.boards if n == node]

    def channels_on_board(self, node: int, board: int) -> list[ChannelKey]:
        """Channels of a board with events, sorted."""
        lo = bisect.bisect_left(self._channels, (node, board, -1, -1))
        hi = bisect.bisect_left(self._channels, (node, board + 1, -1, -1))
        return list(self._channels[lo:hi])

    def step_channel(
        self, key: ChannelKey | tuple[int, int, int, int] | None, delta: int
    ) -> ChannelKey | None:
        """The channel ``delta`` places from ``key`` in the sorted data channels.

        Clamped at both ends. From None (or a channel without data) a
        positive step lands on the first channel after ``key`` (the first
        channel overall for None), a negative one on the last before it.
        """
        channels = self._channels
        if not channels:
            return None
        if key is None:
            return channels[0] if delta >= 0 else channels[-1]
        wanted = ChannelKey(*key)
        index = bisect.bisect_left(channels, wanted)
        if index < len(channels) and channels[index] == wanted:
            target = index + delta
        elif delta >= 0:
            target = index + delta - 1
        else:
            target = index + delta
        return channels[max(0, min(len(channels) - 1, target))]

    def step_from_board(self, node: int, board: int, delta: int) -> ChannelKey | None:
        """The channel ``delta`` places from a board selected without a channel.

        Next (``delta > 0``) lands on the board's first channel with events
        (or the first one after the board), Prev on its last channel (or the
        last one before it); larger steps continue from there. Clamped at
        both ends.
        """
        if delta >= 0:
            return self.step_channel(ChannelKey(node, board, -1, -1), delta)
        return self.step_channel(ChannelKey(node, board, _AFTER_ANY, _AFTER_ANY), delta)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result(self, key: ChannelKey | tuple[int, int, int, int]) -> ChannelResult | None:
        """The merged result of a channel, or None."""
        return self._merged.get(ChannelKey(*key))

    def merged_results(self) -> list[ChannelResult]:
        """The batch results with the overrides applied, sorted (what the exports write)."""
        return list(self._merged.values())

    def options_for(
        self, key: ChannelKey | tuple[int, int, int, int]
    ) -> tuple[FitOptions, str] | None:
        """The options a channel's result was fitted with, and their source.

        Returns:
            ``(options, "override")`` for an override, ``(batch options,
            "batch")`` for a batch row, None without a result.
        """
        stored = self._stored
        result = self.result(key)
        if stored is None or result is None:
            return None
        override = stored.overrides.get(ChannelKey(*key))
        if override is not None and result.options_source == OPTIONS_OVERRIDE:
            return override.options, OPTIONS_OVERRIDE
        return stored.options, OPTIONS_BATCH

    def views(self) -> dict[ChannelKey, ChannelView]:
        """The System Map views of every result."""
        return {key: channel_view(result) for key, result in self._merged.items()}

    # ------------------------------------------------------------------
    # Event data (blocking; any thread)
    # ------------------------------------------------------------------

    def cached_boards(self) -> list[tuple[int, int]]:
        """``(node, board)`` held by the LRU, least recently used first."""
        with self._io_lock:
            return [(node, board) for _path, node, board in self._boards_lru]

    def board_data(self, node: int, board: int, cache: UVCache | None = None) -> BoardUV:
        """A board's events, from the LRU or loaded from the cache (blocking).

        Args:
            node: Node number.
            board: Board number.
            cache: The cache to read (default: the open one).

        Raises:
            SessionError: If no file is open.
            KeyError: If the cache has no events for the board.
            CacheBusyError: If another process holds the cache's lock.
        """
        cache = cache if cache is not None else self.cache
        if cache is None:
            raise SessionError("No file is open")
        lru_key = (str(cache.path), int(node), int(board))
        with self._io_lock:
            data = self._boards_lru.get(lru_key)
            if data is not None:
                self._boards_lru.move_to_end(lru_key)
                return data
            data = cache.load_board(node, board)
            self._boards_lru[lru_key] = data
            while len(self._boards_lru) > self._board_cache_size:
                self._boards_lru.popitem(last=False)
            return data

    def channel_data(
        self, key: ChannelKey | tuple[int, int, int, int], cache: UVCache | None = None
    ) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
        """(U, V) of one channel in file order (blocking; empty arrays if it has no events).

        Raises:
            SessionError, KeyError, CacheBusyError: See :meth:`board_data`.
        """
        node, board, rena, channel = (int(k) for k in key)
        return self.board_data(node, board, cache).channel_data(rena, channel)

    def detail_request(self, key: ChannelKey | tuple[int, int, int, int]) -> DetailRequest:
        """Snapshot what :meth:`compute_detail` needs for ``key`` (GUI thread).

        Raises:
            SessionError: If no file is open.
        """
        cache = self.cache
        if cache is None:
            raise SessionError("No file is open")
        channel = ChannelKey(*(int(k) for k in key))
        used = self.options_for(channel)
        return DetailRequest(
            key=channel,
            cache=cache,
            result=self.result(channel),
            options=used[0] if used is not None else self.batch_options,
            has_results=self.has_results,
        )

    def compute_detail(self, request: DetailRequest) -> ChannelDetail:
        """Load a channel and recompute its kept mask and corrected points (blocking).

        Raises:
            KeyError, CacheBusyError: See :meth:`board_data`.
        """
        t_start = time.perf_counter()
        u16, v16 = self.channel_data(request.key, request.cache)
        return _build_detail(request, u16, v16, t_start)

    def channel_detail(self, key: ChannelKey | tuple[int, int, int, int]) -> ChannelDetail:
        """:meth:`detail_request` and :meth:`compute_detail` in one call (blocking)."""
        return self.compute_detail(self.detail_request(key))

    # ------------------------------------------------------------------
    # Fit All (blocking part runs in a worker thread)
    # ------------------------------------------------------------------

    def run_batch(
        self,
        options: FitOptions,
        *,
        workers: int | None = None,
        progress_cb: BatchProgressCallback | None = None,
        stop_flag: StopFlag | None = None,
    ) -> BatchOutcome:
        """Fit every channel of the open cache and store the results (blocking).

        Runs :func:`~uvcorr.analysis.analyze_all` (a ``spawn`` process pool
        unless ``workers`` is 1), then stores the results as the new
        ``/results/current``, keeping the overrides, and reads them back.
        Does not change the session: pass the outcome to
        :meth:`apply_batch` on the GUI thread.

        Args:
            options: Batch fit options.
            workers: Worker processes (default
                :func:`~uvcorr.analysis.default_workers`).
            progress_cb: ``progress_cb(done, total)``, as ``analyze_all`` reports it.
            stop_flag: Stop request (callable or ``threading.Event``).

        Raises:
            SessionError: If no file is open.
            SessionBusyError: If a batch is already running.
            AnalysisCancelled: If ``stop_flag`` stopped it (nothing is stored).
            AnalysisError: If a board failed.
            CacheBusyError: If another process holds the cache's lock.
        """
        cache = self.cache
        if cache is None:
            raise SessionError("No file is open")
        if not self._batch_lock.acquire(blocking=False):
            raise SessionBusyError("Fit All is already running")
        try:
            t_start = time.perf_counter()
            n_workers = default_workers() if workers is None else int(workers)
            results = analyze_all(
                cache, options, workers=n_workers, progress_cb=progress_cb, stop_flag=stop_flag
            )
            with self._io_lock:
                cache.save_results(results, options, keep_overrides=True)
                stored = cache.load_results()
            if stored is None:  # pragma: no cover - save_results just wrote them
                raise ResultsError(f"{cache.path}: the saved results could not be read back")
            return BatchOutcome(
                cache_path=cache.path,
                stored=stored,
                options=options,
                workers=n_workers,
                seconds=time.perf_counter() - t_start,
            )
        finally:
            self._batch_lock.release()

    def apply_batch(self, outcome: BatchOutcome) -> dict[ChannelKey, ChannelView]:
        """Install the results of a finished batch (GUI thread); returns the new map views.

        Raises:
            SessionError: If the session has since switched to another cache.
        """
        cache = self.cache
        if cache is None or cache.path != outcome.cache_path:
            raise SessionError(
                f"The batch ran on {outcome.cache_path}, which is no longer the open cache"
            )
        self._set_stored(outcome.stored)
        return self.views()

    # ------------------------------------------------------------------
    # Phase 5 hooks: single-channel re-fits and overrides
    # ------------------------------------------------------------------

    def refit_channel(
        self, key: ChannelKey | tuple[int, int, int, int], options: FitOptions
    ) -> ChannelResult:
        """Fit one channel with ``options`` (blocking; nothing is stored).

        The result has ``options_source="batch"`` as returned by
        :func:`~uvcorr.analysis.analyze_channel`; :meth:`store_override`
        marks it as an override.

        Raises:
            SessionError, KeyError, CacheBusyError: See :meth:`board_data`.
            ValueError: If the channel is not active.
        """
        channel = ChannelKey(*(int(k) for k in key))
        u, v = self.channel_data(channel)
        return analyze_channel(channel, u, v, options)

    def store_override(
        self, result: ChannelResult, options: FitOptions
    ) -> dict[ChannelKey, ChannelView]:
        """Persist a channel re-fit as an override and apply it (GUI thread).

        Args:
            result: The re-fitted result (stored with ``options_source="override"``).
            options: The options it was fitted with.

        Returns:
            The changed map view (``{key: view}``) for
            ``SystemMapWidget.update_views``.

        Raises:
            SessionError: If no file is open.
            SessionBusyError: If a batch is running.
            ResultsError: If there are no stored batch results to attach the
                override to (run Fit All first).
            CacheBusyError: If another process holds the cache's lock.
        """
        cache = self.cache
        if cache is None:
            raise SessionError("No file is open")
        if self.batch_running:
            raise SessionBusyError("Fit All is running; wait for it to finish")
        stored = self._stored
        if stored is None:
            raise ResultsError("Overrides need stored batch results: run Fit All first")
        override = replace(result, options_source=OPTIONS_OVERRIDE)
        with self._io_lock:
            cache.save_override(override, options)
        overrides = dict(stored.overrides)
        overrides[override.key] = StoredOverride(override, options)
        self._set_stored(replace(stored, overrides=dict(sorted(overrides.items()))))
        return {override.key: channel_view(override)}

    # ------------------------------------------------------------------
    # Descriptions
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """A short summary of the open file for the status bar (the title names the file)."""
        opened = self._opened
        if opened is None:
            return "No file open"
        n_events = sum(opened.board_counts.values())
        parts = [
            f"{len(opened.board_counts)} boards",
            f"{len(self._channels):,} ch",
            f"{_compact_count(n_events)} events",
        ]
        if self._stored is not None:
            n_over = len(self._stored.overrides)
            text = f"fitted {self._stored.created_at[:16].replace('T', ' ')}"
            if n_over:
                text += f" + {n_over} override{'s' if n_over > 1 else ''}"
            parts.append(text)
        else:
            parts.append("not fitted")
        return " · ".join(parts)

    def channel_summary(self, key: ChannelKey | tuple[int, int, int, int]) -> str:
        """One line about a channel: name, polarity, events."""
        channel = ChannelKey(*(int(k) for k in key))
        text = channel_title(channel)
        if is_active_channel(channel.rena, channel.channel):
            text += f" · {polarity_name(channel.board, channel.rena, channel.channel)}"
        return f"{text} · {self.channel_count(channel):,} events"
