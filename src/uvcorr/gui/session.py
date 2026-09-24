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
:func:`load_cache`), the Fit All batch (:meth:`UVSession.run_batch`), a
channel or board re-fit (:meth:`UVSession.run_refit`) and the scatter's
channel detail (:meth:`UVSession.compute_detail`) are blocking and run in the
worker threads of :mod:`uvcorr.gui.threads`; everything that changes the
session's state (:meth:`UVSession.install`, :meth:`UVSession.apply_batch`,
:meth:`UVSession.apply_refit`, the override reverts) runs on the GUI thread.
A batch and a re-fit store results, so only one of them runs at a time
(:class:`SessionBusyError`), and no revert or open can run meanwhile. No HDF5
handle is held between calls (the cache API opens the file per call), but
in-process read and write handles on one file conflict, so every cache access
of the session holds one lock (``_io_lock``). The one exception is the
read-only board loading inside :func:`~uvcorr.analysis.analyze_all` (with
``workers=1`` it reads in-process without the lock): concurrent reads are
safe, and no session write can run while a batch is running. A cache that
another process holds open for writing raises
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

Re-fits and overrides (plan D7, 6.1). A channel or board is re-fitted with
the control band's options (:meth:`UVSession.refit_request` on the GUI
thread, :meth:`UVSession.run_refit` in a worker, :meth:`UVSession.apply_refit`
back on the GUI thread). The options are compared with the stored batch
options: when they differ, the new results are stored as overrides (one
cache write for a whole board, :meth:`~uvcorr.cache.UVCache.save_overrides`);
when they are equal, the re-fit reproduces the batch, so the channels'
overrides are deleted instead (revert to batch) and nothing is stored.
Re-fits need stored batch results (:class:`~uvcorr.cache.ResultsError`
otherwise). :meth:`UVSession.revert_overrides`, :meth:`UVSession.revert_board`
and :meth:`UVSession.clear_overrides` delete overrides directly.

Exports (plan D3): :meth:`UVSession.export_tec`, :meth:`UVSession.export_csv`
and :meth:`UVSession.export_outputs` write the merged results (batch rows
with the overrides applied, :meth:`UVSession.merged_results`) through the
atomic writers of :mod:`uvcorr.io`.
"""

from __future__ import annotations

import bisect
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from uvcorr.analysis import (
    OPTIONS_BATCH,
    OPTIONS_OVERRIDE,
    AnalysisCancelled,
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
    StaleResultsError,
    StopFlag,
    StoredOverride,
    StoredResults,
    UVCache,
    UVCacheError,
    default_cache_path,
    no_valid_frames_message,
    open_or_build,
)
from uvcorr.channels import electrode_label, is_active_channel, polarity_name
from uvcorr.ellipse import EllipseParams, correct, fit_ellipse
from uvcorr.gui._system_map_model import ChannelView
from uvcorr.io.export import output_paths, prepare_output_dir, write_outputs
from uvcorr.io.summary_csv import write_summary_csv
from uvcorr.io.tec import write_tec
from uvcorr.options import (
    FLAG_GAUSS_FIT_FAILED_PRE,
    ROBUST_ONLY_FIELDS,
    STATUS_FIT_FAILED,
    STATUS_OK,
    STATUS_TOO_FEW_EVENTS,
    FitOptions,
    effective_options,
    same_fit,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BOARD_CACHE_SIZE",
    "INFORMATIONAL_FLAGS",
    "MAP_METRIC_FIELDS",
    "BatchOutcome",
    "ChannelDetail",
    "DetailRequest",
    "ExportSummary",
    "OpenedFile",
    "RawFileCheck",
    "NoEventsError",
    "RefitOutcome",
    "RefitRequest",
    "RevertOutcome",
    "RevertRequest",
    "STALE_RESULTS_MESSAGE",
    "SessionBusyError",
    "SessionError",
    "UVSession",
    "channel_view",
    "channel_title",
    "describe_options_change",
    "effective_options",  # re-exported from uvcorr.options
    "inspect_raw",
    "is_cache_file",
    "load_cache",
    "load_raw",
    "same_fit",  # re-exported from uvcorr.options
    "short_title",
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


STALE_RESULTS_MESSAGE = "stored results changed on disk (another process?); reopen the file"
"""The error when the cache's batch results are not the ones the session loaded."""


@contextmanager
def _stale_as_session_error(cache: UVCache) -> Iterator[None]:
    """Turn a :class:`~uvcorr.cache.StaleResultsError` into a :class:`SessionError`."""
    try:
        yield
    except StaleResultsError as exc:
        logger.warning(str(exc))
        raise SessionError(f"{cache.path.name}: {STALE_RESULTS_MESSAGE}") from exc


def _same_file(path: Path, other: Path) -> bool:
    """Whether two paths name the same file (symbolic or hard links included)."""
    try:
        if path.resolve() == other.resolve():
            return True
        return path.exists() and other.exists() and os.path.samefile(path, other)
    except OSError:
        return False


def _export_error(exc: OSError, targets: list[Path]) -> OSError:
    """An export's OSError, reworded to name the target instead of a temporary file.

    The writers go through hidden temporary files (and the directory check
    through a probe file), whose names mean nothing to the user.
    """
    cause = exc.__cause__ if isinstance(exc.__cause__, OSError) else exc
    reason = cause.strerror or str(exc)
    if cause.strerror is None and not cause.filename:
        return OSError(str(exc))  # already worded for the user (e.g. "... is a directory")
    names = {str(target) for target in targets}
    if cause.filename2 is not None and str(cause.filename2) in names:
        where = str(cause.filename2)
    elif len(targets) == 1:
        where = str(targets[0])
    else:
        where = f"to the directory {targets[0].parent}"
    return OSError(f"Cannot write {where}: {reason}")


def _stop_callable(stop_flag: StopFlag | None) -> Callable[[], bool]:
    """A ``stop_flag`` (None, a callable or a ``threading.Event``) as a callable."""
    if stop_flag is None:
        return lambda: False
    if isinstance(stop_flag, threading.Event):
        return stop_flag.is_set
    return stop_flag


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
    valid frame (not raw data, an empty file) fails the build itself
    (:class:`~uvcorr.cache.CacheBuildError`, no cache is written). A raw
    file without any event on an active channel is refused too, and a cache
    this call built for it is removed again, so no empty "valid" cache is
    left behind.

    Raises:
        FileNotFoundError: If ``dat_path`` does not exist.
        NoEventsError: If the file (or its reused cache) holds no events on
            active channels.
        CacheBuildCancelled: If ``stop_flag`` stopped a build.
        CacheBuildError: If the build failed, e.g. ``"no valid frames in
            <file>: is this a raw .dat file?"``.
        CacheBusyError, ResultsError: See ``open_or_build`` and
            :meth:`~uvcorr.cache.UVCache.load_results`.
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
    """Raise :class:`NoEventsError` if the cache holds no events (optionally deleting it).

    A build no longer writes a cache for a file without valid frames (it
    raises :class:`~uvcorr.cache.CacheBuildError`) and such a cache is never
    reused (:meth:`~uvcorr.cache.UVCache.is_valid_for`), but one built by an
    earlier uvcorr can still be opened directly (:func:`load_cache`); it gets
    the build's message.
    """
    if cache.board_event_counts():
        return
    metadata = cache.metadata()
    if metadata.get("parser_frames") == 0:
        message = no_valid_frames_message(source if source is not None else cache.path)
    else:
        name = source.name if source is not None else cache.path.name
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
    text = short_title((node, board, rena, channel))
    if is_active_channel(rena, channel):
        text += f" ({electrode_label(board, rena, channel)})"
    return text


def short_title(key: ChannelKey | tuple[int, int, int, int]) -> str:
    """Channel name without the electrode, e.g. ``"N2 B16 R0 Ch27"`` (status messages)."""
    node, board, rena, channel = (int(k) for k in key)
    return f"N{node} B{board} R{rena} Ch{channel:02d}"


def _on_off(value: Any) -> str:
    return "on" if value else "off"


_OPTION_CHANGE_TEXT: dict[str, Callable[[Any], str]] = {
    "robust": lambda value: f"robust {_on_off(value)}",
    "clip_k": lambda value: f"clip k {value:g}",
    "max_iter": lambda value: f"max iter {value}",
    "geometric": lambda value: f"geometric {_on_off(value)}",
    "min_events": lambda value: f"min events {value:,}",
    "phase_ref_freq_hz": lambda value: f"ref freq {value:g} Hz",
    "high_rejection_frac": lambda value: f"high rejection > {value:g}",
    "extreme_axis_ratio": lambda value: f"axis ratio < {value:g}",
    "broad_ring_frac": lambda value: f"broad ring > {value:g}",
}


def describe_options_change(options: FitOptions, reference: FitOptions) -> str:
    """The effective options that differ from ``reference``, e.g. ``"robust off, clip k 3"``.

    Both sides are compared as :func:`~uvcorr.options.effective_options`,
    and clip k and max iter are not listed when ``options`` has robust off.

    Returns:
        The changed options in the control band's order (robust, clip k,
        max iter, geometric, min events, then the others), or ``""`` when
        they fit identically.
    """
    before = effective_options(reference).to_dict()
    after = effective_options(options).to_dict()
    # The control band's order first, then any other option
    names = [*_OPTION_CHANGE_TEXT, *(name for name in after if name not in _OPTION_CHANGE_TEXT)]
    if not options.robust:  # clip k and max iter mean nothing without the robust iteration
        names = [name for name in names if name not in ROBUST_ONLY_FIELDS]
    parts: list[str] = []
    for name in names:
        value = after[name]
        if value != before.get(name):
            text = _OPTION_CHANGE_TEXT.get(name)
            parts.append(text(value) if text is not None else f"{name} {value}")
    return ", ".join(parts)


def _plural(n: int, noun: str) -> str:
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


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
        stored: The results as stored after the batch.
        options: The batch options.
        workers: Worker processes used.
        seconds: Wall time (analysis and storing).
        keep_overrides: Whether the overrides were kept (else discarded).
        n_dropped: Stored overrides not carried over: all of them when
            discarded, else those fitted with the new batch options.
    """

    cache_path: Path
    stored: StoredResults
    options: FitOptions
    workers: int
    seconds: float
    keep_overrides: bool = True
    n_dropped: int = 0


# ---------------------------------------------------------------------------
# Re-fits (channel or board) and exports
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RefitRequest:
    """A snapshot of what :meth:`UVSession.run_refit` needs (taken on the GUI thread).

    Attributes:
        cache: The cache to read the events from and store the results in.
        node, board: The board.
        channel: The channel of a channel re-fit; None re-fits every channel
            of the board.
        options: The fit options (the control band's).
        stored: The stored results when the re-fit was requested: their
            batch options decide between storing overrides and reverting.
    """

    cache: UVCache
    node: int
    board: int
    channel: ChannelKey | None
    options: FitOptions
    stored: StoredResults

    @property
    def is_board(self) -> bool:
        """Whether the whole board is re-fitted."""
        return self.channel is None

    @property
    def title(self) -> str:
        """``"N2 B16 R0 Ch27"`` for a channel, ``"N2 B16"`` for a board."""
        if self.channel is not None:
            return short_title(self.channel)
        return f"N{self.node} B{self.board}"

    @property
    def as_override(self) -> bool:
        """The options fit differently from the batch options: the results become overrides.

        Compared as :func:`~uvcorr.options.effective_options` (with robust
        off, clip k and max iter do not matter).
        """
        return not same_fit(self.options, self.stored.options)

    @property
    def n_reverted(self) -> int:
        """Overrides a re-fit with the batch options deletes (0 for an override re-fit)."""
        if self.as_override:
            return 0
        if self.channel is not None:
            return int(self.channel in self.stored.overrides)
        return sum(1 for key in self.stored.overrides if (key.node, key.board) == self.board_key)

    @property
    def board_key(self) -> tuple[int, int]:
        """``(node, board)``."""
        return self.node, self.board

    def describe_start(self) -> str:
        """The status-bar text while it runs, e.g. ``"Fit Channel N2 B16 R0 Ch27 (robust off)…"``."""
        what = "Fit Board" if self.is_board else "Fit Channel"
        if self.as_override:
            change = describe_options_change(self.options, self.stored.options)
            return f"{what} {self.title} ({change})…"
        n = self.n_reverted
        if self.is_board:
            reverts = f"reverts {_plural(n, 'override')}" if n else "no overrides to revert"
        else:
            reverts = "reverts its override" if n else "no override to revert"
        return f"{what} {self.title} with the batch options ({reverts})…"


@dataclass(frozen=True, eq=False)
class RefitOutcome:
    """Result of :meth:`UVSession.run_refit` (already stored in the cache).

    Attributes:
        request: The request.
        results: The re-fitted results, sorted; ``options_source`` is
            ``"override"`` for those stored as overrides.
        stored: The stored results after the re-fit, for
            :meth:`UVSession.apply_refit`.
        saved: Channels whose re-fit was stored as an override.
        removed: Channels whose override was deleted (a re-fit with the
            batch options reverts to the batch).
        seconds: Wall time (fitting and storing).
    """

    request: RefitRequest
    results: tuple[ChannelResult, ...]
    stored: StoredResults
    saved: tuple[ChannelKey, ...]
    removed: tuple[ChannelKey, ...]
    seconds: float

    @property
    def changed(self) -> tuple[ChannelKey, ...]:
        """Channels whose merged result changed (sorted)."""
        return tuple(sorted(set(self.saved) | set(self.removed)))

    def describe(self) -> str:
        """A status-bar summary, e.g. ``"Override saved for N2 B16 R0 Ch27 (robust off)"``."""
        request = self.request
        title = request.title
        if request.as_override:
            change = describe_options_change(request.options, request.stored.options)
            if request.is_board:
                text = f"Overrides saved for the {_plural(len(self.saved), 'channel')} of {title}"
            else:
                text = f"Override saved for {title}"
            text += f" ({change})"
        else:
            n_removed = len(self.removed)
            if n_removed and request.is_board:
                text = f"{title} reverted to batch ({_plural(n_removed, 'override')} removed)"
            elif n_removed:
                text = f"{title} reverted to batch (override removed)"
            elif request.is_board:
                text = (
                    f"{title} re-fitted with the batch options: no overrides to remove, "
                    "nothing stored"
                )
            else:
                text = (
                    f"{title} re-fitted with the batch options: it matches the batch, "
                    "nothing stored"
                )
            if self.saved:  # channels without a batch row keep their re-fit
                text += (
                    f"; {_plural(len(self.saved), 'channel')} without a batch result "
                    "stored as override"
                )
        return f"{text}; {self.seconds:.1f} s"


@dataclass(frozen=True, eq=False)
class RevertRequest:
    """Overrides to delete (:meth:`UVSession.revert_request`, taken on the GUI thread).

    Attributes:
        cache: The cache to write.
        keys: The channels whose overrides are deleted (sorted, all with an
            override).
        stored: The stored results when the revert was requested.
        title: What is reverted, e.g. ``"N2 B16 R0 Ch27"``, ``"N2 B16"``, or
            ``""`` for every override.
    """

    cache: UVCache
    keys: tuple[ChannelKey, ...]
    stored: StoredResults
    title: str

    @property
    def is_all(self) -> bool:
        """Whether every override is deleted (Clear All Overrides)."""
        return not self.title


@dataclass(frozen=True, eq=False)
class RevertOutcome:
    """Result of :meth:`UVSession.run_revert` (already written to the cache).

    Attributes:
        request: The request.
        stored: The stored results after the revert, for
            :meth:`UVSession.apply_revert`.
        removed: The channels reverted (sorted).
        seconds: Wall time.
    """

    request: RevertRequest
    stored: StoredResults
    removed: tuple[ChannelKey, ...]
    seconds: float

    def describe(self) -> str:
        """A status-bar summary, e.g. ``"N2 B16 reverted to batch (3 overrides removed)"``."""
        request = self.request
        removed = _plural(len(self.removed), "override")
        if request.is_all:
            return f"All overrides cleared ({removed} removed; the channels are back to batch)"
        if len(request.keys) == 1 and request.title == short_title(request.keys[0]):
            return f"{request.title} reverted to batch (override removed)"
        return f"{request.title} reverted to batch ({removed} removed)"


@dataclass(frozen=True)
class ExportSummary:
    """What an export wrote (:meth:`UVSession.export_tec` and friends).

    Attributes:
        paths: The files written.
        n_rows: Channels in the results (CSV rows).
        n_ok: ``ok`` channels (``.tec`` blocks).
        n_overrides: Channels whose row comes from an override.
        seconds: Wall time of formatting and writing.
        warning: Set when the results stored in the cache changed on disk
            since they were loaded (the export wrote the results the session
            holds); empty otherwise.
    """

    paths: tuple[Path, ...]
    n_rows: int
    n_ok: int
    n_overrides: int
    seconds: float
    warning: str = ""


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
        # Held while Fit All or a re-fit runs (both store results); _busy_label names it.
        self._batch_lock = threading.Lock()
        self._busy_label: str | None = None
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
        """Whether :meth:`run_batch` or :meth:`run_refit` is running (both store results)."""
        return self._batch_lock.locked()

    def _busy_text(self) -> str:
        return f"{self._busy_label or 'Fit All'} is running"

    @contextmanager
    def _exclusive(self, label: str) -> Iterator[None]:
        """Hold the batch lock while a batch or re-fit runs (``label`` names it in errors)."""
        if not self._batch_lock.acquire(blocking=False):
            raise SessionBusyError(f"{self._busy_text()}; wait for it to finish")
        self._busy_label = label
        try:
            yield
        finally:
            self._busy_label = None
            self._batch_lock.release()

    def _check_not_busy(self, what: str) -> None:
        if self.batch_running:
            raise SessionBusyError(f"{self._busy_text()}; wait for it to finish before {what}")

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------

    def install(self, opened: OpenedFile) -> None:
        """Make ``opened`` the session's file (GUI thread): results, index, empty LRU.

        The fit options become the stored batch options (the defaults
        without results) and the selection is cleared.

        Raises:
            SessionBusyError: If a batch or re-fit is running on the current file.
        """
        if self.batch_running:
            raise SessionBusyError(f"{self._busy_text()}; stop it before opening another file")
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
            raise SessionBusyError(f"{self._busy_text()}; stop it before closing the file")
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

    def view(self, key: ChannelKey | tuple[int, int, int, int]) -> ChannelView | None:
        """The System Map view of one channel's merged result, or None without a result."""
        result = self.result(key)
        return channel_view(result) if result is not None else None

    def override_keys(self, node: int | None = None, board: int | None = None) -> list[ChannelKey]:
        """Channels with an override, sorted: all of them, or those of one board."""
        stored = self._stored
        if stored is None:
            return []
        keys = list(stored.overrides)
        if node is not None and board is not None:
            keys = [key for key in keys if (key.node, key.board) == (node, board)]
        return sorted(keys)

    def override_options(self, key: ChannelKey | tuple[int, int, int, int]) -> FitOptions | None:
        """The options of a channel's override, or None if it has none."""
        stored = self._stored
        if stored is None:
            return None
        override = stored.overrides.get(ChannelKey(*(int(k) for k in key)))
        return override.options if override is not None else None

    def override_change(self, key: ChannelKey | tuple[int, int, int, int]) -> str | None:
        """How a channel's override options differ from the batch options (e.g. ``"robust off"``).

        Returns:
            None if the channel has no override; ``"batch options"`` for an
            override fitted with options equal to the current batch options
            (e.g. after a Fit All with those options that kept it).
        """
        options = self.override_options(key)
        stored = self._stored
        if options is None or stored is None:
            return None
        return describe_options_change(options, stored.options) or "batch options"

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
        keep_overrides: bool = True,
        workers: int | None = None,
        progress_cb: BatchProgressCallback | None = None,
        stop_flag: StopFlag | None = None,
    ) -> BatchOutcome:
        """Fit every channel of the open cache and store the results (blocking).

        Runs :func:`~uvcorr.analysis.analyze_all` (a ``spawn`` process pool
        unless ``workers`` is 1), then stores the results as the new
        ``/results/current``, keeping or discarding the overrides, and reads
        them back. Kept overrides fitted with the new batch options
        (:func:`~uvcorr.options.same_fit`) are dropped in the same write: the
        new batch rows reproduce them (``uvcorr process`` applies the same
        rule). Does not change the session: pass the outcome to
        :meth:`apply_batch` on the GUI thread.

        Args:
            options: Batch fit options.
            keep_overrides: Keep the stored overrides (else they are discarded).
            workers: Worker processes (default
                :func:`~uvcorr.analysis.default_workers`).
            progress_cb: ``progress_cb(done, total)``, as ``analyze_all`` reports it.
            stop_flag: Stop request (callable or ``threading.Event``).

        Raises:
            SessionError: If no file is open.
            SessionBusyError: If a batch or re-fit is already running.
            AnalysisCancelled: If ``stop_flag`` stopped it (nothing is stored).
            AnalysisError: If a board failed.
            CacheBusyError: If another process holds the cache's lock.
        """
        cache = self.cache
        if cache is None:
            raise SessionError("No file is open")
        with self._exclusive("Fit All"):
            t_start = time.perf_counter()
            n_workers = default_workers() if workers is None else int(workers)
            results = analyze_all(
                cache, options, workers=n_workers, progress_cb=progress_cb, stop_flag=stop_flag
            )
            with self._io_lock:
                n_dropped = cache.save_results(
                    results,
                    options,
                    keep_overrides=keep_overrides,
                    drop_overrides=lambda _key, used: same_fit(used, options),
                )
                stored = cache.load_results()
            if stored is None:  # pragma: no cover - save_results just wrote them
                raise ResultsError(f"{cache.path}: the saved results could not be read back")
            return BatchOutcome(
                cache_path=cache.path,
                stored=stored,
                options=options,
                workers=n_workers,
                seconds=time.perf_counter() - t_start,
                keep_overrides=keep_overrides,
                n_dropped=n_dropped,
            )

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
    # Re-fits (request on the GUI thread, run in a worker, apply on the GUI thread)
    # ------------------------------------------------------------------

    def refit_request(
        self,
        options: FitOptions,
        *,
        channel: ChannelKey | tuple[int, int, int, int] | None = None,
        board: tuple[int, int] | None = None,
    ) -> RefitRequest:
        """Snapshot a channel or board re-fit for :meth:`run_refit` (GUI thread).

        Args:
            options: The fit options.
            channel: The channel to re-fit, or
            board: ``(node, board)`` to re-fit every channel of the board.

        Raises:
            ValueError: Unless exactly one of ``channel`` and ``board`` is given.
            SessionError: If no file is open, or the channel or board has no
                events.
            SessionBusyError: If a batch or re-fit is running.
            ResultsError: If there are no stored batch results (the re-fit
                would have nothing to override: run Fit All first).
        """
        if (channel is None) == (board is None):
            raise ValueError("refit_request needs a channel or a board, not both")
        cache = self.cache
        if cache is None:
            raise SessionError("No file is open")
        self._check_not_busy("re-fitting")
        stored = self._stored
        if stored is None:
            raise ResultsError(
                "Re-fits are stored as overrides of the batch results: run Fit All first"
            )
        if channel is not None:
            key = ChannelKey(*(int(k) for k in channel))
            if self.channel_count(key) == 0:
                raise SessionError(f"{short_title(key)} has no events to fit")
            return RefitRequest(cache, key.node, key.board, key, options, stored)
        node, board_number = (int(k) for k in board)  # type: ignore[union-attr]
        if not self.channels_on_board(node, board_number):
            raise SessionError(f"N{node} B{board_number} has no events to fit")
        return RefitRequest(cache, node, board_number, None, options, stored)

    def run_refit(
        self,
        request: RefitRequest,
        *,
        progress_cb: BatchProgressCallback | None = None,
        stop_flag: StopFlag | None = None,
    ) -> RefitOutcome:
        """Re-fit a channel or board in-process and store the outcome (blocking).

        Each channel is fitted with :func:`~uvcorr.analysis.analyze_channel`
        (a board is loaded once, through the session's board cache, and its
        channels are fitted one by one, as
        :func:`~uvcorr.analysis.analyze_board` does). Then:

        - options different from the batch options: the results are stored
          as overrides, in one cache write;
        - options that fit like the batch options
          (:func:`~uvcorr.options.same_fit`): the re-fit reproduces the
          batch, so the channels' overrides are deleted and nothing is
          stored. (A channel without a batch row, which only results written
          by a newer uvcorr can leave, keeps its re-fit as an override, in
          the same write.)

        The cache checks that the stored batch is still the one the request
        saw (another process may have replaced it), also when there is
        nothing to store or delete.

        Does not change the session: pass the outcome to :meth:`apply_refit`
        on the GUI thread.

        Args:
            request: From :meth:`refit_request`.
            progress_cb: ``progress_cb(channels_done, channels_total)``.
            stop_flag: Stop request (callable or ``threading.Event``), checked
                before each channel and once more before storing.

        Raises:
            SessionBusyError: If a batch or another re-fit is running.
            AnalysisCancelled: If ``stop_flag`` stopped it (nothing is stored).
            KeyError, CacheBusyError: See :meth:`board_data`.
            SessionError: If the stored results changed on disk since the
                request (nothing is written).
        """
        should_stop = _stop_callable(stop_flag)
        with self._exclusive("A re-fit"):
            t_start = time.perf_counter()
            fitted = self._fit_request(request, progress_cb, should_stop)
            stored = request.stored
            options = request.options
            if request.as_override:
                to_save = fitted
                to_remove: list[ChannelKey] = []
            else:
                batch_keys = {result.key for result in stored.results}
                to_save = [result for result in fitted if result.key not in batch_keys]
                to_remove = [
                    result.key
                    for result in fitted
                    if result.key in batch_keys and result.key in stored.overrides
                ]
            saved = [replace(result, options_source=OPTIONS_OVERRIDE) for result in to_save]
            if should_stop():
                raise AnalysisCancelled(f"Re-fit of {request.title} cancelled")
            with self._io_lock, _stale_as_session_error(request.cache):
                request.cache.replace_overrides(
                    saved,
                    options,
                    to_remove,
                    expected_created_at=request.stored.created_at,
                )
            overrides = {k: v for k, v in stored.overrides.items() if k not in set(to_remove)}
            for result in saved:
                overrides[result.key] = StoredOverride(result, options)
            by_key = {result.key: result for result in [*fitted, *saved]}
            return RefitOutcome(
                request=request,
                results=tuple(by_key[key] for key in sorted(by_key)),
                stored=replace(stored, overrides=dict(sorted(overrides.items()))),
                saved=tuple(result.key for result in saved),
                removed=tuple(sorted(to_remove)),
                seconds=time.perf_counter() - t_start,
            )

    def _fit_request(
        self,
        request: RefitRequest,
        progress_cb: BatchProgressCallback | None,
        should_stop: Callable[[], bool],
    ) -> list[ChannelResult]:
        """Fit the channel or every active channel of the board of ``request``."""
        report = progress_cb if progress_cb is not None else (lambda _done, _total: None)
        if request.channel is not None:
            report(0, 1)
            u, v = self.channel_data(request.channel, request.cache)
            if should_stop():
                raise AnalysisCancelled(f"Re-fit of {request.title} cancelled")
            result = analyze_channel(request.channel, u, v, request.options)
            report(1, 1)
            return [result]
        data = self.board_data(request.node, request.board, request.cache)
        channels = [(r, c) for r, c, _n in data.channels() if is_active_channel(r, c)]
        results: list[ChannelResult] = []
        report(0, len(channels))
        for done, (rena, channel) in enumerate(channels, start=1):
            if should_stop():
                raise AnalysisCancelled(f"Re-fit of {request.title} cancelled")
            u, v = data.channel_data(rena, channel)
            key = ChannelKey(request.node, request.board, rena, channel)
            results.append(analyze_channel(key, u, v, request.options))
            report(done, len(channels))
        return results

    def apply_refit(self, outcome: RefitOutcome) -> tuple[ChannelKey, ...]:
        """Install a finished re-fit (GUI thread).

        Returns:
            The channels whose merged result changed (see :meth:`view`).

        Raises:
            SessionError: If the session has since switched to another cache,
                or its stored results changed while the re-fit ran.
        """
        self._check_current(outcome.request.cache, outcome.request.stored, "re-fit")
        self._set_stored(outcome.stored)
        return outcome.changed

    # ------------------------------------------------------------------
    # Reverting overrides (GUI thread)
    # ------------------------------------------------------------------

    def revert_request(
        self,
        keys: Iterable[ChannelKey | tuple[int, int, int, int]] | None = None,
        *,
        board: tuple[int, int] | None = None,
        title: str | None = None,
    ) -> RevertRequest | None:
        """Snapshot a revert for :meth:`run_revert` (GUI thread).

        Args:
            keys: The channels to revert; None with no ``board``: every
                override (Clear All Overrides).
            board: ``(node, board)``: every override of the board.
            title: What is reverted, for the status bar (default: the
                channel, the board, or "" for every override).

        Returns:
            The request, or None if none of the channels has an override.

        Raises:
            SessionError: If no file is open.
            SessionBusyError: If a batch or re-fit is running.
        """
        cache = self.cache
        if cache is None:
            raise SessionError("No file is open")
        self._check_not_busy("reverting overrides")
        stored = self._stored
        if stored is None:
            return None
        if board is not None:
            node, board_number = (int(k) for k in board)
            doomed = self.override_keys(node, board_number)
            default_title = f"N{node} B{board_number}"
        elif keys is not None:
            wanted = {ChannelKey(*(int(k) for k in key)) for key in keys}
            doomed = sorted(key for key in stored.overrides if key in wanted)
            default_title = short_title(doomed[0]) if len(wanted) == 1 and doomed else "channels"
        else:
            doomed = sorted(stored.overrides)
            default_title = ""
        if not doomed:
            return None
        return RevertRequest(
            cache, tuple(doomed), stored, default_title if title is None else title
        )

    def run_revert(self, request: RevertRequest) -> RevertOutcome:
        """Delete the overrides of a :class:`RevertRequest` in one cache write (blocking).

        Runs in a worker thread (a busy file can take a second). Checks that
        the stored batch is still the one the request saw. Does not change
        the session: pass the outcome to :meth:`apply_revert`.

        Raises:
            SessionBusyError: If a batch or re-fit is running.
            SessionError: If the stored results changed on disk (nothing is
                written).
            CacheBusyError: If another process holds the cache's lock.
        """
        with self._exclusive("A revert"):
            t_start = time.perf_counter()
            expected = request.stored.created_at
            with self._io_lock, _stale_as_session_error(request.cache):
                if request.is_all:
                    n_deleted = request.cache.clear_overrides(expected_created_at=expected)
                else:
                    n_deleted = request.cache.delete_overrides(
                        request.keys, expected_created_at=expected
                    )
            if n_deleted != len(request.keys):
                logger.warning(
                    f"Reverting {len(request.keys)} override(s) deleted {n_deleted} from "
                    f"{request.cache.path}"
                )
            doomed = set(request.keys)
            overrides = {k: v for k, v in request.stored.overrides.items() if k not in doomed}
            return RevertOutcome(
                request=request,
                stored=replace(request.stored, overrides=overrides),
                removed=request.keys,
                seconds=time.perf_counter() - t_start,
            )

    def apply_revert(self, outcome: RevertOutcome) -> tuple[ChannelKey, ...]:
        """Install a finished revert (GUI thread); returns the channels reverted.

        Raises:
            SessionError: As :meth:`apply_refit`.
        """
        self._check_current(outcome.request.cache, outcome.request.stored, "revert")
        self._set_stored(outcome.stored)
        return outcome.removed

    def _check_current(self, cache: UVCache, stored: StoredResults, what: str) -> None:
        """Refuse to apply an outcome computed for other results than the session holds."""
        current = self.cache
        if current is None or current.path != cache.path:
            raise SessionError(f"The {what} ran on {cache.path}, which is no longer the open cache")
        if self._stored is not stored:
            raise SessionError(
                f"The stored results changed while the {what} ran; reopen the file to see its "
                "overrides"
            )

    def revert_overrides(
        self, keys: Iterable[ChannelKey | tuple[int, int, int, int]]
    ) -> tuple[ChannelKey, ...]:
        """Delete the overrides of some channels now (request, run and apply in one call).

        Channels without an override are ignored. The GUI runs
        :meth:`run_revert` in a worker thread instead.

        Returns:
            The channels reverted, sorted.

        Raises:
            SessionError, SessionBusyError, CacheBusyError: See
            :meth:`revert_request` and :meth:`run_revert`.
        """
        request = self.revert_request(list(keys))
        return () if request is None else self.apply_revert(self.run_revert(request))

    def revert_board(self, node: int, board: int) -> tuple[ChannelKey, ...]:
        """Delete every override of a board now (see :meth:`revert_overrides`)."""
        request = self.revert_request(board=(node, board))
        return () if request is None else self.apply_revert(self.run_revert(request))

    def clear_overrides(self) -> tuple[ChannelKey, ...]:
        """Delete every override now, in one cache write; returns the channels reverted."""
        request = self.revert_request()
        return () if request is None else self.apply_revert(self.run_revert(request))

    def overrides_fitting_like(self, options: FitOptions) -> list[ChannelKey]:
        """Overrides a Fit All with ``options`` reproduces and so drops.

        See :meth:`~uvcorr.cache.StoredOverride.reproduced_by` (options that
        fit alike; never an override with options of a newer uvcorr).
        """
        stored = self._stored
        if stored is None:
            return []
        return sorted(key for key, o in stored.overrides.items() if o.reproduced_by(options))

    def results_changed_on_disk(self) -> bool:
        """Whether the cache's batch results are not the ones the session loaded.

        Raises:
            CacheBusyError: If another process holds the cache's lock.
        """
        cache = self.cache
        if cache is None:
            return False
        expected = self._stored.created_at if self._stored is not None else None
        with self._io_lock:
            return cache.results_created_at() != expected

    # ------------------------------------------------------------------
    # Exports (GUI thread; the writes are atomic)
    # ------------------------------------------------------------------

    @property
    def export_stem(self) -> str:
        """Name stem of the ``.tec`` file: the raw file's stem (``data_…_120628``).

        When no raw path is known, the cache name without ``.uv.h5`` and the
        raw file's extension.
        """
        dat = self.dat_path
        if dat is not None:
            return dat.stem
        cache = self.cache
        if cache is None:
            raise SessionError("No file is open")
        name = cache.path.name
        if name.endswith(CACHE_SUFFIX):
            name = name[: -len(CACHE_SUFFIX)]
        return Path(name).stem or cache.path.stem

    def _export_rows(self) -> list[ChannelResult]:
        if self.cache is None:
            raise SessionError("No file is open")
        if self._stored is None:
            raise SessionError("Nothing to export: run Fit All first")
        return self.merged_results()

    def check_export_target(self, path: str | Path) -> None:
        """Refuse an export target that would destroy data.

        The open raw file and the open cache (also through another name), any
        existing HDF5 file (a UV cache) and any existing ``.dat`` file are
        refused: the exports replace their target.

        Raises:
            SessionError: With the reason.
        """
        target = Path(path)
        opened = [(self.dat_path, "the open raw data file"), (self.cache_path, "the open UV cache")]
        for other, what in opened:
            if other is not None and _same_file(target, other):
                raise SessionError(f"Refusing to overwrite {what} {other} with an export")
        if not target.is_file():
            return
        if target.suffix.lower() == ".dat":
            raise SessionError(
                f"Refusing to overwrite {target}: it is a raw .dat file; choose another name"
            )
        try:
            is_hdf5 = bool(h5py.is_hdf5(target))
        except OSError:  # unreadable: the write reports it
            is_hdf5 = False
        if is_hdf5:
            raise SessionError(
                f"Refusing to overwrite {target}: it is an HDF5 file (a UV cache?); "
                "choose another name"
            )

    @property
    def cache_path(self) -> Path | None:
        """The open cache's path."""
        cache = self.cache
        return cache.path if cache is not None else None

    def _stale_warning(self) -> str:
        """A warning if the cache's results changed on disk since they were loaded."""
        try:
            changed = self.results_changed_on_disk()
        except (UVCacheError, OSError) as exc:
            logger.warning(f"Could not check the stored results before exporting: {exc}")
            return ""
        if not changed:
            return ""
        name = self.cache_path.name if self.cache_path is not None else "the cache"
        return (
            f"The results stored in {name} changed on disk since they were loaded (another "
            "process?). The export holds the results shown in this window; reopen the file "
            "to see the new ones."
        )

    def _export(
        self, targets: list[Path], write: Callable[[list[ChannelResult]], Iterable[Path]]
    ) -> ExportSummary:
        t_start = time.perf_counter()
        rows = self._export_rows()
        for target in targets:
            self.check_export_target(target)
        warning = self._stale_warning()
        try:
            paths = tuple(write(rows))
        except OSError as exc:
            raise _export_error(exc, targets) from exc
        return ExportSummary(
            paths=paths,
            n_rows=len(rows),
            n_ok=sum(1 for row in rows if row.ok),
            n_overrides=sum(1 for row in rows if row.options_source == OPTIONS_OVERRIDE),
            seconds=time.perf_counter() - t_start,
            warning=warning,
        )

    def export_tec(self, path: str | Path) -> ExportSummary:
        """Write the merged results' ``.tec`` file (the ``ok`` channels) to ``path``.

        Raises:
            SessionError: If there are no results, or the target is refused
                (:meth:`check_export_target`).
            ValueError: See :func:`~uvcorr.io.tec.write_tec`.
            OSError: If the file cannot be written (the message names ``path``).
        """
        target = Path(path)
        return self._export([target], lambda rows: [write_tec(target, rows)])

    def export_csv(self, path: str | Path) -> ExportSummary:
        """Write the merged results' ``radial_summary.csv`` to ``path``.

        Raises:
            SessionError, OSError: As :meth:`export_tec`.
        """
        target = Path(path)
        return self._export([target], lambda rows: [write_summary_csv(target, rows)])

    def export_outputs(self, directory: str | Path) -> ExportSummary:
        """Write ``<stem>.tec`` and ``radial_summary.csv`` to ``directory`` (created if needed).

        Both files are replaced together (:func:`~uvcorr.io.export.write_outputs`).

        Raises:
            SessionError: If there are no results, or a target is refused.
            ValueError: See :func:`~uvcorr.io.export.write_outputs`.
            OSError: If the files cannot be written (the message names the
                directory or file, not a temporary file).
        """
        stem = self.export_stem
        folder = Path(directory)

        def write(rows: list[ChannelResult]) -> Iterable[Path]:
            prepare_output_dir(folder, stem)
            return write_outputs(folder, stem, rows)

        return self._export([*output_paths(folder, stem)], write)

    # ------------------------------------------------------------------
    # Single-channel re-fit and override (lower-level helpers)
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
            SessionBusyError: If a batch or re-fit is running.
            ResultsError: If there are no stored batch results to attach the
                override to (run Fit All first).
            CacheBusyError: If another process holds the cache's lock.
        """
        cache = self.cache
        if cache is None:
            raise SessionError("No file is open")
        self._check_not_busy("storing an override")
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
