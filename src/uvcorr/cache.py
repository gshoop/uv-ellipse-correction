"""Sibling HDF5 UV cache (``<name>.dat.uv.h5``) built from a raw ``.dat`` file.

The cache holds the (U, V) fine-timing pairs (plus RENA, channel and PHA) of
every event on an active channel, grouped per board, so an analysis can load
one board in milliseconds instead of re-parsing the multi-GB raw file.

Layout (plan section 6.1)::

    /metadata                   attrs: uv_cache_version, uvcorr_version, source_path,
                                source_size, source_mtime, source_hash, parser_frames,
                                parser_events, parser_dropped, n_events_kept,
                                n_events_inactive, n_events_node0, build_seconds,
                                created_at and build provenance (see ``_build``)
    /events/node_{N}/board_{B}/ rena (int8), channel (int8), u, v, pha (int16),
                                in file order; attr n_events

The build streams the file once with
:meth:`~adc2kev.parser.packet_parser.PacketParser.iter_event_arrays`, drops
node-0 events and events on inactive channels, and appends each batch's events
to per-board buffers that are flushed to resizable, chunked, compressed
datasets. Memory stays bounded by a per-board and a global buffered-events
limit (:class:`BuildSettings`). Each build writes its own uniquely named
temporary file next to the cache (``<stem>.<random><suffix>.tmp``, e.g.
``run.dat.uv.1a2b3c4d.h5.tmp``) and renames it over the cache only on success,
so an interrupted build never leaves a cache that looks valid and concurrent
builds do not interfere. The build refuses to replace anything that is not a
UV cache (the raw file itself, a directory, a foreign file) and a cache that
another process has open.

The ``/results`` group (analysis results and GUI overrides) is added by the
analysis phase; the build creates only ``/metadata`` and ``/events``.

No HDF5 handle is kept open between calls: every accessor opens the file,
reads and closes it, so :class:`UVCache` objects are cheap, picklable and safe
to use from worker processes. When another process holds the file's HDF5 lock
(it has the file open for writing), accessors retry for
``LOCK_RETRY_SECONDS`` and then raise :class:`CacheBusyError`; a busy cache is
never treated as invalid.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import math
import os
import secrets
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
from adc2kev.parser import EventBatch, PacketParser

from uvcorr import __version__
from uvcorr.channels import active_channel_mask

logger = logging.getLogger(__name__)

__all__ = [
    "BYTES_PER_EVENT",
    "CACHE_SUFFIX",
    "EVENT_DTYPES",
    "EVENT_FIELDS",
    "USER_WARNING",
    "UV_CACHE_VERSION",
    "BoardUV",
    "BuildSettings",
    "BuildStats",
    "CacheBuildCancelled",
    "CacheBuildError",
    "CacheBusyError",
    "InsufficientDiskSpaceError",
    "ProgressCallback",
    "StopFlag",
    "UVCache",
    "UVCacheError",
    "check_disk_space",
    "compute_source_hash",
    "default_cache_path",
    "estimate_cache_bytes",
    "open_or_build",
]

UV_CACHE_VERSION = "1.0.0"
"""Layout version; a cache with a different version is rebuilt."""

CACHE_SUFFIX = ".uv.h5"
TMP_SUFFIX = ".tmp"

# Bytes of the source file hashed for identity (same as adc2kev's cache hash).
HASH_CHUNK_BYTES = 1024 * 1024

EVENT_FIELDS: tuple[str, ...] = ("rena", "channel", "u", "v", "pha")
"""Per-board dataset names, in storage order."""

EVENT_DTYPES: dict[str, np.dtype[Any]] = {
    "rena": np.dtype(np.int8),
    "channel": np.dtype(np.int8),
    "u": np.dtype(np.int16),
    "v": np.dtype(np.int16),
    "pha": np.dtype(np.int16),
}

BYTES_PER_EVENT: int = sum(dtype.itemsize for dtype in EVENT_DTYPES.values())
"""Uncompressed cache bytes per stored event (8)."""

# Raw-file bytes per event assumed by the disk-space estimate. An AND-mode
# frame is 19 + 6n bytes for n hits: 25 B/event with one hit per frame, 16 B
# with ~2 (the test acquisition: 16.7 B/event), but only ~6.5 B/event for
# frames with all 36 channels hit (OR-mode frames can go down to ~4.7 B). 16
# B/event is therefore an assumption about typical data, not a bound: dense
# data can need up to ~2.5x the estimate. The estimate is of the uncompressed
# size, while the real (lzf) cache is ~25 % smaller. If the disk fills up
# anyway, the build fails cleanly and removes its temporary file.
DAT_BYTES_PER_EVENT_ESTIMATE = 16
DISK_SPACE_HEADROOM_BYTES = 64 * 1024 * 1024

# Share of the progress range given to the parse; the rest covers the final
# flush, metadata and rename, after which 1.0 is reported.
PARSE_PROGRESS_SHARE = 0.97

# How long an accessor retries when another process holds the file's HDF5
# lock, before raising CacheBusyError.
LOCK_RETRY_SECONDS = 1.0
LOCK_RETRY_INTERVAL = 0.05

USER_WARNING = "uvcorr_user_warning"
"""LogRecord attribute set on warnings addressed to the end user (a rebuild
discarding stored ``/results``; a file without valid frames). Front ends that
report these conditions themselves (the CLI) filter such records out."""

# Board groups are keyed internally by node * 64 + board (a board is 6 bits).
_BOARD_KEY_SHIFT = 6

# HDF5 chunk-cache size per open dataset during the build, in chunks of the
# widest (int16) dataset. The build keeps 5 datasets per board open, and with
# HDF5's default 1 MiB cache each the build's peak RSS grew to 2.2 GB on the
# test file; two chunks are enough for sequential appends (a partial chunk
# stays cached until it is full, so each chunk is compressed once) and bring
# the peak RSS of ``uvcorr build-cache`` down to ~0.95 GB at the same speed.
_BUILD_CHUNK_CACHE_CHUNKS = 2

ProgressCallback = Callable[[float], None]
StopFlag = Callable[[], bool] | threading.Event


class UVCacheError(Exception):
    """Base class for UV cache errors."""


class CacheBuildError(UVCacheError):
    """The cache could not be built (no cache or temporary file is left behind)."""


class InsufficientDiskSpaceError(CacheBuildError):
    """Not enough free disk space at the cache location to build the cache."""


class CacheBuildCancelled(UVCacheError):
    """The build was stopped through its ``stop_flag`` (nothing is left behind)."""


class CacheBusyError(UVCacheError):
    """Another process holds the cache file's HDF5 lock (it has it open for writing)."""


@dataclass(frozen=True)
class BuildSettings:
    """Tunables of :meth:`UVCache.build_from_dat`.

    The defaults were chosen by measurement on the 3.5 GB test acquisition
    (208M kept events, 155 boards; plan section 11): lzf with shuffle builds
    in about 21 s (parse alone: ~11-16 s) to 1.26 GB, against 1.71 GB
    uncompressed (16 s) and 0.94-0.98 GB with gzip level 1-4 plus shuffle
    (35-40 s); the chunk length (64k vs 256k events) changed neither. Loading
    a median board (1.1M events) takes about 20 ms. The thresholds are
    overridable so tests can exercise the flushing paths on small files.

    Attributes:
        batch_events: Events per parser batch (``iter_event_arrays``).
        board_flush_events: Flush a board's buffer once it holds this many
            events.
        max_buffered_events: Flush every board's buffer once the buffers
            together hold more than this many events. With ~160 boards the
            per-board limit alone would buffer the whole file, so this global
            budget is what bounds memory: at most about
            ``max_buffered_events + batch_events`` events (8 B each) are held.
        chunk_events: HDF5 chunk length (events) of the per-board datasets.
        compression: HDF5 compression filter (``"gzip"``, ``"lzf"`` or None).
        compression_opts: Filter option (gzip level), or None.
        shuffle: Apply the byte-shuffle filter before compression.
    """

    batch_events: int = 2_000_000
    board_flush_events: int = 4_000_000
    max_buffered_events: int = 16_000_000
    chunk_events: int = 65_536
    compression: str | None = "lzf"
    compression_opts: int | None = None
    shuffle: bool = True

    def __post_init__(self) -> None:
        for name in ("batch_events", "board_flush_events", "max_buffered_events", "chunk_events"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"BuildSettings.{name} must be >= 1, got {value}")
        if self.compression not in (None, "gzip", "lzf"):
            raise ValueError(f"Unsupported compression {self.compression!r}")


@dataclass(frozen=True)
class BuildStats:
    """Summary of one cache build.

    Attributes:
        n_events_parsed: Events decoded by the parser.
        n_events_kept: Events stored (active channel, node != 0).
        n_events_inactive: Events dropped because the channel is inactive
            (node != 0).
        n_events_node0: Events dropped because they came from node 0 (the DAQ
            PC address), whatever their channel.
        n_events_uv_zero: Kept events with U = V = 0 (no fine-timing
            sample, e.g. OR-mode slow-trigger-only hits).
        n_boards: ``(node, board)`` groups written.
        parser_frames: Frames accepted by the parser.
        parser_dropped: Frames the parser dropped (CRC errors).
        build_seconds: Wall time of the build.
        n_flushes: Appends of a board buffer to its HDF5 datasets.
        max_buffered_events: Most events held in the buffers at once.
        cache_bytes: Size of the finished cache file.
    """

    n_events_parsed: int
    n_events_kept: int
    n_events_inactive: int
    n_events_node0: int
    n_events_uv_zero: int
    n_boards: int
    parser_frames: int
    parser_dropped: int
    build_seconds: float
    n_flushes: int
    max_buffered_events: int
    cache_bytes: int


@dataclass(frozen=True, eq=False)
class BoardUV:
    """All cached events of one board, in file order.

    Attributes:
        node: Node number.
        board: Board number.
        rena: int8 RENA per event.
        channel: int8 channel per event.
        u: int16 U per event.
        v: int16 V per event.
        pha: int16 pulse height per event (kept for a future PHA gate).
    """

    node: int
    board: int
    rena: npt.NDArray[np.int8]
    channel: npt.NDArray[np.int8]
    u: npt.NDArray[np.int16]
    v: npt.NDArray[np.int16]
    pha: npt.NDArray[np.int16]

    @property
    def n_events(self) -> int:
        """Number of events on the board."""
        return int(self.u.shape[0])

    def channel_mask(self, rena: int, channel: int) -> npt.NDArray[np.bool_]:
        """Boolean mask of the events on ``(rena, channel)``."""
        mask: npt.NDArray[np.bool_] = (self.rena == rena) & (self.channel == channel)
        return mask

    def channel_data(
        self, rena: int, channel: int
    ) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
        """(U, V) of one channel's events in file order (empty if it has none)."""
        mask = self.channel_mask(rena, channel)
        return self.u[mask], self.v[mask]

    def channels(self) -> list[tuple[int, int, int]]:
        """``(rena, channel, n_events)`` of every channel with events, sorted."""
        return _channel_counts(self.rena, self.channel)


def default_cache_path(dat_path: str | Path) -> Path:
    """Return the default cache path of a raw file: ``<dat path>.uv.h5``."""
    return Path(str(dat_path) + CACHE_SUFFIX)


def compute_source_hash(path: str | Path, chunk_size: int = HASH_CHUNK_BYTES) -> str:
    """SHA-256 hex digest of the first ``chunk_size`` bytes of a file.

    Reimplements adc2kev's private ``CalibrationCache._compute_file_hash``
    exactly, so both packages identify a source file the same way.

    Args:
        path: File to hash.
        chunk_size: Number of leading bytes hashed (default 1 MiB).

    Returns:
        Hex-encoded SHA-256 digest.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read(chunk_size))
    return hasher.hexdigest()


def estimate_cache_bytes(dat_size: int) -> int:
    """Estimate the uncompressed cache size for a raw file of ``dat_size`` bytes.

    Assumes ``DAT_BYTES_PER_EVENT_ESTIMATE`` raw bytes per event and
    ``BYTES_PER_EVENT`` cache bytes per event (all events kept). The real,
    compressed cache is smaller; the estimate is used as the disk-space
    requirement.
    """
    return math.ceil(dat_size / DAT_BYTES_PER_EVENT_ESTIMATE) * BYTES_PER_EVENT


def check_disk_space(cache_path: str | Path, dat_size: int) -> None:
    """Fail early if the cache location lacks space for a cache of this raw file.

    Args:
        cache_path: Cache file to be written (its directory must exist).
        dat_size: Size of the raw ``.dat`` file in bytes.

    Raises:
        InsufficientDiskSpaceError: If the free space at the cache location is
            below :func:`estimate_cache_bytes` plus a 64 MiB headroom.
    """
    cache_path = Path(cache_path)
    required = estimate_cache_bytes(dat_size) + DISK_SPACE_HEADROOM_BYTES
    free = shutil.disk_usage(cache_path.parent).free
    if free < required:
        raise InsufficientDiskSpaceError(
            f"Not enough free disk space for the UV cache {cache_path}: about "
            f"{_gb(required)} needed (estimated from the {_gb(dat_size)} raw file), "
            f"{_gb(free)} free in {cache_path.parent}. Free some space or choose "
            "another cache location."
        )


class UVCache:
    """Per-board UV event cache of one raw ``.dat`` file.

    The object only stores the path; every method opens the HDF5 file for the
    duration of the call.

    Example:
        >>> cache = UVCache(default_cache_path("run.dat"))
        >>> if not cache.is_valid_for("run.dat"):
        ...     cache.build_from_dat("run.dat")
        >>> for node, board in cache.boards():
        ...     data = cache.load_board(node, board)

    Attributes:
        last_build: Statistics of the build run through this object, or None
            if it has not built the cache (e.g. a valid cache was reused).
    """

    def __init__(self, path: str | Path) -> None:
        """Args:
        path: Path of the HDF5 cache file (created by :meth:`build_from_dat`).
        """
        self._path = Path(path)
        self.last_build: BuildStats | None = None

    def __repr__(self) -> str:
        return f"UVCache({str(self._path)!r})"

    @property
    def path(self) -> Path:
        """Path of the cache file."""
        return self._path

    def tmp_files(self) -> list[Path]:
        """Temporary build files of this cache that exist right now, sorted.

        Each build writes ``<stem>.<8 hex digits><suffix>.tmp`` next to the
        cache and removes it when it finishes; files listed here belong to
        builds in progress, or to builds that were killed.
        """
        pattern = f"{_glob_escape(self._path.stem)}.{'[0-9a-f]' * _TMP_TOKEN_HEX}"
        pattern += f"{_glob_escape(self._path.suffix)}{TMP_SUFFIX}"
        return sorted(self._path.parent.glob(pattern))

    def exists(self) -> bool:
        """Whether the cache file exists (it may still be stale)."""
        return self._path.is_file()

    def has_results(self) -> bool:
        """Whether the cache stores analysis results or overrides (``/results``).

        False when the file does not exist or is not a readable HDF5 file.

        Raises:
            CacheBusyError: If another process holds the file's lock.
        """
        if not self._path.is_file():
            return False
        try:
            with _open_h5(self._path) as h5f:
                return "results" in h5f
        except OSError:
            return False

    # ------------------------------------------------------------------
    # Validity
    # ------------------------------------------------------------------

    def is_valid_for(self, dat_path: str | Path) -> bool:
        """Whether this cache was built from the given raw file, as it is now.

        The cache version, the source size, mtime and the hash of its first
        MiB must all match (pattern of adc2kev's ``DiagnosticCache``). A
        missing or unreadable cache, a file that is not a UV cache, or a
        missing raw file, is not valid. A cache that another process has open
        for writing is *not* reported invalid (a rebuild would discard that
        process's work): :class:`CacheBusyError` is raised instead.

        Args:
            dat_path: The raw ``.dat`` file.

        Returns:
            True if the cache can be used for ``dat_path`` without a rebuild.

        Raises:
            CacheBusyError: If another process holds the cache's HDF5 lock
                for longer than ``LOCK_RETRY_SECONDS``.
        """
        dat_path = Path(dat_path)
        if not self._path.is_file() or not dat_path.is_file():
            return False
        try:
            stat = dat_path.stat()
            with _open_h5(self._path) as h5f:
                meta = h5f["metadata"].attrs
                expected = {
                    "uv_cache_version": UV_CACHE_VERSION,
                    "source_size": stat.st_size,
                    "source_mtime": stat.st_mtime,
                }
                for name, value in expected.items():
                    if not _attr_equals(meta.get(name), value):
                        return False
                if not _attr_equals(meta.get("source_hash"), compute_source_hash(dat_path)):
                    return False
                if "events" not in h5f:
                    return False
        except (OSError, KeyError, ValueError, TypeError):
            # Not a readable UV cache (a lock raises CacheBusyError, not OSError)
            return False
        return True

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_from_dat(
        self,
        dat_path: str | Path,
        progress_cb: ProgressCallback | None = None,
        stop_flag: StopFlag | None = None,
        *,
        settings: BuildSettings | None = None,
    ) -> BuildStats:
        """Build (or rebuild) the cache from a raw ``.dat`` file.

        Before parsing, the target is checked: it must not be the raw file
        itself, a directory, or an existing file that is not a UV cache (such
        a file is never replaced), and no other process may have it open.
        Rebuilding a cache that stores ``/results`` discards them (a warning
        is logged). The new cache is written to a uniquely named temporary
        file (see :meth:`tmp_files`) and renamed over :attr:`path` only on
        success; on failure or cancellation this build's temporary file is
        removed and any previous cache file is left untouched.

        Args:
            dat_path: Raw ``.dat`` acquisition file.
            progress_cb: Called after every parser batch with the fraction of
                the file read scaled to ``[0, PARSE_PROGRESS_SHARE]``
                (non-decreasing), and with 1.0 once the cache is in place.
            stop_flag: A callable returning True, or a ``threading.Event``
                that is set, to request a stop. Checked after every parser
                batch.
            settings: Build tunables (defaults: :class:`BuildSettings`).

        Returns:
            The build statistics (also stored in :attr:`last_build`).

        Raises:
            FileNotFoundError: If ``dat_path`` does not exist.
            CacheBusyError: If another process has the existing cache open.
            InsufficientDiskSpaceError: If the cache location lacks space.
            CacheBuildCancelled: If ``stop_flag`` requested a stop.
            CacheBuildError: If the target is unsuitable, the cache directory
                cannot be created or written, or parsing or writing failed.
        """
        dat_path = Path(dat_path)
        if not dat_path.is_file():
            raise FileNotFoundError(f"Raw data file not found: {dat_path}")
        settings = settings if settings is not None else BuildSettings()
        should_stop = _stop_callable(stop_flag)

        try:
            discards_results = _check_target(self._path, dat_path)
        except OSError as exc:
            raise CacheBuildError(f"Cannot use the cache path {self._path}: {exc}") from exc
        if discards_results:
            logger.warning(
                f"Rebuilding UV cache {self._path}: its stored analysis results and "
                "overrides (/results) are discarded",
                extra={USER_WARNING: True},
            )
        _prepare_directory(self._path, dat_path.stat().st_size)

        logger.info(f"Building UV cache {self._path} from {dat_path}")
        tmp_path: Path | None = None
        try:
            tmp_path = _create_tmp_file(self._path)
            stats = _build(tmp_path, dat_path, progress_cb, should_stop, settings)
            _ensure_not_busy(self._path)
            os.replace(tmp_path, self._path)
        except (CacheBuildCancelled, CacheBuildError, CacheBusyError):
            _remove_quietly(tmp_path)
            raise
        except Exception as exc:
            _remove_quietly(tmp_path)
            raise CacheBuildError(
                f"Failed to build UV cache {self._path} from {dat_path}: {exc}"
            ) from exc
        except BaseException:  # KeyboardInterrupt, SystemExit
            _remove_quietly(tmp_path)
            raise

        stats = replace(stats, cache_bytes=self._path.stat().st_size)
        self.last_build = stats
        logger.info(
            f"UV cache built: {stats.n_events_kept:,} events on {stats.n_boards} boards "
            f"in {stats.build_seconds:.1f} s ({_gb(stats.cache_bytes)})"
        )
        if progress_cb is not None:
            progress_cb(1.0)
        return stats

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        """The ``/metadata`` attributes as plain Python values."""
        with _open_h5(self._path) as h5f:
            return {name: _to_python(value) for name, value in h5f["metadata"].attrs.items()}

    def boards(self) -> list[tuple[int, int]]:
        """``(node, board)`` of every board with cached events, sorted."""
        return sorted(self.board_event_counts())

    def board_event_counts(self) -> dict[tuple[int, int], int]:
        """``{(node, board): n_events}`` from the board groups' ``n_events`` attrs."""
        counts: dict[tuple[int, int], int] = {}
        with _open_h5(self._path) as h5f:
            events = h5f["events"]
            for node_name, node_group in events.items():
                node = _group_number(node_name, "node_")
                for board_name, board_group in node_group.items():
                    board = _group_number(board_name, "board_")
                    counts[(node, board)] = int(board_group.attrs["n_events"])
        return dict(sorted(counts.items()))

    def load_board(self, node: int, board: int) -> BoardUV:
        """Load every cached event of one board.

        Args:
            node: Node number.
            board: Board number.

        Returns:
            The board's arrays, in file order.

        Raises:
            KeyError: If the cache has no events for this board.
        """
        with _open_h5(self._path) as h5f:
            group = self._board_group(h5f, node, board)
            arrays = {name: group[name][()] for name in EVENT_FIELDS}
        return BoardUV(node=node, board=board, **arrays)

    def channel_data(
        self, node: int, board: int, rena: int, channel: int
    ) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
        """(U, V) of one channel's events in file order.

        Args:
            node: Node number.
            board: Board number.
            rena: RENA number.
            channel: Channel number.

        Returns:
            ``(u, v)`` int16 arrays, empty if the channel has no events.

        Raises:
            KeyError: If the cache has no events for this board.
        """
        with _open_h5(self._path) as h5f:
            group = self._board_group(h5f, node, board)
            mask = (group["rena"][()] == rena) & (group["channel"][()] == channel)
            u: npt.NDArray[np.int16] = group["u"][()][mask]
            v: npt.NDArray[np.int16] = group["v"][()][mask]
        return u, v

    def channels(self, node: int, board: int) -> list[tuple[int, int, int]]:
        """``(rena, channel, n_events)`` of every channel with events on a board, sorted.

        Raises:
            KeyError: If the cache has no events for this board.
        """
        with _open_h5(self._path) as h5f:
            group = self._board_group(h5f, node, board)
            return _channel_counts(group["rena"][()], group["channel"][()])

    @staticmethod
    def _board_group(h5f: Any, node: int, board: int) -> Any:
        group = h5f.get(_board_path(node, board))
        if group is None:
            raise KeyError(f"No cached events for node {node} board {board}")
        return group


def open_or_build(
    dat_path: str | Path,
    cache_path: str | Path | None = None,
    force: bool = False,
    progress_cb: ProgressCallback | None = None,
    stop_flag: StopFlag | None = None,
    *,
    settings: BuildSettings | None = None,
) -> UVCache:
    """Return a valid cache for a raw file, building it if needed.

    A cache that :meth:`UVCache.is_valid_for` the raw file is reused (its
    ``last_build`` stays None) unless ``force`` is set; otherwise it is
    (re)built.

    Args:
        dat_path: Raw ``.dat`` acquisition file.
        cache_path: Cache location (default: :func:`default_cache_path`).
        force: Rebuild even if a valid cache exists.
        progress_cb: Progress callback (see :meth:`UVCache.build_from_dat`);
            called with 1.0 when a valid cache is reused.
        stop_flag: Stop request (see :meth:`UVCache.build_from_dat`).
        settings: Build tunables.

    Returns:
        The ready-to-use cache.

    Raises:
        FileNotFoundError: If ``dat_path`` does not exist.
        CacheBusyError: If another process has the cache open for writing
            (it is then neither reused nor rebuilt).
        InsufficientDiskSpaceError: If a build is needed and space is short.
        CacheBuildCancelled: If a build was stopped.
        CacheBuildError: If a build failed.
    """
    dat_path = Path(dat_path)
    if not dat_path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {dat_path}")
    cache = UVCache(cache_path if cache_path is not None else default_cache_path(dat_path))
    if not force and cache.is_valid_for(dat_path):
        logger.info(f"Reusing valid UV cache {cache.path}")
        if progress_cb is not None:
            progress_cb(1.0)
        return cache
    cache.build_from_dat(dat_path, progress_cb, stop_flag, settings=settings)
    return cache


# ----------------------------------------------------------------------
# Build internals
# ----------------------------------------------------------------------

_Columns = tuple[npt.NDArray[Any], ...]


class _BoardWriter:
    """Per-board event buffers flushed to chunked, resizable HDF5 datasets."""

    def __init__(self, events_group: Any, settings: BuildSettings) -> None:
        self._events = events_group
        self._settings = settings
        self._pending: dict[int, list[_Columns]] = {}
        self._pending_counts: dict[int, int] = {}
        self._datasets: dict[int, tuple[Any, ...]] = {}
        self._written: dict[int, int] = {}
        self.buffered = 0
        self.max_buffered = 0
        self.n_flushes = 0

    def add(self, key: int, columns: _Columns) -> None:
        """Buffer one batch's events of one board (flushing the board if full)."""
        n = int(columns[0].shape[0])
        self._pending.setdefault(key, []).append(columns)
        count = self._pending_counts.get(key, 0) + n
        self._pending_counts[key] = count
        self.buffered += n
        self.max_buffered = max(self.max_buffered, self.buffered)
        if count >= self._settings.board_flush_events:
            self.flush_board(key)

    def enforce_budget(self) -> None:
        """Flush every buffer if together they exceed the global budget."""
        if self.buffered > self._settings.max_buffered_events:
            self.flush_all()

    def flush_all(self) -> None:
        for key in sorted(self._pending):
            self.flush_board(key)

    def flush_board(self, key: int) -> None:
        pieces = self._pending.pop(key, None)
        if not pieces:
            return
        n = self._pending_counts.pop(key)
        self.buffered -= n
        if len(pieces) == 1:
            columns = pieces[0]
        else:
            columns = tuple(
                np.concatenate([piece[i] for piece in pieces]) for i in range(len(EVENT_FIELDS))
            )
        datasets = self._datasets.get(key)
        if datasets is None:
            datasets = self._create_datasets(key)
        start = self._written.get(key, 0)
        stop = start + n
        for dataset, data in zip(datasets, columns):
            dataset.resize((stop,))
            dataset[start:stop] = data
        self._written[key] = stop
        self.n_flushes += 1

    def finish(self) -> dict[int, int]:
        """Flush everything, write the ``n_events`` attrs; return events per key."""
        self.flush_all()
        for key, n in self._written.items():
            self._events[_board_path(*_split_key(key), rooted=False)].attrs["n_events"] = n
        return dict(self._written)

    def _create_datasets(self, key: int) -> tuple[Any, ...]:
        node, board = _split_key(key)
        group = self._events.require_group(f"node_{node}").create_group(f"board_{board}")
        s = self._settings
        datasets = tuple(
            group.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=EVENT_DTYPES[name],
                chunks=(s.chunk_events,),
                compression=s.compression,
                compression_opts=s.compression_opts,
                shuffle=s.shuffle,
            )
            for name in EVENT_FIELDS
        )
        self._datasets[key] = datasets
        return datasets


@dataclass
class _Counts:
    parsed: int = 0
    kept: int = 0
    inactive: int = 0
    node0: int = 0
    uv_zero: int = 0


# EventBatch columns used by the build and the dtypes they must have: the
# storage casts are only lossless for these (HDF5 would silently saturate e.g.
# an int32 U of 40000 to 32767 in an int16 dataset).
_BATCH_DTYPES: dict[str, np.dtype[Any]] = {
    "node_num": np.dtype(np.uint8),
    "board_num": np.dtype(np.uint8),
    "rena_num": np.dtype(np.uint8),
    "channel_num": np.dtype(np.uint8),
    "u": np.dtype(np.int16),
    "v": np.dtype(np.int16),
    "pha": np.dtype(np.int16),
}


def _split_batch(batch: EventBatch, counts: _Counts) -> list[tuple[int, _Columns]]:
    """Filter one batch and split it into per-board column tuples (file order kept).

    Every returned array is a fresh copy (not a view of the batch), so a
    buffered board never pins a whole batch in memory.

    Raises:
        CacheBuildError: If a column does not have the expected dtype.
    """
    for name, dtype in _BATCH_DTYPES.items():
        actual = getattr(batch, name).dtype
        if actual != dtype:
            raise CacheBuildError(
                f"Parser batch column {name!r} has dtype {actual}, expected {dtype} "
                "(incompatible adc2kev version?)"
            )
    n = batch.n_events
    counts.parsed += n
    active = active_channel_mask(batch.rena_num, batch.channel_num)
    from_node0 = batch.node_num == 0
    n_node0 = int(np.count_nonzero(from_node0))
    if n_node0:
        counts.node0 += n_node0
        counts.inactive += int(np.count_nonzero(~active & ~from_node0))
        keep = active & ~from_node0
    else:
        counts.inactive += n - int(np.count_nonzero(active))
        keep = active

    index = np.flatnonzero(keep)
    counts.kept += int(index.shape[0])
    if index.shape[0] == 0:
        return []
    counts.uv_zero += int(np.count_nonzero(keep & (batch.u == 0) & (batch.v == 0)))

    node_key = batch.node_num[index].astype(np.uint16) << _BOARD_KEY_SHIFT
    board_key = node_key | batch.board_num[index]
    order = np.argsort(board_key, kind="stable")  # stable: file order within a board
    sorted_keys = board_key[order]
    index = index[order]
    bounds = np.concatenate(
        ([0], np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1, [index.shape[0]])
    )

    pieces: list[tuple[int, _Columns]] = []
    for start, stop in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        sel = index[start:stop]
        columns = (
            batch.rena_num[sel].view(np.int8),
            batch.channel_num[sel].view(np.int8),
            batch.u[sel],
            batch.v[sel],
            batch.pha[sel],
        )
        pieces.append((int(sorted_keys[start]), columns))
    return pieces


def _build(
    tmp_path: Path,
    dat_path: Path,
    progress_cb: ProgressCallback | None,
    should_stop: Callable[[], bool],
    settings: BuildSettings,
) -> BuildStats:
    """Parse ``dat_path`` into a complete cache file at ``tmp_path``."""
    t_start = time.perf_counter()
    source_stat = dat_path.stat()
    source_hash = compute_source_hash(dat_path)
    file_size = source_stat.st_size

    parser = PacketParser(dat_path)
    counts = _Counts()
    last_fraction = 0.0

    widest = max(dtype.itemsize for dtype in EVENT_DTYPES.values())
    chunk_cache_bytes = _BUILD_CHUNK_CACHE_CHUNKS * settings.chunk_events * widest
    with h5py.File(tmp_path, "w", rdcc_nbytes=chunk_cache_bytes) as h5f:
        metadata = h5f.create_group("metadata")
        writer = _BoardWriter(h5f.create_group("events"), settings)

        batches = parser.iter_event_arrays(batch_events=settings.batch_events)
        try:
            for batch in batches:
                if should_stop():
                    raise CacheBuildCancelled(f"UV cache build of {dat_path} cancelled")
                for key, columns in _split_batch(batch, counts):
                    writer.add(key, columns)
                writer.enforce_budget()
                if progress_cb is not None and file_size > 0:
                    fraction = PARSE_PROGRESS_SHARE * min(1.0, batch.bytes_read / file_size)
                    last_fraction = max(last_fraction, fraction)
                    progress_cb(last_fraction)
        finally:
            # Close the parser generator (and its file) when stopping early
            close = getattr(batches, "close", None)
            if close is not None:
                close()
        if should_stop():
            raise CacheBuildCancelled(f"UV cache build of {dat_path} cancelled")

        board_counts = writer.finish()
        h5f.flush()

        end_stat = dat_path.stat()
        if (end_stat.st_size, end_stat.st_mtime) != (source_stat.st_size, source_stat.st_mtime):
            raise CacheBuildError(f"{dat_path} changed while the UV cache was being built")

        pstats = parser.get_statistics()
        if pstats.total_events != counts.parsed:
            raise CacheBuildError(
                f"Parser reported {pstats.total_events:,} events but "
                f"{counts.parsed:,} were delivered"
            )
        accounted = counts.kept + counts.inactive + counts.node0
        if accounted != counts.parsed or counts.kept != sum(board_counts.values()):
            raise CacheBuildError(f"Inconsistent event counts during the build: {counts}")
        if file_size > 0 and pstats.total_frames == 0:
            logger.warning(
                f"No valid frames in {dat_path} ({file_size:,} bytes): is this a raw .dat file?",
                extra={USER_WARNING: True},
            )

        build_seconds = time.perf_counter() - t_start
        attrs = metadata.attrs
        attrs["uv_cache_version"] = UV_CACHE_VERSION
        attrs["uvcorr_version"] = __version__
        attrs["source_path"] = str(dat_path.absolute())
        attrs["source_size"] = np.int64(source_stat.st_size)
        attrs["source_mtime"] = float(source_stat.st_mtime)
        attrs["source_hash"] = source_hash
        attrs["parser_frames"] = np.int64(pstats.total_frames)
        attrs["parser_events"] = np.int64(pstats.total_events)
        attrs["parser_dropped"] = np.int64(pstats.dropped_frames)
        attrs["parser_invalid_headers"] = np.int64(pstats.invalid_headers)
        attrs["parser_invalid_lengths"] = np.int64(pstats.invalid_lengths)
        attrs["parser_bytes_read"] = np.int64(pstats.total_bytes_read)
        attrs["n_events_kept"] = np.int64(counts.kept)
        attrs["n_events_inactive"] = np.int64(counts.inactive)
        attrs["n_events_node0"] = np.int64(counts.node0)
        attrs["n_events_uv_zero"] = np.int64(counts.uv_zero)
        attrs["n_boards"] = np.int64(len(board_counts))
        attrs["build_seconds"] = float(build_seconds)
        attrs["created_at"] = datetime.now().isoformat()
        attrs["chunk_events"] = np.int64(settings.chunk_events)
        attrs["compression"] = settings.compression or "none"
        attrs["shuffle"] = bool(settings.shuffle)

    return BuildStats(
        n_events_parsed=counts.parsed,
        n_events_kept=counts.kept,
        n_events_inactive=counts.inactive,
        n_events_node0=counts.node0,
        n_events_uv_zero=counts.uv_zero,
        n_boards=len(board_counts),
        parser_frames=pstats.total_frames,
        parser_dropped=pstats.dropped_frames,
        build_seconds=build_seconds,
        n_flushes=writer.n_flushes,
        max_buffered_events=writer.max_buffered,
        cache_bytes=0,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _check_target(path: Path, dat_path: Path) -> bool:
    """Refuse cache targets whose replacement would destroy something else.

    Returns:
        True if the target is an existing UV cache that stores ``/results``.

    Raises:
        CacheBuildError: If ``path`` is the raw file itself, a directory, or
            an existing file that is not a (readable) UV cache.
        CacheBusyError: If another process has the existing cache open.
    """
    same = path.resolve() == dat_path.resolve()
    if path.exists() and not same:
        same = os.path.samefile(path, dat_path)
    if same:
        raise CacheBuildError(
            f"The cache path {path} is the raw data file itself; choose another cache path"
        )
    if path.is_dir():
        raise CacheBuildError(f"The cache path {path} is a directory; give a file path")
    if not path.exists():
        return False
    refuse = (
        f"Refusing to replace {path}: it is not a uvcorr UV cache. Delete or move it, "
        "or choose another cache path"
    )
    if not path.is_file() or not h5py.is_hdf5(path):
        raise CacheBuildError(refuse)
    try:
        with _open_h5(path) as h5f:
            metadata = h5f.get("metadata")
            is_cache = metadata is not None and "uv_cache_version" in metadata.attrs
            has_results = "results" in h5f
    except OSError as exc:
        raise CacheBuildError(f"{refuse} (unreadable: {exc})") from exc
    if not is_cache:
        raise CacheBuildError(refuse)
    return has_results


def _prepare_directory(path: Path, dat_size: int) -> None:
    """Create the cache directory and check that it is writable and has space.

    Raises:
        InsufficientDiskSpaceError: If the free space is short.
        CacheBuildError: If the directory cannot be created, written or queried.
    """
    parent = path.parent
    hint = "choose another location (uvcorr build-cache --cache PATH)"
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CacheBuildError(f"Cannot create the cache directory {parent}: {exc}; {hint}") from exc
    if not os.access(parent, os.W_OK | os.X_OK):
        raise CacheBuildError(f"No write permission in the cache directory {parent}; {hint}")
    try:
        check_disk_space(path, dat_size)
    except OSError as exc:
        raise CacheBuildError(f"Cannot check the free space in {parent}: {exc}") from exc


# Random hex digits in a temporary build file name.
_TMP_TOKEN_HEX = 8


def _create_tmp_file(path: Path) -> Path:
    """Atomically create an empty, uniquely named temporary file next to ``path``.

    Named ``<stem>.<8 hex digits><suffix>.tmp`` (so ``*.h5.tmp`` ignore rules
    match it) and created with the default permissions (``0o666`` minus the
    umask; ``tempfile.mkstemp`` would make the finished cache owner-only).
    """
    for _ in range(100):
        token = secrets.token_hex(_TMP_TOKEN_HEX // 2)
        candidate = path.with_name(f"{path.stem}.{token}{path.suffix}{TMP_SUFFIX}")
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue
        os.close(fd)
        return candidate
    raise CacheBuildError(f"Could not create a temporary file next to {path}")


def _ensure_not_busy(path: Path) -> None:
    """Raise CacheBusyError if another process has ``path`` open for writing."""
    if path.is_file() and h5py.is_hdf5(path):
        with _open_h5(path):
            pass


@contextmanager
def _open_h5(path: Path, mode: str = "r") -> Iterator[Any]:
    """Open an HDF5 file, retrying while another process holds its lock.

    Raises:
        CacheBusyError: If the lock is still held after ``LOCK_RETRY_SECONDS``.
        OSError: For any other failure to open the file.
    """
    deadline = time.monotonic() + LOCK_RETRY_SECONDS
    while True:
        try:
            h5f = h5py.File(path, mode)
            break
        except OSError as exc:
            if not _is_lock_error(exc):
                raise
            if time.monotonic() >= deadline:
                raise CacheBusyError(
                    f"The UV cache {path} is in use by another process (HDF5 file lock); "
                    "try again when it has finished"
                ) from exc
            time.sleep(LOCK_RETRY_INTERVAL)
    with h5f:
        yield h5f


def _is_lock_error(exc: OSError) -> bool:
    """Whether an h5py open failed because another process holds the file lock."""
    if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
        return True
    message = str(exc)
    return "unable to lock file" in message and "temporarily unavailable" in message


def _attr_equals(value: Any, expected: Any) -> bool:
    """Scalar comparison of an HDF5 attribute (False for arrays or missing values)."""
    value = _to_python(value)
    if isinstance(value, np.ndarray) or value is None:
        return False
    return bool(value == expected)


def _glob_escape(text: str) -> str:
    return "".join(f"[{c}]" if c in "*?[" else c for c in text)


def _board_path(node: int, board: int, rooted: bool = True) -> str:
    path = f"node_{node}/board_{board}"
    return f"events/{path}" if rooted else path


def _split_key(key: int) -> tuple[int, int]:
    return key >> _BOARD_KEY_SHIFT, key & ((1 << _BOARD_KEY_SHIFT) - 1)


def _group_number(name: str, prefix: str) -> int:
    if not name.startswith(prefix):
        raise KeyError(f"Unexpected group {name!r} in the UV cache (expected {prefix}<n>)")
    return int(name[len(prefix) :])


def _channel_counts(
    rena: npt.NDArray[Any], channel: npt.NDArray[Any]
) -> list[tuple[int, int, int]]:
    """Sorted ``(rena, channel, n)`` of the pairs present (O(n) bincount on a pair key)."""
    rena_i = rena.astype(np.int64)
    channel_i = channel.astype(np.int64)
    if rena_i.size == 0:
        return []
    if min(rena_i.min(), channel_i.min()) < 0 or channel_i.max() > 255:
        # Not values a cache can hold (int8 >= 0); fall back to a general sort.
        pairs, n = np.unique(np.stack([rena_i, channel_i]), axis=1, return_counts=True)
        return [(int(r), int(c), int(k)) for (r, c), k in zip(pairs.T, n)]
    counts = np.bincount(rena_i * 256 + channel_i)
    return [(int(k) // 256, int(k) % 256, int(counts[k])) for k in np.flatnonzero(counts)]


def _stop_callable(stop_flag: StopFlag | None) -> Callable[[], bool]:
    if stop_flag is None:
        return lambda: False
    if isinstance(stop_flag, threading.Event):
        return stop_flag.is_set
    if callable(stop_flag):
        return stop_flag
    raise TypeError(f"stop_flag must be a callable or threading.Event, got {type(stop_flag)}")


def _to_python(value: Any) -> Any:
    """Convert an HDF5 attribute value (numpy scalar, bytes) to a plain Python value."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _remove_quietly(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning(f"Could not remove {path}: {exc}")


def _gb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.2f} GB"
