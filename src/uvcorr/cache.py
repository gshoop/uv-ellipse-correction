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
    /results/current/           the last batch analysis run: attrs options_json,
                                created_at, uvcorr_version, results_version;
                                table (one row per channel, every CSV column)
    /results/current/overrides/ table: per-channel GUI re-fits, same columns
                                plus options_json per row

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

The build creates only ``/metadata`` and ``/events``; ``/results`` is written
by :meth:`UVCache.save_results` and the override methods (see "Results"
below). Rebuilding a cache discards its ``/results`` (a warning is logged).

Results: ``save_results`` writes a complete new group ``/results/_new`` and
only then swaps it in (``current`` -> ``_old``, ``_new`` -> ``current``,
delete ``_old``), so an exception part-way through leaves the previous
``current`` intact. The overrides table is replaced the same way
(``table_new`` -> ``table``), once per call however many overrides the call
stores or deletes (:meth:`UVCache.replace_overrides`, which
:meth:`UVCache.save_overrides` and :meth:`UVCache.delete_overrides` use).
The override writes take an optional ``expected_created_at``: the
``created_at`` of the batch results the caller loaded, checked under the
write handle, so a caller holding results that another process has since
replaced gets a :class:`StaleResultsError` instead of writing into the new
results. Readers ignore leftover ``_new`` groups
and fall back to ``_old`` if a swap was interrupted; the next write cleans
both up.
This protects against errors and ordinary interruptions, not against power
loss or a process killed inside an HDF5 call: HDF5 files are not journaled, so
such a crash can corrupt the file (the cache can always be rebuilt from the
``.dat``, but its results would be lost). HDF5 does not reclaim the space of
replaced tables; a table is ~2-3 MB for the full system, so this only
matters after very many saves (``h5repack`` compacts the file).

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
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import numpy.typing as npt

from uvcorr import __version__
from uvcorr.options import FLAG_SEPARATOR, FLAGS, STATUSES, FitOptions

if TYPE_CHECKING:
    from adc2kev.parser import EventBatch

    # uvcorr.analysis imports this module; the results code imports it lazily.
    from uvcorr.analysis import ChannelKey, ChannelResult

# The build-only imports (adc2kev's parser, and uvcorr.channels, which imports
# adc2kev.tools) happen inside the build functions: importing adc2kev pulls in
# pandas, numba, lmfit and matplotlib (~0.5 s and ~150 MB per process), which
# the analysis worker processes, which only read the cache, do not need.

logger = logging.getLogger(__name__)

__all__ = [
    "BYTES_PER_EVENT",
    "CACHE_SUFFIX",
    "EVENT_DTYPES",
    "EVENT_FIELDS",
    "RESULTS_VERSION",
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
    "ResultsError",
    "StaleResultsError",
    "StopFlag",
    "StoredOverride",
    "StoredResults",
    "UVCache",
    "UVCacheError",
    "check_disk_space",
    "compute_source_hash",
    "default_cache_path",
    "estimate_cache_bytes",
    "merge_results",
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


class ResultsError(UVCacheError):
    """Stored results are missing where required, or unreadable."""


class StaleResultsError(ResultsError):
    """The stored batch results are not the ones the caller loaded (``expected_created_at``).

    Another process (e.g. ``uvcorr process``) replaced or removed
    ``/results/current`` since the caller read it; writing overrides now
    would attach them to results the caller has not seen.
    """


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


RESULTS_VERSION = "1.0.0"
"""Layout version of ``/results`` (stored as the ``results_version`` attr)."""


@dataclass(frozen=True)
class StoredOverride:
    """One per-channel re-fit stored under ``/results/current/overrides``.

    Attributes:
        result: The channel's result (``options_source="override"``).
        options: The options it was fitted with.
    """

    result: ChannelResult
    options: FitOptions


@dataclass(frozen=True)
class StoredResults:
    """The analysis results stored in a cache (``/results/current``).

    Attributes:
        results: The batch run's rows, sorted by channel key.
        options: The batch run's options.
        created_at: ISO time the batch results were saved.
        uvcorr_version: uvcorr version that saved them.
        overrides: Per-channel re-fits by channel key (sorted).
    """

    results: tuple[ChannelResult, ...]
    options: FitOptions
    created_at: str
    uvcorr_version: str
    overrides: Mapping[ChannelKey, StoredOverride] = field(default_factory=dict)

    def merged(self) -> list[ChannelResult]:
        """The batch rows with the overrides applied (see :func:`merge_results`)."""
        return merge_results(self.results, (o.result for o in self.overrides.values()))


def merge_results(
    batch: Iterable[ChannelResult], overrides: Iterable[ChannelResult]
) -> list[ChannelResult]:
    """Apply per-channel overrides to batch results.

    An override replaces the batch row of its channel and is marked
    ``options_source="override"``; an override for a channel without a batch
    row is added. This is what the exports (CLI and GUI) write.

    Args:
        batch: The batch results.
        overrides: The override results.

    Returns:
        The merged results, sorted by channel key.
    """
    merged = {result.key: result for result in batch}
    for result in overrides:
        if result.options_source != "override":
            result = replace(result, options_source="override")
        merged[result.key] = result
    return [merged[key] for key in sorted(merged)]


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

    # ------------------------------------------------------------------
    # Results (plan 6.1)
    # ------------------------------------------------------------------

    def save_results(
        self,
        results: Iterable[ChannelResult],
        options: FitOptions,
        *,
        keep_overrides: bool = True,
        drop_overrides: Callable[[ChannelKey, FitOptions], bool] | None = None,
    ) -> int:
        """Store a batch run's results as ``/results/current``, replacing any previous ones.

        The new group is written completely before it replaces the old one
        (module docstring), so an exception leaves the previous results
        intact.

        Overrides are kept by default: they are deliberate per-channel user
        choices, computed on the same events, and stay valid across batch
        runs. Pass ``keep_overrides=False`` to drop them.

        Args:
            results: One result per channel, each with
                ``options_source="batch"``.
            options: The options of the batch run.
            keep_overrides: Carry the existing overrides over to the new
                results.
            drop_overrides: With ``keep_overrides``: ``drop_overrides(key,
                override_options)`` is called for each stored override, and
                the overrides it returns True for are not carried over (e.g.
                those fitted with the new batch options, which the new batch
                rows reproduce). Decided on the overrides as stored, in the
                same write.

        Returns:
            The number of stored overrides not carried over (every one with
            ``keep_overrides=False``).

        Raises:
            ValueError: If a result is not a batch result or two results
                have the same channel key.
            CacheBusyError: If another process holds the cache's lock.
            OSError: If the file cannot be opened for writing.
        """
        rows = sorted(results, key=lambda result: result.key)
        _check_unique_keys(rows)
        for result in rows:
            if result.options_source != "batch":
                raise ValueError(
                    f"save_results takes batch results; {result.key} has options_source "
                    f"{result.options_source!r} (store re-fits with save_override)"
                )
        table = _encode_results(rows, None)
        with _open_h5(self._path, "r+") as h5f:
            group = h5f.require_group(_RESULTS_GROUP)
            _recover_results(group)
            new = group.create_group(_RESULTS_NEW)
            try:
                attrs = new.attrs
                attrs["options_json"] = options.to_json()
                attrs["created_at"] = datetime.now().isoformat()
                attrs["uvcorr_version"] = __version__
                attrs["results_version"] = RESULTS_VERSION
                attrs["n_channels"] = np.int64(len(rows))
                new.create_dataset(_TABLE, data=table)
                current = group.get(_RESULTS_CURRENT)
                n_dropped = 0
                if current is not None and _OVERRIDES in current:
                    if not keep_overrides:
                        n_dropped = _override_count(current)
                    elif drop_overrides is None:
                        _copy_overrides(current[_OVERRIDES], new)
                    else:
                        entries = _read_override_entries(current)
                        kept = {
                            key: entry
                            for key, entry in entries.items()
                            if not drop_overrides(key, entry[1])
                        }
                        n_dropped = len(entries) - len(kept)
                        if n_dropped:
                            _write_override_entries(new, kept)
                        else:  # nothing dropped: copy the table as it is
                            _copy_overrides(current[_OVERRIDES], new)
                if current is not None:
                    group.move(_RESULTS_CURRENT, _RESULTS_OLD)
                group.move(_RESULTS_NEW, _RESULTS_CURRENT)  # commit point
            except BaseException:
                _undo_results_write(group)
                raise
            # The new results are in place; a leftover _old is harmless (readers
            # ignore it, the next write removes it), so cleanup is best-effort.
            _delete_quietly(group, _RESULTS_OLD)
        logger.info(f"Saved {len(rows)} channel results to {self._path}")
        if n_dropped:
            logger.info(f"{n_dropped} override(s) were not carried over")
        return n_dropped

    def results_created_at(self) -> str | None:
        """``created_at`` of the stored batch results (a cheap attribute read), or None.

        None when the file or its results do not exist. Compare it with
        :attr:`StoredResults.created_at` to tell whether another process has
        replaced the results since they were loaded.

        Raises:
            CacheBusyError: If another process holds the cache's lock.
        """
        if not self._path.is_file():
            return None
        with _open_h5(self._path) as h5f:
            current = _current_results(h5f)
            return None if current is None else _created_at(current)

    def load_results(self) -> StoredResults | None:
        """Load the stored results and overrides.

        Results written by a newer uvcorr still load: options are parsed
        leniently (``FitOptions.from_json(strict=False)``), unknown table
        columns are ignored, unknown flags are dropped and rows with an
        unknown status, polarity or options source are skipped, each with a
        logged warning. (Rewriting the overrides with this version, through
        ``save_override``/``delete_override``, then drops such rows for good.)

        Returns:
            The stored results, or None if the cache has none (or the file
            does not exist).

        Raises:
            ResultsError: If the stored results cannot be decoded.
            CacheBusyError: If another process holds the cache's lock.
        """
        if not self._path.is_file():
            return None
        with _open_h5(self._path) as h5f:
            current = _current_results(h5f)
            if current is None:
                return None
            try:
                attrs = current.attrs
                options = FitOptions.from_json(_attr_text(attrs["options_json"]), strict=False)
                rows = _decode_results(current[_TABLE][()], with_options=False)
                overrides = {
                    key: StoredOverride(result, opts)
                    for key, (result, opts) in _read_override_entries(current).items()
                }
                return StoredResults(
                    results=tuple(result for result, _ in rows),
                    options=options,
                    created_at=_attr_text(attrs.get("created_at", "")),
                    uvcorr_version=_attr_text(attrs.get("uvcorr_version", "")),
                    overrides=dict(sorted(overrides.items())),
                )
            except (KeyError, ValueError, TypeError) as exc:
                raise ResultsError(
                    f"The stored results in {self._path} are unreadable: {exc}"
                ) from exc

    def merged_results(self) -> list[ChannelResult]:
        """The stored batch results with the overrides applied (empty if none are stored).

        Raises:
            ResultsError: If the stored results cannot be decoded.
            CacheBusyError: If another process holds the cache's lock.
        """
        stored = self.load_results()
        return stored.merged() if stored is not None else []

    def save_override(self, result: ChannelResult, options: FitOptions) -> None:
        """Store (insert or replace) one channel's re-fit as an override.

        The result is stored with ``options_source="override"``. Same as
        ``save_overrides([result], options)``.

        Args:
            result: The channel's re-fitted result.
            options: The options it was fitted with.

        Raises:
            ResultsError: If the cache has no batch results to attach the
                override to (run a batch first).
            CacheBusyError: If another process holds the cache's lock.
        """
        self.save_overrides([result], options)

    def save_overrides(
        self,
        results: Iterable[ChannelResult],
        options: OverrideOptions,
        *,
        expected_created_at: str | None = None,
    ) -> int:
        """Store (insert or replace) the re-fits of several channels as overrides, in one write.

        See :meth:`replace_overrides` (with nothing to delete).

        Returns:
            The number of overrides stored (0 for no results; the file is
            then not opened).
        """
        saved, _ = self.replace_overrides(results, options, expected_created_at=expected_created_at)
        return saved

    def delete_override(self, key: ChannelKey | tuple[int, int, int, int]) -> bool:
        """Delete one channel's override.

        Args:
            key: ``(node, board, rena, channel)``.

        Returns:
            True if an override was deleted, False if there was none.

        Raises:
            CacheBusyError: If another process holds the cache's lock.
        """
        return self.delete_overrides([key]) > 0

    def delete_overrides(
        self,
        keys: Iterable[ChannelKey | tuple[int, int, int, int]],
        *,
        expected_created_at: str | None = None,
    ) -> int:
        """Delete the overrides of several channels, in one write.

        See :meth:`replace_overrides` (with nothing to save).

        Returns:
            The number of overrides deleted.
        """
        _, deleted = self.replace_overrides(delete=keys, expected_created_at=expected_created_at)
        return deleted

    def replace_overrides(
        self,
        save: Iterable[ChannelResult] = (),
        options: OverrideOptions | None = None,
        delete: Iterable[ChannelKey | tuple[int, int, int, int]] = (),
        *,
        expected_created_at: str | None = None,
    ) -> tuple[int, int]:
        """Store some overrides and delete others, in one write.

        The overrides table is replaced once, with the write-then-swap of
        the module docstring, so either every change is made or, after an
        exception, none (a board re-fit is one swap, not one per channel).
        Saved results are stored with ``options_source="override"`` and
        replace any override of their channel. Keys in ``delete`` without an
        override are ignored. Nothing is written when there is nothing to
        change; with nothing to save or delete the file is not opened.

        Args:
            save: The re-fitted results to store, one per channel.
            options: The options they were fitted with: one
                :class:`~uvcorr.options.FitOptions` for all, or a mapping
                from channel key to options with an entry for every result.
                Required when ``save`` is not empty.
            delete: The channels whose overrides are deleted.
            expected_created_at: The ``created_at`` of the batch results the
                caller loaded (:attr:`StoredResults.created_at`); if the
                stored results differ (or are gone), nothing is written.

        Returns:
            ``(saved, deleted)``: the numbers of overrides stored and deleted.

        Raises:
            ValueError: If two saved results have the same channel key, or a
                channel is both saved and deleted.
            KeyError: If ``options`` is a mapping without an entry for a
                saved result.
            TypeError: If results are saved without ``options``.
            ResultsError: If results are saved but the cache has no batch
                results to attach them to (run a batch first).
            StaleResultsError: If ``expected_created_at`` does not match the
                stored results.
            CacheBusyError: If another process holds the cache's lock.
            OSError: If results are saved and the file cannot be opened for
                writing.
        """
        rows = sorted(
            (
                (
                    result
                    if result.options_source == "override"
                    else replace(result, options_source="override")
                )
                for result in save
            ),
            key=lambda result: result.key,
        )
        _check_unique_keys(rows)
        per_row = _options_per_row(rows, options)
        wanted = {tuple(int(k) for k in key) for key in delete}
        both = sorted(tuple(row.key) for row in rows if tuple(row.key) in wanted)
        if both:
            raise ValueError(f"Channel(s) both saved and deleted: {both}")
        if not rows and not wanted:
            return 0, 0
        if not rows and not self._path.is_file():
            self._check_created_at(None, expected_created_at)
            return 0, 0
        with _open_h5(self._path, "r+") as h5f:
            current = _writable_current(h5f)
            self._check_created_at(current, expected_created_at)
            if current is None:
                if rows:
                    self._current_for_overrides(h5f)  # raises ResultsError
                return 0, 0
            entries = _read_override_entries(current)
            doomed = [key for key in entries if tuple(key) in wanted]
            if not rows and not doomed:
                return 0, 0
            for key in doomed:
                del entries[key]
            for row, opts in zip(rows, per_row):
                entries[row.key] = (row, opts)
            _write_override_entries(current, entries)
        if len(rows) == 1 and not doomed:
            logger.info(f"Saved the override of {rows[0].key} to {self._path}")
        else:
            logger.info(f"Saved {len(rows)} and deleted {len(doomed)} override(s) in {self._path}")
        return len(rows), len(doomed)

    def clear_overrides(self, *, expected_created_at: str | None = None) -> int:
        """Delete every override.

        Args:
            expected_created_at: As in :meth:`replace_overrides`.

        Returns:
            The number of overrides deleted.

        Raises:
            StaleResultsError: If ``expected_created_at`` does not match.
            CacheBusyError: If another process holds the cache's lock.
        """
        if not self._path.is_file():
            self._check_created_at(None, expected_created_at)
            return 0
        with _open_h5(self._path, "r+") as h5f:
            current = _writable_current(h5f)
            self._check_created_at(current, expected_created_at)
            if current is None:
                return 0
            n = _override_count(current)
            if _OVERRIDES in current:
                del current[_OVERRIDES]
        return n

    def _check_created_at(self, current: Any, expected: str | None) -> None:
        """Raise :class:`StaleResultsError` unless the stored results are the expected ones."""
        if expected is None:
            return
        found = None if current is None else _created_at(current)
        if found != expected:
            what = "were removed" if found is None else f"were replaced (saved {found})"
            raise StaleResultsError(
                f"The stored results in {self._path.name} {what} since they were loaded "
                f"(saved {expected}), probably by another process"
            )

    def _current_for_overrides(self, h5f: Any) -> Any:
        current = _writable_current(h5f)
        if current is None:
            raise ResultsError(
                f"{self._path} stores no batch results to attach an override to; "
                "run a batch analysis first"
            )
        return current


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
    from uvcorr.channels import active_channel_mask  # build-only import, see top

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

    from adc2kev.parser import PacketParser  # build-only import, see top

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
# Results internals
# ----------------------------------------------------------------------

_RESULTS_GROUP = "results"
_RESULTS_CURRENT = "current"
_RESULTS_NEW = "_new"  # a batch write in progress (or left by a crash)
_RESULTS_OLD = "_old"  # the previous results during the swap
_OVERRIDES = "overrides"
_TABLE = "table"
_TABLE_NEW = "table_new"  # an overrides write in progress
_OPTIONS_COLUMN = "options_json"
_MISSING_COUNT = -1  # a count column's None in the HDF5 table

_OverrideEntries = dict["ChannelKey", tuple["ChannelResult", FitOptions]]

OverrideOptions = FitOptions | Mapping["ChannelKey | tuple[int, int, int, int]", FitOptions]
"""The options of saved overrides: one set for all, or one per channel key."""


def _options_per_row(
    rows: list[ChannelResult], options: OverrideOptions | None
) -> list[FitOptions]:
    """The options of each saved override (see :meth:`UVCache.replace_overrides`)."""
    if not rows:
        return []
    if options is None:
        raise TypeError("Saving overrides needs their fit options")
    if isinstance(options, FitOptions):
        return [options] * len(rows)
    by_key = {tuple(int(k) for k in key): opts for key, opts in options.items()}
    per_row = []
    for row in rows:
        opts = by_key.get(tuple(row.key))
        if opts is None:
            raise KeyError(f"No fit options given for the override of {row.key}")
        per_row.append(opts)
    return per_row


def _created_at(current: Any) -> str:
    """The ``created_at`` attribute of a results group ("" if missing)."""
    return _attr_text(current.attrs.get("created_at", ""))


def _override_count(current: Any) -> int:
    """Rows in the overrides table of a results group (read from the shape, not decoded)."""
    table = _override_table(current)
    return 0 if table is None else int(table.shape[0])


def _check_unique_keys(rows: list[ChannelResult]) -> None:
    for prev, row in zip(rows, rows[1:]):
        if prev.key == row.key:
            raise ValueError(f"Duplicate results for {row.key}")


def _results_dtype(with_options: bool) -> np.dtype[Any]:
    """Compound dtype of a results table: one field per CSV column (+ options_json)."""
    from uvcorr import analysis

    text = h5py.string_dtype()  # variable-length UTF-8
    fields: list[tuple[str, Any]] = []
    for column in analysis.RESULT_COLUMNS:
        if column.kind in (analysis.KIND_INT, analysis.KIND_COUNT):
            fields.append((column.name, np.int64))
        elif column.kind in (analysis.KIND_FLOAT, analysis.KIND_FLOAT_PRECISE):
            fields.append((column.name, np.float64))
        else:
            fields.append((column.name, text))
    if with_options:
        fields.append((_OPTIONS_COLUMN, text))
    return np.dtype(fields)


def _object_array(values: list[Any]) -> npt.NDArray[np.object_]:
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def _encode_results(
    rows: list[ChannelResult], options: list[FitOptions] | None
) -> npt.NDArray[np.void]:
    """Encode results as a structured array (None: NaN for floats, -1 for counts)."""
    from uvcorr import analysis

    table = np.zeros(len(rows), dtype=_results_dtype(options is not None))
    for column in analysis.RESULT_COLUMNS:
        values = [getattr(row, column.name) for row in rows]
        kind = column.kind
        if kind == analysis.KIND_COUNT:
            table[column.name] = [_MISSING_COUNT if value is None else value for value in values]
        elif kind in (analysis.KIND_FLOAT, analysis.KIND_FLOAT_PRECISE):
            table[column.name] = [math.nan if value is None else value for value in values]
        elif kind == analysis.KIND_FLAGS:
            table[column.name] = _object_array([row.flags_text for row in rows])
        elif kind == analysis.KIND_STR:
            table[column.name] = _object_array(values)
        else:
            table[column.name] = values
    if options is not None:
        table[_OPTIONS_COLUMN] = _object_array([opts.to_json() for opts in options])
    return table


def _decode_results(
    table: npt.NDArray[np.void], *, with_options: bool
) -> list[tuple[ChannelResult, FitOptions | None]]:
    """Decode a results table; unknown columns are ignored, missing optional ones are None.

    Unknown flags are dropped and rows with an unknown status, polarity or
    options source are skipped, each with a warning, so results written by a
    newer uvcorr still load.

    Raises:
        ValueError, TypeError, KeyError: If the table cannot be decoded.
    """
    from uvcorr import analysis

    names = set(table.dtype.names or ())
    columns: dict[str, list[Any]] = {}
    for column in analysis.RESULT_COLUMNS:
        if column.name not in names:
            continue
        data = table[column.name]
        kind = column.kind
        if kind == analysis.KIND_COUNT:
            columns[column.name] = [None if v < 0 else v for v in data.astype(np.int64).tolist()]
        elif kind == analysis.KIND_INT:
            columns[column.name] = data.astype(np.int64).tolist()
        elif kind in (analysis.KIND_FLOAT, analysis.KIND_FLOAT_PRECISE):
            columns[column.name] = data.astype(np.float64).tolist()  # NaN -> None on construction
        else:
            columns[column.name] = [_attr_text(value) for value in data]
    unknown = names - set(analysis.CSV_COLUMNS) - {_OPTIONS_COLUMN}
    if unknown:
        logger.debug(f"Ignoring unknown results column(s) {sorted(unknown)}")
    options: list[FitOptions | None] = [None] * len(table)
    if with_options:
        options = [
            FitOptions.from_json(_attr_text(value), strict=False)
            for value in table[_OPTIONS_COLUMN]
        ]
    # Values a newer uvcorr may write: unknown flags are dropped, rows with an
    # unknown status, polarity or options source are skipped (with a warning).
    known_flags = set(FLAGS)
    allowed = {
        "status": set(STATUSES),
        "polarity": set(analysis.POLARITIES),
        "options_source": set(analysis.OPTIONS_SOURCES),
    }
    dropped_flags: set[str] = set()
    skipped: dict[str, set[str]] = {}
    decoded: list[tuple[ChannelResult, FitOptions | None]] = []
    for i in range(len(table)):
        row = {name: values[i] for name, values in columns.items()}
        bad = {k: row[k] for k, ok in allowed.items() if k in row and row[k] not in ok}
        if bad:
            for name, value in bad.items():
                skipped.setdefault(name, set()).add(str(value))
            continue
        if "flags" in row:
            flags = [f for f in str(row["flags"]).split(FLAG_SEPARATOR) if f]
            dropped_flags.update(f for f in flags if f not in known_flags)
            row["flags"] = tuple(f for f in flags if f in known_flags)
        decoded.append((analysis.ChannelResult.from_dict(row), options[i]))
    if dropped_flags:
        logger.warning(f"Ignoring unknown flag(s) in the stored results: {sorted(dropped_flags)}")
    if skipped:
        n_skipped = len(table) - len(decoded)
        detail = ", ".join(f"{name} {sorted(values)}" for name, values in skipped.items())
        logger.warning(f"Skipping {n_skipped} stored result row(s) with unknown {detail}")
    return decoded


def _attr_text(value: Any) -> str:
    """An HDF5 string (bytes, numpy or Python str) as a Python str."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _current_results(h5f: Any) -> Any:
    """The complete stored results group: ``current``, else an interrupted swap's ``_old``."""
    group = h5f.get(_RESULTS_GROUP)
    if group is None:
        return None
    for name in (_RESULTS_CURRENT, _RESULTS_OLD):
        candidate = group.get(name)
        if candidate is not None and _TABLE in candidate:
            return candidate
    return None


def _recover_results(group: Any) -> None:
    """Clean up after an interrupted results write (before writing again).

    An orphaned ``_old`` (no ``current``) becomes ``current`` again; leftover
    ``_new``/``_old`` groups are deleted on a best-effort basis (readers ignore
    them).
    """
    _delete_quietly(group, _RESULTS_NEW)
    if _RESULTS_OLD in group:
        if _RESULTS_CURRENT in group:
            _delete_quietly(group, _RESULTS_OLD)
        else:
            group.move(_RESULTS_OLD, _RESULTS_CURRENT)


def _writable_current(h5f: Any) -> Any:
    """``/results/current`` of a file open for writing, after recovering an interrupted write."""
    group = h5f.get(_RESULTS_GROUP)
    if group is None:
        return None
    _recover_results(group)
    return _current_results(h5f)


def _delete_quietly(group: Any, name: str) -> None:
    """Delete ``group[name]`` if present; log instead of raising (post-commit cleanup)."""
    try:
        if name in group:
            del group[name]
    except Exception as exc:
        logger.warning(f"Could not remove {group.name}/{name} (it is cleaned up later): {exc}")


def _undo_results_write(group: Any) -> None:
    """Restore the previous results after a failed :meth:`UVCache.save_results`."""
    try:
        if _RESULTS_NEW in group:
            del group[_RESULTS_NEW]
        if _RESULTS_CURRENT not in group and _RESULTS_OLD in group:
            group.move(_RESULTS_OLD, _RESULTS_CURRENT)
    except Exception as exc:  # keep the original error; readers fall back to _old
        logger.warning(f"Could not clean up after a failed results write: {exc}")


def _override_table(current: Any) -> Any:
    """The overrides table of a results group (an interrupted swap's ``table_new``), or None."""
    overrides = current.get(_OVERRIDES)
    if overrides is None:
        return None
    for name in (_TABLE, _TABLE_NEW):
        table = overrides.get(name)
        if table is not None:
            return table
    return None


def _copy_overrides(source: Any, new: Any) -> None:
    """Copy the overrides table of ``source`` (an overrides group) into group ``new``."""
    table = source.get(_TABLE)
    if table is None:
        table = source.get(_TABLE_NEW)
    if table is None:
        return
    new.create_group(_OVERRIDES).create_dataset(_TABLE, data=table[()])


def _read_override_entries(current: Any) -> _OverrideEntries:
    table = _override_table(current)
    if table is None:
        return {}
    entries: _OverrideEntries = {}
    for result, options in _decode_results(table[()], with_options=True):
        assert options is not None
        entries[result.key] = (result, options)
    return entries


def _write_override_entries(current: Any, entries: _OverrideEntries) -> None:
    """Replace the overrides table (``table_new`` is written first, then swapped in)."""
    if not entries:
        if _OVERRIDES in current:
            del current[_OVERRIDES]
        return
    keys = sorted(entries)
    table = _encode_results([entries[key][0] for key in keys], [entries[key][1] for key in keys])
    overrides = current.require_group(_OVERRIDES)
    if _TABLE_NEW in overrides:
        if _TABLE in overrides:
            del overrides[_TABLE_NEW]  # left by a failed write; `table` is current
        else:
            overrides.move(_TABLE_NEW, _TABLE)  # committed but never renamed
    try:
        overrides.create_dataset(_TABLE_NEW, data=table)
    except BaseException:
        if _TABLE_NEW in overrides:
            del overrides[_TABLE_NEW]
        raise
    try:
        if _TABLE in overrides:
            del overrides[_TABLE]  # commit point: readers now use table_new
    except BaseException:
        _delete_quietly(overrides, _TABLE_NEW)
        raise
    try:
        overrides.move(_TABLE_NEW, _TABLE)
    except Exception as exc:  # committed already: readers fall back to table_new
        logger.warning(f"Could not rename the new overrides table (it is used as is): {exc}")


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
