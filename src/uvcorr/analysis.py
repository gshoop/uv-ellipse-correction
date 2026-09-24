"""Per-channel, per-board and whole-system analysis driver (plan sections 4.1, 5 and 7).

- :class:`ChannelKey`: the ``(node, board, rena, channel)`` address of a channel.
- :class:`ChannelResult`: every ``radial_summary.csv`` field of one channel
  (plan 6.2). Its dataclass fields *are* the CSV columns, in CSV order and
  with the CSV names (``centerU``, ``semiMajor``, ... keep the ``.tec`` key
  spelling). :data:`RESULT_COLUMNS` derives the column list from the fields,
  so the CSV writer, the HDF5 results table and the GUI inspector share one
  source of truth.
- :func:`analyze_channel`: fit, correct and measure one channel.
- :func:`analyze_board`: every active channel of one cached board.
- :func:`analyze_all`: every board of a cache, one process-pool task per board.

``FitOptions`` is defined in :mod:`uvcorr.options` (``uvcorr.ellipse`` needs it
and this module imports ``uvcorr.ellipse``) and re-exported here.

Unavailable values: a metric that is not available (a failed channel, a
Gaussian fit that reports no chi2, phase metrics below 8 points, an undefined
skewness) is ``None`` in :class:`ChannelResult`, never NaN. Non-finite floats
are normalised to ``None`` on construction. On disk it is an empty CSV cell,
NaN in a float column of the HDF5 table and -1 in its count columns.

Process pool and BLAS threads: :func:`analyze_all` runs one task per board in
a ``ProcessPoolExecutor`` with the ``spawn`` start method. ``spawn`` (not
Linux's default ``fork``) is safe from multi-threaded parents such as the GUI,
whose ``QThread`` runs the analysis. The worker initializer limits the
worker's BLAS/OpenMP thread pools to one thread with ``threadpoolctl`` (no
environment variables are touched: ``setenv`` from a non-main thread is not
thread-safe). numpy's bundled OpenBLAS otherwise uses one thread per core in
every worker, which oversubscribes the machine on the 6xN scatter products;
single-threaded workers were 15-18 % faster on the full test acquisition.
It also ignores Ctrl-C (the parent handles it). ``workers=1`` runs in-process
(no pool, the caller's BLAS settings), which is handy for tests and
debugging.

Default worker count: ``min(8, usable CPUs)`` (plan 7). See
:func:`default_workers` for the timings and memory measured on the test
acquisition.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import signal
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from uvcorr.cache import UVCache
from uvcorr.ellipse import (
    EllipseParams,
    correct,
    corrected_residual,
    fit_ellipse,
    radii_about_center,
    residual_to_ellipse,
)
from uvcorr.metrics import phase_stats, radial_stats, residual_stats, timing_jitter_ns
from uvcorr.options import (
    FLAG_GAUSS_FIT_FAILED_POST,
    FLAG_GAUSS_FIT_FAILED_PRE,
    FLAG_SEPARATOR,
    STATUS_OK,
    STATUSES,
    FitOptions,
    order_flags,
)

logger = logging.getLogger(__name__)

__all__ = [
    "COLUMN_KINDS",
    "CSV_COLUMNS",
    "KIND_COUNT",
    "KIND_FLAGS",
    "KIND_FLOAT",
    "KIND_FLOAT_PRECISE",
    "KIND_INT",
    "KIND_STR",
    "OPTIONS_BATCH",
    "OPTIONS_OVERRIDE",
    "OPTIONS_SOURCES",
    "POLARITIES",
    "RESULT_COLUMNS",
    "AnalysisCancelled",
    "AnalysisError",
    "BoardProgressCallback",
    "WorkerCrashedError",
    "ChannelKey",
    "ChannelResult",
    "FitOptions",
    "ResultColumn",
    "analyze_all",
    "analyze_board",
    "analyze_channel",
    "default_workers",
]

OPTIONS_BATCH = "batch"
"""``options_source`` of a result computed by a batch run with the run's options."""

OPTIONS_OVERRIDE = "override"
"""``options_source`` of a per-channel re-fit with its own options (a GUI override)."""

OPTIONS_SOURCES: tuple[str, ...] = (OPTIONS_BATCH, OPTIONS_OVERRIDE)

POLARITIES: tuple[str, ...] = ("anode", "cathode")

MAX_DEFAULT_WORKERS = 8
"""Upper limit of :func:`default_workers`."""

BoardProgressCallback = Callable[[int, int], None]
"""``progress_cb(done_events, total_events)`` of :func:`analyze_all`.

The units are *events*: the events of the boards finished so far and of all
boards, so ``done / total`` is a smooth progress fraction even though the
boards differ in size by 5x and run largest first."""

StopFlag = Callable[[], bool] | threading.Event
"""A callable returning True, or a set ``threading.Event``, requests a stop."""

FloatArray = npt.NDArray[np.float64]


class AnalysisCancelled(Exception):
    """The analysis was stopped through its ``stop_flag``; no results are returned."""


class AnalysisError(Exception):
    """The analysis failed (in a worker process or in-process).

    The original exception is chained as ``__cause__`` (for a worker, with the
    remote traceback).

    Attributes:
        node: Node of the board that failed, or None if the failure cannot
            be attributed to a board (:class:`WorkerCrashedError`).
        board: Board number, or None.
    """

    def __init__(self, message: str, node: int | None = None, board: int | None = None) -> None:
        super().__init__(message)
        self.node = node
        self.board = board

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (str(self), self.node, self.board))


WORKER_CRASHED_MESSAGE = (
    "a worker process terminated abruptly (killed, e.g. out of memory, or crashed); "
    "rerun with --workers 1 (workers=1) to locate the board"
)


class WorkerCrashedError(AnalysisError):
    """A worker process died without reporting an error (killed, out of memory, segfault).

    The process pool is then broken and every pending board fails at once, so
    the board being analysed cannot be identified: ``node`` and ``board`` are
    None. Rerunning with ``workers=1`` runs in-process and finds the board.
    """

    def __init__(
        self,
        message: str = WORKER_CRASHED_MESSAGE,
        node: int | None = None,
        board: int | None = None,
    ) -> None:
        super().__init__(message, node, board)


# ---------------------------------------------------------------------------
# Channel key and result
# ---------------------------------------------------------------------------


class ChannelKey(NamedTuple):
    """Address of one channel; hashes and sorts like the plain 4-tuple."""

    node: int
    board: int
    rena: int
    channel: int

    def __str__(self) -> str:
        return f"node {self.node} board {self.board} rena {self.rena} channel {self.channel}"


KIND_INT = "int"
"""Always-present integer (identity columns, ``n_events``)."""

KIND_COUNT = "count"
"""Integer count or None (``n_used``, ``n_rejected``)."""

KIND_STR = "str"
"""Text (``polarity``, ``electrode``, ``status``, ``options_source``)."""

KIND_FLAGS = "flags"
"""Tuple of flag strings; ``;``-joined in the CSV and the HDF5 table."""

KIND_FLOAT = "float"
"""Finite float or None; ``%.6g`` in the CSV."""

KIND_FLOAT_PRECISE = "float_precise"
"""Finite float or None; ``%.9g`` in the CSV (centre, semi-axes, phi)."""

COLUMN_KINDS: tuple[str, ...] = (
    KIND_INT,
    KIND_COUNT,
    KIND_STR,
    KIND_FLAGS,
    KIND_FLOAT,
    KIND_FLOAT_PRECISE,
)


def _col(kind: str, **kwargs: Any) -> Any:
    """Declare a :class:`ChannelResult` field of a column kind."""
    return field(metadata={"kind": kind}, **kwargs)


def _opt(kind: str) -> Any:
    """Declare an optional (default None) :class:`ChannelResult` field."""
    return _col(kind, default=None)


@dataclass(frozen=True, kw_only=True)
class ChannelResult:
    """Every ``radial_summary.csv`` field of one channel (plan 6.2), in column order.

    The field names are the CSV column names. Values that are not available
    are None (see the module docstring); for a channel whose status is not
    ``ok`` only the identity fields, ``status``, ``flags``, ``n_events`` and
    ``options_source`` are set.

    Construction normalises the values: numpy scalars become Python scalars,
    non-finite floats become None, ``flags`` becomes a tuple in the canonical
    ``uvcorr.options.FLAGS`` order.

    Attributes:
        node, board, rena, channel: The channel address.
        polarity: ``"anode"`` or ``"cathode"``.
        electrode: Electrode label, e.g. ``"A17"`` or ``"C03"``.
        status: One of ``uvcorr.options.STATUSES``.
        flags: Warning flags (``uvcorr.options.FLAGS``), canonical order.
        n_events: Events of the channel.
        n_used: Events the final ellipse fit used.
        n_rejected: ``n_events - n_used``.
        centerU, centerV: Fitted ellipse centre (ADC).
        semiMajor, semiMinor: Semi-axes a >= b (ADC).
        phi: Angle of the major axis, in (-pi/2, pi/2].
        axis_ratio: b/a.
        target_radius: sqrt(ab), the radius of the corrected circle.
        pre_mean ... pre_robust_sigma: Radial statistics of the raw points about
            the fitted centre (Gaussian mean/sigma/FWHM/chi2ndf, unbinned
            skewness and excess kurtosis, 1.4826 MAD).
        post_mean ... post_robust_sigma: The same for the corrected points.
        rawfit_res_mean, rawfit_res_sigma: Gaussian stats of the raw radial
            residuals to the ellipse.
        corr_res_mean, corr_res_sigma: Gaussian stats of the corrected radial
            residuals to the target circle.
        phase_mean_gap_rad, phase_max_gap_rad, phase_max_gap_ns, phase_ks:
            Sorted-phase metrics of the corrected points (N >= 8).
        timing_jitter_ns: ``post_sigma / (2 pi f post_mean)`` in ns.
        options_source: ``"batch"`` or ``"override"``.
    """

    node: int = _col(KIND_INT)
    board: int = _col(KIND_INT)
    rena: int = _col(KIND_INT)
    channel: int = _col(KIND_INT)
    polarity: str = _col(KIND_STR)
    electrode: str = _col(KIND_STR)
    status: str = _col(KIND_STR)
    flags: tuple[str, ...] = _col(KIND_FLAGS, default=())
    n_events: int = _col(KIND_INT)
    n_used: int | None = _opt(KIND_COUNT)
    n_rejected: int | None = _opt(KIND_COUNT)
    centerU: float | None = _opt(KIND_FLOAT_PRECISE)
    centerV: float | None = _opt(KIND_FLOAT_PRECISE)
    semiMajor: float | None = _opt(KIND_FLOAT_PRECISE)
    semiMinor: float | None = _opt(KIND_FLOAT_PRECISE)
    phi: float | None = _opt(KIND_FLOAT_PRECISE)
    axis_ratio: float | None = _opt(KIND_FLOAT)
    target_radius: float | None = _opt(KIND_FLOAT)
    pre_mean: float | None = _opt(KIND_FLOAT)
    pre_sigma: float | None = _opt(KIND_FLOAT)
    pre_fwhm: float | None = _opt(KIND_FLOAT)
    pre_chi2ndf: float | None = _opt(KIND_FLOAT)
    pre_skewness: float | None = _opt(KIND_FLOAT)
    pre_kurtosis: float | None = _opt(KIND_FLOAT)
    pre_robust_sigma: float | None = _opt(KIND_FLOAT)
    post_mean: float | None = _opt(KIND_FLOAT)
    post_sigma: float | None = _opt(KIND_FLOAT)
    post_fwhm: float | None = _opt(KIND_FLOAT)
    post_chi2ndf: float | None = _opt(KIND_FLOAT)
    post_skewness: float | None = _opt(KIND_FLOAT)
    post_kurtosis: float | None = _opt(KIND_FLOAT)
    post_robust_sigma: float | None = _opt(KIND_FLOAT)
    rawfit_res_mean: float | None = _opt(KIND_FLOAT)
    rawfit_res_sigma: float | None = _opt(KIND_FLOAT)
    corr_res_mean: float | None = _opt(KIND_FLOAT)
    corr_res_sigma: float | None = _opt(KIND_FLOAT)
    phase_mean_gap_rad: float | None = _opt(KIND_FLOAT)
    phase_max_gap_rad: float | None = _opt(KIND_FLOAT)
    phase_max_gap_ns: float | None = _opt(KIND_FLOAT)
    phase_ks: float | None = _opt(KIND_FLOAT)
    timing_jitter_ns: float | None = _opt(KIND_FLOAT)
    options_source: str = _col(KIND_STR, default=OPTIONS_BATCH)

    def __post_init__(self) -> None:
        """Normalise the values (see the class docstring) and validate the enumerations.

        Raises:
            ValueError: If ``status``, ``polarity``, ``options_source`` or a
                flag is not a known value, or a count is negative.
            TypeError: If a value has the wrong type.
        """
        for column in RESULT_COLUMNS:
            name, kind = column.name, column.kind
            object.__setattr__(self, name, _normalise(name, kind, getattr(self, name)))
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if self.polarity not in POLARITIES:
            raise ValueError(f"polarity must be one of {POLARITIES}, got {self.polarity!r}")
        if self.options_source not in OPTIONS_SOURCES:
            raise ValueError(
                f"options_source must be one of {OPTIONS_SOURCES}, got {self.options_source!r}"
            )

    @property
    def key(self) -> ChannelKey:
        """The channel address."""
        return ChannelKey(self.node, self.board, self.rena, self.channel)

    @property
    def ok(self) -> bool:
        """True if the ellipse fit succeeded (``status == "ok"``)."""
        return self.status == STATUS_OK

    @property
    def rejected_fraction(self) -> float | None:
        """``n_rejected / n_events`` (not a CSV column; used by the System Map)."""
        if self.n_rejected is None or self.n_events <= 0:
            return None
        return self.n_rejected / self.n_events

    @property
    def params(self) -> EllipseParams | None:
        """The fitted ellipse, or None if it is not available."""
        cx, cy, a, b, phi = self.centerU, self.centerV, self.semiMajor, self.semiMinor, self.phi
        if cx is None or cy is None or a is None or b is None or phi is None:
            return None
        return EllipseParams(cx=cx, cy=cy, a=a, b=b, phi=phi)

    @property
    def flags_text(self) -> str:
        """The flags joined with ``;`` (the CSV / HDF5 representation)."""
        return FLAG_SEPARATOR.join(self.flags)

    def to_dict(self) -> dict[str, Any]:
        """Return ``{column name: value}`` in column order (flags as a tuple)."""
        return {column.name: getattr(self, column.name) for column in RESULT_COLUMNS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, strict: bool = True) -> ChannelResult:
        """Build a result from ``{column name: value}``.

        ``flags`` may be a tuple, a list or the ``;``-joined text.

        Args:
            data: The values. Optional columns that are missing are None
                (``options_source`` defaults to ``"batch"``).
            strict: Raise on keys that are not columns instead of ignoring
                them (ignored keys are logged).

        Returns:
            The result.

        Raises:
            ValueError: If ``strict`` and ``data`` has unknown keys, or a value
                is invalid.
            TypeError: If a required column is missing or a value has the wrong
                type.
        """
        unknown = sorted(set(data) - set(CSV_COLUMNS))
        if unknown:
            if strict:
                raise ValueError(f"Unknown ChannelResult column(s): {unknown}")
            logger.warning("Ignoring unknown ChannelResult column(s): %s", unknown)
        values = {name: data[name] for name in CSV_COLUMNS if name in data}
        flags = values.get("flags")
        if isinstance(flags, str):
            values["flags"] = tuple(part for part in flags.split(FLAG_SEPARATOR) if part)
        return cls(**values)


class ResultColumn(NamedTuple):
    """One ``radial_summary.csv`` column: its name and kind (``KIND_*``)."""

    name: str
    kind: str


RESULT_COLUMNS: tuple[ResultColumn, ...] = tuple(
    ResultColumn(f.name, f.metadata["kind"]) for f in fields(ChannelResult)
)
"""The ``radial_summary.csv`` columns in order, derived from :class:`ChannelResult`."""

CSV_COLUMNS: tuple[str, ...] = tuple(column.name for column in RESULT_COLUMNS)
"""The ``radial_summary.csv`` header (plan 6.2)."""


def _normalise(name: str, kind: str, value: Any) -> Any:
    """Normalise one :class:`ChannelResult` value of a column kind."""
    if kind == KIND_INT:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"ChannelResult.{name} must be an int, got {value!r}")
        return int(value)
    if kind == KIND_COUNT:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"ChannelResult.{name} must be an int or None, got {value!r}")
        if value < 0:
            raise ValueError(f"ChannelResult.{name} must be >= 0, got {value}")
        return int(value)
    if kind in (KIND_FLOAT, KIND_FLOAT_PRECISE):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"ChannelResult.{name} must be a float or None, got {value!r}")
        number = float(value)
        return number if math.isfinite(number) else None
    if kind == KIND_STR:
        if not isinstance(value, str):
            raise TypeError(f"ChannelResult.{name} must be a str, got {value!r}")
        return value
    if kind == KIND_FLAGS:
        if isinstance(value, str):
            raise TypeError(f"ChannelResult.{name} must be a sequence of flags, got {value!r}")
        return order_flags(value)
    raise AssertionError(f"unknown column kind {kind!r}")


# ---------------------------------------------------------------------------
# Electrode labels
# ---------------------------------------------------------------------------

ChannelIdentity = tuple[str, str]
"""``(polarity, electrode label)`` of an active channel."""

# (board, rena, channel) -> identity, or None for an inactive channel. Filled
# lazily from uvcorr.channels, or seeded by the parent in pool workers:
# uvcorr.channels imports adc2kev.tools, whose package import pulls in pandas,
# numba, lmfit and matplotlib (~0.35 s and ~80 MB PSS / ~150 MB RSS per
# worker; 0.65 GB less at 8 workers), which the workers otherwise do not need.
_identities: dict[tuple[int, int, int], ChannelIdentity | None] = {}


def _channel_identity(board: int, rena: int, channel: int) -> ChannelIdentity | None:
    """Polarity and electrode label of a channel, or None if it is not active."""
    key = (board, rena, channel)
    try:
        return _identities[key]
    except KeyError:
        pass
    from uvcorr import channels  # heavy (adc2kev.tools), see above

    identity: ChannelIdentity | None = None
    if channels.is_active_channel(rena, channel):
        identity = (
            channels.polarity_name(board, rena, channel),
            channels.electrode_label(board, rena, channel),
        )
    _identities[key] = identity
    return identity


def _identity_table(
    boards: Iterable[int],
) -> dict[tuple[int, int, int], ChannelIdentity | None]:
    """Identities of every active channel of the given boards (computed in the parent)."""
    from uvcorr.channels import ACTIVE_CHANNELS

    return {
        (board, rena, channel): _channel_identity(board, rena, channel)
        for board in sorted(set(boards))
        for rena, channel in ACTIVE_CHANNELS
    }


# ---------------------------------------------------------------------------
# Per-channel analysis
# ---------------------------------------------------------------------------


def analyze_channel(
    key: ChannelKey | tuple[int, int, int, int],
    u: npt.ArrayLike,
    v: npt.ArrayLike,
    options: FitOptions | None = None,
) -> ChannelResult:
    """Fit, correct and measure one channel (plan 4.1 and 5).

    1. :func:`~uvcorr.ellipse.fit_ellipse`. A ``too_few_events`` or
       ``fit_failed`` channel gets only its identity fields, status, flags
       and ``n_events``.
    2. Radii of the raw points about the *fitted* centre, radial statistics
       (pre); the corrected points (:func:`~uvcorr.ellipse.correct`), their
       radii and radial statistics (post); Gaussian stats of the raw
       residuals to the ellipse and of the corrected residuals to the target
       circle; phase metrics of the corrected points if there are at least 8;
       the timing-jitter proxy from the post mean and sigma.
    3. A pre or post radial Gaussian fit that failed (``ok=False``; its mean
       and sigma are then the sample mean and standard deviation) adds the
       ``gauss_fit_failed_pre`` / ``gauss_fit_failed_post`` flag to the fit's
       flags.

    The metrics are computed over **all** finite events of the channel, not
    only the subset the robust fit kept: the correction is applied to every
    event downstream, so the metrics describe what the ``.tec`` consumers get,
    and the Gaussian fits are themselves robust to the outliers the fit
    rejected (``n_rejected`` reports how many there were).

    Args:
        key: The channel address ``(node, board, rena, channel)``; it must be
            an active channel.
        u: The channel's U values (any real dtype, e.g. the int16 cache
            column).
        v: The channel's V values, same length.
        options: Fit options (default ``FitOptions()``).

    Returns:
        The channel's result with ``options_source="batch"``.

    Raises:
        ValueError: If the channel is not active, or ``u`` and ``v`` are not
            1-D arrays of equal length.
    """
    key = ChannelKey(*(int(k) for k in key))
    identity_of_channel = _channel_identity(key.board, key.rena, key.channel)
    if identity_of_channel is None:
        raise ValueError(f"{key} is not an active channel")
    polarity, electrode = identity_of_channel
    opts = options if options is not None else FitOptions()
    x = np.asarray(u, dtype=np.float64)
    y = np.asarray(v, dtype=np.float64)
    fit = fit_ellipse(x, y, opts)
    identity: dict[str, Any] = {
        "node": key.node,
        "board": key.board,
        "rena": key.rena,
        "channel": key.channel,
        "polarity": polarity,
        "electrode": electrode,
        "n_events": fit.n_events,
    }
    params = fit.params
    if not fit.ok or params is None:
        return ChannelResult(status=fit.status, flags=fit.flags, **identity)

    pre = radial_stats(radii_about_center(x, y, params))
    u_corr, v_corr = correct(x, y, params)
    post = radial_stats(np.hypot(u_corr, v_corr))
    raw_res = residual_stats(residual_to_ellipse(x, y, params))
    corr_res = residual_stats(corrected_residual(u_corr, v_corr, params))
    phase = phase_stats(u_corr, v_corr, opts.phase_ref_freq_hz)

    flags = list(fit.flags)
    if not pre.ok:
        flags.append(FLAG_GAUSS_FIT_FAILED_PRE)
    if not post.ok:
        flags.append(FLAG_GAUSS_FIT_FAILED_POST)

    return ChannelResult(
        status=fit.status,
        flags=order_flags(flags),
        n_used=fit.n_used,
        n_rejected=fit.n_rejected,
        centerU=params.cx,
        centerV=params.cy,
        semiMajor=params.a,
        semiMinor=params.b,
        phi=params.phi,
        axis_ratio=params.axis_ratio,
        target_radius=params.target_radius,
        pre_mean=pre.mean,
        pre_sigma=pre.sigma,
        pre_fwhm=pre.fwhm,
        pre_chi2ndf=pre.chi2ndf,
        pre_skewness=pre.skewness,
        pre_kurtosis=pre.kurtosis,
        pre_robust_sigma=pre.robust_sigma,
        post_mean=post.mean,
        post_sigma=post.sigma,
        post_fwhm=post.fwhm,
        post_chi2ndf=post.chi2ndf,
        post_skewness=post.skewness,
        post_kurtosis=post.kurtosis,
        post_robust_sigma=post.robust_sigma,
        rawfit_res_mean=raw_res.mean,
        rawfit_res_sigma=raw_res.sigma,
        corr_res_mean=corr_res.mean,
        corr_res_sigma=corr_res.sigma,
        phase_mean_gap_rad=phase.mean_gap_rad if phase is not None else None,
        phase_max_gap_rad=phase.max_gap_rad if phase is not None else None,
        phase_max_gap_ns=phase.max_gap_ns if phase is not None else None,
        phase_ks=phase.ks if phase is not None else None,
        timing_jitter_ns=timing_jitter_ns(post.sigma, post.mean, opts.phase_ref_freq_hz),
        **identity,
    )


# ---------------------------------------------------------------------------
# Per-board analysis
# ---------------------------------------------------------------------------


def _stop_callable(stop_flag: StopFlag | None) -> Callable[[], bool]:
    if stop_flag is None:
        return lambda: False
    if isinstance(stop_flag, threading.Event):
        return stop_flag.is_set
    if callable(stop_flag):
        return stop_flag
    raise TypeError(f"stop_flag must be a callable or threading.Event, got {type(stop_flag)}")


def _as_cache(cache: UVCache | str | Path) -> UVCache:
    return cache if isinstance(cache, UVCache) else UVCache(cache)


def analyze_board(
    cache_path: UVCache | str | Path,
    node: int,
    board: int,
    options: FitOptions | None = None,
    *,
    stop_flag: StopFlag | None = None,
) -> list[ChannelResult]:
    """Analyse every active channel of one cached board.

    The board is loaded once; its events are grouped by channel with one
    stable sort (file order is kept within a channel). There is one result
    per active channel with at least one event (plan 6.2), sorted by key.

    Args:
        cache_path: The UV cache (path or :class:`~uvcorr.cache.UVCache`).
        node: Node number.
        board: Board number.
        options: Fit options (default ``FitOptions()``).
        stop_flag: A callable returning True, or a ``threading.Event`` that
            is set, to request a stop; checked before every channel.

    Returns:
        The channel results, sorted by (rena, channel).

    Raises:
        KeyError: If the cache has no events for this board.
        AnalysisCancelled: If ``stop_flag`` requested a stop.
        CacheBusyError: If another process holds the cache's lock.
    """
    opts = options if options is not None else FitOptions()
    should_stop = _stop_callable(stop_flag)
    data = _as_cache(cache_path).load_board(node, board)
    results: list[ChannelResult] = []
    for rena, channel, u, v in _split_channels(data.rena, data.channel, data.u, data.v):
        if should_stop():
            raise AnalysisCancelled(f"Analysis of node {node} board {board} cancelled")
        if _channel_identity(board, rena, channel) is None:
            logger.debug(f"Skipping inactive channel {rena}/{channel} on node {node} board {board}")
            continue
        results.append(analyze_channel(ChannelKey(node, board, rena, channel), u, v, opts))
    return results


def _split_channels(
    rena: npt.NDArray[Any], channel: npt.NDArray[Any], u: npt.NDArray[Any], v: npt.NDArray[Any]
) -> Iterator[tuple[int, int, npt.NDArray[Any], npt.NDArray[Any]]]:
    """Yield ``(rena, channel, u, v)`` per channel with events, sorted, file order kept."""
    if u.shape[0] == 0:
        return
    pair = rena.astype(np.int32) * 256 + channel.astype(np.int32)
    order = np.argsort(pair, kind="stable")
    pair_sorted = pair[order]
    u_sorted = u[order]
    v_sorted = v[order]
    bounds = np.flatnonzero(pair_sorted[1:] != pair_sorted[:-1]) + 1
    starts = np.concatenate(([0], bounds)).tolist()
    stops = np.concatenate((bounds, [pair_sorted.shape[0]])).tolist()
    for start, stop in zip(starts, stops):
        key = int(pair_sorted[start])
        yield key // 256, key % 256, u_sorted[start:stop], v_sorted[start:stop]


# ---------------------------------------------------------------------------
# Whole-system analysis
# ---------------------------------------------------------------------------


def default_workers() -> int:
    """Default worker processes: ``min(8, usable CPUs)``.

    Usable CPUs are ``len(os.sched_getaffinity(0))`` where available (it
    honours ``taskset``/cgroup CPU sets), else ``os.cpu_count()``.

    Measured on the full test acquisition (208M events, 155 boards, 24 cores,
    2026-09-24): 1/4/8/12/16 workers took about 114/33/16/12.5/11 s with a peak
    PSS of the whole process tree of 0.5/1.1/1.9/2.5/3.1 GB (~0.17 GB per
    worker: the scientific libraries, one board's arrays and the per-channel
    float64 temporaries). 8 keeps the run far below the 2-minute target at a
    moderate memory cost.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # not on every platform
        cpus = os.cpu_count() or 1
    return max(1, min(MAX_DEFAULT_WORKERS, cpus))


# Stop event of a worker process (set by ``_init_worker``).
_worker_stop: Any = None
# threadpoolctl limiter of a worker process (kept alive for the process lifetime).
_worker_blas_limit: Any = None


def _init_worker(
    stop_event: Any,
    identities: Mapping[tuple[int, int, int], ChannelIdentity | None] | None = None,
) -> None:
    """Pool initializer: single-threaded BLAS, Ctrl-C ignored, stop event and labels.

    SIGINT arrives blocked (the parent blocks it while it starts the pool, and
    the mask is inherited across ``exec``), so a Ctrl-C during start-up is
    pending here; it is discarded when SIGINT is ignored and then unblocked.
    ``identities`` (from the parent) spares the worker the heavy import of
    ``uvcorr.channels``/adc2kev.
    """
    global _worker_stop, _worker_blas_limit
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    _worker_stop = stop_event
    if identities:
        _identities.update(identities)
    from threadpoolctl import threadpool_limits

    _worker_blas_limit = threadpool_limits(limits=1)


def _worker_diagnostics() -> dict[str, Any]:
    """Thread-pool sizes and heavy imports of the current process (for tests)."""
    import sys

    from threadpoolctl import threadpool_info

    return {
        "blas_threads": [int(info["num_threads"]) for info in threadpool_info()],
        "adc2kev_loaded": any(name.split(".")[0] == "adc2kev" for name in sys.modules),
        "sigint_ignored": signal.getsignal(signal.SIGINT) is signal.SIG_IGN,
    }


def _board_task(cache_path: str, node: int, board: int, options: FitOptions) -> list[ChannelResult]:
    """Pool task: :func:`analyze_board` with the worker's stop event."""
    stop = _worker_stop.is_set if _worker_stop is not None else None
    return analyze_board(cache_path, node, board, options, stop_flag=stop)


@contextmanager
def _sigint_blocked() -> Iterator[None]:
    """Block SIGINT in this thread (and in processes it starts) for the duration.

    A Ctrl-C meanwhile stays pending and is delivered when the block exits.
    """
    if not hasattr(signal, "pthread_sigmask"):
        yield
        return
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def schedule_boards(counts: Mapping[tuple[int, int], int]) -> list[tuple[int, int]]:
    """Order boards for :func:`analyze_all`: largest first, then by (node, board).

    Longest-processing-time-first scheduling keeps the tail of a pooled run
    short (the largest board has 5x the median's events).
    """
    return sorted(counts, key=lambda nb: (-counts[nb], nb))


def analyze_all(
    cache: UVCache | str | Path,
    options: FitOptions | None = None,
    workers: int | None = None,
    progress_cb: BoardProgressCallback | None = None,
    stop_flag: StopFlag | None = None,
) -> list[ChannelResult]:
    """Analyse every board of a UV cache.

    With more than one worker, boards run as tasks of a ``spawn``
    ``ProcessPoolExecutor`` with single-threaded BLAS (module docstring),
    largest boards first so the tail of the run stays short. With
    ``workers=1`` the boards run in-process, in order.

    Args:
        cache: The UV cache (object or path).
        options: Fit options (default ``FitOptions()``).
        workers: Worker processes (default :func:`default_workers`); never
            more than the number of boards.
        progress_cb: Called as ``progress_cb(done_events, total_events)``
            (event counts, see :data:`BoardProgressCallback`): once with 0
            before the first board, then after every board.
        stop_flag: A callable returning True, or a ``threading.Event`` that
            is set, to request a stop. It is polled about every 0.1 s (and
            between channels in-process); running boards stop at their next
            channel and pending boards are cancelled.

    Returns:
        The results of every board, sorted by :class:`ChannelKey`.

    Raises:
        ValueError: If ``workers`` is < 1.
        AnalysisCancelled: If ``stop_flag`` requested a stop.
        AnalysisError: If a board failed (``node``/``board`` identify it; the
            worker's exception is the ``__cause__``).
        WorkerCrashedError: If a worker process died abruptly (an
            ``AnalysisError`` without a board).
        CacheBusyError: If the cache is locked by another process when the
            board list is read.
    """
    cache_obj = _as_cache(cache)
    opts = options if options is not None else FitOptions()
    should_stop = _stop_callable(stop_flag)
    n_workers = default_workers() if workers is None else int(workers)
    if n_workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    counts = cache_obj.board_event_counts()
    boards = schedule_boards(counts)
    progress = _Progress(counts, progress_cb)
    progress.start()
    if not boards:
        return []
    n_workers = min(n_workers, len(boards))
    if n_workers == 1:
        results = _analyze_in_process(cache_obj, boards, opts, progress, should_stop)
    else:
        results = _analyze_in_pool(cache_obj, boards, opts, n_workers, progress, should_stop)
    results.sort(key=lambda result: result.key)
    return results


def _board_error(node: int, board: int, exc: BaseException) -> AnalysisError:
    return AnalysisError(
        f"Analysis of node {node} board {board} failed: {type(exc).__name__}: {exc}", node, board
    )


class _Progress:
    """Event-weighted progress reporting for :func:`analyze_all`."""

    def __init__(
        self, counts: Mapping[tuple[int, int], int], callback: BoardProgressCallback | None
    ) -> None:
        self._counts = counts
        self._callback = callback
        self._total = int(sum(counts.values()))
        self._done = 0

    def start(self) -> None:
        if self._callback is not None:
            self._callback(0, self._total)

    def board_done(self, node: int, board: int) -> None:
        self._done += int(self._counts[(node, board)])
        if self._callback is not None:
            self._callback(self._done, self._total)


def _analyze_in_process(
    cache: UVCache,
    boards: list[tuple[int, int]],
    options: FitOptions,
    progress: _Progress,
    should_stop: Callable[[], bool],
) -> list[ChannelResult]:
    results: list[ChannelResult] = []
    for node, board in boards:
        if should_stop():
            raise AnalysisCancelled("Analysis cancelled")
        try:
            results.extend(analyze_board(cache, node, board, options, stop_flag=should_stop))
        except AnalysisCancelled:
            raise
        except Exception as exc:
            raise _board_error(node, board, exc) from exc
        progress.board_done(node, board)
    return results


# Seconds between checks of the stop flag while waiting for the pool.
_POLL_SECONDS = 0.1


def _analyze_in_pool(
    cache: UVCache,
    boards: list[tuple[int, int]],
    options: FitOptions,
    n_workers: int,
    progress: _Progress,
    should_stop: Callable[[], bool],
) -> list[ChannelResult]:
    ctx = multiprocessing.get_context("spawn")
    stop_event = ctx.Event()
    results: list[ChannelResult] = []
    futures: dict[Future[list[ChannelResult]], tuple[int, int]] = {}
    completed = False
    # Workers are started by submit(). SIGINT is blocked meanwhile so that a Ctrl-C
    # cannot hit a worker before its initializer ignores it (a traceback per
    # worker); the parent receives it when the block ends.
    with _sigint_blocked():
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(stop_event, _identity_table(board for _, board in boards)),
        )
    try:
        with _sigint_blocked():
            for node, board in boards:
                future = executor.submit(_board_task, str(cache.path), node, board, options)
                futures[future] = (node, board)
        pending: set[Future[list[ChannelResult]]] = set(futures)
        while pending:
            if should_stop():
                raise AnalysisCancelled("Analysis cancelled")
            done, pending = wait(pending, timeout=_POLL_SECONDS, return_when=FIRST_COMPLETED)
            for future in done:
                node, board = futures[future]
                try:
                    board_results = future.result()
                except BrokenProcessPool as exc:
                    # Every pending board fails at once: the culprit is unknown
                    if should_stop():
                        raise AnalysisCancelled("Analysis cancelled") from exc
                    raise WorkerCrashedError() from exc
                except Exception as exc:
                    if should_stop():
                        raise AnalysisCancelled("Analysis cancelled") from exc
                    raise _board_error(node, board, exc) from exc
                results.extend(board_results)
                progress.board_done(node, board)
        completed = True
    except BrokenProcessPool as exc:  # from submit(): a worker died while starting
        if should_stop():
            raise AnalysisCancelled("Analysis cancelled") from exc
        raise WorkerCrashedError() from exc
    finally:
        if not completed:
            # Running boards stop at their next channel; pending ones never start.
            stop_event.set()
            for future in futures:
                future.cancel()
        executor.shutdown(wait=True, cancel_futures=not completed)
    return results
