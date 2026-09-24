"""Command-line interface for uvcorr (backs the ``uvcorr`` console script).

Subcommands (plan section 8):

``uvcorr build-cache data.dat [--cache PATH] [--force]``
    Build (or reuse) the sibling HDF5 UV cache for a raw ``.dat`` file.

``uvcorr process data.dat --output-dir out/ [--cache PATH] [--workers N]
[--min-events N] [--no-robust] [--clip-k K] [--max-iter N] [--geometric]
[--discard-overrides]``
    Build or reuse the cache, fit every active channel (a process pool, one
    task per board), write ``<dat stem>.tec`` and ``radial_summary.csv`` from
    the batch results with the cache's per-channel GUI overrides applied
    (unless ``--discard-overrides``), then store the results in the cache
    (``/results/current``, keeping the overrides unless
    ``--discard-overrides``; skipped with a warning if the cache file is
    read-only). The output directory is created and write-tested before the
    analysis starts. Prints a status/flag census and timings. Options that
    are not given keep their ``FitOptions`` defaults.

Exit codes: 0 on success, 1 on an error (e.g. a failed build or analysis, an
unusable cache path, a cache in use by another process, or too little disk
space), 2 for a missing input file or bad arguments, 130 when stopped with
Ctrl-C. Ctrl-C during a build or an analysis sets its stop flag: a build stops
at its next check (within one parser batch), removes its temporary file and
leaves any previous cache untouched; an analysis stops its workers at their
next channel, and nothing is stored or written. Further Ctrl-Cs are ignored
until that cleanup has finished.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, TextIO

from uvcorr import __version__
from uvcorr.analysis import (
    MAX_DEFAULT_WORKERS,
    AnalysisCancelled,
    AnalysisError,
    ChannelResult,
    analyze_all,
    default_workers,
)
from uvcorr.cache import (
    USER_WARNING,
    CacheBuildCancelled,
    ResultsError,
    UVCache,
    UVCacheError,
    default_cache_path,
    merge_results,
)
from uvcorr.ellipse import MIN_FIT_POINTS
from uvcorr.io.export import prepare_output_dir, write_outputs
from uvcorr.options import FLAGS, STATUSES, FitOptions

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def _positive_int(value: str) -> int:
    """argparse type: an int >= 1 (for ``--workers``, ``--max-iter``, ``--min-events``)."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {number}")
    return number


def _min_events(value: str) -> int:
    """argparse type for ``--min-events``: an int >= ``MIN_FIT_POINTS`` (6).

    ``FitOptions`` accepts any ``min_events >= 1``, but below 6 points the
    algebraic fit cannot run, so channels with 1-5 events would be reported
    ``fit_failed`` instead of ``too_few_events``; the CLI refuses such values.
    """
    number = _positive_int(value)
    if number < MIN_FIT_POINTS:
        raise argparse.ArgumentTypeError(
            f"must be >= {MIN_FIT_POINTS} (the ellipse fit needs {MIN_FIT_POINTS} points), "
            f"got {number}"
        )
    return number


def _positive_float(value: str) -> float:
    """argparse type: a finite float > 0 (for ``--clip-k``)."""
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(f"must be a finite number > 0, got {value}")
    return number


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="uvcorr",
        description="RENA-3 fine-timing U/V ellipse correction from raw .dat files.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v for INFO, -vv for DEBUG).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_build_cache_parser(subparsers)
    _add_process_parser(subparsers)
    return parser


def _add_build_cache_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the ``build-cache`` subcommand."""
    p = subparsers.add_parser(
        "build-cache",
        help="Build (or reuse) the HDF5 UV cache for a raw .dat file.",
        description="Build (or reuse) the sibling HDF5 UV cache for a raw .dat file.",
    )
    p.add_argument("dat", type=Path, help="Raw .dat acquisition file.")
    p.add_argument("--cache", type=Path, default=None, help="Cache path (default: <dat>.uv.h5).")
    p.add_argument("--force", action="store_true", help="Rebuild even if a valid cache exists.")
    p.set_defaults(func=_run_build_cache)


def _add_process_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``process`` subcommand (defaults in the help come from ``FitOptions``)."""
    defaults = FitOptions()
    p = subparsers.add_parser(
        "process",
        help="Fit all channels and export .tec + radial_summary.csv.",
        description=(
            "Build or reuse the UV cache, fit every active channel, and write "
            "<stem>.tec and radial_summary.csv to the output directory."
        ),
    )
    p.add_argument("dat", type=Path, help="Raw .dat acquisition file.")
    p.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for the .tec and CSV files."
    )
    p.add_argument("--cache", type=Path, default=None, help="Cache path (default: <dat>.uv.h5).")
    p.add_argument(
        "--workers",
        type=_positive_int,
        default=None,
        help=f"Worker processes (default: min({MAX_DEFAULT_WORKERS}, usable CPUs), "
        f"here {default_workers()}).",
    )
    p.add_argument(
        "--min-events",
        type=_min_events,
        default=None,
        help=f"Minimum events for a channel to be fitted, >= {MIN_FIT_POINTS} "
        f"(default: {defaults.min_events}).",
    )
    p.add_argument(
        "--no-robust", action="store_true", help="Disable the robust (MAD-clipped) refit."
    )
    p.add_argument(
        "--clip-k",
        type=_positive_float,
        default=None,
        help=f"Robust clipping threshold in robust sigmas (default: {defaults.clip_k:g}).",
    )
    p.add_argument(
        "--max-iter",
        type=_positive_int,
        default=None,
        help=f"Maximum robust refit iterations (default: {defaults.max_iter}).",
    )
    p.add_argument(
        "--geometric", action="store_true", help="Refine with a geometric least-squares fit."
    )
    p.add_argument(
        "--discard-overrides",
        action="store_true",
        help="Drop the per-channel overrides stored in the cache (default: keep and apply them).",
    )
    p.set_defaults(func=_run_process)


class _ProgressPrinter:
    """Throttled progress callback that writes to a stream (stderr).

    On a terminal the line is redrawn in place at every whole percent; on a
    pipe or file a new line is written every 10 %.
    """

    def __init__(self, label: str, stream: TextIO | None = None) -> None:
        self._label = label
        self._stream = stream if stream is not None else sys.stderr
        self._tty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._step = 1 if self._tty else 10
        self._last = -1
        self._t0 = time.perf_counter()

    def __call__(self, fraction: float) -> None:
        pct = max(0, min(100, int(fraction * 100)))
        bucket = pct // self._step
        if bucket == self._last:
            return
        self._last = bucket
        elapsed = time.perf_counter() - self._t0
        text = f"{self._label}: {pct:3d}% ({elapsed:5.1f} s)"
        if self._tty:
            print(f"\r{text}", end="", file=self._stream, flush=True)
        else:
            print(text, file=self._stream, flush=True)

    def close(self) -> None:
        """End the in-place progress line (terminal only)."""
        if self._tty and self._last >= 0:
            print(file=self._stream, flush=True)


@contextmanager
def _sigint_sets(stop: threading.Event, what: str = "the build") -> Iterator[None]:
    """While active, Ctrl-C sets ``stop`` instead of raising KeyboardInterrupt.

    The previous handler is restored only when the block exits, i.e. after
    the build (or analysis) has stopped and cleaned up, so no
    KeyboardInterrupt can land in the middle of that cleanup. Only installed
    in the main thread (``signal.signal`` fails elsewhere). ``what`` names the
    operation in the messages.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(_signum: int, _frame: FrameType | None) -> None:
        if stop.is_set():
            print(f"\nstill stopping {what}...", file=sys.stderr, flush=True)
            return
        stop.set()
        print(f"\nstopping {what}...", file=sys.stderr, flush=True)

    previous = signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


class _DropUserWarnings(logging.Filter):
    """Drop the cache's end-user warnings, which the CLI prints itself."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, USER_WARNING, False)


@contextmanager
def _cli_reports_user_warnings() -> Iterator[None]:
    cache_logger = logging.getLogger("uvcorr.cache")
    drop = _DropUserWarnings()
    cache_logger.addFilter(drop)
    try:
        yield
    finally:
        cache_logger.removeFilter(drop)


def _format_bytes(n_bytes: int) -> str:
    return f"{n_bytes / 1e9:.2f} GB" if n_bytes >= 1e8 else f"{n_bytes / 1e6:.2f} MB"


def _print_cache_summary(
    cache: UVCache, meta: dict[str, Any], reused: bool, elapsed: float
) -> None:
    """Print the build-cache summary to stdout."""
    counts = cache.board_event_counts()
    nodes = sorted({node for node, _ in counts})
    boards = sorted({board for _, board in counts})
    kept = int(meta["n_events_kept"])
    inactive = int(meta["n_events_inactive"])
    node0 = int(meta.get("n_events_node0", 0))
    uv_zero = meta.get("n_events_uv_zero")  # absent in caches built before it existed
    if reused:
        status = f"reused (valid for the .dat; built {meta.get('created_at', '?')})"
    else:
        status = f"built in {float(meta['build_seconds']):.1f} s"
    board_range = f"nodes {_ranges(nodes)}, boards {_ranges(boards)}" if counts else "none"
    print(f"UV cache: {cache.path}")
    print(f"  status:        {status}")
    print(f"  events kept:   {kept:,} on {len(counts)} boards ({board_range})")
    if uv_zero:
        print(f"                 {int(uv_zero):,} of them with U = V = 0")
    print(f"  dropped:       {inactive:,} on inactive channels, {node0:,} from node 0")
    print(
        f"  parser:        {int(meta['parser_frames']):,} frames, "
        f"{int(meta['parser_events']):,} events, {int(meta['parser_dropped']):,} dropped frames"
    )
    print(f"  cache size:    {_format_bytes(cache.path.stat().st_size)}")
    print(f"  time:          {elapsed:.1f} s")


def _ranges(values: list[int]) -> str:
    """Compact ``1-10`` / ``1,3,5-7`` rendering of sorted integers."""
    parts: list[str] = []
    start = prev = values[0]
    for value in [*values[1:], None]:
        if value is not None and value == prev + 1:
            prev = value
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if value is not None:
            start = prev = value
    return ",".join(parts)


def _run_build_cache(args: argparse.Namespace) -> int:
    """Run ``uvcorr build-cache``: build or reuse the UV cache, print a summary."""
    dat: Path = args.dat
    if not dat.is_file():
        print(f"error: raw data file not found: {dat}", file=sys.stderr)
        return EXIT_USAGE
    cache = UVCache(args.cache if args.cache is not None else default_cache_path(dat))

    t0 = time.perf_counter()
    try:
        with _cli_reports_user_warnings():
            reused = not args.force and cache.is_valid_for(dat)
            if not reused:
                _build_with_progress(cache, dat, force=args.force)
            meta = cache.metadata()
            _print_cache_summary(cache, meta, reused, time.perf_counter() - t0)
    except CacheBuildCancelled:
        print("build cancelled; no new cache was written.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (UVCacheError, OSError) as exc:  # build errors, busy cache, disk space
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if int(meta.get("parser_frames", 0)) == 0 and int(meta.get("source_size", 0)) > 0:
        print(
            f"warning: no valid frames in {dat}: is this a raw .dat file?",
            file=sys.stderr,
        )
    return EXIT_OK


def _build_with_progress(cache: UVCache, dat: Path, *, force: bool) -> None:
    """Build ``cache`` from ``dat`` with stderr progress and Ctrl-C as stop flag."""
    if force:
        reason = "--force"
    elif cache.exists():
        reason = "existing file is not a valid cache for this .dat"
    else:
        reason = "no cache yet"
    print(f"Building UV cache {cache.path} ({reason})", file=sys.stderr)
    if cache.has_results():
        print(
            f"warning: {cache.path} stores analysis results/overrides (/results); "
            "rebuilding the cache discards them",
            file=sys.stderr,
        )
    stop = threading.Event()
    progress = _ProgressPrinter("parsing", sys.stderr)
    try:
        with _sigint_sets(stop):
            cache.build_from_dat(dat, progress_cb=progress, stop_flag=stop)
    finally:
        progress.close()


def _fit_options(args: argparse.Namespace) -> FitOptions:
    """``FitOptions`` from the defaults plus the options given on the command line."""
    given: dict[str, Any] = {}
    if args.min_events is not None:
        given["min_events"] = args.min_events
    if args.no_robust:
        given["robust"] = False
    if args.clip_k is not None:
        given["clip_k"] = args.clip_k
    if args.max_iter is not None:
        given["max_iter"] = args.max_iter
    if args.geometric:
        given["geometric"] = True
    return FitOptions(**given)


def _describe_options(options: FitOptions) -> str:
    """``defaults`` or the non-default options as ``name=value``."""
    defaults = FitOptions().to_dict()
    changed = [f"{k}={v}" for k, v in options.to_dict().items() if defaults[k] != v]
    return ", ".join(changed) if changed else "defaults"


def _run_process(args: argparse.Namespace) -> int:
    """Run ``uvcorr process``: cache, fit all channels, export, store, census.

    Everything that can be checked cheaply is checked before the analysis: the
    output directory is created and write-tested, the stored overrides are
    read, and a read-only cache is detected (the run then still writes the
    outputs but does not store the results). The outputs are written before
    the results are stored, so a failure to store never loses the outputs.
    """
    dat: Path = args.dat
    if not dat.is_file():
        print(f"error: raw data file not found: {dat}", file=sys.stderr)
        return EXIT_USAGE
    output_dir: Path = args.output_dir
    if output_dir.exists() and not output_dir.is_dir():
        print(f"error: --output-dir {output_dir} exists and is not a directory", file=sys.stderr)
        return EXIT_USAGE
    try:
        options = _fit_options(args)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    cache = UVCache(args.cache if args.cache is not None else default_cache_path(dat))

    t_start = time.perf_counter()
    try:
        prepare_output_dir(output_dir, dat.stem)
        with _cli_reports_user_warnings():
            reused = cache.is_valid_for(dat)
            if not reused:
                _build_with_progress(cache, dat, force=False)
        t_cache = time.perf_counter()

        store = os.access(cache.path, os.W_OK)
        if not store:
            print(
                f"warning: the UV cache {cache.path} is read-only: the results will not be "
                "stored in it (the output files are still written)",
                file=sys.stderr,
            )
        overrides: list[ChannelResult] = []
        if not args.discard_overrides:
            stored = cache.load_results()
            if stored is not None:
                overrides = [override.result for override in stored.overrides.values()]

        n_boards = len(cache.board_event_counts())
        workers = min(args.workers or default_workers(), max(n_boards, 1))
        print(
            f"Fitting {n_boards} boards with {workers} worker(s), options: "
            f"{_describe_options(options)}",
            file=sys.stderr,
        )
        results = _analyze_with_progress(cache, options, workers)
        t_analysis = time.perf_counter()

        merged = merge_results(results, overrides)
        tec_path, csv_path = write_outputs(output_dir, dat.stem, merged)
        t_write = time.perf_counter()
    except CacheBuildCancelled:
        print("build cancelled; no new cache was written.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except AnalysisCancelled:
        print("analysis cancelled; nothing was stored or written.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except ResultsError as exc:
        print(f"error: {exc} (--discard-overrides skips and replaces them)", file=sys.stderr)
        return EXIT_ERROR
    except (AnalysisError, UVCacheError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    store_error: Exception | None = None
    if store:
        try:
            cache.save_results(results, options, keep_overrides=not args.discard_overrides)
        except (UVCacheError, OSError, ValueError) as exc:
            store_error = exc
    t_store = time.perf_counter()

    status = "reused" if reused else "built"
    print(f"UV cache: {cache.path} ({status})")
    _print_census(merged, n_boards, workers, options, len(overrides), args.discard_overrides)
    n_ok = sum(1 for r in merged if r.ok)
    print("Outputs:")
    print(f"  {tec_path}  ({n_ok:,} channel blocks, {_format_bytes(tec_path.stat().st_size)})")
    print(f"  {csv_path}  ({len(merged):,} rows, {_format_bytes(csv_path.stat().st_size)})")
    if store and store_error is None:
        print("  results stored in the cache (/results/current)")
    else:
        print("  results NOT stored in the cache")
    print(
        f"Time: cache {t_cache - t_start:.1f} s, analysis {t_analysis - t_cache:.1f} s, "
        f"write {t_write - t_analysis:.1f} s, store {t_store - t_write:.1f} s, "
        f"total {t_store - t_start:.1f} s"
    )
    if store_error is not None:
        print(
            "error: the output files were written, but storing the results in the cache "
            f"failed: {store_error}",
            file=sys.stderr,
        )
        return EXIT_ERROR
    return EXIT_OK


def _analyze_with_progress(
    cache: UVCache, options: FitOptions, workers: int
) -> list[ChannelResult]:
    """Run ``analyze_all`` with stderr progress and Ctrl-C as stop flag."""
    stop = threading.Event()
    progress = _ProgressPrinter("fitting", sys.stderr)

    def on_progress(done: int, total: int) -> None:
        progress(done / total if total else 1.0)

    try:
        with _sigint_sets(stop, "the analysis"):
            return analyze_all(
                cache, options, workers=workers, progress_cb=on_progress, stop_flag=stop
            )
    finally:
        progress.close()


def _print_census(
    results: list[ChannelResult],
    n_boards: int,
    workers: int,
    options: FitOptions,
    n_overrides: int,
    discarded: bool,
) -> None:
    """Print the status/flag census of the exported (merged) results to stdout."""
    n_events = sum(r.n_events for r in results)
    print(
        f"Analysis: {len(results):,} channels ({n_events:,} events) on {n_boards} boards, "
        f"{workers} worker(s), options: {_describe_options(options)}"
    )
    statuses = {status: sum(1 for r in results if r.status == status) for status in STATUSES}
    print("  status:     " + ", ".join(f"{s} {n:,}" for s, n in statuses.items()))
    flag_counts = {flag: sum(1 for r in results if flag in r.flags) for flag in FLAGS}
    flagged = sum(1 for r in results if r.flags)
    flag_text = ", ".join(f"{f} {n:,}" for f, n in flag_counts.items() if n) or "none"
    print(f"  flags:      {flag_text}")
    print(f"              ({flagged:,} channels with at least one flag)")
    if discarded:
        print("  overrides:  discarded (--discard-overrides)")
    else:
        print(f"  overrides:  {n_overrides:,} applied")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 on success)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    try:
        exit_code: int = args.func(args)
        return exit_code
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
