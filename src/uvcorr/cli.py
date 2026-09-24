"""Command-line interface for uvcorr (backs the ``uvcorr`` console script).

Subcommands (plan section 8):

``uvcorr build-cache data.dat [--cache PATH] [--force]``
    Build (or reuse) the sibling HDF5 UV cache for a raw ``.dat`` file. Phase 1.

``uvcorr process data.dat --output-dir out/ [...]``
    Fit every active channel and write ``<stem>.tec`` and ``radial_summary.csv``.
    Phase 3.

Both subcommands are registered but not implemented yet; they exit with status 1.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

from uvcorr import __version__

logger = logging.getLogger(__name__)

EXIT_NOT_IMPLEMENTED = 1


def _positive_int(value: str) -> int:
    """argparse type: an int >= 1 (for ``--workers``, ``--max-iter``, ``--min-events``)."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {number}")
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
        help="Build the HDF5 UV cache for a raw .dat file (not implemented yet).",
        description="Build (or reuse) the sibling HDF5 UV cache for a raw .dat file.",
    )
    p.add_argument("dat", type=Path, help="Raw .dat acquisition file.")
    p.add_argument("--cache", type=Path, default=None, help="Cache path (default: <dat>.uv.h5).")
    p.add_argument("--force", action="store_true", help="Rebuild even if a valid cache exists.")
    p.set_defaults(func=_run_build_cache)


def _add_process_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``process`` subcommand."""
    p = subparsers.add_parser(
        "process",
        help="Fit all channels and export .tec + radial_summary.csv (not implemented yet).",
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
        help="Worker processes (default: min(8, CPU count)).",
    )
    p.add_argument(
        "--min-events",
        type=_positive_int,
        default=None,
        help="Minimum events for a channel to be fitted (default: 100).",
    )
    p.add_argument(
        "--no-robust", action="store_true", help="Disable the robust (MAD-clipped) refit."
    )
    p.add_argument(
        "--clip-k",
        type=_positive_float,
        default=None,
        help="Robust clipping threshold in robust sigmas (default: 4).",
    )
    p.add_argument(
        "--max-iter",
        type=_positive_int,
        default=None,
        help="Maximum robust refit iterations (default: 5).",
    )
    p.add_argument(
        "--geometric", action="store_true", help="Refine with a geometric least-squares fit."
    )
    p.set_defaults(func=_run_process)


def _not_implemented(command: str, phase: int) -> int:
    print(
        f"uvcorr {command}: not implemented yet (planned for phase {phase}).",
        file=sys.stderr,
    )
    return EXIT_NOT_IMPLEMENTED


def _run_build_cache(args: argparse.Namespace) -> int:
    """Run ``uvcorr build-cache`` (phase 1)."""
    return _not_implemented(args.command, phase=1)


def _run_process(args: argparse.Namespace) -> int:
    """Run ``uvcorr process`` (phase 3)."""
    return _not_implemented(args.command, phase=3)


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
        return 130


if __name__ == "__main__":
    sys.exit(main())
