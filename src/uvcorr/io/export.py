"""Export of a run's results: ``<stem>.tec`` and ``radial_summary.csv`` together.

:func:`prepare_output_dir` checks the output location *before* an analysis
(so a long run does not end in a permission error), and :func:`write_outputs`
writes both files through temporary files that are renamed back to back.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path

from uvcorr.analysis import ChannelResult
from uvcorr.io._atomic import atomic_write_texts
from uvcorr.io.summary_csv import SUMMARY_CSV_NAME, format_summary_csv
from uvcorr.io.tec import format_tec

__all__ = ["output_paths", "prepare_output_dir", "write_outputs"]


def output_paths(output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    """The ``(<stem>.tec, radial_summary.csv)`` paths in ``output_dir``."""
    directory = Path(output_dir)
    return directory / f"{stem}.tec", directory / SUMMARY_CSV_NAME


def prepare_output_dir(output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    """Create the output directory and check that both outputs can be written there.

    Args:
        output_dir: Output directory (created with its parents if missing).
        stem: Name stem of the ``.tec`` file (the ``.dat`` stem).

    Returns:
        The ``(tec, csv)`` output paths.

    Raises:
        OSError: If the directory cannot be created or written, or an output
            path is a directory (the message says which).
    """
    directory = Path(output_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"cannot create the output directory {directory}: {exc}") from exc
    paths = output_paths(directory, stem)
    for path in paths:
        if path.is_dir():
            raise IsADirectoryError(f"the output file {path} is a directory; remove or rename it")
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".uvcorr-write-test-"):
            pass
    except OSError as exc:
        raise OSError(f"cannot write to the output directory {directory}: {exc}") from exc
    return paths


def write_outputs(
    output_dir: str | Path, stem: str, results: Iterable[ChannelResult]
) -> tuple[Path, Path]:
    """Write ``<stem>.tec`` and ``radial_summary.csv`` of the results.

    Both files are formatted first, written to ``fsync``-ed temporary files,
    and then renamed over the destinations back to back, so a failure leaves
    both previous files untouched.

    Args:
        output_dir: Existing output directory.
        stem: Name stem of the ``.tec`` file (the ``.dat`` stem).
        results: The results to export (e.g. batch results with overrides
            applied).

    Returns:
        The ``(tec, csv)`` paths written.

    Raises:
        ValueError: If an ok result cannot be written to the ``.tec`` file.
        OSError: If a file cannot be written.
    """
    rows = list(results)
    tec_path, csv_path = output_paths(output_dir, stem)
    atomic_write_texts(
        [
            (tec_path, format_tec(rows), "ascii"),
            (csv_path, format_summary_csv(rows), "utf-8"),
        ]
    )
    return tec_path, csv_path
