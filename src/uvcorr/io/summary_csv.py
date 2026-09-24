"""``radial_summary.csv`` writer and reader (new schema, plan 6.2, D9).

One row per channel result, sorted by (node, board, rena, channel); the
header is :data:`uvcorr.analysis.CSV_COLUMNS` (the :class:`ChannelResult`
fields, in order). Cells are formatted by column kind:

- integers in decimal; a count that is not available (``n_used``,
  ``n_rejected`` of a failed channel) is empty;
- floats with ``%.6g``, the ellipse centre, semi-axes and phi with ``%.9g``;
  an unavailable value is empty;
- ``flags`` joined with ``;`` (empty if none);
- text as is (``polarity`` is ``anode``/``cathode``, ``electrode`` the
  ``ElectrodeMap`` label, ``status``, ``options_source``).

Lines end with ``\\n``. The values never contain commas or quotes, so no
quoting is needed (the ``csv`` module would add it if they did).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from uvcorr.analysis import (
    CSV_COLUMNS,
    KIND_COUNT,
    KIND_FLAGS,
    KIND_FLOAT,
    KIND_FLOAT_PRECISE,
    KIND_INT,
    RESULT_COLUMNS,
    ChannelResult,
)
from uvcorr.io._atomic import atomic_write_text
from uvcorr.options import FLAG_SEPARATOR

__all__ = [
    "FLOAT_FORMAT",
    "PRECISE_FLOAT_FORMAT",
    "SUMMARY_CSV_NAME",
    "format_cell",
    "format_summary_csv",
    "parse_cell",
    "read_summary_csv",
    "write_summary_csv",
]

SUMMARY_CSV_NAME = "radial_summary.csv"
"""File name of the summary in the output directory."""

FLOAT_FORMAT = ".6g"
PRECISE_FLOAT_FORMAT = ".9g"
"""Formats of ``KIND_FLOAT`` and ``KIND_FLOAT_PRECISE`` cells (plan 6.2)."""


def format_cell(kind: str, value: Any) -> str:
    """Format one CSV cell of a column kind (``uvcorr.analysis.KIND_*``); None is empty."""
    if value is None:
        return ""
    if kind == KIND_FLOAT:
        return format(float(value), FLOAT_FORMAT)
    if kind == KIND_FLOAT_PRECISE:
        return format(float(value), PRECISE_FLOAT_FORMAT)
    if kind in (KIND_INT, KIND_COUNT):
        return str(int(value))
    if kind == KIND_FLAGS:
        return FLAG_SEPARATOR.join(value)
    return str(value)


def parse_cell(kind: str, text: str) -> Any:
    """Parse one CSV cell of a column kind (the inverse of :func:`format_cell`).

    Raises:
        ValueError: If a number cannot be parsed.
    """
    if kind == KIND_FLAGS:
        return tuple(part for part in text.split(FLAG_SEPARATOR) if part)
    if kind in (KIND_INT, KIND_COUNT, KIND_FLOAT, KIND_FLOAT_PRECISE) and text == "":
        return None
    if kind in (KIND_INT, KIND_COUNT):
        return int(text)
    if kind in (KIND_FLOAT, KIND_FLOAT_PRECISE):
        return float(text)
    return text


def format_summary_csv(results: Iterable[ChannelResult]) -> str:
    """Return the ``radial_summary.csv`` text of the results (sorted by channel key)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for result in sorted(results, key=lambda r: r.key):
        writer.writerow(
            format_cell(column.kind, getattr(result, column.name)) for column in RESULT_COLUMNS
        )
    return buffer.getvalue()


def write_summary_csv(path: str | Path, results: Iterable[ChannelResult]) -> Path:
    """Write ``radial_summary.csv`` (atomically replaced).

    Args:
        path: Output file (its directory must exist).
        results: Channel results (e.g. the merged batch and override results).

    Returns:
        The path written.

    Raises:
        OSError: If the file cannot be written.
    """
    return atomic_write_text(path, format_summary_csv(results), encoding="utf-8")


def read_summary_csv(path: str | Path) -> list[ChannelResult]:
    """Read a ``radial_summary.csv`` written by :func:`write_summary_csv`.

    Values come back at the written precision (6 or 9 significant digits).

    Args:
        path: The file.

    Returns:
        The results, in file order.

    Raises:
        ValueError: If the header is not :data:`~uvcorr.analysis.CSV_COLUMNS`
            or a row cannot be parsed.
        OSError: If the file cannot be read.
    """
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or tuple(header) != CSV_COLUMNS:
            raise ValueError(f"{path}: not a uvcorr radial_summary.csv (unexpected header)")
        results: list[ChannelResult] = []
        for lineno, row in enumerate(reader, start=2):
            if len(row) != len(CSV_COLUMNS):
                raise ValueError(
                    f"{path} line {lineno}: {len(row)} cells, expected {len(CSV_COLUMNS)}"
                )
            try:
                values = {
                    column.name: parse_cell(column.kind, cell)
                    for column, cell in zip(RESULT_COLUMNS, row)
                }
                results.append(ChannelResult.from_dict(values))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path} line {lineno}: {exc}") from exc
    return results
