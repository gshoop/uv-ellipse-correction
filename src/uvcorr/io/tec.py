"""``<name>.tec`` writer and reader in the exact RadialAnalysis layout (plan 6.3, D10).

RadialAnalysis (``~/DataProcessing/EllipseCorrection/main_radial.cpp``, the
``tec <<`` block after the CSV row) writes, for every channel whose ellipse
fit succeeded, in channel-loop order::

    channel{\\n
    \\tnode=1\\n
    \\tboard=15\\n
    \\trena=0\\n
    \\tchannel=4\\n
    \\tcenterU=2030.41\\n
    \\tcenterV=2036.89\\n
    \\tsemiMajor=695.584\\n
    \\tsemiMinor=670.071\\n
    \\tphi=3.11564\\n
    \\tradius=682.708\\n
    \\tradiusStd=9.29557\\n
    }\\n

There is no header, no blank line between blocks, and the file ends with the
last block's ``}\\n`` (an empty file if no channel succeeded). Integers are
written in decimal. Doubles use the default ``std::ostream`` formatting
(``std::defaultfloat``, precision 6), which is exactly C's ``%g``: 6
significant digits, no trailing zeros or decimal point for integral values
(``2036``), exponent notation below 1e-4 or from 1e6 on (``-2.99377e-05``),
``-0`` for negative zero. Python's ``format(x, "g")`` (same as ``"%g" % x``)
gives identical bytes: verified against a compiled ``cout << x`` on 30,000
values, including rounding ties, exponent boundaries and subnormals.

``radius`` is sqrt(semiMajor * semiMinor) and ``radiusStd`` the Gaussian sigma
of the corrected radii (the sample standard deviation when that fit failed,
flagged ``gauss_fit_failed_post``). uvcorr writes phi in (-pi/2, pi/2];
RadialAnalysis could write any angle in [-pi/2, pi) (e.g. ``phi=3.11564``), which
is the same ellipse (plan D10).

Older ``.tec`` files from the legacy shear method (``EllipseCorrection``,
``main.cpp``) have ``centerU``, ``centerV``, ``psi`` and ``radius`` instead;
:func:`parse_tec_blocks` reads them, :func:`read_tec` rejects them with a clear
error.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from uvcorr.analysis import ChannelKey, ChannelResult
from uvcorr.ellipse import EllipseParams
from uvcorr.io._atomic import atomic_write_text

__all__ = [
    "TEC_KEYS",
    "TecEntry",
    "TecFormatError",
    "format_tec",
    "format_tec_double",
    "parse_tec_blocks",
    "read_tec",
    "write_tec",
]

TEC_KEYS: tuple[str, ...] = (
    "node",
    "board",
    "rena",
    "channel",
    "centerU",
    "centerV",
    "semiMajor",
    "semiMinor",
    "phi",
    "radius",
    "radiusStd",
)
"""The keys of a RadialAnalysis ``.tec`` block, in the order they are written."""

_INT_KEYS = frozenset({"node", "board", "rena", "channel"})
_LEGACY_KEYS = frozenset({"psi"})
_BLOCK_START = "channel{"
_BLOCK_END = "}"


class TecFormatError(ValueError):
    """A ``.tec`` file is malformed or not in the RadialAnalysis ellipse format."""


@dataclass(frozen=True)
class TecEntry:
    """One channel's block of a RadialAnalysis ``.tec`` file.

    Attributes:
        key: The channel address.
        params: The ellipse (``centerU``, ``centerV``, ``semiMajor``,
            ``semiMinor``, ``phi``), as written (not canonicalised).
        radius: The target radius sqrt(ab).
        radius_std: The corrected radial sigma (``radiusStd``).
    """

    key: ChannelKey
    params: EllipseParams
    radius: float
    radius_std: float


def format_tec_double(value: float) -> str:
    """Format a double like C++ ``std::ostream << value`` with the default settings.

    Args:
        value: A finite number.

    Returns:
        The ``%g`` text (6 significant digits).

    Raises:
        ValueError: If ``value`` is not finite (``.tec`` readers cannot parse it).
    """
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"cannot write the non-finite value {number} to a .tec file")
    return format(number, "g")


def _block(result: ChannelResult) -> str:
    params = result.params
    if params is None or result.post_sigma is None:
        raise ValueError(f"{result.key} has status ok but no ellipse or post sigma")
    radius = result.target_radius if result.target_radius is not None else params.target_radius
    values = (
        ("centerU", params.cx),
        ("centerV", params.cy),
        ("semiMajor", params.a),
        ("semiMinor", params.b),
        ("phi", params.phi),
        ("radius", radius),
        ("radiusStd", result.post_sigma),
    )
    lines = [
        _BLOCK_START,
        f"\tnode={result.node}",
        f"\tboard={result.board}",
        f"\trena={result.rena}",
        f"\tchannel={result.channel}",
    ]
    try:
        lines.extend(f"\t{name}={format_tec_double(value)}" for name, value in values)
    except ValueError as exc:
        raise ValueError(f"{result.key}: {exc}") from exc
    lines.append(_BLOCK_END)
    return "".join(f"{line}\n" for line in lines)


def format_tec(results: Iterable[ChannelResult]) -> str:
    """Return the ``.tec`` text of the ``status == ok`` results, sorted by channel key.

    Args:
        results: Channel results (other statuses are skipped).

    Returns:
        The file content.

    Raises:
        ValueError: If an ok result lacks its ellipse or post sigma, or a value
            is not finite.
    """
    ok = sorted((result for result in results if result.ok), key=lambda result: result.key)
    return "".join(_block(result) for result in ok)


def write_tec(path: str | Path, results: Iterable[ChannelResult]) -> Path:
    """Write the RadialAnalysis ``.tec`` file (atomically replaced).

    Only ``status == ok`` channels are written, in CSV order (sorted by
    node, board, rena, channel). See the module docstring for the layout.

    Args:
        path: Output file (its directory must exist).
        results: Channel results (e.g. the merged batch and override results).

    Returns:
        The path written.

    Raises:
        ValueError: See :func:`format_tec`.
        OSError: If the file cannot be written.
    """
    return atomic_write_text(path, format_tec(results), encoding="ascii")


def parse_tec_blocks(text: str) -> list[dict[str, str]]:
    """Split ``.tec`` text into its ``channel{...}`` blocks.

    Any ``key=value`` lines are accepted, so this also reads the legacy
    shear-method files. Blank lines and surrounding whitespace are ignored.

    Args:
        text: The file content.

    Returns:
        One ``{key: value text}`` dict per block, in file order.

    Raises:
        TecFormatError: If a line is outside a block, a block is not closed,
            a block repeats a key, or a line is not ``key=value``.
    """
    blocks: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if current is None:
            if line != _BLOCK_START:
                raise TecFormatError(f"line {lineno}: expected {_BLOCK_START!r}, got {raw!r}")
            current = {}
        elif line == _BLOCK_END:
            blocks.append(current)
            current = None
        else:
            key, sep, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not sep or not key:
                raise TecFormatError(f"line {lineno}: expected key=value, got {raw!r}")
            if key in current:
                raise TecFormatError(f"line {lineno}: key {key!r} repeated in a block")
            current[key] = value
    if current is not None:
        raise TecFormatError("the last channel block is not closed with '}'")
    return blocks


def _entry(block: dict[str, str], index: int) -> TecEntry:
    keys = set(block)
    if keys & _LEGACY_KEYS and "semiMajor" not in keys:
        raise TecFormatError(
            f"block {index} is in the legacy shear-method format (psi=...), "
            "not the RadialAnalysis ellipse format"
        )
    missing = [key for key in TEC_KEYS if key not in keys]
    unknown = sorted(keys - set(TEC_KEYS))
    if missing or unknown:
        raise TecFormatError(f"block {index}: missing keys {missing}, unknown keys {unknown}")
    try:
        ints = [int(block[key]) for key in TEC_KEYS if key in _INT_KEYS]
        floats = {key: float(block[key]) for key in TEC_KEYS if key not in _INT_KEYS}
    except ValueError as exc:
        raise TecFormatError(f"block {index}: {exc}") from exc
    return TecEntry(
        key=ChannelKey(*ints),
        params=EllipseParams(
            cx=floats["centerU"],
            cy=floats["centerV"],
            a=floats["semiMajor"],
            b=floats["semiMinor"],
            phi=floats["phi"],
        ),
        radius=floats["radius"],
        radius_std=floats["radiusStd"],
    )


def read_tec(path: str | Path) -> dict[ChannelKey, TecEntry]:
    """Read a RadialAnalysis-format ``.tec`` file.

    Args:
        path: The file.

    Returns:
        ``{channel key: entry}`` in file order.

    Raises:
        TecFormatError: If the file is malformed, is a legacy shear-method
            file, or has two blocks for one channel.
        OSError: If the file cannot be read.
    """
    text = Path(path).read_text(encoding="ascii")
    entries: dict[ChannelKey, TecEntry] = {}
    for index, block in enumerate(parse_tec_blocks(text)):
        entry = _entry(block, index)
        if entry.key in entries:
            raise TecFormatError(f"block {index}: a second block for {entry.key}")
        entries[entry.key] = entry
    return entries
