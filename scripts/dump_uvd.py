#!/usr/bin/env python3
"""Write RadialAnalysis ``.uvd`` files from a uvcorr UV cache (plan section 10.3).

The C++ cross-check (``~/DataProcessing/EllipseCorrection/RadialAnalysis``)
reads one ``.uvd`` text file per channel. This script writes them straight from
the HDF5 UV cache for a few boards, so RadialAnalysis and uvcorr fit exactly
the same events::

    venv/bin/python scripts/dump_uvd.py --out /tmp/uvd                  # default boards
    venv/bin/python scripts/dump_uvd.py data.dat --boards 1:15 --out d  # <data.dat>.uv.h5
    venv/bin/python scripts/dump_uvd.py --cache x.uv.h5 --boards 8:24 --channels 0:4-28 --out d

File name: ``node{N}board{BB:02d}rena0{R}channel{CC:02d}.uvd``, as built by
``buildUVDFilename`` in ``uvd_common.h`` (and by extractData's
``ChannelWriter``, ``%02d`` for the RENA), e.g.
``node1board15rena00channel04.uvd``.

Content: the header line ``U Values V Values``, then one ``"%d %d\\n"`` line
(one space) per event in file order, so the file ends with a newline.
RadialAnalysis' ``loadUVD`` always discards the first line and then reads
whitespace-separated number pairs until EOF, dropping the value read at EOF:

- The header is what the original Python DAQ exporter wrote. The current
  extractData ``exportUVD`` (``~/DataProcessing/extractData/data_kimia_edit/
  main.cpp``) writes *no* header, so RadialAnalysis loses the first event of
  every channel it reads from extractData output. With the header, it reads
  every event.
- The trailing newline matters: without it the last event would be read at
  EOF and dropped.

The output directory must be empty (or not exist): RadialAnalysis reads every
``.uvd`` file in it, so stale files from another run would be analysed too.
``--overwrite`` deletes the ``.uvd`` files already there first (other files,
e.g. an old ``RadialAnalysis_output/``, are left alone; RadialAnalysis
rewrites its CSV and ``.tec``).

Only active channels are in the cache (plan D2: RENA 0 channels 4-28, RENA 1
channels 7-28). The cache is only read; it must be valid for the ``.dat``
when one is given (build it with ``uvcorr build-cache``).

Exit codes: 0 on success, 1 if nothing could be written (a board without
cached events, no channel matching ``--channels``, an I/O error), 2 for bad
arguments, a missing or invalid cache, or a non-empty output directory
without ``--overwrite``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from uvcorr.cache import UVCache, UVCacheError, default_cache_path

REFERENCE_DAT = Path.home() / "adc2kev-test-data/full-system/sources/ge/data_20260910_120628.dat"
"""The test acquisition (plan section 1); its cache is the default input."""

DEFAULT_BOARDS: tuple[tuple[int, int], ...] = ((1, 15), (1, 16), (8, 24))
"""Default boards: node 1 boards 15 and 16 (plan 10.3) and node 8 board 24, a
low-count board of the test file whose 47 channels have 1 to 19,516 events
(20 of them below 100 events)."""

UVD_HEADER = "U Values V Values"
"""First line of a ``.uvd`` file (skipped by ``loadUVD``)."""

UVD_SUFFIX = ".uvd"


def uvd_filename(node: int, board: int, rena: int, channel: int) -> str:
    """``.uvd`` file name of a channel, exactly as ``buildUVDFilename`` (uvd_common.h) builds it.

    That is ``"node" + N``, ``"board"`` + B zero-padded below 10, ``"rena0" + R`` and
    ``"channel"`` + C zero-padded below 10, e.g. ``node1board15rena00channel04.uvd``.
    """
    return f"node{node}board{board:02d}rena0{rena}channel{channel:02d}{UVD_SUFFIX}"


def parse_uvd_filename(name: str) -> tuple[int, int, int, int]:
    """Parse ``node{N}board{BB}rena{RR}channel{CC}.uvd`` into ``(node, board, rena, channel)``.

    Raises:
        ValueError: If the name does not have that form.
    """
    stem = name[: -len(UVD_SUFFIX)] if name.endswith(UVD_SUFFIX) else ""
    try:
        if not stem.startswith("node"):
            raise ValueError
        node_text, rest = stem[4:].split("board", 1)
        board_text, rest = rest.split("rena", 1)
        rena_text, channel_text = rest.split("channel", 1)
        return int(node_text), int(board_text), int(rena_text), int(channel_text)
    except ValueError:
        raise ValueError(f"not a .uvd channel file name: {name!r}") from None


def format_uvd(u: npt.ArrayLike, v: npt.ArrayLike) -> str:
    """Text of a ``.uvd`` file: the header, then ``"u v"`` per event, newline-terminated.

    Args:
        u: Integer U values (e.g. the int16 cache column).
        v: Integer V values, same length.

    Returns:
        The file content.

    Raises:
        ValueError: If ``u`` and ``v`` differ in length or are not integers.
    """
    uu = np.asarray(u)
    vv = np.asarray(v)
    if uu.shape != vv.shape or uu.ndim != 1:
        raise ValueError(f"u and v must be 1-D arrays of equal length, got {uu.shape}, {vv.shape}")
    if uu.size and not (
        np.issubdtype(uu.dtype, np.integer) and np.issubdtype(vv.dtype, np.integer)
    ):
        raise ValueError("u and v must be integer arrays (the .uvd files hold ADC counts)")
    lines = [UVD_HEADER]
    lines.extend(f"{a} {b}" for a, b in zip(uu.tolist(), vv.tolist()))
    return "\n".join(lines) + "\n"


def read_uvd(path: str | Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Read a ``.uvd`` file with the semantics of RadialAnalysis' ``loadUVD``.

    ``loadUVD`` discards the first line, whatever it holds, then loops
    ``while (!eof) { file >> u >> v; push_back(u, v); }`` and finally pops the
    last pair. That pair is the garbage from the extraction that hit EOF when
    the file ends with whitespace (e.g. a newline), but a *real* event when the
    last number ends the file. The result is therefore: every complete
    whitespace-separated ``u v`` pair after line 1, minus a trailing unpaired
    value, and minus the last pair if the file does not end with whitespace.

    Returns:
        ``(U, V)`` as float64 arrays.
    """
    text = Path(path).read_text(encoding="ascii")
    _, _, body = text.partition("\n")
    values = np.array(body.split(), dtype=np.float64)
    n = values.size // 2
    if n and values.size % 2 == 0 and not body[-1].isspace():
        n -= 1  # the last pair was read at EOF, then popped
    return values[0 : 2 * n : 2].copy(), values[1 : 2 * n : 2].copy()


def parse_boards(text: str) -> list[tuple[int, int]]:
    """Parse ``"1:15,1:16,8:24"`` into ``[(1, 15), (1, 16), (8, 24)]`` (order kept, no repeats).

    Raises:
        ValueError: If an entry is not ``node:board``.
    """
    boards: list[tuple[int, int]] = []
    for item in _items(text):
        node_text, sep, board_text = item.partition(":")
        try:
            if not sep:
                raise ValueError
            key = (int(node_text), int(board_text))
        except ValueError:
            raise ValueError(f"expected node:board, got {item!r}") from None
        if key not in boards:
            boards.append(key)
    if not boards:
        raise ValueError("no boards given")
    return boards


def parse_channels(text: str) -> set[tuple[int, int]]:
    """Parse ``"0:4-28,1:7,1:9"`` into a set of ``(rena, channel)`` pairs.

    A channel may be a range ``lo-hi`` (inclusive).

    Raises:
        ValueError: If an entry is not ``rena:channel`` or ``rena:lo-hi``.
    """
    channels: set[tuple[int, int]] = set()
    for item in _items(text):
        rena_text, sep, channel_text = item.partition(":")
        try:
            if not sep:
                raise ValueError
            rena = int(rena_text)
            lo_text, dash, hi_text = channel_text.partition("-")
            lo = int(lo_text)
            hi = int(hi_text) if dash else lo
            if hi < lo:
                raise ValueError
        except ValueError:
            raise ValueError(f"expected rena:channel or rena:lo-hi, got {item!r}") from None
        channels.update((rena, ch) for ch in range(lo, hi + 1))
    if not channels:
        raise ValueError("no channels given")
    return channels


def _items(text: str) -> Iterable[str]:
    return (item.strip() for item in text.split(",") if item.strip())


class OutputDirNotEmptyError(Exception):
    """The output directory already holds files and ``overwrite`` was not given."""


def dump_uvd(
    cache: UVCache,
    boards: Sequence[tuple[int, int]],
    out_dir: str | Path,
    channels: set[tuple[int, int]] | None = None,
    *,
    overwrite: bool = False,
) -> list[tuple[Path, int]]:
    """Write one ``.uvd`` file per channel with events on the given boards.

    Everything is checked before the output directory is touched: every board
    must have cached events and at least one channel must be selected.

    Args:
        cache: The UV cache (only read).
        boards: ``(node, board)`` pairs.
        out_dir: Output directory (created if needed). It must be empty unless
            ``overwrite`` is set.
        channels: Optional ``(rena, channel)`` filter applied to every board.
        overwrite: Delete the ``.uvd`` files already in ``out_dir`` first and
            accept a non-empty directory.

    Returns:
        ``(path, n_events)`` of every file written, in board then channel order.

    Raises:
        KeyError: If the cache has no events for one of the boards.
        ValueError: If no channel with events matches ``channels``.
        OutputDirNotEmptyError: If ``out_dir`` is not empty and ``overwrite``
            is False.
        NotADirectoryError: If ``out_dir`` exists and is not a directory.
    """
    available = cache.board_event_counts()
    for node, board in boards:
        if (node, board) not in available:
            raise KeyError(f"No cached events for node {node} board {board}")
    selected = [
        (node, board, rena, channel)
        for node, board in boards
        for rena, channel, _ in cache.channels(node, board)
        if channels is None or (rena, channel) in channels
    ]
    if not selected:
        raise ValueError("no channel with events matches the selected boards and channels")

    out = Path(out_dir)
    if out.exists() and not out.is_dir():
        raise NotADirectoryError(f"{out} exists and is not a directory")
    if out.is_dir() and any(out.iterdir()):
        if not overwrite:
            raise OutputDirNotEmptyError(
                f"{out} is not empty; RadialAnalysis would also read any .uvd file already "
                "there (use --overwrite to delete the old .uvd files first)"
            )
        for old in out.glob(f"*{UVD_SUFFIX}"):
            old.unlink()
    out.mkdir(parents=True, exist_ok=True)

    wanted = set(selected)
    written: list[tuple[Path, int]] = []
    for node, board in boards:
        data = cache.load_board(node, board)
        for rena, channel, n_events in data.channels():
            if (node, board, rena, channel) not in wanted:
                continue
            u, v = data.channel_data(rena, channel)
            path = out / uvd_filename(node, board, rena, channel)
            path.write_text(format_uvd(u, v), encoding="ascii")
            written.append((path, n_events))
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write RadialAnalysis .uvd files (one per channel) from a uvcorr UV cache.",
    )
    parser.add_argument(
        "dat",
        nargs="?",
        type=Path,
        default=None,
        help="Raw .dat file; its cache <dat>.uv.h5 is used (must be valid). "
        f"Default: the test acquisition {REFERENCE_DAT.name}.",
    )
    parser.add_argument("--cache", type=Path, default=None, help="UV cache to read instead.")
    parser.add_argument(
        "--boards",
        type=parse_boards,
        default=list(DEFAULT_BOARDS),
        help="Comma-separated node:board list (default: "
        + ",".join(f"{n}:{b}" for n, b in DEFAULT_BOARDS)
        + ").",
    )
    parser.add_argument(
        "--channels",
        type=parse_channels,
        default=None,
        help="Only these channels, e.g. 0:4-28,1:7 (rena:channel or rena:lo-hi; default: all).",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output directory (must be empty or new)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the .uvd files already in --out first (other files are kept).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the exit code."""
    args = _build_parser().parse_args(argv)
    if args.cache is not None and args.dat is not None:
        print("error: give either a .dat file or --cache, not both", file=sys.stderr)
        return 2
    if args.cache is not None:
        cache = UVCache(args.cache)
        if not cache.exists():
            print(f"error: cache not found: {cache.path}", file=sys.stderr)
            return 2
    else:
        dat: Path = args.dat if args.dat is not None else REFERENCE_DAT
        cache = UVCache(default_cache_path(dat))
        try:
            valid = cache.is_valid_for(dat)
        except UVCacheError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if not valid:
            print(
                f"error: no valid UV cache for {dat} at {cache.path}; "
                f"run: uvcorr build-cache {dat}",
                file=sys.stderr,
            )
            return 2

    t0 = time.perf_counter()
    try:
        written = dump_uvd(cache, args.boards, args.out, args.channels, overwrite=args.overwrite)
    except OutputDirNotEmptyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 1
    except (ValueError, UVCacheError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    n_events = sum(n for _, n in written)
    boards_text = ", ".join(f"{n}:{b}" for n, b in args.boards)
    print(
        f"Wrote {len(written)} .uvd files ({n_events:,} events) for boards {boards_text} "
        f"from {cache.path} to {args.out} in {time.perf_counter() - t0:.1f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
