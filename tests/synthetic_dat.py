"""Synthetic RENA-3 AND-mode (0xC8) ``.dat`` writer for tests.

Frame layout (``~/adc2kev-python/docs/technical/PACKET_FORMAT_SPEC.md``),
``19 + 6 * n`` bytes for ``n`` triggered channels:

====== ==== ==================================================================
Offset Size Field
====== ==== ==================================================================
0      1    Header 0xC8 (AND mode)
1      1    Source node
2      1    Destination (0x00)
3      1    bits[6:1] board (FPGA id), bit 0 RENA
4-9    6    42-bit timestamp, 7 bits per byte, big-endian
10-15  6    36-bit fast trigger list, 6 bits per byte, big-endian (bit i = channel i)
16+    6n   Per triggered channel, in ascending channel order: PHA, U, V as
            12-bit values, each 2 bytes of 6 bits (high, low)
N-3    2    CRC-8 as two nibbles (high, low)
N-1    1    Terminator 0xFF
====== ==== ==================================================================

CRC-8 (polynomial 0x07): ``crc = T[header]``, then ``crc = T[crc ^ b]`` for
every byte from offset 3 up to the last data byte (the node and destination
bytes are not covered; this is the 3-byte lag of the reference C++ decoder).

Adapted from adc2kev's ``tests/test_parser/synthetic_packets.py`` (test code,
not package API); the CRC is reimplemented here so the tests do not depend on
adc2kev internals. The tests check that files written here parse with
``PacketParser.iter_event_arrays`` to exactly :func:`expected_events`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

HEADER_AND = 0xC8
TERMINATOR = 0xFF


def _crc8_table() -> list[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) if crc & 0x80 else (crc << 1)
            crc &= 0xFF
        table.append(crc)
    return table


CRC8_TABLE = _crc8_table()

EVENT_DTYPE = np.dtype(
    [
        ("node", np.uint8),
        ("board", np.uint8),
        ("rena", np.uint8),
        ("channel", np.uint8),
        ("pha", np.int16),
        ("u", np.int16),
        ("v", np.int16),
    ]
)
"""Row layout of :func:`expected_events`."""


@dataclass(frozen=True)
class Hit:
    """One triggered channel of a frame (12-bit PHA, U, V)."""

    channel: int
    pha: int
    u: int
    v: int


@dataclass(frozen=True)
class Frame:
    """One AND-mode frame: a RENA readout with one or more triggered channels."""

    node: int
    board: int
    rena: int
    timestamp: int
    hits: tuple[Hit, ...]

    def __post_init__(self) -> None:
        channels = [hit.channel for hit in self.hits]
        if len(set(channels)) != len(channels):
            raise ValueError(f"duplicate channel in frame: {channels}")
        if not 0 <= self.node <= 0xC7:  # a node byte of 0xC8/0xC9/0xFF would confuse framing
            raise ValueError(f"node {self.node} not encodable in tests")
        if not 0 <= self.board <= 63 or self.rena not in (0, 1):
            raise ValueError(f"bad board/rena {self.board}/{self.rena}")
        for hit in self.hits:
            if not 0 <= hit.channel <= 35:
                raise ValueError(f"bad channel {hit.channel}")
            for value in (hit.pha, hit.u, hit.v):
                if not 0 <= value <= 0xFFF:
                    raise ValueError(f"value {value} is not 12-bit")


def _encode(value: int, n_bytes: int, bits: int) -> bytes:
    mask = (1 << bits) - 1
    return bytes((value >> (i * bits)) & mask for i in range(n_bytes - 1, -1, -1))


def crc8(body: bytes) -> int:
    """CRC-8 of a frame body (header through the last data byte)."""
    crc = CRC8_TABLE[body[0]]
    for byte in body[3:]:
        crc = CRC8_TABLE[crc ^ byte]
    return crc


def encode_frame(frame: Frame) -> bytes:
    """Encode one AND-mode frame with a valid CRC and terminator."""
    hits = sorted(frame.hits, key=lambda hit: hit.channel)
    triggers = 0
    for hit in hits:
        triggers |= 1 << hit.channel
    body = bytearray([HEADER_AND, frame.node, 0x00, ((frame.board & 0x3F) << 1) | frame.rena])
    body += _encode(frame.timestamp & 0x3FFFFFFFFFF, 6, 7)
    body += _encode(triggers, 6, 6)
    for hit in hits:
        for value in (hit.pha, hit.u, hit.v):
            body += bytes([(value >> 6) & 0x3F, value & 0x3F])
    crc = crc8(bytes(body))
    return bytes(body) + bytes([(crc >> 4) & 0x0F, crc & 0x0F, TERMINATOR])


def write_dat(path: str | Path, frames: Iterable[Frame]) -> Path:
    """Write frames to a ``.dat`` file and return its path."""
    path = Path(path)
    path.write_bytes(b"".join(encode_frame(frame) for frame in frames))
    return path


def expected_events(frames: Sequence[Frame]) -> npt.NDArray[np.void]:
    """The events the parser must decode from ``frames``, in file order.

    Within a frame the parser emits channels in ascending order.
    """
    rows = [
        (f.node, f.board, f.rena, h.channel, h.pha, h.u, h.v)
        for f in frames
        for h in sorted(f.hits, key=lambda hit: hit.channel)
    ]
    return np.array(rows, dtype=EVENT_DTYPE)


def random_frames(
    n_frames: int,
    *,
    seed: int = 0,
    nodes: Sequence[int] = (0, 1, 2, 3),
    boards: Sequence[int] = (15, 16, 30),
    max_hits: int = 4,
    channels: Sequence[int] = tuple(range(36)),
) -> list[Frame]:
    """Random frames over the given nodes/boards, both RENAs and ``channels``.

    Args:
        n_frames: Number of frames.
        seed: RNG seed (the output is deterministic).
        nodes: Node numbers to draw from (node 0 events must be dropped).
        boards: Board numbers to draw from.
        max_hits: Maximum triggered channels per frame (at least 1).
        channels: Channel numbers to draw from, active and inactive.

    Returns:
        The frames, in file order.
    """
    rng = np.random.default_rng(seed)
    frames = []
    channel_pool = np.asarray(channels)
    for i in range(n_frames):
        n_hits = int(rng.integers(1, min(max_hits, len(channel_pool)) + 1))
        chosen = rng.choice(channel_pool, size=n_hits, replace=False)
        values = rng.integers(0, 4096, size=(n_hits, 3))
        frames.append(
            Frame(
                node=int(rng.choice(nodes)),
                board=int(rng.choice(boards)),
                rena=int(rng.integers(0, 2)),
                timestamp=1000 + 17 * i,
                hits=tuple(
                    Hit(int(ch), int(p), int(u), int(v)) for ch, (p, u, v) in zip(chosen, values)
                ),
            )
        )
    return frames
