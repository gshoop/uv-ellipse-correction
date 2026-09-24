"""Active-channel definitions, electrode labels and polarity.

A board carries two RENA-3 ASICs with 36 channels each, but only 47 of those
72 channels are wired to electrodes (39 anodes and 8 cathodes): RENA 0
channels 4-28 and RENA 1 channels 7-28. Events on any other channel are
dropped when the UV cache is built (plan decision D2).

Electrode labels (``A01``..``A39``, ``C01``..``C08``) and the cathode rule
come from adc2kev's :class:`~adc2kev.tools.electrode_map.ElectrodeMap`, which
reads the per-parity ``.cmf`` load-balance files. For boards 15-30 its
cathode rule agrees with the C++ RadialAnalysis ``isCathode``
(``main_radial.cpp:279``): cathodes are channels 25-28 of both RENAs on even
boards, and RENA 0 channels 4-7 or RENA 1 channels 7-10 on odd boards.
"""

from __future__ import annotations

import functools
from typing import Literal

import numpy as np
import numpy.typing as npt
from adc2kev.tools.electrode_map import ElectrodeMap
from adc2kev.tools.geometry import ACTIVE_BOARDS, is_active_board

__all__ = [
    "ACTIVE_BOARDS",
    "ACTIVE_CHANNELS",
    "ACTIVE_CHANNEL_RANGES",
    "N_ACTIVE_CHANNELS",
    "PolarityName",
    "active_channel_mask",
    "electrode_label",
    "electrode_map",
    "is_active_board",
    "is_active_channel",
    "is_cathode",
    "polarity_name",
]

PolarityName = Literal["anode", "cathode"]

# Inclusive (first, last) active channel of each RENA-3 ASIC.
ACTIVE_CHANNEL_RANGES: dict[int, tuple[int, int]] = {0: (4, 28), 1: (7, 28)}

ACTIVE_CHANNELS: tuple[tuple[int, int], ...] = tuple(
    (rena, channel)
    for rena, (first, last) in sorted(ACTIVE_CHANNEL_RANGES.items())
    for channel in range(first, last + 1)
)
"""Every active ``(rena, channel)`` pair, sorted (47 per board)."""

N_ACTIVE_CHANNELS: int = len(ACTIVE_CHANNELS)

_ACTIVE_SET: frozenset[tuple[int, int]] = frozenset(ACTIVE_CHANNELS)


def is_active_channel(rena: int, channel: int) -> bool:
    """Return True if ``(rena, channel)`` is wired to an electrode.

    Args:
        rena: RENA-3 ASIC on the board (0 or 1).
        channel: Channel on the ASIC (0-35).

    Returns:
        True for RENA 0 channels 4-28 and RENA 1 channels 7-28, else False.
    """
    return (rena, channel) in _ACTIVE_SET


def active_channel_mask(
    rena: npt.ArrayLike,
    channel: npt.ArrayLike,
) -> npt.NDArray[np.bool_]:
    """Vectorised :func:`is_active_channel` over matching arrays.

    Uses range comparisons rather than a lookup table: on 2M-row uint8
    batches this is several times faster than fancy indexing into a table.

    Args:
        rena: RENA numbers (any integer dtype).
        channel: Channel numbers, broadcastable against ``rena``.

    Returns:
        Boolean array, True where the ``(rena, channel)`` pair is active.
    """
    rena_arr = np.asarray(rena)
    channel_arr = np.asarray(channel)
    mask = np.zeros(np.broadcast_shapes(rena_arr.shape, channel_arr.shape), dtype=np.bool_)
    for rena_id, (first, last) in ACTIVE_CHANNEL_RANGES.items():
        mask |= (rena_arr == rena_id) & (channel_arr >= first) & (channel_arr <= last)
    return mask


@functools.lru_cache(maxsize=1)
def electrode_map() -> ElectrodeMap:
    """Return the shared adc2kev :class:`ElectrodeMap` (packaged ``.cmf`` files)."""
    return ElectrodeMap.default()


def electrode_label(board: int, rena: int, channel: int) -> str:
    """Return the electrode label of a channel, e.g. ``"A17"`` or ``"C03"``.

    The mapping depends on the board parity (even and odd boards are wired
    differently).

    Args:
        board: Board (FPGA) number.
        rena: RENA-3 ASIC on the board (0 or 1).
        channel: Channel on the ASIC.

    Returns:
        The electrode label.

    Raises:
        KeyError: If ``(rena, channel)`` is not an active channel.
    """
    return electrode_map().electrode_label(board, rena, channel)


def is_cathode(board: int, rena: int, channel: int) -> bool:
    """Return True if the channel reads a cathode electrode.

    Args:
        board: Board (FPGA) number.
        rena: RENA-3 ASIC on the board (0 or 1).
        channel: Channel on the ASIC.

    Raises:
        KeyError: If ``(rena, channel)`` is not an active channel.
    """
    return electrode_map().is_cathode(board, rena, channel)


def polarity_name(board: int, rena: int, channel: int) -> PolarityName:
    """Return ``"cathode"`` or ``"anode"`` for a channel (the CSV ``polarity`` column).

    Args:
        board: Board (FPGA) number.
        rena: RENA-3 ASIC on the board (0 or 1).
        channel: Channel on the ASIC.

    Raises:
        KeyError: If ``(rena, channel)`` is not an active channel.
    """
    return "cathode" if is_cathode(board, rena, channel) else "anode"
