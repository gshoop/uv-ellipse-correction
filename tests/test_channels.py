"""Tests for uvcorr.channels: active channels, electrode labels, polarity."""

from __future__ import annotations

import numpy as np
import pytest
from adc2kev.parser import PacketParser

from tests.conftest import SyntheticFile
from uvcorr import channels
from uvcorr.channels import (
    ACTIVE_CHANNELS,
    N_ACTIVE_CHANNELS,
    active_channel_mask,
    electrode_label,
    electrode_map,
    is_active_channel,
    is_cathode,
    polarity_name,
)

ALL_BOARDS = tuple(range(15, 31))


def cpp_is_cathode(board: int, rena: int, channel: int) -> bool:
    """``isCathode`` of ``~/DataProcessing/EllipseCorrection/main_radial.cpp:279``."""
    if board % 2 == 0:
        return 25 <= channel <= 28
    if rena == 0:
        return 4 <= channel <= 7
    if rena == 1:
        return 7 <= channel <= 10
    return False


class TestActiveChannels:
    def test_active_channel_list(self) -> None:
        expected = [(0, ch) for ch in range(4, 29)] + [(1, ch) for ch in range(7, 29)]
        assert list(ACTIVE_CHANNELS) == expected
        assert N_ACTIVE_CHANNELS == len(ACTIVE_CHANNELS) == 47
        assert list(ACTIVE_CHANNELS) == sorted(ACTIVE_CHANNELS)

    @pytest.mark.parametrize(
        ("rena", "channel", "active"),
        [
            (0, 3, False),
            (0, 4, True),
            (0, 28, True),
            (0, 29, False),
            (1, 6, False),
            (1, 7, True),
            (1, 28, True),
            (1, 29, False),
            (0, 0, False),
            (1, 35, False),
            (2, 10, False),
            (-1, 10, False),
        ],
    )
    def test_is_active_channel_boundaries(self, rena: int, channel: int, active: bool) -> None:
        assert is_active_channel(rena, channel) is active

    @pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.int16, np.int64])
    def test_mask_matches_scalar_rule(self, dtype: type[np.integer]) -> None:
        rena, channel = np.meshgrid(np.arange(2), np.arange(64), indexing="ij")
        rena = rena.ravel().astype(dtype)
        channel = channel.ravel().astype(dtype)
        mask = active_channel_mask(rena, channel)
        expected = [is_active_channel(int(r), int(c)) for r, c in zip(rena, channel)]
        assert mask.dtype == np.bool_
        assert mask.tolist() == expected
        assert int(mask.sum()) == 47

    def test_mask_rejects_other_renas_and_broadcasts(self) -> None:
        assert not active_channel_mask(np.array([2, 3, 255]), np.array([10, 10, 10])).any()
        assert active_channel_mask(0, np.arange(36)).tolist() == [4 <= ch <= 28 for ch in range(36)]
        assert active_channel_mask([], []).shape == (0,)


class TestElectrodes:
    def test_electrode_map_is_cached(self) -> None:
        assert electrode_map() is electrode_map()

    @pytest.mark.parametrize("board", [15, 16])
    def test_each_board_has_39_anodes_and_8_cathodes(self, board: int) -> None:
        labels = [electrode_label(board, r, c) for r, c in ACTIVE_CHANNELS]
        assert len(set(labels)) == 47
        assert sorted(label for label in labels if label[0] == "A") == [
            f"A{i:02d}" for i in range(1, 40)
        ]
        assert sorted(label for label in labels if label[0] == "C") == [
            f"C{i:02d}" for i in range(1, 9)
        ]

    def test_known_labels(self) -> None:
        # From adc2kev's packaged .cmf load-balance files.
        assert electrode_label(15, 0, 4) == "C01"
        assert electrode_label(16, 0, 25) == "C08"
        assert electrode_label(16, 1, 28) == "C02"

    def test_inactive_channel_has_no_label(self) -> None:
        with pytest.raises(KeyError):
            electrode_label(16, 0, 2)
        with pytest.raises(KeyError):
            is_cathode(16, 1, 6)

    @pytest.mark.parametrize("board", ALL_BOARDS)
    def test_cathode_rule_matches_cpp_isCathode(self, board: int) -> None:
        # Plan 4.5: the ElectrodeMap cathode rule must agree with RadialAnalysis.
        mismatches = [
            (rena, ch)
            for rena, ch in ACTIVE_CHANNELS
            if is_cathode(board, rena, ch) != cpp_is_cathode(board, rena, ch)
        ]
        assert mismatches == []
        assert sum(is_cathode(board, r, c) for r, c in ACTIVE_CHANNELS) == 8

    def test_labels_and_polarity_agree(self) -> None:
        for board in ALL_BOARDS:
            for rena, ch in ACTIVE_CHANNELS:
                label = electrode_label(board, rena, ch)
                name = polarity_name(board, rena, ch)
                assert name == ("cathode" if label.startswith("C") else "anode")

    def test_parser_polarity_matches_cathode_rule(self, synthetic_file: SyntheticFile) -> None:
        # The polarity decoded by adc2kev's parser (0 = cathode) agrees with is_cathode.
        checked = 0
        for batch in PacketParser(synthetic_file.path).iter_event_arrays():
            keep = active_channel_mask(batch.rena_num, batch.channel_num)
            for board, rena, ch, pol in zip(
                batch.board_num[keep].tolist(),
                batch.rena_num[keep].tolist(),
                batch.channel_num[keep].tolist(),
                batch.polarity[keep].tolist(),
            ):
                assert pol == (0 if is_cathode(board, rena, ch) else 1)
                checked += 1
        assert checked > 1000


def test_reexports_geometry() -> None:
    assert channels.ACTIVE_BOARDS == ALL_BOARDS
    assert channels.is_active_board(15) and not channels.is_active_board(14)
