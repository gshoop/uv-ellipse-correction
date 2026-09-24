"""Tests for the System Map widgets in ``uvcorr.gui.system_map``.

The widgets are driven offscreen through ``qtbot``: cell geometry comes from
the public ``cell_rect`` / ``cell_at`` helpers, clicks land on computed cell
centres and the emitted ``(node, board, rena, channel)`` is compared with the
model's cell. Also covers the colour bar (``uvcorr.gui.map_colors``) and the
flow layout used by the map's top row.
"""

from __future__ import annotations

import math
from contextlib import AbstractContextManager
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from PyQt6.QtCore import QCoreApplication, QEvent, QPoint, QRectF, Qt
from PyQt6.QtGui import QAction, QColor, QHelpEvent
from PyQt6.QtWidgets import QApplication, QMenu, QToolTip, QWidget
from pytestqt.qtbot import QtBot

from uvcorr.gui._flow_layout import FlowLayout
from uvcorr.gui._system_map_model import (
    GRID_NODES,
    NO_DATA_SUMMARY,
    ChannelAddress,
    ChannelTuple,
    ChannelView,
    SystemMapModel,
    build_system_map,
    format_board_summary,
    format_summary,
)
from uvcorr.gui.map_colors import (
    CATEGORIES,
    CATEGORY_COLORS,
    CATEGORY_FAILED,
    CATEGORY_LABELS,
    CATEGORY_NOT_FITTED,
    CATEGORY_OK,
    COLOR_MODES,
    METRIC_SPECS,
    MODE_STATUS,
    NO_DATA_COLOR,
    NOT_FITTED_COLOR,
    ColorBarWidget,
    metric_color,
)
from uvcorr.gui.system_map import (
    OVERRIDE_LEGEND_LABEL,
    OVERRIDE_TICK_COLOR,
    SELECTED_BOARD_OUTLINE,
    SELECTED_CHANNEL_OUTLINE,
    VIEW_ANODES,
    VIEW_CATHODES,
    BoardStripWidget,
    PanelGridWidget,
    SystemMapWidget,
)
from uvcorr.options import (
    FLAG_HIGH_REJECTION,
    STATUS_FIT_FAILED,
    STATUS_OK,
    STATUS_TOO_FEW_EVENTS,
)

pytestmark = pytest.mark.gui

# ---------------------------------------------------------------------------
# Synthetic state
# ---------------------------------------------------------------------------

NODE, BOARD = 3, 17  # odd board, panel 1
EVEN_BOARD = 18
PANEL2 = (8, 25)
NOT_FITTED_BOARD = (1, 15)  # data, no views
NO_DATA_BOARD = (4, 17)

_LAYOUT: list[SystemMapModel] = []


def _layout() -> SystemMapModel:
    if not _LAYOUT:
        _LAYOUT.append(build_system_map({}))
    return _LAYOUT[0]


def _anode(node: int, board: int, position: int) -> ChannelAddress:
    return _layout().boards[(node, board)].anodes[position - 1].channel


def _cathode(node: int, board: int, index: int) -> ChannelAddress:
    return _layout().boards[(node, board)].cathodes[index - 1].channel


def _ok(sigma: float, **extra: float) -> ChannelView:
    return ChannelView(
        STATUS_OK, metrics={"n_events": 1000.0, "post_sigma": sigma, "phase_ks": 0.01, **extra}
    )


def _sample_views() -> dict[ChannelTuple, ChannelView]:
    """Boards (3, 17), (3, 18) and (8, 25): a mix of statuses, an override, NaN metrics."""
    views: dict[ChannelTuple, ChannelView] = {}
    for position in range(1, ANODES + 1):
        views[_anode(NODE, BOARD, position)] = _ok(10.0 + position)
        views[_anode(NODE, EVEN_BOARD, position)] = _ok(20.0 + position)
        views[_anode(*PANEL2, position)] = _ok(30.0 + position)
    views[_anode(NODE, BOARD, 2)] = ChannelView(
        STATUS_OK, (FLAG_HIGH_REJECTION,), metrics={"post_sigma": 12.0}
    )
    views[_anode(NODE, BOARD, 3)] = ChannelView(STATUS_FIT_FAILED, metrics={"n_events": 90.0})
    views[_anode(NODE, BOARD, 4)] = ChannelView(STATUS_TOO_FEW_EVENTS, metrics={"n_events": 9.0})
    views[_anode(NODE, BOARD, 5)] = ChannelView(
        STATUS_OK, options_source="override", metrics={"post_sigma": math.nan}
    )
    views[_cathode(NODE, BOARD, 1)] = ChannelView(STATUS_FIT_FAILED)
    for index in range(2, 9):
        views[_cathode(NODE, BOARD, index)] = _ok(100.0 * index)
    return views


ANODES = 39
ACTIVE_BOARDS = [(NODE, BOARD), (NODE, EVEN_BOARD), PANEL2, NOT_FITTED_BOARD]

CTRL = Qt.KeyboardModifier.ControlModifier
CTRL_SHIFT = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier


def _center(rect: QRectF) -> QPoint:
    """Integer centre of a cell or tile rectangle, where clicks are aimed."""
    return rect.center().toPoint()


@pytest.fixture()
def map_widget(qtbot: QtBot) -> SystemMapWidget:
    """The whole dock widget, shown offscreen, with the sample state."""
    widget = SystemMapWidget()
    qtbot.addWidget(widget)
    widget.resize(1600, 420)
    widget.set_state(_sample_views(), active_boards=ACTIVE_BOARDS)
    widget.show()
    qtbot.waitExposed(widget)
    return widget


def _grid1(widget: SystemMapWidget) -> PanelGridWidget:
    return widget.panel_grids[0]


def _model(widget: SystemMapWidget) -> SystemMapModel:
    model = widget.model
    assert model is not None
    return model


# ---------------------------------------------------------------------------
# PanelGridWidget
# ---------------------------------------------------------------------------


@pytest.fixture()
def grid(qtbot: QtBot) -> PanelGridWidget:
    """Panel 1 grid at a fixed size showing the sample model."""
    widget = PanelGridWidget(1)
    qtbot.addWidget(widget)
    widget.resize(760, 320)
    widget.set_model(build_system_map(_sample_views(), ACTIVE_BOARDS))
    return widget


class TestPanelGrid:
    def test_defaults_and_invalid_panel(self, grid: PanelGridWidget) -> None:
        assert grid.panel == 1
        assert grid.nodes == (1, 2, 3, 4, 5)
        assert grid.view == VIEW_ANODES
        assert grid.sizeHint().width() > grid.minimumSizeHint().width()
        with pytest.raises(ValueError, match="panel"):
            PanelGridWidget(3)

    @pytest.mark.parametrize("view", [VIEW_ANODES, VIEW_CATHODES])
    def test_cell_rect_cell_at_round_trip(self, grid: PanelGridWidget, view: str) -> None:
        grid.set_view(view)
        n_cells = ANODES if view == VIEW_ANODES else 8
        for node, board in ((1, 15), (NODE, BOARD), (5, 30)):
            for position in range(1, n_cells + 1):
                point = _center(grid.cell_rect(node, board, position))
                assert grid.cell_at(point) == (node, board, position)

    def test_outside_and_gaps(self, grid: PanelGridWidget) -> None:
        assert grid.cell_at(QPoint(0, 0)) is None
        tile = grid.tile_rect(1, 15)
        assert grid.tile_at(QPoint(int(tile.right()) + 1, int(tile.center().y()))) is None
        with pytest.raises(ValueError, match="node"):
            grid.tile_rect(6, 15)
        with pytest.raises(ValueError, match="position"):
            grid.cell_rect(1, 15, 40)
        with pytest.raises(ValueError, match="view"):
            grid.set_view("strips")

    def test_click_emits_model_channel(self, qtbot: QtBot, grid: PanelGridWidget) -> None:
        expected = _anode(NODE, BOARD, 7)
        with qtbot.waitSignal(grid.cell_clicked, timeout=1000) as blocker:
            qtbot.mouseClick(
                grid, Qt.MouseButton.LeftButton, pos=_center(grid.cell_rect(NODE, BOARD, 7))
            )
        assert blocker.args == [expected]

    def test_click_no_data_tile_emits_board(self, qtbot: QtBot, grid: PanelGridWidget) -> None:
        with (
            qtbot.assertNotEmitted(grid.cell_clicked),
            qtbot.waitSignal(grid.board_clicked, timeout=1000) as blocker,
        ):
            qtbot.mouseClick(
                grid, Qt.MouseButton.LeftButton, pos=_center(grid.tile_rect(*NO_DATA_BOARD))
            )
        assert blocker.args == list(NO_DATA_BOARD)

    def test_right_click_requests_context(self, qtbot: QtBot, grid: PanelGridWidget) -> None:
        with qtbot.waitSignal(grid.context_requested, timeout=1000) as blocker:
            qtbot.mouseClick(
                grid, Qt.MouseButton.RightButton, pos=_center(grid.cell_rect(NODE, BOARD, 12))
            )
        node, board, channel, global_pos = blocker.args
        assert (node, board, channel) == (NODE, BOARD, _anode(NODE, BOARD, 12))
        assert isinstance(global_pos, QPoint)

    def test_tooltips(self, grid: PanelGridWidget) -> None:
        model = grid.model
        assert model is not None
        cell = model.boards[(NODE, BOARD)].anodes[4]
        text = grid.tooltip_at(_center(grid.cell_rect(NODE, BOARD, 5)))
        assert text == cell.tooltip
        assert "Options: override" in text
        no_data = model.boards[NO_DATA_BOARD]
        assert grid.tooltip_at(_center(grid.tile_rect(*NO_DATA_BOARD))) == format_board_summary(
            no_data
        )
        assert grid.tooltip_at(QPoint(0, 0)) is None

    def test_help_event_shows_cell_tooltip(self, grid: PanelGridWidget) -> None:
        grid.show()
        pos = _center(grid.cell_rect(NODE, BOARD, 1))
        event = QHelpEvent(QEvent.Type.ToolTip, pos, grid.mapToGlobal(pos))
        assert QApplication.sendEvent(grid, event)
        assert QToolTip.text() == grid.tooltip_at(pos)
        QToolTip.hideText()

    def test_cell_colours(self, grid: PanelGridWidget) -> None:
        grid.set_view(VIEW_CATHODES)  # wide cells: the centre pixel is inside the fill
        image = grid.grab().toImage()
        assert image.pixelColor(_center(grid.cell_rect(NODE, BOARD, 1))) == QColor(
            CATEGORY_COLORS[CATEGORY_FAILED]
        )
        assert image.pixelColor(_center(grid.cell_rect(NODE, BOARD, 2))) == QColor(
            CATEGORY_COLORS[CATEGORY_OK]
        )
        assert image.pixelColor(_center(grid.cell_rect(*NOT_FITTED_BOARD, 3))) == QColor(
            NOT_FITTED_COLOR
        )
        assert image.pixelColor(_center(grid.tile_rect(*NO_DATA_BOARD))) == QColor(NO_DATA_COLOR)

    def test_selection_outlines(self, grid: PanelGridWidget) -> None:
        grid.set_view(VIEW_CATHODES)
        grid.set_selection(NODE, BOARD, _cathode(NODE, BOARD, 3))
        image = grid.grab().toImage()
        tile = grid.tile_rect(NODE, BOARD)
        assert image.pixelColor(QPoint(int(tile.left()) - 1, int(tile.center().y()))) == QColor(
            SELECTED_BOARD_OUTLINE
        )
        cell = grid.cell_rect(NODE, BOARD, 3)
        top = QPoint(int(cell.center().x()), int(cell.top()) + 1)
        assert image.pixelColor(top) == QColor(SELECTED_CHANNEL_OUTLINE)
        grid.set_selection(None, None, None)
        assert grid.grab().toImage().pixelColor(top) != QColor(SELECTED_CHANNEL_OUTLINE)


# ---------------------------------------------------------------------------
# BoardStripWidget
# ---------------------------------------------------------------------------


@pytest.fixture()
def strip(qtbot: QtBot) -> BoardStripWidget:
    widget = BoardStripWidget()
    qtbot.addWidget(widget)
    widget.resize(1100, 70)
    widget.set_board(build_system_map(_sample_views(), ACTIVE_BOARDS).boards[(NODE, BOARD)])
    return widget


class TestBoardStrip:
    def test_geometry_and_click(self, qtbot: QtBot, strip: BoardStripWidget) -> None:
        board = strip.board
        assert board is not None
        for kind, count in (("anode", ANODES), ("cathode", 8)):
            for position in range(1, count + 1):
                cell = strip.cell_at(_center(strip.cell_rect(kind, position)))
                assert cell is board.cells(kind)[position - 1]
        with qtbot.waitSignal(strip.cell_clicked, timeout=1000) as blocker:
            qtbot.mouseClick(
                strip, Qt.MouseButton.LeftButton, pos=_center(strip.cell_rect("cathode", 4))
            )
        assert blocker.args == [_cathode(NODE, BOARD, 4)]
        assert strip.selected_channel == _cathode(NODE, BOARD, 4)
        assert strip.header_text == format_board_summary(board)

    def test_keyboard_moves_along_the_board(self, qtbot: QtBot, strip: BoardStripWidget) -> None:
        board = strip.board
        assert board is not None
        with qtbot.waitSignal(strip.cell_clicked, timeout=1000) as blocker:
            qtbot.keyClick(strip, Qt.Key.Key_Right)  # nothing selected: first cell
        assert blocker.args == [board.anodes[0].channel]
        with qtbot.waitSignal(strip.cell_clicked, timeout=1000) as blocker:
            qtbot.keyClick(strip, Qt.Key.Key_End)
        assert blocker.args == [board.cathodes[-1].channel]
        with qtbot.assertNotEmitted(strip.cell_clicked):
            qtbot.keyClick(strip, Qt.Key.Key_Right)  # already at the end
        with qtbot.waitSignal(strip.cell_clicked, timeout=1000) as blocker:
            qtbot.keyClick(strip, Qt.Key.Key_Left)
        assert blocker.args == [board.cathodes[-2].channel]
        # Stepping from the last anode crosses into the cathodes.
        strip.set_selected_channel(tuple(board.anodes[-1].channel))  # type: ignore[arg-type]
        with qtbot.waitSignal(strip.cell_clicked, timeout=1000) as blocker:
            qtbot.keyClick(strip, Qt.Key.Key_Right)
        assert blocker.args == [board.cathodes[0].channel]

    @pytest.mark.parametrize(
        ("key", "modifier", "signal", "value"),
        [
            (Qt.Key.Key_Up, Qt.KeyboardModifier.NoModifier, "board_step_requested", -1),
            (Qt.Key.Key_Down, Qt.KeyboardModifier.NoModifier, "board_step_requested", 1),
            (Qt.Key.Key_Left, CTRL, "node_step_requested", -1),
            (Qt.Key.Key_Right, CTRL, "node_step_requested", 1),
            (Qt.Key.Key_Left, CTRL_SHIFT, "node_step_requested", -5),
            (Qt.Key.Key_Right, CTRL_SHIFT, "node_step_requested", 5),
        ],
    )
    def test_step_requests(
        self,
        qtbot: QtBot,
        strip: BoardStripWidget,
        key: Qt.Key,
        modifier: Qt.KeyboardModifier,
        signal: str,
        value: int,
    ) -> None:
        with qtbot.waitSignal(getattr(strip, signal), timeout=1000) as blocker:
            qtbot.keyClick(strip, key, modifier)
        assert blocker.args == [value]

    def test_no_data_board_paints_dark(self, strip: BoardStripWidget) -> None:
        strip.set_board(build_system_map({}).boards[NO_DATA_BOARD])
        image = strip.grab().toImage()
        rect = strip.cell_rect("cathode", 2)
        corner = QPoint(int(rect.left()) + 2, int(rect.top()) + 2)
        assert image.pixelColor(corner) == QColor(NO_DATA_COLOR)
        assert strip.header_text.endswith(" - no data")


# ---------------------------------------------------------------------------
# SystemMapWidget
# ---------------------------------------------------------------------------


class TestSystemMapWidget:
    def test_construction(self, map_widget: SystemMapWidget) -> None:
        model = _model(map_widget)
        assert map_widget.view == VIEW_ANODES
        assert map_widget.color_mode == MODE_STATUS
        assert map_widget.current_board is None and map_widget.current_channel is None
        assert map_widget.summary_label.text() == format_summary(model)
        assert map_widget.summary_label.text().startswith("Anodes: ")
        assert map_widget.warning_label.isHidden()
        assert map_widget.color_bar.isHidden()
        assert map_widget.legend_labels() == [
            *(CATEGORY_LABELS[c] for c in CATEGORIES),
            OVERRIDE_LEGEND_LABEL,
        ]
        assert not map_widget.override_legend.isHidden()
        assert [nb for nb, b in model.boards.items() if b.has_data] == sorted(ACTIVE_BOARDS)
        assert map_widget.board_strip.header_text == "No board selected"
        assert not map_widget.grab().isNull()

    def test_click_on_cell_emits_channel(self, qtbot: QtBot, map_widget: SystemMapWidget) -> None:
        grid = _grid1(map_widget)
        expected = _anode(NODE, BOARD, 7)
        with (
            qtbot.assertNotEmitted(map_widget.board_selected),
            qtbot.waitSignal(map_widget.channel_selected, timeout=1000) as blocker,
        ):
            qtbot.mouseClick(
                grid, Qt.MouseButton.LeftButton, pos=_center(grid.cell_rect(NODE, BOARD, 7))
            )
        (channel,) = blocker.args
        assert channel == expected
        assert isinstance(channel, ChannelAddress)
        assert tuple(channel) == (NODE, BOARD, expected.rena, expected.channel)
        assert map_widget.current_board == (NODE, BOARD)
        assert map_widget.current_channel == expected
        assert map_widget.board_strip.board is _model(map_widget).boards[(NODE, BOARD)]
        assert map_widget.board_strip.hasFocus()

    def test_click_on_panel2_and_cathode_view(
        self, qtbot: QtBot, map_widget: SystemMapWidget
    ) -> None:
        map_widget.cathode_radio.setChecked(True)
        assert map_widget.view == VIEW_CATHODES
        assert all(grid.view == VIEW_CATHODES for grid in map_widget.panel_grids)
        grid = map_widget.panel_grids[1]
        with qtbot.waitSignal(map_widget.channel_selected, timeout=1000) as blocker:
            qtbot.mouseClick(
                grid, Qt.MouseButton.LeftButton, pos=_center(grid.cell_rect(*PANEL2, 6))
            )
        assert blocker.args == [_cathode(*PANEL2, 6)]
        map_widget.anode_radio.setChecked(True)
        assert map_widget.view == VIEW_ANODES
        map_widget.set_view(VIEW_CATHODES)
        assert map_widget.cathode_radio.isChecked()

    def test_click_on_no_data_tile_selects_board(
        self, qtbot: QtBot, map_widget: SystemMapWidget
    ) -> None:
        grid = _grid1(map_widget)
        with (
            qtbot.assertNotEmitted(map_widget.channel_selected),
            qtbot.waitSignal(map_widget.board_selected, timeout=1000) as blocker,
        ):
            qtbot.mouseClick(
                grid, Qt.MouseButton.LeftButton, pos=_center(grid.tile_rect(*NO_DATA_BOARD))
            )
        assert blocker.args == list(NO_DATA_BOARD)
        assert map_widget.current_board == NO_DATA_BOARD
        assert map_widget.current_channel is None

    def test_keyboard_stepping(self, qtbot: QtBot, map_widget: SystemMapWidget) -> None:
        strip = map_widget.board_strip
        grid = _grid1(map_widget)
        qtbot.mouseClick(
            grid, Qt.MouseButton.LeftButton, pos=_center(grid.cell_rect(NODE, BOARD, 10))
        )
        # Right: next electrode on the board.
        with qtbot.waitSignal(map_widget.channel_selected, timeout=1000) as blocker:
            qtbot.keyClick(strip, Qt.Key.Key_Right)
        assert blocker.args == [_anode(NODE, BOARD, 11)]
        # Down: same position on the next board (even board, data).
        with qtbot.waitSignals(
            [map_widget.board_selected, map_widget.channel_selected], order="strict", timeout=1000
        ) as blockers:
            qtbot.keyClick(strip, Qt.Key.Key_Down)
        board_args, channel_args = (list(sig.args) for sig in blockers.all_signals_and_args)
        assert board_args == [NODE, EVEN_BOARD]
        assert channel_args == [_anode(NODE, EVEN_BOARD, 11)]
        # Ctrl+Right onto a board without data: board only, position remembered.
        qtbot.keyClick(strip, Qt.Key.Key_Up)
        with (
            qtbot.assertNotEmitted(map_widget.channel_selected),
            qtbot.waitSignal(map_widget.board_selected, timeout=1000) as blocker,
        ):
            qtbot.keyClick(strip, Qt.Key.Key_Right, CTRL)
        assert blocker.args == list(NO_DATA_BOARD)
        assert map_widget.current_channel is None
        with qtbot.waitSignal(map_widget.channel_selected, timeout=1000) as blocker:
            qtbot.keyClick(strip, Qt.Key.Key_Left, CTRL)
        assert blocker.args == [_anode(NODE, BOARD, 11)]
        # Ctrl+Shift+Right: same column on the other panel.
        with qtbot.waitSignal(map_widget.board_selected, timeout=1000) as blocker:
            qtbot.keyClick(strip, Qt.Key.Key_Right, CTRL_SHIFT)
        assert blocker.args == [NODE + 5, BOARD]
        # Up from board 15 stays put.
        map_widget.set_selection(1, 15)
        with qtbot.assertNotEmitted(map_widget.board_selected):
            qtbot.keyClick(strip, Qt.Key.Key_Up)

    def test_set_selection_emits_nothing(self, qtbot: QtBot, map_widget: SystemMapWidget) -> None:
        cathode = _cathode(NODE, BOARD, 2)
        with (
            qtbot.assertNotEmitted(map_widget.channel_selected),
            qtbot.assertNotEmitted(map_widget.board_selected),
        ):
            map_widget.set_selection(NODE, BOARD, (NODE, BOARD, cathode.rena, cathode.channel))
            assert map_widget.current_channel == cathode
            assert isinstance(map_widget.current_channel, ChannelAddress)
            assert map_widget.board_strip.selected_channel == cathode
            assert map_widget.current_board == (NODE, BOARD)

            map_widget.set_selection(*PANEL2)
            assert map_widget.current_board == PANEL2
            assert map_widget.current_channel is None

            map_widget.set_current_channel(_anode(NODE, EVEN_BOARD, 3))
            assert map_widget.current_board == (NODE, EVEN_BOARD)

            map_widget.set_selection(NODE, 40, None)  # outside the grid
            assert map_widget.current_board is None
            map_widget.set_current_channel((NODE, BOARD, 0, 1))  # not an electrode
            assert map_widget.current_board == (NODE, BOARD)
            assert map_widget.current_channel is None
            map_widget.set_selection(None, None)
            assert map_widget.current_board is None
        with pytest.raises(ValueError, match="not on node"):
            map_widget.set_selection(NODE, BOARD, _anode(*PANEL2, 1))

    def test_set_state_keeps_selection(self, map_widget: SystemMapWidget) -> None:
        channel = _anode(NODE, BOARD, 8)
        map_widget.set_current_channel(channel)
        views = _sample_views()
        views[channel] = ChannelView(STATUS_FIT_FAILED)
        map_widget.set_state(views, active_boards=ACTIVE_BOARDS)
        assert map_widget.current_channel == channel
        strip_board = map_widget.board_strip.board
        assert strip_board is not None
        assert strip_board.anodes[7].category == CATEGORY_FAILED

    def test_colour_mode_switch(self, qtbot: QtBot, map_widget: SystemMapWidget) -> None:
        grid = _grid1(map_widget)
        map_widget.set_view(VIEW_CATHODES)
        point = _center(grid.cell_rect(NODE, BOARD, 3))
        status_fill = QColor(CATEGORY_COLORS[CATEGORY_OK])
        assert grid.grab().toImage().pixelColor(point) == status_fill
        views_before = [c.view for c in _model(map_widget).boards[(NODE, BOARD)].cathodes]

        with (
            qtbot.assertNotEmitted(map_widget.color_mode_changed),
            qtbot.assertNotEmitted(map_widget.channel_selected),
        ):
            map_widget.set_color_mode("post_sigma")
        model = _model(map_widget)
        assert model.color_mode == "post_sigma"
        # Same views (nothing re-read), new fills.
        assert [c.view for c in model.boards[(NODE, BOARD)].cathodes] == views_before
        limits = model.limits_for("cathode")
        assert limits is not None
        expected = metric_color(300.0, limits)
        assert model.boards[(NODE, BOARD)].cathodes[2].fill == expected
        assert grid.grab().toImage().pixelColor(point) == expected
        assert expected != status_fill
        # NaN-valued (failed) cathode 1 is grey.
        assert grid.grab().toImage().pixelColor(_center(grid.cell_rect(NODE, BOARD, 1))) == QColor(
            NOT_FITTED_COLOR
        )

        # The colour bar shows the metric and the limits of the current view.
        spec = METRIC_SPECS["post_sigma"]
        assert not map_widget.color_bar.isHidden()
        assert map_widget.color_bar.name_label.text() == "Post σ:"
        assert map_widget.color_bar.min_label.text() == spec.format(limits[0])
        assert map_widget.color_bar.max_label.text() == spec.format(limits[1])
        assert map_widget.legend_labels() == ["No value", "No data", OVERRIDE_LEGEND_LABEL]
        assert not map_widget.override_legend.isHidden()
        assert map_widget.summary_label.text().startswith("Post σ | Anodes: median")
        map_widget.set_view(VIEW_ANODES)
        anode_limits = model.limits_for("anode")
        assert anode_limits is not None
        assert map_widget.color_bar.limits == anode_limits
        assert map_widget.color_bar.min_label.text() == spec.format(anode_limits[0])

        map_widget.set_color_mode(MODE_STATUS)
        assert map_widget.color_bar.isHidden()
        assert map_widget.legend_labels()[0] == CATEGORY_LABELS[CATEGORY_OK]
        map_widget.set_view(VIEW_CATHODES)
        assert grid.grab().toImage().pixelColor(point) == status_fill
        with pytest.raises(ValueError, match="color mode"):
            map_widget.set_color_mode("rainbow")

    def test_combo_emits_colour_mode_changed(
        self, qtbot: QtBot, map_widget: SystemMapWidget
    ) -> None:
        with qtbot.waitSignal(map_widget.color_mode_changed, timeout=1000) as blocker:
            map_widget.color_mode_combo.setCurrentIndex(COLOR_MODES.index("phase_ks"))
        assert blocker.args == ["phase_ks"]
        assert map_widget.color_mode == "phase_ks"
        assert _model(map_widget).color_mode == "phase_ks"
        map_widget.set_color_mode("axis_ratio")
        assert map_widget.color_mode_combo.currentData() == "axis_ratio"

    def test_metric_mode_before_state(self, qtbot: QtBot) -> None:
        widget = SystemMapWidget()
        qtbot.addWidget(widget)
        widget.set_color_mode("timing_jitter_ns")
        assert not widget.color_bar.isHidden()
        assert widget.color_bar.min_label.text() == "n/a"
        widget.set_state(_sample_views(), active_boards=ACTIVE_BOARDS)
        assert _model(widget).color_mode == "timing_jitter_ns"
        assert widget.summary_label.text() == "Jitter | Anodes: no values | Cathodes: no values"

    def test_informational_flags(self, map_widget: SystemMapWidget) -> None:
        cell = _model(map_widget).boards[(NODE, BOARD)].anodes[1]
        assert cell.category == "flagged"
        map_widget.set_informational_flags([FLAG_HIGH_REJECTION])
        assert map_widget.informational_flags == frozenset({FLAG_HIGH_REJECTION})
        assert _model(map_widget).boards[(NODE, BOARD)].anodes[1].category == CATEGORY_OK
        # Kept across a rebuild.
        map_widget.set_state(_sample_views(), active_boards=ACTIVE_BOARDS)
        assert _model(map_widget).boards[(NODE, BOARD)].anodes[1].category == CATEGORY_OK

    def test_warning_for_boards_outside_the_grid(self, map_widget: SystemMapWidget) -> None:
        views = _sample_views()
        views[(2, 14, 0, 4)] = _ok(1.0)
        map_widget.set_state(views, active_boards=[*ACTIVE_BOARDS, (0, 20)])
        assert not map_widget.warning_label.isHidden()
        assert "2 board(s)" in map_widget.warning_label.text()
        assert "(0, 20), (2, 14)" in map_widget.warning_label.text()
        map_widget.set_state(_sample_views(), active_boards=ACTIVE_BOARDS)
        assert map_widget.warning_label.isHidden()

    def test_board_with_data_and_no_views(self, qtbot: QtBot) -> None:
        widget = SystemMapWidget()
        qtbot.addWidget(widget)
        widget.set_state({}, active_boards=[NOT_FITTED_BOARD], data_channels=[(1, 15, 0, 4)])
        board = _model(widget).boards[NOT_FITTED_BOARD]
        assert board.has_data
        categories = {cell.category for cell in board.anodes + board.cathodes}
        assert categories == {CATEGORY_NOT_FITTED, "no_data"}
        assert widget.summary_label.text().startswith("Anodes: 0\u00a0ok")

    def test_clear(self, map_widget: SystemMapWidget) -> None:
        map_widget.set_current_channel(_anode(NODE, BOARD, 1))
        map_widget.clear()
        assert map_widget.model is None
        assert map_widget.current_board is None
        assert map_widget.summary_label.text() == NO_DATA_SUMMARY
        assert all(grid.model is None for grid in map_widget.panel_grids)

    def test_context_menu_actions(self, qtbot: QtBot, map_widget: SystemMapWidget) -> None:
        channel = _anode(NODE, BOARD, 1)
        menu = map_widget.build_context_menu(NODE, BOARD, channel)
        actions = [a for a in menu.actions() if not a.isSeparator()]
        assert [a.text().split("  ")[0] for a in actions] == ["Fit Channel", "Fit Board"]
        assert "A39" in actions[0].text()  # odd board: position 1 is A39
        with qtbot.waitSignal(map_widget.fit_channel_requested, timeout=1000) as blocker:
            actions[0].trigger()
        assert blocker.args == [channel]
        with qtbot.waitSignal(map_widget.fit_board_requested, timeout=1000) as blocker:
            actions[1].trigger()
        assert blocker.args == [NODE, BOARD]
        assert len(map_widget.build_context_menu(*NO_DATA_BOARD, None).actions()) == 1


# ---------------------------------------------------------------------------
# ColorBarWidget and FlowLayout
# ---------------------------------------------------------------------------


class TestColorBar:
    def test_labels(self, qtbot: QtBot) -> None:
        bar = ColorBarWidget()
        qtbot.addWidget(bar)
        assert bar.spec is None and bar.name_label.text() == ""
        spec = METRIC_SPECS["rejected_fraction"]
        bar.set_metric(spec, (0.0, 0.08))
        assert (bar.name_label.text(), bar.min_label.text(), bar.max_label.text()) == (
            "Rejected fraction:",
            "0%",
            "8%",
        )
        bar.set_metric(spec, None)
        assert bar.min_label.text() == bar.max_label.text() == "n/a"
        bar.resize(300, 20)
        assert not bar.grab().isNull()


class TestFlowLayout:
    def test_wraps_and_skips_hidden_items(self, qtbot: QtBot) -> None:
        host = QWidget()
        qtbot.addWidget(host)
        layout = FlowLayout(host, margin=0, h_spacing=6, v_spacing=2)
        children = []
        for _ in range(4):
            child = QWidget(host)
            child.setFixedSize(100, 20)
            layout.addWidget(child)
            children.append(child)
        assert layout.heightForWidth(500) == 20
        assert layout.heightForWidth(250) == 42
        children[1].hide()
        children[2].hide()
        # Two visible items fit in 206 px once the hidden ones take no space.
        assert layout.heightForWidth(206) == 20
        host.resize(206, 60)
        host.show()
        qtbot.waitExposed(host)
        assert children[3].geometry().x() == children[0].geometry().x() + 106
        assert layout.count() == 4
        assert layout.minimumSize().width() == 100


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def _distinct_views(node: int, board: int) -> dict[ChannelTuple, ChannelView]:
    """Every electrode of ``(node, board)`` with its own post σ (own fill in that mode)."""
    views: dict[ChannelTuple, ChannelView] = {
        _anode(node, board, p): _ok(float(p)) for p in range(1, ANODES + 1)
    }
    views.update({_cathode(node, board, i): _ok(100.0 + 10.0 * i) for i in range(1, 9)})
    return views


class TestPixelAccurateHits:
    """Every painted pixel of a cell selects that cell (adc2kev hit the neighbour).

    Clicks every pixel column across a tile (or the strip), reads the colour
    painted there from ``grab()`` and checks that the emitted channel is the
    cell painted in that colour. Fills are distinct per cell (post σ mode).
    """

    @pytest.mark.parametrize("size", [(1600, 420), (1366, 214)])
    @pytest.mark.parametrize("view", [VIEW_ANODES, VIEW_CATHODES])
    def test_grid(self, qtbot: QtBot, view: str, size: tuple[int, int]) -> None:
        widget = SystemMapWidget()
        qtbot.addWidget(widget)
        widget.resize(*size)
        widget.set_color_mode("post_sigma")
        widget.set_state(_distinct_views(NODE, BOARD))
        widget.set_view(view)
        widget.show()
        qtbot.waitExposed(widget)
        grid = _grid1(widget)
        kind = "anode" if view == VIEW_ANODES else "cathode"
        cells = _model(widget).boards[(NODE, BOARD)].cells(kind)
        fills = {cell.fill.rgb(): cell for cell in cells}
        assert len(fills) == len(cells)
        tile = grid.tile_rect(NODE, BOARD)
        assert tile.width() / len(cells) < 5 or view == VIEW_CATHODES  # narrow anode cells
        image = grid.grab().toImage()
        y = int(tile.bottom()) - 1
        received: list[ChannelAddress] = []
        widget.channel_selected.connect(received.append)
        checked = set()
        for x in range(int(tile.left()) - 2, int(tile.right()) + 3):
            painted = fills.get(image.pixelColor(x, y).rgb())
            if painted is None:
                continue  # separator, tile gap or a neighbouring tile without data
            received.clear()
            qtbot.mouseClick(grid, Qt.MouseButton.LeftButton, pos=QPoint(x, y))
            assert received == [painted.channel], f"x={x}"
            checked.add(painted.position)
        assert checked == {cell.position for cell in cells}

    @pytest.mark.parametrize("width", [250, 1100])
    def test_strip(self, qtbot: QtBot, width: int) -> None:
        strip = BoardStripWidget()
        qtbot.addWidget(strip)
        strip.resize(width, 70)
        model = build_system_map(_distinct_views(NODE, BOARD), color_mode="post_sigma")
        board = model.boards[(NODE, BOARD)]
        strip.set_board(board)
        image = strip.grab().toImage()
        anode_fills = {cell.fill.rgb(): cell for cell in board.anodes}
        cathode_fills = {cell.fill.rgb(): cell for cell in board.cathodes}
        # Anode 1 and cathode 1 share a fill: split the pixels in the middle
        # of the empty slot between the anodes and the cathodes.
        split = strip.cell_rect("anode", ANODES).right() + strip.cell_rect("anode", 1).width() / 2
        y = int(strip.cell_rect("anode", 1).top()) + 2  # above the labels
        received: list[ChannelAddress] = []
        strip.cell_clicked.connect(received.append)
        checked = 0
        for x in range(strip.width()):
            painted = (anode_fills if x + 0.5 < split else cathode_fills).get(
                image.pixelColor(x, y).rgb()
            )
            if painted is None:
                continue
            received.clear()
            qtbot.mouseClick(strip, Qt.MouseButton.LeftButton, pos=QPoint(x, y))
            assert received == [painted.channel], f"x={x}"
            checked += 1
        assert checked > 0.9 * strip.width() * 47 / 48


class TestOverrideTick:
    """Override cells carry a corner tick where cells are wide enough."""

    @staticmethod
    def _corner(rect: QRectF) -> QPoint:
        return QPoint(int(rect.right()) - 2, int(rect.top()) + 1)

    def test_strip(self, strip: BoardStripWidget) -> None:
        board = strip.board
        assert board is not None
        override, plain = board.anodes[4], board.anodes[5]
        assert override.is_override and not plain.is_override
        image = strip.grab().toImage()
        corner = image.pixelColor(self._corner(strip.cell_rect("anode", 5)))
        assert corner == QColor(OVERRIDE_TICK_COLOR) != override.fill
        assert image.pixelColor(self._corner(strip.cell_rect("anode", 6))) == plain.fill
        # Away from the corner the override cell keeps its fill.
        rect = strip.cell_rect("anode", 5)
        assert image.pixelColor(QPoint(int(rect.left()) + 2, int(rect.bottom()) - 2)) == (
            override.fill
        )

    def test_grid_only_when_cells_are_wide(self, grid: PanelGridWidget) -> None:
        views = _sample_views()
        views[_cathode(NODE, BOARD, 3)] = ChannelView(
            STATUS_OK, options_source="override", metrics={"n_events": 5.0}
        )
        grid.set_model(build_system_map(views, ACTIVE_BOARDS))
        model = grid.model
        assert model is not None
        # Anode view: ~4 px cells, no tick.
        override = model.boards[(NODE, BOARD)].anodes[4]
        assert override.is_override
        rect = grid.cell_rect(NODE, BOARD, 5)
        assert rect.width() < 6
        image = grid.grab().toImage()
        assert image.pixelColor(QPoint(int(rect.center().x()), int(rect.top()) + 1)) == (
            override.fill
        )
        # Cathode view: wide cells, the tick is drawn.
        grid.set_view(VIEW_CATHODES)
        image = grid.grab().toImage()
        cathode = model.boards[(NODE, BOARD)].cathodes[2]
        corner = image.pixelColor(self._corner(grid.cell_rect(NODE, BOARD, 3)))
        assert corner == QColor(OVERRIDE_TICK_COLOR) != cathode.fill
        assert image.pixelColor(self._corner(grid.cell_rect(NODE, BOARD, 4))) == (
            model.boards[(NODE, BOARD)].cathodes[3].fill
        )


class TestOnceAndEdges:
    def test_grid_click_emits_channel_selected_once(
        self, qtbot: QtBot, map_widget: SystemMapWidget
    ) -> None:
        grid = _grid1(map_widget)
        expected = _anode(NODE, BOARD, 9)
        received: list[ChannelAddress] = []
        boards: list[tuple[int, int]] = []
        map_widget.channel_selected.connect(received.append)
        map_widget.board_selected.connect(lambda n, b: boards.append((n, b)))
        with qtbot.waitSignal(map_widget.channel_selected, timeout=1000):
            qtbot.mouseClick(
                grid, Qt.MouseButton.LeftButton, pos=_center(grid.cell_rect(NODE, BOARD, 9))
            )
        assert received == [expected]
        assert boards == []
        assert map_widget.board_strip.selected_channel == expected

    def test_keys_emit_once(self, qtbot: QtBot, map_widget: SystemMapWidget) -> None:
        map_widget.set_current_channel(_anode(NODE, BOARD, 9))
        received: list[ChannelAddress] = []
        boards: list[tuple[int, int]] = []
        map_widget.channel_selected.connect(received.append)
        map_widget.board_selected.connect(lambda n, b: boards.append((n, b)))
        qtbot.keyClick(map_widget.board_strip, Qt.Key.Key_Right)
        assert received == [_anode(NODE, BOARD, 10)] and boards == []
        received.clear()
        qtbot.keyClick(map_widget.board_strip, Qt.Key.Key_Down)
        assert boards == [(NODE, EVEN_BOARD)]
        assert received == [_anode(NODE, EVEN_BOARD, 10)]

    def test_down_stops_at_the_last_board(self, qtbot: QtBot, map_widget: SystemMapWidget) -> None:
        map_widget.set_selection(NODE, 30)
        with qtbot.assertNotEmitted(map_widget.board_selected):
            qtbot.keyClick(map_widget.board_strip, Qt.Key.Key_Down)
        assert map_widget.current_board == (NODE, 30)

    def test_step_without_board_is_ignored(self, qtbot: QtBot, map_widget: SystemMapWidget) -> None:
        with qtbot.assertNotEmitted(map_widget.board_selected):
            map_widget.board_strip.board_step_requested.emit(1)
            map_widget.board_strip.node_step_requested.emit(1)
        assert map_widget.current_board is None

    def test_combo_set_to_the_current_mode_emits_nothing(
        self, qtbot: QtBot, map_widget: SystemMapWidget
    ) -> None:
        model = map_widget.model
        with qtbot.assertNotEmitted(map_widget.color_mode_changed):
            map_widget.color_mode_combo.setCurrentIndex(COLOR_MODES.index(MODE_STATUS))
            map_widget.set_color_mode(MODE_STATUS)
        assert map_widget.model is model  # not even recoloured

    def test_view_changed_only_on_user_toggle(
        self, qtbot: QtBot, map_widget: SystemMapWidget
    ) -> None:
        with qtbot.assertNotEmitted(map_widget.view_changed):
            map_widget.set_view(VIEW_CATHODES)
            map_widget.set_view(VIEW_ANODES)
        seen: list[str] = []
        map_widget.view_changed.connect(seen.append)
        qtbot.mouseClick(map_widget.cathode_radio, Qt.MouseButton.LeftButton)
        qtbot.mouseClick(map_widget.cathode_radio, Qt.MouseButton.LeftButton)  # no change
        qtbot.mouseClick(map_widget.anode_radio, Qt.MouseButton.LeftButton)
        assert seen == [VIEW_CATHODES, VIEW_ANODES]
        assert map_widget.view == VIEW_ANODES


NO_DATA_NODE = 7  # (7, 17) has no data in the ring state


def _ring_views() -> dict[ChannelTuple, ChannelView]:
    """Board 17 of every node but 7 has data (anode 1 fitted); no other board does."""
    return {_anode(node, BOARD, 1): _ok(1.0) for node in GRID_NODES if node != NO_DATA_NODE}


class TestNodeRing:
    """Ctrl / Ctrl+Shift + Left/Right move along the node ring (ported from adc2kev)."""

    @pytest.fixture()
    def ring_map(self, map_widget: SystemMapWidget) -> SystemMapWidget:
        map_widget.set_state(_ring_views())
        return map_widget

    @pytest.mark.parametrize(
        ("start", "key", "modifier", "expected"),
        [
            pytest.param(5, Qt.Key.Key_Right, CTRL, 6, id="ctrl-right-crosses-5-to-6"),
            pytest.param(6, Qt.Key.Key_Left, CTRL, 5, id="ctrl-left-crosses-6-to-5"),
            pytest.param(10, Qt.Key.Key_Right, CTRL, 1, id="ctrl-right-wraps-10-to-1"),
            pytest.param(1, Qt.Key.Key_Left, CTRL, 10, id="ctrl-left-wraps-1-to-10"),
            pytest.param(3, Qt.Key.Key_Right, CTRL_SHIFT, 8, id="ctrl-shift-right-3-to-8"),
            pytest.param(8, Qt.Key.Key_Left, CTRL_SHIFT, 3, id="ctrl-shift-left-8-to-3"),
            pytest.param(8, Qt.Key.Key_Right, CTRL_SHIFT, 3, id="ctrl-shift-right-8-to-3"),
            pytest.param(3, Qt.Key.Key_Left, CTRL_SHIFT, 8, id="ctrl-shift-left-3-to-8"),
        ],
    )
    def test_ring_moves(
        self,
        qtbot: QtBot,
        ring_map: SystemMapWidget,
        start: int,
        key: Qt.Key,
        modifier: Qt.KeyboardModifier,
        expected: int,
    ) -> None:
        ring_map.set_current_channel(_anode(start, BOARD, 5))
        received: list[ChannelAddress] = []
        ring_map.channel_selected.connect(received.append)
        qtbot.keyClick(ring_map.board_strip, key, modifier)
        assert received == [_anode(expected, BOARD, 5)]
        assert ring_map.current_board == (expected, BOARD)

    def test_dismissed_context_menu_keeps_the_sweep_position(
        self, qtbot: QtBot, ring_map: SystemMapWidget
    ) -> None:
        """A right-click dismissed while board-only must not move the sweep."""
        ring_map.set_current_channel(_anode(6, BOARD, 5))
        qtbot.keyClick(ring_map.board_strip, Qt.Key.Key_Right, CTRL)  # (7, 17): board only
        assert ring_map.current_channel is None

        calls: list[_MenuExecCall] = []
        with _patched_menu_exec(ring_map, calls, choose=None):
            ring_map._on_context_requested(8, BOARD, _anode(8, BOARD, 20), QPoint(0, 0))
        assert len(calls) == 1
        assert calls[0].selected_channel == _anode(8, BOARD, 20)
        assert ring_map.current_board == (NO_DATA_NODE, BOARD)
        assert ring_map.current_channel is None

        with qtbot.waitSignal(ring_map.channel_selected, timeout=1000) as blocker:
            qtbot.keyClick(ring_map.board_strip, Qt.Key.Key_Right, CTRL)
        assert blocker.args == [_anode(8, BOARD, 5)]


@dataclass
class _MenuExecCall:
    """What ``QMenu.exec`` saw: the menu's actions and the map's selection at that moment."""

    pos: object
    action_texts: list[str]
    selected_board: tuple[int, int] | None
    selected_channel: ChannelAddress | None


def _patched_menu_exec(
    map_widget: SystemMapWidget, calls: list[_MenuExecCall], choose: int | None = None
) -> AbstractContextManager[object]:
    """Replace ``QMenu.exec`` with a recorder that returns action ``choose`` or None.

    A plain function is installed on the class (not a MagicMock) so it binds
    like a method and receives the menu; the menu's actions are read while it
    is alive, because the widget ``deleteLater``-s it after ``exec``.
    """

    def fake_exec(menu: QMenu, pos: QPoint, action: QAction | None = None) -> QAction | None:
        actions = [a for a in menu.actions() if not a.isSeparator()]
        calls.append(
            _MenuExecCall(
                pos=pos,
                action_texts=[a.text() for a in actions],
                selected_board=map_widget.current_board,
                selected_channel=map_widget.current_channel,
            )
        )
        if choose is None:
            return None
        actions[choose].trigger()
        return actions[choose]

    return patch.object(QMenu, "exec", new=fake_exec)


def _flush_deferred_deletes() -> None:
    app = QCoreApplication.instance()
    assert app is not None
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


class TestContextMenuFlow:
    """The right-click slot: select, show the menu, restore on dismiss (ported)."""

    def test_right_click_selects_cell_and_restores_on_dismiss(
        self, qtbot: QtBot, map_widget: SystemMapWidget
    ) -> None:
        map_widget.set_current_channel(_anode(NODE, BOARD, 1))
        grid = _grid1(map_widget)
        calls: list[_MenuExecCall] = []
        with (
            _patched_menu_exec(map_widget, calls),
            qtbot.assertNotEmitted(map_widget.channel_selected),
            qtbot.assertNotEmitted(map_widget.board_selected),
            qtbot.assertNotEmitted(map_widget.fit_channel_requested),
            qtbot.assertNotEmitted(map_widget.fit_board_requested),
        ):
            qtbot.mouseClick(
                grid, Qt.MouseButton.RightButton, pos=_center(grid.cell_rect(NODE, BOARD, 9))
            )
        assert len(calls) == 1
        call = calls[0]
        assert isinstance(call.pos, QPoint)
        assert call.action_texts[0].startswith("Fit Channel") and "RENA" in call.action_texts[0]
        assert call.action_texts[1] == f"Fit Board  Node {NODE} Board {BOARD}"
        assert call.selected_channel == _anode(NODE, BOARD, 9)  # while the menu was open
        assert map_widget.current_channel == _anode(NODE, BOARD, 1)  # restored
        assert map_widget.board_strip.selected_channel == _anode(NODE, BOARD, 1)
        # The menu was released with deleteLater.
        _flush_deferred_deletes()
        assert map_widget.findChildren(QMenu) == []

    def test_right_click_then_action_keeps_the_selection(
        self, qtbot: QtBot, map_widget: SystemMapWidget
    ) -> None:
        map_widget.set_current_channel(_anode(NODE, BOARD, 1))
        grid = _grid1(map_widget)
        calls: list[_MenuExecCall] = []
        with (
            _patched_menu_exec(map_widget, calls, choose=0),
            qtbot.waitSignal(map_widget.fit_channel_requested, timeout=1000) as blocker,
        ):
            qtbot.mouseClick(
                grid, Qt.MouseButton.RightButton, pos=_center(grid.cell_rect(NODE, BOARD, 9))
            )
        assert blocker.args == [_anode(NODE, BOARD, 9)]
        assert map_widget.current_channel == _anode(NODE, BOARD, 9)

    def test_right_click_on_no_data_tile_selects_board_only(
        self, qtbot: QtBot, map_widget: SystemMapWidget
    ) -> None:
        map_widget.set_current_channel(_anode(NODE, BOARD, 1))
        grid = map_widget.panel_grids[1]
        calls: list[_MenuExecCall] = []
        with (
            _patched_menu_exec(map_widget, calls, choose=0),
            qtbot.waitSignal(map_widget.fit_board_requested, timeout=1000) as blocker,
        ):
            qtbot.mouseClick(grid, Qt.MouseButton.RightButton, pos=_center(grid.tile_rect(7, 21)))
        assert calls[0].action_texts == ["Fit Board  Node 7 Board 21"]
        assert calls[0].selected_board == (7, 21) and calls[0].selected_channel is None
        assert blocker.args == [7, 21]

    def test_right_click_on_strip_selects_cell(
        self, qtbot: QtBot, map_widget: SystemMapWidget
    ) -> None:
        map_widget.set_current_channel(_anode(NODE, BOARD, 1))
        strip = map_widget.board_strip
        calls: list[_MenuExecCall] = []
        with (
            _patched_menu_exec(map_widget, calls),
            qtbot.assertNotEmitted(map_widget.channel_selected),
        ):
            qtbot.mouseClick(
                strip, Qt.MouseButton.RightButton, pos=_center(strip.cell_rect("cathode", 6))
            )
        assert calls[0].selected_channel == _cathode(NODE, BOARD, 6)
        assert map_widget.current_channel == _anode(NODE, BOARD, 1)
        assert strip.selected_channel == _anode(NODE, BOARD, 1)


class TestUpdateViews:
    def test_updates_keep_the_selection(self, map_widget: SystemMapWidget) -> None:
        selected = _anode(NODE, BOARD, 8)
        map_widget.set_current_channel(selected)
        untouched = _model(map_widget).boards[PANEL2].anodes[0]
        hovered = untouched.tooltip
        changed = {_anode(NODE, BOARD, 9): ChannelView(STATUS_FIT_FAILED)}
        map_widget.update_views(changed)
        model = _model(map_widget)
        assert model.boards[(NODE, BOARD)].anodes[8].category == CATEGORY_FAILED
        assert map_widget.current_channel == selected
        strip_board = map_widget.board_strip.board
        assert strip_board is model.boards[(NODE, BOARD)]
        assert model.boards[PANEL2].anodes[0].tooltip is hovered  # not rebuilt
        rebuilt = build_system_map({**_sample_views(), **changed}, ACTIVE_BOARDS)
        assert [c.category for b in model.boards.values() for c in b.anodes] == [
            c.category for b in rebuilt.boards.values() for c in b.anodes
        ]
        assert map_widget.summary_label.text() == format_summary(model)

    def test_without_model_and_empty(self, qtbot: QtBot) -> None:
        widget = SystemMapWidget()
        qtbot.addWidget(widget)
        widget.update_views({_anode(NODE, BOARD, 1): _ok(1.0)})
        model = _model(widget)
        assert model.boards[(NODE, BOARD)].has_data
        widget.update_views({})
        assert widget.model is model


class TestUnplacedWarning:
    def test_lists_results_on_non_electrode_channels(self, map_widget: SystemMapWidget) -> None:
        views = _sample_views()
        views[(NODE, BOARD, 0, 1)] = _ok(1.0)
        map_widget.set_state(views, active_boards=ACTIVE_BOARDS)
        assert not map_widget.warning_label.isHidden()
        assert map_widget.warning_label.text() == ("1 result on unmapped channels: (3, 17, 0, 1)")
        many = {(1, 15, 0, ch): _ok(1.0) for ch in range(4)} | {
            (1, 15, 1, ch): _ok(1.0) for ch in range(3)
        }
        views[(2, 14, 0, 4)] = _ok(1.0)
        map_widget.set_state({**views, **many}, active_boards=ACTIVE_BOARDS)
        text = map_widget.warning_label.text()
        assert text.startswith("1 board(s) with data outside the mapped grid")
        assert "; 8 results on unmapped channels: (1, 15, 0, 0)," in text
        assert text.endswith("... (3 more)")


class TestColorBarLazyLut:
    def test_lut_only_with_a_metric(self, qtbot: QtBot) -> None:
        bar = ColorBarWidget()
        qtbot.addWidget(bar)
        bar.resize(300, 20)
        assert bar.gradient.lut is None
        assert not bar.grab().isNull()  # paints the frame only
        bar.set_metric(METRIC_SPECS["phase_ks"], (0.0, 0.1))
        assert bar.gradient.lut is not None and len(bar.gradient.lut) == 256
        assert "2nd-98th percentile" in bar.toolTip()
        bar.set_metric(None, None)
        assert bar.toolTip() == ""


class TestSparseLabels:
    def test_covered_cells(self, qtbot: QtBot) -> None:
        strip = BoardStripWidget()
        qtbot.addWidget(strip)
        strip.resize(400, 60)  # too narrow for every label
        board = _layout().boards[(NODE, BOARD)]
        strip.set_board(board)
        rect = strip.cell_rect("anode", 20)
        cell_w = rect.width()
        wide = rect.adjusted(-cell_w, 0, cell_w, 0)
        covered = strip._covered_anodes(board, wide, Qt.AlignmentFlag.AlignCenter, 2.5 * cell_w)
        assert [c.position for c in covered] == [19, 20, 21]
        covered = strip._covered_anodes(board, wide, Qt.AlignmentFlag.AlignCenter, 0.5 * cell_w)
        assert [c.position for c in covered] == [20]
        first = strip.cell_rect("anode", 1)
        left = QRectF(first.left(), first.top(), 3 * cell_w, first.height())
        covered = strip._covered_anodes(
            board, left, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, 1.5 * cell_w
        )
        assert [c.position for c in covered] == [1, 2]
        assert not strip.grab().isNull()
