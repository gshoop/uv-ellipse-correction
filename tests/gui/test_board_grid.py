"""Tests for the Board grid tab (``uvcorr.gui.board_grid``)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt
from pytestqt.qtbot import QtBot

from tests.conftest import RingFiles
from tests.gui.ring_cache import (
    CLEAN,
    ECCENTRIC,
    FIT_FAILED,
    OUTLIERS,
    TOO_FEW,
    copy_files,
    store_batch_results,
)
from uvcorr.analysis import ChannelKey
from uvcorr.channels import ACTIVE_CHANNELS, electrode_label, electrode_map, is_cathode
from uvcorr.ellipse import correct
from uvcorr.gui import board_grid as grid_module
from uvcorr.gui.board_grid import (
    AFTER_MARGIN,
    GRID_COLUMNS,
    GRID_ROWS,
    LAYOUT_RENA,
    LAYOUT_STRIP,
    MODE_AFTER,
    MODE_BEFORE,
    BoardGridData,
    BoardGridTab,
    board_keys,
    compute_board_grid,
    grid_slots,
    window_fits,
)
from uvcorr.gui.map_colors import (
    CATEGORY_FAILED,
    CATEGORY_FLAGGED,
    CATEGORY_NO_DATA,
    CATEGORY_NOT_FITTED,
    CATEGORY_OK,
    CATEGORY_TOO_FEW,
)
from uvcorr.gui.scatter import subsample_indices
from uvcorr.gui.session import UVSession, load_cache

pytestmark = pytest.mark.gui

NODE, BOARD = CLEAN.node, CLEAN.board  # (1, 15): an odd board
NO_EVENTS = ChannelKey(NODE, BOARD, 0, 4)


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory, _ring_files_master: RingFiles) -> UVSession:
    """A session on a copy of the ring cache with stored batch results (read only here)."""
    _, cache = copy_files(
        _ring_files_master.dat, _ring_files_master.cache, tmp_path_factory.mktemp("grid")
    )
    store_batch_results(cache)
    s = UVSession()
    s.install(load_cache(cache))
    return s


def _results(session: UVSession, node: int = NODE, board: int = BOARD) -> dict:
    return {key: session.result(key) for key in board_keys(node, board)}


@pytest.fixture(scope="module")
def grid_data(session: UVSession) -> BoardGridData:
    return compute_board_grid(session.board_data(NODE, BOARD), _results(session))


@pytest.fixture
def tab(qtbot: QtBot) -> BoardGridTab:
    widget = BoardGridTab()
    qtbot.addWidget(widget)
    widget.resize(1100, 700)
    return widget


def _items(tab: BoardGridTab, key: ChannelKey):
    return tab._cells[tab.slot_keys().index(key)]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_rena_layout_order() -> None:
    slots = grid_slots(NODE, BOARD, LAYOUT_RENA)
    assert len(slots) == GRID_COLUMNS * GRID_ROWS == 48
    expected = [ChannelKey(NODE, BOARD, rena, ch) for rena in (0,) for ch in range(4, 29)]
    expected += [ChannelKey(NODE, BOARD, 1, ch) for ch in range(7, 29)]
    assert list(slots[:47]) == expected == list(board_keys(NODE, BOARD))
    assert slots[47] is None


@pytest.mark.parametrize("board", [15, 16])
def test_strip_layout_order(board: int) -> None:
    slots = grid_slots(3, board, LAYOUT_STRIP)
    assert len(slots) == 48 and slots[39] is None
    keys = [key for key in slots if key is not None]
    assert sorted(keys) == sorted(board_keys(3, board))
    labels = [electrode_label(board, k.rena, k.channel) for k in keys]
    anodes = [f"A{i:02d}" for i in range(1, 40)]
    # Physical strip position 1 is on the low-node side: odd boards start at A39
    assert labels[:39] == (anodes if board % 2 == 0 else anodes[::-1])
    assert labels[39:] == [f"C{i:02d}" for i in range(1, 9)]
    emap = electrode_map()
    for position, key in enumerate(slots[:39], start=1):
        assert key is not None
        assert emap.physical_strip_position(board, key.rena, key.channel) == position
    assert all(is_cathode(board, k.rena, k.channel) for k in keys[39:])
    assert slots[40:] == tuple(keys[39:])  # the cathodes fill the last row
    with pytest.raises(ValueError, match="layout"):
        grid_slots(3, board, "diagonal")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def test_cells_categories_and_counts(session: UVSession, grid_data: BoardGridData) -> None:
    assert (grid_data.node, grid_data.board) == (NODE, BOARD)
    assert len(grid_data.cells) == len(ACTIVE_CHANNELS) == 47
    expected = {
        CLEAN: CATEGORY_OK,
        OUTLIERS: CATEGORY_FLAGGED,
        ECCENTRIC: CATEGORY_FLAGGED,
        TOO_FEW: CATEGORY_TOO_FEW,
        FIT_FAILED: CATEGORY_FAILED,
        NO_EVENTS: CATEGORY_NO_DATA,
    }
    for key, category in expected.items():
        assert grid_data.cells[key].category == category, key
    for key, cell in grid_data.cells.items():
        assert cell.n_events == session.channel_count(key)
        assert cell.cathode == is_cathode(BOARD, key.rena, key.channel)
        assert cell.electrode == electrode_label(BOARD, key.rena, key.channel)
        assert "Click to open the channel" in cell.tooltip
    assert grid_data.cells[CLEAN].cathode  # R0 Ch05 is a cathode on odd boards
    assert grid_data.n_events == session.board_data(NODE, BOARD).n_events


def test_cells_draw_a_deterministic_subsample(session: UVSession) -> None:
    board_uv = session.board_data(NODE, BOARD)
    data = compute_board_grid(board_uv, _results(session), points_per_cell=300)
    again = compute_board_grid(board_uv, _results(session), points_per_cell=300)
    for key in (CLEAN, OUTLIERS, TOO_FEW):
        u16, v16 = session.channel_data(key)
        index = subsample_indices(u16.shape[0], 300, tuple(key))  # the Scatter tab's seeding
        u = u16 if index is None else u16[index]
        v = v16 if index is None else v16[index]
        cell = data.cells[key]
        np.testing.assert_array_equal(cell.before_x, u.astype(np.float64))
        np.testing.assert_array_equal(cell.before_y, v.astype(np.float64))
        np.testing.assert_array_equal(cell.before_x, again.cells[key].before_x)
        assert cell.n_shown == min(300, u16.shape[0])
    assert data.cells[CLEAN].n_shown == 300 and data.cells[TOO_FEW].n_shown == 40


def test_after_points_and_overlays(grid_data: BoardGridData) -> None:
    ok = grid_data.cells[OUTLIERS]
    params = ok.params
    assert params is not None and ok.after_corrected and ok.result is not None
    u_corr, v_corr = correct(ok.before_x, ok.before_y, params)
    np.testing.assert_allclose(ok.after_x, u_corr)
    np.testing.assert_allclose(ok.after_y, v_corr)
    assert ok.target_radius == pytest.approx(ok.result.target_radius)
    assert ok.ellipse_x is not None and ok.ellipse_y is not None
    raw = grid_data.cells[TOO_FEW]
    assert raw.params is None and not raw.after_corrected and raw.ellipse_x is None
    np.testing.assert_allclose(raw.after_x, raw.before_x - np.median(raw.before_x))
    empty = grid_data.cells[NO_EVENTS]
    assert empty.n_shown == 0 and empty.after_x.size == 0


def test_common_windows(grid_data: BoardGridData) -> None:
    x0, x1, y0, y1 = grid_data.before_window
    assert x1 - x0 == pytest.approx(y1 - y0)  # square
    fitted = [cell for cell in grid_data.cells.values() if cell.params is not None]
    assert len(fitted) == 4
    for cell in fitted:
        assert cell.ellipse_x is not None and cell.ellipse_y is not None
        assert x0 < cell.ellipse_x.min() and cell.ellipse_x.max() < x1
        assert y0 < cell.ellipse_y.min() and cell.ellipse_y.max() < y1
    targets = [cell.target_radius for cell in fitted if cell.target_radius is not None]
    assert grid_data.after_half == pytest.approx(AFTER_MARGIN * max(targets))
    # The eccentric ring (extreme_axis_ratio) does not set the windows, but fits in them
    assert grid_data.window_left_out == (ECCENTRIC,)
    assert grid_data.clipped_before == () and grid_data.clipped_after == ()
    h = grid_data.after_half
    assert grid_data.window(MODE_AFTER) == (-h, h, -h, h)
    assert grid_data.window(MODE_BEFORE) == grid_data.before_window


def test_one_bad_fit_does_not_stretch_the_window(session: UVSession) -> None:
    board_uv = session.board_data(NODE, BOARD)
    results = _results(session)
    clean = results[CLEAN]
    assert clean is not None and clean.centerU is not None
    # A fit whose centre is 1200 ADC away from the other fits of its RENA
    bad = dict(results)
    bad[CLEAN] = replace(clean, centerU=clean.centerU + 1200.0)
    data = compute_board_grid(board_uv, bad)
    without = compute_board_grid(board_uv, {k: r for k, r in results.items() if k != CLEAN})
    assert CLEAN in data.window_left_out and ECCENTRIC in data.window_left_out
    assert data.before_window == pytest.approx(without.before_window)
    assert data.after_half == pytest.approx(without.after_half)
    assert data.clipped_before == (CLEAN,) and data.clipped(MODE_BEFORE) == (CLEAN,)
    assert "Left out of the board's common window" in data.cells[CLEAN].tooltip
    used, left_out = window_fits(data.cells.values())
    assert {cell.key for cell in left_out} == {CLEAN, ECCENTRIC}
    assert {cell.key for cell in used} == {OUTLIERS, ChannelKey(NODE, BOARD, 1, 25)}


def test_a_shifted_rena_is_not_an_outlier(grid_data: BoardGridData) -> None:
    # A whole ASIC can sit at its own U/V offset: fits are compared within their RENA
    template = grid_data.cells[CLEAN]
    result = template.result
    assert result is not None and result.centerU is not None and result.centerV is not None
    rng = np.random.default_rng(2)
    cells = []
    for rena, channels, shift in ((0, range(4, 26), 0.0), (1, range(7, 27), 770.0)):
        for channel in channels:
            key = ChannelKey(NODE, BOARD, rena, channel)
            fit = replace(
                result,
                channel=channel,
                rena=rena,
                centerU=result.centerU + shift + rng.normal(0.0, 8.0),
                centerV=result.centerV + rng.normal(0.0, 8.0),
            )
            cells.append(replace(template, key=key, result=fit))
    outlier = replace(cells[3], result=replace(cells[3].result, centerU=result.centerU + 400.0))
    cells[3] = outlier
    extreme = replace(cells[30], result=replace(cells[30].result, flags=("extreme_axis_ratio",)))
    cells[30] = extreme
    used, left_out = window_fits(cells)
    assert [cell.key for cell in left_out] == sorted([outlier.key, extreme.key])
    assert len(used) == len(cells) - 2
    assert sum(1 for cell in used if cell.key.rena == 1) == 19  # the shifted RENA stays in


def test_info_line_names_the_clipped_fits(session: UVSession, tab: BoardGridTab) -> None:
    results = dict(_results(session))
    clean = results[CLEAN]
    assert clean is not None and clean.centerU is not None
    results[CLEAN] = replace(clean, centerU=clean.centerU + 1200.0)
    tab.show_data(compute_board_grid(session.board_data(NODE, BOARD), results))
    electrode = electrode_label(BOARD, CLEAN.rena, CLEAN.channel)
    assert f"2 fits left out of it (1 clipped: {electrode})" in tab.info_text()
    assert _items(tab, CLEAN).note.textItem.toPlainText() == "outside\nthe window"
    tab.set_mode(MODE_AFTER)
    assert "2 fits left out of it" in tab.info_text() and "clipped" not in tab.info_text()
    assert _items(tab, CLEAN).note.textItem.toPlainText() == ""


def test_board_without_results(session: UVSession, tab: BoardGridTab) -> None:
    data = compute_board_grid(session.board_data(NODE, BOARD), {})
    categories = {cell.category for cell in data.cells.values()}
    assert categories == {CATEGORY_NOT_FITTED, CATEGORY_NO_DATA}
    assert all(cell.params is None for cell in data.cells.values())
    x0, x1, y0, y1 = data.before_window
    clean = data.cells[CLEAN]
    assert x0 < clean.before_x.min() and clean.before_x.max() < x1
    assert math.isfinite(data.after_half) and data.after_half > 500.0
    for mode in (MODE_BEFORE, MODE_AFTER):
        tab.set_mode(mode)
        tab.show_data(data)
        assert _items(tab, CLEAN).note.textItem.toPlainText().startswith("Not fitted")
        assert not _items(tab, CLEAN).curve.isVisible()


def test_switching_boards_reuses_the_items(session: UVSession, tab: BoardGridTab) -> None:
    tab.show_data(compute_board_grid(session.board_data(1, 15), _results(session, 1, 15)))

    def snapshot() -> list[tuple[int, ...]]:
        return [
            (id(items.vb), id(items.scatter), id(items.curve), len(items.vb.addedItems))
            for items in tab._cells
        ]

    before = snapshot()
    even = compute_board_grid(session.board_data(1, 16), _results(session, 1, 16))
    tab.show_data(even)
    assert tab.board == (1, 16) and tab.slot_keys() == grid_slots(1, 16, LAYOUT_RENA)
    assert snapshot() == before  # same cells and items, nothing added
    cathode = ChannelKey(1, 16, 1, 25)  # a cathode on even boards
    assert _items(tab, cathode).cathode and even.cells[cathode].cathode


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


def test_layouts_and_cathode_outlines(
    tab: BoardGridTab, grid_data: BoardGridData, qtbot: QtBot
) -> None:
    tab.show_data(grid_data)
    assert tab.board == (NODE, BOARD) and tab.grid_layout == LAYOUT_RENA
    assert tab.slot_keys() == grid_slots(NODE, BOARD, LAYOUT_RENA)
    with qtbot.waitSignal(tab.display_changed):
        tab.layout_combo.setCurrentIndex(1)
    assert tab.grid_layout == LAYOUT_STRIP
    assert tab.slot_keys() == grid_slots(NODE, BOARD, LAYOUT_STRIP)
    with qtbot.assertNotEmitted(tab.display_changed):
        tab.set_grid_layout(LAYOUT_RENA)
    assert tab.slot_keys() == grid_slots(NODE, BOARD, LAYOUT_RENA)
    cathode_color = grid_module._BORDER_CATHODE.color().name()
    for items in tab._cells:
        if items.key is None:
            continue
        outlined = items.vb.border.color().name() == cathode_color
        assert outlined == is_cathode(BOARD, items.key.rena, items.key.channel), items.key
    assert sum(1 for items in tab._cells if items.cathode) == 8
    # Titles carry the electrode, notes mark the channels without an ellipse
    title = _items(tab, CLEAN).label.text
    assert f"<b>{electrode_label(BOARD, CLEAN.rena, CLEAN.channel)}</b>" in title
    assert "R0·05 · 700" in title
    assert _items(tab, TOO_FEW).note.textItem.toPlainText() == "Too few events"
    assert _items(tab, NO_EVENTS).note.textItem.toPlainText() == "No data"
    assert _items(tab, OUTLIERS).layout.toolTip() == grid_data.cells[OUTLIERS].tooltip
    # Empty slots have no frame
    empty = [items for items in tab._cells if items.key is None]
    assert empty and all(items.vb.border.style() == Qt.PenStyle.NoPen for items in empty)


def test_no_data_and_not_fitted_titles_differ(session: UVSession, tab: BoardGridTab) -> None:
    data = compute_board_grid(session.board_data(NODE, BOARD), {})
    tab.show_data(data)
    not_fitted = _items(tab, CLEAN).label.text
    no_data = _items(tab, NO_EVENTS).label.text
    assert "■" in not_fitted and "■" not in no_data and "no data" in no_data
    assert grid_module._NO_DATA_TEXT not in not_fitted
    note_colors = {
        key: _items(tab, key).note.textItem.defaultTextColor().name() for key in (CLEAN, NO_EVENTS)
    }
    assert note_colors[NO_EVENTS] == grid_module._NO_DATA_TEXT
    assert note_colors[CLEAN] != note_colors[NO_EVENTS]


def _center(tab: BoardGridTab, key: ChannelKey) -> QPoint:
    top_left, bottom_right = tab.cell_view_rect(key)
    return QPoint((top_left.x() + bottom_right.x()) // 2, (top_left.y() + bottom_right.y()) // 2)


@pytest.mark.parametrize("layout", [LAYOUT_RENA, LAYOUT_STRIP])
def test_all_cells_fit_a_small_tab(
    session: UVSession, grid_data: BoardGridData, qtbot: QtBot, layout: str
) -> None:
    # A real tab area at 1280 x 800 is ~934 x 402: all eight columns must fit
    tab = BoardGridTab()
    qtbot.addWidget(tab)
    tab.resize(930, 400)
    tab.set_grid_layout(layout)
    tab.show_data(compute_board_grid(session.board_data(1, 16), _results(session, 1, 16)))
    tab.show()
    qtbot.waitExposed(tab)
    assert tab.minimumSizeHint().width() <= 930
    viewport = tab.graphics.viewport()
    width, height = viewport.width(), viewport.height()
    keys = [key for key in tab.slot_keys() if key is not None]
    assert len(keys) == 47
    for key in keys:
        top_left, bottom_right = tab.cell_view_rect(key)
        assert top_left.x() >= 0 and bottom_right.x() <= width, key
        assert top_left.y() >= 0 and bottom_right.y() <= height, key
        assert tab.cell_key_at(_center(tab, key)) == key
    # Titles are shortened to fit, never wider than their cell
    for items in tab._cells:
        if items.key is not None:
            shown = grid_module._TEXT_WIDTHS.width(items.label.shown)
            assert shown <= items.label.size().width() or items.label.shown == items.label.text
    for key in keys[:: 5 if layout == LAYOUT_RENA else 7]:
        with qtbot.waitSignal(tab.channel_activated, timeout=1000) as blocker:
            qtbot.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=_center(tab, key))
        assert blocker.args == [key]


def test_click_emits_the_channel(tab: BoardGridTab, grid_data: BoardGridData, qtbot: QtBot) -> None:
    tab.show_data(grid_data)
    tab.show()
    qtbot.waitExposed(tab)
    for key in (OUTLIERS, CLEAN, NO_EVENTS):
        centre = _center(tab, key)
        assert tab.cell_key_at(centre) == key
        with qtbot.waitSignal(tab.channel_activated, timeout=1000) as blocker:
            qtbot.mouseClick(tab.graphics.viewport(), Qt.MouseButton.LeftButton, pos=centre)
        assert blocker.args == [key]
    tab.set_grid_layout(LAYOUT_STRIP)
    centre = _center(tab, ECCENTRIC)
    with qtbot.waitSignal(tab.channel_activated, timeout=1000) as blocker:
        qtbot.mouseClick(tab.graphics.viewport(), Qt.MouseButton.LeftButton, pos=centre)
    assert blocker.args == [ECCENTRIC]


def _ranges(tab: BoardGridTab) -> list[tuple[float, float, float, float]]:
    ranges = []
    for items in tab._cells:
        (x0, x1), (y0, y1) = items.vb.viewRange()
        ranges.append((x0, x1, y0, y1))
    return ranges


def test_common_axes_and_mode_toggle(
    tab: BoardGridTab, grid_data: BoardGridData, qtbot: QtBot
) -> None:
    tab.show_data(grid_data)
    tab.show()
    qtbot.waitExposed(tab)
    for mode in (MODE_BEFORE, MODE_AFTER):
        tab.set_mode(mode)
        x0, x1, y0, y1 = grid_data.window(mode)
        for rx0, rx1, ry0, ry1 in _ranges(tab):
            # The same centre in every cell; the locked aspect may widen one axis
            assert 0.5 * (rx0 + rx1) == pytest.approx(0.5 * (x0 + x1), abs=1e-6 * (x1 - x0))
            assert 0.5 * (ry0 + ry1) == pytest.approx(0.5 * (y0 + y1), abs=1e-6 * (y1 - y0))
            assert rx0 <= x0 + 1e-6 and rx1 >= x1 - 1e-6 and ry0 <= y0 + 1e-6
            assert (rx1 - rx0) == pytest.approx(_ranges(tab)[0][1] - _ranges(tab)[0][0], rel=0.05)
    assert "U′, V′ ±" in tab.info_text()
    corrected = _items(tab, OUTLIERS).scatter.getData()
    np.testing.assert_array_equal(corrected[0], grid_data.cells[OUTLIERS].after_x)
    assert "(raw − median)" in _items(tab, TOO_FEW).note.textItem.toPlainText()
    with qtbot.waitSignal(tab.display_changed):
        tab.before_radio.setChecked(True)
    assert tab.mode == MODE_BEFORE and "common window U" in tab.info_text()
    np.testing.assert_array_equal(
        _items(tab, OUTLIERS).scatter.getData()[0], grid_data.cells[OUTLIERS].before_x
    )
    with pytest.raises(ValueError, match="mode"):
        tab.set_mode("sideways")


def test_selection_outline_and_clear(tab: BoardGridTab, grid_data: BoardGridData) -> None:
    tab.show_data(grid_data)
    selected_color = grid_module._BORDER_SELECTED.color().name()
    tab.set_selected(OUTLIERS)
    assert tab.selected == OUTLIERS
    outlined = [
        items.key for items in tab._cells if items.vb.border.color().name() == selected_color
    ]
    assert outlined == [OUTLIERS]
    tab.set_grid_layout(LAYOUT_STRIP)  # the outline follows the channel
    outlined = [
        items.key for items in tab._cells if items.vb.border.color().name() == selected_color
    ]
    assert outlined == [OUTLIERS]
    tab.set_selected(None)
    assert not any(items.vb.border.color().name() == selected_color for items in tab._cells)
    tab.show_loading("N1 B16")
    assert tab.info_text().startswith("Loading")
    tab.clear("Node 2 Board 18: no events on this board")
    assert tab.data is None and tab.board is None
    assert tab.message() == "Node 2 Board 18: no events on this board"
    assert tab.slot_keys() == (None,) * 48
