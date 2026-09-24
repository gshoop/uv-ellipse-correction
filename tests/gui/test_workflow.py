"""The plan 10.5 smoke sequence, end to end through the real main window, and the tab wiring.

10.5: on a small synthetic cache, open it, select a channel through the map,
switch through the tabs, re-fit the channel with robust off, export, and
check the files (parsed back: the override is in the CSV). The wiring tests
cover the Radial, Radius vs angle and Board grid tabs' loading, laziness,
refresh after a re-fit and settings persistence.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from pytestqt.qtbot import QtBot

from tests.conftest import RingFiles
from tests.gui.ring_cache import CLEAN, ECCENTRIC, OUTLIERS
from tests.gui.window_helpers import MakeWindow, click_map_cell, open_and_wait, wait_idle
from uvcorr.analysis import OPTIONS_BATCH, OPTIONS_OVERRIDE
from uvcorr.cache import UVCache
from uvcorr.gui._layout import app_settings
from uvcorr.gui.board_grid import LAYOUT_STRIP, MODE_AFTER
from uvcorr.gui.window import TAB_TITLES, MainWindow
from uvcorr.io.summary_csv import read_summary_csv
from uvcorr.io.tec import read_tec
from uvcorr.options import STATUS_OK

pytestmark = pytest.mark.gui


def show_tab(qtbot: QtBot, window: MainWindow, title: str) -> None:
    index = TAB_TITLES.index(title)
    QTest.mouseClick(
        window.tabs.tabBar(),  # type: ignore[arg-type]
        Qt.MouseButton.LeftButton,
        pos=window.tabs.tabBar().tabRect(index).center(),  # type: ignore[union-attr]
    )
    assert window.tabs.currentIndex() == index
    wait_idle(qtbot, window)


def click_grid_cell(window: MainWindow, key: object) -> None:
    grid = window.board_grid
    top_left, bottom_right = grid.cell_view_rect(key)  # type: ignore[arg-type]
    centre = QPoint((top_left.x() + bottom_right.x()) // 2, (top_left.y() + bottom_right.y()) // 2)
    assert grid.cell_key_at(centre) == key
    QTest.mouseClick(grid.graphics.viewport(), Qt.MouseButton.LeftButton, pos=centre)


def test_plan_10_5_smoke_sequence(
    make_window: MakeWindow, qtbot: QtBot, results_files: RingFiles, tmp_path: Path
) -> None:
    window, dialogs = make_window()
    window.resize(1400, 1000)
    # Open
    open_and_wait(qtbot, window, results_files.cache)
    assert "stored results loaded" in window.status_message()

    # Select a channel through the map (a real mouse click on its cell)
    click_map_cell(window, OUTLIERS)
    wait_idle(qtbot, window)
    assert window.session.selection == OUTLIERS
    detail = window.scatter.detail
    assert detail is not None and detail.key == OUTLIERS and detail.n_rejected > 0

    # Switch through the tabs: each shows the selected channel (or its board)
    for title in TAB_TITLES[1:]:
        show_tab(qtbot, window, title)
    radial, angle = window.radial.data, window.angle.data
    assert radial is not None and radial.key == OUTLIERS and radial.matches_stored
    assert angle is not None and angle.key == OUTLIERS and angle.has_mask
    assert window.board_grid.board == (1, 15) and window.board_grid.selected == OUTLIERS
    # A click on a Board grid cell opens that channel everywhere
    click_grid_cell(window, ECCENTRIC)
    wait_idle(qtbot, window)
    assert window.session.selection == ECCENTRIC and window.board_grid.selected == ECCENTRIC
    assert window.system_map.current_channel == ECCENTRIC
    radial = window.radial.data
    assert radial is not None and radial.key == ECCENTRIC
    click_grid_cell(window, OUTLIERS)
    wait_idle(qtbot, window)
    show_tab(qtbot, window, "Scatter")

    # Re-fit the channel with robust off (the band's controls and button)
    QTest.mouseClick(window.control_band.robust_check, Qt.MouseButton.LeftButton)
    assert not window.control_band.robust_check.isChecked()
    QTest.mouseClick(window.control_band.fit_channel_button, Qt.MouseButton.LeftButton)
    wait_idle(qtbot, window)
    assert dialogs.errors == []
    assert window.status_message().startswith("Override saved for N1 B15 R0 Ch12 (robust off)")
    stored = UVCache(results_files.cache).load_results()
    assert stored is not None and list(stored.overrides) == [OUTLIERS]
    model = window.system_map.model
    assert model is not None
    cell = model.cell(OUTLIERS)
    assert cell is not None and cell.is_override  # the map marks it
    detail = window.scatter.detail
    assert detail is not None and detail.n_rejected == 0
    angle = window.angle.data
    assert angle is not None and angle.key == OUTLIERS

    # Export both files and parse them back
    out = tmp_path / "export"
    summary = window.export_both(out)
    assert summary is not None and dialogs.errors == []
    tec, csv = out / "rings.tec", out / "radial_summary.csv"
    assert tec.is_file() and csv.is_file()
    rows = {row.key: row for row in read_summary_csv(csv)}
    assert len(rows) == 18
    assert rows[OUTLIERS].options_source == OPTIONS_OVERRIDE and rows[OUTLIERS].n_rejected == 0
    assert rows[CLEAN].options_source == OPTIONS_BATCH
    entries = read_tec(tec)
    assert set(entries) == {key for key, row in rows.items() if row.status == STATUS_OK}
    assert entries[OUTLIERS].params.a == pytest.approx(rows[OUTLIERS].semiMajor, rel=1e-5)


def test_channel_tabs_follow_the_selection(
    make_window: MakeWindow, qtbot: QtBot, results_files: RingFiles
) -> None:
    window, _ = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    first = window.session.data_channels[0]
    radial = window.radial.data
    assert radial is not None and radial.key == first  # loaded with the first channel
    assert window.last_views_seconds is not None and window.last_switch_seconds is not None
    assert window.last_views_seconds >= window.last_switch_seconds
    for key in window.session.data_channels[:6]:  # rapid stepping: the last one wins
        window.select_channel(key)
    wait_idle(qtbot, window)
    last = window.session.data_channels[5]
    for data in (window.radial.data, window.angle.data):
        assert data is not None and data.key == last
    # A board without events clears the channel tabs
    window.system_map.board_selected.emit(2, 20)
    wait_idle(qtbot, window)
    assert window.radial.data is None and "no events" in window.radial.message()
    assert window.angle.data is None and "no events" in window.angle.message()


def test_board_grid_loads_lazily_and_refreshes_after_a_refit(
    make_window: MakeWindow, qtbot: QtBot, results_files: RingFiles
) -> None:
    window, _ = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    assert window.board_grid.board is None  # not loaded while its tab is hidden
    window.select_channel(ECCENTRIC)
    wait_idle(qtbot, window)
    assert window.board_grid.board is None and window.board_grid.selected == ECCENTRIC
    window.tabs.setCurrentWidget(window.board_grid)
    wait_idle(qtbot, window)
    data = window.board_grid.data
    assert data is not None and (data.node, data.board) == (1, 15)
    assert window.last_grid_seconds is not None
    # Another board through the map's board click
    window.system_map.board_selected.emit(4, 29)
    wait_idle(qtbot, window)
    assert window.board_grid.board == (4, 29) and window.board_grid.selected is None
    # A re-fit on the shown board recomputes it (the status squares follow the results)
    key = OUTLIERS._replace(node=4, board=29)
    window.select_channel(key)
    wait_idle(qtbot, window)
    before = window.board_grid.data
    window.control_band.robust_check.setChecked(False)
    assert window.refit_channel()
    wait_idle(qtbot, window)
    after = window.board_grid.data
    assert after is not None and after is not before and after.board == 29
    result = after.cells[key].result
    assert result is not None and result.options_source == OPTIONS_OVERRIDE
    # A re-fit on another board leaves the grid alone
    window.select_channel(CLEAN)
    wait_idle(qtbot, window)
    shown = window.board_grid.data
    assert shown is not None and shown.board == 15
    assert window.refit_board(4, 29)
    wait_idle(qtbot, window)
    assert window.board_grid.data is shown


def test_tab_settings_round_trip(make_window: MakeWindow, qtbot: QtBot) -> None:
    window, _ = make_window()
    assert window.radial.common_axis and not window.angle.include_rejected  # defaults
    window.radial.common_axis_check.setChecked(False)  # the user path (emits)
    window.angle.include_rejected_check.setChecked(True)
    window.board_grid.after_radio.setChecked(True)
    window.board_grid.layout_combo.setCurrentIndex(1)
    settings = app_settings()
    assert settings.value("radial/common_axis") in (False, "false")
    assert settings.value("grid/layout") == LAYOUT_STRIP
    window.close()

    restored, _ = make_window()
    assert not restored.radial.common_axis and restored.angle.include_rejected
    assert restored.board_grid.grid_layout == LAYOUT_STRIP
    assert restored.board_grid.mode == MODE_AFTER
