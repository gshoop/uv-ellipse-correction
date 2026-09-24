"""Smoke tests of the uvcorr main window (``uvcorr.gui.window``) and ``uvcorr-gui``.

Every test drives the real worker threads (cache build, cache open, Fit All
with ``workers=1``, scatter loading) on the synthetic ring files and waits for
them with ``qtbot.waitUntil``. Dialogs are replaced by recorders
(``window._show_error`` / ``window._ask``); see ``window_helpers.py``.
Re-fits, overrides and exports are tested in ``test_refit.py`` and
``test_export.py``.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import h5py
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from tests.conftest import RingFiles
from tests.gui.ring_cache import CLEAN, OUTLIERS, TOO_FEW
from tests.gui.window_helpers import (
    WAIT_MS,
    MakeWindow,
    aim,
    click_map_cell,
    open_and_wait,
    wait_idle,
)
from uvcorr.analysis import ChannelKey
from uvcorr.cache import BuildSettings, UVCache
from uvcorr.gui import main as main_module
from uvcorr.gui import session as session_module
from uvcorr.gui._layout import app_settings
from uvcorr.gui._system_map_model import ChannelAddress
from uvcorr.gui.map_colors import COLOR_MODES
from uvcorr.gui.session import channel_title
from uvcorr.gui.system_map import VIEW_CATHODES
from uvcorr.gui.window import TAB_TITLES, MainWindow
from uvcorr.options import FLAG_HIGH_REJECTION, STATUS_OK

pytestmark = pytest.mark.gui


def test_open_cache_with_results_and_select_from_map(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
) -> None:
    window, dialogs = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    assert dialogs.errors == []
    assert window.last_open_seconds is not None
    assert "stored results loaded" in window.status_message()
    # The map shows the stored results; the first channel is selected
    model = window.system_map.model
    assert model is not None
    cell = model.cell(OUTLIERS)
    assert cell is not None and cell.view is not None and cell.view.status == STATUS_OK
    first = window.session.data_channels[0]
    assert window.session.selection == first and window.scatter.detail is not None
    assert window.scatter.detail.key == first
    assert "3 boards" in window.file_label.text()
    assert window.fit_all_action.isEnabled() and not window.stop_action.isEnabled()

    # A map click updates the band, the inspector and the scatter
    window.system_map.channel_selected.emit(ChannelAddress(*OUTLIERS))
    wait_idle(qtbot, window)
    assert window.session.selection == OUTLIERS
    detail = window.scatter.detail
    result = window.session.result(OUTLIERS)
    assert detail is not None and detail.key == OUTLIERS and result is not None
    assert detail.consistent and detail.n_rejected == result.n_rejected > 0
    assert f"all {result.n_rejected} rejected shown" in window.scatter.info_text()
    assert window.inspector.value_text("status") == "ok"
    assert FLAG_HIGH_REJECTION in window.inspector.flags_shown()
    assert "N1 B15 R0 Ch12" in window.control_band.label_text()
    assert window.last_switch_seconds is not None

    # A channel without an ellipse: raw points and a message
    window.system_map.channel_selected.emit(ChannelAddress(*TOO_FEW))
    wait_idle(qtbot, window)
    assert "Too few events" in window.scatter.message()
    assert window.inspector.value_text("status") == "too_few_events"


def test_band_navigation_drives_the_map_silently(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
) -> None:
    window, _ = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    channels = window.session.data_channels
    with qtbot.assertNotEmitted(window.system_map.channel_selected):
        window.control_band.next_button.click()
        wait_idle(qtbot, window)
    assert window.session.selection == channels[1]
    assert window.system_map.current_channel == channels[1]
    # Board selector: same RENA/channel on the next board
    band = window.control_band
    band.board_combo.activated.emit(band.board_combo.findData(16))
    wait_idle(qtbot, window)
    assert window.session.selection == ChannelKey(1, 16, *channels[1][2:])
    assert window.system_map.current_board == (1, 16)
    assert window.scatter.detail is not None
    assert window.scatter.detail.key == window.session.selection


def test_rapid_selection_shows_the_last_channel(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
) -> None:
    window, _ = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    for key in window.session.data_channels[:8]:
        window.select_channel(key)
    wait_idle(qtbot, window)
    assert window.scatter.detail is not None
    assert window.scatter.detail.key == window.session.data_channels[7]


def test_map_board_without_data(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
) -> None:
    window, _ = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    window.system_map.board_selected.emit(2, 20)
    assert window.session.selection is None
    assert "no events" in window.scatter.message()
    assert "Node 2 Board 20" in window.inspector.title_label.text()
    assert "no events" in window.radial.message() and "no events" in window.angle.message()
    assert "no events" in window.board_grid.message() and window.board_grid.board is None
    # Nothing to re-fit on a board without events; the results can be exported
    band = window.control_band
    assert not band.fit_channel_button.isEnabled() and not band.fit_board_button.isEnabled()
    assert window.export_tec_action.isEnabled() and window.export_csv_action.isEnabled()
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert titles == list(TAB_TITLES)
    assert all(window.tabs.isTabEnabled(i) for i in range(window.tabs.count()))


def test_fit_all_stores_results_and_updates_the_map(
    make_window: MakeWindow,
    qtbot: QtBot,
    ring_files: RingFiles,
) -> None:
    window, dialogs = make_window()
    open_and_wait(qtbot, window, ring_files.cache)
    assert not window.session.has_results
    model = window.system_map.model
    assert model is not None and model.cell(CLEAN) is not None
    assert model.cell(CLEAN).view is None  # not fitted yet
    assert "Run Fit All" in window.scatter.message()

    window.control_band.clip_spin.setValue(3.5)
    assert window.start_fit_all(confirm=True)
    assert "Fit all 18 channels" in dialogs.questions[-1]
    assert window.stop_action.isEnabled() and not window.open_raw_action.isEnabled()
    wait_idle(qtbot, window)
    assert dialogs.errors == []
    assert window.session.has_results and window.session.batch_options is not None
    assert window.session.batch_options.clip_k == 3.5
    stored = UVCache(ring_files.cache).load_results()
    assert stored is not None and len(stored.results) == 18
    model = window.system_map.model
    assert model is not None and model.cell(CLEAN).view is not None
    assert "results stored" in window.status_message()
    # The selected channel now shows its fit
    assert window.scatter.detail is not None and window.scatter.detail.params is not None
    assert window.inspector.value_text("status") == "ok"


def test_open_raw_builds_the_cache(
    make_window: MakeWindow,
    qtbot: QtBot,
    ring_files: RingFiles,
) -> None:
    ring_files.cache.unlink()
    window, dialogs = make_window()
    progress: list[float] = []
    assert window.open_raw(ring_files.dat)
    assert window.open_thread is not None
    window.open_thread.progress.connect(progress.append)
    wait_idle(qtbot, window)
    assert dialogs.errors == [] and dialogs.questions == []
    assert ring_files.cache.is_file() and window.session.is_open
    assert "built the UV cache" in window.status_message()
    assert len(window.session.data_channels) == 18
    assert window.windowTitle().startswith("rings.dat")


def test_open_raw_stop(
    make_window: MakeWindow,
    qtbot: QtBot,
    ring_files: RingFiles,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop in the middle of a build: after the first batches were parsed and written."""
    real_open_or_build = session_module.open_or_build
    fractions: list[float] = []

    def gated(*args: object, progress_cb: Callable[[float], None], **kwargs: object) -> object:
        stop_flag = kwargs["stop_flag"]
        assert isinstance(stop_flag, threading.Event)

        def blocking_progress(fraction: float) -> None:
            progress_cb(fraction)
            fractions.append(fraction)
            if 0.0 < fraction < 1.0:
                assert stop_flag.wait(10)  # hold the build here until Stop is pressed

        return real_open_or_build(*args, progress_cb=blocking_progress, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(session_module, "open_or_build", gated)
    ring_files.cache.unlink()
    window, dialogs = make_window(build_settings=BuildSettings(batch_events=64))
    assert window.open_raw(ring_files.dat)
    assert "Building the UV cache" in window.status_message()
    assert window.stop_action.isEnabled()
    qtbot.waitUntil(lambda: bool(fractions), timeout=WAIT_MS)  # mid-build, tmp file written
    assert UVCache(ring_files.cache).tmp_files() != []
    window.stop_current()
    wait_idle(qtbot, window)
    assert dialogs.errors == []
    assert 0.0 < fractions[0] < 1.0 and 1.0 not in fractions
    assert "stopped" in window.status_message()
    assert not window.session.is_open and not ring_files.cache.exists()
    assert UVCache(ring_files.cache).tmp_files() == []
    assert window.open_raw_action.isEnabled() and not window.stop_action.isEnabled()


def _assert_in_sync(window: MainWindow, key: ChannelKey) -> None:
    """Band, map, strip, scatter and inspector all show ``key``."""
    band, smap = window.control_band, window.system_map
    assert window.session.selection == key
    assert band.current == key
    assert band.node_combo.currentData() == key.node
    assert band.board_combo.currentData() == key.board
    assert band.channel_combo.currentData() == key.rena * 256 + key.channel
    assert f"N{key.node} B{key.board} R{key.rena} Ch{key.channel:02d}" in band.label_text()
    assert smap.current_board == (key.node, key.board)
    assert smap.current_channel == key
    strip = smap.board_strip
    assert strip.selected_channel == key
    assert strip.header_text.startswith(f"Node {key.node} Board {key.board} ")
    detail = window.scatter.detail
    assert detail is not None and detail.key == key
    assert window.scatter.info_text().startswith(channel_title(key))
    assert window.inspector.title_label.text() == channel_title(key)


def test_mouse_clicks_on_grid_and_strip_keep_everything_in_sync(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
) -> None:
    window, _ = make_window()
    window.resize(1400, 1000)
    open_and_wait(qtbot, window, results_files.cache)
    smap = window.system_map
    model = smap.model
    assert model is not None

    # A grid cell (panel 1 holds nodes 1-5)
    click_map_cell(window, OUTLIERS)
    wait_idle(qtbot, window)
    _assert_in_sync(window, OUTLIERS)

    # A strip cell of the same board
    target = ChannelKey(1, 15, 1, 25)
    located = model.locate(target)
    assert located is not None
    _, kind, position = located
    strip = smap.board_strip
    point = aim(
        strip.cell_rect(kind, position),
        lambda pt: (cell := strip.cell_at(pt)) is not None and cell.channel == target,
    )
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, pos=point)
    wait_idle(qtbot, window)
    _assert_in_sync(window, target)

    # Keyboard stepping in the strip (Right) and the band's Next agree too
    QTest.keyClick(strip, Qt.Key.Key_Right)
    wait_idle(qtbot, window)
    stepped = window.session.selection
    assert stepped is not None and stepped != target
    _assert_in_sync(window, stepped)
    QTest.mouseClick(window.control_band.next_button, Qt.MouseButton.LeftButton)
    wait_idle(qtbot, window)
    _assert_in_sync(window, window.session.step_channel(stepped, +1))


def test_prev_next_after_a_board_only_selection(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
) -> None:
    window, _ = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    session, band, smap = window.session, window.control_band, window.system_map
    on_16 = session.channels_on_board(1, 16)
    # A board with data selected without a channel: Next is its first channel, Prev its last
    smap.board_selected.emit(1, 16)
    assert session.selection is None and "Node 1 Board 16" in band.label_text()
    band.next_button.click()
    wait_idle(qtbot, window)
    _assert_in_sync(window, on_16[0])
    smap.board_selected.emit(2, 20)  # a no-data tile
    band.prev_button.click()
    wait_idle(qtbot, window)
    _assert_in_sync(window, on_16[-1])  # the last channel before board (2, 20)
    smap.board_selected.emit(2, 20)
    band.next_button.click()
    wait_idle(qtbot, window)
    _assert_in_sync(window, session.channels_on_board(4, 29)[0])


def test_channel_without_events_says_no_data(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
) -> None:
    window, _ = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    empty = ChannelKey(1, 15, 0, 4)
    assert window.session.channel_count(empty) == 0
    window.system_map.channel_selected.emit(ChannelAddress(*empty))
    wait_idle(qtbot, window)
    assert "0 events · no data" in window.control_band.label_text()
    assert window.scatter.message() == "No events on this channel (no data)."
    assert window.inspector.value_text("status") == "no data"


def test_unreadable_results_open_with_a_warning(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
) -> None:
    with h5py.File(results_files.cache, "r+") as h5f:
        h5f["results/current"].attrs["options_json"] = "{not json"
    window, dialogs = make_window()
    warnings: list[tuple[str, str]] = []
    window._show_warning = lambda title, text: warnings.append((title, text))  # type: ignore[method-assign]
    open_and_wait(qtbot, window, results_files.cache)
    assert dialogs.errors == [] and len(warnings) == 1
    assert "Fit All stores new results" in warnings[0][1]
    assert "stored results are unreadable" in window.status_message()
    model = window.system_map.model
    assert model is not None and model.cell(CLEAN) is not None
    assert model.cell(CLEAN).view is None  # browsable, not fitted
    assert window.scatter.detail is not None and window.scatter.detail.n_events > 0
    assert window.start_fit_all(confirm=False)
    wait_idle(qtbot, window)
    assert window.session.has_results and dialogs.errors == []


def test_open_dialog_filters_match_is_cache_file() -> None:
    from uvcorr.gui import window as window_module
    from uvcorr.gui.session import is_cache_file

    def patterns(name_filter: str) -> set[str]:
        return set(name_filter[name_filter.index("(") + 1 : name_filter.index(")")].split())

    cache_patterns = patterns(window_module._CACHE_FILTER.split(";;")[0])
    assert cache_patterns == {"*.uv.h5", "*.h5", "*.hdf5"}
    assert all(is_cache_file("run" + pattern[1:]) for pattern in cache_patterns)
    any_filters = window_module._ANY_FILTER.split(";;")
    assert patterns(any_filters[0]) == cache_patterns | {"*.dat"}
    assert patterns(any_filters[2]) == cache_patterns


def test_open_raw_routes_caches_and_refuses_non_raw_files(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
    tmp_path: Path,
) -> None:
    window, dialogs = make_window()
    # A cache given to Open Raw is opened as a cache (no "<cache>.uv.h5" is built)
    assert window.open_raw(results_files.cache)
    wait_idle(qtbot, window)
    assert window.session.cache is not None
    assert window.session.cache.path == results_files.cache
    assert not Path(str(results_files.cache) + ".uv.h5").exists()
    first_dir = window.last_directory()
    # A file without frames: an error, no cache left, the open session is kept
    junk_dir = tmp_path / "junk"
    junk_dir.mkdir()
    junk = junk_dir / "junk.dat"
    junk.write_bytes(bytes(range(256)) * 64)
    assert window.open_raw(junk)
    wait_idle(qtbot, window)
    assert dialogs.errors and "is this a raw .dat file" in dialogs.errors[-1][1]
    assert not (junk_dir / "junk.dat.uv.h5").exists()
    assert window.session.cache.path == results_files.cache
    assert window.last_directory() == first_dir  # only a successful open is remembered


def test_stop_is_disabled_while_opening_a_cache(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
) -> None:
    window, _ = make_window()
    assert window.open_cache(results_files.cache)
    assert window.open_thread is not None and not window.stop_action.isEnabled()
    wait_idle(qtbot, window)


def test_rebuild_asks_before_discarding_results(
    make_window: MakeWindow,
    results_files: RingFiles,
) -> None:
    stat = results_files.dat.stat()
    os.utime(results_files.dat, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    window, dialogs = make_window()
    dialogs.answer = False
    assert not window.open_raw(results_files.dat)
    assert "discarded" in dialogs.questions[-1]
    assert window.open_thread is None


def test_open_errors_are_reported(make_window: MakeWindow, tmp_path: Path) -> None:
    window, dialogs = make_window()
    assert not window.open_path(tmp_path / "missing.dat")
    assert not window.open_path(tmp_path / "missing.uv.h5")
    assert len(dialogs.errors) == 2
    assert not window.start_fit_all(confirm=False)  # nothing open
    assert "Open a raw" in dialogs.errors[-1][1]


def test_open_error_from_the_thread(make_window: MakeWindow, qtbot: QtBot, tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.uv.h5"
    bogus.write_bytes(b"not hdf5")
    window, dialogs = make_window()
    assert window.open_path(bogus)
    wait_idle(qtbot, window)
    assert dialogs.errors and dialogs.errors[-1][0] == "Open failed"
    assert not window.session.is_open


def test_reset_layout_and_settings_round_trip(make_window: MakeWindow, qtbot: QtBot) -> None:
    window, _ = make_window()
    mode = COLOR_MODES[2]
    window.system_map.color_mode_combo.setCurrentIndex(2)  # the user path (emits)
    window.system_map.cathode_radio.setChecked(True)
    window.scatter.cap_spin.setValue(20_000)
    window.scatter.density_check.setChecked(True)
    window.inspector_dock.hide()
    window.close()

    settings = app_settings()
    assert settings.value("map/color_mode") == mode
    assert settings.contains("layout/state")

    restored, _ = make_window()
    assert restored.system_map.color_mode == mode
    assert restored.system_map.view == VIEW_CATHODES
    assert restored.scatter.point_cap == 20_000 and restored.scatter.density
    assert restored.inspector_dock.isHidden()
    restored.reset_layout()
    assert not restored.inspector_dock.isHidden() and not restored.system_map_dock.isHidden()
    assert not app_settings().contains("layout/state")


def test_close_stops_a_running_fit_all(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close while Fit All runs: it is stopped and the stored results are untouched."""
    real_analyze_all = session_module.analyze_all
    started = threading.Event()

    def gated(*args: object, **kwargs: object) -> object:
        stop_flag = kwargs["stop_flag"]
        assert isinstance(stop_flag, threading.Event)
        started.set()
        assert stop_flag.wait(10)  # running until the window stops it
        return real_analyze_all(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(session_module, "analyze_all", gated)
    window, dialogs = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    before = UVCache(results_files.cache).load_results()
    assert before is not None
    assert window.start_fit_all(confirm=False)
    qtbot.waitUntil(started.is_set, timeout=WAIT_MS)
    assert window.fit_all_thread is not None and window.fit_all_thread.isRunning()
    window.close()
    assert any("still running" in question for question in dialogs.questions)
    assert window.fit_all_thread is None
    after = UVCache(results_files.cache).load_results()
    assert after is not None and after.created_at == before.created_at  # nothing stored


def test_main_entry_point_opens_the_file(
    qtbot: QtBot, results_files: RingFiles, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: dict[str, int] = {}

    def fake_exec(_app: QApplication) -> int:
        windows = [w for w in QApplication.topLevelWidgets() if isinstance(w, MainWindow)]
        window = windows[-1]
        qtbot.addWidget(window)
        qtbot.waitUntil(lambda: window.session.is_open, timeout=WAIT_MS)
        wait_idle(qtbot, window)
        opened["channels"] = len(window.session.data_channels)
        opened["workers"] = window.workers or 0
        # Ctrl+C in the terminal closes the window through the event loop
        signal.raise_signal(signal.SIGINT)
        qtbot.waitUntil(lambda: not window.isVisible(), timeout=WAIT_MS)
        opened["closed_by_sigint"] = 1
        return 0

    levels: list[int] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: levels.append(kwargs["level"]))
    monkeypatch.setattr(QApplication, "exec", fake_exec)
    previous = signal.getsignal(signal.SIGINT)
    assert main_module.main([str(results_files.cache), "--workers", "2", "-v"]) == 0
    assert opened == {"channels": 18, "workers": 2, "closed_by_sigint": 1}
    assert levels == [logging.INFO]
    assert signal.getsignal(signal.SIGINT) is previous  # restored after the event loop
    assert main_module.MainWindow is MainWindow  # lazy re-export
    with pytest.raises(SystemExit):
        main_module.main(["--workers", "0"])


def test_entry_module_imports_no_qt() -> None:
    """Spawned Fit All workers re-import the console script (``uvcorr.gui.main``)."""
    code = (
        "import sys, uvcorr.gui.main; "
        "print(sorted(m for m in ('PyQt6.QtWidgets', 'pyqtgraph', 'uvcorr.gui.window') "
        "if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert out.stdout.strip() == "[]"
