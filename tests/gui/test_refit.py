"""Re-fits, overrides and reverts through the main window (plan 9 control band, D7, 6.1).

Every test drives the real worker threads on the synthetic ring cache (see
``window_helpers.py``); the dialogs are recorded, and the Fit All
keep/discard question is answered by patching ``QMessageBox.exec``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QMessageBox
from pytestqt.qtbot import QtBot

from tests.conftest import RingFiles
from tests.gui.ring_cache import CLEAN, ECCENTRIC, OUTLIERS, store_batch_results
from tests.gui.window_helpers import WAIT_MS, MakeWindow, open_and_wait, wait_idle
from uvcorr import cache as cache_module
from uvcorr.analysis import OPTIONS_BATCH, OPTIONS_OVERRIDE, ChannelKey
from uvcorr.cache import UVCache
from uvcorr.gui import session as session_module
from uvcorr.gui._system_map_model import ChannelAddress
from uvcorr.gui.controls import REFIT_NEEDS_BATCH_TIP
from uvcorr.gui.window import MainWindow
from uvcorr.options import FitOptions

pytestmark = pytest.mark.gui

ON_16 = CLEAN._replace(board=16)


def is_override_on_map(window: MainWindow, key: ChannelKey) -> bool:
    model = window.system_map.model
    assert model is not None
    cell = model.cell(key)
    assert cell is not None
    return cell.is_override


def stored_overrides(cache: Path) -> list[ChannelKey]:
    stored = UVCache(cache).load_results()
    assert stored is not None
    return list(stored.overrides)


def refit(qtbot: QtBot, window: MainWindow, *, board: bool = False) -> None:
    """Click Fit Channel (or Fit Board) and wait for the re-fit and the scatter reload."""
    band = window.control_band
    button = band.fit_board_button if board else band.fit_channel_button
    assert button.isEnabled(), button.toolTip()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert window.refit_thread is not None or not window.is_busy()
    wait_idle(qtbot, window)


def test_refit_needs_a_batch(make_window: MakeWindow, qtbot: QtBot, ring_files: RingFiles) -> None:
    window, dialogs = make_window()
    open_and_wait(qtbot, window, ring_files.cache)
    band = window.control_band
    assert window.session.selection is not None
    for button in (band.fit_channel_button, band.fit_board_button):
        assert not button.isEnabled() and button.toolTip() == REFIT_NEEDS_BATCH_TIP
    menu = window.system_map.build_context_menu(1, 15, ChannelAddress(*CLEAN))
    fits = [action for action in menu.actions() if action.text().startswith("Fit ")]
    assert len(fits) == 2
    assert all(not a.isEnabled() and a.toolTip() == REFIT_NEEDS_BATCH_TIP for a in fits)
    # The API refuses too, with a message
    assert not window.refit_channel(CLEAN)
    assert dialogs.errors and "run Fit All first" in dialogs.errors[-1][1]
    # After Fit All the buttons work
    assert window.start_fit_all(confirm=False)
    wait_idle(qtbot, window)
    assert band.fit_channel_button.isEnabled() and band.fit_board_button.isEnabled()
    assert "override" in band.fit_channel_button.toolTip()


def test_refit_channel_saves_an_override(
    make_window: MakeWindow, qtbot: QtBot, results_files: RingFiles
) -> None:
    window, dialogs = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    window.select_channel(OUTLIERS)
    wait_idle(qtbot, window)
    band = window.control_band
    batch = window.session.batch_options
    assert batch is not None and band.options() == batch  # the band opens on the batch options
    assert window.scatter.detail is not None and window.scatter.detail.n_rejected > 0
    assert not window.revert_channel_action.isEnabled() and not band.revert_button.isEnabled()

    band.robust_check.setChecked(False)
    refit(qtbot, window)
    assert dialogs.errors == []
    assert window.status_message().startswith("Override saved for N1 B15 R0 Ch12 (robust off)")
    assert stored_overrides(results_files.cache) == [OUTLIERS]
    assert is_override_on_map(window, OUTLIERS) and not is_override_on_map(window, CLEAN)
    assert band.label_text().endswith("· override (robust off)")
    assert window.inspector.value_text("options_used") == "override (robust off)"
    assert window.inspector.value_text("option:robust") == "off"
    # The scatter was reloaded with the override's options: no point is rejected
    detail = window.scatter.detail
    assert detail is not None and detail.key == OUTLIERS and detail.consistent
    assert detail.options == FitOptions(robust=False) and detail.n_rejected == 0
    assert window.scatter.shown_rejected == 0 and "rejected" not in window.scatter.info_text()
    assert window.last_refit is not None and window.last_refit.saved == (OUTLIERS,)
    assert "1 override" in window.file_label.text()
    assert window.revert_channel_action.isEnabled() and band.revert_button.isEnabled()
    assert band.use_options_button.isEnabled()

    # The same re-fit through the map's context menu replaces the override (selecting it)
    window.select_channel(CLEAN)
    wait_idle(qtbot, window)
    menu = window.system_map.build_context_menu(1, 15, ChannelAddress(*OUTLIERS))
    labels = [action.text() for action in menu.actions()]
    assert any(text.startswith("Revert Channel to Batch") for text in labels)
    next(a for a in menu.actions() if a.text().startswith("Fit Channel")).trigger()
    wait_idle(qtbot, window)
    assert window.session.selection == OUTLIERS
    assert window.status_message().startswith("Override saved for N1 B15 R0 Ch12")


def test_refit_with_the_batch_options_reverts(
    make_window: MakeWindow, qtbot: QtBot, results_files: RingFiles
) -> None:
    window, _ = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    window.select_channel(OUTLIERS)
    wait_idle(qtbot, window)
    band = window.control_band
    band.robust_check.setChecked(False)
    refit(qtbot, window)
    assert is_override_on_map(window, OUTLIERS)

    band.robust_check.setChecked(True)  # the band shows the batch options again
    refit(qtbot, window)
    assert window.status_message().startswith("N1 B15 R0 Ch12 reverted to batch (override removed)")
    assert stored_overrides(results_files.cache) == []
    assert not is_override_on_map(window, OUTLIERS)
    result = window.session.result(OUTLIERS)
    assert result is not None and result.options_source == OPTIONS_BATCH
    detail = window.scatter.detail
    assert detail is not None and detail.n_rejected == result.n_rejected > 0
    assert window.scatter.shown_rejected == result.n_rejected  # drawn grey again
    assert "override" not in band.label_text()
    # Without an override: nothing to remove, nothing stored
    refit(qtbot, window)
    assert "it matches the batch, nothing stored" in window.status_message()


def test_use_this_channels_options(
    make_window: MakeWindow, qtbot: QtBot, results_files: RingFiles
) -> None:
    window, _ = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    window.select_channel(OUTLIERS)
    wait_idle(qtbot, window)
    band = window.control_band
    band.robust_check.setChecked(False)
    band.min_events_spin.setValue(50)
    refit(qtbot, window)
    override = window.session.override_options(OUTLIERS)
    assert override == FitOptions(robust=False, min_events=50)
    batch = window.session.batch_options
    assert batch is not None
    band.set_options(batch)
    window.select_channel(CLEAN)
    assert not band.use_options_button.isEnabled()
    window.select_channel(OUTLIERS)
    wait_idle(qtbot, window)
    assert "robust off, min events 50" in band.use_options_button.toolTip()
    QTest.mouseClick(band.use_options_button, Qt.MouseButton.LeftButton)
    assert band.options() == override == window.session.options
    assert not band.robust_check.isChecked() and band.min_events_spin.value() == 50
    assert "loaded (robust off, min events 50)" in window.status_message()


def test_board_refit_and_stop(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, dialogs = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    window.select_channel(OUTLIERS)
    wait_idle(qtbot, window)
    band = window.control_band
    band.clip_spin.setValue(3.0)
    progress: list[tuple[int, int]] = []
    assert window.refit_board()
    assert window.refit_thread is not None
    window.refit_thread.progress.connect(lambda done, total: progress.append((done, total)))
    assert window.stop_action.isEnabled() and not window.fit_all_action.isEnabled()
    wait_idle(qtbot, window)
    assert dialogs.errors == []
    assert window.status_message().startswith(
        "Overrides saved for the 6 channels of N1 B15 (clip k 3)"
    )
    on_board = window.session.channels_on_board(1, 15)
    assert stored_overrides(results_files.cache) == on_board
    assert all(is_override_on_map(window, key) for key in on_board)
    assert not is_override_on_map(window, ON_16)
    assert window.revert_board_action.text() == "Revert N1 B15 to Batch (6 overrides)"
    assert not window.progress_bar.isVisible() and not window.stop_action.isEnabled()

    # Stop in the middle of a board re-fit: nothing is stored
    real_analyze = session_module.analyze_channel
    started, release = threading.Event(), threading.Event()
    calls: list[ChannelKey] = []

    def gated(key, u, v, options):  # type: ignore[no-untyped-def]
        calls.append(key)
        if len(calls) == 2:
            started.set()
            assert release.wait(10)
        return real_analyze(key, u, v, options)

    monkeypatch.setattr(session_module, "analyze_channel", gated)
    window.select_channel(ON_16)
    wait_idle(qtbot, window)
    assert window.refit_board()
    qtbot.waitUntil(started.is_set, timeout=WAIT_MS)
    assert window.stop_action.isEnabled()
    assert not window.refit_channel()  # one operation at a time
    assert "Cannot re-fit now" in window.status_message()
    window.stop_current()
    release.set()
    wait_idle(qtbot, window)
    assert len(calls) == 2  # stopped before the third channel
    assert window.status_message() == "Re-fit of N1 B16 stopped; nothing was stored"
    assert stored_overrides(results_files.cache) == on_board
    assert window.session.override_keys(1, 16) == []
    assert dialogs.errors == []


def test_revert_channel_board_and_clear_all(
    make_window: MakeWindow, qtbot: QtBot, results_files: RingFiles
) -> None:
    window, dialogs = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    band = window.control_band
    band.robust_check.setChecked(False)
    window.select_channel(OUTLIERS)
    refit(qtbot, window, board=True)
    window.select_channel(ON_16)
    refit(qtbot, window)
    assert len(window.session.override_keys()) == 7
    assert window.clear_overrides_action.text() == "Clear All &Overrides (7)..."

    # Channel: the band's menu button holds the window's revert actions
    window.select_channel(OUTLIERS)
    wait_idle(qtbot, window)
    menu = band.revert_button.menu()
    assert menu is not None
    assert menu.actions() == [window.revert_channel_action, window.revert_board_action]
    assert window.revert_channel_action.text() == "Revert N1 B15 R0 Ch12 to Batch"
    window.revert_channel_action.trigger()
    assert window.revert_thread is not None  # the write runs in a worker thread
    wait_idle(qtbot, window)
    assert window.status_message() == "N1 B15 R0 Ch12 reverted to batch (override removed)"
    assert not window.revert_channel_action.isEnabled()
    assert window.revert_channel_action.text() == "Revert Channel to Batch"
    assert window.revert_board_action.text() == "Revert N1 B15 to Batch (5 overrides)"
    detail = window.scatter.detail
    assert detail is not None and detail.n_rejected > 0  # the batch fit's mask again

    # Board
    window.revert_board_action.trigger()
    wait_idle(qtbot, window)
    assert window.status_message() == "N1 B15 reverted to batch (5 overrides removed)"
    assert window.session.override_keys() == [ON_16]
    assert not band.revert_button.isEnabled()

    # The map's context menu offers the revert for the channel and board with overrides
    menu = window.system_map.build_context_menu(1, 16, ChannelAddress(*ON_16))
    reverts = [a for a in menu.actions() if a.text().startswith("Revert")]
    assert [a.text().split("  ")[0] for a in reverts] == [
        "Revert Channel to Batch",
        "Revert Board to Batch",
    ]
    assert "(1 override)" in reverts[1].text()
    assert not any(
        a.text().startswith("Revert")
        for a in window.system_map.build_context_menu(1, 15, ChannelAddress(*CLEAN)).actions()
    )

    # Clear all: asks first
    dialogs.answer = False
    assert not window.clear_overrides()
    assert "Delete all 1 channel override" in dialogs.questions[-1]
    assert window.session.override_keys() == [ON_16]
    dialogs.answer = True
    window.clear_overrides_action.trigger()
    wait_idle(qtbot, window)
    assert window.session.override_keys() == [] == stored_overrides(results_files.cache)
    assert window.status_message().startswith("All overrides cleared (1 override removed")
    assert not is_override_on_map(window, ON_16)
    assert not window.clear_overrides_action.isEnabled()
    assert dialogs.errors == []


def test_map_context_revert_actions(
    make_window: MakeWindow, qtbot: QtBot, results_files: RingFiles
) -> None:
    window, _ = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    window.control_band.robust_check.setChecked(False)
    assert window.refit_board(1, 15)
    wait_idle(qtbot, window)
    menu = window.system_map.build_context_menu(1, 15, ChannelAddress(*OUTLIERS))
    next(a for a in menu.actions() if a.text().startswith("Revert Channel")).trigger()
    wait_idle(qtbot, window)
    assert window.session.selection == OUTLIERS  # the menu's channel is shown
    assert OUTLIERS not in window.session.override_keys()
    menu = window.system_map.build_context_menu(1, 15, None)
    next(a for a in menu.actions() if a.text().startswith("Revert Board")).trigger()
    wait_idle(qtbot, window)
    assert window.session.override_keys() == []


def _answer_keep_question(monkeypatch: pytest.MonkeyPatch, button: str) -> list[str]:
    """Make ``QMessageBox.exec`` click ``button`` (by its text); returns the questions shown."""
    shown: list[str] = []

    def fake_exec(box: QMessageBox) -> int:
        shown.append(f"{box.text()}\n{box.informativeText()}")
        choice = next(b for b in box.buttons() if b.text().replace("&", "") == button)
        choice.click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    return shown


@pytest.mark.parametrize("button", ["Cancel", "Keep", "Discard"])
def test_fit_all_asks_to_keep_the_overrides(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
    monkeypatch: pytest.MonkeyPatch,
    button: str,
) -> None:
    window, dialogs = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    window.select_channel(OUTLIERS)
    window.control_band.robust_check.setChecked(False)
    refit(qtbot, window)
    before = UVCache(results_files.cache).load_results()
    assert before is not None and len(before.overrides) == 1

    shown = _answer_keep_question(monkeypatch, button)
    window.control_band.robust_check.setChecked(True)
    window.control_band.clip_spin.setValue(3.5)
    started = window.start_fit_all(confirm=True)
    assert len(shown) == 1  # one question: Keep / Discard / Cancel instead of the yes/no one
    question, details = shown[0].split("\n", 1)
    assert question.startswith("Fit all 18 channels on 3 boards (robust k=3.5, 5 iter")
    assert question.endswith("and store the results in the cache, keeping the 1 channel override?")
    assert question.count("?") == 1
    assert details.startswith("This replaces the stored batch results.")
    assert "Discard: every channel takes the new batch fit" in details
    assert "dropped" not in details  # the override (robust off) fits differently
    assert dialogs.questions == []
    if button == "Cancel":
        assert not started and not window.is_busy()
        after = UVCache(results_files.cache).load_results()
        assert after is not None and after.created_at == before.created_at
        return
    assert started
    wait_idle(qtbot, window)
    stored = UVCache(results_files.cache).load_results()
    assert stored is not None and stored.options == FitOptions(clip_k=3.5)
    if button == "Keep":
        assert list(stored.overrides) == [OUTLIERS]
        assert window.status_message().endswith("; 1 override kept")
        assert is_override_on_map(window, OUTLIERS)
        result = window.session.result(OUTLIERS)
        assert result is not None and result.options_source == OPTIONS_OVERRIDE
    else:
        assert stored.overrides == {}
        assert window.status_message().endswith("; 1 override discarded")
        assert not is_override_on_map(window, OUTLIERS)
    assert window.control_band.options() == FitOptions(clip_k=3.5)


def test_fit_all_and_refits_exclude_each_other(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_analyze_all = session_module.analyze_all
    started = threading.Event()

    def gated(*args: object, **kwargs: object) -> object:
        stop_flag = kwargs["stop_flag"]
        assert isinstance(stop_flag, threading.Event)
        started.set()
        assert stop_flag.wait(10)
        return real_analyze_all(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(session_module, "analyze_all", gated)
    window, dialogs = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    assert window.start_fit_all(confirm=False)
    qtbot.waitUntil(started.is_set, timeout=WAIT_MS)
    band = window.control_band
    assert not band.fit_channel_button.isEnabled()
    assert band.fit_channel_button.toolTip() == "Wait for the running operation to finish"
    assert not window.refit_channel() and not window.refit_board()
    assert "Cannot re-fit now" in window.status_message()
    assert not window.export_both_action.isEnabled()
    window.stop_current()
    wait_idle(qtbot, window)
    assert dialogs.errors == [] and band.fit_channel_button.isEnabled()


def test_overrides_persist_across_close_and_reopen(
    make_window: MakeWindow, qtbot: QtBot, results_files: RingFiles
) -> None:
    window, _ = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    window.select_channel(OUTLIERS)
    window.control_band.robust_check.setChecked(False)
    refit(qtbot, window)
    window.close()

    reopened, dialogs = make_window()
    open_and_wait(qtbot, reopened, results_files.cache)
    assert dialogs.errors == []
    assert reopened.session.override_keys() == [OUTLIERS]
    assert is_override_on_map(reopened, OUTLIERS)
    assert "1 override" in reopened.file_label.text()
    batch = reopened.session.batch_options
    assert batch is not None and reopened.control_band.options() == batch
    reopened.select_channel(OUTLIERS)
    wait_idle(qtbot, reopened)
    assert reopened.control_band.label_text().endswith("· override (robust off)")
    assert reopened.inspector.value_text("options_used") == "override (robust off)"
    detail = reopened.scatter.detail
    assert detail is not None and detail.n_rejected == 0
    # The band still holds the batch options, so an unchanged Fit Channel reverts
    refit(qtbot, reopened)
    assert "reverted to batch" in reopened.status_message()
    assert reopened.session.override_keys() == []


def test_busy_cache_errors_are_reported(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
    hold_h5_open: Callable[[Path], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module, "LOCK_RETRY_SECONDS", 0.1)
    window, dialogs = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    window.select_channel(OUTLIERS)
    wait_idle(qtbot, window)  # the board is in the session's memory now
    window.control_band.robust_check.setChecked(False)
    with hold_h5_open(results_files.cache):  # type: ignore[attr-defined]
        refit(qtbot, window)
        assert dialogs.errors and dialogs.errors[-1][0] == "Re-fit failed"
        assert "in use" in dialogs.errors[-1][1]
        assert "failed; nothing was stored" in window.status_message()
        assert window.session.override_keys() == []
    refit(qtbot, window)
    assert window.session.override_keys() == [OUTLIERS]
    with hold_h5_open(results_files.cache):  # type: ignore[attr-defined]
        assert window.revert_channel()  # started in a worker thread, which waits for the file
        wait_idle(qtbot, window)
        assert dialogs.errors[-1][0] == "Revert to batch" and "in use" in dialogs.errors[-1][1]
        assert window.status_message() == "Revert failed; nothing was deleted"
    assert window.session.override_keys() == [OUTLIERS]


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def test_reverts_run_in_a_worker_and_block_the_option_buttons(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, dialogs = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    window.select_channel(OUTLIERS)
    band = window.control_band
    band.robust_check.setChecked(False)
    refit(qtbot, window)
    assert band.use_options_button.isEnabled() and band.batch_options_button.isEnabled()

    real_delete = UVCache.delete_overrides
    started, release = threading.Event(), threading.Event()

    def gated(self: UVCache, keys, **kwargs):  # type: ignore[no-untyped-def]
        started.set()
        assert release.wait(10)
        return real_delete(self, keys, **kwargs)

    monkeypatch.setattr(UVCache, "delete_overrides", gated)
    assert window.revert_channel()
    qtbot.waitUntil(started.is_set, timeout=WAIT_MS)
    # The GUI stays responsive and refuses other writes meanwhile
    assert window.is_busy() and window.status_message() == "Reverting N1 B15 R0 Ch12 to batch…"
    assert not band.fit_channel_button.isEnabled() and not window.fit_all_action.isEnabled()
    assert not band.use_options_button.isEnabled() and not band.batch_options_button.isEnabled()
    assert not window.revert_board_action.isEnabled() and not window.stop_action.isEnabled()
    assert window.fit_all_action.toolTip() == "Wait for the running operation to finish"
    assert band.fit_all_button.toolTip() == "Wait for the running operation to finish"
    assert not window.refit_channel() and not window.revert_board()
    release.set()
    wait_idle(qtbot, window)
    assert window.status_message() == "N1 B15 R0 Ch12 reverted to batch (override removed)"
    assert window.session.override_keys() == [] and dialogs.errors == []
    assert band.batch_options_button.isEnabled() and not band.use_options_button.isEnabled()


def test_batch_options_button(
    make_window: MakeWindow, qtbot: QtBot, results_files: RingFiles
) -> None:
    window, _ = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    band = window.control_band
    batch = window.session.batch_options
    assert batch is not None
    window.select_channel(OUTLIERS)
    band.robust_check.setChecked(False)
    refit(qtbot, window)
    band.clip_spin.setValue(2.5)
    band.robust_check.setChecked(True)
    QTest.mouseClick(band.use_options_button, Qt.MouseButton.LeftButton)
    assert band.options() == FitOptions(robust=False)
    # Back to the batch options: Fit Channel now reverts the override
    QTest.mouseClick(band.batch_options_button, Qt.MouseButton.LeftButton)
    assert band.options() == batch == window.session.options
    assert window.status_message().startswith("Batch options loaded (robust k=4")
    refit(qtbot, window)
    assert "reverted to batch" in window.status_message()


def test_batch_options_without_a_batch_are_the_defaults(
    make_window: MakeWindow, qtbot: QtBot, ring_files: RingFiles
) -> None:
    window, _ = make_window()
    open_and_wait(qtbot, window, ring_files.cache)
    band = window.control_band
    band.clip_spin.setValue(2.5)
    band.batch_options_button.click()
    assert band.options() == FitOptions()
    assert window.status_message() == "Default options loaded (no batch results yet)"


def test_disabled_actions_say_why(
    make_window: MakeWindow, qtbot: QtBot, ring_files: RingFiles
) -> None:
    window, _ = make_window()
    band = window.control_band
    assert not window.fit_all_action.isEnabled()
    assert window.fit_all_action.toolTip() == "Open a raw .dat file or a UV cache first"
    assert band.fit_all_button.toolTip() == "Open a raw .dat file or a UV cache first"
    open_and_wait(qtbot, window, ring_files.cache)
    assert window.fit_all_action.isEnabled()
    assert window.fit_all_action.toolTip().startswith("Fit every channel")
    for action in (window.export_tec_action, window.export_csv_action, window.export_both_action):
        assert not action.isEnabled()
        assert action.toolTip() == "Nothing to export yet: run Fit All first"
    assert window.start_fit_all(confirm=False)
    wait_idle(qtbot, window)
    assert window.export_both_action.toolTip().startswith("Write <name>.tec")


def test_start_messages_of_refits(
    make_window: MakeWindow, qtbot: QtBot, results_files: RingFiles
) -> None:
    window, _ = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    window.select_channel(OUTLIERS)
    band = window.control_band
    band.robust_check.setChecked(False)
    assert window.refit_board()
    assert window.status_message() == "Fit Board N1 B15 (robust off)…"
    wait_idle(qtbot, window)
    band.robust_check.setChecked(True)
    assert window.refit_board()
    assert window.status_message() == (
        "Fit Board N1 B15 with the batch options (reverts 6 overrides)…"
    )
    wait_idle(qtbot, window)
    assert window.refit_channel()
    assert window.status_message() == (
        "Fit Channel N1 B15 R0 Ch12 with the batch options (no override to revert)…"
    )
    wait_idle(qtbot, window)


def test_fit_all_keep_drops_overrides_with_the_new_options(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review s4: an override fitted with exactly the new batch options is dropped."""
    window, _ = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    band = window.control_band
    window.select_channel(OUTLIERS)
    band.robust_check.setChecked(False)
    refit(qtbot, window)
    batch = window.session.batch_options
    assert batch is not None
    band.set_options(batch)
    window.select_channel(CLEAN)
    band.clip_spin.setValue(3.5)
    refit(qtbot, window)
    assert window.session.override_keys() == [CLEAN, OUTLIERS]

    shown = _answer_keep_question(monkeypatch, "Keep")
    assert window.start_fit_all(confirm=True)  # the band still has clip k 3.5
    assert "1 of the 2 overrides uses exactly these options and will be dropped" in shown[0]
    wait_idle(qtbot, window)
    assert window.status_message().endswith(
        "; 1 override kept; 1 override with exactly these options dropped"
    )
    assert window.session.override_keys() == [OUTLIERS]
    assert stored_overrides(results_files.cache) == [OUTLIERS]
    assert not is_override_on_map(window, CLEAN) and is_override_on_map(window, OUTLIERS)


def test_writes_refuse_results_replaced_on_disk(
    make_window: MakeWindow, qtbot: QtBot, results_files: RingFiles, tmp_path: Path
) -> None:
    window, dialogs = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    window.select_channel(OUTLIERS)
    window.control_band.robust_check.setChecked(False)
    refit(qtbot, window)
    # Another process (e.g. uvcorr process) stores a new batch meanwhile
    store_batch_results(results_files.cache, FitOptions(clip_k=3.0))
    on_disk = UVCache(results_files.cache).load_results()
    window.select_channel(CLEAN)
    refit(qtbot, window)
    assert dialogs.errors[-1][0] == "Re-fit failed"
    assert "stored results changed on disk (another process?); reopen the file" in (
        dialogs.errors[-1][1]
    )
    assert window.revert_channel(OUTLIERS)
    wait_idle(qtbot, window)
    assert dialogs.errors[-1][0] == "Revert to batch" and "changed on disk" in dialogs.errors[-1][1]
    assert UVCache(results_files.cache).load_results() == on_disk  # untouched
    # An export still writes what the window shows, with a warning
    assert window.export_csv(tmp_path / "x.csv") is not None
    title, text = dialogs.warnings[-1]
    assert title == "Export CSV" and "changed on disk" in text


def test_board_click_skips_the_views_of_a_pending_load(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, _ = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    real_compute = session_module.UVSession.compute_detail
    started, release = threading.Event(), threading.Event()

    def gated(self, request):  # type: ignore[no-untyped-def]
        started.set()
        assert release.wait(10)
        return real_compute(self, request)

    monkeypatch.setattr(session_module.UVSession, "compute_detail", gated)
    window.select_channel(ECCENTRIC)
    qtbot.waitUntil(started.is_set, timeout=WAIT_MS)
    thread = window._detail_thread
    assert thread is not None and not thread._skip_views.is_set()
    window.system_map.board_selected.emit(4, 29)
    assert thread._skip_views.is_set()
    release.set()
    wait_idle(qtbot, window)
    assert window.radial.data is None and window.scatter.detail is None
