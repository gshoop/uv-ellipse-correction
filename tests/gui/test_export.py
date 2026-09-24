"""File > Export .tec / Export CSV / Export Both through the main window (plan D3).

The exports write the merged results (batch rows with the overrides
applied). The file dialogs are replaced by recorders that return fixed paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot

from tests.conftest import RingFiles
from tests.gui.ring_cache import OUTLIERS
from tests.gui.window_helpers import MakeWindow, open_and_wait, wait_idle
from uvcorr.analysis import OPTIONS_BATCH, OPTIONS_OVERRIDE
from uvcorr.gui._layout import app_settings
from uvcorr.gui.window import KEY_EXPORT_DIR, MainWindow
from uvcorr.io.summary_csv import read_summary_csv
from uvcorr.io.tec import read_tec
from uvcorr.options import STATUS_OK, FitOptions

pytestmark = pytest.mark.gui

EXPORT_ACTIONS = ("export_tec_action", "export_csv_action", "export_both_action")


class FileDialogs:
    """Replaces the save and directory dialogs; records their titles and start paths."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[str, str]] = []
        self.answer = ""
        monkeypatch.setattr(QFileDialog, "getSaveFileName", self._save)
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", self._directory)

    def _save(self, _parent: object, title: str, start: str, _filter: str) -> tuple[str, str]:
        self.calls.append((title, start))
        return self.answer, ""

    def _directory(self, _parent: object, title: str, start: str) -> str:
        self.calls.append((title, start))
        return self.answer


def add_override(qtbot: QtBot, window: MainWindow) -> None:
    """Re-fit OUTLIERS with robust off (an override)."""
    window.select_channel(OUTLIERS)
    wait_idle(qtbot, window)
    window.control_band.robust_check.setChecked(False)
    assert window.refit_channel()
    wait_idle(qtbot, window)
    window.control_band.robust_check.setChecked(True)


def test_export_is_disabled_without_results(
    make_window: MakeWindow, qtbot: QtBot, ring_files: RingFiles, tmp_path: Path
) -> None:
    window, dialogs = make_window()
    assert not any(getattr(window, name).isEnabled() for name in EXPORT_ACTIONS)
    open_and_wait(qtbot, window, ring_files.cache)
    assert not any(getattr(window, name).isEnabled() for name in EXPORT_ACTIONS)
    # The API says why, and writes nothing
    assert window.export_both(tmp_path / "out") is None
    assert dialogs.errors and "run Fit All first" in dialogs.errors[-1][1]
    assert not (tmp_path / "out").exists()
    assert window.start_fit_all(confirm=False)
    wait_idle(qtbot, window)
    assert all(getattr(window, name).isEnabled() for name in EXPORT_ACTIONS)


def test_export_both_writes_the_merged_results(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, dialogs = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    add_override(qtbot, window)
    files = FileDialogs(monkeypatch)
    out = tmp_path / "exports"
    files.answer = str(out)
    window.export_both_action.trigger()
    # The dialog starts in the cache's directory (nothing exported yet)
    assert files.calls == [
        ("Export rings.tec and radial_summary.csv to", str(results_files.cache.parent))
    ]
    assert dialogs.errors == []
    tec, csv = out / "rings.tec", out / "radial_summary.csv"
    assert tec.is_file() and csv.is_file()
    rows = read_summary_csv(csv)
    by_key = {row.key: row for row in rows}
    assert len(rows) == 18 and by_key[OUTLIERS].options_source == OPTIONS_OVERRIDE
    assert by_key[OUTLIERS].n_rejected == 0
    assert sum(1 for row in rows if row.options_source == OPTIONS_BATCH) == 17
    entries = read_tec(tec)
    n_ok = sum(1 for row in rows if row.status == STATUS_OK)
    assert len(entries) == n_ok and OUTLIERS in entries
    title, text = dialogs.infos[-1]
    assert title == "Export .tec and CSV"
    assert str(tec) in text and str(csv) in text
    assert f"18 rows, {n_ok} ok blocks, 1 override applied" in text
    assert window.status_message().startswith("Exported rings.tec and radial_summary.csv (18 rows")
    summary = window.last_export
    assert summary is not None and summary.paths == (tec, csv) and summary.n_overrides == 1
    assert app_settings().value(KEY_EXPORT_DIR) == str(out.resolve())


def test_export_tec_and_csv_separately(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, dialogs = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    add_override(qtbot, window)
    files = FileDialogs(monkeypatch)
    remembered = tmp_path / "remembered"
    remembered.mkdir()
    app_settings().setValue(KEY_EXPORT_DIR, str(remembered))

    files.answer = str(tmp_path / "chosen" / "ellipses")  # no extension: .tec is added
    (tmp_path / "chosen").mkdir()
    window.export_tec_action.trigger()
    assert files.calls[-1] == ("Export .tec", str(remembered / "rings.tec"))
    tec = tmp_path / "chosen" / "ellipses.tec"
    assert OUTLIERS in read_tec(tec)
    assert dialogs.infos[-1][0] == "Export .tec" and str(tec) in dialogs.infos[-1][1]

    files.answer = str(tmp_path / "chosen" / "summary.csv")
    window.export_csv_action.trigger()
    # The next dialog starts where the last export went
    assert files.calls[-1] == ("Export CSV", str(tmp_path / "chosen" / "radial_summary.csv"))
    rows = read_summary_csv(tmp_path / "chosen" / "summary.csv")
    assert {row.key: row for row in rows}[OUTLIERS].options_source == OPTIONS_OVERRIDE

    files.answer = ""  # cancelled: nothing happens
    n_infos = len(dialogs.infos)
    window.export_csv_action.trigger()
    assert len(dialogs.infos) == n_infos and dialogs.errors == []


def test_export_errors_are_reported(
    make_window: MakeWindow, qtbot: QtBot, results_files: RingFiles, tmp_path: Path
) -> None:
    window, dialogs = make_window()
    open_and_wait(qtbot, window, results_files.cache)
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    assert window.export_both(blocker) is None
    assert dialogs.errors[-1][0] == "Export .tec and CSV failed"
    assert window.export_tec(tmp_path / "missing" / "x.tec") is None
    assert dialogs.errors[-1] == (
        "Export .tec failed",
        f"Cannot write {tmp_path / 'missing' / 'x.tec'}: No such file or directory",
    )  # the target, not the hidden temporary file
    assert window.status_message() == "Export .tec failed"
    assert window.last_export is None
    # The batch options did not change on the way
    assert window.control_band.options() == FitOptions()


def test_export_never_overwrites_the_open_files(
    make_window: MakeWindow,
    qtbot: QtBot,
    results_files: RingFiles,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review s10: the save dialog accepts any name; the export refuses the data files."""
    window, dialogs = make_window()
    open_and_wait(qtbot, window, results_files.dat)  # raw open: the private .dat copy
    dat, cache = results_files.dat, results_files.cache
    assert window.session.dat_path == dat
    before = {
        path: path.read_bytes()[:4096] + str(path.stat().st_size).encode() for path in (dat, cache)
    }
    files = FileDialogs(monkeypatch)
    files.answer = str(dat)
    window.export_tec_action.trigger()
    assert dialogs.errors[-1][0] == "Export .tec failed"
    assert "Refusing to overwrite the open raw data file" in dialogs.errors[-1][1]
    files.answer = str(cache)
    window.export_csv_action.trigger()
    assert "Refusing to overwrite the open UV cache" in dialogs.errors[-1][1]
    other = tmp_path / "another.uv.h5"
    other.write_bytes(cache.read_bytes()[:1_000_000])  # an HDF5 signature is enough
    assert window.export_csv(other) is None
    assert "HDF5 file" in dialogs.errors[-1][1]
    after = {
        path: path.read_bytes()[:4096] + str(path.stat().st_size).encode() for path in (dat, cache)
    }
    assert after == before and dialogs.infos == []
    # The window still works on its files
    window.select_channel(OUTLIERS)
    wait_idle(qtbot, window)
    assert window.scatter.detail is not None and window.scatter.message() == ""
