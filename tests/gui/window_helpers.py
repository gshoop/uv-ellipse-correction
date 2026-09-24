"""Helpers for the main-window tests (the fixtures using them live in ``conftest.py``).

Every window test drives the real worker threads (cache build, cache open,
Fit All with ``workers=1``, re-fits, scatter loading) on the synthetic ring
files and waits for them with ``qtbot.waitUntil``. Dialogs are replaced by a
:class:`Dialogs` recorder (``window._show_error`` / ``_show_info`` / ``_ask``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QPoint, QRectF, Qt
from PyQt6.QtTest import QTest
from pytestqt.qtbot import QtBot

from uvcorr.analysis import ChannelKey
from uvcorr.gui.system_map import VIEW_ANODES, VIEW_CATHODES
from uvcorr.gui.window import MainWindow

WAIT_MS = 20_000


class Dialogs:
    """Records the window's error, warning and info dialogs and answers its yes/no questions."""

    def __init__(self) -> None:
        self.errors: list[tuple[str, str]] = []
        self.infos: list[tuple[str, str]] = []
        self.warnings: list[tuple[str, str]] = []
        self.questions: list[str] = []
        self.answer = True

    def error(self, title: str, text: str) -> None:
        self.errors.append((title, text))

    def info(self, title: str, text: str) -> None:
        self.infos.append((title, text))

    def warning(self, title: str, text: str) -> None:
        self.warnings.append((title, text))

    def ask(self, title: str, text: str) -> bool:
        self.questions.append(f"{title}: {text}")
        return self.answer

    def install(self, window: MainWindow) -> None:
        """Replace the window's dialog methods by this recorder."""
        window._show_error = self.error  # type: ignore[method-assign]
        window._show_info = self.info  # type: ignore[method-assign]
        window._show_warning = self.warning  # type: ignore[method-assign]
        window._ask = self.ask  # type: ignore[method-assign]


MakeWindow = Callable[..., tuple[MainWindow, Dialogs]]


def wait_idle(qtbot: QtBot, window: MainWindow) -> None:
    """Wait until no open/build, Fit All, re-fit, channel load or Board grid load is running."""
    qtbot.waitUntil(
        lambda: not window.is_busy() and not window.detail_pending and not window.grid_pending,
        timeout=WAIT_MS,
    )


def open_and_wait(qtbot: QtBot, window: MainWindow, path: Path) -> None:
    """Open a raw file or cache and wait until everything is loaded."""
    assert window.open_path(path, confirm=False)
    wait_idle(qtbot, window)
    assert window.session.is_open


def aim(rect: QRectF, hits: Callable[[QPoint], bool]) -> QPoint:
    """A pixel of ``rect`` (a cell a few pixels wide) that hit-tests to that cell."""
    y = round(rect.center().y())
    for x in range(math.floor(rect.left()), math.ceil(rect.right()) + 1):
        point = QPoint(x, y)
        if hits(point):
            return point
    raise AssertionError(f"no pixel of {rect} hits the cell")


def click_map_cell(
    window: MainWindow, key: ChannelKey, button: Qt.MouseButton = Qt.MouseButton.LeftButton
) -> None:
    """Click ``key``'s cell in its panel grid (switching the map to the channel's view)."""
    smap = window.system_map
    model = smap.model
    assert model is not None
    located = model.locate(key)
    assert located is not None, f"{key} is not on the map"
    _, kind, position = located
    smap.set_view(VIEW_ANODES if kind == "anode" else VIEW_CATHODES)
    grid = smap.panel_grids[0 if key.node <= 5 else 1]
    point = aim(
        grid.cell_rect(key.node, key.board, position),
        lambda pt: grid.cell_at(pt) == (key.node, key.board, position),
    )
    QTest.mouseClick(grid, button, pos=point)
