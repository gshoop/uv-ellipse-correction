"""Check that pytest-qt runs headless with PyQt6 (the GUI arrives in phase 4)."""

from __future__ import annotations

import os

import pytest
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QLabel
from pytestqt.qtbot import QtBot


@pytest.mark.gui
def test_qtbot_creates_widget(qtbot: QtBot) -> None:
    label = QLabel("uvcorr")
    qtbot.addWidget(label)
    label.show()
    qtbot.waitExposed(label)
    assert label.text() == "uvcorr"
    assert QGuiApplication.platformName() == os.environ["QT_QPA_PLATFORM"]
