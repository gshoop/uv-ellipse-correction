"""Pytest configuration and shared fixtures for uvcorr tests.

GUI tests (pytest-qt, ``qt_api = "pyqt6"`` in ``pyproject.toml``) run on Qt's
``offscreen`` platform by default so the suite works headless. Export
``QT_QPA_PLATFORM`` (e.g. ``xcb``) before running pytest to watch them on a
real display instead.
"""

from __future__ import annotations

import os

# Must happen before the first QApplication is created (pytest-qt creates it
# lazily in the ``qapp``/``qtbot`` fixtures).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
