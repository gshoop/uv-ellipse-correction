"""GUI test package configuration: deterministic destruction of Qt objects.

Copied from adc2kev 2.2.7 ``tests/test_gui/conftest.py`` and adapted to uvcorr.

Widget trees that hold pyqtgraph plots (and main windows) end their test
inside a reference cycle, so they are freed by Python's cyclic garbage
collector rather than by reference counting. The collector runs at
allocation-driven moments, usually in the middle of a later test, and clears
a cycle in arbitrary order: a Python-owned pyqtgraph item can be deleted
while its C++ parent is still alive with a ``LayoutRequest`` queued, and Qt
then walks a dangling child (adc2kev saw a segfault in
``QGraphicsWidget::setGeometry`` under ``QGraphicsGridLayout::setGeometry``,
delivered from pytest-qt's ``processEvents()`` after a test). Whether the
collector lands on such a moment depends on the exact allocation count, so
adding or removing any test could flip the whole package between green and a
core dump.

The fixtures below take the collector out of the test bodies and run it at
one safe point per test: after pytest-qt has closed the test's widgets,
their pending events have been delivered and their deferred deletions have
been flushed, so Qt destroys the C++ trees top-down before any Python
wrapper goes away.

The package also redirects ``QSettings`` (Ini format, user and system scope)
to a per-test temporary directory, so a window that persists its layout can
never read or overwrite the developer's real ``~/.config`` files, and tests
never see each other's settings.

``QT_QPA_PLATFORM=offscreen`` is set by the top-level ``tests/conftest.py``.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from pathlib import Path

import pytest
from PyQt6.QtCore import QCoreApplication, QEvent, QSettings


@pytest.fixture(scope="package", autouse=True)
def _no_automatic_gc() -> Iterator[None]:
    """Switch off allocation-driven garbage collection for the GUI package.

    Package scope (``tests/gui`` is a package) so the collector is re-enabled
    after the last GUI test: a session fixture would keep it off for every
    package that runs afterwards in a full ``pytest tests/`` run.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()


@pytest.fixture(autouse=True)
def _collect_qt_garbage_at_teardown() -> Iterator[None]:
    """Destroy the test's Qt objects once it is over, C++ side first.

    pytest-qt closes and ``deleteLater()``-s the ``qtbot`` widgets before
    the fixture finalisers run. Those deletions are posted outside an event
    loop, which a plain ``processEvents()`` never delivers, so they are
    flushed explicitly here; the young-generation collection then frees the
    Python cycle whose C++ objects are already gone. With automatic
    collection off, everything the test created is still in generation 0,
    which keeps this cheap.
    """
    yield
    app = QCoreApplication.instance()
    if app is not None:
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
    gc.collect(0)


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path: Path) -> None:
    """Point ``QSettings`` at a per-test directory.

    ``setPath`` is process-global and only affects ``QSettings`` objects
    created afterwards, which is why this runs before every test rather than
    once per session: each test starts with an empty store under its own
    ``tmp_path`` and never touches the real user configuration. The system
    scope is redirected as well so a machine-wide file cannot leak in either.
    """
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.SystemScope, str(tmp_path / "system")
    )
