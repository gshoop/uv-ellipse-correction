"""Window sizing policy and settings helpers for the uvcorr GUI.

Adapted from adc2kev 2.2.7 ``gui/_layout.py`` (a private module, so it is
reimplemented here rather than imported). The main window has no hard-coded
size: it opens at ``SIZE_FRACTION`` of the available screen area, clamped
between the window's layout minimum and ``SIZE_CAP``, and maximized when that
fraction would leave the window cramped (within ``MAXIMIZE_MARGIN`` of its
minimum) or when the screen is short (available height below
``SHORT_SCREEN_HEIGHT``): on a 1280 x 800 laptop 85 % of the height leaves the
scatter panels only ~170 px tall once the System Map takes its share.

Persistence: ``QSettings`` in Ini format under the user scope
(``~/.config/uvcorr/uvcorr-gui.ini`` on Linux). The window geometry and dock
state are tagged with ``LAYOUT_VERSION``; bump it whenever docks are added,
removed or renamed so that a state written for the old dock set is ignored
instead of being restored partially. Tests redirect the user-scope path to a
temporary directory (``tests/gui/conftest.py``).

Values read back from an Ini file are text in a new process (``"true"``,
``"50000"``), so the ``read_*`` helpers accept both the typed value Qt caches
in the writing process and its text form.
"""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, QSettings

__all__ = [
    "FALLBACK_SCREEN",
    "LAYOUT_VERSION",
    "MAXIMIZE_MARGIN",
    "SETTINGS_APP",
    "SETTINGS_ORG",
    "SHORT_SCREEN_HEIGHT",
    "SIZE_CAP",
    "SIZE_FRACTION",
    "app_settings",
    "initial_window_size",
    "read_bool",
    "read_bytes",
    "read_int",
    "read_str",
]

LAYOUT_VERSION = 1  # bump whenever docks are added, removed or renamed
SETTINGS_ORG = "uvcorr"
SETTINGS_APP = "uvcorr-gui"
SIZE_FRACTION = 0.85
SIZE_CAP = (1920, 1200)  # logical pixels
MAXIMIZE_MARGIN = 1.25  # maximize when the fraction is within 25 % of the minimum
FALLBACK_SCREEN = (1600, 900)  # assumed available area when no screen is known
SHORT_SCREEN_HEIGHT = 900  # below this available height the window opens maximized


def app_settings() -> QSettings:
    """The GUI's settings store (Ini format, user scope, no fallbacks)."""
    settings = QSettings(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, SETTINGS_ORG, SETTINGS_APP
    )
    settings.setFallbacksEnabled(False)
    return settings


def initial_window_size(
    available: tuple[int, int], minimum: tuple[int, int]
) -> tuple[tuple[int, int], bool]:
    """Choose the launch window size for a screen.

    Args:
        available: Width and height of the available screen area.
        minimum: The window's layout minimum (``minimumSizeHint``).

    Returns:
        ``((width, height), maximize)``: ``round(available * SIZE_FRACTION)``
        clamped to ``[minimum, SIZE_CAP]`` (the minimum wins over the cap),
        and whether to maximize: the unclamped fraction falls below
        ``minimum * MAXIMIZE_MARGIN`` in either dimension, or the available
        height is below ``SHORT_SCREEN_HEIGHT``.
    """
    wanted_width = round(available[0] * SIZE_FRACTION)
    wanted_height = round(available[1] * SIZE_FRACTION)
    maximize = (
        wanted_width < minimum[0] * MAXIMIZE_MARGIN
        or wanted_height < minimum[1] * MAXIMIZE_MARGIN
        or available[1] < SHORT_SCREEN_HEIGHT
    )
    width = max(minimum[0], min(wanted_width, SIZE_CAP[0]))
    height = max(minimum[1], min(wanted_height, SIZE_CAP[1]))
    return (width, height), maximize


def read_int(settings: QSettings, key: str, default: int | None = None) -> int | None:
    """Integer stored under ``key``, or ``default`` when absent or not an integer."""
    raw = settings.value(key)
    if raw is None or isinstance(raw, bool):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def read_bool(settings: QSettings, key: str, default: bool = False) -> bool:
    """Boolean stored under ``key`` (a bool or the text ``true``/``false``), else ``default``."""
    raw = settings.value(key)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return default


def read_str(settings: QSettings, key: str, default: str = "") -> str:
    """Text stored under ``key``, or ``default`` when absent or not text."""
    raw = settings.value(key)
    return raw if isinstance(raw, str) else default


def read_bytes(settings: QSettings, key: str) -> QByteArray:
    """Byte array stored under ``key``; empty when absent or of another type."""
    raw = settings.value(key)
    if isinstance(raw, QByteArray):
        return raw
    if isinstance(raw, bytes | bytearray):
        return QByteArray(bytes(raw))
    return QByteArray()
