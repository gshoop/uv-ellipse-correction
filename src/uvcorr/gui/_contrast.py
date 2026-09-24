"""Readable text colours for status labels on the widget palette.

The System Map's category colours (:data:`uvcorr.gui.map_colors.CATEGORY_COLORS`)
are fills chosen for the map's dark background. Used as *text* on a light
widget palette they are too faint (the ok green has a contrast ratio of about
2:1 on white). :func:`readable_color` keeps the hue but mixes the colour
toward black (light background) or white (dark background) until it reaches
the WCAG AA contrast of 4.5:1, so the control band and the Fit Inspector read
well on light and dark themes alike.
"""

from __future__ import annotations

from collections.abc import Iterable

from PyQt6.QtGui import QColor

__all__ = ["MIN_TEXT_CONTRAST", "contrast_ratio", "readable_color", "relative_luminance"]

MIN_TEXT_CONTRAST = 4.5
"""WCAG AA contrast ratio for normal text."""

_MIX_STEPS = 20


def relative_luminance(color: QColor) -> float:
    """WCAG relative luminance of an sRGB colour (0 black .. 1 white)."""

    def channel(value: float) -> float:
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(color.redF())
        + 0.7152 * channel(color.greenF())
        + 0.0722 * channel(color.blueF())
    )


def contrast_ratio(a: QColor, b: QColor) -> float:
    """WCAG contrast ratio of two colours (1 .. 21)."""
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _mix(color: QColor, target: QColor, t: float) -> QColor:
    return QColor.fromRgbF(
        color.redF() + (target.redF() - color.redF()) * t,
        color.greenF() + (target.greenF() - color.greenF()) * t,
        color.blueF() + (target.blueF() - color.blueF()) * t,
    )


def readable_color(
    color: QColor | str,
    backgrounds: QColor | Iterable[QColor],
    minimum: float = MIN_TEXT_CONTRAST,
) -> QColor:
    """``color`` darkened or lightened just enough to read on every background.

    Args:
        color: The wanted text colour.
        backgrounds: The background colour(s) the text is drawn on (e.g. a
            tree's base and alternate-row colours).
        minimum: Contrast ratio to reach.

    Returns:
        The first mix of ``color`` toward black (light backgrounds) or white
        (dark ones), in 5 % steps, that reaches ``minimum`` on all
        backgrounds; black or white at worst.
    """
    base = QColor(color)
    grounds = [backgrounds] if isinstance(backgrounds, QColor) else list(backgrounds)
    mean_luminance = sum(relative_luminance(b) for b in grounds) / max(len(grounds), 1)
    target = QColor("#000000") if mean_luminance > 0.18 else QColor("#ffffff")
    for step in range(_MIX_STEPS + 1):
        candidate = _mix(base, target, step / _MIX_STEPS)
        if all(contrast_ratio(candidate, ground) >= minimum for ground in grounds):
            return candidate
    return target
