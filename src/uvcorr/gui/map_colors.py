"""Colour logic and colour bar for the System Map.

New in uvcorr (adc2kev 2.2.7's map knows only its four calibration colours,
``STATUS_COLORS`` in ``adc2kev/gui/system_map.py``). The map has two kinds of
colour mode (plan section 9, System Map row):

- **Status** (:data:`MODE_STATUS`): every channel falls into one of six
  categories (:data:`CATEGORIES`), derived from its fit status and warning
  flags by :func:`status_category`. The category colours extend adc2kev's
  palette (green ok, orange flagged, red failed, grey not fitted, dark no
  data) with a muted blue for ``too_few_events``. The six colours were checked
  with a palette validator (OKLab ΔE x 100) on the ``#1e1e1e`` map
  background, over all pairs: the colour-vision-deficiency separation passes
  (worst: ok vs flagged, ΔE 8.8 for deuteranopia) and every pair is at least
  ΔE 16 apart for normal vision. The two neutrals are grey on purpose (they
  mean "nothing to show"). The legend labels every colour, so the colour
  never stands alone.
- **Metric** (the keys of :data:`METRIC_SPECS`): a channel's metric value is
  mapped onto the viridis sequential colormap. The limits are the 2nd and
  98th percentiles of the finite values (:func:`metric_limits`), so a few
  outliers do not wash out the rest of the detector. NaN or missing values get
  the not-fitted grey, and channels without events keep the no-data colour.

Everything here except :class:`ColorBarWidget` is pure: it needs no
``QApplication`` (``QColor`` is a value type), so it is unit tested without a
display. :class:`ColorBarWidget` shows the colormap with its limits and the
metric name while a metric mode is active.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from uvcorr.options import STATUS_FIT_FAILED, STATUS_OK, STATUS_TOO_FEW_EVENTS, STATUSES

__all__ = [
    "CATEGORIES",
    "CATEGORY_COLORS",
    "CATEGORY_FAILED",
    "CATEGORY_FLAGGED",
    "CATEGORY_LABELS",
    "CATEGORY_NOT_FITTED",
    "CATEGORY_NO_DATA",
    "CATEGORY_OK",
    "CATEGORY_TOO_FEW",
    "CLIP_PERCENTILES",
    "COLORMAP_NAME",
    "COLOR_MODES",
    "LABEL_MIN_CONTRAST",
    "LUT_SIZE",
    "METRIC_SPECS",
    "MODE_LABELS",
    "MODE_STATUS",
    "NO_DATA_COLOR",
    "NOT_FITTED_COLOR",
    "ColorBarWidget",
    "MetricSpec",
    "category_qcolor",
    "check_color_mode",
    "clip_range_text",
    "is_metric_mode",
    "label_colors_for_fills",
    "legend_entries",
    "lut_indices",
    "metric_color",
    "metric_colors",
    "metric_limits",
    "sequential_lut",
    "status_category",
    "text_color_for",
    "text_color_for_fills",
]

# ---------------------------------------------------------------------------
# Status categories
# ---------------------------------------------------------------------------

CATEGORY_OK = "ok"
"""Status ``ok`` without warning flags (or only informational ones)."""

CATEGORY_FLAGGED = "flagged"
"""Status ``ok`` with at least one warning flag outside the informational set."""

CATEGORY_FAILED = "failed"
"""Status ``fit_failed``."""

CATEGORY_TOO_FEW = "too_few"
"""Status ``too_few_events`` (the channel has events, but fewer than ``min_events``)."""

CATEGORY_NOT_FITTED = "not_fitted"
"""The channel has events but no result yet."""

CATEGORY_NO_DATA = "no_data"
"""The channel (or its whole board) has no events."""

CATEGORIES: tuple[str, ...] = (
    CATEGORY_OK,
    CATEGORY_FLAGGED,
    CATEGORY_FAILED,
    CATEGORY_TOO_FEW,
    CATEGORY_NOT_FITTED,
    CATEGORY_NO_DATA,
)
"""All categories, in legend and summary order."""

CATEGORY_LABELS: dict[str, str] = {
    CATEGORY_OK: "OK",
    CATEGORY_FLAGGED: "Flagged",
    CATEGORY_FAILED: "Fit failed",
    CATEGORY_TOO_FEW: "Too few events",
    CATEGORY_NOT_FITTED: "Not fitted",
    CATEGORY_NO_DATA: "No data",
}

# Dark-theme palette: adc2kev's calibrated/failed/rejected/not-fitted colours,
# its no-data tile colour, and a muted blue for too few events.
CATEGORY_COLORS: dict[str, str] = {
    CATEGORY_OK: "#2ecc71",
    CATEGORY_FLAGGED: "#f39c12",
    CATEGORY_FAILED: "#e74c3c",
    CATEGORY_TOO_FEW: "#5d86b8",
    CATEGORY_NOT_FITTED: "#555555",
    CATEGORY_NO_DATA: "#2b2b2b",
}

NO_DATA_COLOR = CATEGORY_COLORS[CATEGORY_NO_DATA]
"""Fill of boards and channels without events (adc2kev's ``NO_DATA_COLOR``)."""

NOT_FITTED_COLOR = CATEGORY_COLORS[CATEGORY_NOT_FITTED]
"""Fill of channels with events but no result, and of NaN metric values."""

# Parsed once and shared by every cell of a category: the map fills up to
# 7520 cells. Treat these as read-only.
_CATEGORY_QCOLORS: dict[str, QColor] = {
    category: QColor(color) for category, color in CATEGORY_COLORS.items()
}


def status_category(
    status: str | None,
    flags: Iterable[str] = (),
    informational_flags: Collection[str] = frozenset(),
    *,
    has_data: bool = True,
) -> str:
    """Return the status category of one channel.

    Rules, evaluated in order:

    1. ``has_data`` is False -> ``no_data``.
    2. No result (``status`` is None) -> ``not_fitted``.
    3. ``ok`` with any flag not in ``informational_flags`` -> ``flagged``,
       otherwise ``ok``.
    4. ``fit_failed`` -> ``failed``.
    5. ``too_few_events`` -> ``too_few``.

    Args:
        status: The channel's fit status (one of ``uvcorr.options.STATUSES``),
            or None when the channel has no result.
        flags: The channel's warning flags.
        informational_flags: Flags that do not make an ``ok`` channel
            ``flagged`` (they are only shown in the tooltip).
        has_data: Whether the channel has any events.

    Returns:
        One of :data:`CATEGORIES`.

    Raises:
        ValueError: If ``status`` is not None and not a known status.
    """
    if not has_data:
        return CATEGORY_NO_DATA
    if status is None:
        return CATEGORY_NOT_FITTED
    if status == STATUS_OK:
        if any(flag not in informational_flags for flag in flags):
            return CATEGORY_FLAGGED
        return CATEGORY_OK
    if status == STATUS_FIT_FAILED:
        return CATEGORY_FAILED
    if status == STATUS_TOO_FEW_EVENTS:
        return CATEGORY_TOO_FEW
    raise ValueError(f"status must be one of {STATUSES} or None, got {status!r}")


def category_qcolor(category: str) -> QColor:
    """Return the shared fill colour of ``category`` (do not mutate it).

    Raises:
        ValueError: If ``category`` is not one of :data:`CATEGORIES`.
    """
    try:
        return _CATEGORY_QCOLORS[category]
    except KeyError:
        raise ValueError(f"category must be one of {CATEGORIES}, got {category!r}") from None


# ---------------------------------------------------------------------------
# Colour modes
# ---------------------------------------------------------------------------

MODE_STATUS = "status"
"""Colour mode: status categories."""


def _format_general(value: float) -> str:
    return f"{value:.4g}"


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.3g}%"


@dataclass(frozen=True)
class MetricSpec:
    """One metric colour mode.

    Attributes:
        key: Metric key in ``ChannelView.metrics`` (the CSV column name where
            there is one) and the colour mode string.
        label: Short display name for the mode selector, colour bar and
            summary.
        unit: Unit appended to formatted values (empty for none).
        formatter: Formats a finite value without its unit.
    """

    key: str
    label: str
    unit: str = ""
    formatter: Callable[[float], str] = _format_general

    def format(self, value: float) -> str:
        """Format ``value`` with its unit; NaN gives ``"n/a"``."""
        if not math.isfinite(value):
            return "n/a"
        text = self.formatter(value)
        return f"{text} {self.unit}" if self.unit else text


METRIC_SPECS: dict[str, MetricSpec] = {
    spec.key: spec
    for spec in (
        MetricSpec("post_sigma", "Post σ", "ADC"),
        MetricSpec("timing_jitter_ns", "Jitter", "ns"),
        MetricSpec("phase_ks", "KS D"),
        MetricSpec("axis_ratio", "Axis ratio b/a"),
        MetricSpec("rejected_fraction", "Rejected fraction", formatter=_format_percent),
    )
}
"""Metric colour modes by key, in selector order.

``rejected_fraction`` is ``n_rejected / n_events`` (not a CSV column); the
others are the CSV columns of the same name (plan section 6.2).
"""

COLOR_MODES: tuple[str, ...] = (MODE_STATUS, *METRIC_SPECS)
"""All colour modes, in selector order."""

MODE_LABELS: dict[str, str] = {
    MODE_STATUS: "Status",
    **{key: spec.label for key, spec in METRIC_SPECS.items()},
}

CLIP_PERCENTILES: tuple[float, float] = (2.0, 98.0)
"""Percentiles of the finite metric values used as the colormap limits."""


def _ordinal(value: float) -> str:
    """``2`` -> ``"2nd"``, ``98`` -> ``"98th"``, ``11`` -> ``"11th"``, ``2.5`` -> ``"2.5th"``."""
    if value != int(value):
        return f"{value:g}th"
    number = int(value)
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def clip_range_text(percentiles: tuple[float, float] = CLIP_PERCENTILES) -> str:
    """Describe the clipping percentiles, e.g. ``"2nd-98th percentile"``."""
    return f"{_ordinal(percentiles[0])}-{_ordinal(percentiles[1])} percentile"


COLORMAP_NAME = "viridis"
"""Sequential colormap of the metric modes (from ``pyqtgraph.colormap``)."""

LUT_SIZE = 256
"""Number of colours in the metric lookup table."""


def check_color_mode(mode: str) -> None:
    """Raise ``ValueError`` unless ``mode`` is one of :data:`COLOR_MODES`."""
    if mode not in COLOR_MODES:
        raise ValueError(f"color mode must be one of {COLOR_MODES}, got {mode!r}")


def is_metric_mode(mode: str) -> bool:
    """Return True for a metric colour mode, False for the status mode.

    Raises:
        ValueError: If ``mode`` is not one of :data:`COLOR_MODES`.
    """
    check_color_mode(mode)
    return mode != MODE_STATUS


def legend_entries(mode: str) -> tuple[tuple[str, str], ...]:
    """Return the ``(label, hex colour)`` swatches of the legend for ``mode``.

    The status mode lists every category. A metric mode lists only the two
    fills that are not on the colour bar: no value (NaN or missing) and no
    data.

    Raises:
        ValueError: If ``mode`` is not one of :data:`COLOR_MODES`.
    """
    if is_metric_mode(mode):
        return (("No value", NOT_FITTED_COLOR), ("No data", NO_DATA_COLOR))
    return tuple((CATEGORY_LABELS[category], CATEGORY_COLORS[category]) for category in CATEGORIES)


# ---------------------------------------------------------------------------
# Metric colours
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def sequential_lut(name: str = COLORMAP_NAME, size: int = LUT_SIZE) -> tuple[QColor, ...]:
    """Return ``size`` colours of the pyqtgraph colormap ``name``, low to high.

    Built once per ``(name, size)``; the colours are shared, do not mutate
    them.

    Raises:
        ValueError: If ``size`` is below 2.
    """
    if size < 2:
        raise ValueError(f"size must be at least 2, got {size}")
    import pyqtgraph as pg  # deferred: only the metric modes need it

    table = np.asarray(
        pg.colormap.get(name).getLookupTable(0.0, 1.0, size, alpha=False), dtype=np.int64
    )
    return tuple(QColor(int(r), int(g), int(b)) for r, g, b in table[:, :3])


def metric_limits(
    values: npt.ArrayLike,
    percentiles: tuple[float, float] = CLIP_PERCENTILES,
) -> tuple[float, float] | None:
    """Return the colour limits of a set of metric values.

    Args:
        values: Metric values (a sequence or array); NaN and infinite values
            are ignored.
        percentiles: Lower and upper percentile (0-100) used as the limits.

    Returns:
        ``(low, high)`` with ``low <= high`` (equal when all finite values
        are equal), or None when there is no finite value.

    Raises:
        ValueError: If the percentiles are not ``0 <= low <= high <= 100``.
    """
    low_pct, high_pct = percentiles
    if not 0.0 <= low_pct <= high_pct <= 100.0:
        raise ValueError(f"percentiles must satisfy 0 <= low <= high <= 100, got {percentiles}")
    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    low, high = np.percentile(finite, [low_pct, high_pct])
    return float(low), float(high)


def lut_indices(
    values: npt.ArrayLike, limits: tuple[float, float] | None, size: int = LUT_SIZE
) -> npt.NDArray[np.int64]:
    """Map values onto lookup-table indices, clipping at the limits.

    ``low`` maps to 0 and ``high`` to ``size - 1``; values outside the limits
    are clipped to the ends. When ``low == high`` every finite value maps to
    the middle of the table.

    Args:
        values: Metric values (any shape).
        limits: ``(low, high)`` from :func:`metric_limits`, or None.
        size: Lookup table size.

    Returns:
        Integer indices of the same shape; -1 for a non-finite value, and
        everywhere when ``limits`` is None.
    """
    array = np.asarray(values, dtype=float)
    indices = np.full(array.shape, -1, dtype=np.int64)
    if limits is None:
        return indices
    low, high = limits
    finite = np.isfinite(array)
    if high > low:
        scaled = (array[finite] - low) / (high - low) * (size - 1)
        indices[finite] = np.clip(np.rint(scaled), 0, size - 1).astype(np.int64)
    else:
        indices[finite] = (size - 1) // 2
    return indices


def metric_colors(values: npt.ArrayLike, limits: tuple[float, float] | None) -> list[QColor]:
    """Return the fill colour of each metric value (vectorised).

    Non-finite values (and every value when ``limits`` is None) get
    :data:`NOT_FITTED_COLOR`. The returned colours are shared; do not mutate
    them.
    """
    lut = sequential_lut()
    nan_color = _CATEGORY_QCOLORS[CATEGORY_NOT_FITTED]
    return [
        lut[index] if index >= 0 else nan_color
        for index in lut_indices(values, limits, len(lut)).tolist()
    ]


def metric_color(value: float, limits: tuple[float, float] | None) -> QColor:
    """Return the fill colour of one metric value (see :func:`metric_colors`)."""
    return metric_colors([value], limits)[0]


# ---------------------------------------------------------------------------
# Text on fills
# ---------------------------------------------------------------------------

_DARK_TEXT = QColor("#1e1e1e")
_LIGHT_TEXT = QColor("#f0f0f0")


def _relative_luminance(color: QColor) -> float:
    """WCAG relative luminance of an sRGB colour."""

    def channel(value: float) -> float:
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(color.redF())
        + 0.7152 * channel(color.greenF())
        + 0.0722 * channel(color.blueF())
    )


@functools.lru_cache(maxsize=1024)
def _text_color_for_rgb(rgb: int) -> QColor:
    luminance = _relative_luminance(QColor.fromRgb(rgb))
    dark = _relative_luminance(_DARK_TEXT)
    light = _relative_luminance(_LIGHT_TEXT)
    contrast_dark = (luminance + 0.05) / (dark + 0.05)
    contrast_light = (light + 0.05) / (luminance + 0.05)
    return _DARK_TEXT if contrast_dark >= contrast_light else _LIGHT_TEXT


def text_color_for(fill: QColor) -> QColor:
    """Return the dark or light label colour with the higher contrast on ``fill``.

    adc2kev draws every strip label in the dark background colour, which is
    unreadable on the dark end of viridis; this picks per fill instead. The
    override tick uses the same colour.
    """
    return _text_color_for_rgb(fill.rgb())


def _contrast(a: float, b: float) -> float:
    """WCAG contrast ratio of two relative luminances."""
    high, low = max(a, b), min(a, b)
    return (high + 0.05) / (low + 0.05)


LABEL_MIN_CONTRAST = 3.0
"""WCAG contrast a label must reach on every fill under it to go without a halo."""


def label_colors_for_fills(fills: Iterable[QColor]) -> tuple[QColor, QColor | None]:
    """Return the text colour for a label over several fills, and a halo colour if needed.

    For a label that spills over neighbouring cells. The text colour is the
    dark or light one whose lowest contrast against any of ``fills`` is the
    higher. When even that stays below :data:`LABEL_MIN_CONTRAST` (e.g. a
    label over both ends of viridis), the other colour is returned as a halo
    to draw around the text; otherwise the halo is None.
    """
    luminances = [_relative_luminance(fill) for fill in fills]
    if not luminances:
        return _LIGHT_TEXT, None
    dark = _relative_luminance(_DARK_TEXT)
    light = _relative_luminance(_LIGHT_TEXT)
    worst_dark = min(_contrast(lum, dark) for lum in luminances)
    worst_light = min(_contrast(lum, light) for lum in luminances)
    if worst_dark >= worst_light:
        text, halo, worst = _DARK_TEXT, _LIGHT_TEXT, worst_dark
    else:
        text, halo, worst = _LIGHT_TEXT, _DARK_TEXT, worst_light
    return text, (halo if worst < LABEL_MIN_CONTRAST else None)


def text_color_for_fills(fills: Iterable[QColor]) -> QColor:
    """Text colour of :func:`label_colors_for_fills` (with one fill: :func:`text_color_for`)."""
    return label_colors_for_fills(fills)[0]


# ---------------------------------------------------------------------------
# Colour bar
# ---------------------------------------------------------------------------


class _GradientBar(QWidget):
    """The colormap drawn left (low) to right (high).

    The lookup table is handed over by :meth:`ColorBarWidget.set_metric`, so
    painting never triggers the deferred pyqtgraph import; without one only
    the frame is drawn.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(80, 10)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.lut: tuple[QColor, ...] | None = None

    def sizeHint(self) -> QSize:
        """Preferred size: 160 x 12 px."""
        return QSize(160, 12)

    def paintEvent(self, a0: QPaintEvent | None) -> None:  # noqa: ARG002
        """Paint the gradient with a thin grey frame."""
        painter = QPainter(self)
        try:
            rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            lut = self.lut
            if lut:
                gradient = QLinearGradient(QPointF(rect.left(), 0.0), QPointF(rect.right(), 0.0))
                last = len(lut) - 1
                for index in range(0, len(lut), 8):
                    gradient.setColorAt(index / last, lut[index])
                gradient.setColorAt(1.0, lut[last])
                painter.fillRect(rect, gradient)
            painter.setPen(QPen(QColor("#888888")))
            painter.drawRect(rect)
        finally:
            painter.end()


class ColorBarWidget(QWidget):
    """Metric name, low limit, the colormap and high limit on one line.

    Shown by the System Map while a metric colour mode is active. The limits
    are those of the view the grids show (anodes or cathodes).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.name_label = QLabel()
        self.min_label = QLabel()
        self.min_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.gradient = _GradientBar()
        self.max_label = QLabel()
        for widget in (self.name_label, self.min_label, self.gradient, self.max_label):
            layout.addWidget(widget)
        self._spec: MetricSpec | None = None
        self._limits: tuple[float, float] | None = None
        self.set_metric(None, None)

    @property
    def spec(self) -> MetricSpec | None:
        """Metric shown, if any."""
        return self._spec

    @property
    def limits(self) -> tuple[float, float] | None:
        """Limits shown, if any."""
        return self._limits

    def set_metric(self, spec: MetricSpec | None, limits: tuple[float, float] | None) -> None:
        """Show ``spec``'s name and ``limits`` (None limits show ``n/a``).

        Args:
            spec: The metric, or None to blank the labels and the tooltip.
            limits: ``(low, high)`` colour limits, or None when no channel
                has a finite value.
        """
        self._spec = spec
        self._limits = limits
        if spec is None:
            self.name_label.setText("")
            self.min_label.setText("")
            self.max_label.setText("")
            self.setToolTip("")
            return
        if self.gradient.lut is None:
            self.gradient.lut = sequential_lut()
            self.gradient.update()
        self.name_label.setText(f"{spec.label}:")
        if limits is None:
            self.min_label.setText("n/a")
            self.max_label.setText("n/a")
        else:
            self.min_label.setText(spec.format(limits[0]))
            self.max_label.setText(spec.format(limits[1]))
        self.setToolTip(
            f"{spec.label}: viridis colour scale clipped to the {clip_range_text()} "
            "of the channels shown (anodes or cathodes). Grey: no value; dark: no data."
        )
