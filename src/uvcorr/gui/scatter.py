"""Scatter tab: a channel's (U, V) points before and after the correction (plan section 9).

Two pyqtgraph panels with a locked 1:1 aspect ratio:

- **Raw** (U, V) with the fitted ellipse, its centre and both semi-axes;
  points the robust fit rejected are drawn in a muted grey.
- **Corrected** (U', V') with the target circle ``r = sqrt(ab)``.

The panels are linked by a translation: the corrected panel shows the same
span as the raw one, shifted by the ellipse centre (the corrected points are
centred on the origin), so zooming or panning either panel follows the same
part of the ring in both.

Display options:

- *Overlay*: raw-minus-centre and corrected points on one centred axis pair
  (needs a fitted ellipse).
- *Density*: log-coloured 2D histograms (``pg.ImageItem``) instead of
  points, over a square around the ring (the info line says how many points
  fall inside it). Bins are a whole number of ADC wide and aligned to the
  integer U/V lattice, so no moire pattern appears. In overlay mode the two
  histograms are blended additively (orange raw, blue corrected, white where
  they coincide).
- *Point cap* (default 50k, at most 300k): the kept points above the cap are
  a deterministic random subsample seeded by the channel address
  (specview's point cap); the points the robust fit rejected are drawn in
  addition, all of them up to :data:`REJECTED_CAP`, so outliers are never
  hidden by the subsample. The info line reports both ("showing N of K kept
  points · all R rejected shown"). pyqtgraph scatter plots slow down above
  ~100k points; the density mode is always complete and fast (one
  ``bincount``).

Channels without an ellipse (no result, ``too_few_events``,
``fit_failed``) show the raw points only and a message.

The tab only draws a :class:`~uvcorr.gui.session.ChannelDetail`; loading and
the fit-mask recomputation happen in a worker thread (see
:class:`~uvcorr.gui.threads.ChannelDetailThread`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from uvcorr.ellipse import EllipseParams, ellipse_points
from uvcorr.gui._flow_layout import FlowLayout
from uvcorr.gui.session import ChannelDetail, channel_title

__all__ = [
    "CORR_COLOR",
    "DEFAULT_POINT_CAP",
    "DENSITY_BINS",
    "MAX_POINT_CAP",
    "MIN_POINT_CAP",
    "MUTED_COLOR",
    "RAW_COLOR",
    "REJECTED_CAP",
    "ScatterTab",
    "density_grid",
    "density_image",
    "subsample_indices",
]

DEFAULT_POINT_CAP = 50_000
MIN_POINT_CAP = 1_000
MAX_POINT_CAP = 300_000
"""Above ~300k points a pyqtgraph scatter takes over a second per redraw; use Density."""
POINT_CAP_STEP = 10_000

REJECTED_CAP = 50_000
"""Most rejected (grey) points drawn per panel; below it every rejected point is drawn."""

DENSITY_BINS = 256
"""Most histogram bins per axis. Bins are a whole number of ADC wide (the raw data
are integers), aligned so every bin holds the same number of integer values."""

# Colour-vision-safe pair (orange / blue) on the black plot background.
RAW_COLOR = "#f0a04b"
CORR_COLOR = "#4aa3df"
MUTED_COLOR = (128, 128, 128, 150)
CURVE_COLOR = (242, 242, 242, 200)  # slightly translucent: thin rings show underneath
MARKER_COLOR = "#ffe14d"
MESSAGE_COLOR = "#f39c12"

_OVERLAY_ALPHA = 150
_CURVE_POINTS = 721
_VIEW_MARGIN = 1.15  # half-span of the initial view, in units of the semi-major axis
_Z_IMAGE = -10
_Z_REJECTED = 0
_Z_POINTS = 1
_Z_CURVES = 5
_Z_MARKER = 6

FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.intp]


def subsample_indices(n: int, cap: int, seed: tuple[int, ...] | list[int]) -> IndexArray | None:
    """Deterministic random subsample of ``range(n)`` of size ``cap``, sorted.

    Args:
        n: Number of points.
        cap: Maximum number of points to draw.
        seed: Seed (e.g. the channel address), so a channel always shows the
            same subsample.

    Returns:
        Sorted indices, or None when ``n <= cap`` (draw everything).
    """
    if n <= cap:
        return None
    rng = np.random.default_rng([abs(int(s)) for s in seed])
    indices = rng.choice(n, size=cap, replace=False)
    indices.sort()
    return indices.astype(np.intp, copy=False)


def density_image(
    x: npt.ArrayLike, y: npt.ArrayLike, x0: float, y0: float, span: float, bins: int
) -> npt.NDArray[np.int64]:
    """2D histogram of the points in the square ``[x0, x0+span) x [y0, y0+span)``.

    One ``bincount`` over flattened bin indices (several times faster than
    ``np.histogram2d`` at 1M points). Points outside the square are ignored.

    Returns:
        Counts of shape ``(bins, bins)`` indexed ``[ix, iy]`` (pyqtgraph's
        default column-major image order).
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    scale = bins / span
    with np.errstate(invalid="ignore"):
        fx = np.floor((xs - x0) * scale)
        fy = np.floor((ys - y0) * scale)
        inside = (fx >= 0) & (fx < bins) & (fy >= 0) & (fy < bins)
    ix = fx[inside].astype(np.intp)
    iy = fy[inside].astype(np.intp)
    counts = np.bincount(ix * bins + iy, minlength=bins * bins)
    return counts.reshape(bins, bins).astype(np.int64, copy=False)


def density_grid(
    cx: float, cy: float, half: float, lattice: tuple[float, float] | None = None
) -> tuple[float, float, float, int]:
    """Square histogram grid covering ``[cx - half, cx + half]`` (and ``cy``).

    Bins are ``width = ceil(2 half / DENSITY_BINS)`` ADC wide (at least 1).
    With ``lattice = (ox, oy)`` the data lie on the integer lattice shifted
    by ``(ox, oy)`` (raw U/V: ``(0, 0)``; raw minus the centre: ``(-cU, -cV)``)
    and the bin edges fall half-way between lattice points, so every bin
    holds exactly ``width`` lattice columns and rows (no moire modulation).

    Returns:
        ``(x0, y0, width, bins)``: the lower-left corner, the bin width and
        the bins per axis (the square spans ``bins * width``).
    """
    width = float(max(1, math.ceil(2.0 * half / DENSITY_BINS)))
    x0, y0 = cx - half, cy - half
    if lattice is not None:
        ox, oy = lattice
        x0 = math.floor(x0 - ox - 0.5) + 0.5 + ox
        y0 = math.floor(y0 - oy - 0.5) + 0.5 + oy
    bins = max(8, math.ceil((cx + half - x0) / width), math.ceil((cy + half - y0) / width))
    return x0, y0, width, int(bins)


@dataclass(frozen=True)
class _Sample:
    """Which points a panel draws: kept points subsampled, rejected ones up to their cap."""

    kept: IndexArray | None  # None: every point (a channel without a fit mask, not capped)
    rejected: IndexArray
    n_kept: int  # kept points in the channel (all points without a mask)
    n_rejected: int

    @property
    def shown_kept(self) -> int:
        return self.n_kept if self.kept is None else int(self.kept.shape[0])

    @property
    def shown_rejected(self) -> int:
        return int(self.rejected.shape[0])


_NO_INDEX: IndexArray = np.zeros(0, dtype=np.intp)


def _viridis_lut() -> npt.NDArray[np.uint8]:
    """Viridis with a transparent first entry (empty bins show the background)."""
    lut = np.asarray(pg.colormap.get("viridis").getLookupTable(nPts=256, alpha=True))
    lut = lut.astype(np.uint8, copy=True)
    lut[0, 3] = 0
    return lut


def _ramp_lut(color: str) -> npt.NDArray[np.uint8]:
    """Black to ``color`` (for additive blending)."""
    rgb = np.array(pg.mkColor(color).getRgb()[:3], dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, 256)[:, None] * rgb[None, :]
    lut = np.empty((256, 4), dtype=np.uint8)
    lut[:, :3] = np.rint(ramp).astype(np.uint8)
    lut[:, 3] = 255
    return lut


def _take(values: npt.NDArray[Any], index: IndexArray | None) -> npt.NDArray[Any]:
    return values if index is None else values[index]


def _alpha(color: str, alpha: int) -> tuple[int, int, int, int]:
    r, g, b, _ = pg.mkColor(color).getRgb()
    return (r, g, b, alpha)


class ScatterTab(QWidget):
    """The Scatter tab (see the module docstring).

    Signals:
        point_cap_changed(int): The user changed the point cap.
        display_changed(): The user toggled Overlay or Density.
    """

    point_cap_changed = pyqtSignal(int)
    display_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._detail: ChannelDetail | None = None
        self._sample_for: tuple[tuple[int, ...], int, int] | None = None
        self._sample: _Sample | None = None
        self._offset: tuple[float, float] | None = None
        self._syncing = False
        self._drawn: _Sample | None = None
        self._density_counted: int | None = None
        self._viridis = _viridis_lut()
        self._raw_ramp = _ramp_lut(RAW_COLOR)
        self._corr_ramp = _ramp_lut(CORR_COLOR)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        controls = FlowLayout(h_spacing=8, v_spacing=2)
        self.overlay_check = QCheckBox("Overlay")
        self.overlay_check.setToolTip(
            "Show the raw points minus the ellipse centre and the corrected points on one "
            "centred axis pair"
        )
        self.density_check = QCheckBox("Density")
        self.density_check.setToolTip(
            "Show log-coloured 2D histograms of all points instead of (capped) points"
        )
        self.cap_spin = QSpinBox()
        self.cap_spin.setRange(MIN_POINT_CAP, MAX_POINT_CAP)
        self.cap_spin.setSingleStep(POINT_CAP_STEP)
        self.cap_spin.setValue(DEFAULT_POINT_CAP)
        self.cap_spin.setGroupSeparatorShown(True)
        self.cap_spin.setKeyboardTracking(False)
        self.cap_spin.setToolTip(
            "Most kept points drawn per panel (larger channels show a fixed random subsample; "
            f"rejected points are drawn in addition, up to {REJECTED_CAP:,})"
        )
        self.info_label = QLabel()
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cap_box = QWidget()
        cap_layout = QHBoxLayout(cap_box)
        cap_layout.setContentsMargins(0, 0, 0, 0)
        cap_layout.setSpacing(4)
        cap_layout.addWidget(QLabel("Point cap:"))
        cap_layout.addWidget(self.cap_spin)
        controls.addWidget(self.overlay_check)
        controls.addWidget(self.density_check)
        controls.addWidget(cap_box)
        controls.addWidget(self.info_label)
        layout.addLayout(controls)

        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet(f"color: {MESSAGE_COLOR};")
        self.message_label.hide()
        layout.addWidget(self.message_label)

        self.raw_plot = self._make_plot("Raw (U, V)", "U (ADC)", "V (ADC)")
        self.raw_plot.setToolTip(
            "Raw points (orange) with the fitted ellipse: + centre, solid line the "
            "semi-major axis a, dashed line the semi-minor axis b; grey points were rejected "
            "by the robust fit"
        )
        self.corr_plot = self._make_plot("Corrected (U′, V′)", "U′ (ADC)", "V′ (ADC)")
        self.corr_plot.setToolTip(
            "Corrected points (blue), centred on the origin, with the target circle "
            "r = √(ab); the view follows the raw panel's zoom and pan"
        )
        self.overlay_plot = self._make_plot(
            "Overlay: raw − centre and corrected", "U − cU, U′ (ADC)", "V − cV, V′ (ADC)"
        )
        side = QWidget()
        side_layout = QHBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(4)
        side_layout.addWidget(self.raw_plot, 1)
        side_layout.addWidget(self.corr_plot, 1)
        self.stack = QStackedWidget()
        self.stack.addWidget(side)
        self.stack.addWidget(self.overlay_plot)
        layout.addWidget(self.stack, 1)

        self.raw_plot.getViewBox().sigRangeChanged.connect(self._on_raw_range_changed)
        self.corr_plot.getViewBox().sigRangeChanged.connect(self._on_corr_range_changed)
        self.overlay_check.toggled.connect(self._on_display_toggled)
        self.density_check.toggled.connect(self._on_display_toggled)
        self.cap_spin.valueChanged.connect(self._on_cap_changed)
        self._update_info()

    @staticmethod
    def _make_plot(title: str, x_label: str, y_label: str) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        item = plot.getPlotItem()
        item.setTitle(title)
        item.setLabel("bottom", x_label)
        item.setLabel("left", y_label)
        item.showGrid(x=True, y=True, alpha=0.2)
        item.setAspectLocked(True, ratio=1.0)
        item.addLegend(
            offset=(6, 6), labelTextSize="8pt", verSpacing=-6, brush=pg.mkBrush(0, 0, 0, 170)
        )
        return plot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def detail(self) -> ChannelDetail | None:
        """The channel shown, if any."""
        return self._detail

    @property
    def overlay(self) -> bool:
        """Whether the overlay toggle is on."""
        return self.overlay_check.isChecked()

    @property
    def density(self) -> bool:
        """Whether the density toggle is on."""
        return self.density_check.isChecked()

    @property
    def point_cap(self) -> int:
        """Most points drawn per panel."""
        return int(self.cap_spin.value())

    @property
    def shown_points(self) -> int:
        """Points drawn per panel, kept and rejected (0 in density mode or without a channel)."""
        drawn = self._drawn
        return 0 if drawn is None else drawn.shown_kept + drawn.shown_rejected

    @property
    def shown_rejected(self) -> int:
        """Rejected points drawn per panel."""
        return 0 if self._drawn is None else self._drawn.shown_rejected

    @property
    def density_counted(self) -> int | None:
        """Points inside the raw density histogram (None outside density mode)."""
        return self._density_counted

    @property
    def overlay_active(self) -> bool:
        """Whether the single overlay panel is shown (toggle on and an ellipse to centre on)."""
        return self.stack.currentIndex() == 1

    def set_overlay(self, on: bool) -> None:
        """Switch the overlay on or off (redraws; emits nothing)."""
        self._set_checked(self.overlay_check, on)

    def set_density(self, on: bool) -> None:
        """Switch the density mode on or off (redraws; emits nothing)."""
        self._set_checked(self.density_check, on)

    def set_point_cap(self, cap: int) -> None:
        """Set the point cap (clamped to its range; redraws; emits nothing)."""
        self.cap_spin.blockSignals(True)
        try:
            self.cap_spin.setValue(int(cap))
        finally:
            self.cap_spin.blockSignals(False)
        self._render(autorange=False)

    def set_detail(self, detail: ChannelDetail | None) -> None:
        """Show a channel (None clears); the view is re-centred on it."""
        self._detail = detail
        self._offset = None
        if detail is not None and detail.params is not None:
            self._offset = (detail.params.cx, detail.params.cy)
        self._set_message(detail.message if detail is not None else "")
        self._render(autorange=True)

    def show_loading(self, text: str) -> None:
        """Note that a channel is loading (the previous one stays drawn meanwhile)."""
        self.info_label.setText(f"Loading {text} …")

    def clear(self, message: str = "") -> None:
        """Show nothing, with an optional message."""
        self._detail = None
        self._offset = None
        self._set_message(message)
        self._render(autorange=False)

    def message(self) -> str:
        """The message shown above the plots (empty if hidden)."""
        return self.message_label.text() if not self.message_label.isHidden() else ""

    def info_text(self) -> str:
        """The info line ("showing N of M points", ...)."""
        return str(self.info_label.text())

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _set_checked(self, box: QCheckBox, on: bool) -> None:
        box.blockSignals(True)
        try:
            box.setChecked(on)
        finally:
            box.blockSignals(False)
        self._render(autorange=False)

    def _set_message(self, text: str) -> None:
        self.message_label.setText(text)
        self.message_label.setVisible(bool(text))

    def _sample_for_detail(self, detail: ChannelDetail) -> _Sample:
        """The points to draw (cached per channel and cap; deterministic per channel)."""
        seed = tuple(int(k) for k in detail.key)
        wanted = (seed, detail.n_events, self.point_cap)
        if self._sample is not None and self._sample_for == wanted:
            return self._sample
        if detail.kept is None:
            sample = _Sample(
                kept=subsample_indices(detail.n_events, self.point_cap, seed),
                rejected=_NO_INDEX,
                n_kept=detail.n_events,
                n_rejected=0,
            )
        else:
            kept = np.flatnonzero(detail.kept)
            rejected = np.flatnonzero(~detail.kept)
            sub_kept = subsample_indices(kept.shape[0], self.point_cap, seed)
            sub_rejected = subsample_indices(rejected.shape[0], REJECTED_CAP, (*seed, 1))
            sample = _Sample(
                kept=kept if sub_kept is None else kept[sub_kept],
                rejected=rejected if sub_rejected is None else rejected[sub_rejected],
                n_kept=int(kept.shape[0]),
                n_rejected=int(rejected.shape[0]),
            )
        self._sample, self._sample_for = sample, wanted
        return sample

    def _render(self, autorange: bool) -> None:
        """Redraw the current channel; re-centre the view if asked or if the layout changed."""
        for plot in (self.raw_plot, self.corr_plot, self.overlay_plot):
            plot.clear()
        self.corr_plot.getPlotItem().setTitle("Corrected (U′, V′)")
        detail = self._detail
        overlay = self.overlay and detail is not None and detail.params is not None
        page = 1 if overlay else 0
        if page != self.stack.currentIndex():
            self.stack.setCurrentIndex(page)
            autorange = True
        self._drawn = None
        self._density_counted = None
        if detail is not None:
            if overlay:
                self._render_overlay(detail)
            else:
                self._render_side(detail)
            if autorange:
                self._auto_range(detail, overlay)
        self._update_info()

    def _render_side(self, detail: ChannelDetail) -> None:
        raw, corr = self.raw_plot, self.corr_plot
        params = detail.params
        if self.density:
            cx, cy, half = self._view_square(detail)
            x0, y0, width, bins = density_grid(cx, cy, half, lattice=(0.0, 0.0))
            self._density_counted = self._add_density(
                raw, detail.u, detail.v, x0, y0, width, bins, self._viridis
            )
            if detail.u_corr is not None and detail.v_corr is not None:
                cx0, cy0, _, _ = density_grid(0.0, 0.0, half)
                self._add_density(
                    corr, detail.u_corr, detail.v_corr, cx0, cy0, width, bins, self._viridis
                )
        else:
            sample = self._sample_for_detail(detail)
            self._drawn = sample
            self._add_points(raw, detail.u, detail.v, sample, RAW_COLOR, None)
            if detail.u_corr is not None and detail.v_corr is not None:
                self._add_points(corr, detail.u_corr, detail.v_corr, sample, CORR_COLOR, None)
        if params is None:
            corr.getPlotItem().setTitle("Corrected (U′, V′): no ellipse fitted")
            return
        self._add_ellipse_overlays(raw, params)
        self._add_circle(corr, params.target_radius, "target circle √(ab)")
        self._add_marker(corr, 0.0, 0.0)

    def _render_overlay(self, detail: ChannelDetail) -> None:
        params = detail.params
        assert params is not None and detail.u_corr is not None and detail.v_corr is not None
        plot = self.overlay_plot
        centred = replace(params, cx=0.0, cy=0.0)
        x_raw = detail.u - params.cx
        y_raw = detail.v - params.cy
        if self.density:
            _, _, half = self._view_square(detail)
            x0, y0, width, bins = density_grid(0.0, 0.0, half, lattice=(-params.cx, -params.cy))
            self._density_counted = self._add_density(
                plot, x_raw, y_raw, x0, y0, width, bins, self._raw_ramp, additive=True
            )
            cx0, cy0, _, _ = density_grid(0.0, 0.0, half)
            self._add_density(
                plot,
                detail.u_corr,
                detail.v_corr,
                cx0,
                cy0,
                width,
                bins,
                self._corr_ramp,
                additive=True,
            )
        else:
            sample = self._sample_for_detail(detail)
            self._drawn = sample
            color_raw = _alpha(RAW_COLOR, _OVERLAY_ALPHA)
            color_corr = _alpha(CORR_COLOR, _OVERLAY_ALPHA)
            self._add_points(plot, x_raw, y_raw, sample, color_raw, "raw − centre")
            self._add_points(
                plot,
                detail.u_corr,
                detail.v_corr,
                sample,
                color_corr,
                "corrected",
                rejected_name=None,
            )
        self._add_ellipse_curve(plot, centred, "fitted ellipse")
        self._add_circle(plot, params.target_radius, "target circle", dashed=True)
        self._add_marker(plot, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------

    @staticmethod
    def _scatter(
        x: FloatArray, y: FloatArray, color: Any, name: str | None, z: float
    ) -> pg.ScatterPlotItem:
        size = 3 if x.shape[0] <= 20_000 else 2
        item = pg.ScatterPlotItem(
            x=x, y=y, pen=None, brush=pg.mkBrush(color), size=size, pxMode=True, name=name
        )
        item.setZValue(z)
        return item

    def _add_points(
        self,
        plot: pg.PlotWidget,
        x: FloatArray,
        y: FloatArray,
        sample: _Sample,
        color: Any,
        name: str | None,
        rejected_name: str | None = "rejected",
    ) -> None:
        """Draw the sample's kept points in ``color`` and its rejected points in grey."""
        if sample.shown_rejected:
            index = sample.rejected
            plot.addItem(self._scatter(x[index], y[index], MUTED_COLOR, rejected_name, _Z_REJECTED))
        kept = sample.kept
        plot.addItem(self._scatter(_take(x, kept), _take(y, kept), color, name, _Z_POINTS))

    def _add_density(
        self,
        plot: pg.PlotWidget,
        x: FloatArray,
        y: FloatArray,
        x0: float,
        y0: float,
        width: float,
        bins: int,
        lut: npt.NDArray[np.uint8],
        additive: bool = False,
    ) -> int:
        """Add a log-coloured histogram image; returns the number of points inside it."""
        span = bins * width
        counts = density_image(x, y, x0, y0, span, bins)
        image = np.log1p(counts.astype(np.float64))
        item = pg.ImageItem(image)
        item.setLookupTable(lut)
        top = float(image.max())
        item.setLevels((0.0, top if top > 0 else 1.0))
        item.setRect(QRectF(x0, y0, span, span))
        item.setZValue(_Z_IMAGE)
        if additive:
            item.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        plot.addItem(item)
        return int(counts.sum())

    def _add_ellipse_curve(self, plot: pg.PlotWidget, params: EllipseParams, name: str) -> None:
        t = np.linspace(0.0, 2.0 * math.pi, _CURVE_POINTS)
        x, y = ellipse_points(params, t)
        curve = pg.PlotCurveItem(x, y, pen=pg.mkPen(CURVE_COLOR, width=1.5), name=name)
        curve.setZValue(_Z_CURVES)
        plot.addItem(curve)

    def _add_ellipse_overlays(self, plot: pg.PlotWidget, params: EllipseParams) -> None:
        """Fitted ellipse, centre marker and the two semi-axes."""
        self._add_ellipse_curve(plot, params, "fitted ellipse")
        cos_p, sin_p = math.cos(params.phi), math.sin(params.phi)
        major = pg.PlotCurveItem(
            [params.cx, params.cx + params.a * cos_p],
            [params.cy, params.cy + params.a * sin_p],
            pen=pg.mkPen(MARKER_COLOR, width=2),
        )
        minor = pg.PlotCurveItem(
            [params.cx, params.cx - params.b * sin_p],
            [params.cy, params.cy + params.b * cos_p],
            pen=pg.mkPen(MARKER_COLOR, width=2, style=Qt.PenStyle.DashLine),
        )
        for item in (major, minor):
            item.setZValue(_Z_CURVES)
            plot.addItem(item)
        self._add_marker(plot, params.cx, params.cy)

    def _add_circle(
        self, plot: pg.PlotWidget, radius: float, name: str, dashed: bool = False
    ) -> None:
        t = np.linspace(0.0, 2.0 * math.pi, _CURVE_POINTS)
        style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        curve = pg.PlotCurveItem(
            radius * np.cos(t),
            radius * np.sin(t),
            pen=pg.mkPen(CURVE_COLOR if not dashed else MARKER_COLOR, width=1.5, style=style),
            name=name,
        )
        curve.setZValue(_Z_CURVES)
        plot.addItem(curve)

    @staticmethod
    def _add_marker(plot: pg.PlotWidget, x: float, y: float) -> None:
        marker = pg.ScatterPlotItem(
            [x], [y], symbol="+", size=16, pen=pg.mkPen(MARKER_COLOR, width=2), brush=None
        )
        marker.setZValue(_Z_MARKER)
        plot.addItem(marker)

    # ------------------------------------------------------------------
    # Ranges
    # ------------------------------------------------------------------

    @staticmethod
    def _view_square(detail: ChannelDetail) -> tuple[float, float, float]:
        """Centre and half-span of the initial view (and of the density square)."""
        params = detail.params
        n = detail.n_events
        if n == 0:
            if params is not None:
                return params.cx, params.cy, _VIEW_MARGIN * params.a
            return 0.0, 0.0, 1.0
        step = max(1, n // 100_000)
        u = detail.u[::step]
        v = detail.v[::step]
        if params is not None:
            cx, cy = params.cx, params.cy
        else:
            cx, cy = float(np.median(u)), float(np.median(v))
        extent = float(np.percentile(np.hypot(u - cx, v - cy), 99.5))
        half = 1.05 * extent
        if params is not None:
            half = max(half, _VIEW_MARGIN * params.a)
        return cx, cy, max(half, 5.0)

    def _auto_range(self, detail: ChannelDetail, overlay: bool) -> None:
        cx, cy, half = self._view_square(detail)
        if overlay:
            self.overlay_plot.setRange(xRange=(-half, half), yRange=(-half, half), padding=0.0)
            return
        self.raw_plot.setRange(
            xRange=(cx - half, cx + half), yRange=(cy - half, cy + half), padding=0.0
        )
        if self._offset is None:
            self.corr_plot.setRange(xRange=(-half, half), yRange=(-half, half), padding=0.0)

    def _sync_range(self, source: pg.PlotWidget, target: pg.PlotWidget, sign: float) -> None:
        if self._syncing or self._offset is None:
            return
        (x0, x1), (y0, y1) = source.getViewBox().viewRange()
        dx, dy = sign * self._offset[0], sign * self._offset[1]
        self._syncing = True
        try:
            target.setRange(xRange=(x0 + dx, x1 + dx), yRange=(y0 + dy, y1 + dy), padding=0.0)
        finally:
            self._syncing = False

    def _on_raw_range_changed(self, *_args: Any) -> None:
        self._sync_range(self.raw_plot, self.corr_plot, -1.0)

    def _on_corr_range_changed(self, *_args: Any) -> None:
        self._sync_range(self.corr_plot, self.raw_plot, +1.0)

    # ------------------------------------------------------------------
    # Info line and controls
    # ------------------------------------------------------------------

    def _update_info(self) -> None:
        detail = self._detail
        if detail is None:
            self.info_label.setText("No channel selected")
            return
        parts = [channel_title(detail.key)]
        n = detail.n_events
        counted = self._density_counted
        drawn = self._drawn
        if counted is not None:
            if counted == n:
                parts.append(f"density of all {n:,} points (log colour)")
            else:
                parts.append(f"density of {counted:,} of {n:,} points in range (log colour)")
        elif drawn is not None:
            if detail.kept is None:
                parts.append(f"showing {drawn.shown_kept:,} of {n:,} points")
            else:
                parts.append(f"showing {drawn.shown_kept:,} of {drawn.n_kept:,} kept points")
                if drawn.n_rejected:
                    if drawn.shown_rejected == drawn.n_rejected:
                        parts.append(f"all {drawn.n_rejected:,} rejected shown (grey)")
                    else:
                        parts.append(
                            f"{drawn.shown_rejected:,} of {drawn.n_rejected:,} rejected "
                            "shown (grey)"
                        )
        if self.overlay and not self.overlay_active:
            parts.append("overlay needs a fitted ellipse")
        self.info_label.setText(" · ".join(parts))

    def _on_display_toggled(self, _checked: bool) -> None:
        self._render(autorange=False)
        self.display_changed.emit()

    def _on_cap_changed(self, value: int) -> None:
        self._render(autorange=False)
        self.point_cap_changed.emit(int(value))
