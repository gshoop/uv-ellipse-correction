"""Radial tab: pre- and post-correction radius histograms with their Gaussian fits (plan 9).

Two stacked panels:

- **Pre**: the radii of the raw points about the *fitted* ellipse centre
  (:func:`~uvcorr.ellipse.radii_about_center`), with the semi-axes b and a
  marked (an uncorrected ellipse spreads its radii between them, so its
  Gaussian fit often fails: an informational flag).
- **Post**: the radii ``|(U', V')|`` of the corrected points
  (:func:`~uvcorr.ellipse.correct`), with the target radius ``sqrt(ab)``
  marked.

Each panel draws exactly what :func:`~uvcorr.metrics.radial_fit_detail`
fitted: its histogram and its Gaussian curve over the accepted fit range (the
shaded band). The text box next to it gives the histogram's binning, the fit
range and its bins, the Gaussian mean, sigma, FWHM and chi2/ndf, and the
unbinned sample mean and std, robust sigma, skewness and excess kurtosis of
:func:`~uvcorr.metrics.radial_stats`, the routine behind the stored
``pre_*`` / ``post_*`` columns, over **all** finite events of the channel
(as :func:`~uvcorr.analysis.analyze_channel` measures them). With a stored
result the numbers therefore equal the stored values; the data record checks
this (:attr:`RadialViewData.matches_stored`) and the tab says so if they
differ. A failed Gaussian fit is reported as such, and its fallback sample
mean and standard deviation are labelled as sample statistics.

The view opens on ``mu +- 5 sigma`` of the fit (and the markers) when that is
narrower than the histogram, which spans the [0.5, 99.5] percentiles and so
stretches far for a channel with a second population. *Same radius axis*
(default on) shows both panels over the union of their ranges (linked
zoom). When the plotted range hides part of a histogram, its text box says
how many radii lie outside the plotted range; both counts (own and common
range) are computed up front, so the checkbox only redraws. The text boxes
scroll when the tab is short.

A channel without an ellipse (no result, ``too_few_events``, ``fit_failed``)
shows its raw radius histogram about the median point in the upper panel,
labelled as such, and no post panel. A channel without events shows a
message.

As in the Scatter tab, the data are computed off the GUI thread by
:func:`compute_radial_view` (~0.13 s for the largest real channel, 946k
events: one :func:`~uvcorr.metrics.radial_stats_and_detail` per radius set);
:meth:`RadialTab.show_data` only draws.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from uvcorr.analysis import ChannelKey
from uvcorr.gui._contrast import readable_color
from uvcorr.gui._flow_layout import FlowLayout
from uvcorr.gui._tab_data import (
    CENTRE_FITTED,
    CORR_COLOR,
    CURVE_COLOR,
    MARKER_COLOR,
    MESSAGE_COLOR,
    RAW_COLOR,
    add_center_note,
    alpha_color,
    channel_radii,
    fmt,
    make_plot,
)
from uvcorr.gui.session import ChannelDetail, channel_title
from uvcorr.metrics import (
    MIN_GAUSS_SAMPLES,
    GaussFitDetail,
    RadialStats,
    radial_stats_and_detail,
)

__all__ = [
    "PANEL_POST",
    "PANEL_PRE",
    "STAT_FIELDS",
    "VIEW_PADDING",
    "VIEW_SIGMAS",
    "RadialPanel",
    "RadialTab",
    "RadialViewData",
    "TEXT_TOOLTIP",
    "compute_radial_view",
    "panel_html",
]

PANEL_PRE = "pre"
PANEL_POST = "post"

STAT_FIELDS: tuple[str, ...] = (
    "mean",
    "sigma",
    "fwhm",
    "chi2ndf",
    "skewness",
    "kurtosis",
    "robust_sigma",
)
"""The :class:`~uvcorr.metrics.RadialStats` fields stored as ``pre_<field>`` / ``post_<field>``."""

VIEW_SIGMAS = 5.0
"""The view opens on ``mu +- VIEW_SIGMAS * sigma`` of the fit (when narrower than the histogram)."""

VIEW_PADDING = 0.03
"""Padding added on each side of a view range, as a fraction of its span."""

CURVE_POINTS = 241
"""Points of the drawn Gaussian curve over the fit range."""

FALLBACK_MAX_BINS = 40
"""Most bins of the display-only histogram drawn when the fit routine built none."""

_MIN_DISPLAY_RANGE_REL = 1e-9
_STORED_RTOL = 1e-9  # the recomputation runs the same code on the same float64 values
_STORED_ATOL = 1e-12
_HIST_ALPHA = 110
_Z_REGION = -10
_Z_HIST = 0
_Z_CURVE = 5
_Z_MARKER = 6
_LEFT_AXIS_WIDTH = 58
_TEXT_WIDTH = 275
_NOWRAP = "white-space:nowrap"

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


# ---------------------------------------------------------------------------
# Data (Qt-free)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RadialPanel:
    """One radius histogram with its Gaussian fit and statistics.

    Attributes:
        name: :data:`PANEL_PRE` or :data:`PANEL_POST`.
        heading: Short name for the text box (``"Pre"``, ``"Post"`` or ``"Raw"``).
        title: The panel title (says what the radii are measured about).
        stats: :func:`~uvcorr.metrics.radial_stats` of the radii: the
            numbers shown (and stored as ``<name>_*`` for a fitted channel).
        fit: :func:`~uvcorr.metrics.radial_fit_detail` of the radii: the
            histogram and curve drawn (``fit.stats == stats.gauss``).
        n_values: Number of finite radii.
        edges, counts: The histogram drawn: the fit routine's, or (when the
            routine built none, e.g. fewer than 50 values) a display-only
            histogram (see ``fitted_histogram``).
        fitted_histogram: Whether ``edges``/``counts`` are the fit routine's.
        curve_x, curve_y: The Gaussian over the fit range (empty if the fit failed).
        markers: ``(label, radius)`` vertical markers (b and a, or sqrt(ab)).
        view_range: The radius range the panel shows on its own axis (*Same
            radius axis* off): ``mu +- VIEW_SIGMAS sigma`` of the fit (at least
            the fit range) within the histogram, widened to the markers and
            padded by ``VIEW_PADDING``; None without a histogram.
        view_clipped: Whether ``view_range`` hides part of the histogram.
        n_outside_view: Finite radii outside ``view_range``.
        common_clipped: Whether the common range
            (:attr:`RadialViewData.common_range`, *Same radius axis* on) hides
            part of the histogram.
        n_outside_common: Finite radii outside the common range.
        expected_failure: A failed Gaussian fit is expected here (the pre
            radii of an uncorrected ellipse).
        stored: The stored ``<name>_<field>`` values by field (None without
            a stored result for these radii).
    """

    name: str
    heading: str
    title: str
    stats: RadialStats
    fit: GaussFitDetail
    n_values: int
    edges: FloatArray
    counts: IntArray
    fitted_histogram: bool
    curve_x: FloatArray
    curve_y: FloatArray
    markers: tuple[tuple[str, float], ...]
    view_range: tuple[float, float] | None
    view_clipped: bool
    n_outside_view: int
    expected_failure: bool
    stored: Mapping[str, float | None] | None
    common_clipped: bool = False
    n_outside_common: int = 0

    def value(self, field: str) -> float:
        """The computed statistic ``field`` (one of :data:`STAT_FIELDS`)."""
        return float(getattr(self.stats, field))

    def outside(self, common_axis: bool) -> tuple[int, bool]:
        """``(radii outside the plotted range, whether it hides part of the histogram)``.

        Args:
            common_axis: Whether *Same radius axis* is on (the common range is plotted).
        """
        if common_axis:
            return self.n_outside_common, self.common_clipped
        return self.n_outside_view, self.view_clipped

    @property
    def n_fit_bins(self) -> int:
        """Histogram bins the Gaussian was fitted to (centres inside the fit range)."""
        fit_range = self.fit_range
        if fit_range is None or self.edges.size < 2:
            return 0
        centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        return int(np.count_nonzero((centers >= fit_range[0]) & (centers <= fit_range[1])))

    @property
    def bin_width(self) -> float:
        """Width of the histogram bins (NaN without a histogram)."""
        if self.edges.size < 2:
            return math.nan
        return float(self.edges[1] - self.edges[0])

    @property
    def fit_range(self) -> tuple[float, float] | None:
        """The accepted fit range, or None if the fit failed."""
        lo, hi = self.fit.fit_range
        return (lo, hi) if self.fit.stats.ok and math.isfinite(lo) and math.isfinite(hi) else None

    @property
    def mismatches(self) -> tuple[str, ...]:
        """Fields whose computed value differs from the stored one (empty without one)."""
        if self.stored is None:
            return ()
        return tuple(
            field
            for field in STAT_FIELDS
            if not _same_value(self.value(field), self.stored.get(field))
        )

    @property
    def matches_stored(self) -> bool | None:
        """Whether every statistic equals its stored value (None without a stored result)."""
        return None if self.stored is None else not self.mismatches


@dataclass(frozen=True, eq=False)
class RadialViewData:
    """Everything the Radial tab draws for one channel (:func:`compute_radial_view`).

    Attributes:
        key: The channel.
        title: Its short name (:func:`~uvcorr.gui.session.channel_title`).
        centre_kind: ``"fitted"`` or ``"median"`` (see
            :mod:`uvcorr.gui._tab_data`); None without events.
        pre: The upper panel (pre radii, or raw radii about the median point).
        post: The lower panel (corrected radii); None without an ellipse.
        n_events: Points of the channel.
        message: A note for the user (why there is no ellipse, a mismatch
            with the stored statistics, ...); empty when all is well.
        seconds: Time spent computing.
        common_range: The radius range both panels show with *Same radius
            axis* on: the union of the panels' unpadded view ranges, padded
            by ``VIEW_PADDING`` (None without a histogram).
    """

    key: ChannelKey
    title: str
    centre_kind: str | None
    pre: RadialPanel | None
    post: RadialPanel | None
    n_events: int
    message: str
    seconds: float
    common_range: tuple[float, float] | None = None

    @property
    def panels(self) -> tuple[RadialPanel, ...]:
        """The panels present, pre first."""
        return tuple(panel for panel in (self.pre, self.post) if panel is not None)

    @property
    def matches_stored(self) -> bool | None:
        """Whether both panels equal the stored statistics (None without a stored result)."""
        checks = [panel.matches_stored for panel in self.panels]
        if not checks or any(check is None for check in checks):
            return None
        return all(checks)


def _same_value(computed: float, stored: float | None) -> bool:
    """A stored None stands for a non-finite value (``ChannelResult`` normalisation)."""
    if stored is None:
        return not math.isfinite(computed)
    return math.isfinite(computed) and math.isclose(
        computed, stored, rel_tol=_STORED_RTOL, abs_tol=_STORED_ATOL
    )


def _display_histogram(values: FloatArray) -> tuple[FloatArray, IntArray]:
    """A plain histogram for values the fit routine did not histogram (few or equal values)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    n_bins = int(min(FALLBACK_MAX_BINS, max(1, math.ceil(math.sqrt(finite.size)))))
    lo, hi = float(finite.min()), float(finite.max())
    if not hi - lo > _MIN_DISPLAY_RANGE_REL * max(abs(lo), abs(hi), 1.0):
        # (nearly) equal values: one ADC around them (finer bins than the float
        # resolution would make np.histogram raise)
        centre = 0.5 * (lo + hi)
        lo, hi = centre - 0.5, centre + 0.5
    counts, edges = np.histogram(finite, bins=n_bins, range=(lo, hi))
    return edges.astype(np.float64), counts.astype(np.int64)


def _core_range(
    fit: GaussFitDetail, edges: FloatArray, markers: tuple[tuple[str, float], ...]
) -> tuple[float, float] | None:
    """A panel's unpadded view range: ``mu +- VIEW_SIGMAS sigma`` in the histogram, plus markers."""
    if edges.size == 0:
        return None
    lo_h, hi_h = float(edges[0]), float(edges[-1])
    lo, hi = lo_h, hi_h
    stats = fit.stats
    if stats.ok:
        fit_lo, fit_hi = fit.fit_range
        lo = max(lo_h, min(stats.mean - VIEW_SIGMAS * stats.sigma, fit_lo))
        hi = min(hi_h, max(stats.mean + VIEW_SIGMAS * stats.sigma, fit_hi))
    for _, radius in markers:
        lo, hi = min(lo, radius), max(hi, radius)
    if not hi > lo:
        lo, hi = lo_h, hi_h
    return lo, hi


def _padded(core: tuple[float, float]) -> tuple[float, float]:
    """``core`` padded by ``VIEW_PADDING`` of its span on each side (at least 0.5 ADC wide)."""
    lo, hi = core
    if not hi > lo:
        return lo - 0.5, hi + 0.5
    pad = VIEW_PADDING * (hi - lo)
    return lo - pad, hi + pad


def _outside(radii: FloatArray, view: tuple[float, float] | None) -> int:
    if view is None:
        return 0
    with np.errstate(invalid="ignore"):
        return int(np.count_nonzero((radii < view[0]) | (radii > view[1])))


def _hides_histogram(edges: FloatArray, view: tuple[float, float] | None) -> bool:
    return view is not None and edges.size > 0 and (view[0] > edges[0] or view[1] < edges[-1])


def _panel(
    name: str,
    heading: str,
    title: str,
    radii: FloatArray,
    markers: tuple[tuple[str, float], ...],
    stored: Mapping[str, float | None] | None,
    *,
    expected_failure: bool = False,
) -> RadialPanel:
    """One panel without its common-range fields (see :func:`_with_common_range`)."""
    # One Gaussian fit: the numbers and the drawn histogram/curve come from it
    stats, fit = radial_stats_and_detail(radii)
    markers = tuple((label, radius) for label, radius in markers if math.isfinite(radius))
    if fit.edges.size:
        edges, counts, fitted = fit.edges, fit.counts, True
    else:
        (edges, counts), fitted = _display_histogram(radii), False
    curve_x = np.empty(0, dtype=np.float64)
    curve_y = np.empty(0, dtype=np.float64)
    lo, hi = fit.fit_range
    if fit.stats.ok and math.isfinite(lo) and math.isfinite(hi):
        curve_x = np.linspace(lo, hi, CURVE_POINTS)
        curve_y = fit.curve(curve_x)
    core = _core_range(fit, edges, markers)
    view = _padded(core) if core is not None else None
    finite = np.isfinite(radii)
    return RadialPanel(
        name=name,
        heading=heading,
        title=title,
        stats=stats,
        fit=fit,
        n_values=int(np.count_nonzero(finite)),
        edges=edges,
        counts=counts,
        fitted_histogram=fitted,
        curve_x=curve_x,
        curve_y=curve_y,
        markers=markers,
        view_range=view,
        view_clipped=_hides_histogram(edges, view),
        n_outside_view=_outside(radii, view),
        expected_failure=expected_failure,
        stored=stored,
    )


def _common_range(panels: list[RadialPanel]) -> tuple[float, float] | None:
    cores = [
        _core_range(panel.fit, panel.edges, panel.markers) for panel in panels if panel.edges.size
    ]
    ranges = [core for core in cores if core is not None]
    if not ranges:
        return None
    return _padded((min(r[0] for r in ranges), max(r[1] for r in ranges)))


def _with_common_range(
    panel: RadialPanel, radii: FloatArray, common: tuple[float, float] | None
) -> RadialPanel:
    return replace(
        panel,
        common_clipped=_hides_histogram(panel.edges, common),
        n_outside_common=_outside(radii, common),
    )


def _stored_values(detail: ChannelDetail, name: str) -> dict[str, float | None] | None:
    result = detail.result
    if result is None or detail.params is None:
        return None
    return {field: getattr(result, f"{name}_{field}") for field in STAT_FIELDS}


def compute_radial_view(detail: ChannelDetail) -> RadialViewData:
    """Compute the Radial tab's data for a channel (Qt-free; run it in a worker thread).

    Args:
        detail: The channel (:meth:`~uvcorr.gui.session.UVSession.compute_detail`).

    Returns:
        The histograms, fits and statistics. Never raises for degenerate
        data (few, equal or no points).
    """
    t_start = time.perf_counter()
    title = channel_title(detail.key)
    radii = channel_radii(detail)
    if radii is None:
        return RadialViewData(
            key=detail.key,
            title=title,
            centre_kind=None,
            pre=None,
            post=None,
            n_events=0,
            message=detail.message or "No events on this channel (no data).",
            seconds=time.perf_counter() - t_start,
        )
    params = detail.params
    if radii.fitted and params is not None and radii.post is not None:
        pre = _panel(
            PANEL_PRE,
            "Pre",
            "Pre: radius about the fitted centre",
            radii.pre,
            (("b", params.b), ("a", params.a)),
            _stored_values(detail, PANEL_PRE),
            expected_failure=True,
        )
        post: RadialPanel | None = _panel(
            PANEL_POST,
            "Post",
            "Post: radius of the corrected points",
            radii.post,
            (("√(ab)", params.target_radius),),
            _stored_values(detail, PANEL_POST),
        )
    else:
        pre = _panel(
            PANEL_PRE,
            "Raw",
            "Raw: radius about the median point (no ellipse)",
            radii.pre,
            (),
            None,
        )
        post = None
    common = _common_range([panel for panel in (pre, post) if panel is not None])
    pre = _with_common_range(pre, radii.pre, common)
    if post is not None and radii.post is not None:
        post = _with_common_range(post, radii.post, common)
    notes = [detail.message] if detail.message else []
    mismatched = [
        f"{panel.name} {', '.join(panel.mismatches)}"
        for panel in (pre, post)
        if panel is not None and panel.mismatches
    ]
    if mismatched:
        notes.append(
            "The recomputed statistics differ from the stored result "
            f"({'; '.join(mismatched)}); the recomputed values are shown."
        )
    return RadialViewData(
        key=detail.key,
        title=title,
        centre_kind=radii.centre_kind,
        pre=pre,
        post=post,
        n_events=detail.n_events,
        message=" ".join(notes),
        seconds=time.perf_counter() - t_start,
        common_range=common,
    )


def _pairs_table(sections: list[tuple[str, list[tuple[str, str]]]]) -> str:
    """Titled groups of ``(name, value)`` pairs, two pairs per row (a compact table)."""
    rows = []
    for title, pairs in sections:
        if title:
            rows.append(f"<tr><td colspan='5'><i>{title}</i></td></tr>")
        cells = [
            f"<td style='{_NOWRAP}'>{name}</td><td style='{_NOWRAP}' align='right'>{value}</td>"
            for name, value in pairs
        ]
        for index in range(0, len(cells), 2):
            pair = cells[index : index + 2]
            gap = "<td>&nbsp;&nbsp;</td>" if len(pair) == 2 else ""
            rows.append(f"<tr>{pair[0]}{gap}{''.join(pair[1:])}</tr>")
    return "<table cellspacing='0' cellpadding='1'>" + "".join(rows) + "</table>"


def panel_html(
    panel: RadialPanel, warn_color: str = MESSAGE_COLOR, *, common_axis: bool = True
) -> str:
    """The text box of a panel (rich text): the fit and the unbinned statistics.

    Values use ``%.6g``, as the CSV and the Fit Inspector do; radii are in ADC.

    Args:
        panel: The panel.
        warn_color: Colour of warnings (a failed fit, a stored-value mismatch).
        common_axis: Whether *Same radius axis* is on: the "outside the
            plotted range" note counts against the range actually plotted.
    """
    stats = panel.stats
    lines = [f"<b>{panel.heading}</b> · N = {panel.n_values:,} · radii in ADC"]
    if panel.edges.size > 1:
        kind = "histogram" if panel.fitted_histogram else "display histogram"
        lines.append(f"{kind}: {panel.edges.size - 1} bins of {fmt(panel.bin_width, 3)}")
    pairs: list[tuple[str, str]] = []
    if stats.ok:
        lo, hi = panel.fit.fit_range
        k = panel.n_fit_bins
        lines.append(f"fit over [{fmt(lo, 5)}, {fmt(hi, 5)}]: {k} bins (ndf {k - 3})")
        pairs += [
            ("μ", fmt(stats.mean)),
            ("σ", fmt(stats.sigma)),
            ("FWHM", fmt(stats.fwhm)),
            ("χ²/ndf", fmt(stats.chi2ndf)),
        ]
    else:
        if not panel.fitted_histogram:
            if panel.n_values < MIN_GAUSS_SAMPLES:
                why = f"fewer than {MIN_GAUSS_SAMPLES} values"
            else:
                why = "no spread to histogram"
        elif panel.expected_failure:
            why = "usual before the correction"
        else:
            why = "no usable fit"
        text = f"<b>Gaussian fit failed</b> ({why}): μ, σ = sample mean, std"
        if not panel.expected_failure:
            text = f"<span style='color:{warn_color}'>{text}</span>"
        lines.append(text)
        pairs += [("FWHM", fmt(stats.fwhm)), ("χ²/ndf", "n/a")]
    unbinned = [
        ("mean", fmt(stats.sample_mean)),
        ("std", fmt(stats.sample_std)),
        ("robust σ", fmt(stats.robust_sigma)),
        ("skewness", fmt(stats.skewness)),
        ("ex. kurt.", fmt(stats.kurtosis)),
    ]
    unbinned += [(label, fmt(radius)) for label, radius in panel.markers]
    notes: list[str] = []
    n_outside, clipped = panel.outside(common_axis)
    if clipped and panel.n_values:
        share = 100.0 * n_outside / panel.n_values
        notes.append(f"{n_outside:,} radii ({fmt(share, 3)} %) outside the plotted range")
    if panel.matches_stored is True:
        notes.append("<i>= the stored result</i>")
    elif panel.matches_stored is False:
        notes.append(
            f"<span style='color:{warn_color}'>differs from the stored result: "
            f"{', '.join(panel.mismatches)}</span>"
        )
    table = _pairs_table([("", pairs), ("unbinned (sample)", unbinned)])
    return "<br>".join(lines) + table + "<br>".join(notes)


TEXT_TOOLTIP = (
    "μ, σ, FWHM, χ²/ndf: the Gaussian fit of radial_fit_detail (binned Poisson likelihood; "
    "χ²/ndf is the Baker-Cousins deviance per degree of freedom over the fitted bins). "
    "sample mean/std, robust σ (1.4826 MAD), skewness and excess kurtosis are unbinned, "
    "over all finite events of the channel: the stored pre_*/post_* columns. Before the "
    "correction the radii spread between b and a, so the pre fit often fails (an "
    "informational flag); μ and σ are then the sample mean and std. The histogram covers "
    "the [0.5, 99.5] percentiles of the radii; when the plotted range hides part of it, the "
    "box counts every radius outside the plotted range."
)
"""Tooltip of the text boxes."""


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


class _PanelItems:
    """The plot items of one panel, created once and refilled for every channel."""

    def __init__(self, plot: pg.PlotWidget, color: str) -> None:
        self.plot = plot
        self.view_range: tuple[float, float] | None = None
        # stepMode goes with each setData: an empty step curve is refused
        self.hist = pg.PlotCurveItem(
            fillLevel=0,
            brush=pg.mkBrush(alpha_color(color, _HIST_ALPHA)),
            pen=pg.mkPen(color, width=1),
        )
        self.hist.setZValue(_Z_HIST)
        self.region = pg.LinearRegionItem(
            movable=False,
            brush=pg.mkBrush(255, 255, 255, 28),
            pen=pg.mkPen(255, 255, 255, 90, style=Qt.PenStyle.DashLine),
        )
        self.region.setZValue(_Z_REGION)
        self.curve = pg.PlotCurveItem(pen=pg.mkPen(CURVE_COLOR, width=2))
        self.curve.setZValue(_Z_CURVE)
        self.markers = [
            pg.InfiniteLine(
                angle=90,
                movable=False,
                pen=pg.mkPen(MARKER_COLOR, width=1.5, style=Qt.PenStyle.DashLine),
                label="",
                labelOpts={
                    "position": 0.95,
                    "color": MARKER_COLOR,
                    "fill": (0, 0, 0, 170),
                    "movable": False,
                },
            )
            for _ in range(2)
        ]
        for line in self.markers:
            line.setZValue(_Z_MARKER)
        item = plot.getPlotItem()
        for graphics in (self.region, self.hist, self.curve, *self.markers):
            item.addItem(graphics)
        item.getViewBox().setAutoVisible(y=True)
        self.note = add_center_note(plot)
        self.clear()

    def clear(self, title: str = "", note: str = "") -> None:
        self.view_range = None
        self.hist.clear()
        self.curve.clear()
        for graphics in (self.hist, self.curve, self.region, *self.markers):
            graphics.setVisible(False)
        item = self.plot.getPlotItem()
        item.setTitle(title)
        item.enableAutoRange(axis="y", enable=False)
        item.setYRange(0.0, 1.0, padding=0.0)
        self.note.setText(note)

    def show(self, panel: RadialPanel) -> None:
        self.plot.getPlotItem().setTitle(panel.title)
        self.note.setText("" if panel.counts.size else "No histogram")
        self.view_range = panel.view_range
        if panel.counts.size:
            self.hist.setData(panel.edges, panel.counts.astype(np.float64), stepMode="center")
            self.hist.setVisible(True)
        else:
            self.hist.clear()
            self.hist.setVisible(False)
        if panel.curve_x.size:
            self.curve.setData(panel.curve_x, panel.curve_y)
            self.curve.setVisible(True)
        else:
            self.curve.clear()
            self.curve.setVisible(False)
        fit_range = panel.fit_range
        if fit_range is not None:
            self.region.setRegion(fit_range)
        self.region.setVisible(fit_range is not None)
        for line, marker in zip(self.markers, (*panel.markers, None, None)):
            if marker is None:
                line.setVisible(False)
                continue
            label, radius = marker
            # Visible first: a hidden InfLineLabel ignores setFormat's text update
            line.setVisible(True)
            line.setValue(radius)
            line.label.setFormat(label)
        self.plot.getPlotItem().enableAutoRange(axis="y")


class RadialTab(QWidget):
    """The Radial tab (see the module docstring).

    Signals:
        display_changed(): The user toggled *Same radius axis* (for persistence).
    """

    display_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data: RadialViewData | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        controls = FlowLayout(h_spacing=8, v_spacing=2)
        self.common_axis_check = QCheckBox("Same radius axis")
        self.common_axis_check.setChecked(True)
        self.common_axis_check.setToolTip(
            "Show both histograms over one radius range (linked zoom), so the narrowing by "
            "the correction is visible; off: each panel is scaled to its own histogram"
        )
        self.info_label = QLabel()
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        controls.addWidget(self.common_axis_check)
        controls.addWidget(self.info_label)
        layout.addLayout(controls)

        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet(f"color: {MESSAGE_COLOR};")
        self.message_label.hide()
        layout.addWidget(self.message_label)

        self.pre_plot = make_plot("Pre", "radius (ADC)", "counts / bin")
        self.post_plot = make_plot("Post", "radius (ADC)", "counts / bin")
        for plot in (self.pre_plot, self.post_plot):
            # Equal axis widths line the two views up, so linked ranges are identical
            plot.getPlotItem().getAxis("left").setWidth(_LEFT_AXIS_WIDTH)
        tip = (
            "histogram and Gaussian fit (white) of radial_fit_detail, the routine behind the "
            "stored statistics; the shaded band is the fit range"
        )
        self.pre_plot.setToolTip(
            f"Radii of the raw points about the fitted ellipse centre (orange): {tip}; "
            "dashed lines: the semi-axes b and a"
        )
        self.post_plot.setToolTip(
            f"Radii of the corrected points (blue): {tip}; dashed line: the target radius √(ab)"
        )
        self._pre_items = _PanelItems(self.pre_plot, RAW_COLOR)
        self._post_items = _PanelItems(self.post_plot, CORR_COLOR)
        self.pre_text, self.pre_scroll = self._make_text()
        self.post_text, self.post_scroll = self._make_text()
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        grid.addWidget(self.pre_plot, 0, 0)
        grid.addWidget(self.pre_scroll, 0, 1)
        grid.addWidget(self.post_plot, 1, 0)
        grid.addWidget(self.post_scroll, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        layout.addLayout(grid, 1)

        self.common_axis_check.toggled.connect(self._on_common_axis_toggled)
        self._render()

    @staticmethod
    def _make_text() -> tuple[QLabel, QScrollArea]:
        """A text box in a frameless, vertically scrolling area (it never gets clipped)."""
        label = QLabel()
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setWordWrap(True)
        label.setToolTip(TEXT_TOOLTIP)
        area = QScrollArea()
        area.setWidget(label)
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setFixedWidth(_TEXT_WIDTH)
        viewport = area.viewport()
        if viewport is not None:
            viewport.setBackgroundRole(QPalette.ColorRole.Window)
        return label, area

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def data(self) -> RadialViewData | None:
        """The channel shown, if any."""
        return self._data

    @property
    def common_axis(self) -> bool:
        """Whether both panels share one radius axis."""
        return self.common_axis_check.isChecked()

    def set_common_axis(self, on: bool) -> None:
        """Switch the shared radius axis on or off (redraws; emits nothing)."""
        self.common_axis_check.blockSignals(True)
        try:
            self.common_axis_check.setChecked(on)
        finally:
            self.common_axis_check.blockSignals(False)
        self._apply_ranges()
        self._update_texts()

    def show_data(self, data: RadialViewData | None) -> None:
        """Draw a channel's histograms (None clears)."""
        self._data = data
        self._render()

    def show_loading(self, text: str) -> None:
        """Note that a channel is loading (the previous one stays drawn meanwhile)."""
        self.info_label.setText(f"Loading {text} …")

    def clear(self, message: str = "") -> None:
        """Show nothing, with an optional message."""
        self._data = None
        self._render(message)

    def message(self) -> str:
        """The message shown above the plots (empty if hidden)."""
        return self.message_label.text() if not self.message_label.isHidden() else ""

    def info_text(self) -> str:
        """The info line."""
        return str(self.info_label.text())

    def stats_html(self, name: str) -> str:
        """The text box of the ``"pre"`` or ``"post"`` panel (rich text)."""
        label = self.pre_text if name == PANEL_PRE else self.post_text
        return str(label.text())

    def x_ranges(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The radius ranges the pre and post panels show."""
        pre = self.pre_plot.getPlotItem().getViewBox().viewRange()[0]
        post = self.post_plot.getPlotItem().getViewBox().viewRange()[0]
        return (float(pre[0]), float(pre[1])), (float(post[0]), float(post[1]))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _warn_color(self) -> str:
        background = self.palette().color(QPalette.ColorRole.Window)
        return str(readable_color(MESSAGE_COLOR, background).name())

    def _set_message(self, text: str) -> None:
        self.message_label.setText(text)
        self.message_label.setVisible(bool(text))

    def _update_texts(self) -> None:
        """Refill both text boxes (their "outside the plotted range" note follows the axis)."""
        data = self._data
        warn = self._warn_color()
        for panel, label in (
            (None if data is None else data.pre, self.pre_text),
            (None if data is None else data.post, self.post_text),
        ):
            label.setText(
                "" if panel is None else panel_html(panel, warn, common_axis=self.common_axis)
            )

    def _render(self, message: str = "") -> None:
        data = self._data
        self._update_texts()
        if data is None:
            self._pre_items.clear("Pre")
            self._post_items.clear("Post")
            self.info_label.setText("No channel selected")
            self._set_message(message)
            self._apply_ranges()
            return
        self._set_message(data.message)
        if data.pre is not None:
            self._pre_items.show(data.pre)
        else:
            self._pre_items.clear("Pre", "No events on this channel")
        if data.post is not None:
            self._post_items.show(data.post)
        elif data.n_events:
            self._post_items.clear("Post", "No ellipse fitted: no corrected radii")
        else:
            self._post_items.clear("Post", "No events on this channel")
        parts = [data.title]
        if data.n_events:
            parts.append(f"{data.n_events:,} points, all used")
            if data.centre_kind == CENTRE_FITTED:
                parts.append("pre radii about the fitted centre")
            else:
                parts.append("radii about the median point (no ellipse)")
        self.info_label.setText(" · ".join(parts))
        self._apply_ranges()

    def _apply_ranges(self) -> None:
        """Set the radius ranges the data record's notes were counted against."""
        pre_item = self.pre_plot.getPlotItem()
        post_item = self.post_plot.getPlotItem()
        data = self._data
        common = None if data is None else data.common_range
        if self.common_axis and common is not None:
            post_item.setXLink(pre_item)
            pre_item.setXRange(common[0], common[1], padding=0.0)
            post_item.setXRange(common[0], common[1], padding=0.0)
            return
        post_item.setXLink(None)
        for item, view in (
            (pre_item, self._pre_items.view_range),
            (post_item, self._post_items.view_range),
        ):
            if view is not None:
                item.setXRange(view[0], view[1], padding=0.0)
            else:
                item.setXRange(0.0, 1.0, padding=0.0)

    def _on_common_axis_toggled(self, _checked: bool) -> None:
        self._apply_ranges()
        self._update_texts()
        self.display_changed.emit()
