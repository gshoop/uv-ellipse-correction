"""Radius vs angle tab: the mean radius and its spread in 72 bins of 5° (plan section 9).

A diagnostic of residual ellipticity. Two stacked plots share the angle axis
(0-360°):

- **Pre**: radius ``hypot(U - cx, V - cy)`` about the fitted centre against
  the angle ``atan2(V - cy, U - cx)``. An uncorrected ellipse shows a
  two-cycle modulation between b and a.
- **Post**: radius ``hypot(U', V')`` of the corrected points against
  ``atan2(V', U')``; flat at ``sqrt(ab)`` for a perfect correction.

Each 5° bin shows the mean radius ± the standard deviation (``np.std``,
ddof 0) of its points; empty bins are gaps in the line. A dashed horizontal
line marks ``sqrt(ab)``. The one-line summary gives the peak-to-peak
modulation of the bin means, pre and post, over the bins with at least
:data:`P2P_MIN_COUNT` points, ± the standard error of the difference of the
two extreme bin means (``sqrt(sem_max^2 + sem_min^2)``). Points with a
non-finite coordinate are left out (and counted in the message).

By default only the points the robust fit **kept** are binned: the rejected
events (a second population in ~900 real channels) would otherwise pull the
bin means. *Include rejected points* bins every event instead; the tab always
says which set is shown. Both sets are computed up front, so the checkbox
redraws without recomputing. The kept mask comes from the session's refit
(:attr:`~uvcorr.gui.session.ChannelDetail.kept`) and is used only when that
refit reproduced the stored ellipse (``ChannelDetail.consistent``); without a
mask, or with an inconsistent one, all points are binned, the checkbox is
disabled and the tab says why.

A channel without an ellipse shows its raw profile about the median point
(upper plot only); a channel without events shows a message.

:func:`compute_angle_view` runs in a worker thread (~0.1 s for a 1M-event
channel); :meth:`AngleTab.show_data` only draws.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QLabel, QVBoxLayout, QWidget

from uvcorr.analysis import ChannelKey
from uvcorr.gui._flow_layout import FlowLayout
from uvcorr.gui._tab_data import (
    CENTRE_FITTED,
    CORR_COLOR,
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

__all__ = [
    "ANGLE_BIN_DEG",
    "N_ANGLE_BINS",
    "P2P_MIN_COUNT",
    "SET_ALL",
    "SET_KEPT",
    "AngleProfile",
    "AngleSet",
    "AngleTab",
    "AngleViewData",
    "angle_bin_index",
    "angle_profile",
    "compute_angle_view",
    "set_description",
    "summary_text",
]

N_ANGLE_BINS = 72
ANGLE_BIN_DEG = 360.0 / N_ANGLE_BINS
"""72 bins of 5°."""

BIN_CENTERS_DEG: npt.NDArray[np.float64] = (np.arange(N_ANGLE_BINS) + 0.5) * ANGLE_BIN_DEG

P2P_MIN_COUNT = 10
"""Bins with fewer points are left out of the peak-to-peak modulation (noisy means)."""

SET_KEPT = "kept"
SET_ALL = "all"

_ERROR_ALPHA = 150
_Z_ERRORS = 0
_Z_MEANS = 2
_Z_REFERENCE = 3

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
IndexArray = npt.NDArray[np.intp]


# ---------------------------------------------------------------------------
# Data (Qt-free)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AngleProfile:
    """Radius statistics in the 72 angle bins.

    Attributes:
        count: Points per bin.
        mean: Mean radius per bin (NaN for an empty bin).
        std: Standard deviation (ddof 0) per bin (NaN for an empty bin).
    """

    count: IntArray
    mean: FloatArray
    std: FloatArray

    @property
    def centers(self) -> FloatArray:
        """Bin centres in degrees (2.5, 7.5, ..., 357.5)."""
        return BIN_CENTERS_DEG

    @property
    def sem(self) -> FloatArray:
        """Standard error of each bin mean, ``std / sqrt(count)`` (NaN for an empty bin)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            sem: FloatArray = self.std / np.sqrt(self.count)
        return np.where(self.count > 0, sem, np.nan)

    @property
    def n_empty(self) -> int:
        """Number of empty bins."""
        return int(np.count_nonzero(self.count == 0))

    def p2p_bins(self, min_count: int = P2P_MIN_COUNT) -> npt.NDArray[np.bool_]:
        """The bins used for the peak-to-peak modulation (at least ``min_count`` points)."""
        mask: npt.NDArray[np.bool_] = self.count >= max(1, min_count)
        return mask

    def peak_to_peak(self, min_count: int = P2P_MIN_COUNT) -> float | None:
        """Largest minus smallest bin mean over the bins with ``min_count`` points or more.

        Returns:
            The modulation, or None if fewer than two bins qualify.
        """
        used = self.p2p_bins(min_count)
        if int(np.count_nonzero(used)) < 2:
            return None
        means = self.mean[used]
        return float(np.max(means) - np.min(means))

    def peak_to_peak_error(self, min_count: int = P2P_MIN_COUNT) -> float | None:
        """Standard error of :meth:`peak_to_peak`: ``sqrt(sem_max^2 + sem_min^2)``.

        ``sem_max`` and ``sem_min`` are the standard errors of the largest and
        smallest bin means used (the extreme bins can be sparse, so a typical
        bin's error would understate it). None if the modulation is.
        """
        used = np.flatnonzero(self.p2p_bins(min_count))
        if used.size < 2:
            return None
        means = self.mean[used]
        sem = self.sem[used]
        high, low = sem[int(np.argmax(means))], sem[int(np.argmin(means))]
        return float(math.hypot(high, low))

    def typical_sem(self, min_count: int = P2P_MIN_COUNT) -> float | None:
        """Median standard error of the bin means used for the peak-to-peak value."""
        used = self.p2p_bins(min_count)
        if not np.any(used):
            return None
        return float(np.median(self.sem[used]))


def angle_bin_index(du: npt.ArrayLike, dv: npt.ArrayLike) -> IndexArray:
    """The 5° bin (0..71) of each point's angle ``atan2(dv, du)`` in [0°, 360°).

    A point with a non-finite coordinate gets -1 (no bin); never warns.
    """
    x = np.asarray(du, dtype=np.float64)
    y = np.asarray(dv, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    all_finite = bool(np.all(finite))
    if not all_finite:
        x, y = x[finite], y[finite]
    deg = np.degrees(np.arctan2(y, x))
    deg = np.where(deg < 0.0, deg + 360.0, deg)
    binned = np.minimum((deg / ANGLE_BIN_DEG).astype(np.intp), N_ANGLE_BINS - 1)
    if all_finite:
        return binned
    index: IndexArray = np.full(finite.shape, -1, dtype=np.intp)
    index[finite] = binned
    return index


def _profile(index: IndexArray, radius: FloatArray) -> AngleProfile:
    """Binned mean and std (two-pass: no cancellation at radius >> std).

    Entries without a bin (index -1) or with a non-finite radius are skipped.
    """
    valid = (index >= 0) & np.isfinite(radius)
    if not bool(np.all(valid)):
        index, radius = index[valid], radius[valid]
    count = np.bincount(index, minlength=N_ANGLE_BINS).astype(np.int64)
    sums = np.bincount(index, weights=radius, minlength=N_ANGLE_BINS)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, sums / count, np.nan)
        dev = radius - mean[index]
        var = np.bincount(index, weights=dev * dev, minlength=N_ANGLE_BINS) / count
    std = np.where(count > 0, np.sqrt(var), np.nan)
    return AngleProfile(count=count, mean=mean.astype(np.float64), std=std.astype(np.float64))


def angle_profile(
    du: npt.ArrayLike, dv: npt.ArrayLike, mask: npt.ArrayLike | None = None
) -> AngleProfile:
    """Mean and std of ``hypot(du, dv)`` in 72 bins of the angle ``atan2(dv, du)``.

    Args:
        du, dv: Point coordinates relative to the centre (e.g. ``U - cx``,
            or the corrected ``U'`` and ``V'``).
        mask: Optional boolean selection of the points to bin.

    Returns:
        The profile; empty bins have NaN mean and std. Points with a
        non-finite coordinate are skipped (no warning).
    """
    x = np.asarray(du, dtype=np.float64)
    y = np.asarray(dv, dtype=np.float64)
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        x, y = x[keep], y[keep]
    return _profile(angle_bin_index(x, y), np.hypot(x, y))


@dataclass(frozen=True, eq=False)
class AngleSet:
    """The profiles of one point set.

    Attributes:
        kind: :data:`SET_KEPT` (the robust fit's kept points) or :data:`SET_ALL`.
        n_points: Points in the set.
        pre: Profile about the fitted centre (or the median point without an ellipse).
        post: Profile of the corrected points (None without an ellipse).
    """

    kind: str
    n_points: int
    pre: AngleProfile
    post: AngleProfile | None


@dataclass(frozen=True, eq=False)
class AngleViewData:
    """Everything the Radius vs angle tab draws for one channel (:func:`compute_angle_view`).

    Attributes:
        key: The channel.
        title: Its short name.
        centre_kind: ``"fitted"`` or ``"median"``; None without events.
        target_radius: ``sqrt(ab)`` (None without an ellipse).
        n_events: Points of the channel.
        n_rejected: Points the robust fit rejected (None without a fit mask).
        kept: Profiles of the kept points (None without a fit mask).
        all: Profiles of every point (None without events).
        message: A note for the user; empty when all is well.
        seconds: Time spent computing.
    """

    key: ChannelKey
    title: str
    centre_kind: str | None
    target_radius: float | None
    n_events: int
    n_rejected: int | None
    kept: AngleSet | None
    all: AngleSet | None
    message: str
    seconds: float

    @property
    def has_mask(self) -> bool:
        """Whether the kept/rejected split is known (the checkbox applies)."""
        return self.kept is not None

    def profile_set(self, include_rejected: bool) -> AngleSet | None:
        """The set shown: the kept points unless ``include_rejected`` or there is no mask."""
        if not include_rejected and self.kept is not None:
            return self.kept
        return self.all


def compute_angle_view(detail: ChannelDetail) -> AngleViewData:
    """Compute the Radius vs angle tab's data for a channel (Qt-free; run it in a worker thread).

    Both point sets (kept only, all) are binned, so the *Include rejected
    points* checkbox needs no recomputation.

    Args:
        detail: The channel (:meth:`~uvcorr.gui.session.UVSession.compute_detail`).

    Returns:
        The profiles. Never raises for degenerate data.
    """
    t_start = time.perf_counter()
    title = channel_title(detail.key)
    radii = channel_radii(detail)
    if radii is None:
        return AngleViewData(
            key=detail.key,
            title=title,
            centre_kind=None,
            target_radius=None,
            n_events=0,
            n_rejected=None,
            kept=None,
            all=None,
            message=detail.message or "No events on this channel (no data).",
            seconds=time.perf_counter() - t_start,
        )
    pre_index = angle_bin_index(radii.du, radii.dv)
    post_index: IndexArray | None = None
    if radii.post is not None and detail.u_corr is not None and detail.v_corr is not None:
        post_index = angle_bin_index(detail.u_corr, detail.v_corr)

    def profiles(kind: str, keep: npt.NDArray[np.bool_] | None) -> AngleSet:
        post: AngleProfile | None = None
        if keep is None:
            pre = _profile(pre_index, radii.pre)
            if post_index is not None and radii.post is not None:
                post = _profile(post_index, radii.post)
            return AngleSet(kind, detail.n_events, pre, post)
        pre = _profile(pre_index[keep], radii.pre[keep])
        if post_index is not None and radii.post is not None:
            post = _profile(post_index[keep], radii.post[keep])
        return AngleSet(kind, int(np.count_nonzero(keep)), pre, post)

    notes = [detail.message] if detail.message else []
    kept_mask = detail.kept if radii.fitted else None
    if kept_mask is not None and not detail.consistent:
        # A mask from a refit that disagrees with the stored ellipse belongs to another fit
        kept_mask = None
        notes.append(
            "The kept/rejected split is not known for the stored ellipse: all points are binned."
        )
    n_unbinned = int(np.count_nonzero(pre_index < 0))
    if n_unbinned:
        notes.append(f"{n_unbinned:,} point(s) with a non-finite coordinate are left out.")
    kept = profiles(SET_KEPT, kept_mask) if kept_mask is not None else None
    return AngleViewData(
        key=detail.key,
        title=title,
        centre_kind=radii.centre_kind,
        target_radius=radii.target_radius,
        n_events=detail.n_events,
        n_rejected=detail.n_rejected if kept_mask is not None else None,
        kept=kept,
        all=profiles(SET_ALL, None),
        message=" ".join(notes),
        seconds=time.perf_counter() - t_start,
    )


def set_description(data: AngleViewData, include_rejected: bool) -> str:
    """Which points are binned, e.g. ``"robust-kept points: 945,713 of 946,025 (312 rejected
    left out)"``."""
    shown = data.profile_set(include_rejected)
    if shown is None:
        return "no points"
    if shown.kind == SET_KEPT:
        rejected = data.n_rejected or 0
        return (
            f"robust-kept points: {shown.n_points:,} of {data.n_events:,} "
            f"({rejected:,} rejected left out)"
        )
    if data.has_mask:
        return f"all {shown.n_points:,} points, including {data.n_rejected or 0:,} rejected"
    if data.centre_kind == CENTRE_FITTED:
        return f"all {shown.n_points:,} points (the rejected points are not known)"
    return f"all {shown.n_points:,} points (no ellipse: about the median point)"


def _p2p_part(name: str, profile: AngleProfile | None, target: float | None) -> str:
    if profile is None:
        return f"{name} n/a"
    p2p = profile.peak_to_peak()
    if p2p is None:
        return f"{name} n/a (fewer than 2 bins with ≥ {P2P_MIN_COUNT} points)"
    text = f"{name} {fmt(p2p, 3)}"
    error = profile.peak_to_peak_error()
    if error is not None:
        text += f" ± {fmt(error, 2)}"
    text += " ADC"
    if target is not None and target > 0:
        text += f" ({_percent(100.0 * p2p / target)} % of √(ab))"
    return text


def _percent(value: float) -> str:
    """Two significant digits below 10 %, whole percent above (never an exponent)."""
    return f"{value:.0f}" if abs(value) >= 10.0 else fmt(value, 2)


def summary_text(data: AngleViewData, include_rejected: bool) -> str:
    """The one-line peak-to-peak summary of the bin means, pre and post."""
    shown = data.profile_set(include_rejected)
    if shown is None:
        return ""
    parts = [
        _p2p_part("pre", shown.pre, data.target_radius),
    ]
    if shown.post is not None:
        parts.append(_p2p_part("post", shown.post, data.target_radius))
    text = "Peak-to-peak of the 5° bin means: " + ", ".join(parts)
    reference = shown.post if shown.post is not None else shown.pre
    n_nonempty = N_ANGLE_BINS - reference.n_empty
    n_used = int(np.count_nonzero(reference.p2p_bins()))
    if n_used < n_nonempty:
        text += f" · {n_nonempty - n_used} bin(s) with < {P2P_MIN_COUNT} points left out"
    return text


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


class _ProfileItems:
    """The items of one profile plot, created once and refilled."""

    def __init__(self, plot: pg.PlotWidget, color: str) -> None:
        self.plot = plot
        self.errors = pg.ErrorBarItem(
            x=np.zeros(0), y=np.zeros(0), beam=1.6, pen=pg.mkPen(alpha_color(color, _ERROR_ALPHA))
        )
        self.errors.setZValue(_Z_ERRORS)
        self.means = pg.PlotDataItem(
            pen=pg.mkPen(color, width=1.5),
            symbol="o",
            symbolSize=4,
            symbolPen=None,
            symbolBrush=pg.mkBrush(color),
            connect="finite",
        )
        self.means.setZValue(_Z_MEANS)
        self.reference = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(MARKER_COLOR, width=1.5, style=Qt.PenStyle.DashLine),
            label="√(ab)",
            labelOpts={
                "position": 0.03,
                "color": MARKER_COLOR,
                "fill": (0, 0, 0, 170),
                "movable": False,
            },
        )
        self.reference.setZValue(_Z_REFERENCE)
        item = plot.getPlotItem()
        for graphics in (self.errors, self.means, self.reference):
            item.addItem(graphics)
        self.note = add_center_note(plot)
        self.clear()

    def clear(self, title: str = "", note: str = "") -> None:
        self.errors.setData(x=np.zeros(0), y=np.zeros(0), top=np.zeros(0), bottom=np.zeros(0))
        self.means.setData(x=np.zeros(0), y=np.zeros(0))
        self.errors.setVisible(False)
        self.means.setVisible(False)
        self.reference.setVisible(False)
        item = self.plot.getPlotItem()
        item.setTitle(title)
        item.enableAutoRange(axis="y", enable=False)
        item.setYRange(0.0, 1.0, padding=0.0)
        self.note.setText(note)

    def show(self, profile: AngleProfile, target: float | None, title: str) -> None:
        filled = profile.count > 0
        if not np.any(filled):
            # All-NaN data would make pyqtgraph warn on every repaint
            self.clear(title, "No points to bin")
            return
        self.plot.getPlotItem().setTitle(title)
        self.note.setText("")
        x = profile.centers
        self.means.setData(x=x, y=profile.mean)
        self.errors.setData(
            x=x[filled],
            y=profile.mean[filled],
            top=profile.std[filled],
            bottom=profile.std[filled],
        )
        self.means.setVisible(True)
        self.errors.setVisible(bool(np.any(filled)))
        if target is not None:
            self.reference.setValue(target)
        self.reference.setVisible(target is not None)
        self.plot.getPlotItem().enableAutoRange(axis="y")


def _angle_ticks() -> list[list[tuple[float, str]]]:
    major = [(float(v), f"{v}°") for v in range(0, 361, 45)]
    minor = [(float(v), "") for v in range(0, 361, 15) if v % 45]
    return [major, minor]


class AngleTab(QWidget):
    """The Radius vs angle tab (see the module docstring).

    Signals:
        display_changed(): The user toggled *Include rejected points* (for persistence).
    """

    display_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data: AngleViewData | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        controls = FlowLayout(h_spacing=8, v_spacing=2)
        self.include_rejected_check = QCheckBox("Include rejected points")
        self.include_rejected_check.setToolTip(
            "Bin every event, including the points the robust fit rejected; by default only "
            "the kept points are binned, so a second population does not pull the bin means"
        )
        self.info_label = QLabel()
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        controls.addWidget(self.include_rejected_check)
        controls.addWidget(self.info_label)
        layout.addLayout(controls)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary_label.setToolTip(
            "Largest minus smallest bin mean, over the bins with at least "
            f"{P2P_MIN_COUNT} points, ± the standard error of that difference "
            "(√(sem_max² + sem_min²) of the two extreme bins, sem = σ/√n); a flat profile "
            "gives a few times that error"
        )
        layout.addWidget(self.summary_label)

        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet(f"color: {MESSAGE_COLOR};")
        self.message_label.hide()
        layout.addWidget(self.message_label)

        self.pre_plot = make_plot("Pre", "angle (°)", "mean radius ± σ (ADC)")
        self.post_plot = make_plot("Post", "angle (°)", "mean radius ± σ (ADC)")
        self.pre_plot.setToolTip(
            "Mean radius ± σ in 5° bins of atan2(V − cV, U − cU), about the fitted centre "
            "(orange); dashed line: √(ab). Gaps are empty bins."
        )
        self.post_plot.setToolTip(
            "Mean radius ± σ of the corrected points in 5° bins of atan2(V′, U′) (blue); "
            "dashed line: √(ab). Gaps are empty bins."
        )
        for plot in (self.pre_plot, self.post_plot):
            item = plot.getPlotItem()
            item.getAxis("bottom").setTicks(_angle_ticks())
            item.setXRange(0.0, 360.0, padding=0.01)
            item.setLimits(xMin=-5.0, xMax=365.0)
            item.enableAutoRange(axis="x", enable=False)
        self.post_plot.getPlotItem().setXLink(self.pre_plot.getPlotItem())
        self._pre_items = _ProfileItems(self.pre_plot, RAW_COLOR)
        self._post_items = _ProfileItems(self.post_plot, CORR_COLOR)
        layout.addWidget(self.pre_plot, 1)
        layout.addWidget(self.post_plot, 1)

        self.include_rejected_check.toggled.connect(self._on_include_toggled)
        self._render()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def data(self) -> AngleViewData | None:
        """The channel shown, if any."""
        return self._data

    @property
    def include_rejected(self) -> bool:
        """Whether the rejected points are binned too."""
        return self.include_rejected_check.isChecked()

    def set_include_rejected(self, on: bool) -> None:
        """Bin the rejected points too, or not (redraws; emits nothing)."""
        self.include_rejected_check.blockSignals(True)
        try:
            self.include_rejected_check.setChecked(on)
        finally:
            self.include_rejected_check.blockSignals(False)
        self._render()

    @property
    def shown_set(self) -> AngleSet | None:
        """The point set drawn."""
        data = self._data
        return None if data is None else data.profile_set(self.include_rejected)

    def show_data(self, data: AngleViewData | None) -> None:
        """Draw a channel's profiles (None clears)."""
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
        """The info line (channel and point set)."""
        return str(self.info_label.text())

    def summary(self) -> str:
        """The peak-to-peak summary line."""
        return str(self.summary_label.text())

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _set_message(self, text: str) -> None:
        self.message_label.setText(text)
        self.message_label.setVisible(bool(text))

    def _render(self, message: str = "") -> None:
        data = self._data
        self.include_rejected_check.setEnabled(data is not None and data.has_mask)
        if data is None:
            self._pre_items.clear("Pre")
            self._post_items.clear("Post")
            self.info_label.setText("No channel selected")
            self.summary_label.setText("")
            self._set_message(message)
            return
        self._set_message(data.message)
        shown = data.profile_set(self.include_rejected)
        if shown is None:
            self._pre_items.clear("Pre", "No events on this channel")
            self._post_items.clear("Post", "No events on this channel")
            self.info_label.setText(data.title)
            self.summary_label.setText("")
            return
        fitted = data.centre_kind == CENTRE_FITTED
        pre_title = (
            "Pre: radius about the fitted centre"
            if fitted
            else "Raw: radius about the median point (no ellipse)"
        )
        self._pre_items.show(shown.pre, data.target_radius, pre_title)
        if shown.post is not None:
            self._post_items.show(
                shown.post, data.target_radius, "Post: radius of the corrected points"
            )
        else:
            self._post_items.clear("Post", "No ellipse fitted: no corrected points")
        self.info_label.setText(f"{data.title} · {set_description(data, self.include_rejected)}")
        self.summary_label.setText(summary_text(data, self.include_rejected))

    def _on_include_toggled(self, _checked: bool) -> None:
        self._render()
        self.display_changed.emit()
