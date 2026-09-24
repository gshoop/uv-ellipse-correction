"""Shared pieces of the Radial, Radius vs angle and Board grid tabs (plan section 9).

Each of these tabs is split the same way as the Scatter tab: a pure, Qt-free
``compute_*`` function builds a frozen data record in a worker thread (a
channel has up to ~1M points), and the widget's ``show_data`` only draws it.
This module holds what the compute functions share:

- :func:`channel_radii`: the radii of a :class:`~uvcorr.gui.session.ChannelDetail`
  about the fitted ellipse centre (pre) and of the corrected points (post),
  exactly as :func:`~uvcorr.analysis.analyze_channel` computes them for the
  stored ``pre_*`` / ``post_*`` columns. A channel without an ellipse (no
  result, ``too_few_events``, ``fit_failed``) gets its radii about the
  median point instead (:data:`CENTRE_MEDIAN`), so its ring can still be
  inspected; the tabs label it as such.
- number formatting (``%.6g`` as in the CSV and the Fit Inspector);
- the plot factory and colours of the tabs (the Scatter tab's orange for
  raw/pre and blue for corrected/post, on pyqtgraph's black background).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg

from uvcorr.ellipse import radii_about_center
from uvcorr.gui.scatter import CORR_COLOR, CURVE_COLOR, MARKER_COLOR, MESSAGE_COLOR, RAW_COLOR
from uvcorr.gui.session import ChannelDetail

__all__ = [
    "CENTRE_FITTED",
    "CENTRE_MEDIAN",
    "CORR_COLOR",
    "CURVE_COLOR",
    "MARKER_COLOR",
    "MESSAGE_COLOR",
    "RAW_COLOR",
    "ChannelRadii",
    "add_center_note",
    "alpha_color",
    "channel_radii",
    "fmt",
    "make_plot",
]

CENTRE_FITTED = "fitted"
"""Radii measured about the fitted ellipse centre (the stored pre-correction radii)."""

CENTRE_MEDIAN = "median"
"""Radii measured about the median (U, V) point (a channel without an ellipse)."""

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class ChannelRadii:
    """The pre and post radii of one channel, all points in file order.

    Attributes:
        centre_kind: :data:`CENTRE_FITTED` or :data:`CENTRE_MEDIAN`.
        cx, cy: The centre the pre radii are measured about.
        du, dv: ``U - cx`` and ``V - cy``.
        pre: ``hypot(du, dv)``.
        post: ``hypot(U', V')`` of the corrected points (None without an ellipse).
        target_radius: ``sqrt(ab)`` (None without an ellipse or when not finite).

    Non-finite points give non-finite radii (the compute functions skip them).
    """

    centre_kind: str
    cx: float
    cy: float
    du: FloatArray
    dv: FloatArray
    pre: FloatArray
    post: FloatArray | None
    target_radius: float | None

    @property
    def fitted(self) -> bool:
        """Whether the radii are about the fitted centre (an ellipse exists)."""
        return self.centre_kind == CENTRE_FITTED


def channel_radii(detail: ChannelDetail) -> ChannelRadii | None:
    """Pre and post radii of a channel (None when it has no events).

    With an ellipse the pre radii are :func:`~uvcorr.ellipse.radii_about_center`
    and the post radii ``hypot(U', V')`` of the detail's corrected points: the
    same arrays :func:`~uvcorr.analysis.analyze_channel` measures. Without one
    the pre radii are about the median point and there are no post radii.
    """
    if detail.n_events == 0:
        return None
    params = detail.params
    target: float | None
    if params is not None and detail.u_corr is not None and detail.v_corr is not None:
        cx, cy = params.cx, params.cy
        pre = radii_about_center(detail.u, detail.v, params)
        post: FloatArray | None = np.hypot(detail.u_corr, detail.v_corr)
        kind, target = CENTRE_FITTED, params.target_radius
        if not math.isfinite(target):
            target = None
    else:
        finite = np.isfinite(detail.u) & np.isfinite(detail.v)
        if np.any(finite):
            cx, cy = float(np.median(detail.u[finite])), float(np.median(detail.v[finite]))
        else:
            cx = cy = math.nan
        with np.errstate(invalid="ignore"):
            pre = np.hypot(detail.u - cx, detail.v - cy)
        post, kind, target = None, CENTRE_MEDIAN, None
    with np.errstate(invalid="ignore"):
        du, dv = detail.u - cx, detail.v - cy
    return ChannelRadii(
        centre_kind=kind,
        cx=cx,
        cy=cy,
        du=du,
        dv=dv,
        pre=pre,
        post=post,
        target_radius=target,
    )


def fmt(value: float | None, digits: int = 6) -> str:
    """``value`` with ``digits`` significant digits (``%.6g``); ``"n/a"`` for None or NaN."""
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}g}"


def alpha_color(color: str, alpha: int) -> tuple[int, int, int, int]:
    """``color`` as an RGBA tuple with the given alpha (0-255)."""
    r, g, b, _ = pg.mkColor(color).getRgb()
    return (r, g, b, alpha)


def make_plot(title: str, x_label: str, y_label: str) -> pg.PlotWidget:
    """A plot in the Scatter tab's style: black background and a faint grid.

    No legend: the titles, text boxes and tooltips name the items (a pyqtgraph
    legend would show hidden items with an eye icon).
    """
    plot = pg.PlotWidget()
    item = plot.getPlotItem()
    item.setTitle(title)
    item.setLabel("bottom", x_label)
    item.setLabel("left", y_label)
    item.showGrid(x=True, y=True, alpha=0.2)
    return plot


def add_center_note(plot: pg.PlotWidget) -> pg.LabelItem:
    """A text label pinned to the centre of a plot's view (for "no data" notes)."""
    note = pg.LabelItem("", size="10pt", color=MESSAGE_COLOR)
    note.setParentItem(plot.getPlotItem().getViewBox())
    note.anchor(itemPos=(0.5, 0.5), parentPos=(0.5, 0.5))
    return note
