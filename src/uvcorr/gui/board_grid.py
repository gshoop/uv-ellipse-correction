"""Board grid tab: the 47 channels of one board as small U/V plots (plan section 9).

The cells sit in an 8 x 6 grid, in one of two orders (:func:`grid_slots`):

- **RENA / channel** (:data:`LAYOUT_RENA`): RENA 0 channels 4-28, then
  RENA 1 channels 7-28 (:data:`~uvcorr.channels.ACTIVE_CHANNELS`).
- **Physical strip order** (:data:`LAYOUT_STRIP`): the 39 anodes by physical
  strip position 1..39, as the System Map's board strip shows them
  (``ElectrodeMap.channel_for_strip_position``: position 1 on the low-node
  side, so odd boards start at A39), then one empty slot, then the cathodes
  C01..C08 on the last row.

A *before/after* toggle switches every cell between

- **Before**: the raw (U, V) points with the fitted ellipse, over one common
  U/V window for the whole board (the union of the fitted ellipses' bounding
  boxes plus a margin), so the cells compare directly;
- **After**: the corrected (U', V') points with the target circle
  ``sqrt(ab)``, centred at 0 with one common half-width (the largest target
  radius plus a margin).

One bad fit must not squash the other 46 rings, so the windows leave out
fits flagged ``extreme_axis_ratio`` and fits whose centre is far from the
other fits on the same RENA (:func:`window_fits`). Those channels are still
drawn (possibly clipped); the info line names the clipped ones.

Channels without an ellipse (``too_few_events``, ``fit_failed``, not fitted)
show their raw points without an overlay: in the after view centred on their
median point, so they stay in the common centred window (the cell says so).
Channels without events show "no data". Cathodes are outlined, the title of
every cell starts with a square in its status colour (the System Map's
category colours, :mod:`uvcorr.gui.map_colors`) and a hover tooltip gives the
details; the selected channel is outlined in yellow. A click on a cell emits
:attr:`BoardGridTab.channel_activated` with its :class:`~uvcorr.analysis.ChannelKey`.

Each cell draws a deterministic random subsample of at most
:data:`DEFAULT_CELL_POINTS` points, seeded per channel as the Scatter tab
seeds its point cap (:func:`~uvcorr.gui.scatter.subsample_indices`). The
robust fit's kept/rejected split is not known here (it needs a refit per
channel), so all points share one colour.

:func:`compute_board_grid` runs in a worker thread: one stable sort of the
board's channel column, the subsamples and their corrections (~0.1 s for the
largest real board, 5.3M events). :class:`BoardGridTab` creates its 48 cells
once and only refills their items on :meth:`~BoardGridTab.show_data`, on a
layout or mode switch.
"""

from __future__ import annotations

import functools
import math
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from adc2kev.tools.geometry import ANODES_PER_BOARD, CATHODES_PER_BOARD
from PyQt6.QtCore import QPoint, QPointF, QSizeF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QTextDocument
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGraphicsItem,
    QGraphicsTextItem,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from uvcorr.analysis import ChannelKey, ChannelResult
from uvcorr.cache import BoardUV
from uvcorr.channels import ACTIVE_CHANNELS, electrode_label, electrode_map, is_cathode
from uvcorr.ellipse import EllipseParams, correct, ellipse_points
from uvcorr.gui._contrast import readable_color
from uvcorr.gui._flow_layout import FlowLayout
from uvcorr.gui._tab_data import (
    CORR_COLOR,
    MARKER_COLOR,
    MESSAGE_COLOR,
    RAW_COLOR,
    alpha_color,
    fmt,
)
from uvcorr.gui.map_colors import (
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    CATEGORY_NO_DATA,
    status_category,
)
from uvcorr.gui.scatter import subsample_indices
from uvcorr.gui.session import INFORMATIONAL_FLAGS, channel_title
from uvcorr.metrics import MAD_TO_SIGMA
from uvcorr.options import FLAG_EXTREME_AXIS_RATIO, STATUS_OK

__all__ = [
    "DEFAULT_CELL_POINTS",
    "GRID_COLUMNS",
    "GRID_ROWS",
    "LAYOUTS",
    "LAYOUT_LABELS",
    "LAYOUT_RENA",
    "LAYOUT_STRIP",
    "MODES",
    "MODE_AFTER",
    "MODE_BEFORE",
    "BoardGridData",
    "BoardGridTab",
    "GridCell",
    "board_keys",
    "compute_board_grid",
    "grid_slots",
    "window_fits",
]

LAYOUT_RENA = "rena"
LAYOUT_STRIP = "strip"
LAYOUTS: tuple[str, ...] = (LAYOUT_RENA, LAYOUT_STRIP)
LAYOUT_LABELS: dict[str, str] = {
    LAYOUT_RENA: "RENA / channel",
    LAYOUT_STRIP: "Physical strip order",
}

MODE_BEFORE = "before"
MODE_AFTER = "after"
MODES: tuple[str, ...] = (MODE_BEFORE, MODE_AFTER)

GRID_COLUMNS = 8
GRID_ROWS = 6
N_SLOTS = GRID_COLUMNS * GRID_ROWS

DEFAULT_CELL_POINTS = 2000
"""Most points drawn per cell (a deterministic subsample above it)."""

BEFORE_MARGIN = 0.12
"""Margin around the fitted ellipses of the before window, as a fraction of the largest a."""

AFTER_MARGIN = 1.2
"""Half-width of the after window in units of the largest target radius."""

WINDOW_MIN_POINTS = 50
"""Without any fitted ellipse, only channels with this many drawn points set the window."""

WINDOW_OUTLIER_K = 5.0
"""A fit whose centre is more than this many robust sigmas from its RENA's median centre
is left out of the common windows (:func:`window_fits`)."""

WINDOW_MIN_GROUP = 3
"""A RENA with fewer normal fits is compared with the whole board's fits (:func:`window_fits`)."""

WINDOW_FLOOR_FRACTION = 0.1
WINDOW_FLOOR_ADC = 10.0
"""The centre tolerance is at least ``max(WINDOW_FLOOR_FRACTION * median a, WINDOW_FLOOR_ADC)``."""

_CURVE_POINTS = 121
_POINT_ALPHA = 190
_CODE_STRIDE = 64  # channel numbers are < 36

_BORDER_ANODE = pg.mkPen("#3a3a3a", width=1)
_BORDER_CATHODE = pg.mkPen("#d8d8d8", width=2)
_BORDER_SELECTED = pg.mkPen(MARKER_COLOR, width=3)
_OVERLAY_COLOR = (242, 242, 242, 120)  # translucent: the ring's points show through
_TITLE_COLOR = "#dddddd"
_DIM_COLOR = "#9a9a9a"
_NO_DATA_TEXT = "#6e6e6e"  # dimmer than the "not fitted" grey (#777777 after contrast)
_TITLE_MARGIN = 1.0
_BLACK = QColor("#000000")

FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.intp]
Window = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Layout (Qt-free)
# ---------------------------------------------------------------------------


def board_keys(node: int, board: int) -> tuple[ChannelKey, ...]:
    """The board's 47 active channels in RENA/channel order."""
    return tuple(ChannelKey(node, board, rena, channel) for rena, channel in ACTIVE_CHANNELS)


@functools.lru_cache(maxsize=64)
def _slot_channels(board: int, layout: str) -> tuple[tuple[int, int] | None, ...]:
    if layout == LAYOUT_RENA:
        slots: list[tuple[int, int] | None] = list(ACTIVE_CHANNELS)
    elif layout == LAYOUT_STRIP:
        emap = electrode_map()
        slots = [
            emap.channel_for_strip_position(board, position)
            for position in range(1, ANODES_PER_BOARD + 1)
        ]
        slots += [None] * (-len(slots) % GRID_COLUMNS)  # the cathodes start a row
        slots += [
            emap.channel_for_cathode_label(board, index)
            for index in range(1, CATHODES_PER_BOARD + 1)
        ]
    else:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    slots += [None] * (N_SLOTS - len(slots))
    return tuple(slots)


def grid_slots(node: int, board: int, layout: str) -> tuple[ChannelKey | None, ...]:
    """The channel in each of the 48 grid slots (row-major, 8 per row), None for an empty slot.

    Raises:
        ValueError: If ``layout`` is not one of :data:`LAYOUTS`.
    """
    return tuple(
        None if slot is None else ChannelKey(node, board, *slot)
        for slot in _slot_channels(board, layout)
    )


# ---------------------------------------------------------------------------
# Data (Qt-free)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class GridCell:
    """One channel's cell.

    Attributes:
        key: The channel.
        electrode: Its electrode label (``A17``, ``C03``).
        cathode: Whether it is a cathode.
        category: Status category (:data:`uvcorr.gui.map_colors.CATEGORIES`).
        result: Its result, or None.
        n_events: Events of the channel.
        before_x, before_y: The raw subsample.
        after_x, after_y: The corrected subsample, or (without an ellipse)
            the raw subsample minus its median point.
        after_corrected: Whether ``after_*`` are corrected points.
        ellipse_x, ellipse_y: The fitted ellipse (None without one).
        target_radius: ``sqrt(ab)`` (None without an ellipse).
        tooltip: Hover text.
    """

    key: ChannelKey
    electrode: str
    cathode: bool
    category: str
    result: ChannelResult | None
    n_events: int
    before_x: FloatArray
    before_y: FloatArray
    after_x: FloatArray
    after_y: FloatArray
    after_corrected: bool
    ellipse_x: FloatArray | None
    ellipse_y: FloatArray | None
    target_radius: float | None
    tooltip: str

    @property
    def n_shown(self) -> int:
        """Points drawn."""
        return int(self.before_x.shape[0])

    @property
    def params(self) -> EllipseParams | None:
        """The ellipse drawn (None without one)."""
        if self.result is None or self.result.status != STATUS_OK:
            return None
        return self.result.params


@dataclass(frozen=True, eq=False)
class BoardGridData:
    """Everything the Board grid tab draws for one board (:func:`compute_board_grid`).

    Attributes:
        node, board: The board.
        cells: The 47 cells by channel.
        before_window: ``(u0, u1, v0, v1)``: the common raw window (square).
        after_half: Half-width of the common corrected window (centred at 0).
        n_events: Events on the board.
        points_per_cell: The subsample cap.
        seconds: Time spent computing.
        window_left_out: Fitted channels left out of the common windows
            (:func:`window_fits`); they are still drawn, possibly clipped.
        clipped_before: Those whose ellipse extends beyond ``before_window``.
        clipped_after: Those whose target circle extends beyond the after window.
    """

    node: int
    board: int
    cells: Mapping[ChannelKey, GridCell]
    before_window: Window
    after_half: float
    n_events: int
    points_per_cell: int
    seconds: float
    window_left_out: tuple[ChannelKey, ...] = ()
    clipped_before: tuple[ChannelKey, ...] = ()
    clipped_after: tuple[ChannelKey, ...] = ()

    def clipped(self, mode: str) -> tuple[ChannelKey, ...]:
        """Left-out channels whose ellipse (before) or target circle (after) is clipped."""
        return self.clipped_after if mode == MODE_AFTER else self.clipped_before

    def window(self, mode: str) -> Window:
        """The common ``(x0, x1, y0, y1)`` window of ``mode``."""
        if mode == MODE_AFTER:
            h = self.after_half
            return (-h, h, -h, h)
        return self.before_window


def _channel_groups(board_uv: BoardUV) -> dict[tuple[int, int], IndexArray]:
    """Event indices of every channel, in file order (one stable sort of the channel codes)."""
    if board_uv.n_events == 0:
        return {}
    code = board_uv.rena.astype(np.int16) * _CODE_STRIDE + board_uv.channel.astype(np.int16)
    if int(code.min()) < 0:  # pragma: no cover - the cache holds active channels only
        raise ValueError("negative RENA or channel number in the board data")
    order = np.argsort(code, kind="stable")
    counts = np.bincount(code)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return {
        (c // _CODE_STRIDE, c % _CODE_STRIDE): order[starts[c] : starts[c] + counts[c]]
        for c in np.flatnonzero(counts).tolist()
    }


def _compact(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k" if n < 100_000 else f"{n / 1000:.0f}k"
    return f"{n / 1e6:.2f}M"


def _tooltip(
    key: ChannelKey, cathode: bool, category: str, result: ChannelResult | None, n: int, shown: int
) -> str:
    emap = electrode_map()
    lines = [channel_title(key) + (" · cathode" if cathode else " · anode")]
    if not cathode:
        position = emap.physical_strip_position(key.board, key.rena, key.channel)
        lines[0] += f" · strip position {position}"
    status = CATEGORY_LABELS[category]
    if result is not None and result.flags:
        status += f" ({', '.join(result.flags)})"
    lines.append(f"Status: {status}")
    if n:
        drawn = f"all {n:,} drawn" if shown == n else f"{shown:,} of {n:,} drawn"
        lines.append(f"Events: {n:,} ({drawn})")
    if result is not None and result.status == STATUS_OK:
        parts = [f"b/a {fmt(result.axis_ratio, 4)}", f"√(ab) {fmt(result.target_radius, 5)} ADC"]
        if result.post_sigma is not None:
            parts.append(f"post σ {fmt(result.post_sigma, 4)} ADC")
        if result.rejected_fraction is not None:
            parts.append(f"rejected {fmt(100.0 * result.rejected_fraction, 3)} %")
        lines.append(" · ".join(parts))
    lines.append("Click to open the channel")
    return "\n".join(lines)


def _cell(
    key: ChannelKey,
    index: IndexArray | None,
    board_uv: BoardUV,
    result: ChannelResult | None,
    informational_flags: Collection[str],
    cap: int,
) -> GridCell:
    n = 0 if index is None else int(index.shape[0])
    if index is not None and n:
        sub = subsample_indices(n, cap, tuple(key))
        chosen = index if sub is None else index[sub]
        u = board_uv.u[chosen].astype(np.float64)
        v = board_uv.v[chosen].astype(np.float64)
    else:
        u = v = np.empty(0, dtype=np.float64)
    category = status_category(
        result.status if result is not None else None,
        result.flags if result is not None else (),
        informational_flags,
        has_data=n > 0,
    )
    params = result.params if result is not None and result.status == STATUS_OK else None
    ellipse_x = ellipse_y = None
    target: float | None = None
    if params is not None:
        ellipse_x, ellipse_y = ellipse_points(params, np.linspace(0.0, 2 * math.pi, _CURVE_POINTS))
        after_x, after_y = correct(u, v, params)
        target = params.target_radius
    elif u.size:
        after_x, after_y = u - float(np.median(u)), v - float(np.median(v))
    else:
        after_x, after_y = u, v
    cathode = is_cathode(key.board, key.rena, key.channel)
    return GridCell(
        key=key,
        electrode=electrode_label(key.board, key.rena, key.channel),
        cathode=cathode,
        category=category,
        result=result,
        n_events=n,
        before_x=u,
        before_y=v,
        after_x=after_x,
        after_y=after_y,
        after_corrected=params is not None,
        ellipse_x=ellipse_x,
        ellipse_y=ellipse_y,
        target_radius=target,
        tooltip=_tooltip(key, cathode, category, result, n, int(u.shape[0])),
    )


def _square(x0: float, x1: float, y0: float, y1: float) -> Window:
    xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = max(0.5 * (x1 - x0), 0.5 * (y1 - y0), 1.0)
    return (xc - half, xc + half, yc - half, yc + half)


def _ellipse_box(p: EllipseParams, pad: float = 0.0) -> Window:
    """Axis-aligned bounding box of an ellipse, grown by ``pad`` on every side."""
    c, s = math.cos(p.phi), math.sin(p.phi)
    hx = math.hypot(p.a * c, p.b * s)
    hy = math.hypot(p.a * s, p.b * c)
    return (p.cx - hx - pad, p.cx + hx + pad, p.cy - hy - pad, p.cy + hy + pad)


def _inside(box: Window, window: Window) -> bool:
    return (
        box[0] >= window[0] and box[1] <= window[1] and box[2] >= window[2] and box[3] <= window[3]
    )


def _reference(centres: FloatArray, floor: float) -> tuple[FloatArray, FloatArray]:
    """Median centre and per-axis tolerance ``max(K * 1.4826 * MAD, floor)``."""
    median = np.median(centres, axis=0)
    mad = np.median(np.abs(centres - median), axis=0)
    return median, np.maximum(WINDOW_OUTLIER_K * MAD_TO_SIGMA * mad, floor)


def window_fits(cells: Collection[GridCell]) -> tuple[list[GridCell], list[GridCell]]:
    """Split the fitted cells into those that set the common windows and those left out.

    A fit is left out when it carries ``extreme_axis_ratio``, or when its
    centre lies more than ``max(WINDOW_OUTLIER_K * 1.4826 * MAD,
    WINDOW_FLOOR_FRACTION * median a)`` from the median centre in U or V.
    The median and MAD are those of the other normal fits on the same RENA (a
    whole ASIC can sit at its own U/V offset, e.g. RENA 1 of N8 B26 in the
    test acquisition, ~770 ADC away), or of the whole board for a RENA with
    fewer than :data:`WINDOW_MIN_GROUP` fits. Without any fit left, every fit
    is used.

    Returns:
        ``(used, left_out)``, each sorted by channel.
    """
    fitted = [cell for cell in cells if cell.params is not None]
    if not fitted:
        return [], []
    params = {id(cell): cell.params for cell in fitted if cell.params is not None}
    floor = max(
        WINDOW_FLOOR_FRACTION * float(np.median([p.a for p in params.values()])),
        WINDOW_FLOOR_ADC,
    )
    normal = [
        cell
        for cell in fitted
        if cell.result is None or FLAG_EXTREME_AXIS_RATIO not in cell.result.flags
    ]
    left_out = [cell for cell in fitted if cell not in normal]

    def centres(group: list[GridCell]) -> FloatArray:
        return np.array([(params[id(c)].cx, params[id(c)].cy) for c in group], dtype=np.float64)

    board_reference = _reference(centres(normal), floor) if normal else None
    used: list[GridCell] = []
    for rena in sorted({cell.key.rena for cell in normal}):
        group = [cell for cell in normal if cell.key.rena == rena]
        points = centres(group)
        if len(group) >= WINDOW_MIN_GROUP:
            median, limit = _reference(points, floor)
        else:
            assert board_reference is not None
            median, limit = board_reference
        for cell, centre in zip(group, points):
            (used if np.all(np.abs(centre - median) <= limit) else left_out).append(cell)
    if not used:
        return sorted(fitted, key=lambda cell: cell.key), []
    return sorted(used, key=lambda cell: cell.key), sorted(left_out, key=lambda cell: cell.key)


def _before_window(cells: Collection[GridCell], used: list[GridCell]) -> Window:
    """The union of the used ellipses' boxes (plus a margin), or of the drawn points."""
    boxes: list[Window] = []
    fits = [cell.params for cell in used if cell.params is not None]
    if fits:
        pad = BEFORE_MARGIN * max(p.a for p in fits)
        boxes = [_ellipse_box(p, pad) for p in fits]
    else:
        drawn = [cell for cell in cells if cell.n_shown]
        enough = [cell for cell in drawn if cell.n_shown >= WINDOW_MIN_POINTS] or drawn
        for cell in enough:
            x0, x1 = (float(q) for q in np.percentile(cell.before_x, [0.5, 99.5]))
            y0, y1 = (float(q) for q in np.percentile(cell.before_y, [0.5, 99.5]))
            pad = 0.05 * max(x1 - x0, y1 - y0, 1.0)
            boxes.append((x0 - pad, x1 + pad, y0 - pad, y1 + pad))
    if not boxes:
        return (0.0, 4096.0, 0.0, 4096.0)
    return _square(
        min(b[0] for b in boxes),
        max(b[1] for b in boxes),
        min(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _after_half(cells: Collection[GridCell], used: list[GridCell]) -> float:
    """``AFTER_MARGIN`` x the largest used target radius, or the drawn points' extent."""
    targets = [cell.target_radius for cell in used if cell.target_radius is not None]
    if targets:
        return AFTER_MARGIN * max(targets)
    drawn = [cell for cell in cells if cell.n_shown]
    enough = [cell for cell in drawn if cell.n_shown >= WINDOW_MIN_POINTS] or drawn
    extents = [float(np.percentile(np.hypot(cell.after_x, cell.after_y), 99.5)) for cell in enough]
    return max(1.05 * max(extents), 1.0) if extents else 1.0


_LEFT_OUT_NOTE = (
    "Left out of the board's common window (extreme axis ratio, or a centre far from its "
    "RENA's other fits): it may be clipped"
)


def compute_board_grid(
    board_uv: BoardUV,
    results_by_key: Mapping[ChannelKey, ChannelResult | None],
    *,
    informational_flags: Collection[str] = INFORMATIONAL_FLAGS,
    points_per_cell: int = DEFAULT_CELL_POINTS,
) -> BoardGridData:
    """Compute the Board grid tab's data for a board (Qt-free; run it in a worker thread).

    Args:
        board_uv: The board's events (``UVSession.board_data``).
        results_by_key: The merged results; only the board's 47 channels are
            read, and a missing key or None means no result.
        informational_flags: Flags that do not make an ``ok`` channel
            "flagged" (as on the System Map).
        points_per_cell: The subsample cap per cell.

    Returns:
        The 47 cells and the common windows (see :func:`window_fits` for the
        fits that set them).
    """
    t_start = time.perf_counter()
    node, board = int(board_uv.node), int(board_uv.board)
    groups = _channel_groups(board_uv)
    cells = {
        key: _cell(
            key,
            groups.get((key.rena, key.channel)),
            board_uv,
            results_by_key.get(key),
            informational_flags,
            points_per_cell,
        )
        for key in board_keys(node, board)
    }
    used, left_out = window_fits(cells.values())
    before = _before_window(cells.values(), used)
    after_half = _after_half(cells.values(), used)
    after = (-after_half, after_half, -after_half, after_half)
    clipped_before: list[ChannelKey] = []
    clipped_after: list[ChannelKey] = []
    for cell in left_out:
        params = cell.params
        assert params is not None
        if not _inside(_ellipse_box(params), before):
            clipped_before.append(cell.key)
        target = cell.target_radius
        if target is not None and not _inside((-target, target, -target, target), after):
            clipped_after.append(cell.key)
        cells[cell.key] = replace(cell, tooltip=f"{cell.tooltip}\n{_LEFT_OUT_NOTE}")
    return BoardGridData(
        node=node,
        board=board,
        cells=cells,
        before_window=before,
        after_half=after_half,
        n_events=board_uv.n_events,
        points_per_cell=points_per_cell,
        seconds=time.perf_counter() - t_start,
        window_left_out=tuple(cell.key for cell in left_out),
        clipped_before=tuple(clipped_before),
        clipped_after=tuple(clipped_after),
    )


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _readable_on_black(color: str) -> str:
    return str(readable_color(color, _BLACK).name())


def _title_variants(cell: GridCell) -> tuple[str, ...]:
    """The cell title from long to short: the RENA/channel goes first, then the count.

    A channel without events gets no status marker and a dimmed name, so it
    never reads like a "not fitted" channel (both categories are greys).
    """
    rena_ch = f"R{cell.key.rena}·{cell.key.channel:02d}"
    if cell.category == CATEGORY_NO_DATA:
        name = f"<span style='color:{_NO_DATA_TEXT}'>{cell.electrode}</span>"
        count = f"<span style='color:{_NO_DATA_TEXT}'><i>no data</i></span>"
        return (
            f"{name} <span style='color:{_NO_DATA_TEXT}'>{rena_ch}</span> {count}",
            f"{name} {count}",
            name,
        )
    marker = f"<span style='color:{_readable_on_black(CATEGORY_COLORS[cell.category])}'>■</span>"
    name = f"{marker} <b>{cell.electrode}</b>"
    count = _compact(cell.n_events)
    return (
        f"{name} <span style='color:{_DIM_COLOR}'>{rena_ch} · {count}</span>",
        f"{name} <span style='color:{_DIM_COLOR}'>{count}</span>",
        name,
    )


class _TextWidths:
    """Rendered widths of rich-text titles (cached; one shared ``QTextDocument``)."""

    def __init__(self) -> None:
        self._document: QTextDocument | None = None
        self._widths: dict[str, float] = {}

    def width(self, html: str) -> float:
        width = self._widths.get(html)
        if width is None:
            document = self._document
            if document is None:
                document = self._document = QTextDocument()
                document.setDefaultFont(_title_font())
                document.setDocumentMargin(_TITLE_MARGIN)
            document.setHtml(html)
            width = float(document.idealWidth()) + 2 * _TITLE_MARGIN
            if len(self._widths) > 4096:
                self._widths.clear()
            self._widths[html] = width
        return width


_TEXT_WIDTHS = _TextWidths()


@functools.lru_cache(maxsize=1)
def _title_font() -> QFont:
    font = QFont()
    font.setPointSizeF(8.0)
    return font


class _CellTitle(pg.GraphicsWidget):
    """A one-line rich-text title that never makes its grid column wider.

    pyqtgraph's ``LabelItem`` sets its minimum width to its text, so eight
    titles forced the grid to ~1030 px and clipped the eighth column in a
    narrower tab. This item asks for no width, shows the longest title
    variant that fits (see :func:`_title_variants`) and clips the rest.
    """

    def __init__(self) -> None:
        # set before the base class runs: resizing (in its constructor) calls _fit
        self._variants: tuple[str, ...] = ()
        self._shown = ""
        self.text = ""
        """The full title (the first variant), whatever fits."""
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape)
        self.item = QGraphicsTextItem(self)
        self.item.setFont(_title_font())
        self.item.setDefaultTextColor(QColor(_TITLE_COLOR))
        document = self.item.document()
        if document is not None:
            document.setDocumentMargin(_TITLE_MARGIN)
        self._height = math.ceil(QFontMetricsF(_title_font()).height() + 2 * _TITLE_MARGIN)
        self.setMinimumSize(0.0, self._height)
        self.setMaximumHeight(self._height)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def sizeHint(  # Qt override
        self, which: Qt.SizeHint, constraint: QSizeF | None = None  # noqa: ARG002
    ) -> QSizeF:
        """No width wanted (the grid shares the width equally); the height of one line."""
        if which == Qt.SizeHint.MaximumSize:
            return QSizeF(1e6, self._height)
        return QSizeF(0.0, self._height)

    @property
    def shown(self) -> str:
        """The variant drawn."""
        return self._shown

    def set_variants(self, variants: tuple[str, ...]) -> None:
        """Set the title variants, longest first ("" or () clears)."""
        self._variants = tuple(variants)
        self.text = self._variants[0] if self._variants else ""
        self._fit()

    def resizeEvent(self, ev: Any) -> None:
        """Pick the variant that fits the new width."""
        super().resizeEvent(ev)
        self._fit()

    def _fit(self) -> None:
        if not hasattr(self, "item"):
            return  # still in the base class constructor
        width = float(self.size().width())
        chosen = ""
        for html in self._variants:
            chosen = html
            if _TEXT_WIDTHS.width(html) <= width:
                break
        if chosen != self._shown:
            self.item.setHtml(chosen)
            self._shown = chosen
        text_width = _TEXT_WIDTHS.width(chosen) if chosen else 0.0
        self.item.setPos(max(0.0, 0.5 * (width - text_width)), 0.0)


class _CellItems:
    """The graphics of one grid slot, created once and refilled."""

    def __init__(self, graphics: pg.GraphicsLayoutWidget, row: int, col: int) -> None:
        self.key: ChannelKey | None = None
        self.cathode = False
        self.layout = graphics.addLayout(row=row, col=col)
        self.layout.setContentsMargins(1, 1, 1, 1)
        self.layout.setSpacing(0)
        self.label = _CellTitle()
        self.layout.addItem(self.label, row=0, col=0)
        self.vb = self.layout.addViewBox(
            row=1, col=0, lockAspect=True, enableMouse=False, enableMenu=False
        )
        self.vb.setMinimumSize(10.0, 10.0)
        self.vb.setBorder(None)
        self.scatter = pg.ScatterPlotItem(pen=None, size=2, pxMode=True)
        self.curve = pg.PlotCurveItem(pen=pg.mkPen(_OVERLAY_COLOR, width=1))
        self.note = pg.TextItem("", anchor=(0.5, 0.5))
        self.vb.addItem(self.scatter)
        self.vb.addItem(self.curve)
        self.vb.addItem(self.note, ignoreBounds=True)

    def set_border(self, selected: bool) -> None:
        if self.key is None:
            self.vb.setBorder(None)  # an empty slot has no frame
        elif selected:
            self.vb.setBorder(_BORDER_SELECTED)
        elif self.cathode:
            self.vb.setBorder(_BORDER_CATHODE)
        else:
            self.vb.setBorder(_BORDER_ANODE)

    def show(
        self,
        cell: GridCell | None,
        mode: str,
        window: Window,
        selected: bool,
        clipped: bool = False,
    ) -> None:
        self.key = None if cell is None else cell.key
        self.cathode = cell is not None and cell.cathode
        self.set_border(selected)
        x0, x1, y0, y1 = window
        self.vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.0)
        if cell is None:
            self.label.set_variants(())
            self.layout.setToolTip("")
            self.scatter.clear()
            self.curve.setVisible(False)
            self.note.setText("")
            return
        self.label.set_variants(_title_variants(cell))
        self.layout.setToolTip(cell.tooltip)
        after = mode == MODE_AFTER
        x, y = (cell.after_x, cell.after_y) if after else (cell.before_x, cell.before_y)
        corrected = after and cell.after_corrected
        color = CORR_COLOR if corrected else RAW_COLOR
        self.scatter.setData(x=x, y=y, pen=None, brush=pg.mkBrush(alpha_color(color, _POINT_ALPHA)))
        if after and cell.target_radius is not None:
            t = np.linspace(0.0, 2 * math.pi, _CURVE_POINTS)
            self.curve.setData(cell.target_radius * np.cos(t), cell.target_radius * np.sin(t))
            self.curve.setVisible(True)
        elif not after and cell.ellipse_x is not None and cell.ellipse_y is not None:
            self.curve.setData(cell.ellipse_x, cell.ellipse_y)
            self.curve.setVisible(True)
        else:
            self.curve.setVisible(False)
        note = ""
        note_color = _readable_on_black(CATEGORY_COLORS[cell.category])
        if cell.category == CATEGORY_NO_DATA:
            note, note_color = "No data", _NO_DATA_TEXT
        elif cell.params is None:
            note = CATEGORY_LABELS[cell.category]
            if after and cell.n_events:
                note += "\n(raw − median)"
        elif clipped:
            note, note_color = "outside\nthe window", _readable_on_black(MESSAGE_COLOR)
        self.note.setText(note, color=note_color)
        self.note.setPos(0.5 * (x0 + x1), 0.5 * (y0 + y1))


class BoardGridTab(QWidget):
    """The Board grid tab (see the module docstring).

    Signals:
        channel_activated(ChannelKey): A cell was clicked.
        display_changed(): The user changed the layout or the before/after
            mode (for persistence).
    """

    channel_activated = pyqtSignal(object)
    display_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data: BoardGridData | None = None
        self._selected: ChannelKey | None = None
        self.last_render_seconds = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        controls = FlowLayout(h_spacing=8, v_spacing=2)
        self.layout_combo = QComboBox()
        for key in LAYOUTS:
            self.layout_combo.addItem(LAYOUT_LABELS[key], key)
        self.layout_combo.setToolTip(
            "Cell order: RENA 0 channels 4-28 then RENA 1 channels 7-28, or the anodes in "
            "physical strip order (as the System Map's board strip) then the cathodes C01-C08"
        )
        self.before_radio = QRadioButton("Before (raw U, V)")
        self.after_radio = QRadioButton("After (corrected U′, V′)")
        self.before_radio.setToolTip(
            "Raw points with the fitted ellipse, over one U/V window common to the board"
        )
        self.after_radio.setToolTip(
            "Corrected points with the target circle √(ab), centred at 0 with one common "
            "half-width; channels without an ellipse show their raw points minus the median"
        )
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.before_radio)
        self.mode_group.addButton(self.after_radio)
        self.before_radio.setChecked(True)
        mode_box = QWidget()
        mode_layout = QHBoxLayout(mode_box)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(6)
        mode_layout.addWidget(self.before_radio)
        mode_layout.addWidget(self.after_radio)
        layout_box = QWidget()
        layout_row = QHBoxLayout(layout_box)
        layout_row.setContentsMargins(0, 0, 0, 0)
        layout_row.setSpacing(4)
        layout_row.addWidget(QLabel("Order:"))
        layout_row.addWidget(self.layout_combo)
        self.info_label = QLabel()
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        controls.addWidget(layout_box)
        controls.addWidget(mode_box)
        controls.addWidget(self.info_label)
        layout.addLayout(controls)

        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet(f"color: {MESSAGE_COLOR};")
        self.message_label.hide()
        layout.addWidget(self.message_label)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.ci.setContentsMargins(2, 2, 2, 2)
        self.graphics.ci.setSpacing(3)
        self._cells = [
            _CellItems(self.graphics, index // GRID_COLUMNS, index % GRID_COLUMNS)
            for index in range(N_SLOTS)
        ]
        layout.addWidget(self.graphics, 1)

        scene = self.graphics.scene()
        assert scene is not None
        scene.sigMouseClicked.connect(self._on_scene_clicked)
        self.layout_combo.currentIndexChanged.connect(self._on_display_changed)
        self.after_radio.toggled.connect(self._on_display_changed)
        self._render()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def data(self) -> BoardGridData | None:
        """The board shown, if any."""
        return self._data

    @property
    def board(self) -> tuple[int, int] | None:
        """``(node, board)`` shown, if any."""
        data = self._data
        return None if data is None else (data.node, data.board)

    @property
    def grid_layout(self) -> str:
        """The cell order (:data:`LAYOUT_RENA` or :data:`LAYOUT_STRIP`)."""
        return str(self.layout_combo.currentData())

    @property
    def mode(self) -> str:
        """:data:`MODE_BEFORE` or :data:`MODE_AFTER`."""
        return MODE_AFTER if self.after_radio.isChecked() else MODE_BEFORE

    @property
    def selected(self) -> ChannelKey | None:
        """The channel outlined as selected."""
        return self._selected

    def set_grid_layout(self, layout: str) -> None:
        """Set the cell order (redraws; emits nothing).

        Raises:
            ValueError: If ``layout`` is not one of :data:`LAYOUTS`.
        """
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
        self.layout_combo.blockSignals(True)
        try:
            self.layout_combo.setCurrentIndex(LAYOUTS.index(layout))
        finally:
            self.layout_combo.blockSignals(False)
        self._render()

    def set_mode(self, mode: str) -> None:
        """Show the raw (before) or corrected (after) points (redraws; emits nothing).

        Raises:
            ValueError: If ``mode`` is not one of :data:`MODES`.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        for button in (self.before_radio, self.after_radio):
            button.blockSignals(True)
        try:
            (self.after_radio if mode == MODE_AFTER else self.before_radio).setChecked(True)
        finally:
            for button in (self.before_radio, self.after_radio):
                button.blockSignals(False)
        self._render()

    def set_selected(self, key: ChannelKey | tuple[int, int, int, int] | None) -> None:
        """Outline ``key`` as the selected channel (None: no outline; emits nothing)."""
        self._selected = None if key is None else ChannelKey(*(int(k) for k in key))
        for items in self._cells:
            items.set_border(items.key is not None and items.key == self._selected)

    def show_data(self, data: BoardGridData | None) -> None:
        """Draw a board (None clears)."""
        self._data = data
        self._render()

    def show_loading(self, text: str) -> None:
        """Note that a board is loading (the previous one stays drawn meanwhile)."""
        self.info_label.setText(f"Loading {text} …")

    def clear(self, message: str = "") -> None:
        """Show empty cells, with an optional message."""
        self._data = None
        self._render(message)

    def message(self) -> str:
        """The message shown above the grid (empty if hidden)."""
        return self.message_label.text() if not self.message_label.isHidden() else ""

    def info_text(self) -> str:
        """The info line."""
        return str(self.info_label.text())

    def slot_keys(self) -> tuple[ChannelKey | None, ...]:
        """The channel drawn in each of the 48 slots (row-major)."""
        return tuple(items.key for items in self._cells)

    def cell_view_rect(self, key: ChannelKey | tuple[int, int, int, int]) -> tuple[QPoint, QPoint]:
        """Top-left and bottom-right corners of a channel's cell in viewport coordinates.

        Raises:
            KeyError: If the channel is not shown.
        """
        wanted = ChannelKey(*(int(k) for k in key))
        for items in self._cells:
            if items.key == wanted:
                rect = items.layout.sceneBoundingRect()
                return (
                    self.graphics.mapFromScene(rect.topLeft()),
                    self.graphics.mapFromScene(rect.bottomRight()),
                )
        raise KeyError(f"{wanted} is not shown in the board grid")

    def cell_key_at(self, point: QPoint) -> ChannelKey | None:
        """The channel whose cell contains a viewport point, if any."""
        return self._key_at_scene(self.graphics.mapToScene(point))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _set_message(self, text: str) -> None:
        self.message_label.setText(text)
        self.message_label.setVisible(bool(text))

    def _render(self, message: str = "") -> None:
        t_start = time.perf_counter()
        data = self._data
        mode = self.mode
        if data is None:
            for items in self._cells:
                items.show(None, mode, (0.0, 1.0, 0.0, 1.0), False)
            self.info_label.setText("No board selected")
            self._set_message(message)
            self.last_render_seconds = time.perf_counter() - t_start
            return
        self._set_message(message)
        window = data.window(mode)
        slots = grid_slots(data.node, data.board, self.grid_layout)
        clipped_keys = set(data.clipped(mode))
        for key, items in zip(slots, self._cells):
            cell = None if key is None else data.cells.get(key)
            selected = key is not None and key == self._selected
            items.show(cell, mode, window, selected, key in clipped_keys)
        parts = [f"N{data.node} B{data.board}", f"{data.n_events:,} events"]
        parts.append(f"≤ {data.points_per_cell:,} points per cell")
        x0, x1, y0, y1 = window
        if mode == MODE_AFTER:
            parts.append(f"common window U′, V′ ± {fmt(0.5 * (x1 - x0), 4)} ADC")
        else:
            parts.append(
                f"common window U {fmt(x0, 4)}–{fmt(x1, 4)}, V {fmt(y0, 4)}–{fmt(y1, 4)} ADC"
            )
        left_out = data.window_left_out
        if left_out:
            text = f"{len(left_out)} fit{'s' if len(left_out) > 1 else ''} left out of it"
            clipped = data.clipped(mode)
            if clipped:
                names = ", ".join(data.cells[key].electrode for key in clipped[:4])
                more = f" +{len(clipped) - 4}" if len(clipped) > 4 else ""
                text += f" ({len(clipped)} clipped: {names}{more})"
            parts.append(text)
        self.info_label.setText(" · ".join(parts))
        self.last_render_seconds = time.perf_counter() - t_start

    def _key_at_scene(self, pos: QPointF) -> ChannelKey | None:
        for items in self._cells:
            if items.key is not None and items.layout.sceneBoundingRect().contains(pos):
                return items.key
        return None

    def _on_scene_clicked(self, event: Any) -> None:
        """Emit ``channel_activated`` for a left click on a cell (pyqtgraph ``MouseClickEvent``)."""
        if event.button() != Qt.MouseButton.LeftButton or event.double():
            return
        key = self._key_at_scene(event.scenePos())
        if key is not None:
            event.accept()
            self.channel_activated.emit(key)

    def _on_display_changed(self, *_args: object) -> None:
        self._render()
        self.display_changed.emit()
