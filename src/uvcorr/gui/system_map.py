"""Qt widgets for the System Map dock.

Forked from adc2kev 2.2.7 ``gui/system_map.py``. Keep the structure close to
it so fixes can be carried across. Changes from adc2kev:

- The map is fed :class:`~uvcorr.gui._system_map_model.ChannelView` records
  (:meth:`SystemMapWidget.set_state`) instead of adc2kev's calibration
  dictionaries, and addresses channels by
  :class:`~uvcorr.gui._system_map_model.ChannelAddress` (a
  ``(node, board, rena, channel)`` ``NamedTuple``) instead of ``ChannelKey``.
- Cells are painted with the fill colour the model computed, so the map has
  colour modes: status categories, or a metric on the viridis colormap. A
  mode selector sits next to the Anodes/Cathodes toggle, the legend follows
  the mode, and a :class:`~uvcorr.gui.map_colors.ColorBarWidget` shows the
  metric's colour limits while a metric mode is active. Switching the mode
  recolours the current model (:func:`recolor_system_map`) and re-reads
  nothing.
- :meth:`SystemMapWidget.set_selection` highlights a board and optionally a
  channel without emitting, next to adc2kev's ``set_current_channel``.
- The top row is a :class:`~uvcorr.gui._flow_layout.FlowLayout`, so the wider
  legend wraps instead of setting the dock's minimum width.
- Strip labels are drawn in a dark or light colour depending on the fill
  (for a label spilling over its neighbours, the one that reads best on all
  of them), since the dark end of viridis would hide adc2kev's dark labels.
- Cells whose result comes from an override fit carry a small white
  "dog-ear" in their top-right corner (in the strip, and in the grids when
  cells are at least 6 px wide); the legend explains it.
- :attr:`SystemMapWidget.view_changed` reports the user's Anodes/Cathodes
  choice, and :meth:`SystemMapWidget.update_views` applies a re-fit of a few
  channels without a full rebuild.
- The warning line also counts results on channels that are not electrodes.
- **Bug fix to carry back to adc2kev:** hit-testing follows the painting
  rule. An aliased ``QPainter.fillRect(QRectF)`` fills the pixels whose
  centre lies in ``(left, right]``, but adc2kev's ``tile_at`` / ``cell_at``
  (grid) and ``cell_at`` (strip) tested the pixel's top-left corner, so a
  click on 12-16 % of the painted pixels of the ~4 px anode cells (about 1 %
  for cathodes, 3 % in the strip) selected the neighbouring cell. The fix is
  :func:`_span_index` (``ceil((pixel + 0.5 - origin) / pitch) - 1``). Only a
  centre exactly on a cell edge can still go either way, as Qt then rounds
  a sum computed in a different order.

Three plain-``QPainter`` widgets consume the model:

- :class:`PanelGridWidget` draws one detector panel (five node columns by
  sixteen board rows); each tile is split into 39 anode or 8 cathode cells.
- :class:`BoardStripWidget` magnifies one board as a single row of cells and
  handles keyboard stepping.
- :class:`SystemMapWidget` composes the view toggle, colour mode selector,
  legend, colour bar, summary line, both panel grids and the strip, and
  exposes the selection signals the main window wires up.

Rendering uses ``QPainter`` in ``paintEvent`` (no pyqtgraph, no
``QGraphicsView``): cells are rectangles computed from the widget size, so
hit-testing is exact arithmetic and ``qtbot`` can drive every interaction
offscreen. The public ``cell_rect`` / ``cell_at`` helpers expose that
geometry so tests can click at a computed cell centre.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from adc2kev.tools.geometry import (
    ACTIVE_BOARDS,
    ANODES_PER_BOARD,
    CATHODES_PER_BOARD,
    PANEL_NODES,
)
from PyQt6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QHelpEvent,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QRadioButton,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from uvcorr.gui._flow_layout import FlowLayout
from uvcorr.gui._system_map_model import (
    GRID_BOARDS,
    GRID_NODES,
    NO_DATA_SUMMARY,
    BoardCells,
    ChannelAddress,
    ChannelKeyT,
    ChannelTuple,
    ChannelView,
    MapCell,
    SystemMapModel,
    as_address,
    build_system_map,
    format_board_summary,
    format_summary,
    is_in_grid,
    recolor_system_map,
    update_system_map,
)
from uvcorr.gui.map_colors import (
    COLOR_MODES,
    METRIC_SPECS,
    MODE_LABELS,
    MODE_STATUS,
    NO_DATA_COLOR,
    NOT_FITTED_COLOR,
    ColorBarWidget,
    check_color_mode,
    clip_range_text,
    is_metric_mode,
    label_colors_for_fills,
    legend_entries,
    text_color_for,
)

__all__ = [
    "BACKGROUND_COLOR",
    "CELL_SEPARATOR",
    "LABEL_COLOR",
    "NO_DATA_COLOR",
    "NO_DATA_SUMMARY",
    "OVERRIDE_LEGEND_LABEL",
    "OVERRIDE_TICK_COLOR",
    "SELECTED_BOARD_OUTLINE",
    "SELECTED_CHANNEL_OUTLINE",
    "VIEW_ANODES",
    "VIEW_CATHODES",
    "WARNING_COLOR",
    "BoardStripWidget",
    "PanelGridWidget",
    "SystemMapWidget",
]

logger = logging.getLogger(__name__)

# Dark-theme palette matching the pyqtgraph plots. The cell fills come from
# the model (see uvcorr.gui.map_colors).
BACKGROUND_COLOR = "#1e1e1e"
LABEL_COLOR = "#cccccc"
SELECTED_BOARD_OUTLINE = "#ffffff"
SELECTED_CHANNEL_OUTLINE = "#00ffff"
CELL_SEPARATOR = "#1e1e1e"
WARNING_COLOR = "#f39c12"

# Parsed once: paintEvent fills up to 3120 cells per grid.
_NO_DATA_QCOLOR = QColor(NO_DATA_COLOR)
_BACKGROUND_QCOLOR = QColor(BACKGROUND_COLOR)
_LABEL_QCOLOR = QColor(LABEL_COLOR)
_SEPARATOR_QCOLOR = QColor(CELL_SEPARATOR)

VIEW_ANODES = "anodes"
VIEW_CATHODES = "cathodes"
_VIEWS = (VIEW_ANODES, VIEW_CATHODES)
_VIEW_KIND = {VIEW_ANODES: "anode", VIEW_CATHODES: "cathode"}
_VIEW_CELLS = {VIEW_ANODES: ANODES_PER_BOARD, VIEW_CATHODES: CATHODES_PER_BOARD}

# Separator lines between cells are only worth drawing above this cell width.
_MIN_SEPARATOR_CELL_WIDTH = 6.0
_OUTLINE_WIDTH = 2
_LABEL_POINT_SIZE = 8

# Override tick: a right triangle in the top-right corner of a cell, sized
# from the cell and drawn only when the cell is at least this wide.
_MIN_TICK_CELL_WIDTH = 6.0
_TICK_FRACTION = 0.45
_TICK_MIN_SIZE = 3.0
_TICK_MAX_SIZE = 7.0
_TICK_EDGE_MIN_SIZE = 5.0
OVERRIDE_TICK_COLOR = "#ffffff"
_OVERRIDE_TICK_QCOLOR = QColor(OVERRIDE_TICK_COLOR)

# Legend swatch groups: one set for the status mode, one for the metric modes.
_LEGEND_STATUS = "status"
_LEGEND_METRIC = "metric"
OVERRIDE_LEGEND_LABEL = "Override"

# Unplaced result keys listed by name in the warning line.
_MAX_LISTED_UNPLACED = 5


def _check_view(view: str) -> None:
    if view not in _VIEWS:
        raise ValueError(f"view must be one of {_VIEWS}, got {view!r}")


def _span_index(pixel: int, origin: float, pitch: float) -> tuple[int, float]:
    """Locate a pixel in a row of equal spans the way ``QPainter`` paints them.

    An aliased ``fillRect(QRectF)`` rounds both edges of the rectangle
    (``qRound``), so it fills the pixels whose centre ``c = pixel + 0.5``
    satisfies ``left < c <= right``. adc2kev hit-tested ``floor(pixel -
    origin) / pitch`` instead, i.e. the pixel's top-left corner, which put
    12-16 % of the painted pixels of a 4 px anode cell into its neighbour;
    this is the fix to carry back.

    Args:
        pixel: Pixel coordinate along the row.
        origin: Coordinate where span 0 starts.
        pitch: Span width (plus any gap that follows it).

    Returns:
        ``(index, offset)``: the span whose ``(start, start + pitch]`` holds
        the centre (negative before the origin) and the centre's distance
        from that span's start, in ``(0, pitch]``.
    """
    centre = pixel + 0.5 - origin
    index = math.ceil(centre / pitch) - 1
    return index, centre - index * pitch


def tick_size(cell_w: float, h: float) -> float:
    """Side of the override tick in a ``cell_w`` x ``h`` cell, in pixels."""
    return min(max(min(cell_w, h) * _TICK_FRACTION, _TICK_MIN_SIZE), _TICK_MAX_SIZE)


def _draw_override_tick(painter: QPainter, right: float, top: float, size: float) -> None:
    """Draw the override "dog-ear" in the top-right corner of a cell.

    A white triangle, edged along its hypotenuse in the background colour
    once it is big enough: the white reads on dark fills and the dark edge on
    light ones (a single contrasting colour would be the background's own
    dark on light fills and look like a missing corner).
    """
    painter.save()
    try:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_OVERRIDE_TICK_QCOLOR)
        painter.drawPolygon(
            QPolygonF([QPointF(right - size, top), QPointF(right, top), QPointF(right, top + size)])
        )
        if size >= _TICK_EDGE_MIN_SIZE:
            pen = QPen(_BACKGROUND_QCOLOR)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawLine(QPointF(right - size, top), QPointF(right, top + size))
    finally:
        painter.restore()


def _draw_cell_row(
    painter: QPainter,
    cells: Sequence[MapCell],
    x0: float,
    y: float,
    cell_w: float,
    h: float,
    fill: QColor | None = None,
) -> None:
    """Fill ``cells`` left to right from ``x0`` and separate them when wide enough.

    ``fill`` overrides the cells' own colours (used to grey out a board without
    data while keeping its cell geometry and labels). Cells with an override
    result get a corner tick when they are wide enough.
    """
    for index, cell in enumerate(cells):
        painter.fillRect(
            QRectF(x0 + index * cell_w, y, cell_w, h),
            fill if fill is not None else cell.fill,
        )
    if cell_w >= _MIN_SEPARATOR_CELL_WIDTH:
        pen = QPen(_SEPARATOR_QCOLOR)
        pen.setWidth(1)
        painter.setPen(pen)
        for index in range(1, len(cells)):
            x = x0 + index * cell_w
            painter.drawLine(QPointF(x, y), QPointF(x, y + h))
    if fill is None and cell_w >= _MIN_TICK_CELL_WIDTH:
        size = tick_size(cell_w, h)
        for index, cell in enumerate(cells):
            if cell.is_override:
                _draw_override_tick(painter, x0 + (index + 1) * cell_w, y, size)


def _draw_outline(painter: QPainter, rect: QRectF, color: str) -> None:
    """Draw a 2 px outline just inside ``rect``."""
    pen = QPen(QColor(color))
    pen.setWidth(_OUTLINE_WIDTH)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(rect.adjusted(1, 1, -1, -1))


def _draw_board_outline(painter: QPainter, tile: QRectF) -> None:
    """Draw the 2 px selected-board outline in the gap around ``tile``.

    Tiles are only about 10 px tall at dock size, so an outline drawn inside
    the tile would hide most of the board's colours; the 2 px tile gap is
    exactly wide enough to hold it instead.
    """
    pen = QPen(QColor(SELECTED_BOARD_OUTLINE))
    pen.setWidth(_OUTLINE_WIDTH)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(tile.adjusted(-1, -1, 1, 1))


def _shrink_font(widget: QWidget) -> None:
    """Use a compact font for cell and axis labels."""
    font = widget.font()
    font.setPointSize(_LABEL_POINT_SIZE)
    widget.setFont(font)


def _hbox_group(*widgets: QWidget, spacing: int = 4) -> QWidget:
    """Put ``widgets`` on one line so the flow layout never splits them."""
    group = QWidget()
    layout = QHBoxLayout(group)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for widget in widgets:
        layout.addWidget(widget)
    return group


def _legend_swatch(label: str, color: str) -> QWidget:
    """A 12 px colour swatch followed by its label."""
    swatch = QLabel()
    swatch.setFixedSize(12, 12)
    swatch.setStyleSheet(f"background-color: {color}; border: 1px solid #888888;")
    return _hbox_group(swatch, QLabel(label))


class _OverrideSwatch(QWidget):
    """12 px legend swatch: a grey cell with the override corner tick."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(12, 12)

    def paintEvent(self, a0: QPaintEvent | None) -> None:  # noqa: ARG002
        """Paint the grey cell, its frame and the tick."""
        painter = QPainter(self)
        try:
            fill = QColor(NOT_FITTED_COLOR)
            painter.fillRect(self.rect(), fill)
            _draw_override_tick(painter, 12.0, 0.0, 6.0)
            painter.setPen(QPen(QColor("#888888")))
            painter.drawRect(QRectF(0.5, 0.5, 11.0, 11.0))
        finally:
            painter.end()


class PanelGridWidget(QWidget):
    """One detector panel: node columns by board rows, each tile split into cells.

    Column labels ``N1..N5`` (or ``N6..N10``) run along the top and row labels
    ``B15..B30`` down the left. In the anode view every tile holds 39 cells in
    physical strip order; in the cathode view it holds 8. Tiles for boards
    without data are drawn as uniform dark blocks and have no cells.

    Signals:
        cell_clicked(ChannelAddress): Left click on a cell of a board with data.
        board_clicked(int, int): Left click on a tile without data
            (``node, board``).
        context_requested(int, int, object, object): Right click on any tile
            (``node, board, ChannelAddress | None, global QPoint``).
    """

    LEFT_MARGIN = 34
    TOP_MARGIN = 18
    RIGHT_MARGIN = 4
    BOTTOM_MARGIN = 4
    TILE_GAP = 2

    cell_clicked = pyqtSignal(object)
    board_clicked = pyqtSignal(int, int)
    context_requested = pyqtSignal(int, int, object, object)

    def __init__(self, panel: int, parent: QWidget | None = None) -> None:
        """Create the grid for ``panel`` (1 or 2, see ``geometry.PANEL_NODES``)."""
        super().__init__(parent)
        if panel not in PANEL_NODES:
            raise ValueError(f"panel must be one of {tuple(PANEL_NODES)}, got {panel!r}")
        self._panel = panel
        self._nodes: tuple[int, ...] = PANEL_NODES[panel]
        self._boards: tuple[int, ...] = ACTIVE_BOARDS
        self._model: SystemMapModel | None = None
        self._view = VIEW_ANODES
        self._selected_board: tuple[int, int] | None = None
        self._selected_channel: ChannelAddress | None = None

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        _shrink_font(self)

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def panel(self) -> int:
        """Panel number (1 or 2)."""
        return self._panel

    @property
    def nodes(self) -> tuple[int, ...]:
        """Nodes drawn as columns, left to right."""
        return self._nodes

    @property
    def view(self) -> str:
        """Current view, ``VIEW_ANODES`` or ``VIEW_CATHODES``."""
        return self._view

    @property
    def model(self) -> SystemMapModel | None:
        """Model currently drawn, if any."""
        return self._model

    def set_model(self, model: SystemMapModel | None) -> None:
        """Replace the drawn model (``None`` paints the empty grid)."""
        self._model = model
        self.update()

    def set_view(self, view: str) -> None:
        """Switch between the anode and cathode views."""
        _check_view(view)
        if view != self._view:
            self._view = view
            self.update()

    def set_selection(
        self, node: int | None, board: int | None, channel: ChannelTuple | None
    ) -> None:
        """Set the outlined board and channel (either may be ``None``)."""
        self._selected_board = (node, board) if node is not None and board is not None else None
        self._selected_channel = as_address(channel) if channel is not None else None
        self.update()

    def sizeHint(self) -> QSize:
        """Preferred size: about 3 px per anode cell, 14 px per row."""
        cols, rows = len(self._nodes), len(self._boards)
        return QSize(cols * ANODES_PER_BOARD * 3 + 40, rows * 14 + 24)

    def minimumSizeHint(self) -> QSize:
        """Minimum size: half the preferred width, 1 px rows.

        Kept small on purpose: the dock's minimum adds directly to the main
        window's minimum height.
        """
        cols, rows = len(self._nodes), len(self._boards)
        return QSize(cols * ANODES_PER_BOARD * 3 // 2 + 40, rows + 24)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _tile_size(self) -> tuple[float, float]:
        cols, rows = len(self._nodes), len(self._boards)
        avail_w = self.width() - self.LEFT_MARGIN - self.RIGHT_MARGIN - (cols - 1) * self.TILE_GAP
        avail_h = self.height() - self.TOP_MARGIN - self.BOTTOM_MARGIN - (rows - 1) * self.TILE_GAP
        return max(avail_w, 0) / cols, max(avail_h, 0) / rows

    def tile_rect(self, node: int, board: int) -> QRectF:
        """Rectangle of the tile for ``(node, board)`` in widget coordinates.

        Raises:
            ValueError: If the node is not in this panel or the board is not
                in the grid.
        """
        if node not in self._nodes:
            raise ValueError(f"node {node} is not in panel {self._panel} ({self._nodes})")
        if board not in self._boards:
            raise ValueError(f"board {board} is not in the grid ({self._boards})")
        col = self._nodes.index(node)
        row = self._boards.index(board)
        tile_w, tile_h = self._tile_size()
        x = self.LEFT_MARGIN + col * (tile_w + self.TILE_GAP)
        y = self.TOP_MARGIN + row * (tile_h + self.TILE_GAP)
        return QRectF(x, y, tile_w, tile_h)

    def tile_at(self, pos: QPoint) -> tuple[int, int] | None:
        """Return ``(node, board)`` of the tile under ``pos``, or ``None``.

        Uses the rule by which the tiles are painted (see :func:`_span_index`),
        so every pixel painted as a tile hits that tile.
        """
        tile_w, tile_h = self._tile_size()
        if tile_w <= 0 or tile_h <= 0:
            return None
        col, x_in = _span_index(pos.x(), self.LEFT_MARGIN, tile_w + self.TILE_GAP)
        row, y_in = _span_index(pos.y(), self.TOP_MARGIN, tile_h + self.TILE_GAP)
        if not (0 <= col < len(self._nodes) and 0 <= row < len(self._boards)):
            return None
        if x_in > tile_w or y_in > tile_h:
            return None  # in the gap between tiles
        return self._nodes[col], self._boards[row]

    def cell_rect(self, node: int, board: int, position: int) -> QRectF:
        """Rectangle of one cell in the current view.

        Args:
            node: Node of the tile.
            board: Board of the tile.
            position: 1-based position within the current view: 1..39 in the
                anode view, 1..8 in the cathode view.
        """
        n_cells = _VIEW_CELLS[self._view]
        if not 1 <= position <= n_cells:
            raise ValueError(f"position must be in 1..{n_cells} for the {self._view} view")
        tile = self.tile_rect(node, board)
        cell_w = tile.width() / n_cells
        return QRectF(tile.left() + (position - 1) * cell_w, tile.top(), cell_w, tile.height())

    def cell_at(self, pos: QPoint) -> tuple[int, int, int] | None:
        """Return ``(node, board, position)`` of the cell under ``pos``, or ``None``.

        ``position`` is 1-based within the current view. Tiles without data
        are still subdivided so callers can tell which board was hit. Like
        :meth:`tile_at`, this follows the painting rule, so every painted
        pixel of a cell selects that cell.
        """
        hit = self.tile_at(pos)
        if hit is None:
            return None
        node, board = hit
        tile = self.tile_rect(node, board)
        n_cells = _VIEW_CELLS[self._view]
        cell_w = tile.width() / n_cells
        if cell_w <= 0:
            return None
        index, _ = _span_index(pos.x(), tile.left(), cell_w)
        return node, board, min(max(index + 1, 1), n_cells)

    # ------------------------------------------------------------------
    # Model lookups
    # ------------------------------------------------------------------

    def _board_cells(self, node: int, board: int) -> BoardCells | None:
        if self._model is None:
            return None
        return self._model.boards.get((node, board))

    def _data_cell(self, node: int, board: int, position: int) -> MapCell | None:
        """The cell at ``position`` when the board has data, else ``None``."""
        board_cells = self._board_cells(node, board)
        if board_cells is None or not board_cells.has_data:
            return None
        cells = board_cells.cells(_VIEW_KIND[self._view])
        if not 1 <= position <= len(cells):
            return None
        return cells[position - 1]

    def tooltip_at(self, pos: QPoint) -> str | None:
        """Hover text for ``pos``: the cell tooltip, the board summary, or ``None``."""
        hit = self.cell_at(pos)
        if hit is None:
            return None
        node, board, position = hit
        board_cells = self._board_cells(node, board)
        if board_cells is None:
            return None
        if not board_cells.has_data:
            return format_board_summary(board_cells)
        cell = self._data_cell(node, board, position)
        return cell.tooltip if cell is not None else None

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _tooltip_rect_at(self, pos: QPoint) -> QRect:
        """Area a tooltip shown for ``pos`` stays valid in: its cell or no-data tile."""
        hit = self.cell_at(pos)
        if hit is None:
            return QRect()
        node, board, position = hit
        board_cells = self._board_cells(node, board)
        if board_cells is None or not board_cells.has_data:
            return self.tile_rect(node, board).toRect()
        return self.cell_rect(node, board, position).toRect()

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        """Emit ``cell_clicked`` / ``board_clicked`` / ``context_requested``."""
        if a0 is None:
            return
        hit = self.cell_at(a0.position().toPoint())
        if hit is None:
            super().mousePressEvent(a0)
            return
        node, board, position = hit
        cell = self._data_cell(node, board, position)
        button = a0.button()
        if button == Qt.MouseButton.LeftButton:
            if cell is not None:
                self.cell_clicked.emit(cell.channel)
            else:
                self.board_clicked.emit(node, board)
            a0.accept()
        elif button == Qt.MouseButton.RightButton:
            channel = cell.channel if cell is not None else None
            self.context_requested.emit(node, board, channel, a0.globalPosition().toPoint())
            a0.accept()
        else:
            super().mousePressEvent(a0)

    def event(self, a0: QEvent | None) -> bool:
        """Show the cell or board tooltip on hover.

        The tip is bound to the hovered cell's rectangle so it hides and
        re-arms as the cursor slides across neighbouring (4 px wide) cells.
        """
        if a0 is not None and a0.type() == QEvent.Type.ToolTip:
            help_event = cast(QHelpEvent, a0)
            pos = help_event.pos()
            text = self.tooltip_at(pos)
            if text is None:
                QToolTip.hideText()
                a0.ignore()
            else:
                QToolTip.showText(help_event.globalPos(), text, self, self._tooltip_rect_at(pos))
            return True
        return bool(super().event(a0))

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, a0: QPaintEvent | None) -> None:  # noqa: ARG002
        """Paint labels, tiles/cells and the selection outlines."""
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), _BACKGROUND_QCOLOR)
            self._paint_labels(painter)
            self._paint_tiles(painter)
            self._paint_selection(painter)
        finally:
            painter.end()

    def _paint_labels(self, painter: QPainter) -> None:
        painter.setPen(_LABEL_QCOLOR)
        first_board = self._boards[0]
        first_node = self._nodes[0]
        for node in self._nodes:
            tile = self.tile_rect(node, first_board)
            painter.drawText(
                QRectF(tile.left(), 0, tile.width(), self.TOP_MARGIN),
                int(Qt.AlignmentFlag.AlignCenter),
                f"N{node}",
            )
        # Rows are ~10 px at dock size; when the font is taller than the row
        # pitch, label every k-th row (always the first) so labels never overlap.
        _tile_w, tile_h = self._tile_size()
        pitch = tile_h + self.TILE_GAP
        font_h = painter.fontMetrics().height()
        step = 1 if pitch <= 0 else max(1, math.ceil(font_h / pitch))
        flags = int(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextDontClip
        )
        for index, board in enumerate(self._boards):
            if index % step:
                continue
            tile = self.tile_rect(first_node, board)
            label_rect = QRectF(0, tile.center().y() - font_h / 2, self.LEFT_MARGIN - 3, font_h)
            painter.drawText(label_rect, flags, f"B{board}")

    def _paint_tiles(self, painter: QPainter) -> None:
        kind = _VIEW_KIND[self._view]
        for node in self._nodes:
            for board in self._boards:
                tile = self.tile_rect(node, board)
                board_cells = self._board_cells(node, board)
                if board_cells is None or not board_cells.has_data:
                    painter.fillRect(tile, _NO_DATA_QCOLOR)
                    continue
                cells = board_cells.cells(kind)
                _draw_cell_row(
                    painter,
                    cells,
                    tile.left(),
                    tile.top(),
                    tile.width() / len(cells),
                    tile.height(),
                )

    def _paint_selection(self, painter: QPainter) -> None:
        selected_board = self._selected_board
        if selected_board is not None and selected_board[0] in self._nodes:
            node, board = selected_board
            if board in self._boards:
                _draw_board_outline(painter, self.tile_rect(node, board))

        channel = self._selected_channel
        if channel is None or self._model is None or channel.node not in self._nodes:
            return
        located = self._model.locate(channel)
        if located is None:
            return
        board_cells, kind, position = located
        if kind != _VIEW_KIND[self._view]:
            return  # e.g. a cathode channel while the anode view is shown
        rect = self.cell_rect(board_cells.node, board_cells.board, position)
        _draw_outline(painter, rect, SELECTED_CHANNEL_OUTLINE)


class BoardStripWidget(QWidget):
    """One board magnified: 39 anode cells, a gap, then 8 cathode cells.

    Signals:
        cell_clicked(ChannelAddress): Left click on a cell, or a keyboard move
            that changed the selection.
        context_requested(int, int, object, object): Right click anywhere on
            the board (``node, board, ChannelAddress | None, global QPoint``).
        board_step_requested(int): Up (-1) or Down (+1) arrow key.
        node_step_requested(int): -1/+1: same board on the previous/next node
            along the ring 1..10 (Ctrl+Left/Right); -5/+5: the same column on
            the other panel (Ctrl+Shift+Left/Right).
    """

    HEADER_HEIGHT = 18
    MARGIN = 4
    # Anodes, one empty slot as the gap, then cathodes share the row width.
    SLOTS = ANODES_PER_BOARD + 1 + CATHODES_PER_BOARD
    _CATHODE_SLOT_OFFSET = ANODES_PER_BOARD + 1
    # Anode positions labelled when the cells are too narrow for every label.
    SPARSE_ANODE_POSITIONS: frozenset[int] = frozenset(
        {1, *range(5, ANODES_PER_BOARD, 5), ANODES_PER_BOARD}
    )

    cell_clicked = pyqtSignal(object)
    context_requested = pyqtSignal(int, int, object, object)
    board_step_requested = pyqtSignal(int)
    node_step_requested = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._board: BoardCells | None = None
        self._selected_channel: ChannelAddress | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        _shrink_font(self)

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def board(self) -> BoardCells | None:
        """Board currently shown, if any."""
        return self._board

    @property
    def selected_channel(self) -> ChannelAddress | None:
        """Outlined channel, if any."""
        return self._selected_channel

    @property
    def header_text(self) -> str:
        """Header line: the board summary, or a placeholder without a board."""
        if self._board is None:
            return "No board selected"
        return format_board_summary(self._board)

    def set_board(self, board: BoardCells | None) -> None:
        """Show ``board`` (``None`` clears the strip)."""
        self._board = board
        self.update()

    def set_selected_channel(self, channel: ChannelTuple | None) -> None:
        """Outline ``channel`` (``None`` removes the outline)."""
        self._selected_channel = as_address(channel) if channel is not None else None
        self.update()

    def sizeHint(self) -> QSize:
        """Preferred size: 22 px per cell plus margins, 60 px tall."""
        return QSize((self.SLOTS - 1) * 22 + 2 * self.MARGIN, 60)

    def minimumSizeHint(self) -> QSize:
        """Minimum size: 8 px per cell, a 4 px cell row (see the grid's note)."""
        return QSize((self.SLOTS - 1) * 8, self.HEADER_HEIGHT + 2 * self.MARGIN + 4)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _cell_width(self) -> float:
        return float(max(self.width() - 2 * self.MARGIN, 0)) / self.SLOTS

    def _row_rect(self) -> QRectF:
        top = self.MARGIN + self.HEADER_HEIGHT
        return QRectF(
            self.MARGIN,
            top,
            max(self.width() - 2 * self.MARGIN, 0),
            max(self.height() - self.MARGIN - top, 0),
        )

    def _slot(self, kind: str, position: int) -> int:
        if kind == "anode":
            if not 1 <= position <= ANODES_PER_BOARD:
                raise ValueError(f"anode position must be in 1..{ANODES_PER_BOARD}")
            return position - 1
        if kind == "cathode":
            if not 1 <= position <= CATHODES_PER_BOARD:
                raise ValueError(f"cathode position must be in 1..{CATHODES_PER_BOARD}")
            return self._CATHODE_SLOT_OFFSET + position - 1
        raise ValueError(f"kind must be 'anode' or 'cathode', got {kind!r}")

    def cell_rect(self, kind: str, position: int) -> QRectF:
        """Rectangle of the ``kind`` (``"anode"``/``"cathode"``) cell at ``position``."""
        slot = self._slot(kind, position)
        row = self._row_rect()
        cell_w = self._cell_width()
        return QRectF(self.MARGIN + slot * cell_w, row.top(), cell_w, row.height())

    def cell_at(self, pos: QPoint) -> MapCell | None:
        """Cell under ``pos``, or ``None`` (no board, header, margins or the gap).

        Uses the painting rule, like the grid (see :func:`_span_index`).
        """
        if self._board is None:
            return None
        row = self._row_rect()
        cell_w = self._cell_width()
        if cell_w <= 0 or not row.contains(QPointF(pos) + QPointF(0.5, 0.5)):
            return None
        slot, _ = _span_index(pos.x(), self.MARGIN, cell_w)
        if 0 <= slot < ANODES_PER_BOARD:
            return self._board.anodes[slot]
        cathode_index = slot - self._CATHODE_SLOT_OFFSET
        if 0 <= cathode_index < CATHODES_PER_BOARD:
            return self._board.cathodes[cathode_index]
        return None

    def tooltip_at(self, pos: QPoint) -> str | None:
        """Hover text for ``pos``: the cell tooltip or ``None``."""
        cell = self.cell_at(pos)
        return cell.tooltip if cell is not None else None

    def _tooltip_rect_at(self, pos: QPoint) -> QRect:
        """Rectangle of the cell under ``pos`` (empty when nothing is hit)."""
        cell = self.cell_at(pos)
        if cell is None or self._board is None:
            return QRect()
        return self.cell_rect(cell.kind, cell.position).toRect()

    def _all_cells(self) -> tuple[MapCell, ...]:
        if self._board is None:
            return ()
        return self._board.anodes + self._board.cathodes

    def _selected_index(self, cells: Sequence[MapCell]) -> int | None:
        if self._selected_channel is None:
            return None
        for index, cell in enumerate(cells):
            if cell.channel == self._selected_channel:
                return index
        return None

    def _locate_selected(self) -> tuple[str, int] | None:
        """``(kind, position)`` of the selected channel on the shown board."""
        if self._board is None or self._selected_channel is None:
            return None
        for kind in ("anode", "cathode"):
            for cell in self._board.cells(kind):
                if cell.channel == self._selected_channel:
                    return kind, cell.position
        return None

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _select_and_emit(self, cell: MapCell) -> None:
        self._selected_channel = cell.channel
        self.update()
        self.cell_clicked.emit(cell.channel)

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        """Left click selects a cell; right click requests the context menu."""
        if a0 is None:
            return
        if self._board is None:
            super().mousePressEvent(a0)
            return
        cell = self.cell_at(a0.position().toPoint())
        button = a0.button()
        if button == Qt.MouseButton.LeftButton:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            if cell is not None:
                self._select_and_emit(cell)
            a0.accept()
        elif button == Qt.MouseButton.RightButton:
            channel = cell.channel if cell is not None else None
            self.context_requested.emit(
                self._board.node, self._board.board, channel, a0.globalPosition().toPoint()
            )
            a0.accept()
        else:
            super().mousePressEvent(a0)

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        """Move the selection along the board or request a board or node step.

        Left/Right/Home/End move the selection; Up/Down step boards (with any
        modifier); Ctrl+Left/Right step nodes and Ctrl+Shift+Left/Right jump
        to the other panel.
        """
        if a0 is None:
            return
        key = a0.key()
        modifiers = a0.modifiers()
        if (
            key in (Qt.Key.Key_Left, Qt.Key.Key_Right)
            and modifiers & Qt.KeyboardModifier.ControlModifier
        ):
            delta = 1 if key == Qt.Key.Key_Right else -1
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                delta *= 5
            self.node_step_requested.emit(delta)
            a0.accept()
            return
        if key == Qt.Key.Key_Up:
            self.board_step_requested.emit(-1)
            a0.accept()
            return
        if key == Qt.Key.Key_Down:
            self.board_step_requested.emit(+1)
            a0.accept()
            return
        if key not in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Home, Qt.Key.Key_End):
            super().keyPressEvent(a0)
            return

        cells = self._all_cells()
        if not cells:
            a0.accept()
            return
        index = self._selected_index(cells)
        last = len(cells) - 1
        new_index: int | None
        if key == Qt.Key.Key_Home:
            new_index = 0
        elif key == Qt.Key.Key_End:
            new_index = last
        elif key == Qt.Key.Key_Right:
            new_index = 0 if index is None else min(index + 1, last)
        else:  # Left
            new_index = None if index is None else max(index - 1, 0)
        if new_index is not None and new_index != index:
            self._select_and_emit(cells[new_index])
        a0.accept()

    def event(self, a0: QEvent | None) -> bool:
        """Show the cell tooltip on hover, bound to the hovered cell's rectangle."""
        if a0 is not None and a0.type() == QEvent.Type.ToolTip:
            help_event = cast(QHelpEvent, a0)
            pos = help_event.pos()
            text = self.tooltip_at(pos)
            if text is None:
                QToolTip.hideText()
                a0.ignore()
            else:
                QToolTip.showText(help_event.globalPos(), text, self, self._tooltip_rect_at(pos))
            return True
        return bool(super().event(a0))

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, a0: QPaintEvent | None) -> None:  # noqa: ARG002
        """Paint the header, both cell rows, labels and the selection outline."""
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), _BACKGROUND_QCOLOR)
            painter.setPen(_LABEL_QCOLOR)
            painter.drawText(
                QRectF(
                    self.MARGIN, self.MARGIN, self.width() - 2 * self.MARGIN, self.HEADER_HEIGHT
                ),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                self.header_text,
            )
            board = self._board
            if board is None:
                return
            row = self._row_rect()
            cell_w = self._cell_width()
            if cell_w <= 0 or row.height() <= 0:
                return
            fill = None if board.has_data else _NO_DATA_QCOLOR
            _draw_cell_row(
                painter, board.anodes, self.MARGIN, row.top(), cell_w, row.height(), fill
            )
            _draw_cell_row(
                painter,
                board.cathodes,
                self.MARGIN + self._CATHODE_SLOT_OFFSET * cell_w,
                row.top(),
                cell_w,
                row.height(),
                fill,
            )
            self._paint_labels(painter, board, cell_w, fill)
            located = self._locate_selected()
            if located is not None:
                _draw_outline(painter, self.cell_rect(*located), SELECTED_CHANNEL_OUTLINE)
        finally:
            painter.end()

    def _paint_labels(
        self, painter: QPainter, board: BoardCells, cell_w: float, fill: QColor | None
    ) -> None:
        """Draw electrode labels: all of them when they fit, else a sparse set.

        Each label is drawn in the dark or light colour that contrasts best
        with the fills under it (``fill`` when it overrides the cells'
        colours): its own cell, or every cell a spilled label covers. A spilled
        label that cannot contrast with all of them gets a 1 px halo.
        """
        metrics = self.fontMetrics()
        limit = cell_w - 2

        def draw(cell: MapCell, rect: QRectF, flags: Qt.AlignmentFlag) -> None:
            halo: QColor | None = None
            if fill is not None:
                color = text_color_for(fill)
            elif rect.width() <= cell_w + 1e-6:
                color = text_color_for(cell.fill)
            else:
                covered = self._covered_anodes(
                    board, rect, flags, metrics.horizontalAdvance(cell.label)
                )
                color, halo = label_colors_for_fills(c.fill for c in covered)
            if halo is not None:
                painter.setPen(halo)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    painter.drawText(rect.translated(dx, dy), int(flags), cell.label)
            painter.setPen(color)
            painter.drawText(rect, int(flags), cell.label)

        cells = board.anodes + board.cathodes
        if all(metrics.horizontalAdvance(cell.label) <= limit for cell in cells):
            for kind in ("anode", "cathode"):
                for cell in board.cells(kind):
                    draw(cell, self.cell_rect(kind, cell.position), Qt.AlignmentFlag.AlignCenter)
            return

        # Too narrow for every label: label a few anode positions, letting the
        # text spill over the unlabelled neighbours, and cathodes that fit.
        spill_limit = 3 * cell_w - 2
        for cell in board.anodes:
            if cell.position not in self.SPARSE_ANODE_POSITIONS:
                continue
            if metrics.horizontalAdvance(cell.label) > spill_limit:
                continue
            rect = self.cell_rect("anode", cell.position)
            if cell.position == 1:
                rect = QRectF(rect.left(), rect.top(), 3 * cell_w, rect.height())
                flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            elif cell.position == ANODES_PER_BOARD:
                rect = QRectF(rect.right() - 3 * cell_w, rect.top(), 3 * cell_w, rect.height())
                flags = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            else:
                rect = rect.adjusted(-cell_w, 0, cell_w, 0)
                flags = Qt.AlignmentFlag.AlignCenter
            draw(cell, rect, flags)
        for cell in board.cathodes:
            if metrics.horizontalAdvance(cell.label) <= limit:
                draw(cell, self.cell_rect("cathode", cell.position), Qt.AlignmentFlag.AlignCenter)

    def _covered_anodes(
        self, board: BoardCells, rect: QRectF, flags: Qt.AlignmentFlag, text_width: float
    ) -> tuple[MapCell, ...]:
        """Anode cells under a label of ``text_width`` drawn in ``rect`` with ``flags``."""
        if flags & Qt.AlignmentFlag.AlignLeft:
            start = rect.left()
        elif flags & Qt.AlignmentFlag.AlignRight:
            start = rect.right() - text_width
        else:
            start = rect.center().x() - text_width / 2
        cell_w = self._cell_width()
        first = int((start - self.MARGIN) / cell_w)
        last = int((start + text_width - self.MARGIN - 1e-6) / cell_w)
        first = min(max(first, 0), ANODES_PER_BOARD - 1)
        last = min(max(last, first), ANODES_PER_BOARD - 1)
        return board.anodes[first : last + 1]


class SystemMapWidget(QWidget):
    """The System Map dock: toggles, legend, summary, panel grids and strip.

    Signals:
        channel_selected(ChannelAddress): The user picked a channel (grid or
            strip click, keyboard move along the board or a board/node step
            onto a board with data). Not emitted by :meth:`set_selection` or
            :meth:`set_current_channel`.
        board_selected(int, int): The user picked a board: a no-data tile
            click, or a board/node step (Up/Down, Ctrl+Left/Right,
            Ctrl+Shift+Left/Right), which emits this before
            ``channel_selected``.
        fit_board_requested(int, int): "Fit Board" chosen from the context
            menu (``node, board``).
        fit_channel_requested(ChannelAddress): "Fit Channel" chosen from the
            context menu.
        color_mode_changed(str): The user picked another colour mode in the
            selector (not emitted by :meth:`set_color_mode`).
        view_changed(str): The user switched between the Anodes and Cathodes
            views (``VIEW_ANODES`` / ``VIEW_CATHODES``; not emitted by
            :meth:`set_view`).
    """

    channel_selected = pyqtSignal(object)
    board_selected = pyqtSignal(int, int)
    fit_board_requested = pyqtSignal(int, int)
    fit_channel_requested = pyqtSignal(object)
    color_mode_changed = pyqtSignal(str)
    view_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model: SystemMapModel | None = None
        self._current_board: tuple[int, int] | None = None
        self._current_channel: ChannelAddress | None = None
        # (kind, position) of the last selected channel; see _move_to_board.
        self._last_position: tuple[str, int] | None = None
        self._view = VIEW_ANODES
        self._color_mode = MODE_STATUS
        self._informational_flags: frozenset[str] = frozenset()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Row 1: view toggle, colour mode, legend or colour bar. A flow layout
        # so a narrow dock wraps the legend instead of clipping it.
        top_row = FlowLayout(h_spacing=8, v_spacing=2)
        self.anode_radio = QRadioButton("Anodes")
        self.anode_radio.setChecked(True)
        self.cathode_radio = QRadioButton("Cathodes")
        self.view_group = QButtonGroup(self)
        self.view_group.addButton(self.anode_radio)
        self.view_group.addButton(self.cathode_radio)
        top_row.addWidget(_hbox_group(self.anode_radio, self.cathode_radio))

        self.color_mode_combo = QComboBox()
        for mode in COLOR_MODES:
            self.color_mode_combo.addItem(MODE_LABELS[mode], mode)
        self.color_mode_combo.setToolTip(
            "Colour the cells by fit status or by a metric (viridis scale, "
            f"clipped to the {clip_range_text()})"
        )
        top_row.addWidget(_hbox_group(QLabel("Colour:"), self.color_mode_combo))

        # Legend swatches for the status mode and for the metric modes; the
        # ones of the other kind of mode are hidden.
        self._legend_groups: dict[str, list[QWidget]] = {}
        self._legend_texts: dict[str, list[str]] = {}
        first_metric = next(iter(METRIC_SPECS))
        for group, mode in ((_LEGEND_STATUS, MODE_STATUS), (_LEGEND_METRIC, first_metric)):
            entries = legend_entries(mode)
            swatches = [_legend_swatch(label, color) for label, color in entries]
            for swatch in swatches:
                top_row.addWidget(swatch)
            self._legend_groups[group] = swatches
            self._legend_texts[group] = [label for label, _color in entries]
        # Override marker: shown in every mode.
        self.override_legend = _hbox_group(_OverrideSwatch(), QLabel(OVERRIDE_LEGEND_LABEL))
        self.override_legend.setToolTip(
            "Corner tick: the channel was re-fitted with its own options (override)"
        )
        top_row.addWidget(self.override_legend)

        self.color_bar = ColorBarWidget()
        top_row.addWidget(self.color_bar)
        layout.addLayout(top_row)

        # Row 2: system-wide counts. On its own row so a narrow dock never
        # clips them against the legend, and wrapped: the metric summary is
        # long enough to set the dock's minimum width otherwise.
        self.summary_label = QLabel(NO_DATA_SUMMARY)
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        # Row 3: warning about boards outside the grid and results on channels
        # that are not electrodes (hidden unless needed).
        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(f"color: {WARNING_COLOR};")
        self.warning_label.hide()
        layout.addWidget(self.warning_label)

        # Row 4: the two panel grids side by side.
        grid_row = QHBoxLayout()
        self.panel_grids: tuple[PanelGridWidget, PanelGridWidget] = (
            PanelGridWidget(1),
            PanelGridWidget(2),
        )
        for grid in self.panel_grids:
            grid_row.addWidget(grid, 1)
        layout.addLayout(grid_row, 1)

        # Row 5: the magnified board.
        self.board_strip = BoardStripWidget()
        layout.addWidget(self.board_strip)

        self.anode_radio.toggled.connect(self._on_anode_radio_toggled)
        self.color_mode_combo.currentIndexChanged.connect(self._on_color_mode_combo_changed)
        for grid in self.panel_grids:
            grid.cell_clicked.connect(self._on_cell_clicked)
            grid.board_clicked.connect(self._on_board_clicked)
            grid.context_requested.connect(self._on_context_requested)
        self.board_strip.cell_clicked.connect(self._on_cell_clicked)
        self.board_strip.context_requested.connect(self._on_context_requested)
        self.board_strip.board_step_requested.connect(self._on_board_step_requested)
        self.board_strip.node_step_requested.connect(self._on_node_step_requested)

        self._update_color_widgets()

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def model(self) -> SystemMapModel | None:
        """Model currently shown, if any."""
        return self._model

    @property
    def current_board(self) -> tuple[int, int] | None:
        """``(node, board)`` shown in the strip, if any."""
        return self._current_board

    @property
    def current_channel(self) -> ChannelAddress | None:
        """Outlined channel, if any."""
        return self._current_channel

    @property
    def view(self) -> str:
        """Current view, ``VIEW_ANODES`` or ``VIEW_CATHODES``."""
        return self._view

    @property
    def color_mode(self) -> str:
        """Current colour mode (one of ``map_colors.COLOR_MODES``)."""
        return self._color_mode

    @property
    def informational_flags(self) -> frozenset[str]:
        """Flags that do not make an ``ok`` channel ``flagged``."""
        return self._informational_flags

    def set_view(self, view: str) -> None:
        """Switch both grids between the anode and cathode views."""
        _check_view(view)
        self._view = view
        for grid in self.panel_grids:
            grid.set_view(view)
        self.anode_radio.blockSignals(True)
        try:
            self.anode_radio.setChecked(view == VIEW_ANODES)
            self.cathode_radio.setChecked(view == VIEW_CATHODES)
        finally:
            self.anode_radio.blockSignals(False)
        self._update_color_widgets()  # the colour bar shows this view's limits

    def set_color_mode(self, mode: str) -> None:
        """Colour the cells by status or by a metric; emits nothing.

        Recolours the current model (nothing is rebuilt or re-read) and shows
        the colour bar in a metric mode, the status legend otherwise.

        Raises:
            ValueError: If ``mode`` is not one of ``map_colors.COLOR_MODES``.
        """
        check_color_mode(mode)
        self.color_mode_combo.blockSignals(True)
        try:
            self.color_mode_combo.setCurrentIndex(COLOR_MODES.index(mode))
        finally:
            self.color_mode_combo.blockSignals(False)
        if mode == self._color_mode:
            return
        self._color_mode = mode
        if self._model is not None:
            self._apply_model(recolor_system_map(self._model, color_mode=mode))
        else:
            self._update_color_widgets()

    def set_informational_flags(self, flags: Iterable[str]) -> None:
        """Set the flags that do not make an ``ok`` channel ``flagged`` and recolour."""
        informational = frozenset(flags)
        if informational == self._informational_flags:
            return
        self._informational_flags = informational
        if self._model is not None:
            self._apply_model(recolor_system_map(self._model, informational_flags=informational))

    def set_state(
        self,
        views: Mapping[ChannelKeyT, ChannelView],
        active_boards: Iterable[tuple[int, int]] | None = None,
        data_channels: Iterable[ChannelTuple] | None = None,
    ) -> None:
        """Rebuild the map from the channel views and repaint.

        The current board and channel selection are kept when they are still
        in the grid, and so are the colour mode and informational flags. See
        :func:`~uvcorr.gui._system_map_model.build_system_map` for the
        argument semantics.
        """
        model = build_system_map(
            views,
            active_boards=active_boards,
            data_channels=data_channels,
            color_mode=self._color_mode,
            informational_flags=self._informational_flags,
        )
        self._apply_model(model)

    def update_views(self, changed: Mapping[ChannelKeyT, ChannelView]) -> None:
        """Add or replace the views of a few channels and repaint.

        For single-channel and board re-fits: only the boards holding a
        changed channel are re-classified (see
        :func:`~uvcorr.gui._system_map_model.update_system_map`), the rest
        is recoloured, and the selection is kept. Without a model this is
        ``set_state(changed)``.
        """
        if self._model is None:
            self.set_state(changed)
            return
        if not changed:
            return
        self._apply_model(update_system_map(self._model, changed))

    def set_selection(
        self, node: int | None, board: int | None, channel: ChannelTuple | None = None
    ) -> None:
        """Highlight a board and optionally one of its channels; emits nothing.

        For keeping the map in step with other selectors in the main window.
        ``node`` or ``board`` None (or a board outside the grid) clears the
        selection. A channel the model does not lay out selects the board
        alone. The view is left unchanged, so a cathode selected while the
        anode view is shown is outlined only in the strip.

        Raises:
            ValueError: If ``channel`` is not on ``(node, board)``.
        """
        if node is None or board is None or not is_in_grid(node, board):
            self._clear_selection()
            return
        address = as_address(channel) if channel is not None else None
        if address is not None and (address.node, address.board) != (node, board):
            raise ValueError(f"channel {tuple(address)} is not on node {node} board {board}")
        self._current_board = (node, board)
        if address is not None and self._model is not None and self._model.locate(address) is None:
            address = None
        self._current_channel = address
        self._remember_position()
        self._refresh_strip()
        self._push_selection()

    def set_current_channel(self, channel: ChannelTuple) -> None:
        """Highlight ``channel`` and show its board; emits nothing.

        Same as ``set_selection(node, board, channel)`` with the channel's
        own node and board (adc2kev's API).
        """
        self.set_selection(channel[0], channel[1], channel)

    def clear(self) -> None:
        """Drop the model and selection and reset the summary."""
        self._model = None
        for grid in self.panel_grids:
            grid.set_model(None)
        self._last_position = None
        self._clear_selection()
        self.summary_label.setText(NO_DATA_SUMMARY)
        self._update_warning((), ())
        self._update_color_widgets()

    def legend_labels(self) -> list[str]:
        """Texts of the legend entries shown for the current colour mode.

        The fill swatches of the mode, then the override marker.
        """
        group = _LEGEND_METRIC if is_metric_mode(self._color_mode) else _LEGEND_STATUS
        return [*self._legend_texts[group], OVERRIDE_LEGEND_LABEL]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_model(self, model: SystemMapModel) -> None:
        """Show ``model``, keeping the selection where it is still valid."""
        self._model = model
        for grid in self.panel_grids:
            grid.set_model(model)
        if self._current_board is not None and self._current_board not in model.boards:
            self._current_board = None
            self._current_channel = None
        if self._current_channel is not None and model.locate(self._current_channel) is None:
            self._current_channel = None
        self._remember_position()
        self._refresh_strip()
        self._push_selection()
        self.summary_label.setText(format_summary(model))
        self._update_warning(model.unmapped_boards, model.unplaced_channels)
        self._update_color_widgets()

    def _update_color_widgets(self) -> None:
        """Show the legend of the current colour mode and, in a metric mode, the colour bar."""
        metric = is_metric_mode(self._color_mode)
        for group, swatches in self._legend_groups.items():
            visible = (group == _LEGEND_METRIC) == metric
            for swatch in swatches:
                swatch.setVisible(visible)
        if metric:
            limits = (
                self._model.limits_for(_VIEW_KIND[self._view]) if self._model is not None else None
            )
            self.color_bar.set_metric(METRIC_SPECS[self._color_mode], limits)
        else:
            self.color_bar.set_metric(None, None)
        self.color_bar.setVisible(metric)

    def _clear_selection(self) -> None:
        self._current_board = None
        self._current_channel = None
        self._refresh_strip()
        self._push_selection()

    def _refresh_strip(self) -> None:
        board: BoardCells | None = None
        if self._model is not None and self._current_board is not None:
            board = self._model.boards.get(self._current_board)
        self.board_strip.set_board(board)
        self.board_strip.set_selected_channel(self._current_channel)

    def _push_selection(self) -> None:
        node, board = self._current_board if self._current_board is not None else (None, None)
        for grid in self.panel_grids:
            grid.set_selection(node, board, self._current_channel)
        self.board_strip.set_selected_channel(self._current_channel)

    def _remember_position(self) -> None:
        """Record the ``(kind, position)`` of the current channel, if the model knows it."""
        if self._model is None or self._current_channel is None:
            return
        located = self._model.locate(self._current_channel)
        if located is not None:
            _, kind, position = located
            self._last_position = (kind, position)

    def _update_warning(
        self, unmapped: Sequence[tuple[int, int]], unplaced: Sequence[ChannelAddress]
    ) -> None:
        """Warn about boards outside the grid and results on non-electrode channels."""
        parts: list[str] = []
        if unmapped:
            listed = ", ".join(f"({node}, {board})" for node, board in unmapped)
            parts.append(
                f"{len(unmapped)} board(s) with data outside the mapped grid "
                f"(nodes {GRID_NODES[0]}-{GRID_NODES[-1]}, "
                f"boards {GRID_BOARDS[0]}-{GRID_BOARDS[-1]}): {listed}"
            )
        if unplaced:
            shown = ", ".join(str(tuple(channel)) for channel in unplaced[:_MAX_LISTED_UNPLACED])
            more = len(unplaced) - _MAX_LISTED_UNPLACED
            if more > 0:
                shown += f", ... ({more} more)"
            noun = "result" if len(unplaced) == 1 else "results"
            parts.append(f"{len(unplaced)} {noun} on unmapped channels: {shown}")
        if not parts:
            self.warning_label.clear()
            self.warning_label.hide()
            return
        self.warning_label.setText("; ".join(parts))
        self.warning_label.show()

    def _select_channel(self, channel: ChannelAddress) -> None:
        self._current_board = (channel.node, channel.board)
        self._current_channel = channel
        self._remember_position()
        self._refresh_strip()
        self._push_selection()

    def _on_anode_radio_toggled(self, checked: bool) -> None:
        view = VIEW_ANODES if checked else VIEW_CATHODES
        if view == self._view:
            return
        self.set_view(view)
        self.view_changed.emit(view)

    def _on_color_mode_combo_changed(self, index: int) -> None:
        if not 0 <= index < len(COLOR_MODES):
            return
        mode = COLOR_MODES[index]
        if mode == self._color_mode:
            return
        self.set_color_mode(mode)
        self.color_mode_changed.emit(mode)

    def _on_cell_clicked(self, channel: ChannelAddress) -> None:
        self._select_channel(channel)
        # Keyboard stepping lives in the strip; a grid click should arm it too.
        self.board_strip.setFocus(Qt.FocusReason.OtherFocusReason)
        self.channel_selected.emit(channel)

    def _on_board_clicked(self, node: int, board: int) -> None:
        self._current_board = (node, board)
        self._current_channel = None
        self._refresh_strip()
        self._push_selection()
        self.board_strip.setFocus(Qt.FocusReason.OtherFocusReason)
        self.board_selected.emit(node, board)

    def _on_board_step_requested(self, delta: int) -> None:
        """Move to the previous/next board on the same node, keeping the position."""
        if self._current_board is None:
            return
        node, board = self._current_board
        if board not in ACTIVE_BOARDS:
            return
        new_index = ACTIVE_BOARDS.index(board) + delta
        if not 0 <= new_index < len(ACTIVE_BOARDS):
            return
        self._move_to_board(node, ACTIVE_BOARDS[new_index])

    def _on_node_step_requested(self, delta: int) -> None:
        """Move to the same board on another node along the ring 1..10.

        ``delta`` -1/+1 is the previous/next node (1 -> 10 and 10 -> 1 wrap,
        5 -> 6 crosses the panels); -5/+5 is the same column on the other
        panel. The electrode position is kept as for a board step.
        """
        if self._current_board is None:
            return
        node, board = self._current_board
        if node not in GRID_NODES:
            return
        new_node = GRID_NODES[(GRID_NODES.index(node) + delta) % len(GRID_NODES)]
        self._move_to_board(new_node, board)

    def _move_to_board(self, node: int, board: int) -> None:
        """Select ``(node, board)`` at the remembered electrode position.

        Shared tail of the board and node steps. The electrode comes from
        ``_last_position`` (``kind``, 1-based position), recorded whenever a
        channel is selected, rather than from the current channel: a step
        onto a board without data selects the board alone and leaves no
        channel, and the next step must still continue at the same
        electrode. Emits ``board_selected`` and, when the target has data,
        ``channel_selected``.
        """
        target = self._model.boards.get((node, board)) if self._model is not None else None

        self._current_board = (node, board)
        self._current_channel = None
        if self._last_position is not None and target is not None and target.has_data:
            kind, position = self._last_position
            cells = target.cells(kind)
            if 1 <= position <= len(cells):
                self._current_channel = cells[position - 1].channel
        self._refresh_strip()
        self._push_selection()
        self.board_selected.emit(node, board)
        if self._current_channel is not None:
            self.channel_selected.emit(self._current_channel)

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _cell_label(self, channel: ChannelAddress) -> str:
        """Electrode label of ``channel`` from the model, else its board."""
        if self._model is not None:
            cell = self._model.cell(channel)
            if cell is not None:
                return cell.label
        return f"Node {channel.node} Board {channel.board}"

    def build_context_menu(self, node: int, board: int, channel: ChannelTuple | None) -> QMenu:
        """Build the right-click menu for ``(node, board)`` and optionally ``channel``.

        With a channel the menu holds "Fit Channel", a separator and "Fit
        Board"; without one only "Fit Board". Triggering an action emits
        :attr:`fit_channel_requested` / :attr:`fit_board_requested`. The
        menu is parented to this widget; the caller owns showing it, which
        keeps the actions inspectable without a modal ``exec``.
        """
        menu = QMenu(self)
        if channel is not None:
            address = as_address(channel)
            fit_channel = QAction(
                f"Fit Channel  {self._cell_label(address)} "
                f"(RENA {address.rena} Ch {address.channel})",
                menu,
            )
            fit_channel.triggered.connect(lambda: self.fit_channel_requested.emit(address))
            menu.addAction(fit_channel)
            menu.addSeparator()
        fit_board = QAction(f"Fit Board  Node {node} Board {board}", menu)
        fit_board.triggered.connect(lambda: self.fit_board_requested.emit(node, board))
        menu.addAction(fit_board)
        return menu

    def _select_silently(self, node: int, board: int, channel: ChannelAddress | None) -> None:
        """Select what was right-clicked without emitting the selection signals.

        Same state change as a left click, so the user sees which board and
        channel the menu applies to; the viewer is only updated once an
        action is actually chosen.
        """
        self._current_board = (node, board)
        self._current_channel = channel
        self._remember_position()
        self._refresh_strip()
        self._push_selection()

    def _on_context_requested(
        self, node: int, board: int, channel: ChannelAddress | None, global_pos: QPoint
    ) -> None:
        """Select the right-clicked cell/board and pop up the context menu.

        The selection moves to the right-clicked cell while the menu is open
        so the user sees what the actions apply to; dismissing the menu puts
        the previous selection back, since the rest of the GUI still shows
        that channel.
        """
        previous_board, previous_channel = self._current_board, self._current_channel
        # The silent select records the right-clicked cell as the remembered
        # sweep position; a dismissed menu must put that back too, or a
        # board-only landing (no channel to re-record) would inherit it.
        previous_position = self._last_position
        self._select_silently(node, board, channel)
        menu = self.build_context_menu(node, board, channel)
        try:
            chosen = menu.exec(global_pos)
        finally:
            # exec() only hides the menu; release it rather than letting one
            # QMenu per right click pile up under this widget.
            menu.deleteLater()
        if chosen is None:
            if previous_board is None:
                self._clear_selection()
            else:
                self._select_silently(previous_board[0], previous_board[1], previous_channel)
            self._last_position = previous_position
