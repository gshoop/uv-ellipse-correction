"""Control band above the central tabs (plan section 9).

Two wrapping rows (:class:`~uvcorr.gui._flow_layout.FlowLayout`, as in
adc2kev's band):

1. Channel navigation: Prev / Node / Board / Channel / Next, then the channel
   label (node, board, RENA, channel, electrode, polarity, events) and its
   fit status, coloured like the System Map's status categories (darkened or
   lightened to a 4.5:1 contrast on the band's background).
2. The fit options (Robust, clip k, max iter, Geometric refine, min events)
   and the Fit Channel / Fit Board / Fit All buttons.

The band holds no data of its own: the main window gives it the channels
with events (:meth:`ControlBand.set_channels`) and the current channel
(:meth:`ControlBand.set_current`, which emits nothing). The user's navigation
is reported as :attr:`ControlBand.channel_requested` or
:attr:`ControlBand.step_requested`.

Options: the band edits five fields of a base :class:`~uvcorr.options.FitOptions`
(the stored batch options after an open) and keeps the others
(``phase_ref_freq_hz`` and the flag thresholds) from the base.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from dataclasses import replace

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPalette, QTextDocument
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from uvcorr.analysis import ChannelKey, ChannelResult
from uvcorr.channels import electrode_label, is_active_channel
from uvcorr.ellipse import MIN_FIT_POINTS
from uvcorr.gui._contrast import readable_color
from uvcorr.gui._flow_layout import FlowLayout
from uvcorr.gui.map_colors import (
    CATEGORY_COLORS,
    CATEGORY_NO_DATA,
    CATEGORY_NOT_FITTED,
    status_category,
)
from uvcorr.gui.session import INFORMATIONAL_FLAGS
from uvcorr.options import FitOptions

__all__ = ["PHASE5_TOOLTIP", "ControlBand", "channel_item_text"]

PHASE5_TOOLTIP = "Re-fitting a channel or board with its own options arrives in phase 5"

_MAX_MIN_EVENTS = 10_000_000
_MAX_ITER_LIMIT = 100


_CODE_BASE = 256


def _channel_code(key: ChannelKey) -> int:
    """Combo item data of a channel: ``rena * 256 + channel`` (plain ints compare reliably)."""
    return key.rena * _CODE_BASE + key.channel


def channel_item_text(key: ChannelKey) -> str:
    """Channel combo entry, e.g. ``"R0 Ch27 · C04"``."""
    text = f"R{key.rena} Ch{key.channel:02d}"
    if is_active_channel(key.rena, key.channel):
        text += f" · {electrode_label(key.board, key.rena, key.channel)}"
    return text


def _hbox(*widgets: QWidget, spacing: int = 4) -> QWidget:
    """Group widgets so the flow layout keeps them on one line."""
    box = QWidget()
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for widget in widgets:
        layout.addWidget(widget)
    return box


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


class ControlBand(QWidget):
    """Channel navigation, channel label/status, fit options and fit buttons.

    Signals:
        channel_requested(ChannelKey): The user picked a channel in the
            selectors.
        step_requested(int): Prev (-1) or Next (+1) was clicked.
        options_changed(FitOptions): An option widget changed.
        fit_channel_clicked(), fit_board_clicked(), fit_all_clicked(): The
            fit buttons.
    """

    channel_requested = pyqtSignal(object)
    step_requested = pyqtSignal(int)
    options_changed = pyqtSignal(object)
    fit_channel_clicked = pyqtSignal()
    fit_board_clicked = pyqtSignal()
    fit_all_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._base_options = FitOptions()
        self._by_board: dict[tuple[int, int], list[ChannelKey]] = {}
        self._nodes: list[int] = []
        self._boards: dict[int, list[int]] = {}
        self._current: ChannelKey | None = None
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        # Row 1: navigation and channel label
        row1 = FlowLayout(h_spacing=8, v_spacing=2)
        self.prev_button = QPushButton("◀ Prev")
        self.prev_button.setToolTip("Previous channel with events")
        self.next_button = QPushButton("Next ▶")
        self.next_button.setToolTip("Next channel with events")
        self.node_combo = QComboBox()
        self.node_combo.setToolTip("Node")
        self.board_combo = QComboBox()
        self.board_combo.setToolTip("Board (boards with events)")
        self.channel_combo = QComboBox()
        self.channel_combo.setToolTip("RENA / channel (channels with events)")
        self.channel_combo.setMinimumContentsLength(12)
        row1.addWidget(
            _hbox(
                self.prev_button,
                QLabel("Node"),
                self.node_combo,
                QLabel("Board"),
                self.board_combo,
                QLabel("Channel"),
                self.channel_combo,
                self.next_button,
            )
        )
        self.channel_label = QLabel("No channel selected")
        self.channel_label.setTextFormat(Qt.TextFormat.RichText)
        row1.addWidget(self.channel_label)
        layout.addLayout(row1)

        # Row 2: fit options and buttons
        row2 = FlowLayout(h_spacing=8, v_spacing=2)
        self.robust_check = QCheckBox("Robust")
        self.robust_check.setToolTip("Robust clip-and-refit iteration (reject outlier points)")
        self.clip_spin = QDoubleSpinBox()
        self.clip_spin.setRange(0.5, 50.0)
        self.clip_spin.setSingleStep(0.5)
        self.clip_spin.setDecimals(1)
        self.clip_spin.setToolTip("Clip threshold k: keep |res - median| <= k x 1.4826 MAD")
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, _MAX_ITER_LIMIT)
        self.iter_spin.setToolTip("Maximum robust clip-and-refit iterations")
        self.geometric_check = QCheckBox("Geometric refine")
        self.geometric_check.setToolTip(
            "Refine the algebraic ellipse by least squares on the radial residuals"
        )
        self.min_events_spin = QSpinBox()
        self.min_events_spin.setRange(MIN_FIT_POINTS, _MAX_MIN_EVENTS)
        self.min_events_spin.setGroupSeparatorShown(True)
        self.min_events_spin.setToolTip(
            f"Channels with fewer events are not fitted (too_few_events); at least {MIN_FIT_POINTS}"
        )
        row2.addWidget(self.robust_check)
        row2.addWidget(_hbox(QLabel("clip k"), self.clip_spin))
        row2.addWidget(_hbox(QLabel("max iter"), self.iter_spin))
        row2.addWidget(self.geometric_check)
        row2.addWidget(_hbox(QLabel("min events"), self.min_events_spin))
        row2.addWidget(_separator())
        self.fit_channel_button = QPushButton("Fit Channel")
        self.fit_board_button = QPushButton("Fit Board")
        self.fit_all_button = QPushButton("Fit All")
        self.fit_all_button.setToolTip(
            "Fit every channel with these options and store the results in the cache"
        )
        for button in (self.fit_channel_button, self.fit_board_button):
            button.setEnabled(False)
            button.setToolTip(PHASE5_TOOLTIP)
        self.fit_all_button.setEnabled(False)
        row2.addWidget(_hbox(self.fit_channel_button, self.fit_board_button, self.fit_all_button))
        layout.addLayout(row2)

        self._write_options(self._base_options)
        self._update_nav_enabled()

        self.prev_button.clicked.connect(lambda: self.step_requested.emit(-1))
        self.next_button.clicked.connect(lambda: self.step_requested.emit(+1))
        self.node_combo.activated.connect(self._on_node_activated)
        self.board_combo.activated.connect(self._on_board_activated)
        self.channel_combo.activated.connect(self._on_channel_activated)
        self.robust_check.toggled.connect(self._on_option_changed)
        self.clip_spin.valueChanged.connect(self._on_option_changed)
        self.iter_spin.valueChanged.connect(self._on_option_changed)
        self.geometric_check.toggled.connect(self._on_option_changed)
        self.min_events_spin.valueChanged.connect(self._on_option_changed)
        self.fit_channel_button.clicked.connect(self.fit_channel_clicked.emit)
        self.fit_board_button.clicked.connect(self.fit_board_clicked.emit)
        self.fit_all_button.clicked.connect(self.fit_all_clicked.emit)

    # ------------------------------------------------------------------
    # Channels and selection
    # ------------------------------------------------------------------

    @property
    def current(self) -> ChannelKey | None:
        """The channel shown in the selectors."""
        return self._current

    def set_channels(self, channels: Iterable[ChannelKey]) -> None:
        """Set the channels with events (sorted or not); clears the current channel."""
        self._by_board = {}
        for key in sorted(ChannelKey(*k) for k in channels):
            self._by_board.setdefault((key.node, key.board), []).append(key)
        self._boards = {}
        for node, board in self._by_board:
            self._boards.setdefault(node, []).append(board)
        self._nodes = sorted(self._boards)
        self._current = None
        self._updating = True
        try:
            self.node_combo.clear()
            for node in self._nodes:
                self.node_combo.addItem(str(node), node)
            self.board_combo.clear()
            self.channel_combo.clear()
        finally:
            self._updating = False
        self.channel_label.setText("No channel selected")
        self._update_nav_enabled()

    def set_current(
        self,
        key: ChannelKey | None,
        result: ChannelResult | None = None,
        n_events: int | None = None,
        *,
        node: int | None = None,
        board: int | None = None,
    ) -> None:
        """Show a channel (or only a board) in the selectors and the label; emits nothing.

        Args:
            key: The channel, or None for a board without a channel.
            result: Its result (status shown), or None if not fitted.
            n_events: Its event count.
            node, board: The board to show when ``key`` is None.
        """
        self._current = key
        if key is not None:
            node, board = key.node, key.board
        self._updating = True
        try:
            if node is not None:
                self._select_data(self.node_combo, node)
                self._fill_boards(node)
                if board is not None:
                    self._select_data(self.board_combo, board)
                    self._fill_channels(node, board)
                    if key is not None:
                        self._select_data(self.channel_combo, _channel_code(key))
                    else:
                        self.channel_combo.setCurrentIndex(-1)
        finally:
            self._updating = False
        self.channel_label.setText(self._label_html(key, result, n_events, node, board))
        self._update_nav_enabled()

    def label_text(self) -> str:
        """The channel label as plain text."""
        document = QTextDocument()
        document.setHtml(self.channel_label.text())
        return document.toPlainText()

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------

    def options(self) -> FitOptions:
        """The base options with the band's five fields applied."""
        return replace(
            self._base_options,
            robust=self.robust_check.isChecked(),
            clip_k=float(self.clip_spin.value()),
            max_iter=int(self.iter_spin.value()),
            geometric=self.geometric_check.isChecked(),
            min_events=int(self.min_events_spin.value()),
        )

    def set_options(self, options: FitOptions) -> None:
        """Make ``options`` the base and show it (emits nothing)."""
        self._base_options = options
        self._write_options(options)

    def set_fit_all_enabled(self, enabled: bool) -> None:
        """Enable or disable the Fit All button (a file is open and nothing runs)."""
        self.fit_all_button.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_options(self, options: FitOptions) -> None:
        widgets: Sequence[QWidget] = (
            self.robust_check,
            self.clip_spin,
            self.iter_spin,
            self.geometric_check,
            self.min_events_spin,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.robust_check.setChecked(options.robust)
            self.clip_spin.setValue(options.clip_k)
            self.iter_spin.setValue(options.max_iter)
            self.geometric_check.setChecked(options.geometric)
            self.min_events_spin.setValue(max(options.min_events, MIN_FIT_POINTS))
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._update_robust_widgets()

    def _update_robust_widgets(self) -> None:
        robust = self.robust_check.isChecked()
        self.clip_spin.setEnabled(robust)
        self.iter_spin.setEnabled(robust)

    def _on_option_changed(self, *_args: object) -> None:
        self._update_robust_widgets()
        self.options_changed.emit(self.options())

    def _update_nav_enabled(self) -> None:
        has_channels = bool(self._by_board)
        for widget in (
            self.prev_button,
            self.next_button,
            self.node_combo,
            self.board_combo,
            self.channel_combo,
        ):
            widget.setEnabled(has_channels)

    @staticmethod
    def _select_data(combo: QComboBox, value: object) -> None:
        index = combo.findData(value)
        combo.setCurrentIndex(index)

    def _fill_boards(self, node: int) -> None:
        self.board_combo.clear()
        for board in self._boards.get(node, []):
            self.board_combo.addItem(str(board), board)

    def _fill_channels(self, node: int, board: int) -> None:
        self.channel_combo.clear()
        for key in self._by_board.get((node, board), []):
            self.channel_combo.addItem(channel_item_text(key), _channel_code(key))

    def _on_node_activated(self, index: int) -> None:
        """Go to the same board number on the new node (as the map's node step), else its first."""
        if self._updating or index < 0:
            return
        node = self.node_combo.itemData(index)
        boards = self._boards.get(node, [])
        if not boards:
            return
        board = self.board_combo.currentData()
        self._request_first(node, board if board in boards else boards[0])

    def _on_board_activated(self, index: int) -> None:
        if self._updating or index < 0:
            return
        node = self.node_combo.currentData()
        board = self.board_combo.itemData(index)
        if node is not None and board is not None:
            self._request_first(node, board)

    def _on_channel_activated(self, index: int) -> None:
        if self._updating or index < 0:
            return
        node = self.node_combo.currentData()
        board = self.board_combo.currentData()
        code = self.channel_combo.itemData(index)
        if node is None or board is None or code is None:
            return
        rena, channel = divmod(int(code), _CODE_BASE)
        self.channel_requested.emit(ChannelKey(int(node), int(board), rena, channel))

    def _request_first(self, node: int, board: int) -> None:
        """Go to the channel at the current RENA/channel on the new board, else its first."""
        channels = self._by_board.get((node, board), [])
        if not channels:
            return
        target = channels[0]
        if self._current is not None:
            for key in channels:
                if (key.rena, key.channel) == (self._current.rena, self._current.channel):
                    target = key
                    break
        self.channel_requested.emit(target)

    def _status_color(self, category: str) -> str:
        """The category colour, adjusted to read on the band's background."""
        background = self.channel_label.palette().color(QPalette.ColorRole.Window)
        return str(readable_color(CATEGORY_COLORS[category], background).name())

    def _label_html(
        self,
        key: ChannelKey | None,
        result: ChannelResult | None,
        n_events: int | None,
        node: int | None,
        board: int | None,
    ) -> str:
        if key is None:
            if node is not None and board is not None:
                return html.escape(f"Node {node} Board {board}: no channel selected")
            return "No channel selected"
        parts = [f"<b>N{key.node} B{key.board} R{key.rena} Ch{key.channel:02d}</b>"]
        if is_active_channel(key.rena, key.channel):
            electrode = electrode_label(key.board, key.rena, key.channel)
            polarity = result.polarity if result is not None else None
            parts.append(html.escape(f"{electrode} ({polarity})" if polarity else electrode))
        if n_events is not None:
            parts.append(f"{n_events:,} events")
        if result is None and n_events == 0:
            status_text, category = "no data", CATEGORY_NO_DATA
        elif result is None:
            status_text, category = "not fitted", CATEGORY_NOT_FITTED
        else:
            category = status_category(result.status, result.flags, INFORMATIONAL_FLAGS)
            status_text = result.status
            if result.flags:
                status_text += f" ({len(result.flags)} flag{'s' if len(result.flags) > 1 else ''})"
            if result.options_source != "batch":
                status_text += f" · {result.options_source}"
        color = self._status_color(category)
        parts.append(
            f'<span style="color:{color}; font-weight:bold">{html.escape(status_text)}</span>'
        )
        return " · ".join(parts)
