"""Fit Inspector dock: every result field of the selected channel (plan section 9).

The field list is driven by :data:`uvcorr.analysis.RESULT_COLUMNS` (the
``radial_summary.csv`` columns), so a column added to ``ChannelResult``
appears here without a change: it is placed in a group by its name
(:func:`group_of`) and formatted by its kind (:func:`format_value`).
Unavailable values (None) are shown empty.

Below the result fields the inspector lists the channel's warning flags, each
with a short explanation (:data:`FLAG_DESCRIPTIONS`, thresholds filled in from
the options used), and the options the result was fitted with, with their
source: the batch run or a per-channel override.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QPalette
from PyQt6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from uvcorr.analysis import (
    KIND_COUNT,
    KIND_FLAGS,
    KIND_FLOAT,
    KIND_FLOAT_PRECISE,
    KIND_INT,
    OPTIONS_OVERRIDE,
    RESULT_COLUMNS,
    ChannelKey,
    ChannelResult,
    ResultColumn,
)
from uvcorr.channels import electrode_label, is_active_channel, polarity_name
from uvcorr.gui._contrast import readable_color
from uvcorr.gui.map_colors import (
    CATEGORY_COLORS,
    CATEGORY_NO_DATA,
    CATEGORY_NOT_FITTED,
    status_category,
)
from uvcorr.gui.session import INFORMATIONAL_FLAGS, channel_title
from uvcorr.options import (
    FLAG_BROAD_RING,
    FLAG_CENTER_OUTSIDE_DATA,
    FLAG_EXTREME_AXIS_RATIO,
    FLAG_GAUSS_FIT_FAILED_POST,
    FLAG_GAUSS_FIT_FAILED_PRE,
    FLAG_GEOMETRIC_REFIT_FAILED,
    FLAG_HIGH_REJECTION,
    FLAG_ROBUST_REFIT_FAILED,
    FitOptions,
)

__all__ = [
    "FLAG_DESCRIPTIONS",
    "GROUP_FLAGS",
    "GROUP_OPTIONS",
    "MIN_WIDTH",
    "OPTION_LABELS",
    "RESULT_GROUPS",
    "FitInspector",
    "describe_flag",
    "format_option",
    "format_value",
    "group_of",
]

FLAG_DESCRIPTIONS: dict[str, str] = {
    FLAG_HIGH_REJECTION: "The robust fit rejected more than {high_rejection_frac:.0%} of the events.",
    FLAG_EXTREME_AXIS_RATIO: "The axis ratio b/a is below {extreme_axis_ratio:g}: a very flat ellipse.",
    FLAG_CENTER_OUTSIDE_DATA: "The fitted centre lies outside the bounding box of the points.",
    FLAG_BROAD_RING: (
        "The points do not form a thin ring: their radial spread exceeds "
        "{broad_ring_frac:g} of the radius (e.g. a blob or a dead channel)."
    ),
    FLAG_ROBUST_REFIT_FAILED: "A robust refit failed; the previous successful fit was kept.",
    FLAG_GEOMETRIC_REFIT_FAILED: "The geometric refinement failed; the algebraic fit was kept.",
    FLAG_GAUSS_FIT_FAILED_PRE: (
        "The Gaussian fit of the raw radii failed (common for an ellipse, whose radii about "
        "the centre are not Gaussian); sample statistics are reported."
    ),
    FLAG_GAUSS_FIT_FAILED_POST: (
        "The Gaussian fit of the corrected radii failed; sample statistics are reported."
    ),
}
"""A short explanation of each warning flag; ``{name}`` fields are FitOptions values."""

OPTION_LABELS: dict[str, str] = {
    "min_events": "min events",
    "robust": "robust",
    "clip_k": "clip k",
    "max_iter": "max iter",
    "geometric": "geometric",
    "phase_ref_freq_hz": "ref freq (Hz)",
    "high_rejection_frac": "high rej. >",
    "extreme_axis_ratio": "axis ratio <",
    "broad_ring_frac": "broad ring >",
}
"""Short labels of the options (they share the Field column with the result names)."""

# Result groups in display order: (title, predicate on the column name).
_IDENTITY = ("node", "board", "rena", "channel", "polarity", "electrode")
_FIT = ("status", "flags", "options_source", "n_events", "n_used", "n_rejected")
_ELLIPSE = (
    "centerU",
    "centerV",
    "semiMajor",
    "semiMinor",
    "phi",
    "axis_ratio",
    "target_radius",
)
RESULT_GROUPS: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("Channel", lambda name: name in _IDENTITY),
    ("Fit", lambda name: name in _FIT),
    ("Ellipse", lambda name: name in _ELLIPSE),
    ("Raw radii about the centre (pre)", lambda name: name.startswith("pre_")),
    ("Corrected radii (post)", lambda name: name.startswith("post_")),
    ("Residuals", lambda name: "_res_" in name),
    ("Phase and timing", lambda name: name.startswith("phase_") or name.endswith("_ns")),
)
GROUP_OTHER = "Other"
GROUP_FLAGS = "Flags"
GROUP_OPTIONS = "Options used"

_UNITS: dict[str, str] = {
    "centerU": "ADC",
    "centerV": "ADC",
    "semiMajor": "ADC",
    "semiMinor": "ADC",
    "target_radius": "ADC",
}

MIN_WIDTH = 340
"""Minimum inspector width: the Field column plus a value such as ``-0.760797 rad (-43.59°)``."""

_FIELD_PADDING = 14
_HEADER_BACKGROUND = QColor("#2b2b2b")
_HEADER_FOREGROUND = QColor("#ffffff")


def group_of(name: str) -> str:
    """The inspector group of a result column (``"Other"`` for an unknown one)."""
    for title, matches in RESULT_GROUPS:
        if matches(name):
            return title
    return GROUP_OTHER


def _unit(name: str) -> str:
    if name in _UNITS:
        return _UNITS[name]
    if name.endswith("_ns"):
        return "ns"
    if name.endswith("_rad"):
        return "rad"
    if name.startswith(("pre_", "post_")) and name.endswith(
        ("_mean", "_sigma", "_fwhm", "_robust_sigma")
    ):
        return "ADC"
    if "_res_" in name:
        return "ADC"
    return ""


def format_value(column: ResultColumn, value: Any) -> str:
    """Display text of one result value (empty for None).

    Counts get thousands separators; floats 6 significant digits (9 for the
    centre and semi-axes, as in the CSV); phi is given in radians (6 digits)
    and degrees; flags are joined with ``"; "``; a unit is appended where
    one is known.
    """
    if value is None:
        return ""
    kind = column.kind
    if kind == KIND_FLAGS:
        return "; ".join(value) if value else "none"
    if kind == KIND_COUNT or (kind == KIND_INT and column.name == "n_events"):
        return f"{int(value):,}"
    if kind == KIND_INT:
        return str(int(value))
    if kind in (KIND_FLOAT, KIND_FLOAT_PRECISE):
        number = float(value)
        if column.name == "phi":
            return f"{number:.6g} rad ({math.degrees(number):.2f}°)"
        text = f"{number:.9g}" if kind == KIND_FLOAT_PRECISE else f"{number:.6g}"
        unit = _unit(column.name)
        return f"{text} {unit}" if unit else text
    return str(value)


def format_option(name: str, value: Any) -> str:
    """Display text of one fit option."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if name.endswith("_frac") and isinstance(value, float):
        return f"{value:g} ({value:.0%})"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def describe_flag(flag: str, options: FitOptions | None) -> str:
    """The explanation of a flag, with thresholds from ``options`` (defaults if None)."""
    template = FLAG_DESCRIPTIONS.get(flag)
    if template is None:
        return "Unknown flag."
    values = (options if options is not None else FitOptions()).to_dict()
    try:
        return template.format(**values)
    except (KeyError, ValueError):
        return template


class FitInspector(QWidget):
    """Tree of the selected channel's result fields, flags and options.

    The value of every displayed field can be read back with
    :meth:`value_text` (tests, copy/paste); values are selectable.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self.title_label = QLabel("No channel selected")
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet(
            "QLabel { font-weight: bold; padding: 4px; background-color: #2b2b2b; color: #ffffff; }"
        )
        layout.addWidget(self.title_label)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Field", "Value"])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        header = self.tree.header()
        assert header is not None
        # Field: fixed from the longest name (not ResizeToContents, which let long labels
        # squeeze the values); Value: the rest. The user can still drag the divider.
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        header.setMinimumSectionSize(40)
        self.tree.setColumnWidth(0, self.field_column_width())
        layout.addWidget(self.tree, 1)
        self.setMinimumWidth(MIN_WIDTH)
        self._values: dict[str, str] = {}
        self._flags: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def field_column_width(self) -> int:
        """Width of the Field column: the longest field or option name, indented."""
        names = [column.name for column in RESULT_COLUMNS] + list(OPTION_LABELS.values())
        metrics = self.tree.fontMetrics()
        longest = max(metrics.horizontalAdvance(name) for name in names)
        return longest + self.tree.indentation() + _FIELD_PADDING

    def clear(self, message: str = "No channel selected") -> None:
        """Show nothing but a message."""
        self.title_label.setText(message)
        self.tree.clear()
        self._values = {}
        self._flags = []

    def show_channel(
        self,
        key: ChannelKey,
        result: ChannelResult | None,
        options: FitOptions | None,
        source_note: str = "",
        n_events: int | None = None,
    ) -> None:
        """Show a channel's result (or only its identity when it has none).

        Args:
            key: The channel.
            result: Its merged result, or None if it has not been fitted.
            options: The options the result was fitted with.
            source_note: Describes where the options come from (e.g. "batch
                run of 2026-09-24 10:12" or "override").
            n_events: The channel's event count (shown without a result).
        """
        self.tree.clear()
        self._values = {}
        self._flags = []
        self.title_label.setText(channel_title(key))
        if result is None:
            self._show_identity(key, n_events)
            return
        groups: dict[str, QTreeWidgetItem] = {}
        for column in RESULT_COLUMNS:
            group = group_of(column.name)
            parent = groups.get(group)
            if parent is None:
                parent = self._group_item(group)
                groups[group] = parent
            text = format_value(column, getattr(result, column.name))
            item = self._row(parent, column.name, text)
            if column.name == "status":
                self._color_status(item, result)
        self._show_flags(result, options)
        self._show_options(options, result.options_source, source_note)
        self.tree.expandAll()

    def value_text(self, name: str) -> str | None:
        """The displayed text of a result field or option (None if not shown)."""
        return self._values.get(name)

    def flags_shown(self) -> list[str]:
        """The flags listed in the Flags group, in order."""
        return list(self._flags)

    def group_titles(self) -> list[str]:
        """The top-level group titles, in order."""
        titles = []
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item is not None:
                titles.append(item.text(0))
        return titles

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _group_item(self, title: str) -> QTreeWidgetItem:
        """A bold group header spanning both columns (so it never widens the field column)."""
        item = QTreeWidgetItem([title])
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        item.setBackground(0, QBrush(_HEADER_BACKGROUND))
        item.setForeground(0, QBrush(_HEADER_FOREGROUND))
        self.tree.addTopLevelItem(item)
        item.setFirstColumnSpanned(True)
        return item

    def _row(
        self, parent: QTreeWidgetItem, name: str, text: str, tooltip: str = ""
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem([name, text])
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable)
        item.setToolTip(0, tooltip or name)
        item.setToolTip(1, tooltip or text)  # the full value when the column elides it
        parent.addChild(item)
        self._values[name] = text
        return item

    def _show_identity(self, key: ChannelKey, n_events: int | None) -> None:
        """Identity rows of a channel without a result ("no data" without events)."""
        group = self._group_item("Channel")
        for name in ("node", "board", "rena", "channel"):
            self._row(group, name, str(getattr(key, name)))
        if is_active_channel(key.rena, key.channel):
            self._row(group, "polarity", polarity_name(key.board, key.rena, key.channel))
            self._row(group, "electrode", electrode_label(key.board, key.rena, key.channel))
        fit = self._group_item("Fit")
        no_data = n_events == 0
        status = self._row(fit, "status", "no data" if no_data else "not fitted")
        category = CATEGORY_NO_DATA if no_data else CATEGORY_NOT_FITTED
        status.setForeground(1, QBrush(self._text_color(CATEGORY_COLORS[category])))
        if n_events is not None:
            self._row(fit, "n_events", f"{n_events:,}")
        self.tree.expandAll()

    def _text_color(self, color: str) -> QColor:
        """``color`` adjusted to read on the tree's row backgrounds."""
        palette = self.tree.palette()
        grounds = (
            palette.color(QPalette.ColorRole.Base),
            palette.color(QPalette.ColorRole.AlternateBase),
        )
        return readable_color(color, grounds)

    def _color_status(self, item: QTreeWidgetItem, result: ChannelResult) -> None:
        category = status_category(result.status, result.flags, INFORMATIONAL_FLAGS)
        item.setForeground(1, QBrush(self._text_color(CATEGORY_COLORS[category])))

    def _show_flags(self, result: ChannelResult, options: FitOptions | None) -> None:
        group = self._group_item(f"{GROUP_FLAGS}: {len(result.flags) if result.flags else 'none'}")
        for flag in result.flags:
            description = describe_flag(flag, options)
            item = QTreeWidgetItem([flag, description])
            item.setToolTip(0, description)
            item.setToolTip(1, description)
            group.addChild(item)
            self._flags.append(flag)

    def _show_options(self, options: FitOptions | None, source: str, note: str) -> None:
        label = "override" if source == OPTIONS_OVERRIDE else "batch"
        text = f"{label} ({note})" if note else label
        group = self._group_item(f"{GROUP_OPTIONS}: {text}")
        self._values["options_used"] = text
        if options is None:
            return
        for name, value in options.to_dict().items():
            self._row(group, OPTION_LABELS.get(name, name), format_option(name, value))
            self._values[f"option:{name}"] = format_option(name, value)
