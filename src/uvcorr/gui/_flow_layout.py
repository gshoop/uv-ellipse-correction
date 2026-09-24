"""A wrapping layout for control rows.

Forked from adc2kev 2.2.7 ``gui/_flow_layout.py``. Keep it close to the
original so fixes can be carried across. One change: hidden widgets take no
space (adc2kev still reserved the spacing after each one). The System Map
swaps its status legend for a colour bar by hiding widgets in its top row, so
the hidden ones must not leave gaps.

A row of small controls laid out with ``QHBoxLayout`` is rigid: the widest
row forces a minimum width on the window, which cannot fit a 1366 px laptop
once the side docks take their share. A flow layout instead packs the
controls left to right and starts a new line when the width runs out, so the
row is one line at normal widths and a few more lines when narrow, and the
window's minimum width is set by the widest single control rather than by a
whole row.

This is a port of the Qt "Flow Layout" example with two additions: the
items on a line are vertically centred on that line's height (so a short
radio button lines up with a taller line edit), and the horizontal and
vertical spacings are separate constructor arguments.
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QLayout, QLayoutItem, QWidget


class FlowLayout(QLayout):
    """Lay items out left to right, wrapping to a new line when out of width.

    The layout reports ``hasHeightForWidth()`` so a parent ``QBoxLayout``
    can grow the owning widget as the row wraps. ``addWidget`` is
    inherited from ``QLayout`` (it calls ``addItem``). Hidden widgets are
    skipped.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        margin: int = 0,
        h_spacing: int = 6,
        v_spacing: int = 2,
    ) -> None:
        """Create the layout.

        Args:
            parent: Widget to install the layout on, or None
            margin: Contents margin on all four sides, in pixels
            h_spacing: Gap between items on a line, in pixels
            v_spacing: Gap between lines, in pixels
        """
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def __del__(self) -> None:
        """Release the items, as the Qt example does."""
        item = self.takeAt(0)
        while item is not None:
            item = self.takeAt(0)

    # -- QLayout interface ---------------------------------------------------

    def addItem(self, a0: QLayoutItem | None) -> None:
        """Append an item (``addWidget`` routes through here)."""
        if a0 is not None:
            self._items.append(a0)

    def count(self) -> int:
        """Return the number of items."""
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        """Return the item at ``index``, or None when out of range."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QLayoutItem | None:
        """Remove and return the item at ``index``, or None when out of range."""
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientation:
        """The layout never asks for more space than its items need."""
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        """The height depends on the width (more lines when narrower)."""
        return True

    def heightForWidth(self, a0: int) -> int:
        """Return the height needed to lay the items out in width ``a0``."""
        return self._do_layout(QRect(0, 0, a0, 0), test_only=True)

    def setGeometry(self, a0: QRect) -> None:
        """Place the items inside ``a0``."""
        super().setGeometry(a0)
        self._do_layout(a0, test_only=False)

    def sizeHint(self) -> QSize:
        """Same as :meth:`minimumSize`; the real height comes from the width."""
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        """The largest visible item's minimum size plus the margins."""
        size = QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(
            margins.left() + margins.right(),
            margins.top() + margins.bottom(),
        )

    # -- Implementation ------------------------------------------------------

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        """Lay the items out in ``rect`` and return the height used.

        Items are gathered line by line so that each one can be centred on
        its line's height once that height is known. With ``test_only`` the
        geometry is computed but not applied (``heightForWidth``). Hidden
        widgets (``isEmpty()``) are skipped.

        Args:
            rect: Area to lay out in (its width decides the wrapping)
            test_only: Only measure; do not move the items

        Returns:
            Height used, including the top and bottom margins
        """
        # The locals taken from Qt calls are typed explicitly so the returned
        # height is an int even where mypy sees PyQt6 as Any.
        margins = self.contentsMargins()
        top: int = rect.y()
        bottom_margin: int = margins.bottom()
        effective = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        right_edge: int = effective.x() + effective.width()
        x: int = effective.x()
        y: int = effective.y()
        line_height = 0
        line: list[tuple[QLayoutItem, int]] = []

        def place_line() -> None:
            if test_only:
                return
            for item, item_x in line:
                item_size = item.sizeHint()
                item_y = y + (line_height - item_size.height()) // 2
                item.setGeometry(QRect(QPoint(item_x, item_y), item_size))

        for item in self._items:
            if item.isEmpty():
                continue
            size = item.sizeHint()
            if x + size.width() > right_edge and line_height > 0:
                place_line()
                line = []
                x = effective.x()
                y += line_height + self._v_spacing
                line_height = 0
            line.append((item, x))
            x += size.width() + self._h_spacing
            line_height = max(line_height, size.height())
        place_line()

        return y + line_height - top + bottom_margin
