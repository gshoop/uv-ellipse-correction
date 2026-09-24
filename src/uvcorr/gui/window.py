"""The uvcorr main window (plan section 9); the entry point is :mod:`uvcorr.gui.main`.

Built on the adc2kev 2.2.7 GUI patterns (``adc2kev/gui/main.py`` and
``docs/implementation/GUI_LAYOUT.md``): a ``QMainWindow`` with docks that
carry object names, ``QSettings`` persistence of the layout, a View menu with
dock toggles and Reset Layout, and ``QThread`` workers with
``progress``/``finished``/``error``/``stopped`` signals and a Stop action.

Layout:

- **Menu bar**: File (Open Raw, Open Cache, Export .tec, Export CSV, Export
  Both, Exit), Process (Fit All, Stop, Revert Channel / Board to Batch,
  Clear All Overrides), View (dock toggles, Focus System Map, Reset Layout),
  Help (About).
- **Toolbar** (``MainToolbar``): Open, Fit All, Stop.
- **Central widget**: the :class:`~uvcorr.gui.controls.ControlBand` above
  the tabs (:data:`TAB_TITLES`): Scatter
  (:class:`~uvcorr.gui.scatter.ScatterTab`), Radial
  (:class:`~uvcorr.gui.radial.RadialTab`), Radius vs angle
  (:class:`~uvcorr.gui.angle.AngleTab`) and Board grid
  (:class:`~uvcorr.gui.board_grid.BoardGridTab`).
- **Docks**: the System Map at the bottom (``SystemMapDock``) and the Fit
  Inspector on the right (``FitInspectorDock``).
- **Status bar**: messages, the open file's summary and a progress bar.

Data flow. The widgets talk only to the :class:`~uvcorr.gui.session.UVSession`.
Opening a file runs in a :class:`~uvcorr.gui.threads.CacheBuildThread`
(``.dat``: reuse or build the cache) or
:class:`~uvcorr.gui.threads.CacheOpenThread` (``.uv.h5``); either reads the
stored ``/results/current`` and overrides, so a cached analysis shows at once
without a refit. Fit All runs in a :class:`~uvcorr.gui.threads.FitAllThread`
and stores its results in the cache. Selecting a channel (System Map click,
control-band navigation, a Board grid cell) updates the band and the
inspector at once and loads the channel in a
:class:`~uvcorr.gui.threads.ChannelDetailThread`: first the scatter data
(drawn as soon as they arrive), then the Radial and Radius vs angle data. A
newer selection supersedes a pending one (generation counter; the second
stage of a superseded load is skipped), so stepping quickly through large
channels never queues work. The Board grid is computed in a
:class:`~uvcorr.gui.threads.BoardGridThread` with its own generation counter,
lazily: only while its tab is current (and on switching to it), when the
selected board differs from the one shown, or after the results changed
(Fit All, a re-fit, a revert).

Selection sync: the map's ``channel_selected`` drives the band, inspector and
scatter; the band drives the map through ``set_current_channel`` /
``set_selection``, which emit nothing, so there is no feedback loop.

Re-fits and overrides (plan D7, 6.1). Fit Channel / Fit Board (the band's
buttons, or the map's context menu) re-fit the selected channel or board
with the band's options in a :class:`~uvcorr.gui.threads.RefitThread`
(in-process; a board re-fit can be stopped). The session compares the
options with the stored batch options: different options store the results
as overrides, equal ones revert the channels to their batch rows (see
:mod:`uvcorr.gui.session`). The status bar says which happened; the map, the
band, the inspector and the scatter (whose kept/rejected mask follows the
override's options) are refreshed. Re-fits need stored batch results, so the
buttons are disabled with the tooltip "Run Fit All first" until a batch
exists. Options are compared as the fit uses them (with robust off, clip k
and max iter do not matter). Overrides are deleted by Revert Channel / Board
to Batch (band menu button, Process menu, map context menu) and Process >
Clear All Overrides, in a :class:`~uvcorr.gui.threads.RevertThread` (a busy
file can make the write wait). Only one of Fit All, a re-fit and a revert
runs at a time, and a write refuses results that another process replaced
on disk since they were loaded. Fit All with overrides asks whether to keep
them; kept overrides fitted with exactly the new batch options are dropped.
On open the band shows the stored batch options, so an unchanged Fit Channel
reverts; "Use this channel's options" loads the selected channel's override
options and "Batch options" loads the batch options again.

Exports (plan D3): File > Export .tec, Export CSV and Export Both write the
merged results (batch plus overrides) atomically and report the paths and
counts. They refuse to overwrite the open raw file or cache, any HDF5 file
and any ``.dat`` file, and warn if the cache's results changed on disk.

Persistence (``~/.config/uvcorr/uvcorr-gui.ini``): window geometry and dock
state (``layout/*``, tagged with ``LAYOUT_VERSION``), the last directory, the
last export directory, the map's colour mode and Anodes/Cathodes view, the
scatter's point cap, Overlay and Density toggles, the Radial tab's *Same
radius axis*, the Radius vs angle tab's *Include rejected points*, and the
Board grid's order and before/after mode.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent, QGuiApplication, QKeySequence
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from uvcorr import __version__
from uvcorr.analysis import OPTIONS_OVERRIDE, ChannelKey
from uvcorr.cache import BuildSettings, UVCacheError
from uvcorr.gui._layout import (
    FALLBACK_SCREEN,
    LAYOUT_VERSION,
    SHORT_SCREEN_HEIGHT,
    app_settings,
    initial_window_size,
    read_bool,
    read_bytes,
    read_int,
    read_str,
)
from uvcorr.gui._system_map_model import ChannelAddress, ChannelTuple, as_address
from uvcorr.gui.angle import AngleTab
from uvcorr.gui.board_grid import (
    LAYOUT_RENA,
    LAYOUTS,
    MODE_BEFORE,
    MODES,
    BoardGridData,
    BoardGridTab,
    board_keys,
)
from uvcorr.gui.controls import REFIT_NEEDS_BATCH_TIP, ControlBand
from uvcorr.gui.inspector import MIN_WIDTH as INSPECTOR_MIN_WIDTH
from uvcorr.gui.inspector import FitInspector
from uvcorr.gui.map_colors import COLOR_MODES
from uvcorr.gui.radial import RadialTab
from uvcorr.gui.scatter import DEFAULT_POINT_CAP, ScatterTab
from uvcorr.gui.session import (
    INFORMATIONAL_FLAGS,
    BatchOutcome,
    ChannelDetail,
    DetailRequest,
    ExportSummary,
    OpenedFile,
    RefitOutcome,
    RevertOutcome,
    RevertRequest,
    SessionError,
    UVSession,
    channel_title,
    inspect_raw,
    is_cache_file,
    short_title,
)
from uvcorr.gui.system_map import VIEW_ANODES, VIEW_CATHODES, SystemMapWidget
from uvcorr.gui.threads import (
    BoardGridThread,
    CacheBuildThread,
    CacheOpenThread,
    ChannelDetailThread,
    DetailViews,
    FitAllThread,
    RefitThread,
    RevertThread,
)
from uvcorr.io.summary_csv import SUMMARY_CSV_NAME
from uvcorr.options import FitOptions

logger = logging.getLogger(__name__)

__all__ = ["TAB_TITLES", "MainWindow", "RefitSystemMap"]

TAB_TITLES: tuple[str, ...] = ("Scatter", "Radial", "Radius vs angle", "Board grid")
"""The central tabs, in order."""

# Settings keys (layout/* is written with LAYOUT_VERSION)
KEY_LAYOUT = "layout"
KEY_LAYOUT_VERSION = "layout/version"
KEY_GEOMETRY = "layout/geometry"
KEY_STATE = "layout/state"
KEY_LAST_DIR = "session/last_dir"
KEY_EXPORT_DIR = "export/last_dir"
KEY_COLOR_MODE = "map/color_mode"
KEY_MAP_VIEW = "map/view"
KEY_POINT_CAP = "scatter/point_cap"
KEY_OVERLAY = "scatter/overlay"
KEY_DENSITY = "scatter/density"
KEY_RADIAL_COMMON_AXIS = "radial/common_axis"
KEY_ANGLE_INCLUDE_REJECTED = "angle/include_rejected"
KEY_GRID_LAYOUT = "grid/layout"
KEY_GRID_MODE = "grid/mode"

# Default dock splits as fractions of the window. On a short window the map gets
# less (it stays usable at ~250 px), so the scatter panels keep a useful height.
_MAP_HEIGHT_FRACTION = 0.36
_MAP_HEIGHT_FRACTION_SHORT = 0.30
_MAP_HEIGHT_MAX_SHORT = 280
_INSPECTOR_WIDTH_FRACTION = 0.24

_RAW_FILTER = "Raw data (*.dat);;All files (*)"
_CACHE_FILTER = "UV cache (*.uv.h5 *.h5);;All files (*)"
_ANY_FILTER = (
    "Raw data or UV cache (*.dat *.uv.h5 *.h5);;Raw data (*.dat);;UV cache (*.uv.h5 *.h5);;"
    "All files (*)"
)
_TEC_FILTER = "RadialAnalysis ellipse file (*.tec);;All files (*)"
_BUSY_TIP = "Wait for the running operation to finish"
_FIT_ALL_TIP = "Fit every channel with the control band's options and store the results"
_EXPORT_TEC_TIP = "Write the ok channels' ellipses (batch plus overrides) to a .tec file"
_EXPORT_CSV_TIP = "Write every channel's results (batch plus overrides) to radial_summary.csv"
_EXPORT_BOTH_TIP = "Write <name>.tec and radial_summary.csv (batch plus overrides) to a directory"
_CSV_FILTER = "CSV (*.csv);;All files (*)"


def _describe_options(options: FitOptions) -> str:
    """Short description of the options the band edits (for dialogs and the status bar)."""
    robust = f"robust k={options.clip_k:g}, {options.max_iter} iter" if options.robust else "plain"
    geometric = ", geometric" if options.geometric else ""
    return f"{robust}{geometric}, min events {options.min_events:,}"


def _plural(n: int, noun: str) -> str:
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


class RefitSystemMap(SystemMapWidget):
    """The System Map with the window's re-fit state in its context menu.

    The map's own menu holds Fit Channel and Fit Board. This subclass
    disables them, with the reason as their tooltip, while no re-fit is
    possible (:attr:`refit_blocked_reason`: no batch results yet, or an
    operation running), and adds "Revert Channel to Batch" and "Revert Board
    to Batch" for a right-clicked channel or board with overrides (read from
    the cells' override markers).

    Signals:
        revert_channel_requested(ChannelAddress): "Revert Channel to Batch".
        revert_board_requested(int, int): "Revert Board to Batch".
    """

    revert_channel_requested = pyqtSignal(object)
    revert_board_requested = pyqtSignal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.refit_blocked_reason = ""
        self.reverts_blocked = False

    def build_context_menu(self, node: int, board: int, channel: ChannelTuple | None) -> QMenu:
        """The map's menu plus the revert actions (see the class docstring)."""
        menu = super().build_context_menu(node, board, channel)
        menu.setToolTipsVisible(True)
        if self.refit_blocked_reason:
            for action in menu.actions():
                if not action.isSeparator():
                    action.setEnabled(False)
                    action.setToolTip(self.refit_blocked_reason)
        model = self.model
        address = as_address(channel) if channel is not None else None
        cell = model.cell(address) if model is not None and address is not None else None
        cells = model.boards.get((node, board)) if model is not None else None
        n_board = (
            sum(1 for item in (*cells.anodes, *cells.cathodes) if item.is_override)
            if cells is not None
            else 0
        )
        if (cell is None or not cell.is_override) and not n_board:
            return menu
        menu.addSeparator()
        if address is not None and cell is not None and cell.is_override:
            revert_channel = QAction(f"Revert Channel to Batch  {cell.label}", menu)
            revert_channel.triggered.connect(lambda: self.revert_channel_requested.emit(address))
            revert_channel.setEnabled(not self.reverts_blocked)
            menu.addAction(revert_channel)
        if n_board:
            revert_board = QAction(
                f"Revert Board to Batch  Node {node} Board {board} "
                f"({_plural(n_board, 'override')})",
                menu,
            )
            revert_board.triggered.connect(lambda: self.revert_board_requested.emit(node, board))
            revert_board.setEnabled(not self.reverts_blocked)
            menu.addAction(revert_board)
        return menu


class MainWindow(QMainWindow):
    """The uvcorr main window (see the module docstring).

    Attributes:
        session: The data layer.
        workers: Fit All worker processes (None: the analysis default).
        build_settings: Cache build tunables (None: defaults; tests use small
            batches).
        last_open_seconds: Wall time of the last open, from the click to the
            populated map.
        last_switch_seconds: Wall time of the last channel switch, from the
            selection to the drawn scatter.
        last_views_seconds: Wall time of the last channel switch, from the
            selection to the drawn Radial and Radius vs angle tabs.
        last_grid_seconds: Wall time of the last Board grid load, from the
            request to the drawn grid.
        last_refit: The last applied re-fit (timings, what was stored).
        last_export: What the last export wrote.
    """

    def __init__(
        self,
        *,
        workers: int | None = None,
        build_settings: BuildSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = UVSession()
        self.workers = workers
        self.build_settings = build_settings
        self.open_thread: CacheBuildThread | CacheOpenThread | None = None
        self.fit_all_thread: FitAllThread | None = None
        self.refit_thread: RefitThread | None = None
        self.revert_thread: RevertThread | None = None
        self._detail_thread: ChannelDetailThread | None = None
        self._detail_generation = 0
        self._pending_detail: tuple[int, DetailRequest] | None = None
        # Board grid: the board it should show, the one being loaded, and whether the
        # results changed since it was computed (see _update_board_grid)
        self._grid_thread: BoardGridThread | None = None
        self._grid_generation = 0
        self._pending_grid: tuple[int, int, int] | None = None
        self._grid_wanted: tuple[int, int] | None = None
        self._grid_loading: tuple[int, int] | None = None
        self._grid_dirty = False
        self._grid_started = 0.0
        # (node, board) shown without a channel (a map click on a board tile)
        self._board_only: tuple[int, int] | None = None
        self._open_started = 0.0
        self._opening_path: Path | None = None
        self._open_stoppable = False
        self._quit_requested = False
        self._switch_started = 0.0
        self.last_open_seconds: float | None = None
        self.last_switch_seconds: float | None = None
        self.last_views_seconds: float | None = None
        self.last_grid_seconds: float | None = None
        self.last_refit: RefitOutcome | None = None
        self.last_export: ExportSummary | None = None

        self._init_central()
        self._create_actions()
        self._create_menus()
        self._create_toolbar()
        self._create_status_bar()
        self._create_docks()
        self._populate_view_menu()
        self._restore_settings()
        self._update_title()
        self._update_actions()
        logger.info("uvcorr GUI initialised")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_central(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.control_band = ControlBand()
        layout.addWidget(self.control_band)
        self.tabs = QTabWidget()
        self.scatter = ScatterTab()
        self.radial = RadialTab()
        self.angle = AngleTab()
        self.board_grid = BoardGridTab()
        for widget, title in zip(
            (self.scatter, self.radial, self.angle, self.board_grid), TAB_TITLES
        ):
            self.tabs.addTab(widget, title)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        band = self.control_band
        band.channel_requested.connect(self._on_band_channel_requested)
        band.step_requested.connect(self._on_band_step_requested)
        band.options_changed.connect(self._on_band_options_changed)
        band.fit_all_clicked.connect(self._on_fit_all_triggered)
        band.fit_channel_clicked.connect(self._on_fit_channel_clicked)
        band.fit_board_clicked.connect(self._on_fit_board_clicked)
        band.use_channel_options_clicked.connect(self._on_use_channel_options)
        band.batch_options_clicked.connect(self._on_batch_options)
        self.scatter.point_cap_changed.connect(self._on_view_settings_changed)
        self.scatter.display_changed.connect(self._on_view_settings_changed)
        self.radial.display_changed.connect(self._on_view_settings_changed)
        self.angle.display_changed.connect(self._on_view_settings_changed)
        self.board_grid.display_changed.connect(self._on_view_settings_changed)
        self.board_grid.channel_activated.connect(self.select_channel)

    def _action(
        self, text: str, slot: Callable[..., object], *, shortcut: str | None = None, tip: str = ""
    ) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            action.setStatusTip(tip)
            action.setToolTip(tip)
        action.triggered.connect(slot)
        return action

    def _create_actions(self) -> None:
        self.open_action = self._action(
            "Open", self._on_open_any, tip="Open a raw .dat file or a UV cache"
        )
        self.open_raw_action = self._action(
            "Open &Raw...",
            self._on_open_raw,
            shortcut="Ctrl+O",
            tip="Open a raw .dat file (builds its UV cache, or reuses a valid one)",
        )
        self.open_cache_action = self._action(
            "Open &Cache...",
            self._on_open_cache,
            shortcut="Ctrl+Shift+O",
            tip="Open a UV cache (.uv.h5) directly",
        )
        self.export_tec_action = self._action(
            "Export .&tec...", self._on_export_tec, tip=_EXPORT_TEC_TIP
        )
        self.export_csv_action = self._action(
            "Export &CSV...", self._on_export_csv, tip=_EXPORT_CSV_TIP
        )
        self.export_both_action = self._action(
            "Export &Both...",
            self._on_export_both,
            shortcut="Ctrl+E",
            tip=_EXPORT_BOTH_TIP,
        )
        self.revert_channel_action = self._action(
            "Revert Channel to Batch",
            self._on_revert_channel,
            tip="Delete the selected channel's override: back to its batch result",
        )
        self.revert_board_action = self._action(
            "Revert Board to Batch",
            self._on_revert_board,
            tip="Delete every override of the selected board: back to the batch results",
        )
        self.clear_overrides_action = self._action(
            "Clear All &Overrides...",
            self._on_clear_overrides,
            tip="Delete every channel override: back to the batch results",
        )
        self.exit_action = self._action("E&xit", self.close, shortcut="Ctrl+Q", tip="Quit")
        self.fit_all_action = self._action(
            "&Fit All",
            self._on_fit_all_triggered,
            shortcut="Ctrl+F",
            tip=_FIT_ALL_TIP,
        )
        self.stop_action = self._action(
            "&Stop", self.stop_current, tip="Stop the running cache build, Fit All or board re-fit"
        )
        self.reset_layout_action = self._action(
            "Reset Layout",
            self.reset_layout,
            tip="Restore the default window size and dock layout",
        )
        self.focus_map_action = self._action(
            "Focus System Map",
            self._focus_system_map,
            shortcut="Ctrl+M",
            tip="Give the System Map's board strip the keyboard focus",
        )
        self.about_action = self._action("&About", self._show_about, tip="About uvcorr")

    def _create_menus(self) -> None:
        menubar = self.menuBar()
        assert menubar is not None
        file_menu = menubar.addMenu("&File")
        assert file_menu is not None
        file_menu.addAction(self.open_raw_action)
        file_menu.addAction(self.open_cache_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_tec_action)
        file_menu.addAction(self.export_csv_action)
        file_menu.addAction(self.export_both_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        process_menu = menubar.addMenu("&Process")
        assert process_menu is not None
        process_menu.addAction(self.fit_all_action)
        process_menu.addAction(self.stop_action)
        process_menu.addSeparator()
        process_menu.addAction(self.revert_channel_action)
        process_menu.addAction(self.revert_board_action)
        process_menu.addSeparator()
        process_menu.addAction(self.clear_overrides_action)
        self.control_band.set_revert_actions(self.revert_channel_action, self.revert_board_action)
        view_menu = menubar.addMenu("&View")
        assert view_menu is not None
        self.view_menu: QMenu = view_menu
        help_menu = menubar.addMenu("&Help")
        assert help_menu is not None
        help_menu.addAction(self.about_action)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("MainToolbar")  # saveState identifies toolbars by name
        toolbar.setMovable(False)
        toolbar.addAction(self.open_action)
        toolbar.addSeparator()
        toolbar.addAction(self.fit_all_action)
        toolbar.addAction(self.stop_action)
        self.addToolBar(toolbar)
        self.toolbar = toolbar

    def _create_status_bar(self) -> None:
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self._status_bar = status_bar
        self.file_label = QLabel("No file open")  # a summary; the window title names the file
        status_bar.addPermanentWidget(self.file_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMaximumWidth(220)
        self.progress_bar.hide()
        status_bar.addPermanentWidget(self.progress_bar)
        status_bar.showMessage("Open a raw .dat file or a UV cache to start")

    def _create_docks(self) -> None:
        self.system_map = RefitSystemMap()
        self.system_map.set_informational_flags(INFORMATIONAL_FLAGS)
        self.system_map.channel_selected.connect(self._on_map_channel_selected)
        self.system_map.board_selected.connect(self._on_map_board_selected)
        self.system_map.fit_channel_requested.connect(self._on_map_fit_channel)
        self.system_map.fit_board_requested.connect(self._on_map_fit_board)
        self.system_map.revert_channel_requested.connect(self._on_map_revert_channel)
        self.system_map.revert_board_requested.connect(self._on_map_revert_board)
        self.system_map.color_mode_changed.connect(self._on_map_color_mode_changed)
        self.system_map.view_changed.connect(self._on_map_view_changed)
        self.system_map_dock = QDockWidget("System Map", self)
        self.system_map_dock.setObjectName("SystemMapDock")
        self.system_map_dock.setWidget(self.system_map)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.system_map_dock)

        self.inspector = FitInspector()
        self.inspector_dock = QDockWidget("Fit Inspector", self)
        self.inspector_dock.setObjectName("FitInspectorDock")
        self.inspector_dock.setWidget(self.inspector)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.inspector_dock)

    def _populate_view_menu(self) -> None:
        for dock in (self.inspector_dock, self.system_map_dock):
            toggle = dock.toggleViewAction()
            assert toggle is not None
            self.view_menu.addAction(toggle)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.focus_map_action)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.reset_layout_action)

    # ------------------------------------------------------------------
    # Layout and settings
    # ------------------------------------------------------------------

    def _apply_default_layout(self) -> bool:
        """Default docks and a screen-derived size; returns whether the window maximizes."""
        self._apply_default_docks()
        return self._apply_default_size()

    def _apply_default_docks(self) -> None:
        """Both docks shown, docked in their default areas."""
        for dock, area in (
            (self.system_map_dock, Qt.DockWidgetArea.BottomDockWidgetArea),
            (self.inspector_dock, Qt.DockWidgetArea.RightDockWidgetArea),
        ):
            dock.setFloating(False)
            self.addDockWidget(area, dock)
            dock.show()

    def _apply_default_size(self) -> bool:
        """Size the window for its screen (adc2kev's policy) and split the docks by fractions."""
        layout = self.layout()
        assert layout is not None
        layout.activate()
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            available = FALLBACK_SCREEN
        else:
            rect = screen.availableGeometry()
            available = (rect.width(), rect.height())
        hint = self.minimumSizeHint()
        (width, height), maximize = initial_window_size(available, (hint.width(), hint.height()))
        if maximize:
            self.resize(*available)
            self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)
            width, height = available
        else:
            self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMaximized)
            self.resize(width, height)
            if screen is not None:
                frame = self.frameGeometry()
                frame.moveCenter(screen.availableGeometry().center())
                self.move(frame.topLeft())
        if height < SHORT_SCREEN_HEIGHT:
            map_height = min(round(height * _MAP_HEIGHT_FRACTION_SHORT), _MAP_HEIGHT_MAX_SHORT)
        else:
            map_height = round(height * _MAP_HEIGHT_FRACTION)
        self.resizeDocks([self.system_map_dock], [map_height], Qt.Orientation.Vertical)
        self.resizeDocks(
            [self.inspector_dock],
            [max(INSPECTOR_MIN_WIDTH, round(width * _INSPECTOR_WIDTH_FRACTION))],
            Qt.Orientation.Horizontal,
        )
        return maximize

    def _restore_settings(self) -> None:
        settings = app_settings()
        version = read_int(settings, KEY_LAYOUT_VERSION)
        geometry = read_bytes(settings, KEY_GEOMETRY)
        state = read_bytes(settings, KEY_STATE)
        restored = False
        if version == LAYOUT_VERSION and not geometry.isEmpty() and not state.isEmpty():
            state_ok = self.restoreState(state, LAYOUT_VERSION)
            if state_ok:
                restored = True
                if not self.restoreGeometry(geometry):
                    # A geometry saved on a very different screen: keep the docks
                    self._apply_default_size()
            else:
                logger.warning("Ignoring an unreadable window layout in %s", settings.fileName())
        if not restored:
            self._apply_default_layout()

        mode = read_str(settings, KEY_COLOR_MODE)
        if mode in COLOR_MODES:
            self.system_map.set_color_mode(mode)
        view = read_str(settings, KEY_MAP_VIEW)
        if view in (VIEW_ANODES, VIEW_CATHODES):
            self.system_map.set_view(view)
        cap = read_int(settings, KEY_POINT_CAP, DEFAULT_POINT_CAP)
        self.scatter.set_point_cap(cap if cap is not None else DEFAULT_POINT_CAP)
        self.scatter.set_overlay(read_bool(settings, KEY_OVERLAY))
        self.scatter.set_density(read_bool(settings, KEY_DENSITY))
        self.radial.set_common_axis(read_bool(settings, KEY_RADIAL_COMMON_AXIS, True))
        self.angle.set_include_rejected(read_bool(settings, KEY_ANGLE_INCLUDE_REJECTED))
        grid_layout = read_str(settings, KEY_GRID_LAYOUT, LAYOUT_RENA)
        self.board_grid.set_grid_layout(grid_layout if grid_layout in LAYOUTS else LAYOUT_RENA)
        grid_mode = read_str(settings, KEY_GRID_MODE, MODE_BEFORE)
        self.board_grid.set_mode(grid_mode if grid_mode in MODES else MODE_BEFORE)

    def save_settings(self) -> None:
        """Persist the layout, last directory, map mode/view and scatter options."""
        settings = app_settings()
        settings.setValue(KEY_LAYOUT_VERSION, LAYOUT_VERSION)
        settings.setValue(KEY_GEOMETRY, self.saveGeometry())
        settings.setValue(KEY_STATE, self.saveState(LAYOUT_VERSION))
        self._save_view_settings(settings)
        settings.sync()

    def _save_view_settings(self, settings: QSettings | None = None) -> None:
        store = app_settings() if settings is None else settings
        store.setValue(KEY_COLOR_MODE, self.system_map.color_mode)
        store.setValue(KEY_MAP_VIEW, self.system_map.view)
        store.setValue(KEY_POINT_CAP, self.scatter.point_cap)
        store.setValue(KEY_OVERLAY, self.scatter.overlay)
        store.setValue(KEY_DENSITY, self.scatter.density)
        store.setValue(KEY_RADIAL_COMMON_AXIS, self.radial.common_axis)
        store.setValue(KEY_ANGLE_INCLUDE_REJECTED, self.angle.include_rejected)
        store.setValue(KEY_GRID_LAYOUT, self.board_grid.grid_layout)
        store.setValue(KEY_GRID_MODE, self.board_grid.mode)

    def reset_layout(self) -> None:
        """View > Reset Layout: forget the persisted layout and apply the default."""
        settings = app_settings()
        settings.remove(KEY_LAYOUT)
        settings.sync()
        maximize = self._apply_default_layout()
        if self.isVisible():
            if maximize:
                self.showMaximized()
            else:
                self.showNormal()

    def last_directory(self) -> str:
        """The directory of the last opened file (for the file dialogs)."""
        return read_str(app_settings(), KEY_LAST_DIR)

    def _remember_directory(self, path: Path) -> None:
        settings = app_settings()
        settings.setValue(KEY_LAST_DIR, str(path.resolve().parent))
        settings.sync()

    # ------------------------------------------------------------------
    # Status, messages and busy state
    # ------------------------------------------------------------------

    def _status(self, message: str, timeout_ms: int = 0) -> None:
        self._status_bar.showMessage(message, timeout_ms)
        logger.info(message)

    def status_message(self) -> str:
        """The status bar's current message."""
        return str(self._status_bar.currentMessage())

    def _show_error(self, title: str, text: str) -> None:
        """Report an error in a dialog (tests replace this method)."""
        logger.error(f"{title}: {text}")
        QMessageBox.critical(self, title, text)

    def _show_warning(self, title: str, text: str) -> None:
        """Report a problem that did not stop the operation (tests replace this method)."""
        logger.warning(f"{title}: {text}")
        QMessageBox.warning(self, title, text)

    def _show_info(self, title: str, text: str) -> None:
        """Report a finished operation in a dialog (tests replace this method)."""
        logger.info(f"{title}: {text}")
        QMessageBox.information(self, title, text)

    def _ask(self, title: str, text: str) -> bool:
        """Ask a yes/no question (tests replace this method)."""
        reply = QMessageBox.question(
            self,
            title,
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _ask_keep_overrides(self, question: str, details: str) -> bool | None:
        """Ask the Fit All question when overrides are stored (Keep / Discard / Cancel).

        Returns:
            True for Keep, False for Discard, None for Cancel.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Fit All")
        box.setText(question)
        box.setInformativeText(details)
        keep = box.addButton("Keep", QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(keep)
        box.setEscapeButton(cancel)
        try:
            box.exec()
            clicked = box.clickedButton()
        finally:
            box.deleteLater()
        if clicked is keep:
            return True
        if clicked is discard:
            return False
        return None

    def is_busy(self) -> bool:
        """Whether an open/build, Fit All or re-fit is running."""
        return (
            self.open_thread is not None
            or self.fit_all_thread is not None
            or self.refit_thread is not None
            or self.revert_thread is not None
        )

    def _can_start(self, what: str) -> bool:
        if self.is_busy():
            self._status(f"Cannot {what} now: wait for the running operation, or Stop it", 5000)
            return False
        return True

    def _current_board(self) -> tuple[int, int] | None:
        """The selected channel's board, or the board selected without a channel."""
        selection = self.session.selection
        if selection is not None:
            return selection.node, selection.board
        return self._board_only

    def _refit_blocked_reason(self) -> str:
        """Why no re-fit can start now ("" if one can)."""
        if not self.session.is_open:
            return "Open a raw .dat file or a UV cache first"
        if not self.session.has_results:
            return REFIT_NEEDS_BATCH_TIP
        if self.is_busy():
            return "Wait for the running operation to finish"
        return ""

    def _update_actions(self) -> None:
        busy = self.is_busy()
        session = self.session
        for action in (self.open_action, self.open_raw_action, self.open_cache_action):
            action.setEnabled(not busy)
        can_fit = session.is_open and not busy
        if not session.is_open:
            fit_reason = "Open a raw .dat file or a UV cache first"
        else:
            fit_reason = _BUSY_TIP if busy else ""
        self.fit_all_action.setEnabled(can_fit)
        self._set_reason(self.fit_all_action, _FIT_ALL_TIP, fit_reason)
        self.control_band.set_fit_all_enabled(can_fit, fit_reason)
        self.control_band.set_option_buttons_blocked(busy, _BUSY_TIP)

        reason = self._refit_blocked_reason()
        selection = session.selection
        board = self._current_board()
        can_channel = not reason and selection is not None and session.channel_count(selection) > 0
        can_board = not reason and board is not None and bool(session.channels_on_board(*board))
        self.control_band.set_refit_enabled(can_channel, can_board, reason)
        self.system_map.refit_blocked_reason = reason
        self.system_map.reverts_blocked = busy

        channel_override = selection is not None and session.override_options(selection) is not None
        n_board = len(session.override_keys(*board)) if board is not None else 0
        self.revert_channel_action.setEnabled(not busy and channel_override)
        self.revert_channel_action.setText(
            f"Revert {short_title(selection)} to Batch"
            if selection is not None and channel_override
            else "Revert Channel to Batch"
        )
        self.revert_board_action.setEnabled(not busy and n_board > 0)
        self.revert_board_action.setText(
            f"Revert N{board[0]} B{board[1]} to Batch ({_plural(n_board, 'override')})"
            if board is not None and n_board
            else "Revert Board to Batch"
        )
        n_all = len(session.override_keys())
        self.clear_overrides_action.setEnabled(not busy and n_all > 0)
        self.clear_overrides_action.setText(
            f"Clear All &Overrides ({n_all:,})..." if n_all else "Clear All &Overrides..."
        )
        if not session.has_results:
            export_reason = "Nothing to export yet: run Fit All first"
        else:
            export_reason = _BUSY_TIP if busy else ""
        for action, tip in (
            (self.export_tec_action, _EXPORT_TEC_TIP),
            (self.export_csv_action, _EXPORT_CSV_TIP),
            (self.export_both_action, _EXPORT_BOTH_TIP),
        ):
            action.setEnabled(not export_reason)
            self._set_reason(action, tip, export_reason)

        refit = self.refit_thread
        stoppable = (
            self.fit_all_thread is not None
            or (self.open_thread is not None and self._open_stoppable)
            or (refit is not None and refit.request.is_board)
        )
        self.stop_action.setEnabled(stoppable)

    @staticmethod
    def _set_reason(action: QAction, tip: str, reason: str) -> None:
        """Show why a disabled action is disabled (its tooltip and status tip), else ``tip``."""
        text = reason or tip
        action.setToolTip(text)
        action.setStatusTip(text)

    def _update_title(self) -> None:
        title = f"uvcorr {__version__}"
        cache = self.session.cache
        if cache is not None:
            name = (
                self.session.dat_path.name if self.session.dat_path is not None else cache.path.name
            )
            title = f"{name} — {title}"
        self.setWindowTitle(title)

    def _show_progress(self, percent: int | None) -> None:
        """Show the progress bar at ``percent``, or as a busy indicator for None."""
        if percent is None:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(max(0, min(100, percent)))
        self.progress_bar.show()

    def _hide_progress(self) -> None:
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 100)

    @staticmethod
    def _retire(thread: QThread | None) -> None:
        """Wait for a finishing worker (its last signal was queued before ``run`` returned)."""
        if thread is not None:
            thread.wait()

    # ------------------------------------------------------------------
    # Opening files
    # ------------------------------------------------------------------

    def _on_open_any(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Raw Data or UV Cache", self.last_directory(), _ANY_FILTER
        )
        if path:
            self.open_path(path)

    def _on_open_raw(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Raw Data", self.last_directory(), _RAW_FILTER
        )
        if path:
            self.open_raw(path)

    def _on_open_cache(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open UV Cache", self.last_directory(), _CACHE_FILTER
        )
        if path:
            self.open_cache(path)

    def open_path(self, path: str | Path, *, confirm: bool = True) -> bool:
        """Open a raw ``.dat`` file or a UV cache, by its name; returns whether it started."""
        if is_cache_file(path):
            return self.open_cache(path)
        return self.open_raw(path, confirm=confirm)

    def open_raw(self, dat_path: str | Path, *, force: bool = False, confirm: bool = True) -> bool:
        """Open a raw file: reuse its valid cache or build it in a :class:`CacheBuildThread`.

        A rebuild that would discard stored results asks first (``confirm``).

        Returns:
            Whether the open was started.
        """
        if is_cache_file(dat_path):
            # A cache chosen through Open Raw would otherwise be parsed as raw data
            return self.open_cache(dat_path)
        if not self._can_start("open a file"):
            return False
        dat = Path(dat_path)
        try:
            check = inspect_raw(dat)
        except (OSError, UVCacheError) as exc:
            self._show_error("Open Raw Data", str(exc))
            return False
        if confirm and check.discards_results:
            question = (
                f"The UV cache {check.cache_path.name} is out of date for {dat.name} and will "
                "be rebuilt. Its stored results and overrides will be discarded.\n\nRebuild it?"
            )
            if not self._ask("Rebuild the UV cache?", question):
                return False
        thread = CacheBuildThread(dat, check.cache_path, force=force, settings=self.build_settings)
        thread.progress.connect(self._on_build_progress)
        builds = check.needs_build or force
        if builds:
            message = f"Building the UV cache of {dat.name}…"
            self._show_progress(0)
        else:
            message = f"Opening {dat.name} (its UV cache is up to date)…"
            self._show_progress(None)
        self._start_open_thread(thread, message, dat, stoppable=builds)
        return True

    def open_cache(self, cache_path: str | Path) -> bool:
        """Open a UV cache directly in a :class:`CacheOpenThread`; returns whether it started."""
        if not self._can_start("open a file"):
            return False
        path = Path(cache_path)
        if not path.is_file():
            self._show_error("Open UV Cache", f"File not found: {path}")
            return False
        self._show_progress(None)
        self._start_open_thread(CacheOpenThread(path), f"Opening {path.name}…", path)
        return True

    def _start_open_thread(
        self,
        thread: CacheBuildThread | CacheOpenThread,
        message: str,
        path: Path,
        *,
        stoppable: bool = False,
    ) -> None:
        thread.finished.connect(self._on_open_finished)
        thread.error.connect(self._on_open_error)
        thread.stopped.connect(self._on_open_stopped)
        self.open_thread = thread
        self._opening_path = path
        self._open_stoppable = stoppable
        self._open_started = time.perf_counter()
        self._status(message)
        self._update_actions()
        thread.start()

    def _end_open_thread(self) -> None:
        self._retire(self.open_thread)
        self.open_thread = None
        self._open_stoppable = False
        self._hide_progress()
        self._update_actions()

    def _on_build_progress(self, fraction: float) -> None:
        self._show_progress(round(100 * fraction))

    def _on_open_finished(self, opened: OpenedFile) -> None:
        self._end_open_thread()
        try:
            self.session.install(opened)
        except SessionError as exc:
            self._show_error("Open", str(exc))
            return
        self._show_opened()
        if self._opening_path is not None:
            self._remember_directory(self._opening_path)
        self.last_open_seconds = time.perf_counter() - self._open_started
        how = "built the UV cache" if opened.built else "read the UV cache"
        if opened.results_error is not None:
            results = "its stored results are unreadable (ignored)"
        elif opened.stored is not None:
            results = "stored results loaded"
        else:
            results = "not fitted yet"
        message = (
            f"Opened {opened.cache.path.name}: {how}, {results} ({self.last_open_seconds:.1f} s)"
        )
        if opened.stale:
            message += (
                "; warning: the cache is out of date for its raw file (open the .dat to rebuild)"
            )
        self._status(message)
        first = self.session.step_channel(None, +1)
        if first is not None:
            self.select_channel(first)
        if opened.results_error is not None:
            self._show_warning(
                "Stored results unreadable",
                f"The results stored in {opened.cache.path.name} cannot be read, so they are "
                "ignored: the events can be browsed, and Fit All stores new results in "
                f"their place.\n\n{opened.results_error}",
            )

    def _show_opened(self) -> None:
        """Populate every widget from the freshly installed session."""
        session = self.session
        # Loads of the previous file may still be running: drop their results
        self._drop_detail_load()
        self._grid_generation += 1
        self._pending_grid = None
        self._grid_wanted = None
        self._grid_loading = None
        self._grid_dirty = False
        self._board_only = None
        self.system_map.set_state(session.views(), session.boards, session.data_channels)
        self.control_band.set_channels(session.data_channels)
        self.control_band.set_options(session.options)
        self.control_band.set_channel_options(None)
        self.inspector.clear()
        self.scatter.clear()
        self.radial.clear()
        self.angle.clear()
        self.board_grid.clear()
        self.file_label.setText(session.describe())
        cache = session.cache
        self.file_label.setToolTip(str(cache.path) if cache is not None else "")
        self._update_title()
        self._update_actions()

    def _on_open_error(self, message: str) -> None:
        self._end_open_thread()
        self._status("Open failed", 5000)
        self._show_error("Open failed", message)

    def _on_open_stopped(self) -> None:
        self._end_open_thread()
        self._status("Cache build stopped; no cache was written", 5000)

    # ------------------------------------------------------------------
    # Fit All
    # ------------------------------------------------------------------

    def _on_fit_all_triggered(self) -> None:
        self.start_fit_all()

    def start_fit_all(self, *, confirm: bool = True, keep_overrides: bool | None = None) -> bool:
        """Fit every channel in a :class:`FitAllThread` with the band's options.

        Args:
            confirm: Ask before starting. With overrides stored and
                ``keep_overrides`` None, the question is "Keep N channel
                overrides?" with Keep / Discard / Cancel.
            keep_overrides: Keep (True) or discard (False) the stored
                overrides; None asks (``confirm``) or keeps them.

        Returns:
            Whether the run was started.
        """
        if not self.session.is_open:
            self._show_error("Fit All", "Open a raw .dat file or a UV cache first.")
            return False
        if not self._can_start("run Fit All"):
            return False
        options = self.control_band.options()
        n_overrides = len(self.session.override_keys())
        what = (
            f"Fit all {len(self.session.data_channels):,} channels on "
            f"{len(self.session.boards)} boards ({_describe_options(options)}) and store "
            "the results in the cache"
        )
        text = f"{what}?"
        if confirm and n_overrides and keep_overrides is None:
            overrides = _plural(n_overrides, "channel override")
            question = f"{what}, keeping the {overrides}?"
            keep_text = "Keep: the overridden channels keep their own fits"
            n_same = len(self.session.overrides_fitting_like(options))
            if n_same:
                verb = "uses" if n_same == 1 else "use"
                keep_text += (
                    f"; {n_same:,} of the {n_overrides:,} overrides {verb} exactly these "
                    "options and will be dropped (the new batch reproduces them)"
                )
            details = (
                f"This replaces the stored batch results.\n\n{keep_text}.\n\nDiscard: every "
                "channel takes the new batch fit and the overrides are deleted."
            )
            answer = self._ask_keep_overrides(question, details)
            if answer is None:
                return False
            keep = answer
        else:
            keep = True if keep_overrides is None else keep_overrides
            if confirm:
                if self.session.has_results:
                    text += "\n\nThis replaces the stored batch results"
                    if n_overrides:
                        overrides = _plural(n_overrides, "channel override")
                        text += f"; the {overrides} are {'kept' if keep else 'deleted'}"
                    text += "."
                if not self._ask("Fit All", text):
                    return False
        self.session.options = options
        thread = FitAllThread(self.session, options, workers=self.workers, keep_overrides=keep)
        thread.progress.connect(self._on_fit_progress)
        thread.finished.connect(self._on_fit_all_finished)
        thread.error.connect(self._on_fit_all_error)
        thread.stopped.connect(self._on_fit_all_stopped)
        self.fit_all_thread = thread
        self._show_progress(0)
        self._status(f"Fit All ({_describe_options(options)})…")
        self._update_actions()
        thread.start()
        return True

    def _end_fit_all_thread(self) -> None:
        self._retire(self.fit_all_thread)
        self.fit_all_thread = None
        self._hide_progress()
        self._update_actions()

    def _on_fit_progress(self, done: int, total: int) -> None:
        """Show Fit All progress; ``done``/``total`` may be boards or events (any size)."""
        percent = int(100 * done / total) if total else 0
        self._show_progress(percent)
        self._status_bar.showMessage(f"Fit All: {percent}%")

    def _on_fit_all_finished(self, outcome: BatchOutcome) -> None:
        self._end_fit_all_thread()
        try:
            views = self.session.apply_batch(outcome)
        except SessionError as exc:
            self._show_error("Fit All", str(exc))
            return
        self.system_map.set_state(views, self.session.boards, self.session.data_channels)
        self.control_band.set_options(outcome.options)
        self.file_label.setText(self.session.describe())
        n = len(outcome.stored.results)
        message = (
            f"Fit All: {n:,} channels in {outcome.seconds:.1f} s ({outcome.workers} worker(s)); "
            "results stored in the cache"
        )
        n_kept = len(outcome.stored.overrides)
        if not outcome.keep_overrides:
            if outcome.n_dropped:
                message += f"; {_plural(outcome.n_dropped, 'override')} discarded"
        else:
            if n_kept:
                message += f"; {_plural(n_kept, 'override')} kept"
            if outcome.n_dropped:
                message += (
                    f"; {_plural(outcome.n_dropped, 'override')} with exactly these options "
                    "dropped"
                )
        self._status(message)
        self._want_board_grid(self._grid_wanted, refresh=True)
        self._refresh_selection()

    def _on_fit_all_error(self, message: str) -> None:
        self._end_fit_all_thread()
        self._status("Fit All failed; nothing was stored", 5000)
        self._show_error("Fit All failed", message)

    def _on_fit_all_stopped(self) -> None:
        self._end_fit_all_thread()
        self._status("Fit All stopped; nothing was stored", 5000)

    def stop_current(self) -> None:
        """Stop the running cache build, Fit All or board re-fit."""
        stopped = False
        for thread in (self.open_thread, self.fit_all_thread, self.refit_thread):
            if thread is not None and thread.isRunning():
                thread.stop()
                stopped = True
        if stopped:
            self._status("Stopping…")

    def _refresh_selection(self) -> None:
        """Show the selection again after the results changed (band, map, inspector, scatter)."""
        selection = self.session.selection
        if selection is not None:
            self.select_channel(selection)
            return
        if self._board_only is not None:
            node, board = self._board_only
            self.control_band.set_current(None, node=node, board=board)
        self._update_actions()

    def _refresh_views(self, keys: tuple[ChannelKey, ...]) -> None:
        """Update the map for channels whose merged result changed."""
        views = {}
        for key in keys:
            view = self.session.view(key)
            if view is None:  # the channel lost its result: rebuild the whole map
                session = self.session
                self.system_map.set_state(session.views(), session.boards, session.data_channels)
                return
            views[key] = view
        self.system_map.update_views(views)

    def _after_results_change(self, keys: tuple[ChannelKey, ...]) -> None:
        """Refresh everything that shows results after a re-fit or revert changed ``keys``."""
        self._refresh_views(keys)
        self.file_label.setText(self.session.describe())
        shown = {self._grid_wanted, self.board_grid.board}
        if any((key.node, key.board) in shown for key in keys):
            self._want_board_grid(self._grid_wanted, refresh=True)
        selection = self.session.selection
        if selection is not None and selection in keys:
            self.select_channel(selection)  # reloads the scatter with the new options
        elif selection is None:
            self._refresh_selection()
        else:
            self._update_actions()

    # ------------------------------------------------------------------
    # Re-fits (worker thread) and overrides
    # ------------------------------------------------------------------

    def _on_fit_channel_clicked(self) -> None:
        self.refit_channel()

    def _on_fit_board_clicked(self) -> None:
        self.refit_board()

    def refit_channel(self, key: ChannelKey | tuple[int, int, int, int] | None = None) -> bool:
        """Re-fit a channel (default: the selected one) with the band's options.

        Runs in a :class:`RefitThread`; see the module docstring for what is
        stored. Returns whether the re-fit was started.
        """
        target = ChannelKey(*(int(k) for k in key)) if key is not None else self.session.selection
        if target is None:
            self._status("Select a channel to re-fit", 5000)
            return False
        return self._start_refit(channel=target)

    def refit_board(self, node: int | None = None, board: int | None = None) -> bool:
        """Re-fit every channel of a board (default: the selected one) with the band's options.

        Returns whether the re-fit was started.
        """
        if node is None or board is None:
            current = self._current_board()
            if current is None:
                self._status("Select a board to re-fit", 5000)
                return False
            node, board = current
        return self._start_refit(board=(int(node), int(board)))

    def _start_refit(
        self, *, channel: ChannelKey | None = None, board: tuple[int, int] | None = None
    ) -> bool:
        what = "Fit Board" if board is not None else "Fit Channel"
        if not self._can_start("re-fit"):
            return False
        options = self.control_band.options()
        try:
            request = self.session.refit_request(options, channel=channel, board=board)
        except (SessionError, UVCacheError) as exc:
            self._show_error(what, str(exc))
            return False
        thread = RefitThread(self.session, request)
        thread.progress.connect(self._on_refit_progress)
        thread.finished.connect(self._on_refit_finished)
        thread.error.connect(self._on_refit_error)
        thread.stopped.connect(self._on_refit_stopped)
        self.refit_thread = thread
        if request.is_board:
            self._show_progress(0)
        else:
            self._show_progress(None)
        self._status(request.describe_start())
        self._update_actions()
        thread.start()
        return True

    def _end_refit_thread(self) -> RefitThread | None:
        thread = self.refit_thread
        self._retire(thread)
        self.refit_thread = None
        self._hide_progress()
        self._update_actions()
        return thread

    def _on_refit_progress(self, done: int, total: int) -> None:
        thread = self.refit_thread
        if thread is None or not thread.request.is_board:
            return
        self._show_progress(int(100 * done / total) if total else 0)
        self._status_bar.showMessage(f"Fit Board {thread.request.title}: {done}/{total} channels")

    def _on_refit_finished(self, outcome: RefitOutcome) -> None:
        self._end_refit_thread()
        try:
            changed = self.session.apply_refit(outcome)
        except SessionError as exc:
            self._show_error("Re-fit", str(exc))
            return
        self.last_refit = outcome
        self._status(outcome.describe())
        self._after_results_change(changed)

    def _on_refit_error(self, message: str) -> None:
        thread = self._end_refit_thread()
        title = thread.request.title if thread is not None else ""
        self._status(f"Re-fit of {title} failed; nothing was stored", 5000)
        self._show_error("Re-fit failed", message)

    def _on_refit_stopped(self) -> None:
        thread = self._end_refit_thread()
        title = thread.request.title if thread is not None else ""
        self._status(f"Re-fit of {title} stopped; nothing was stored", 5000)

    def _on_use_channel_options(self) -> None:
        """Load the selected channel's override options into the band."""
        selection = self.session.selection
        options = self.session.override_options(selection) if selection is not None else None
        if selection is None or options is None:
            return
        self.control_band.set_options(options)
        self.session.options = options
        note = self.session.override_change(selection)
        self._status(f"Options of the override of {short_title(selection)} loaded ({note})")

    def _on_batch_options(self) -> None:
        """Load the stored batch options (the defaults before a batch) into the band."""
        batch = self.session.batch_options
        options = batch if batch is not None else FitOptions()
        self.control_band.set_options(options)
        self.session.options = options
        if batch is None:
            self._status("Default options loaded (no batch results yet)")
        else:
            self._status(f"Batch options loaded ({_describe_options(options)})")

    def revert_channel(self, key: ChannelKey | tuple[int, int, int, int] | None = None) -> bool:
        """Delete a channel's override (default: the selected channel's) in a worker thread.

        Returns:
            Whether the revert was started (False without an override).
        """
        target = ChannelKey(*(int(k) for k in key)) if key is not None else self.session.selection
        if target is None:
            return False
        return self._start_revert(
            lambda: self.session.revert_request([target]), f"{short_title(target)} has no override"
        )

    def revert_board(self, node: int | None = None, board: int | None = None) -> bool:
        """Delete every override of a board (default: the selected one) in a worker thread."""
        if node is None or board is None:
            current = self._current_board()
            if current is None:
                return False
            node, board = current
        n, b = int(node), int(board)
        return self._start_revert(
            lambda: self.session.revert_request(board=(n, b)), f"N{n} B{b} has no overrides"
        )

    def clear_overrides(self, *, confirm: bool = True) -> bool:
        """Delete every override (after asking) in a worker thread; returns whether it started."""
        n = len(self.session.override_keys())
        if n == 0:
            self._status("There are no overrides to clear", 5000)
            return False
        if not self._can_start("clear the overrides"):
            return False
        if confirm and not self._ask(
            "Clear all overrides?",
            f"Delete all {_plural(n, 'channel override')} from the cache? Those channels go "
            "back to their batch results. This cannot be undone.",
        ):
            return False
        return self._start_revert(self.session.revert_request, "There are no overrides to clear")

    def _start_revert(self, make_request: Callable[[], RevertRequest | None], nothing: str) -> bool:
        """Start a :class:`RevertThread` (the write can wait for a busy file)."""
        if not self._can_start("revert overrides"):
            return False
        try:
            request = make_request()
        except (SessionError, UVCacheError) as exc:
            self._show_error("Revert to batch", str(exc))
            return False
        if request is None:
            self._status(nothing, 5000)
            return False
        thread = RevertThread(self.session, request)
        thread.finished.connect(self._on_revert_finished)
        thread.error.connect(self._on_revert_error)
        self.revert_thread = thread
        self._show_progress(None)
        what = "every override" if request.is_all else request.title
        self._status(f"Reverting {what} to batch…")
        self._update_actions()
        thread.start()
        return True

    def _end_revert_thread(self) -> None:
        self._retire(self.revert_thread)
        self.revert_thread = None
        self._hide_progress()
        self._update_actions()

    def _on_revert_finished(self, outcome: RevertOutcome) -> None:
        self._end_revert_thread()
        try:
            removed = self.session.apply_revert(outcome)
        except SessionError as exc:
            self._show_error("Revert to batch", str(exc))
            return
        self._status(outcome.describe())
        self._after_results_change(removed)

    def _on_revert_error(self, message: str) -> None:
        self._end_revert_thread()
        self._status("Revert failed; nothing was deleted", 5000)
        self._show_error("Revert to batch", message)

    def _on_revert_channel(self) -> None:
        self.revert_channel()

    def _on_revert_board(self) -> None:
        self.revert_board()

    def _on_clear_overrides(self) -> None:
        self.clear_overrides()

    def _follow_map_selection(self, node: int, board: int) -> None:
        """Show what the map's context menu selected for a board action (it selects silently).

        The right-clicked channel if the menu was opened on one of the
        board's cells, else the current channel if it is on the board, else
        the board alone.
        """
        channel = self.system_map.current_channel
        if channel is not None and (channel.node, channel.board) == (node, board):
            key = ChannelKey(*channel)
            if self.session.selection != key:
                self.select_channel(key)
            return
        selection = self.session.selection
        if selection is not None and (selection.node, selection.board) == (node, board):
            self.system_map.set_current_channel(selection)
            return
        self._on_map_board_selected(node, board)

    def _show_menu_channel(self, channel: ChannelAddress) -> ChannelKey:
        """Select the channel a context-menu action applies to; returns its key."""
        key = ChannelKey(*channel)
        if self.session.selection != key:
            self.select_channel(key)
        return key

    def _on_map_fit_channel(self, channel: ChannelAddress) -> None:
        self.refit_channel(self._show_menu_channel(channel))

    def _on_map_fit_board(self, node: int, board: int) -> None:
        self._follow_map_selection(node, board)
        self.refit_board(node, board)

    def _on_map_revert_channel(self, channel: ChannelAddress) -> None:
        self.revert_channel(self._show_menu_channel(channel))

    def _on_map_revert_board(self, node: int, board: int) -> None:
        self._follow_map_selection(node, board)
        self.revert_board(node, board)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_directory(self) -> Path:
        """Where the export dialogs start: the last export directory, else the cache's.

        The cache's directory (next to the raw file for a cache built by
        default) rather than the raw file recorded in a cache opened
        directly, which may be a shared data directory.
        """
        remembered = read_str(app_settings(), KEY_EXPORT_DIR)
        if remembered and Path(remembered).is_dir():
            return Path(remembered)
        cache = self.session.cache
        if cache is not None and cache.path.parent.is_dir():
            return cache.path.parent
        last = self.last_directory()
        return Path(last) if last else Path.home()

    def _remember_export_directory(self, directory: Path) -> None:
        settings = app_settings()
        settings.setValue(KEY_EXPORT_DIR, str(directory.resolve()))
        settings.sync()

    def _can_export(self) -> bool:
        if not self.session.has_results:
            self._show_error("Export", "Nothing to export yet: run Fit All first.")
            return False
        return self._can_start("export")

    @staticmethod
    def _with_suffix(path: str, suffix: str) -> Path:
        """``path``, with ``suffix`` added when the name has no extension at all."""
        chosen = Path(path)
        return chosen if chosen.suffix else chosen.with_name(chosen.name + suffix)

    def _on_export_tec(self) -> None:
        if not self._can_export():
            return
        default = self.export_directory() / f"{self.session.export_stem}.tec"
        path, _ = QFileDialog.getSaveFileName(self, "Export .tec", str(default), _TEC_FILTER)
        if path:
            self.export_tec(self._with_suffix(path, ".tec"))

    def _on_export_csv(self) -> None:
        if not self._can_export():
            return
        default = self.export_directory() / SUMMARY_CSV_NAME
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", str(default), _CSV_FILTER)
        if path:
            self.export_csv(self._with_suffix(path, ".csv"))

    def _on_export_both(self) -> None:
        if not self._can_export():
            return
        directory = QFileDialog.getExistingDirectory(
            self,
            f"Export {self.session.export_stem}.tec and {SUMMARY_CSV_NAME} to",
            str(self.export_directory()),
        )
        if directory:
            self.export_both(directory)

    def export_tec(self, path: str | Path) -> ExportSummary | None:
        """Write the merged results' ``.tec`` file; returns what was written (None on error)."""
        target = Path(path)
        return self._export("Export .tec", lambda: self.session.export_tec(target), target.parent)

    def export_csv(self, path: str | Path) -> ExportSummary | None:
        """Write the merged results' ``radial_summary.csv``; returns what was written."""
        target = Path(path)
        return self._export("Export CSV", lambda: self.session.export_csv(target), target.parent)

    def export_both(self, directory: str | Path) -> ExportSummary | None:
        """Write ``<stem>.tec`` and ``radial_summary.csv`` to ``directory``."""
        target = Path(directory)
        return self._export(
            "Export .tec and CSV", lambda: self.session.export_outputs(target), target
        )

    def _export(
        self, title: str, action: Callable[[], ExportSummary], directory: Path
    ) -> ExportSummary | None:
        try:
            summary = action()
        except (SessionError, OSError, ValueError) as exc:
            self._status(f"{title} failed", 5000)
            self._show_error(f"{title} failed", str(exc))
            return None
        self.last_export = summary
        self._remember_export_directory(directory)
        counts = (
            f"{_plural(summary.n_rows, 'row')}, {_plural(summary.n_ok, 'ok block')}, "
            f"{_plural(summary.n_overrides, 'override')} applied"
        )
        names = " and ".join(path.name for path in summary.paths)
        self._status(f"Exported {names} ({counts}) in {summary.seconds:.2f} s")
        paths = "\n".join(f"  {path}" for path in summary.paths)
        text = (
            f"Wrote the batch results with the overrides applied:\n{paths}\n\n{counts} "
            "(.tec: ok channels only; CSV: every channel)."
        )
        if summary.warning:
            self._show_warning(title, f"{text}\n\nWarning: {summary.warning}")
        else:
            self._show_info(title, text)
        return summary

    # ------------------------------------------------------------------
    # Channel selection
    # ------------------------------------------------------------------

    def select_channel(self, key: ChannelKey | tuple[int, int, int, int]) -> None:
        """Show a channel everywhere: band, map, inspector and the tabs.

        The channel is loaded in the background (scatter first, then the
        Radial and Radius vs angle tabs); the Board grid follows its board.
        The map is always told (``set_current_channel`` is idempotent and
        emits nothing), so it cannot drift from the rest of the window
        whichever widget made the selection.
        """
        if not self.session.is_open:
            return
        channel = self.session.select(key)
        assert channel is not None
        self._board_only = None
        result = self.session.result(channel)
        n_events = self.session.channel_count(channel)
        override_note = self.session.override_change(channel)
        self.control_band.set_current(channel, result, n_events, override_note=override_note)
        self.control_band.set_channel_options(
            self.session.override_options(channel), override_note or ""
        )
        self.system_map.set_current_channel(channel)
        used = self.session.options_for(channel)
        self.inspector.show_channel(
            channel,
            result,
            used[0] if used is not None else None,
            self._options_note(used[1] if used is not None else None, override_note),
            n_events,
        )
        title = channel_title(channel)
        self.scatter.show_loading(title)
        self.radial.show_loading(title)
        self.angle.show_loading(title)
        self.board_grid.set_selected(channel)
        self._request_detail(channel)
        self._want_board_grid((channel.node, channel.board))
        self._update_actions()

    def _options_note(self, source: str | None, override_note: str | None = None) -> str:
        stored = self.session.stored
        if source is None or stored is None:
            return ""
        if source == OPTIONS_OVERRIDE:
            return override_note or "the channel's own options"
        return f"Fit All of {stored.created_at[:16].replace('T', ' ')}"

    def _on_map_channel_selected(self, channel: ChannelAddress) -> None:
        self.select_channel(ChannelKey(*channel))

    def _on_map_board_selected(self, node: int, board: int) -> None:
        current = self.session.selection
        if current is not None and (current.node, current.board) == (node, board):
            return
        self.session.select(None)
        self._board_only = (node, board)
        self._drop_detail_load()
        has_data = bool(self.session.channels_on_board(node, board))
        self.control_band.set_current(None, node=node, board=board)
        self.control_band.set_channel_options(None)
        text = f"Node {node} Board {board}: " + (
            "pick a channel" if has_data else "no events on this board"
        )
        self.inspector.clear(text)
        self.scatter.clear(text)
        self.radial.clear(text)
        self.angle.clear(text)
        self.board_grid.set_selected(None)
        if has_data:
            self._want_board_grid((node, board))
        else:
            self._want_board_grid(None)
            self.board_grid.clear(text)
        self._update_actions()

    def _on_band_channel_requested(self, key: ChannelKey) -> None:
        self.select_channel(key)

    def _on_band_step_requested(self, delta: int) -> None:
        """Prev/Next: step from the channel, or from the board shown without a channel."""
        selection = self.session.selection
        if selection is None and self._board_only is not None:
            target = self.session.step_from_board(*self._board_only, delta)
        else:
            target = self.session.step_channel(selection, delta)
        if target is not None and target != selection:
            self.select_channel(target)

    def _on_band_options_changed(self, options: FitOptions) -> None:
        self.session.options = options

    # ------------------------------------------------------------------
    # Scatter data (worker thread)
    # ------------------------------------------------------------------

    @property
    def detail_pending(self) -> bool:
        """Whether a channel load (scatter, radial and angle data) is running or queued."""
        return self._detail_thread is not None or self._pending_detail is not None

    @property
    def grid_pending(self) -> bool:
        """Whether a Board grid load is running or queued."""
        return self._grid_thread is not None or self._pending_grid is not None

    def _drop_detail_load(self) -> None:
        """Supersede the channel load in flight (its results are dropped, its views skipped)."""
        self._detail_generation += 1
        self._pending_detail = None
        if self._detail_thread is not None:
            self._detail_thread.skip_views()

    def _request_detail(self, key: ChannelKey) -> None:
        self._detail_generation += 1
        self._switch_started = time.perf_counter()
        request = self.session.detail_request(key)
        if self._detail_thread is not None:
            self._detail_thread.skip_views()  # its channel is superseded
            self._pending_detail = (self._detail_generation, request)
            return
        self._start_detail(self._detail_generation, request)

    def _start_detail(self, generation: int, request: DetailRequest) -> None:
        thread = ChannelDetailThread(self.session, request, generation)
        thread.done.connect(self._on_detail_done)
        thread.views.connect(self._on_detail_views)
        thread.failed.connect(self._on_detail_failed)
        self._detail_thread = thread
        thread.start()

    def _finish_detail_thread(self) -> None:
        self._retire(self._detail_thread)
        self._detail_thread = None
        pending, self._pending_detail = self._pending_detail, None
        if pending is not None and pending[0] == self._detail_generation:
            self._start_detail(*pending)

    def _on_detail_done(self, generation: int, detail: ChannelDetail) -> None:
        """First stage: draw the scatter (the thread goes on with the other tabs' data)."""
        if generation == self._detail_generation:
            self.scatter.set_detail(detail)
            self.last_switch_seconds = time.perf_counter() - self._switch_started
            logger.debug(
                f"{detail.key}: {detail.n_events:,} points loaded in {detail.seconds:.3f} s, "
                f"shown after {self.last_switch_seconds:.3f} s"
            )
        elif self._detail_thread is not None:
            self._detail_thread.skip_views()

    def _on_detail_views(self, generation: int, views: DetailViews | None) -> None:
        """Second (last) stage: draw the Radial and Radius vs angle tabs."""
        if views is not None and generation == self._detail_generation:
            note = f"Could not compute this view: {views.error}"
            if views.radial is not None:
                self.radial.show_data(views.radial)
            else:
                self.radial.clear(note)
            if views.angle is not None:
                self.angle.show_data(views.angle)
            else:
                self.angle.clear(note)
            self.last_views_seconds = time.perf_counter() - self._switch_started
            logger.debug(
                f"Radial and angle data computed in {views.seconds:.3f} s, shown after "
                f"{self.last_views_seconds:.3f} s"
            )
        self._finish_detail_thread()

    def _on_detail_failed(self, generation: int, message: str) -> None:
        if generation == self._detail_generation:
            text = f"Could not load the channel: {message}"
            self.scatter.clear(text)
            self.radial.clear(text)
            self.angle.clear(text)
            self._status(text, 8000)
        self._finish_detail_thread()

    # ------------------------------------------------------------------
    # Board grid (worker thread, loaded lazily)
    # ------------------------------------------------------------------

    def _want_board_grid(self, board: tuple[int, int] | None, *, refresh: bool = False) -> None:
        """Note the board the Board grid should show (``refresh``: its results changed).

        It is loaded at once if the Board grid tab is current, else when the
        tab is shown (:meth:`_update_board_grid`).
        """
        self._grid_wanted = board
        if refresh:
            self._grid_dirty = True
        if board is None:
            self._grid_generation += 1  # drop a load in flight
            self._grid_loading = None
            self._pending_grid = None
        self._update_board_grid()

    def _on_tab_changed(self, _index: int) -> None:
        self._update_board_grid()

    def _update_board_grid(self) -> None:
        """Load the wanted board into the Board grid if it is current and out of date."""
        wanted = self._grid_wanted
        if wanted is None or self.tabs.currentWidget() is not self.board_grid:
            return
        if not self._grid_dirty and (
            self._grid_loading == wanted
            or (self._grid_loading is None and self.board_grid.board == wanted)
        ):
            return
        self._grid_dirty = False
        self._grid_generation += 1
        self._grid_loading = wanted
        self._grid_started = time.perf_counter()
        node, board = wanted
        self.board_grid.show_loading(f"N{node} B{board}")
        request = (self._grid_generation, node, board)
        if self._grid_thread is not None:
            self._pending_grid = request
            return
        self._start_grid(*request)

    def _start_grid(self, generation: int, node: int, board: int) -> None:
        # The results are snapshot here, on the GUI thread
        cache = self.session.cache
        if cache is None:
            return
        results = {key: self.session.result(key) for key in board_keys(node, board)}
        thread = BoardGridThread(self.session, cache, node, board, results, generation)
        thread.done.connect(self._on_grid_done)
        thread.failed.connect(self._on_grid_failed)
        self._grid_thread = thread
        thread.start()

    def _finish_grid_thread(self) -> None:
        self._retire(self._grid_thread)
        self._grid_thread = None
        pending, self._pending_grid = self._pending_grid, None
        if pending is not None and pending[0] == self._grid_generation:
            self._start_grid(*pending)

    def _on_grid_done(self, generation: int, data: BoardGridData) -> None:
        if generation == self._grid_generation:
            self._grid_loading = None
            self.board_grid.set_selected(self.session.selection)
            self.board_grid.show_data(data)
            self.last_grid_seconds = time.perf_counter() - self._grid_started
            logger.debug(
                f"Board grid of N{data.node} B{data.board} computed in {data.seconds:.3f} s, "
                f"shown after {self.last_grid_seconds:.3f} s"
            )
        self._finish_grid_thread()

    def _on_grid_failed(self, generation: int, message: str) -> None:
        if generation == self._grid_generation:
            self._grid_loading = None
            self.board_grid.clear(f"Could not load the board: {message}")
        self._finish_grid_thread()

    # ------------------------------------------------------------------
    # Map, scatter settings, misc
    # ------------------------------------------------------------------

    def _on_map_color_mode_changed(self, _mode: str) -> None:
        self._save_view_settings()

    def _on_map_view_changed(self, _view: str) -> None:
        self._save_view_settings()

    def _on_view_settings_changed(self, *_args: object) -> None:
        self._save_view_settings()

    def _focus_system_map(self) -> None:
        if self.system_map_dock.isHidden():
            self.system_map_dock.show()
        self.system_map_dock.raise_()
        self.system_map.board_strip.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About uvcorr",
            f"<b>uvcorr {__version__}</b><br>RENA-3 fine-timing U/V ellipse correction "
            "from raw PET acquisition files.<br><br>Built on adc2kev.",
        )

    def request_quit(self) -> None:
        """Close without asking, stopping any running work (Ctrl+C in the terminal)."""
        self._quit_requested = True
        self.close()

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Stop running work (after asking), wait for the workers and save the settings."""
        if event is None:
            return
        running = [
            thread
            for thread in (
                self.open_thread,
                self.fit_all_thread,
                self.refit_thread,
                self.revert_thread,
            )
            if thread is not None and thread.isRunning()
        ]
        if running:
            if not self._quit_requested and not self._ask(
                "Quit uvcorr?",
                "A cache build, Fit All or re-fit is still running. Stop it and quit?",
            ):
                event.ignore()
                return
            for thread in running:
                thread.blockSignals(True)
                thread.stop()
            for thread in running:
                thread.wait()
            self.open_thread = None
            self.fit_all_thread = None
            self.refit_thread = None
            self.revert_thread = None
        for loader in (self._detail_thread, self._grid_thread):
            if loader is not None:
                loader.blockSignals(True)
                loader.wait()
        self._detail_thread = None
        self._grid_thread = None
        self._pending_detail = None
        self._pending_grid = None
        self.save_settings()
        event.accept()
