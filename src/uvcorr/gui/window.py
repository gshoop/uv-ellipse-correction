"""The uvcorr main window (plan section 9); the entry point is :mod:`uvcorr.gui.main`.

Built on the adc2kev 2.2.7 GUI patterns (``adc2kev/gui/main.py`` and
``docs/implementation/GUI_LAYOUT.md``): a ``QMainWindow`` with docks that
carry object names, ``QSettings`` persistence of the layout, a View menu with
dock toggles and Reset Layout, and ``QThread`` workers with
``progress``/``finished``/``error``/``stopped`` signals and a Stop action.

Layout:

- **Menu bar**: File (Open Raw, Open Cache, Export .tec and Export CSV,
  which arrive in phase 5, Exit), Process (Fit All, Stop), View (dock
  toggles, Focus System Map, Reset Layout), Help (About).
- **Toolbar** (``MainToolbar``): Open, Fit All, Stop.
- **Central widget**: the :class:`~uvcorr.gui.controls.ControlBand` above
  the tabs: Scatter (:class:`~uvcorr.gui.scatter.ScatterTab`), and the Radial,
  Radius vs angle and Board grid tabs as disabled placeholders until phase 5
  (:data:`PHASE5_TABS`; phase 5 replaces the widget of each).
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
control-band navigation) updates the band and the inspector at once and
loads the scatter data in a :class:`~uvcorr.gui.threads.ChannelDetailThread`;
a newer selection supersedes a pending one (generation counter), so stepping
quickly through large channels never queues work.

Selection sync: the map's ``channel_selected`` drives the band, inspector and
scatter; the band drives the map through ``set_current_channel`` /
``set_selection``, which emit nothing, so there is no feedback loop.

Persistence (``~/.config/uvcorr/uvcorr-gui.ini``): window geometry and dock
state (``layout/*``, tagged with ``LAYOUT_VERSION``), the last directory, the
map's colour mode and Anodes/Cathodes view, and the scatter's point cap,
Overlay and Density toggles.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QThread
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
from uvcorr.gui._system_map_model import ChannelAddress
from uvcorr.gui.controls import ControlBand
from uvcorr.gui.inspector import MIN_WIDTH as INSPECTOR_MIN_WIDTH
from uvcorr.gui.inspector import FitInspector
from uvcorr.gui.map_colors import COLOR_MODES
from uvcorr.gui.scatter import DEFAULT_POINT_CAP, ScatterTab
from uvcorr.gui.session import (
    INFORMATIONAL_FLAGS,
    BatchOutcome,
    ChannelDetail,
    DetailRequest,
    OpenedFile,
    SessionError,
    UVSession,
    channel_title,
    inspect_raw,
    is_cache_file,
)
from uvcorr.gui.system_map import VIEW_ANODES, VIEW_CATHODES, SystemMapWidget
from uvcorr.gui.threads import (
    CacheBuildThread,
    CacheOpenThread,
    ChannelDetailThread,
    FitAllThread,
)
from uvcorr.options import FitOptions

logger = logging.getLogger(__name__)

__all__ = ["PHASE5_EXPORT_TIP", "PHASE5_TABS", "MainWindow"]

PHASE5_TABS: tuple[str, ...] = ("Radial", "Radius vs angle", "Board grid")
"""Central tabs that arrive in phase 5 (disabled placeholders until then)."""

PHASE5_EXPORT_TIP = "Export arrives in phase 5 (until then: uvcorr process)"

# Settings keys (layout/* is written with LAYOUT_VERSION)
KEY_LAYOUT = "layout"
KEY_LAYOUT_VERSION = "layout/version"
KEY_GEOMETRY = "layout/geometry"
KEY_STATE = "layout/state"
KEY_LAST_DIR = "session/last_dir"
KEY_COLOR_MODE = "map/color_mode"
KEY_MAP_VIEW = "map/view"
KEY_POINT_CAP = "scatter/point_cap"
KEY_OVERLAY = "scatter/overlay"
KEY_DENSITY = "scatter/density"

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


def _describe_options(options: FitOptions) -> str:
    """Short description of the options the band edits (for dialogs and the status bar)."""
    robust = f"robust k={options.clip_k:g}, {options.max_iter} iter" if options.robust else "plain"
    geometric = ", geometric" if options.geometric else ""
    return f"{robust}{geometric}, min events {options.min_events:,}"


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
        self._detail_thread: ChannelDetailThread | None = None
        self._detail_generation = 0
        self._pending_detail: tuple[int, DetailRequest] | None = None
        # (node, board) shown without a channel (a map click on a board tile)
        self._board_only: tuple[int, int] | None = None
        self._open_started = 0.0
        self._opening_path: Path | None = None
        self._open_stoppable = False
        self._quit_requested = False
        self._switch_started = 0.0
        self.last_open_seconds: float | None = None
        self.last_switch_seconds: float | None = None

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
        self.tabs.addTab(self.scatter, "Scatter")
        self.placeholder_tabs: dict[str, QWidget] = {}
        for title in PHASE5_TABS:
            placeholder = QLabel(f"{title}: coming in phase 5")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            index = self.tabs.addTab(placeholder, title)
            self.tabs.setTabEnabled(index, False)
            self.tabs.setTabToolTip(index, "Coming in phase 5")
            self.placeholder_tabs[title] = placeholder
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        band = self.control_band
        band.channel_requested.connect(self._on_band_channel_requested)
        band.step_requested.connect(self._on_band_step_requested)
        band.options_changed.connect(self._on_band_options_changed)
        band.fit_all_clicked.connect(self._on_fit_all_triggered)
        band.fit_channel_clicked.connect(lambda: self._phase5_stub("Fit Channel"))
        band.fit_board_clicked.connect(lambda: self._phase5_stub("Fit Board"))
        self.scatter.point_cap_changed.connect(self._on_scatter_settings_changed)
        self.scatter.display_changed.connect(self._on_scatter_settings_changed)

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
            "Export .&tec...", lambda: self._phase5_stub("Export .tec"), tip=PHASE5_EXPORT_TIP
        )
        self.export_csv_action = self._action(
            "Export &CSV...", lambda: self._phase5_stub("Export CSV"), tip=PHASE5_EXPORT_TIP
        )
        self.export_tec_action.setEnabled(False)
        self.export_csv_action.setEnabled(False)
        self.exit_action = self._action("E&xit", self.close, shortcut="Ctrl+Q", tip="Quit")
        self.fit_all_action = self._action(
            "&Fit All",
            self._on_fit_all_triggered,
            shortcut="Ctrl+F",
            tip="Fit every channel with the control band's options and store the results",
        )
        self.stop_action = self._action(
            "&Stop", self.stop_current, tip="Stop the running cache build or Fit All"
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
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        process_menu = menubar.addMenu("&Process")
        assert process_menu is not None
        process_menu.addAction(self.fit_all_action)
        process_menu.addAction(self.stop_action)
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
        self.system_map = SystemMapWidget()
        self.system_map.set_informational_flags(INFORMATIONAL_FLAGS)
        self.system_map.channel_selected.connect(self._on_map_channel_selected)
        self.system_map.board_selected.connect(self._on_map_board_selected)
        self.system_map.fit_channel_requested.connect(
            lambda _channel: self._phase5_stub("Fit Channel")
        )
        self.system_map.fit_board_requested.connect(
            lambda _node, _board: self._phase5_stub("Fit Board")
        )
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

    def is_busy(self) -> bool:
        """Whether an open/build or Fit All is running."""
        return self.open_thread is not None or self.fit_all_thread is not None

    def _can_start(self, what: str) -> bool:
        if self.is_busy():
            self._status(f"Cannot {what} now: wait for the running operation, or Stop it", 5000)
            return False
        return True

    def _update_actions(self) -> None:
        busy = self.is_busy()
        for action in (self.open_action, self.open_raw_action, self.open_cache_action):
            action.setEnabled(not busy)
        can_fit = self.session.is_open and not busy
        self.fit_all_action.setEnabled(can_fit)
        self.control_band.set_fit_all_enabled(can_fit)
        stoppable = self.fit_all_thread is not None or (
            self.open_thread is not None and self._open_stoppable
        )
        self.stop_action.setEnabled(stoppable)

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

    def _phase5_stub(self, what: str) -> None:
        self._status(f"{what} arrives in phase 5; use Fit All for now", 5000)

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
        # A scatter load of the previous file may still be running: drop its result
        self._detail_generation += 1
        self._pending_detail = None
        self._board_only = None
        self.system_map.set_state(session.views(), session.boards, session.data_channels)
        self.control_band.set_channels(session.data_channels)
        self.control_band.set_options(session.options)
        self.inspector.clear()
        self.scatter.clear()
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

    def start_fit_all(self, *, confirm: bool = True) -> bool:
        """Fit every channel in a :class:`FitAllThread` with the band's options.

        Returns:
            Whether the run was started.
        """
        if not self.session.is_open:
            self._show_error("Fit All", "Open a raw .dat file or a UV cache first.")
            return False
        if not self._can_start("run Fit All"):
            return False
        options = self.control_band.options()
        if confirm:
            text = (
                f"Fit all {len(self.session.data_channels):,} channels on "
                f"{len(self.session.boards)} boards ({_describe_options(options)}) and store "
                "the results in the cache?"
            )
            if self.session.has_results:
                text += "\n\nThis replaces the stored batch results; overrides are kept."
            if not self._ask("Fit All", text):
                return False
        self.session.options = options
        thread = FitAllThread(self.session, options, workers=self.workers)
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
        self._status(
            f"Fit All: {n:,} channels in {outcome.seconds:.1f} s ({outcome.workers} worker(s)); "
            "results stored in the cache"
        )
        selection = self.session.selection
        if selection is not None:
            self.select_channel(selection)

    def _on_fit_all_error(self, message: str) -> None:
        self._end_fit_all_thread()
        self._status("Fit All failed; nothing was stored", 5000)
        self._show_error("Fit All failed", message)

    def _on_fit_all_stopped(self) -> None:
        self._end_fit_all_thread()
        self._status("Fit All stopped; nothing was stored", 5000)

    def stop_current(self) -> None:
        """Stop the running cache build or Fit All."""
        stopped = False
        for thread in (self.open_thread, self.fit_all_thread):
            if thread is not None and thread.isRunning():
                thread.stop()
                stopped = True
        if stopped:
            self._status("Stopping…")

    # ------------------------------------------------------------------
    # Channel selection
    # ------------------------------------------------------------------

    def select_channel(self, key: ChannelKey | tuple[int, int, int, int]) -> None:
        """Show a channel everywhere: band, map, inspector and scatter.

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
        self.control_band.set_current(channel, result, n_events)
        self.system_map.set_current_channel(channel)
        used = self.session.options_for(channel)
        self.inspector.show_channel(
            channel,
            result,
            used[0] if used is not None else None,
            self._options_note(used[1] if used is not None else None),
            n_events,
        )
        self.scatter.show_loading(channel_title(channel))
        self._request_detail(channel)

    def _options_note(self, source: str | None) -> str:
        stored = self.session.stored
        if source is None or stored is None:
            return ""
        if source == OPTIONS_OVERRIDE:
            return "the channel's own options"
        return f"Fit All of {stored.created_at[:16].replace('T', ' ')}"

    def _on_map_channel_selected(self, channel: ChannelAddress) -> None:
        self.select_channel(ChannelKey(*channel))

    def _on_map_board_selected(self, node: int, board: int) -> None:
        current = self.session.selection
        if current is not None and (current.node, current.board) == (node, board):
            return
        self.session.select(None)
        self._board_only = (node, board)
        self._detail_generation += 1  # drop any scatter load still in flight
        has_data = bool(self.session.channels_on_board(node, board))
        self.control_band.set_current(None, node=node, board=board)
        text = f"Node {node} Board {board}: " + (
            "pick a channel" if has_data else "no events on this board"
        )
        self.inspector.clear(text)
        self.scatter.clear(text)

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
        """Whether a scatter load is running or queued."""
        return self._detail_thread is not None or self._pending_detail is not None

    def _request_detail(self, key: ChannelKey) -> None:
        self._detail_generation += 1
        self._switch_started = time.perf_counter()
        request = self.session.detail_request(key)
        if self._detail_thread is not None:
            self._pending_detail = (self._detail_generation, request)
            return
        self._start_detail(self._detail_generation, request)

    def _start_detail(self, generation: int, request: DetailRequest) -> None:
        thread = ChannelDetailThread(self.session, request, generation)
        thread.done.connect(self._on_detail_done)
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
        current = generation == self._detail_generation
        if current:
            self.scatter.set_detail(detail)
            self.last_switch_seconds = time.perf_counter() - self._switch_started
            logger.debug(
                f"{detail.key}: {detail.n_events:,} points loaded in {detail.seconds:.3f} s, "
                f"shown after {self.last_switch_seconds:.3f} s"
            )
        self._finish_detail_thread()

    def _on_detail_failed(self, generation: int, message: str) -> None:
        if generation == self._detail_generation:
            self.scatter.clear(f"Could not load the channel: {message}")
            self._status(f"Could not load the channel: {message}", 8000)
        self._finish_detail_thread()

    # ------------------------------------------------------------------
    # Map, scatter settings, misc
    # ------------------------------------------------------------------

    def _on_map_color_mode_changed(self, _mode: str) -> None:
        self._save_view_settings()

    def _on_map_view_changed(self, _view: str) -> None:
        self._save_view_settings()

    def _on_scatter_settings_changed(self, *_args: object) -> None:
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
            for thread in (self.open_thread, self.fit_all_thread)
            if thread is not None and thread.isRunning()
        ]
        if running:
            if not self._quit_requested and not self._ask(
                "Quit uvcorr?", "A cache build or Fit All is still running. Stop it and quit?"
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
        if self._detail_thread is not None:
            self._detail_thread.blockSignals(True)
            self._detail_thread.wait()
            self._detail_thread = None
        self._pending_detail = None
        self.save_settings()
        event.accept()
