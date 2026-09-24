"""Tests for the Scatter tab, Fit Inspector and control band (``uvcorr.gui``)."""

from __future__ import annotations

import math
import re
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPalette
from PyQt6.QtWidgets import QHeaderView
from pytestqt.qtbot import QtBot

from uvcorr.analysis import RESULT_COLUMNS, ChannelKey, ChannelResult, analyze_channel
from uvcorr.channels import electrode_label
from uvcorr.ellipse import EllipseParams, correct, ellipse_points, fit_ellipse
from uvcorr.gui import scatter as scatter_module
from uvcorr.gui._contrast import MIN_TEXT_CONTRAST, contrast_ratio, readable_color
from uvcorr.gui.controls import ControlBand
from uvcorr.gui.inspector import (
    GROUP_FLAGS,
    GROUP_OPTIONS,
    FitInspector,
    describe_flag,
    format_value,
    group_of,
)
from uvcorr.gui.inspector import MIN_WIDTH as INSPECTOR_MIN_WIDTH
from uvcorr.gui.map_colors import CATEGORY_COLORS
from uvcorr.gui.scatter import (
    DEFAULT_POINT_CAP,
    MAX_POINT_CAP,
    MUTED_COLOR,
    ScatterTab,
    density_grid,
    density_image,
    subsample_indices,
)
from uvcorr.gui.session import ChannelDetail
from uvcorr.options import FLAG_EXTREME_AXIS_RATIO, FLAG_HIGH_REJECTION, FitOptions

pytestmark = pytest.mark.gui

KEY = ChannelKey(2, 16, 0, 27)
PARAMS = EllipseParams(cx=2030.0, cy=2025.0, a=440.0, b=400.0, phi=0.3)


def _ring(n: int, outliers: int = 0, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = ellipse_points(PARAMS, rng.uniform(0, 2 * math.pi, n))
    u = u + rng.normal(0, 2.0, n)
    v = v + rng.normal(0, 2.0, n)
    if outliers:
        u[:outliers] = rng.uniform(1700, 2300, outliers)
        v[:outliers] = rng.uniform(1700, 2300, outliers)
    return u, v


def make_detail(n: int = 5000, outliers: int = 50, fitted: bool = True) -> ChannelDetail:
    """A ChannelDetail built like the session builds one."""
    u, v = _ring(n, outliers)
    if not fitted:
        return ChannelDetail(KEY, u, v, None, None, None, None, None, None, True, "no fit", 0.0)
    fit = fit_ellipse(u, v)
    assert fit.params is not None
    u_corr, v_corr = correct(u, v, fit.params)
    return ChannelDetail(
        key=KEY,
        u=u,
        v=v,
        result=None,
        options=FitOptions(),
        params=fit.params,
        kept=fit.mask_used,
        u_corr=u_corr,
        v_corr=v_corr,
        consistent=True,
        message="",
        seconds=0.0,
    )


def _items(plot: pg.PlotWidget, kind: type) -> list:
    return [item for item in plot.getPlotItem().items if isinstance(item, kind)]


@pytest.fixture
def scatter(qtbot: QtBot) -> ScatterTab:
    tab = ScatterTab()
    qtbot.addWidget(tab)
    tab.resize(900, 500)
    return tab


# ---------------------------------------------------------------------------
# Scatter helpers
# ---------------------------------------------------------------------------


def test_subsample_is_deterministic_per_channel() -> None:
    assert subsample_indices(100, 200, (1, 2, 3, 4)) is None
    a = subsample_indices(10_000, 500, (1, 15, 0, 5))
    b = subsample_indices(10_000, 500, (1, 15, 0, 5))
    c = subsample_indices(10_000, 500, (1, 15, 0, 6))
    assert a is not None and b is not None and c is not None
    assert a.shape == (500,) and np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.all(np.diff(a) > 0)  # sorted, no repeats


def test_density_grid_is_aligned_to_the_adc_lattice() -> None:
    x0, y0, width, bins = density_grid(2030.4, 2025.6, 500.0, lattice=(0.0, 0.0))
    assert width == math.ceil(1000 / 256) == 4.0
    # Edges half-way between integers: each bin holds exactly `width` integer values
    assert (x0 - 0.5) == int(x0 - 0.5) and (y0 - 0.5) == int(y0 - 0.5)
    assert x0 <= 2030.4 - 500 and x0 + bins * width >= 2030.4 + 500
    # Raw minus a fractional centre: the lattice is shifted, and so are the edges
    x0, _, _, _ = density_grid(0.0, 0.0, 500.0, lattice=(-2030.4, -2025.6))
    assert (x0 + 2030.4 - 0.5) == pytest.approx(round(x0 + 2030.4 - 0.5))
    # Integer data: every bin of a uniform patch holds the same count
    u, v = np.meshgrid(np.arange(1600, 2400), np.arange(1600, 2400))
    x0, y0, width, bins = density_grid(2000.0, 2000.0, 300.0, lattice=(0.0, 0.0))
    counts = density_image(u.ravel(), v.ravel(), x0, y0, bins * width, bins)
    inner = counts[2:-2, 2:-2]
    assert inner.min() == inner.max() == width * width
    assert density_grid(0.0, 0.0, 10.0)[2] == 1.0  # never narrower than 1 ADC


def test_density_counts_points_in_range(scatter: ScatterTab) -> None:
    detail = make_detail()
    far = replace(detail, u=detail.u.copy(), v=detail.v.copy())
    far.u[:10] = 0.0  # ten points far outside the ring's square
    far.v[:10] = 0.0
    scatter.set_density(True)
    scatter.set_detail(far)
    assert scatter.density_counted == 4990
    assert "density of 4,990 of 5,000 points in range" in scatter.info_text()


def test_density_image_matches_histogram2d() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(0, 10, 5000), rng.normal(5, 10, 5000)
    counts = density_image(x, y, -30.0, -25.0, 60.0, 40)
    expected, _, _ = np.histogram2d(x, y, bins=40, range=[[-30, 30], [-25, 35]])
    # Same binning except points exactly on the upper edge (none here)
    assert counts.shape == (40, 40)
    assert np.array_equal(counts, expected.astype(np.int64))


# ---------------------------------------------------------------------------
# Scatter tab
# ---------------------------------------------------------------------------


def _drawn(plot: pg.PlotWidget) -> dict[str, int]:
    """Points drawn per brush colour class: kept (coloured) and rejected (grey)."""
    counts = {"kept": 0, "rejected": 0}
    for item in _items(plot, pg.ScatterPlotItem):
        if item.data.shape[0] == 1 and item.opts["symbol"] == "+":
            continue  # the centre marker
        brush = item.opts["brush"].color().getRgb()
        counts["rejected" if brush == MUTED_COLOR else "kept"] += item.data.shape[0]
    return counts


def test_scatter_points_with_rejected_and_fit(scatter: ScatterTab) -> None:
    detail = make_detail()
    scatter.set_detail(detail)
    n_rej = detail.n_rejected
    assert scatter.point_cap == DEFAULT_POINT_CAP and n_rej > 0
    assert scatter.shown_points == 5000 and scatter.shown_rejected == n_rej
    assert f"showing {5000 - n_rej:,} of {5000 - n_rej:,} kept points" in scatter.info_text()
    assert f"all {n_rej} rejected shown (grey)" in scatter.info_text()
    assert _drawn(scatter.raw_plot) == {"kept": 5000 - n_rej, "rejected": n_rej}
    assert _drawn(scatter.corr_plot) == {"kept": 5000 - n_rej, "rejected": n_rej}
    assert _items(scatter.raw_plot, pg.PlotCurveItem)  # ellipse and semi-axes
    assert _items(scatter.corr_plot, pg.PlotCurveItem)  # target circle
    assert not scatter.overlay_active and scatter.message() == ""


def test_scatter_cap_subsamples_only_kept_points(scatter: ScatterTab) -> None:
    detail = make_detail(n=5000, outliers=800)
    n_rej = detail.n_rejected
    assert n_rej >= 700
    scatter.set_detail(detail)
    scatter.set_point_cap(1000)
    assert scatter.point_cap == 1000
    # Every rejected point is still drawn, on top of the 1000 kept ones
    assert _drawn(scatter.raw_plot) == {"kept": 1000, "rejected": n_rej}
    assert f"showing 1,000 of {5000 - n_rej:,} kept points" in scatter.info_text()
    assert f"all {n_rej} rejected shown" in scatter.info_text()
    first = [item.data["x"].copy() for item in _items(scatter.raw_plot, pg.ScatterPlotItem)]
    scatter.set_point_cap(1000)  # same subsample again
    again = [item.data["x"] for item in _items(scatter.raw_plot, pg.ScatterPlotItem)]
    assert all(np.array_equal(a, b) for a, b in zip(first, again))


def test_scatter_rejected_cap(scatter: ScatterTab, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scatter_module, "REJECTED_CAP", 100)
    detail = make_detail(n=5000, outliers=800)
    scatter.set_detail(detail)
    assert _drawn(scatter.raw_plot)["rejected"] == 100
    assert f"100 of {detail.n_rejected} rejected shown (grey)" in scatter.info_text()


def test_scatter_point_cap_limit(scatter: ScatterTab) -> None:
    scatter.set_point_cap(2_000_000)
    assert scatter.point_cap == MAX_POINT_CAP <= 300_000


def test_scatter_overlay_and_density(scatter: ScatterTab, qtbot: QtBot) -> None:
    scatter.set_detail(make_detail())
    with qtbot.waitSignal(scatter.display_changed):
        scatter.overlay_check.setChecked(True)
    assert scatter.overlay_active
    overlay_points = _items(scatter.overlay_plot, pg.ScatterPlotItem)
    assert sum(item.data.shape[0] for item in overlay_points) == 2 * 5000 + 1
    # The raw points are centred on the ellipse centre
    (x0, x1), (y0, y1) = scatter.overlay_plot.getViewBox().viewRange()
    assert x0 < 0 < x1 and y0 < 0 < y1

    scatter.set_density(True)
    assert scatter.shown_points == 0 and "density of all 5,000 points" in scatter.info_text()
    assert scatter.density_counted == 5000
    images = _items(scatter.overlay_plot, pg.ImageItem)
    assert len(images) == 2
    assert all(image.paintMode == QPainter.CompositionMode.CompositionMode_Plus for image in images)
    scatter.set_overlay(False)
    assert not scatter.overlay_active
    assert len(_items(scatter.raw_plot, pg.ImageItem)) == 1
    assert len(_items(scatter.corr_plot, pg.ImageItem)) == 1


def test_scatter_without_ellipse(scatter: ScatterTab) -> None:
    scatter.set_overlay(True)
    scatter.set_detail(make_detail(n=300, fitted=False))
    assert not scatter.overlay_active  # nothing to centre on
    assert "overlay needs a fitted ellipse" in scatter.info_text()
    assert scatter.message() == "no fit"
    assert scatter.shown_points == 300
    assert not _items(scatter.corr_plot, pg.ScatterPlotItem)
    assert "no ellipse" in scatter.corr_plot.getPlotItem().titleLabel.text
    scatter.clear("gone")
    assert scatter.detail is None and scatter.message() == "gone"
    assert scatter.info_text() == "No channel selected"


def test_scatter_panels_are_linked(scatter: ScatterTab) -> None:
    detail = make_detail()
    scatter.set_detail(detail)
    assert detail.params is not None
    scatter.raw_plot.setRange(xRange=(2400, 2500), yRange=(2000, 2100), padding=0.0)
    (x0, x1), (y0, y1) = scatter.corr_plot.getViewBox().viewRange()
    cx = 0.5 * (x0 + x1) + detail.params.cx
    cy = 0.5 * (y0 + y1) + detail.params.cy
    assert cx == pytest.approx(2450, abs=5) and cy == pytest.approx(2050, abs=5)


def test_scatter_million_points_is_capped(scatter: ScatterTab) -> None:
    """A 1M-point channel draws only the cap (and the density mode bins all points)."""
    n = 1_000_000
    rng = np.random.default_rng(3)
    u, v = ellipse_points(PARAMS, rng.uniform(0, 2 * math.pi, n))
    kept = np.ones(n, dtype=bool)
    u_corr, v_corr = correct(u, v, PARAMS)
    detail = ChannelDetail(
        KEY, u, v, None, FitOptions(), PARAMS, kept, u_corr, v_corr, True, "", 0.0
    )
    scatter.set_detail(detail)
    assert scatter.shown_points == DEFAULT_POINT_CAP
    assert f"showing {DEFAULT_POINT_CAP:,} of 1,000,000 kept points" in scatter.info_text()
    scatter.set_density(True)
    image = _items(scatter.raw_plot, pg.ImageItem)[0]
    bins = image.image.shape[0]
    assert image.image.shape == (bins, bins) and bins <= 256 and image.image.max() > 0


# ---------------------------------------------------------------------------
# Fit Inspector
# ---------------------------------------------------------------------------


def _result(outliers: int = 60) -> ChannelResult:
    u, v = _ring(800, outliers, seed=5)
    return analyze_channel(ChannelKey(1, 16, 0, 12), u, v)


def test_format_value() -> None:
    columns = {column.name: column for column in RESULT_COLUMNS}
    assert format_value(columns["n_events"], 946025) == "946,025"
    assert format_value(columns["node"], 2) == "2"
    assert format_value(columns["centerU"], 2030.548243) == "2030.54824 ADC"
    assert format_value(columns["phi"], math.pi / 4) == "0.785398 rad (45.00°)"
    assert format_value(columns["phi"], -0.760796992) == "-0.760797 rad (-43.59°)"
    assert format_value(columns["timing_jitter_ns"], 0.51234567) == "0.512346 ns"
    assert format_value(columns["phase_ks"], 0.01) == "0.01"
    assert format_value(columns["flags"], ()) == "none"
    assert format_value(columns["post_sigma"], None) == ""
    assert group_of("pre_sigma").startswith("Raw radii") and group_of("brand_new") == "Other"


def test_describe_flag_uses_the_thresholds() -> None:
    assert "10%" in describe_flag(FLAG_HIGH_REJECTION, FitOptions(high_rejection_frac=0.1))
    assert "0.4" in describe_flag(FLAG_EXTREME_AXIS_RATIO, FitOptions(extreme_axis_ratio=0.4))
    assert describe_flag("nonsense", None) == "Unknown flag."


def test_inspector_shows_every_column(qtbot: QtBot) -> None:
    inspector = FitInspector()
    qtbot.addWidget(inspector)
    result = _result()
    assert FLAG_HIGH_REJECTION in result.flags
    options = FitOptions(clip_k=3.0)
    inspector.show_channel(result.key, result, options, "Fit All of today")
    for column in RESULT_COLUMNS:
        assert inspector.value_text(column.name) == format_value(
            column, getattr(result, column.name)
        )
    assert inspector.value_text("status") == "ok"
    assert inspector.flags_shown() == list(result.flags)
    titles = inspector.group_titles()
    assert titles[0] == "Channel" and f"{GROUP_FLAGS}: {len(result.flags)}" in titles
    assert f"{GROUP_OPTIONS}: batch (Fit All of today)" in titles
    assert inspector.value_text("option:clip_k") == "3"
    assert inspector.value_text("option:robust") == "on"
    assert "N1 B16 R0 Ch12" in inspector.title_label.text()


def test_inspector_without_result(qtbot: QtBot) -> None:
    inspector = FitInspector()
    qtbot.addWidget(inspector)
    inspector.show_channel(ChannelKey(1, 15, 0, 5), None, None, n_events=40)
    assert inspector.value_text("status") == "not fitted"
    assert inspector.value_text("n_events") == "40"
    assert inspector.value_text("electrode") == "C02"
    assert inspector.value_text("post_sigma") is None
    inspector.clear("Nothing")
    assert inspector.group_titles() == [] and inspector.title_label.text() == "Nothing"


# ---------------------------------------------------------------------------
# Control band
# ---------------------------------------------------------------------------

BAND_CHANNELS = [
    ChannelKey(1, 15, 0, 5),
    ChannelKey(1, 15, 0, 12),
    ChannelKey(1, 16, 0, 5),
    ChannelKey(1, 16, 1, 9),
    ChannelKey(4, 29, 0, 12),
]


@pytest.fixture
def band(qtbot: QtBot) -> ControlBand:
    widget = ControlBand()
    qtbot.addWidget(widget)
    widget.set_channels(BAND_CHANNELS)
    return widget


def test_band_shows_the_current_channel(band: ControlBand, qtbot: QtBot) -> None:
    result = _result()
    with qtbot.assertNotEmitted(band.channel_requested):
        band.set_current(ChannelKey(1, 15, 0, 12), result, 800)
    assert band.node_combo.currentData() == 1 and band.board_combo.currentData() == 15
    assert band.channel_combo.currentText() == f"R0 Ch12 · {electrode_label(15, 0, 12)}"
    text = band.label_text()
    assert "N1 B15 R0 Ch12" in text and "800 events" in text and "ok" in text
    band.set_current(None, node=1, board=15)
    assert band.channel_combo.currentIndex() == -1
    assert "Node 1 Board 15" in band.label_text()


def test_band_navigation_requests(band: ControlBand, qtbot: QtBot) -> None:
    band.set_current(BAND_CHANNELS[1])
    # Another board: the same RENA/channel if it has events, else the first channel
    with qtbot.waitSignal(band.channel_requested) as blocker:
        band.board_combo.activated.emit(band.board_combo.findData(16))
    assert blocker.args == [ChannelKey(1, 16, 0, 5)]
    band.set_current(BAND_CHANNELS[1])
    with qtbot.waitSignal(band.channel_requested) as blocker:
        band.node_combo.activated.emit(band.node_combo.findData(4))
    assert blocker.args == [ChannelKey(4, 29, 0, 12)]
    band.set_current(BAND_CHANNELS[0])
    with qtbot.waitSignal(band.channel_requested) as blocker:
        band.channel_combo.activated.emit(1)
    assert blocker.args == [ChannelKey(1, 15, 0, 12)]
    with qtbot.waitSignal(band.step_requested) as blocker:
        band.next_button.click()
    assert blocker.args == [1]


def test_band_options(band: ControlBand, qtbot: QtBot) -> None:
    base = FitOptions(phase_ref_freq_hz=500e3, clip_k=3.0, min_events=50)
    band.set_options(base)
    assert band.options() == base
    with qtbot.waitSignal(band.options_changed) as blocker:
        band.robust_check.setChecked(False)
    assert blocker.args[0] == FitOptions(
        phase_ref_freq_hz=500e3, clip_k=3.0, min_events=50, robust=False
    )
    assert not band.clip_spin.isEnabled() and not band.iter_spin.isEnabled()
    band.min_events_spin.setValue(1)
    assert band.options().min_events == 6  # the algebraic fit needs 6 points
    assert not band.fit_channel_button.isEnabled() and "phase 5" in band.fit_board_button.toolTip()


def test_inspector_columns_fit_the_values(qtbot: QtBot) -> None:
    inspector = FitInspector()
    qtbot.addWidget(inspector)
    inspector.resize(INSPECTOR_MIN_WIDTH, 600)
    inspector.show()
    result = _result()
    inspector.show_channel(result.key, result, FitOptions(), "Fit All of today")
    header = inspector.tree.header()
    assert header is not None
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Interactive
    metrics = inspector.tree.fontMetrics()
    longest = max(metrics.horizontalAdvance(column.name) for column in RESULT_COLUMNS)
    field = inspector.tree.columnWidth(0)
    assert longest < field < longest + 60
    assert inspector.minimumWidth() == INSPECTOR_MIN_WIDTH
    # At the minimum width a long value still fits in the Value column
    value_width = inspector.tree.viewport().width() - field
    assert metrics.horizontalAdvance("-0.760797 rad (-43.59°)") < value_width
    # A value that does not fit is available in full as a tooltip
    item = inspector.tree.findItems("centerU", Qt.MatchFlag.MatchRecursive, 0)[0]
    assert item.toolTip(1) == inspector.value_text("centerU")


def test_inspector_no_data_channel(qtbot: QtBot) -> None:
    inspector = FitInspector()
    qtbot.addWidget(inspector)
    inspector.show_channel(ChannelKey(1, 15, 0, 4), None, None, n_events=0)
    assert inspector.value_text("status") == "no data"


def test_band_no_data_and_readable_status(band: ControlBand) -> None:
    band.set_current(ChannelKey(1, 15, 0, 4), None, 0)
    assert "0 events · no data" in band.label_text()
    band.set_current(BAND_CHANNELS[0], _result(outliers=0), 800)
    background = band.channel_label.palette().color(QPalette.ColorRole.Window)
    match = re.search(r"color:(#[0-9a-f]{6})", band.channel_label.text())
    assert match is not None
    assert contrast_ratio(QColor(match.group(1)), background) >= MIN_TEXT_CONTRAST


def test_band_node_change_keeps_the_board_number(band: ControlBand, qtbot: QtBot) -> None:
    channels = [*BAND_CHANNELS, ChannelKey(4, 15, 0, 5), ChannelKey(4, 16, 0, 5)]
    band.set_channels(channels)
    band.set_current(ChannelKey(1, 16, 0, 5))
    with qtbot.waitSignal(band.channel_requested) as blocker:
        band.node_combo.activated.emit(band.node_combo.findData(4))
    assert blocker.args == [ChannelKey(4, 16, 0, 5)]  # board 16 kept, not node 4's first


def test_readable_color() -> None:
    white, dark = QColor("#ffffff"), QColor("#1e1e1e")
    green = QColor(CATEGORY_COLORS["ok"])
    assert contrast_ratio(green, white) < 3  # the map fill is too faint as text on white
    on_white = readable_color(green, white)
    assert contrast_ratio(on_white, white) >= MIN_TEXT_CONTRAST
    assert on_white.green() > on_white.red()  # still green
    assert readable_color(green, dark) == green  # already fine on the dark map background
    both = readable_color("#f39c12", [white, QColor("#f0f0f0")])
    assert min(contrast_ratio(both, g) for g in (white, QColor("#f0f0f0"))) >= 4.5
