"""Tests for the Radial tab (``uvcorr.gui.radial``)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from tests.conftest import RingFiles
from tests.gui.ring_cache import (
    CLEAN,
    ECCENTRIC,
    FIT_FAILED,
    OUTLIERS,
    TOO_FEW,
    copy_files,
    store_batch_results,
)
from uvcorr.analysis import ChannelKey
from uvcorr.ellipse import EllipseParams, correct, ellipse_points
from uvcorr.gui._tab_data import CENTRE_FITTED, CENTRE_MEDIAN, fmt
from uvcorr.gui.radial import (
    PANEL_POST,
    PANEL_PRE,
    STAT_FIELDS,
    RadialTab,
    compute_radial_view,
    panel_html,
)
from uvcorr.gui.session import ChannelDetail, UVSession, load_cache
from uvcorr.metrics import radial_fit_detail, radial_stats
from uvcorr.options import FitOptions

pytestmark = pytest.mark.gui

NO_EVENTS = ChannelKey(1, 15, 0, 4)  # active, on a board with data, no events


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory, _ring_files_master: RingFiles) -> UVSession:
    """A session on a copy of the ring cache with stored batch results (read only here)."""
    _, cache = copy_files(
        _ring_files_master.dat, _ring_files_master.cache, tmp_path_factory.mktemp("radial")
    )
    store_batch_results(cache)
    s = UVSession()
    s.install(load_cache(cache))
    return s


@pytest.fixture
def tab(qtbot: QtBot) -> RadialTab:
    widget = RadialTab()
    qtbot.addWidget(widget)
    widget.resize(1000, 600)
    return widget


def _independent_radii(session: UVSession, key: ChannelKey) -> tuple[np.ndarray, np.ndarray]:
    """Pre and post radii computed from the cache and the stored ellipse, without the tab."""
    result = session.result(key)
    assert result is not None and result.params is not None
    u16, v16 = session.channel_data(key)
    u, v = u16.astype(np.float64), v16.astype(np.float64)
    p = result.params
    pre = np.hypot(u - p.cx, v - p.cy)
    u_corr, v_corr = correct(u, v, p)
    return pre, np.hypot(u_corr, v_corr)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", [CLEAN, OUTLIERS, ECCENTRIC])
def test_view_equals_the_fit_routine_and_the_stored_result(session: UVSession, key) -> None:
    result = session.result(key)
    assert result is not None
    data = compute_radial_view(session.channel_detail(key))
    assert data.centre_kind == CENTRE_FITTED and data.message == ""
    assert data.pre is not None and data.post is not None
    for panel, radii in zip((data.pre, data.post), _independent_radii(session, key)):
        expected = radial_fit_detail(radii)
        # The histogram and curve drawn are exactly what radial_fit_detail fitted
        np.testing.assert_array_equal(panel.edges, expected.edges)
        np.testing.assert_array_equal(panel.counts, expected.counts)
        assert panel.fit.stats == expected.stats == radial_stats(radii).gauss
        assert panel.fitted_histogram
        if expected.stats.ok:
            lo, hi = expected.fit_range
            assert panel.curve_x[0] == lo and panel.curve_x[-1] == hi
            np.testing.assert_allclose(panel.curve_y, expected.curve(panel.curve_x))
        # ... and the numbers are the stored pre_* / post_* columns (all events used)
        for field in STAT_FIELDS:
            stored = getattr(result, f"{panel.name}_{field}")
            computed = panel.value(field)
            if stored is None:
                assert not math.isfinite(computed)
            else:
                assert computed == pytest.approx(stored, rel=1e-12, abs=1e-12)
        assert panel.n_values == result.n_events
        assert panel.matches_stored is True
    assert data.matches_stored is True
    assert [m[0] for m in data.pre.markers] == ["b", "a"]
    assert data.post.markers == (("√(ab)", pytest.approx(result.target_radius)),)


def test_text_box_shows_the_stored_values(session: UVSession, tab: RadialTab) -> None:
    result = session.result(OUTLIERS)
    assert result is not None
    tab.show_data(compute_radial_view(session.channel_detail(OUTLIERS)))
    for name in (PANEL_PRE, PANEL_POST):
        html = tab.stats_html(name)
        assert "= the stored result" in html
        for field in ("mean", "sigma", "fwhm", "robust_sigma", "skewness", "kurtosis"):
            assert fmt(getattr(result, f"{name}_{field}")) in html, (name, field)
    post = tab.stats_html(PANEL_POST)
    assert "μ" in post and "χ²/ndf" in post and "ex. kurt." in post
    assert fmt(result.post_chi2ndf) in post
    assert fmt(result.target_radius) in post and "√(ab)" in post
    # The unbinned sample mean and std are shown too (plan 5.3), even with a good fit
    radii = _independent_radii(session, OUTLIERS)[1]
    cell = "</td><td style='white-space:nowrap' align='right'>"
    assert "<i>unbinned (sample)</i>" in post
    assert f">mean{cell}{fmt(float(np.mean(radii)))}<" in post
    assert f">std{cell}{fmt(float(np.std(radii, ddof=1)))}<" in post


def test_text_box_counts_the_fitted_bins(session: UVSession) -> None:
    data = compute_radial_view(session.channel_detail(OUTLIERS))
    for panel in data.panels:
        if not panel.stats.ok:
            continue
        lo, hi = panel.fit.fit_range
        centers = 0.5 * (panel.edges[:-1] + panel.edges[1:])
        k = int(np.count_nonzero((centers >= lo) & (centers <= hi)))
        assert panel.n_fit_bins == k and 5 <= k < panel.edges.size - 1
        width = panel.edges[1] - panel.edges[0]
        assert panel.bin_width == pytest.approx(width)
        html = panel_html(panel)
        assert f"histogram: {panel.edges.size - 1} bins of {fmt(width, 3)}" in html
        assert f"fit over [{fmt(lo, 5)}, {fmt(hi, 5)}]: {k} bins (ndf {k - 3})" in html


def test_mismatch_with_the_stored_result_is_reported(session: UVSession) -> None:
    detail = session.channel_detail(CLEAN)
    assert detail.result is not None and detail.result.post_sigma is not None
    tampered = replace(detail.result, post_sigma=detail.result.post_sigma * 1.01)
    data = compute_radial_view(replace(detail, result=tampered))
    assert data.matches_stored is False
    assert data.post is not None and data.post.mismatches == ("sigma",)
    assert "differ from the stored result" in data.message
    assert "differs from the stored result: sigma" in panel_html(data.post)


def _detail(u: np.ndarray, v: np.ndarray, params: EllipseParams | None) -> ChannelDetail:
    u_corr = v_corr = None
    if params is not None:
        u_corr, v_corr = correct(u, v, params)
    return ChannelDetail(
        key=CLEAN,
        u=u,
        v=v,
        result=None,
        options=FitOptions(),
        params=params,
        kept=None,
        u_corr=u_corr,
        v_corr=v_corr,
        consistent=True,
        message="",
        seconds=0.0,
    )


def test_failed_fits_show_labelled_sample_statistics() -> None:
    params = EllipseParams(cx=2000.0, cy=2000.0, a=500.0, b=480.0, phi=0.2)
    rng = np.random.default_rng(3)
    # 30 points: too few for the fit routine, a display-only histogram is drawn
    u, v = ellipse_points(params, rng.uniform(0, 2 * math.pi, 30))
    data = compute_radial_view(_detail(u, v, params))
    assert data.post is not None and not data.post.fitted_histogram
    assert data.post.counts.sum() == 30 and data.post.curve_x.size == 0
    assert data.post.stored is None and data.matches_stored is None
    html = panel_html(data.post)
    assert "Gaussian fit failed" in html and "fewer than 50 values" in html
    assert "μ, σ = sample mean, std" in html and "display histogram" in html
    assert ">mean<" in html and ">std<" in html and "unbinned (sample)" in html
    assert fmt(float(np.mean(np.hypot(*correct(u, v, params))))) in html
    # Pre radii of an uncorrected ellipse: the failure is expected and said so
    assert data.pre is not None and data.pre.expected_failure
    # Every corrected radius equal: no spread to histogram
    t = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    u2, v2 = ellipse_points(EllipseParams(0.0, 0.0, 400.0, 400.0, 0.0), t)
    circle = EllipseParams(cx=2000.0, cy=2000.0, a=400.0, b=400.0, phi=0.0)
    data = compute_radial_view(_detail(u2 + 2000.0, v2 + 2000.0, circle))
    assert data.post is not None and not data.post.stats.ok
    assert "no spread to histogram" in panel_html(data.post)


def _second_population() -> tuple[ChannelDetail, np.ndarray, np.ndarray]:
    """A ring (b/a 0.9) with 10 % of its points spread inside it; returns the pre/post radii."""
    params = EllipseParams(cx=2000.0, cy=2000.0, a=520.0, b=470.0, phi=0.0)
    rng = np.random.default_rng(5)
    u, v = ellipse_points(params, rng.uniform(0, 2 * math.pi, 20_000))
    u, v = u + rng.normal(0.0, 2.0, u.size), v + rng.normal(0.0, 2.0, u.size)
    inner = rng.uniform(0.2, 0.9, 2_000)
    u[:2_000] = 2000.0 + (u[:2_000] - 2000.0) * inner
    v[:2_000] = 2000.0 + (v[:2_000] - 2000.0) * inner
    pre = np.hypot(u - 2000.0, v - 2000.0)
    post = np.hypot(*correct(u, v, params))
    return _detail(u, v, params), pre, post


def _count_outside(radii: np.ndarray, view: tuple[float, float]) -> int:
    return int(np.count_nonzero((radii < view[0]) | (radii > view[1])))


def test_outside_counts_follow_the_plotted_range() -> None:
    detail, pre_r, post_r = _second_population()
    data = compute_radial_view(detail)
    common = data.common_range
    assert data.pre is not None and data.post is not None and common is not None
    for panel, radii in ((data.pre, pre_r), (data.post, post_r)):
        own = panel.view_range
        assert own is not None
        assert panel.n_outside_view == _count_outside(radii, own)
        assert panel.n_outside_common == _count_outside(radii, common)
        assert common[0] <= own[0] and own[1] <= common[1]  # the union of both panels
        assert panel.outside(False) == (panel.n_outside_view, panel.view_clipped)
        assert panel.outside(True) == (panel.n_outside_common, panel.common_clipped)
    # The post view (mu +- 5 sigma) hides the inner population on its own axis...
    assert data.post.view_clipped and data.post.n_outside_view > 1_900
    # ...and the common range (which reaches down to b) hides fewer post radii
    assert data.post.n_outside_common < data.post.n_outside_view
    html = panel_html(data.post, common_axis=False)
    share = fmt(100.0 * data.post.n_outside_view / data.post.n_values, 3)
    assert f"{data.post.n_outside_view:,} radii ({share} %) outside the plotted range" in html


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


def test_tab_draws_a_fitted_channel(session: UVSession, tab: RadialTab) -> None:
    data = compute_radial_view(session.channel_detail(ECCENTRIC))
    tab.show_data(data)
    assert tab.data is data and tab.message() == ""
    assert "pre radii about the fitted centre" in tab.info_text()
    pre_items = tab._pre_items
    assert pre_items.hist.isVisible() and pre_items.region.isVisible() == data.pre.stats.ok
    # The marker labels are set even when the lines were hidden before (a pyqtgraph quirk)
    labels = [line.label.textItem.toPlainText() for line in pre_items.markers]
    assert labels == ["b", "a"]
    post_line = tab._post_items.markers[0]
    assert post_line.label.textItem.toPlainText() == "√(ab)"
    assert post_line.value() == pytest.approx(data.post.markers[0][1])
    assert not tab._post_items.markers[1].isVisible()


def test_outside_note_matches_the_plotted_range(tab: RadialTab, qtbot: QtBot) -> None:
    tab.resize(930, 420)
    tab.show()
    qtbot.waitExposed(tab)
    detail, pre_r, post_r = _second_population()
    data = compute_radial_view(detail)
    tab.show_data(data)
    for common in (True, False):
        tab.set_common_axis(common)
        for name, radii, shown in zip((PANEL_PRE, PANEL_POST), (pre_r, post_r), tab.x_ranges()):
            panel = data.pre if name == PANEL_PRE else data.post
            assert panel is not None
            n_outside, clipped = panel.outside(common)
            # The count is against the range actually plotted
            assert n_outside == _count_outside(radii, shown), (name, common)
            html = tab.stats_html(name)
            assert (f"{n_outside:,} radii" in html) == clipped, (name, common)


def test_text_boxes_scroll_instead_of_clipping(session: UVSession, qtbot: QtBot) -> None:
    tab = RadialTab()
    qtbot.addWidget(tab)
    tab.resize(930, 420)
    tab.show()
    qtbot.waitExposed(tab)
    tab.show_data(compute_radial_view(session.channel_detail(OUTLIERS)))
    qtbot.wait(10)
    for label, area in ((tab.pre_text, tab.pre_scroll), (tab.post_text, tab.post_scroll)):
        assert area.widget() is label and area.widgetResizable()
        # The whole text is laid out (reachable by scrolling), none of it is cut off
        assert label.height() >= label.heightForWidth(label.width())
        assert area.viewport().height() < 250  # a short tab: the boxes are small


def test_same_radius_axis_toggle(session: UVSession, tab: RadialTab, qtbot: QtBot) -> None:
    tab.show()
    qtbot.waitExposed(tab)
    tab.show_data(compute_radial_view(session.channel_detail(ECCENTRIC)))
    assert tab.common_axis
    pre, post = tab.x_ranges()
    assert pre == pytest.approx(post)
    assert pre == pytest.approx(tab.data.common_range)
    with qtbot.waitSignal(tab.display_changed):
        tab.common_axis_check.setChecked(False)
    pre, post = tab.x_ranges()
    assert (post[1] - post[0]) < 0.5 * (pre[1] - pre[0])  # b/a = 0.4: the post peak is narrow
    with qtbot.assertNotEmitted(tab.display_changed):
        tab.set_common_axis(True)
    pre, post = tab.x_ranges()
    assert pre == pytest.approx(post)


@pytest.mark.parametrize(
    ("key", "phrase"),
    [(TOO_FEW, "Too few events"), (FIT_FAILED, "fit failed")],
)
def test_tab_without_ellipse(session: UVSession, tab: RadialTab, key, phrase: str) -> None:
    data = compute_radial_view(session.channel_detail(key))
    assert data.centre_kind == CENTRE_MEDIAN and data.post is None
    assert data.pre is not None and data.pre.stored is None
    assert data.pre.counts.sum() > 0
    tab.show_data(data)
    assert phrase in tab.message()
    assert "median point" in tab.info_text()
    assert tab._post_items.note.text == "No ellipse fitted: no corrected radii"
    assert "Raw" in tab.stats_html(PANEL_PRE) and tab.stats_html(PANEL_POST) == ""


def test_tab_without_results_and_without_events(
    ring_files: RingFiles, session: UVSession, tab: RadialTab
) -> None:
    fresh = UVSession()
    fresh.install(load_cache(ring_files.cache))
    data = compute_radial_view(fresh.channel_detail(CLEAN))
    assert data.centre_kind == CENTRE_MEDIAN and data.matches_stored is None
    tab.show_data(data)
    assert "Run Fit All" in tab.message()

    empty = compute_radial_view(session.channel_detail(NO_EVENTS))
    assert empty.pre is None and empty.post is None and empty.n_events == 0
    tab.show_data(empty)
    assert "no data" in tab.message()
    assert tab._pre_items.note.text == "No events on this channel"

    tab.show_loading("N1 B15 R0 Ch05")
    assert tab.info_text().startswith("Loading")
    tab.clear("Node 1 Board 15: pick a channel")
    assert tab.data is None and tab.message() == "Node 1 Board 15: pick a channel"
    assert tab.stats_html(PANEL_PRE) == ""
