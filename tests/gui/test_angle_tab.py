"""Tests for the Radius vs angle tab (``uvcorr.gui.angle``)."""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from tests.conftest import RingFiles
from tests.gui.ring_cache import (
    CLEAN,
    FIT_FAILED,
    OUTLIERS,
    TOO_FEW,
    copy_files,
    store_batch_results,
)
from uvcorr.analysis import ChannelKey
from uvcorr.ellipse import EllipseParams, correct
from uvcorr.gui._tab_data import CENTRE_FITTED, CENTRE_MEDIAN, fmt
from uvcorr.gui.angle import (
    N_ANGLE_BINS,
    P2P_MIN_COUNT,
    SET_ALL,
    SET_KEPT,
    AngleProfile,
    AngleTab,
    angle_bin_index,
    angle_profile,
    compute_angle_view,
    set_description,
    summary_text,
)
from uvcorr.gui.session import ChannelDetail, UVSession, load_cache
from uvcorr.options import FitOptions

pytestmark = pytest.mark.gui

NO_EVENTS = ChannelKey(1, 15, 0, 4)


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory, _ring_files_master: RingFiles) -> UVSession:
    """A session on a copy of the ring cache with stored batch results (read only here)."""
    _, cache = copy_files(
        _ring_files_master.dat, _ring_files_master.cache, tmp_path_factory.mktemp("angle")
    )
    store_batch_results(cache)
    s = UVSession()
    s.install(load_cache(cache))
    return s


@pytest.fixture
def tab(qtbot: QtBot) -> AngleTab:
    widget = AngleTab()
    qtbot.addWidget(widget)
    widget.resize(1000, 600)
    return widget


def direct_profile(du: np.ndarray, dv: np.ndarray) -> tuple[np.ndarray, ...]:
    """Mean and std of hypot(du, dv) per 5° bin, one mask per bin (the reference)."""
    angle = np.degrees(np.arctan2(dv, du)) % 360.0
    radius = np.hypot(du, dv)
    count = np.zeros(N_ANGLE_BINS, dtype=np.int64)
    mean = np.full(N_ANGLE_BINS, np.nan)
    std = np.full(N_ANGLE_BINS, np.nan)
    for b in range(N_ANGLE_BINS):
        inside = (angle >= 5.0 * b) & (angle < 5.0 * (b + 1))
        count[b] = np.count_nonzero(inside)
        if count[b]:
            mean[b] = np.mean(radius[inside])
            std[b] = np.std(radius[inside])
    return count, mean, std


def assert_profile(profile: AngleProfile, du: np.ndarray, dv: np.ndarray) -> None:
    count, mean, std = direct_profile(du, dv)
    np.testing.assert_array_equal(profile.count, count)
    np.testing.assert_allclose(profile.mean, mean, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(profile.std, std, rtol=1e-9, atol=1e-9, equal_nan=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def test_angle_profile_matches_direct_binning() -> None:
    rng = np.random.default_rng(12)
    n = 50_000
    # Angles only in [0, 250) degrees: the bins above stay empty (gaps)
    t = np.radians(rng.uniform(0.0, 250.0, n))
    r = 600.0 + 20.0 * np.cos(2 * t) + rng.normal(0.0, 5.0, n)
    du, dv = r * np.cos(t), r * np.sin(t)
    profile = angle_profile(du, dv)
    assert_profile(profile, du, dv)
    assert profile.n_empty == 22 and np.all(np.isnan(profile.mean[50:]))
    assert profile.centers[0] == 2.5 and profile.centers[-1] == 357.5
    # A mask selects the points binned
    mask = rng.random(n) < 0.3
    assert_profile(angle_profile(du, dv, mask), du[mask], dv[mask])
    # Peak-to-peak over the bins with enough points; the standard errors
    used = profile.count >= P2P_MIN_COUNT
    p2p = profile.peak_to_peak()
    assert p2p == pytest.approx(np.nanmax(profile.mean[used]) - np.nanmin(profile.mean[used]))
    np.testing.assert_allclose(
        profile.sem[:50], profile.std[:50] / np.sqrt(profile.count[:50]), rtol=1e-12
    )
    assert angle_profile(du[:1], dv[:1]).peak_to_peak() is None


def test_angle_view_bins_kept_and_all_points(session: UVSession) -> None:
    result = session.result(OUTLIERS)
    assert result is not None and result.n_rejected and result.params is not None
    detail = session.channel_detail(OUTLIERS)
    assert detail.kept is not None
    data = compute_angle_view(detail)
    assert data.centre_kind == CENTRE_FITTED and data.has_mask
    assert data.n_rejected == result.n_rejected
    assert data.target_radius == pytest.approx(result.target_radius)
    assert data.kept is not None and data.all is not None
    assert data.kept.kind == SET_KEPT and data.kept.n_points == result.n_used
    assert data.all.kind == SET_ALL and data.all.n_points == result.n_events
    # Independent reference: the cache's points, the stored ellipse and one mask per bin
    u16, v16 = session.channel_data(OUTLIERS)
    u, v = u16.astype(np.float64), v16.astype(np.float64)
    p = result.params
    u_corr, v_corr = correct(u, v, p)
    kept = detail.kept
    assert_profile(data.all.pre, u - p.cx, v - p.cy)
    assert_profile(data.kept.pre, (u - p.cx)[kept], (v - p.cy)[kept])
    assert data.all.post is not None and data.kept.post is not None
    assert_profile(data.all.post, u_corr, v_corr)
    assert_profile(data.kept.post, u_corr[kept], v_corr[kept])
    assert data.profile_set(False) is data.kept and data.profile_set(True) is data.all
    # The rejected points pull the post bin means; the kept ones are flat
    kept_p2p = data.kept.post.peak_to_peak()
    all_p2p = data.all.post.peak_to_peak()
    assert kept_p2p is not None and all_p2p is not None and kept_p2p < all_p2p


def test_summary_and_set_description(session: UVSession) -> None:
    data = compute_angle_view(session.channel_detail(OUTLIERS))
    assert data.kept is not None and data.kept.post is not None and data.target_radius
    text = summary_text(data, include_rejected=False)
    pre_p2p = data.kept.pre.peak_to_peak()
    post_p2p = data.kept.post.peak_to_peak()
    assert pre_p2p is not None and post_p2p is not None
    pre_err = data.kept.pre.peak_to_peak_error()
    post_err = data.kept.post.peak_to_peak_error()
    assert pre_err is not None and post_err is not None
    assert f"pre {fmt(pre_p2p, 3)} ± {fmt(pre_err, 2)} ADC" in text
    assert f"post {fmt(post_p2p, 3)} ± {fmt(post_err, 2)} ADC" in text
    percent = 100 * post_p2p / data.target_radius
    shown = fmt(percent, 2) if percent < 10 else f"{percent:.0f}"
    assert f"{shown} % of √(ab)" in text
    kept_text = set_description(data, include_rejected=False)
    assert (
        kept_text.startswith("robust-kept points") and f"{data.n_rejected:,} rejected" in kept_text
    )
    all_text = set_description(data, include_rejected=True)
    assert all_text.startswith(f"all {data.n_events:,} points, including")


def test_percentages_never_print_an_exponent() -> None:
    # A tiny pedestal ring (sqrt(ab) ~ 8 ADC) can modulate by more than its radius
    rng = np.random.default_rng(3)
    t = rng.uniform(0.0, 2 * np.pi, 20_000)
    r = 8.0 + 4.0 * np.cos(2 * t) + rng.normal(0.0, 0.5, t.size)
    params = EllipseParams(cx=0.0, cy=0.0, a=8.0, b=8.0, phi=0.0)
    u, v = r * np.cos(t), r * np.sin(t)
    detail = replace(_synthetic_detail(10, 360.0), u=u, v=v, params=params, u_corr=u, v_corr=v)
    text = summary_text(compute_angle_view(detail), include_rejected=True)
    assert "e+" not in text and "% of √(ab)" in text


def test_peak_to_peak_error_uses_the_extreme_bins() -> None:
    rng = np.random.default_rng(21)
    # A dense flat ring and one sparse bin (12 points around 30-35 degrees) far above it
    t = np.radians(rng.uniform(0.0, 360.0, 72_000))
    r = rng.normal(500.0, 1.0, t.size)
    sparse = np.radians(rng.uniform(30.5, 34.5, 12))
    t = np.r_[t[(np.degrees(t) < 30.0) | (np.degrees(t) >= 35.0)], sparse]
    r = np.r_[r[: t.size - 12], rng.normal(530.0, 20.0, 12)]
    profile = angle_profile(r * np.cos(t), r * np.sin(t))
    assert profile.count[6] == 12
    used = np.flatnonzero(profile.count >= P2P_MIN_COUNT)
    hi = used[np.argmax(profile.mean[used])]
    lo = used[np.argmin(profile.mean[used])]
    assert hi == 6
    expected = np.hypot(profile.sem[hi], profile.sem[lo])
    assert profile.peak_to_peak_error() == pytest.approx(expected)
    # The sparse extreme bin dominates: far above a typical bin's error
    typical = profile.typical_sem()
    assert typical is not None and expected > 20 * typical


def test_non_finite_points_are_skipped_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        profile = angle_profile(
            np.array([1.0, np.nan, 1.0, np.inf]), np.array([1.0, 1.0, 0.0, 2.0])
        )
        assert int(profile.count.sum()) == 2
        assert angle_bin_index(np.array([np.nan, -1.0]), np.array([0.0, 0.0])).tolist() == [-1, 36]
        detail = _synthetic_detail(2000, 360.0)
        u = detail.u.copy()
        u[:10] = np.nan
        nan_detail = replace(
            detail, u=u, u_corr=correct(u, detail.v, detail.params)[0], consistent=True
        )
        data = compute_angle_view(nan_detail)
        assert data.all is not None and data.all.post is not None
        assert int(data.all.pre.count.sum()) == 1990 and int(data.all.post.count.sum()) == 1990
        assert "10 point(s) with a non-finite coordinate are left out" in data.message
        # NaN ellipse parameters (never stored, but must not crash the tab)
        nan_params = EllipseParams(cx=np.nan, cy=np.nan, a=np.nan, b=np.nan, phi=np.nan)
        u_c, v_c = correct(detail.u, detail.v, nan_params)
        broken = replace(detail, params=nan_params, u_corr=u_c, v_corr=v_c)
        data = compute_angle_view(broken)
        assert data.target_radius is None and data.all is not None
        assert int(data.all.pre.count.sum()) == 0


def _synthetic_detail(n: int, sector_deg: float) -> ChannelDetail:
    params = EllipseParams(cx=2000.0, cy=2050.0, a=600.0, b=560.0, phi=0.4)
    rng = np.random.default_rng(4)
    t = np.radians(rng.uniform(0.0, sector_deg, n))
    u = params.cx + 580.0 * np.cos(t)
    v = params.cy + 580.0 * np.sin(t)
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
        consistent=False,
        message="The refit did not reproduce the stored ellipse.",
        seconds=0.0,
    )


def test_view_without_a_fit_mask_bins_every_point() -> None:
    data = compute_angle_view(_synthetic_detail(3000, 180.0))
    assert data.kept is None and not data.has_mask and data.n_rejected is None
    assert data.profile_set(False) is data.all and data.all is not None
    assert data.all.pre.n_empty >= 30  # half the circle is empty: gaps
    assert "rejected points are not known" in set_description(data, False)


def test_an_inconsistent_mask_is_not_used(session: UVSession) -> None:
    detail = session.channel_detail(OUTLIERS)
    assert detail.consistent and detail.kept is not None
    good = compute_angle_view(detail)
    assert good.has_mask and good.kept is not None
    # The refit gave a mask for another ellipse: bin every point and say why
    stale = compute_angle_view(replace(detail, consistent=False))
    assert not stale.has_mask and stale.kept is None and stale.n_rejected is None
    assert stale.profile_set(False) is stale.all
    assert "kept/rejected split is not known" in stale.message
    assert "rejected points are not known" in set_description(stale, False)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


def test_checkbox_switches_the_point_set(session: UVSession, tab: AngleTab, qtbot: QtBot) -> None:
    data = compute_angle_view(session.channel_detail(OUTLIERS))
    tab.show_data(data)
    assert tab.include_rejected_check.isEnabled() and not tab.include_rejected
    assert tab.shown_set is data.kept
    assert "robust-kept points" in tab.info_text()
    assert tab.summary() == summary_text(data, False)
    _, y = tab._post_items.means.getData()
    assert data.kept is not None and data.kept.post is not None
    np.testing.assert_array_equal(y, data.kept.post.mean)
    with qtbot.waitSignal(tab.display_changed):
        tab.include_rejected_check.setChecked(True)
    assert tab.shown_set is data.all and "including" in tab.info_text()
    assert data.all is not None and data.all.post is not None
    np.testing.assert_array_equal(tab._post_items.means.getData()[1], data.all.post.mean)
    with qtbot.assertNotEmitted(tab.display_changed):
        tab.set_include_rejected(False)
    assert tab.shown_set is data.kept
    assert tab._pre_items.reference.isVisible()
    assert tab._pre_items.reference.value() == pytest.approx(data.target_radius)


def test_an_empty_point_set_draws_nothing(tab: AngleTab, qtbot: QtBot) -> None:
    detail = _synthetic_detail(500, 360.0)
    none_kept = replace(detail, kept=np.zeros(500, dtype=bool), consistent=True)
    data = compute_angle_view(none_kept)
    assert data.kept is not None and data.kept.n_points == 0
    tab.show()
    qtbot.waitExposed(tab)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "All-NaN slice" from pyqtgraph on repaint
        tab.show_data(data)
        tab.grab()
        assert not tab._pre_items.means.isVisible() and not tab._post_items.means.isVisible()
        assert tab._pre_items.note.text == "No points to bin"
        tab.set_include_rejected(True)
        tab.grab()
        assert tab._pre_items.means.isVisible()


def test_gaps_and_disabled_checkbox(tab: AngleTab) -> None:
    data = compute_angle_view(_synthetic_detail(3000, 180.0))
    tab.show_data(data)
    assert not tab.include_rejected_check.isEnabled()
    _, y = tab._pre_items.means.getData()
    assert data.all is not None and data.all.pre.n_empty >= 30
    assert int(np.isnan(y).sum()) == data.all.pre.n_empty
    assert "refit did not reproduce" in tab.message()


@pytest.mark.parametrize(
    ("key", "phrase"),
    [(TOO_FEW, "Too few events"), (FIT_FAILED, "fit failed")],
)
def test_tab_without_ellipse(session: UVSession, tab: AngleTab, key, phrase: str) -> None:
    data = compute_angle_view(session.channel_detail(key))
    assert data.centre_kind == CENTRE_MEDIAN and data.kept is None and data.all is not None
    assert data.all.post is None and data.target_radius is None
    tab.show_data(data)
    assert phrase in tab.message()
    assert "about the median point" in tab.info_text()
    assert tab._post_items.note.text == "No ellipse fitted: no corrected points"
    assert not tab._pre_items.reference.isVisible()


def test_tab_without_results_and_without_events(
    ring_files: RingFiles, session: UVSession, tab: AngleTab
) -> None:
    fresh = UVSession()
    fresh.install(load_cache(ring_files.cache))
    tab.show_data(compute_angle_view(fresh.channel_detail(CLEAN)))
    assert "Run Fit All" in tab.message() and not tab.include_rejected_check.isEnabled()

    empty = compute_angle_view(session.channel_detail(NO_EVENTS))
    assert empty.all is None and empty.n_events == 0
    tab.show_data(empty)
    assert "no data" in tab.message() and tab.summary() == ""
    assert tab._pre_items.note.text == "No events on this channel"

    tab.clear("Node 1 Board 15: pick a channel")
    assert tab.data is None and tab.message() == "Node 1 Board 15: pick a channel"
    assert tab.info_text() == "No channel selected"
    for plot in (tab.pre_plot, tab.post_plot):  # the angle axis stays 0-360 degrees
        x0, x1 = plot.getPlotItem().getViewBox().viewRange()[0]
        assert -5.0 <= x0 <= 0.0 and 360.0 <= x1 <= 365.0
