"""Tests for uvcorr.analysis: ChannelResult, analyze_channel, analyze_board, analyze_all."""

from __future__ import annotations

import dataclasses
import math
import multiprocessing
import os
import signal
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from tests.conftest import RING_BOARDS, RING_CHANNELS, RingChannel, RingFiles, ring_points
from tests.synthetic_dat import Frame, Hit, write_dat
from uvcorr import analysis
from uvcorr.analysis import (
    CSV_COLUMNS,
    KIND_COUNT,
    KIND_FLAGS,
    KIND_FLOAT,
    KIND_FLOAT_PRECISE,
    KIND_INT,
    KIND_STR,
    RESULT_COLUMNS,
    AnalysisCancelled,
    AnalysisError,
    ChannelKey,
    ChannelResult,
    FitOptions,
    WorkerCrashedError,
    analyze_all,
    analyze_board,
    analyze_channel,
    default_workers,
)
from uvcorr.cache import UVCache, open_or_build
from uvcorr.channels import electrode_label, polarity_name
from uvcorr.ellipse import correct, fit_ellipse, radii_about_center, residual_to_ellipse
from uvcorr.io.tec import format_tec
from uvcorr.metrics import (
    RadialStats,
    phase_stats,
    radial_stats,
    residual_stats,
    timing_jitter_ns,
)
from uvcorr.options import (
    FLAG_BROAD_RING,
    FLAG_EXTREME_AXIS_RATIO,
    FLAG_GAUSS_FIT_FAILED_POST,
    FLAG_GAUSS_FIT_FAILED_PRE,
    FLAG_HIGH_REJECTION,
    FLAGS,
    STATUS_FIT_FAILED,
    STATUS_OK,
    STATUS_TOO_FEW_EVENTS,
    order_flags,
)

pytestmark = pytest.mark.filterwarnings("error")

# Plan section 6.2, verbatim.
PLAN_COLUMNS = [
    "node",
    "board",
    "rena",
    "channel",
    "polarity",
    "electrode",
    "status",
    "flags",
    "n_events",
    "n_used",
    "n_rejected",
    "centerU",
    "centerV",
    "semiMajor",
    "semiMinor",
    "phi",
    "axis_ratio",
    "target_radius",
    "pre_mean",
    "pre_sigma",
    "pre_fwhm",
    "pre_chi2ndf",
    "pre_skewness",
    "pre_kurtosis",
    "pre_robust_sigma",
    "post_mean",
    "post_sigma",
    "post_fwhm",
    "post_chi2ndf",
    "post_skewness",
    "post_kurtosis",
    "post_robust_sigma",
    "rawfit_res_mean",
    "rawfit_res_sigma",
    "corr_res_mean",
    "corr_res_sigma",
    "phase_mean_gap_rad",
    "phase_max_gap_rad",
    "phase_max_gap_ns",
    "phase_ks",
    "timing_jitter_ns",
    "options_source",
]

IDENTITY = ("node", "board", "rena", "channel", "polarity", "electrode")
ALWAYS_SET = (*IDENTITY, "status", "flags", "n_events", "options_source")

KEY = ChannelKey(1, 15, 0, 5)  # an odd-board cathode (C02)


def points(spec: RingChannel, seed: int = 3) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    u, v = ring_points(spec, seed)
    return u.astype(np.int16), v.astype(np.int16)


def minimal_result(**overrides: Any) -> ChannelResult:
    values: dict[str, Any] = {
        "node": 1,
        "board": 15,
        "rena": 0,
        "channel": 5,
        "polarity": "cathode",
        "electrode": "C02",
        "status": STATUS_TOO_FEW_EVENTS,
        "flags": (),
        "n_events": 12,
    }
    values.update(overrides)
    return ChannelResult(**values)


# ---------------------------------------------------------------------------
# Columns and ChannelResult
# ---------------------------------------------------------------------------


def test_columns_are_the_plan_csv_schema() -> None:
    assert list(CSV_COLUMNS) == PLAN_COLUMNS
    assert [f.name for f in dataclasses.fields(ChannelResult)] == PLAN_COLUMNS
    kinds = {c.name: c.kind for c in RESULT_COLUMNS}
    assert {n for n, k in kinds.items() if k == KIND_FLOAT_PRECISE} == {
        "centerU",
        "centerV",
        "semiMajor",
        "semiMinor",
        "phi",
    }
    assert {n for n, k in kinds.items() if k == KIND_INT} == {
        "node",
        "board",
        "rena",
        "channel",
        "n_events",
    }
    assert {n for n, k in kinds.items() if k == KIND_COUNT} == {"n_used", "n_rejected"}
    assert {n for n, k in kinds.items() if k == KIND_FLAGS} == {"flags"}
    assert {n for n, k in kinds.items() if k == KIND_STR} == {
        "polarity",
        "electrode",
        "status",
        "options_source",
    }
    assert all(k == KIND_FLOAT for n, k in kinds.items() if n.startswith(("pre_", "post_")))


def test_channel_result_normalises_values() -> None:
    result = minimal_result(
        status=STATUS_OK,
        node=np.uint8(1),
        n_events=np.int64(12),
        n_used=np.int32(10),
        n_rejected=2,
        flags=[FLAG_GAUSS_FIT_FAILED_POST, FLAG_HIGH_REJECTION, FLAG_HIGH_REJECTION],
        pre_mean=np.float32(1.5),
        pre_sigma=math.nan,
        post_sigma=math.inf,
        phase_ks=3,
    )
    assert type(result.node) is int and type(result.n_events) is int
    assert type(result.n_used) is int and result.n_used == 10
    assert result.flags == (FLAG_HIGH_REJECTION, FLAG_GAUSS_FIT_FAILED_POST)
    assert type(result.pre_mean) is float and result.pre_mean == 1.5
    assert result.pre_sigma is None and result.post_sigma is None
    assert result.phase_ks == 3.0 and type(result.phase_ks) is float
    assert result.options_source == "batch"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"status": "great"}, ValueError),
        ({"polarity": "both"}, ValueError),
        ({"options_source": "gui"}, ValueError),
        ({"flags": ("not_a_flag",)}, ValueError),
        ({"flags": "high_rejection"}, TypeError),
        ({"n_used": -1}, ValueError),
        ({"n_used": 1.5}, TypeError),
        ({"node": True}, TypeError),
        ({"node": 1.0}, TypeError),
        ({"pre_mean": "1.0"}, TypeError),
        ({"electrode": 3}, TypeError),
    ],
)
def test_channel_result_validation(overrides: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        minimal_result(**overrides)


def test_channel_result_properties() -> None:
    result = minimal_result(
        status=STATUS_OK,
        n_events=200,
        n_used=150,
        n_rejected=50,
        centerU=2000.0,
        centerV=2010.0,
        semiMajor=600.0,
        semiMinor=580.0,
        phi=0.1,
        flags=(FLAG_HIGH_REJECTION, FLAG_EXTREME_AXIS_RATIO),
    )
    assert result.key == ChannelKey(1, 15, 0, 5) == (1, 15, 0, 5)
    assert result.ok
    assert result.rejected_fraction == 0.25
    assert result.params is not None and result.params.a == 600.0 and result.params.phi == 0.1
    assert result.flags_text == "high_rejection;extreme_axis_ratio"
    failed = minimal_result()
    assert not failed.ok and failed.params is None and failed.rejected_fraction is None
    assert minimal_result(n_events=0, n_rejected=0).rejected_fraction is None


def test_channel_result_dict_round_trip() -> None:
    result = minimal_result(status=STATUS_OK, flags=(FLAG_BROAD_RING,), pre_mean=2.0)
    data = result.to_dict()
    assert list(data) == PLAN_COLUMNS
    assert ChannelResult.from_dict(data) == result
    text = {**data, "flags": "broad_ring"}
    assert ChannelResult.from_dict(text) == result
    assert ChannelResult.from_dict({**data, "flags": ""}).flags == ()
    with pytest.raises(ValueError, match="Unknown"):
        ChannelResult.from_dict({**data, "extra": 1})
    assert ChannelResult.from_dict({**data, "extra": 1}, strict=False) == result
    with pytest.raises(TypeError):
        ChannelResult.from_dict({k: v for k, v in data.items() if k != "node"})


def test_channel_key_orders_like_a_tuple() -> None:
    keys = [ChannelKey(2, 15, 0, 4), ChannelKey(1, 16, 1, 7), ChannelKey(1, 16, 0, 28)]
    assert sorted(keys) == [(1, 16, 0, 28), (1, 16, 1, 7), (2, 15, 0, 4)]
    assert hash(ChannelKey(1, 2, 3, 4)) == hash((1, 2, 3, 4))
    assert str(ChannelKey(1, 2, 0, 4)) == "node 1 board 2 rena 0 channel 4"


# ---------------------------------------------------------------------------
# analyze_channel
# ---------------------------------------------------------------------------


def assert_only_identity(result: ChannelResult) -> None:
    for name in PLAN_COLUMNS:
        value = getattr(result, name)
        if name in ALWAYS_SET:
            assert value is not None, name
        else:
            assert value is None, name


def test_ok_channel_fills_every_field() -> None:
    spec = RingChannel(rena=0, channel=5, n=2000, a=600.0, b=585.0, phi=0.3, noise=6.0)
    u, v = points(spec)
    result = analyze_channel(KEY, u, v)
    assert result.status == STATUS_OK
    for name in PLAN_COLUMNS:
        assert getattr(result, name) is not None, name
    assert result.flags == ()
    assert result.key == KEY
    assert result.polarity == polarity_name(15, 0, 5) == "cathode"
    assert result.electrode == electrode_label(15, 0, 5) == "C02"
    assert result.options_source == "batch"
    assert result.n_events == 2000
    assert result.n_used is not None and result.n_rejected is not None
    assert result.n_used + result.n_rejected == 2000
    # Parameters near the truth (noise 6, N = 2000; integer rounding)
    assert result.centerU == pytest.approx(spec.cx, abs=1.5)
    assert result.centerV == pytest.approx(spec.cy, abs=1.5)
    assert result.semiMajor == pytest.approx(spec.a, abs=1.5)
    assert result.semiMinor == pytest.approx(spec.b, abs=1.5)
    assert result.phi == pytest.approx(spec.phi, abs=0.05)
    assert result.axis_ratio == pytest.approx(result.semiMinor / result.semiMajor, rel=1e-12)
    assert result.target_radius == pytest.approx(
        math.sqrt(result.semiMajor * result.semiMinor), rel=1e-12
    )
    assert result.post_sigma == pytest.approx(spec.noise, rel=0.15)
    assert result.post_mean == pytest.approx(result.target_radius, rel=1e-3)
    assert result.post_fwhm == pytest.approx(2.35482 * result.post_sigma)
    assert result.timing_jitter_ns == pytest.approx(
        result.post_sigma / (2 * math.pi * 490e3 * result.post_mean) * 1e9
    )
    assert result.phase_mean_gap_rad == pytest.approx(2 * math.pi / 2000)
    assert 0 < result.phase_ks < 0.1


def test_metrics_are_computed_over_all_events() -> None:
    spec = RingChannel(rena=0, channel=12, n=1500, outlier_frac=0.12)
    u, v = points(spec)
    key = ChannelKey(1, 15, 0, 12)
    result = analyze_channel(key, u, v)
    assert result.status == STATUS_OK
    assert FLAG_HIGH_REJECTION in result.flags
    assert result.n_rejected is not None and result.n_rejected >= 150

    fit = fit_ellipse(u, v)
    assert fit.params is not None
    p = fit.params
    x, y = u.astype(np.float64), v.astype(np.float64)
    pre_all = radial_stats(radii_about_center(x, y, p))
    uc, vc = correct(x, y, p)
    post_all = radial_stats(np.hypot(uc, vc))
    raw_all = residual_stats(residual_to_ellipse(x, y, p))
    phase_all = phase_stats(uc, vc)
    assert phase_all is not None
    assert result.pre_mean == pre_all.mean and result.pre_skewness == pre_all.skewness
    assert result.pre_robust_sigma == pre_all.robust_sigma
    assert result.post_sigma == post_all.sigma and result.post_kurtosis == post_all.kurtosis
    assert result.rawfit_res_sigma == raw_all.sigma
    assert result.phase_ks == phase_all.ks
    assert result.phase_mean_gap_rad == pytest.approx(2 * math.pi / 1500)

    kept = fit.mask_used
    pre_kept = radial_stats(radii_about_center(x[kept], y[kept], p))
    assert result.pre_skewness != pre_kept.skewness  # outliers are included
    assert result.pre_kurtosis != pre_kept.kurtosis


def test_too_few_events_fills_only_identity() -> None:
    u, v = points(RingChannel(rena=1, channel=28, n=40))
    result = analyze_channel(ChannelKey(2, 16, 1, 28), u, v)
    assert result.status == STATUS_TOO_FEW_EVENTS
    assert result.n_events == 40 and result.flags == ()
    assert result.polarity == "cathode" and result.electrode == electrode_label(16, 1, 28)
    assert_only_identity(result)


def test_fit_failed_fills_only_identity() -> None:
    u, v = points(RingChannel(rena=0, channel=20, n=300, kind="line"))
    result = analyze_channel(ChannelKey(1, 16, 0, 20), u, v)
    assert result.status == STATUS_FIT_FAILED
    assert result.n_events == 300
    assert_only_identity(result)


def test_eccentric_ring_flags_extreme_axis_ratio() -> None:
    u, v = points(RingChannel(rena=1, channel=9, n=1500, a=620.0, b=250.0, phi=-1.2, noise=3.0))
    result = analyze_channel(ChannelKey(1, 15, 1, 9), u, v)
    assert result.status == STATUS_OK
    assert result.axis_ratio is not None and result.axis_ratio == pytest.approx(250 / 620, rel=0.01)
    assert FLAG_EXTREME_AXIS_RATIO in result.flags
    # Pre radii spread between b and a; the corrected ring is thin
    assert result.pre_sigma is not None and result.post_sigma is not None
    assert result.post_sigma < result.pre_sigma / 10
    assert (FLAG_GAUSS_FIT_FAILED_PRE in result.flags) == (result.pre_chi2ndf is None)
    assert (FLAG_GAUSS_FIT_FAILED_POST in result.flags) == (result.post_chi2ndf is None)


@pytest.mark.parametrize(
    ("fail_pre", "fail_post"), [(True, False), (False, True), (True, True), (False, False)]
)
def test_failed_radial_gaussians_become_flags(
    monkeypatch: pytest.MonkeyPatch, fail_pre: bool, fail_post: bool
) -> None:
    # analyze_channel calls radial_stats twice: pre radii, then post radii
    calls: list[RadialStats] = []

    def forced(r: Any) -> RadialStats:
        stats = radial_stats(r)
        fail = fail_pre if not calls else fail_post
        if fail:
            stats = dataclasses.replace(
                stats,
                ok=False,
                mean=stats.sample_mean,
                sigma=stats.sample_std,
                fwhm=2.35482 * stats.sample_std,
                chi2ndf=math.nan,
            )
        else:
            stats = dataclasses.replace(stats, ok=True, chi2ndf=1.25)
        calls.append(stats)
        return stats

    monkeypatch.setattr(analysis, "radial_stats", forced)
    u, v = points(RingChannel(rena=1, channel=9, n=1500, a=620.0, b=250.0, phi=-1.2, noise=3.0))
    result = analyze_channel(ChannelKey(1, 15, 1, 9), u, v)
    pre, post = calls
    assert (FLAG_GAUSS_FIT_FAILED_PRE in result.flags) == fail_pre
    assert (FLAG_GAUSS_FIT_FAILED_POST in result.flags) == fail_post
    assert FLAG_EXTREME_AXIS_RATIO in result.flags  # the fit's flags are kept
    assert result.flags == order_flags(result.flags)
    assert (result.pre_mean, result.pre_sigma) == (pre.mean, pre.sigma)
    assert (result.post_mean, result.post_sigma) == (post.mean, post.sigma)
    assert (result.pre_chi2ndf is None) == fail_pre
    assert (result.post_chi2ndf is None) == fail_post


def test_failed_post_gaussian_reports_the_sample_std() -> None:
    # Fewer than 50 values: the Gaussian routine reports sample statistics (ok=False)
    spec = RingChannel(rena=0, channel=5, n=40, a=600.0, b=590.0, noise=5.0)
    u, v = points(spec)
    result = analyze_channel(KEY, u, v, FitOptions(min_events=6))
    assert result.status == STATUS_OK
    assert FLAG_GAUSS_FIT_FAILED_PRE in result.flags
    assert FLAG_GAUSS_FIT_FAILED_POST in result.flags
    assert result.pre_chi2ndf is None and result.post_chi2ndf is None
    params = result.params
    assert params is not None
    uc, vc = correct(u.astype(np.float64), v.astype(np.float64), params)
    radii = np.hypot(uc, vc)
    assert result.post_mean == pytest.approx(float(np.mean(radii)), rel=1e-12)
    assert result.post_sigma == pytest.approx(float(np.std(radii, ddof=1)), rel=1e-12)
    # The .tec radiusStd is this post sigma
    assert format_tec([result]).split("radiusStd=")[1] == f"{result.post_sigma:g}\n}}\n"


def test_phase_metrics_need_eight_points() -> None:
    t = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    u = np.rint(2000 + 600 * np.cos(t) + 3 * np.sin(3 * t))
    v = np.rint(2000 + 580 * np.sin(t))
    result = analyze_channel(KEY, u, v, FitOptions(min_events=6))
    assert result.status == STATUS_OK
    for name in ("phase_mean_gap_rad", "phase_max_gap_rad", "phase_max_gap_ns", "phase_ks"):
        assert getattr(result, name) is None, name
    assert result.timing_jitter_ns is not None
    # Fewer than 50 values: sample statistics, flagged
    assert FLAG_GAUSS_FIT_FAILED_PRE in result.flags and FLAG_GAUSS_FIT_FAILED_POST in result.flags


def test_timing_jitter_uses_the_reference_frequency() -> None:
    u, v = points(RingChannel(rena=0, channel=5, n=800))
    base = analyze_channel(KEY, u, v)
    other = analyze_channel(KEY, u, v, FitOptions(phase_ref_freq_hz=245e3))
    assert base.timing_jitter_ns is not None and other.timing_jitter_ns is not None
    assert other.timing_jitter_ns == pytest.approx(2 * base.timing_jitter_ns)
    assert base.post_sigma is not None and base.post_mean is not None
    assert base.timing_jitter_ns == timing_jitter_ns(base.post_sigma, base.post_mean)


def test_non_finite_points_are_counted_but_not_used() -> None:
    u, v = points(RingChannel(rena=0, channel=5, n=600))
    uf = np.concatenate([u.astype(np.float64), [np.nan, 1.0, np.inf]])
    vf = np.concatenate([v.astype(np.float64), [1.0, np.nan, 2.0]])
    result = analyze_channel(KEY, uf, vf)
    assert result.status == STATUS_OK
    assert result.n_events == 603
    assert result.n_rejected is not None and result.n_rejected >= 3
    assert all(
        getattr(result, c.name) is not None
        for c in RESULT_COLUMNS
        if c.kind in (KIND_FLOAT, KIND_FLOAT_PRECISE)
    )


def test_options_reach_the_fit() -> None:
    u, v = points(RingChannel(rena=0, channel=12, n=800, outlier_frac=0.12))
    for options in (FitOptions(), FitOptions(robust=False), FitOptions(clip_k=2.0, max_iter=1)):
        result = analyze_channel(KEY, u, v, options)
        fit = fit_ellipse(u, v, options)
        assert fit.params is not None
        assert (result.n_rejected, result.centerU, result.semiMinor) == (
            fit.n_rejected,
            fit.params.cx,
            fit.params.b,
        )
    robust = analyze_channel(KEY, u, v)
    plain = analyze_channel(KEY, u, v, FitOptions(robust=False))
    assert plain.centerU != robust.centerU
    assert analyze_channel(KEY, u, v, FitOptions(min_events=900)).status == STATUS_TOO_FEW_EVENTS


def test_analyze_channel_key_and_input_checks() -> None:
    u, v = points(RingChannel(rena=0, channel=5, n=200))
    assert analyze_channel((1, 15, 0, 5), u, v).key == KEY
    with pytest.raises(ValueError, match="not an active channel"):
        analyze_channel(ChannelKey(1, 15, 0, 2), u, v)
    with pytest.raises(ValueError):
        analyze_channel(KEY, u, v[:-1])


def test_flags_are_in_canonical_order() -> None:
    for spec in RING_CHANNELS:
        u, v = points(spec)
        result = analyze_channel(ChannelKey(1, 15, spec.rena, spec.channel), u, v)
        assert result.flags == tuple(f for f in FLAGS if f in result.flags)


# ---------------------------------------------------------------------------
# analyze_board / analyze_all
# ---------------------------------------------------------------------------

EXPECTED_STATUS = {
    (0, 5): STATUS_OK,
    (0, 12): STATUS_OK,
    (1, 9): STATUS_OK,
    (1, 25): STATUS_OK,
    (1, 28): STATUS_TOO_FEW_EVENTS,
    (0, 20): STATUS_FIT_FAILED,
}


def test_analyze_board_matches_analyze_channel(ring_files: RingFiles) -> None:
    cache = UVCache(ring_files.cache)
    results = analyze_board(ring_files.cache, 1, 16)
    assert [(r.rena, r.channel) for r in results] == sorted(EXPECTED_STATUS)
    assert {(r.rena, r.channel): r.status for r in results} == EXPECTED_STATUS
    for result in results:
        u, v = cache.channel_data(1, 16, result.rena, result.channel)
        assert result == analyze_channel(result.key, u, v)
        assert result.n_events == u.shape[0]
    flagged = {(r.rena, r.channel): r.flags for r in results}
    assert FLAG_HIGH_REJECTION in flagged[(0, 12)]
    assert FLAG_EXTREME_AXIS_RATIO in flagged[(1, 9)]
    # A UVCache object works too; options are passed on
    again = analyze_board(cache, 1, 16, FitOptions(min_events=1000))
    assert {r.status for r in again} == {STATUS_TOO_FEW_EVENTS}


def test_analyze_board_missing_board(ring_files: RingFiles) -> None:
    with pytest.raises(KeyError):
        analyze_board(ring_files.cache, 9, 20)


def test_analyze_board_stop_flag(ring_files: RingFiles) -> None:
    with pytest.raises(AnalysisCancelled):
        analyze_board(ring_files.cache, 1, 16, stop_flag=lambda: True)
    event = threading.Event()
    event.set()
    with pytest.raises(AnalysisCancelled):
        analyze_board(ring_files.cache, 1, 16, stop_flag=event)


def assert_event_progress(calls: list[tuple[int, int]], counts: dict[Any, int]) -> None:
    """Progress is (done events, total events): 0 first, then one step per board."""
    total = sum(counts.values())
    assert calls[0] == (0, total)
    assert all(t == total for _, t in calls)
    steps = sorted(b - a for (a, _), (b, _) in zip(calls, calls[1:]))
    assert steps == sorted(counts.values())
    assert calls[-1] == (total, total)


def test_analyze_all_in_process_and_pool_agree(ring_files: RingFiles) -> None:
    counts = UVCache(ring_files.cache).board_event_counts()
    calls: list[tuple[int, int]] = []
    single = analyze_all(ring_files.cache, workers=1, progress_cb=lambda d, t: calls.append((d, t)))
    assert_event_progress(calls, counts)
    assert len(single) == len(RING_BOARDS) * len(RING_CHANNELS)
    assert [r.key for r in single] == sorted(r.key for r in single)
    assert {(r.node, r.board) for r in single} == set(RING_BOARDS)

    calls.clear()
    env_before = dict(os.environ)
    mask_before = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    pooled = analyze_all(
        UVCache(ring_files.cache), workers=2, progress_cb=lambda d, t: calls.append((d, t))
    )
    assert pooled == single
    assert_event_progress(calls, counts)
    assert dict(os.environ) == env_before  # no environment mutation
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == mask_before  # SIGINT unblocked again
    assert multiprocessing.active_children() == []


def test_analyze_all_from_a_worker_thread(ring_files: RingFiles) -> None:
    # The GUI runs the analysis in a QThread: the pool must work off the main thread
    expected = analyze_all(ring_files.cache, workers=1)
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["results"] = analyze_all(ring_files.cache, workers=2)
        except BaseException as exc:  # pragma: no cover - reported below
            box["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=120)
    assert not thread.is_alive()
    assert "error" not in box, box.get("error")
    assert box["results"] == expected
    assert multiprocessing.active_children() == []


def test_boards_are_scheduled_largest_first(
    ring_files: RingFiles, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = {(1, 15): 10, (2, 16): 300, (1, 16): 300, (3, 20): 5000, (9, 30): 1}
    assert analysis.schedule_boards(counts) == [(3, 20), (1, 16), (2, 16), (1, 15), (9, 30)]

    # analyze_all runs the boards in that order (in-process: in exactly that order)
    order: list[tuple[int, int]] = []
    real = analysis.analyze_board

    def recording(cache: Any, node: int, board: int, *args: Any, **kwargs: Any) -> Any:
        order.append((node, board))
        return real(cache, node, board, *args, **kwargs)

    monkeypatch.setattr(analysis, "analyze_board", recording)
    with h5py.File(ring_files.cache, "r+") as h5f:  # make the board sizes differ
        h5f["events/node_1/board_16"].attrs["n_events"] = 5000
        h5f["events/node_4/board_29"].attrs["n_events"] = 3000
    board_counts = UVCache(ring_files.cache).board_event_counts()
    analyze_all(ring_files.cache, workers=1)
    assert order == analysis.schedule_boards(board_counts) == [(1, 16), (4, 29), (1, 15)]


def test_worker_initializer() -> None:
    # Workers: single-threaded BLAS, Ctrl-C ignored, no adc2kev import (the parent
    # seeds the electrode labels), and they still label channels correctly.
    ctx = multiprocessing.get_context("spawn")
    identities = analysis._identity_table([15])
    with ProcessPoolExecutor(
        1, mp_context=ctx, initializer=analysis._init_worker, initargs=(ctx.Event(), identities)
    ) as pool:
        diagnostics = pool.submit(analysis._worker_diagnostics).result()
        u, v = points(RingChannel(rena=0, channel=5, n=300))
        result = pool.submit(analyze_channel, KEY, u, v).result()
        after = pool.submit(analysis._worker_diagnostics).result()
    assert diagnostics["blas_threads"] and set(diagnostics["blas_threads"]) == {1}
    assert diagnostics["sigint_ignored"] is True
    assert diagnostics["adc2kev_loaded"] is False
    assert result.electrode == "C02" and result.polarity == "cathode"
    assert after["adc2kev_loaded"] is False
    assert len(identities) == 47 and identities[(15, 0, 5)] == ("cathode", "C02")


def test_worker_crash_is_not_blamed_on_a_board(ring_files: RingFiles) -> None:
    killed: list[int] = []

    def kill_a_worker() -> bool:
        # Called right after the boards were submitted, while the workers start
        if not killed:
            for child in multiprocessing.active_children():
                if child.pid is not None:
                    os.kill(child.pid, signal.SIGKILL)
                    killed.append(child.pid)
                    break
        return False

    with pytest.raises(WorkerCrashedError) as excinfo:
        analyze_all(ring_files.cache, workers=2, stop_flag=kill_a_worker)
    assert killed
    error = excinfo.value
    assert isinstance(error, AnalysisError)
    assert error.node is None and error.board is None
    assert "terminated abruptly" in str(error) and "--workers 1" in str(error)
    assert isinstance(error.__cause__, BrokenProcessPool)
    assert multiprocessing.active_children() == []


def test_worker_crash_after_a_stop_request_is_a_cancel(ring_files: RingFiles) -> None:
    calls: list[int] = []

    def kill_then_stop() -> bool:
        calls.append(1)
        if len(calls) == 1:
            for child in multiprocessing.active_children():
                if child.pid is not None:
                    os.kill(child.pid, signal.SIGKILL)
                    break
            return False
        return True

    with pytest.raises(AnalysisCancelled):
        analyze_all(ring_files.cache, workers=2, stop_flag=kill_then_stop)
    assert multiprocessing.active_children() == []


def test_analyze_all_options(ring_files: RingFiles) -> None:
    options = FitOptions(robust=False, min_events=30)
    results = analyze_all(ring_files.cache, options, workers=2)
    assert not any(r.status == STATUS_TOO_FEW_EVENTS for r in results)  # 40-event channels
    cache = UVCache(ring_files.cache)
    for result in results:
        u, v = cache.channel_data(*result.key)
        assert result == analyze_channel(result.key, u, v, options)


def test_analyze_all_workers_validation(ring_files: RingFiles) -> None:
    with pytest.raises(ValueError, match="workers"):
        analyze_all(ring_files.cache, workers=0)
    assert 1 <= default_workers() <= analysis.MAX_DEFAULT_WORKERS


def test_analyze_all_empty_cache(tmp_path: Path) -> None:
    # Only an inactive channel: the cache has no boards
    dat = write_dat(tmp_path / "empty.dat", [Frame(1, 15, 0, 1, (Hit(2, 1, 5, 5),))])
    cache = open_or_build(dat)
    calls: list[tuple[int, int]] = []
    assert analyze_all(cache, progress_cb=lambda d, t: calls.append((d, t))) == []
    assert calls == [(0, 0)]  # (done events, total events)


@pytest.mark.parametrize("workers", [1, 2])
def test_analyze_all_cancelled_before_start(ring_files: RingFiles, workers: int) -> None:
    event = threading.Event()
    event.set()
    with pytest.raises(AnalysisCancelled):
        analyze_all(ring_files.cache, workers=workers, stop_flag=event)
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize("workers", [1, 2])
def test_analyze_all_cancelled_midway(ring_files: RingFiles, workers: int) -> None:
    stop = threading.Event()
    calls: list[int] = []

    def progress(done: int, total: int) -> None:
        calls.append(done)
        if done >= 1:
            stop.set()
        assert total == sum(UVCache(ring_files.cache).board_event_counts().values())

    with pytest.raises(AnalysisCancelled):
        analyze_all(ring_files.cache, workers=workers, progress_cb=progress, stop_flag=stop.is_set)
    assert max(calls) < sum(UVCache(ring_files.cache).board_event_counts().values())
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize("workers", [1, 2])
def test_worker_error_identifies_the_board(ring_files: RingFiles, workers: int) -> None:
    with h5py.File(ring_files.cache, "r+") as h5f:
        del h5f["events/node_1/board_16/u"]
    with pytest.raises(AnalysisError, match="node 1 board 16") as excinfo:
        analyze_all(ring_files.cache, workers=workers)
    assert (excinfo.value.node, excinfo.value.board) == (1, 16)
    assert excinfo.value.__cause__ is not None
    assert multiprocessing.active_children() == []


def test_default_workers_uses_the_cpu_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2}, raising=False)
    assert default_workers() == 3
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(64)), raising=False)
    assert default_workers() == analysis.MAX_DEFAULT_WORKERS
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert default_workers() == 2


def test_analysis_error_pickles() -> None:
    import pickle

    error = pickle.loads(pickle.dumps(AnalysisError("boom", 3, 25)))
    assert (str(error), error.node, error.board) == ("boom", 3, 25)
    crashed = pickle.loads(pickle.dumps(WorkerCrashedError()))
    assert isinstance(crashed, WorkerCrashedError) and crashed.node is None
    assert str(crashed) == analysis.WORKER_CRASHED_MESSAGE
