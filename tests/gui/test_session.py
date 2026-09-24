"""Tests for the GUI data layer ``uvcorr.gui.session`` (no widgets needed)."""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.conftest import RingFiles
from tests.gui.ring_cache import (
    CHANNELS_PER_BOARD,
    CLEAN,
    ECCENTRIC,
    FIT_FAILED,
    OUTLIERS,
    RING_BOARDS,
    TOO_FEW,
    store_batch_results,
)
from uvcorr import cache as cache_module
from uvcorr.analysis import OPTIONS_BATCH, OPTIONS_OVERRIDE, AnalysisCancelled, ChannelKey
from uvcorr.cache import CacheBusyError, ResultsError, UVCache, UVCacheError
from uvcorr.ellipse import EllipseParams
from uvcorr.gui._system_map_model import ChannelView
from uvcorr.gui.session import (
    DetailRequest,
    NoEventsError,
    SessionBusyError,
    SessionError,
    UVSession,
    channel_title,
    channel_view,
    describe_options_change,
    effective_options,
    inspect_raw,
    is_cache_file,
    load_cache,
    load_raw,
    same_fit,
    short_title,
)
from uvcorr.io.summary_csv import read_summary_csv
from uvcorr.io.tec import read_tec
from uvcorr.options import (
    FLAG_EXTREME_AXIS_RATIO,
    FLAG_HIGH_REJECTION,
    STATUS_FIT_FAILED,
    STATUS_OK,
    STATUS_TOO_FEW_EVENTS,
    FitOptions,
)

# ``results_files`` (the ring files with stored batch results) comes from conftest.py


@pytest.fixture
def session(results_files: RingFiles) -> UVSession:
    """A session with the results cache installed."""
    s = UVSession()
    s.install(load_cache(results_files.cache))
    return s


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


def test_is_cache_file() -> None:
    assert is_cache_file("run.dat.uv.h5")
    assert is_cache_file(Path("/x/other.H5"))
    assert not is_cache_file("run.dat")


def test_load_cache_reads_index_and_results(results_files: RingFiles) -> None:
    opened = load_cache(results_files.cache)
    assert sorted(opened.board_counts) == list(RING_BOARDS)
    assert len(opened.channel_counts) == CHANNELS_PER_BOARD * len(RING_BOARDS)
    assert opened.channel_counts[TOO_FEW] == 40
    assert opened.stored is not None and len(opened.stored.results) == 18
    # The raw file recorded at build time (the master copy the fixture copied)
    assert opened.dat_path is not None and opened.dat_path.name == "rings.dat"
    assert opened.stale is False and opened.built is False


def test_load_cache_flags_a_stale_cache(ring_files: RingFiles) -> None:
    ring_files.cache.unlink()
    load_raw(ring_files.dat)  # built here, so it records this raw file
    assert load_cache(ring_files.cache).stale is False
    stat = ring_files.dat.stat()
    os.utime(ring_files.dat, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert load_cache(ring_files.cache).stale is True


def test_unreadable_results_do_not_block_the_open(results_files: RingFiles) -> None:
    with h5py.File(results_files.cache, "r+") as h5f:
        h5f["results/current"].attrs["options_json"] = "{not json"
    opened = load_cache(results_files.cache)
    assert opened.stored is None and opened.results_error is not None
    assert "unreadable" in opened.results_error
    assert len(opened.channel_counts) == 18  # scanned from the events instead
    session = UVSession()
    session.install(opened)
    assert not session.has_results
    # Fit All replaces the unreadable results
    session.apply_batch(session.run_batch(FitOptions(), workers=1))
    reopened = load_cache(results_files.cache)
    assert reopened.results_error is None and reopened.stored is not None


def test_files_without_events_are_refused(tmp_path: Path) -> None:
    junk = tmp_path / "junk.dat"
    junk.write_bytes(bytes(range(256)) * 64)
    with pytest.raises(NoEventsError, match="No valid frames in junk.dat"):
        load_raw(junk)
    assert not (tmp_path / "junk.dat.uv.h5").exists()  # the empty cache is removed again
    # An empty cache opened directly is refused but left alone
    cache = UVCache(tmp_path / "empty.uv.h5")
    cache.build_from_dat(junk)
    with pytest.raises(NoEventsError, match="is this a raw .dat file"):
        load_cache(cache.path)
    assert cache.path.exists()


def test_load_cache_rejects_other_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cache(tmp_path / "missing.uv.h5")
    foreign = tmp_path / "foreign.h5"
    with h5py.File(foreign, "w") as h5f:
        h5f.create_group("something")
    with pytest.raises(UVCacheError, match="not a uvcorr UV cache"):
        load_cache(foreign)


def test_load_raw_builds_then_reuses(ring_files: RingFiles) -> None:
    ring_files.cache.unlink()
    progress: list[float] = []
    built = load_raw(ring_files.dat, progress_cb=progress.append)
    assert built.built and progress[-1] == 1.0
    assert built.stored is None
    # Without results every board is scanned for its channels
    assert len(built.channel_counts) == 18
    assert built.channel_counts[OUTLIERS] == 500
    reused = load_raw(ring_files.dat)
    assert not reused.built
    assert dict(reused.channel_counts) == dict(built.channel_counts)


def test_inspect_raw(results_files: RingFiles) -> None:
    check = inspect_raw(results_files.dat)
    assert check.valid and not check.needs_build and not check.discards_results
    stat = results_files.dat.stat()
    os.utime(results_files.dat, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    check = inspect_raw(results_files.dat)
    assert check.needs_build and check.cache_exists and check.discards_results
    with pytest.raises(FileNotFoundError):
        inspect_raw(results_files.dat.with_name("nope.dat"))


# ---------------------------------------------------------------------------
# Index and results
# ---------------------------------------------------------------------------


def test_index_queries(session: UVSession) -> None:
    assert session.is_open and session.has_results
    assert session.boards == list(RING_BOARDS)
    assert session.nodes() == [1, 4]
    assert session.boards_on_node(1) == [15, 16]
    on_board = session.channels_on_board(1, 15)
    assert len(on_board) == CHANNELS_PER_BOARD and on_board == sorted(on_board)
    assert session.data_channels == tuple(sorted(session.data_channels))
    assert session.channel_count(CLEAN) == 700
    assert session.channel_count((9, 9, 0, 5)) == 0
    assert session.options == FitOptions()  # the stored batch options
    summary = session.describe()
    assert summary.startswith("3 boards · 18 ch · 7,170 events · fitted ")
    assert "rings" not in summary  # the window title names the file


def test_step_channel(session: UVSession) -> None:
    channels = session.data_channels
    assert session.step_channel(None, +1) == channels[0]
    assert session.step_channel(None, -1) == channels[-1]
    assert session.step_channel(channels[0], -1) == channels[0]  # clamped
    assert session.step_channel(channels[-1], +1) == channels[-1]
    assert session.step_channel(channels[3], +2) == channels[5]
    # From a channel without data: the next / previous one with data
    gap = ChannelKey(1, 15, 0, 13)  # between R0 Ch12 and R0 Ch20
    assert session.step_channel(gap, +1) == ChannelKey(1, 15, 0, 20)
    assert session.step_channel(gap, -1) == OUTLIERS


def test_step_from_board(session: UVSession) -> None:
    """Prev/Next from a board shown without a channel (a click on a board tile)."""
    on_16 = session.channels_on_board(1, 16)
    assert session.step_from_board(1, 16, +1) == on_16[0]
    assert session.step_from_board(1, 16, -1) == on_16[-1]
    assert session.step_from_board(1, 16, +2) == on_16[1]
    # A board without data: the nearest channel after / before it
    assert session.step_from_board(2, 20, +1) == session.channels_on_board(4, 29)[0]
    assert session.step_from_board(2, 20, -1) == on_16[-1]
    # Beyond either end: clamped
    assert session.step_from_board(10, 30, +1) == session.data_channels[-1]
    assert session.step_from_board(1, 14, -1) == session.data_channels[0]


def test_results_and_options(session: UVSession) -> None:
    assert session.result(CLEAN) is not None and session.result(CLEAN).status == STATUS_OK
    assert session.result(TOO_FEW).status == STATUS_TOO_FEW_EVENTS
    assert session.result(FIT_FAILED).status == STATUS_FIT_FAILED
    assert FLAG_HIGH_REJECTION in session.result(OUTLIERS).flags
    assert FLAG_EXTREME_AXIS_RATIO in session.result(ECCENTRIC).flags
    assert session.result((9, 9, 0, 5)) is None
    assert session.options_for(CLEAN) == (FitOptions(), OPTIONS_BATCH)
    assert len(session.merged_results()) == 18


def test_views_carry_status_flags_and_metrics(session: UVSession) -> None:
    views = session.views()
    assert len(views) == 18 and all(isinstance(v, ChannelView) for v in views.values())
    outliers = views[OUTLIERS]
    result = session.result(OUTLIERS)
    assert result is not None
    assert outliers.status == STATUS_OK and outliers.flags == result.flags
    assert outliers.metrics["n_events"] == 500
    assert outliers.metric("post_sigma") == pytest.approx(result.post_sigma)
    assert outliers.metric("rejected_fraction") == pytest.approx(result.rejected_fraction)
    # Unavailable metrics are left out (NaN when read)
    too_few = views[TOO_FEW]
    assert "post_sigma" not in too_few.metrics and math.isnan(too_few.metric("post_sigma"))
    assert channel_view(result) == outliers


def test_channel_title() -> None:
    assert channel_title(CLEAN) == "N1 B15 R0 Ch05 (C02)"
    assert channel_title((1, 15, 0, 2)) == "N1 B15 R0 Ch02"  # inactive: no electrode


def test_session_without_file() -> None:
    s = UVSession()
    assert not s.is_open and s.boards == [] and s.data_channels == ()
    assert s.step_channel(None, 1) is None
    assert s.describe() == "No file open"
    with pytest.raises(SessionError):
        s.detail_request(CLEAN)
    with pytest.raises(SessionError):
        s.board_data(1, 15)
    with pytest.raises(ValueError):
        UVSession(board_cache_size=0)


# ---------------------------------------------------------------------------
# Board LRU
# ---------------------------------------------------------------------------


def test_board_lru(results_files: RingFiles, monkeypatch: pytest.MonkeyPatch) -> None:
    session = UVSession(board_cache_size=2)
    session.install(load_cache(results_files.cache))
    loads: list[tuple[int, int]] = []
    original = UVCache.load_board

    def counting_load(self: UVCache, node: int, board: int):  # type: ignore[no-untyped-def]
        loads.append((node, board))
        return original(self, node, board)

    monkeypatch.setattr(UVCache, "load_board", counting_load)
    session.board_data(1, 15)
    session.board_data(1, 16)
    session.board_data(1, 15)  # hit: moves to the end
    assert loads == [(1, 15), (1, 16)]
    assert session.cached_boards() == [(1, 16), (1, 15)]
    session.board_data(4, 29)  # evicts (1, 16)
    assert session.cached_boards() == [(1, 15), (4, 29)]
    u, v = session.channel_data(CLEAN)
    assert u.shape == v.shape == (700,) and u.dtype == np.int16
    assert loads == [(1, 15), (1, 16), (4, 29)]
    session.install(load_cache(results_files.cache))  # a new open empties the LRU
    assert session.cached_boards() == []


# ---------------------------------------------------------------------------
# Channel detail
# ---------------------------------------------------------------------------


def test_detail_reproduces_the_stored_fit(session: UVSession) -> None:
    result = session.result(OUTLIERS)
    assert result is not None and result.n_rejected
    detail = session.channel_detail(OUTLIERS)
    assert detail.consistent and detail.message == ""
    assert detail.params == result.params
    assert detail.kept is not None
    assert int(detail.kept.sum()) == result.n_used
    assert detail.n_rejected == result.n_rejected
    assert detail.u_corr is not None and detail.v_corr is not None
    radii = np.hypot(detail.u_corr[detail.kept], detail.v_corr[detail.kept])
    assert float(np.mean(radii)) == pytest.approx(result.target_radius, rel=0.01)


@pytest.mark.parametrize(
    ("key", "phrase"),
    [
        (TOO_FEW, f"Too few events (40 < min events {FitOptions().min_events})"),
        (FIT_FAILED, "fit failed"),
    ],
)
def test_detail_without_ellipse(session: UVSession, key: ChannelKey, phrase: str) -> None:
    detail = session.channel_detail(key)
    assert detail.params is None and detail.kept is None and detail.u_corr is None
    assert detail.n_events == session.channel_count(key)
    assert phrase in detail.message


def test_detail_without_results(ring_files: RingFiles) -> None:
    session = UVSession()
    session.install(load_cache(ring_files.cache))
    assert not session.has_results and session.result(CLEAN) is None
    detail = session.channel_detail(CLEAN)
    assert detail.params is None and "Run Fit All" in detail.message
    assert detail.n_events == 700


def test_detail_of_a_channel_without_events(session: UVSession) -> None:
    empty = ChannelKey(1, 15, 0, 4)  # active, on a board with data, no events
    assert session.channel_count(empty) == 0 and session.result(empty) is None
    detail = session.channel_detail(empty)
    assert detail.n_events == 0 and detail.message == "No events on this channel (no data)."


def test_detail_warns_when_the_refit_disagrees(
    session: UVSession, caplog: pytest.LogCaptureFixture
) -> None:
    request = session.detail_request(CLEAN)
    assert request.result is not None
    shifted = replace(request.result, centerU=request.result.centerU + 1.0)
    with caplog.at_level(logging.WARNING, logger="uvcorr.gui.session"):
        detail = session.compute_detail(replace(request, result=shifted))
    assert not detail.consistent and "did not reproduce" in detail.message
    assert any("re-running the fit" in record.message for record in caplog.records)
    # The stored (here: shifted) ellipse is what is drawn
    assert detail.params == shifted.params


def test_detail_request_is_a_snapshot(session: UVSession) -> None:
    request = session.detail_request(CLEAN)
    assert isinstance(request, DetailRequest) and request.has_results
    assert request.options == FitOptions()
    session.close()
    # Computing it later still works (the request carries the cache)
    assert session.compute_detail(request).consistent


def test_params_check_ignores_phi_of_a_circle() -> None:
    from uvcorr.gui.session import _params_agree

    circle = EllipseParams(cx=0.0, cy=0.0, a=100.0, b=100.0, phi=0.0)
    assert _params_agree(replace(circle, phi=1.0), circle)
    ellipse = replace(circle, b=50.0)
    assert _params_agree(replace(ellipse, phi=math.pi), ellipse)  # same axis
    assert not _params_agree(replace(ellipse, phi=0.1), ellipse)


# ---------------------------------------------------------------------------
# Batch and overrides
# ---------------------------------------------------------------------------


def test_run_batch_stores_and_applies(ring_files: RingFiles) -> None:
    session = UVSession()
    session.install(load_cache(ring_files.cache))
    progress: list[tuple[int, int]] = []
    options = FitOptions(clip_k=3.5)
    outcome = session.run_batch(
        options, workers=1, progress_cb=lambda done, total: progress.append((done, total))
    )
    # Units are whatever analyze_all reports (boards or events): check the ends only
    assert progress[0][0] == 0 and progress[-1][0] == progress[-1][1] > 0
    assert not session.has_results  # nothing changes before apply_batch
    views = session.apply_batch(outcome)
    assert len(views) == 18 and session.batch_options == options
    stored = UVCache(ring_files.cache).load_results()
    assert stored is not None and stored.options == options


def test_apply_batch_refuses_another_cache(session: UVSession, ring_files: RingFiles) -> None:
    other = UVSession()
    other.install(load_cache(ring_files.cache))
    outcome = other.run_batch(FitOptions(), workers=1)
    with pytest.raises(SessionError, match="no longer the open cache"):
        session.apply_batch(outcome)


def test_override_round_trip(session: UVSession, results_files: RingFiles) -> None:
    options = FitOptions(robust=False)
    refit = session.refit_channel(OUTLIERS, options)
    assert refit.options_source == OPTIONS_BATCH and refit.n_rejected == 0
    changed = session.store_override(refit, options)
    assert list(changed) == [OUTLIERS] and changed[OUTLIERS].is_override
    assert session.result(OUTLIERS).options_source == OPTIONS_OVERRIDE
    assert session.options_for(OUTLIERS) == (options, OPTIONS_OVERRIDE)
    # The detail uses the override's own options: nothing is rejected
    detail = session.channel_detail(OUTLIERS)
    assert detail.consistent and detail.kept is not None and bool(detail.kept.all())
    # Persisted, and kept by the next batch run
    reopened = load_cache(results_files.cache)
    assert reopened.stored is not None and OUTLIERS in reopened.stored.overrides
    outcome = session.run_batch(FitOptions(), workers=1)
    assert OUTLIERS in outcome.stored.overrides


def test_override_needs_batch_results(ring_files: RingFiles) -> None:
    session = UVSession()
    session.install(load_cache(ring_files.cache))
    result = session.refit_channel(CLEAN, FitOptions())
    with pytest.raises(ResultsError, match="Fit All first"):
        session.store_override(result, FitOptions())


def test_session_is_busy_during_a_batch(session: UVSession) -> None:
    result = session.result(CLEAN)
    assert result is not None
    session._batch_lock.acquire()
    try:
        assert session.batch_running
        with pytest.raises(SessionBusyError):
            session.run_batch(FitOptions(), workers=1)
        with pytest.raises(SessionBusyError):
            session.store_override(result, FitOptions())
        with pytest.raises(SessionBusyError):
            session.close()
    finally:
        session._batch_lock.release()
    assert not session.batch_running


def test_cache_busy_is_reported(
    session: UVSession, results_files: RingFiles, hold_h5_open, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache_module, "LOCK_RETRY_SECONDS", 0.1)
    with hold_h5_open(results_files.cache), pytest.raises(CacheBusyError, match="in use"):
        session.channel_detail(CLEAN)


# ---------------------------------------------------------------------------
# Re-fits, reverts and exports (phase 5)
# ---------------------------------------------------------------------------

ROBUST_OFF = FitOptions(robust=False)


def test_describe_options_change() -> None:
    batch = FitOptions()
    assert describe_options_change(batch, batch) == ""
    assert describe_options_change(ROBUST_OFF, batch) == "robust off"
    changed = FitOptions(clip_k=3.0, max_iter=8, geometric=True, min_events=1500)
    assert describe_options_change(changed, batch) == (
        "clip k 3, max iter 8, geometric on, min events 1,500"
    )
    assert describe_options_change(FitOptions(phase_ref_freq_hz=5e5), batch) == (
        "ref freq 500000 Hz"
    )
    assert short_title(ChannelKey(2, 16, 0, 27)) == "N2 B16 R0 Ch27"


def test_refit_channel_with_other_options_stores_an_override(
    session: UVSession, results_files: RingFiles
) -> None:
    before = session.result(OUTLIERS)
    assert before is not None and before.n_rejected
    request = session.refit_request(ROBUST_OFF, channel=OUTLIERS)
    assert request.as_override and request.title == "N1 B15 R0 Ch12" and not request.is_board
    progress: list[tuple[int, int]] = []
    outcome = session.run_refit(request, progress_cb=lambda d, t: progress.append((d, t)))
    assert progress == [(0, 1), (1, 1)]
    assert outcome.saved == (OUTLIERS,) and outcome.removed == () and outcome.changed == (OUTLIERS,)
    assert outcome.results[0].options_source == OPTIONS_OVERRIDE
    assert outcome.describe().startswith("Override saved for N1 B15 R0 Ch12 (robust off); ")
    assert session.result(OUTLIERS) == before  # nothing changes before apply_refit
    assert session.apply_refit(outcome) == (OUTLIERS,)
    result = session.result(OUTLIERS)
    assert result is not None and result.options_source == OPTIONS_OVERRIDE
    assert result.n_rejected == 0
    assert session.override_keys() == [OUTLIERS] == session.override_keys(1, 15)
    assert session.override_keys(1, 16) == []
    assert session.override_options(OUTLIERS) == ROBUST_OFF
    assert session.override_change(OUTLIERS) == "robust off"
    assert session.override_change(CLEAN) is None
    view = session.view(OUTLIERS)
    assert view is not None and view.is_override
    stored = UVCache(results_files.cache).load_results()
    assert stored is not None and stored.overrides[OUTLIERS].options == ROBUST_OFF
    assert "1 override" in session.describe()


def test_refit_with_the_batch_options_reverts(session: UVSession) -> None:
    session.apply_refit(session.run_refit(session.refit_request(ROBUST_OFF, channel=OUTLIERS)))
    batch = session.batch_options
    assert batch is not None
    outcome = session.run_refit(session.refit_request(batch, channel=OUTLIERS))
    assert not outcome.request.as_override
    assert outcome.saved == () and outcome.removed == (OUTLIERS,)
    assert outcome.describe().startswith("N1 B15 R0 Ch12 reverted to batch (override removed)")
    assert session.apply_refit(outcome) == (OUTLIERS,)
    result = session.result(OUTLIERS)
    assert result is not None and result.options_source == OPTIONS_BATCH
    assert session.override_keys() == []
    cached = session.cache
    assert cached is not None
    stored = cached.load_results()
    assert stored is not None and stored.overrides == {}
    # Again: no override to remove, nothing stored
    again = session.run_refit(session.refit_request(batch, channel=OUTLIERS))
    assert again.changed == () and "it matches the batch, nothing stored" in again.describe()


def test_refit_board_stores_every_channel_in_one_write(
    session: UVSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = session.cache
    assert cache is not None
    writes: list[int] = []
    real_replace = UVCache.replace_overrides

    def counting(self: UVCache, save=(), options=None, delete=(), **kwargs):  # type: ignore[no-untyped-def]
        save = list(save)
        writes.append(len(save))
        return real_replace(self, save, options, delete, **kwargs)

    monkeypatch.setattr(UVCache, "replace_overrides", counting)
    options = FitOptions(clip_k=3.0)
    request = session.refit_request(options, board=(1, 16))
    assert request.is_board and request.title == "N1 B16"
    progress: list[tuple[int, int]] = []
    outcome = session.run_refit(request, progress_cb=lambda d, t: progress.append((d, t)))
    assert writes == [CHANNELS_PER_BOARD]
    assert progress[0] == (0, CHANNELS_PER_BOARD) and progress[-1] == (6, 6)
    assert outcome.saved == tuple(session.channels_on_board(1, 16))
    assert outcome.describe().startswith("Overrides saved for the 6 channels of N1 B16 (clip k 3)")
    session.apply_refit(outcome)
    assert session.override_keys(1, 16) == session.channels_on_board(1, 16)
    # A board re-fit with the batch options reverts the whole board
    batch = session.batch_options
    assert batch is not None
    revert = session.run_refit(session.refit_request(batch, board=(1, 16)))
    assert revert.removed == outcome.saved and revert.saved == ()
    assert revert.describe().startswith("N1 B16 reverted to batch (6 overrides removed)")
    session.apply_refit(revert)
    assert session.override_keys() == []


def test_refit_stop_stores_nothing(session: UVSession) -> None:
    stop = threading.Event()

    def progress(done: int, _total: int) -> None:
        if done == 2:
            stop.set()

    request = session.refit_request(ROBUST_OFF, board=(1, 15))
    with pytest.raises(AnalysisCancelled):
        session.run_refit(request, progress_cb=progress, stop_flag=stop)
    assert not session.batch_running
    cache = session.cache
    assert cache is not None
    stored = cache.load_results()
    assert stored is not None and stored.overrides == {}


def test_refit_request_errors(session: UVSession, ring_files: RingFiles) -> None:
    with pytest.raises(ValueError, match="channel or a board"):
        session.refit_request(ROBUST_OFF)
    with pytest.raises(ValueError, match="channel or a board"):
        session.refit_request(ROBUST_OFF, channel=CLEAN, board=(1, 15))
    with pytest.raises(SessionError, match="no events"):
        session.refit_request(ROBUST_OFF, channel=ChannelKey(1, 15, 0, 4))
    with pytest.raises(SessionError, match="no events"):
        session.refit_request(ROBUST_OFF, board=(2, 20))
    request = session.refit_request(ROBUST_OFF, channel=CLEAN)
    session._batch_lock.acquire()
    try:
        with pytest.raises(SessionBusyError):
            session.refit_request(ROBUST_OFF, channel=CLEAN)
        with pytest.raises(SessionBusyError):
            session.run_refit(request)
        with pytest.raises(SessionBusyError):
            session.revert_overrides([CLEAN])
        with pytest.raises(SessionBusyError):
            session.clear_overrides()
    finally:
        session._batch_lock.release()
    unfitted = UVSession()
    with pytest.raises(SessionError, match="No file"):
        unfitted.refit_request(ROBUST_OFF, channel=CLEAN)
    unfitted.install(load_cache(ring_files.cache))
    with pytest.raises(ResultsError, match="run Fit All first"):
        unfitted.refit_request(ROBUST_OFF, channel=CLEAN)


def test_apply_refit_refuses_a_stale_outcome(session: UVSession) -> None:
    session.apply_refit(session.run_refit(session.refit_request(ROBUST_OFF, channel=OUTLIERS)))
    outcome = session.run_refit(session.refit_request(ROBUST_OFF, channel=CLEAN))
    session.revert_overrides([OUTLIERS])  # the stored results changed since the request
    with pytest.raises(SessionError, match="changed while the re-fit ran"):
        session.apply_refit(outcome)


def test_a_refit_and_a_batch_exclude_each_other(session: UVSession) -> None:
    request = session.refit_request(ROBUST_OFF, channel=CLEAN)
    with session._exclusive("A re-fit"), pytest.raises(SessionBusyError, match="A re-fit is"):
        session.run_batch(FitOptions(), workers=1)
    with session._exclusive("Fit All"), pytest.raises(SessionBusyError, match="Fit All is"):
        session.run_refit(request)


def test_revert_board_and_clear(session: UVSession) -> None:
    session.apply_refit(session.run_refit(session.refit_request(ROBUST_OFF, board=(1, 15))))
    session.apply_refit(
        session.run_refit(session.refit_request(ROBUST_OFF, channel=CLEAN._replace(board=16)))
    )
    assert len(session.override_keys()) == 7
    assert session.revert_overrides([CLEAN, ChannelKey(1, 15, 0, 4)]) == (CLEAN,)
    assert session.revert_overrides([CLEAN]) == ()
    assert session.revert_board(1, 15) == tuple(
        k for k in session.channels_on_board(1, 15) if k != CLEAN
    )
    assert session.override_keys() == [CLEAN._replace(board=16)]
    assert session.clear_overrides() == (CLEAN._replace(board=16),)
    assert session.clear_overrides() == ()
    cache = session.cache
    assert cache is not None
    stored = cache.load_results()
    assert stored is not None and stored.overrides == {}
    assert all(r.options_source == OPTIONS_BATCH for r in session.merged_results())


def test_run_batch_can_discard_the_overrides(session: UVSession) -> None:
    session.apply_refit(session.run_refit(session.refit_request(ROBUST_OFF, channel=CLEAN)))
    kept = session.run_batch(FitOptions(), workers=1)
    assert kept.keep_overrides and CLEAN in kept.stored.overrides
    session.apply_batch(kept)
    discarded = session.run_batch(FitOptions(), keep_overrides=False, workers=1)
    assert not discarded.keep_overrides and discarded.stored.overrides == {}
    session.apply_batch(discarded)
    assert session.override_keys() == []


def test_exports_write_the_merged_results(session: UVSession, tmp_path: Path) -> None:
    assert session.export_stem == "rings"
    session.apply_refit(session.run_refit(session.refit_request(ROBUST_OFF, channel=OUTLIERS)))
    summary = session.export_outputs(tmp_path / "out")
    tec, csv = tmp_path / "out" / "rings.tec", tmp_path / "out" / "radial_summary.csv"
    assert summary.paths == (tec, csv)
    rows = read_summary_csv(csv)
    assert summary.n_rows == len(rows) == 18
    assert summary.n_overrides == 1
    by_key = {row.key: row for row in rows}
    assert by_key[OUTLIERS].options_source == OPTIONS_OVERRIDE
    assert by_key[OUTLIERS].n_rejected == 0
    entries = read_tec(tec)
    assert summary.n_ok == len(entries) == sum(1 for row in rows if row.status == STATUS_OK)
    assert entries[OUTLIERS].params.a == pytest.approx(by_key[OUTLIERS].semiMajor, rel=1e-5)
    single = session.export_tec(tmp_path / "single.tec")
    assert single.paths == (tmp_path / "single.tec",) and read_tec(single.paths[0]) == entries
    assert session.export_csv(tmp_path / "s.csv").n_rows == 18


def test_exports_need_results(ring_files: RingFiles, tmp_path: Path) -> None:
    session = UVSession()
    with pytest.raises(SessionError, match="No file"):
        session.export_tec(tmp_path / "x.tec")
    session.install(load_cache(ring_files.cache))
    with pytest.raises(SessionError, match="run Fit All first"):
        session.export_outputs(tmp_path / "out")
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# Review fixes: effective options, stale results, export guard, single writes
# ---------------------------------------------------------------------------


def test_effective_options_ignore_clip_settings_without_robust() -> None:
    plain = FitOptions(robust=False)
    assert effective_options(plain) == plain
    odd = FitOptions(robust=False, clip_k=3.0, max_iter=9)
    assert effective_options(odd) == plain and same_fit(odd, plain)
    robust = FitOptions(clip_k=3.0)
    assert effective_options(robust) is robust and not same_fit(robust, FitOptions())
    assert not same_fit(odd, FitOptions(robust=False, min_events=50))
    # Descriptions compare effective options on both sides
    assert describe_options_change(odd, plain) == ""
    assert describe_options_change(plain, FitOptions(clip_k=3.0)) == "robust off"
    assert describe_options_change(FitOptions(clip_k=3.0), odd) == "robust on, clip k 3"


def test_robust_off_batch_ignores_the_clip_settings(ring_files: RingFiles) -> None:
    """A batch fitted without the robust iteration: clip k decides nothing (review s9)."""
    store_batch_results(ring_files.cache, FitOptions(robust=False))
    session = UVSession()
    session.install(load_cache(ring_files.cache))
    request = session.refit_request(FitOptions(clip_k=3.0), channel=OUTLIERS)
    assert request.as_override
    session.apply_refit(session.run_refit(request))
    assert session.override_change(OUTLIERS) == "robust on, clip k 3"
    # Robust off again, with the clip spin still showing 3: the batch fit, so it reverts
    request = session.refit_request(FitOptions(robust=False, clip_k=3.0), channel=OUTLIERS)
    assert not request.as_override and request.n_reverted == 1
    assert request.describe_start() == (
        "Fit Channel N1 B15 R0 Ch12 with the batch options (reverts its override)…"
    )
    outcome = session.run_refit(request)
    assert outcome.removed == (OUTLIERS,) and outcome.saved == ()
    session.apply_refit(outcome)
    assert session.override_keys() == []


def test_fit_all_drops_overrides_fitted_with_the_new_options(session: UVSession) -> None:
    session.apply_refit(
        session.run_refit(session.refit_request(FitOptions(robust=False), channel=OUTLIERS))
    )
    session.apply_refit(
        session.run_refit(session.refit_request(FitOptions(clip_k=3.0), channel=CLEAN))
    )
    new_batch = FitOptions(robust=False, clip_k=2.0)  # fits like the OUTLIERS override
    assert session.overrides_fitting_like(new_batch) == [OUTLIERS]
    outcome = session.run_batch(new_batch, workers=1)
    assert outcome.keep_overrides and outcome.n_dropped == 1
    assert list(outcome.stored.overrides) == [CLEAN]
    session.apply_batch(outcome)
    assert session.override_keys() == [CLEAN]
    discard = session.run_batch(new_batch, keep_overrides=False, workers=1)
    assert discard.n_dropped == 1 and discard.stored.overrides == {}


def _replace_batch_on_disk(results_files: RingFiles) -> None:
    """What ``uvcorr process`` in another terminal does: store a new batch (overrides kept)."""
    store_batch_results(results_files.cache, FitOptions(clip_k=3.0))


def test_refit_refuses_results_replaced_on_disk(
    session: UVSession, results_files: RingFiles
) -> None:
    session.apply_refit(
        session.run_refit(session.refit_request(FitOptions(robust=False), channel=CLEAN))
    )
    request = session.refit_request(FitOptions(robust=False), channel=OUTLIERS)
    _replace_batch_on_disk(results_files)
    on_disk = UVCache(results_files.cache).load_results()
    assert on_disk is not None and list(on_disk.overrides) == [CLEAN]
    assert session.results_changed_on_disk()
    with pytest.raises(SessionError, match=r"stored results changed on disk \(another process"):
        session.run_refit(request)
    with pytest.raises(SessionError, match="reopen the file"):
        session.revert_overrides([CLEAN])
    with pytest.raises(SessionError, match="changed on disk"):
        session.clear_overrides()
    after = UVCache(results_files.cache).load_results()
    assert after == on_disk  # nothing was written into the new results
    assert not session.batch_running
    # Reopening picks up the new batch; writes work again
    session.install(load_cache(results_files.cache))
    assert not session.results_changed_on_disk()
    assert session.revert_overrides([CLEAN]) == (CLEAN,)


def test_export_warns_about_results_replaced_on_disk(
    session: UVSession, results_files: RingFiles, tmp_path: Path
) -> None:
    assert session.export_csv(tmp_path / "a.csv").warning == ""
    _replace_batch_on_disk(results_files)
    summary = session.export_csv(tmp_path / "b.csv")
    assert "changed on disk" in summary.warning and summary.paths == (tmp_path / "b.csv",)


def test_refit_revert_with_a_save_is_one_write(
    session: UVSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch-options board re-fit that also stores a channel without a batch row."""
    session.apply_refit(
        session.run_refit(session.refit_request(FitOptions(robust=False), board=(1, 15)))
    )
    batch = session.batch_options
    assert batch is not None
    request = session.refit_request(batch, board=(1, 15))
    stored = request.stored
    # A batch row that a newer uvcorr could leave unreadable: CLEAN has none
    request = replace(
        request, stored=replace(stored, results=tuple(r for r in stored.results if r.key != CLEAN))
    )
    calls: list[tuple[int, int]] = []
    real_replace = UVCache.replace_overrides

    def counting(self: UVCache, save=(), options=None, delete=(), **kwargs):  # type: ignore[no-untyped-def]
        save, delete = list(save), list(delete)
        calls.append((len(save), len(delete)))
        return real_replace(self, save, options, delete, **kwargs)

    monkeypatch.setattr(UVCache, "replace_overrides", counting)
    outcome = session.run_refit(request)
    assert calls == [(1, 5)]
    assert outcome.saved == (CLEAN,) and len(outcome.removed) == 5
    # A failure of that one write changes nothing
    monkeypatch.setattr(cache_module, "_write_override_entries", _raise_os_error)
    before = UVCache(request.cache.path).load_results()
    with pytest.raises(OSError, match="simulated"):
        session.run_refit(request)
    assert UVCache(request.cache.path).load_results() == before


def _raise_os_error(*_args: object) -> None:
    raise OSError("simulated HDF5 error")


def test_revert_requests(session: UVSession) -> None:
    assert session.revert_request([CLEAN]) is None  # no override
    session.apply_refit(
        session.run_refit(session.refit_request(FitOptions(robust=False), board=(1, 15)))
    )
    channel = session.revert_request([CLEAN])
    assert channel is not None and channel.keys == (CLEAN,) and not channel.is_all
    board = session.revert_request(board=(1, 15))
    assert board is not None and len(board.keys) == 6 and board.title == "N1 B15"
    everything = session.revert_request()
    assert everything is not None and everything.is_all
    outcome = session.run_revert(channel)
    assert outcome.describe() == "N1 B15 R0 Ch05 reverted to batch (override removed)"
    assert session.override_keys(1, 15) != []  # not applied yet
    assert session.apply_revert(outcome) == (CLEAN,)
    with pytest.raises(SessionError, match="changed while the revert ran"):
        session.apply_revert(session.run_revert(board))  # the request predates the revert
    board = session.revert_request(board=(1, 15))
    assert board is not None
    assert session.run_revert(board).describe() == "N1 B15 reverted to batch (5 overrides removed)"


def test_export_refuses_to_overwrite_data(results_files: RingFiles, tmp_path: Path) -> None:
    """Review s10: an export must never replace the raw file, the cache or another cache."""
    dat, cache = results_files.dat, results_files.cache
    session = UVSession()
    session.install(load_raw(dat))  # reuses the valid cache next to the private .dat copy
    assert session.dat_path == dat and session.cache_path == cache
    sizes = {path: path.stat().st_size for path in (dat, cache)}
    link = tmp_path / "link.uv.h5"
    link.symlink_to(cache)
    hard = tmp_path / "hard.bin"
    os.link(dat, hard)
    other_cache = tmp_path / "other.h5"
    with h5py.File(other_cache, "w") as h5f:
        h5f["x"] = [1]
    other_dat = tmp_path / "old_run.dat"
    other_dat.write_bytes(b"raw bytes")
    cases = [
        (session.export_tec, dat, "open raw data file"),
        (session.export_csv, cache, "open UV cache"),
        (session.export_tec, link, "open UV cache"),
        (session.export_csv, hard, "open raw data file"),
        (session.export_csv, other_cache, "HDF5 file"),
        (session.export_tec, other_dat, "raw .dat file"),
    ]
    for export, target, reason in cases:
        with pytest.raises(SessionError, match=reason):
            export(target)
    # Export both: either output path is checked
    folder = tmp_path / "out"
    folder.mkdir()
    csv_as_cache = folder / "radial_summary.csv"
    csv_as_cache.write_bytes(other_cache.read_bytes())
    with pytest.raises(SessionError, match="HDF5 file"):
        session.export_outputs(folder)
    assert not (folder / "rings.tec").exists()  # nothing was written
    with pytest.raises(SessionError, match="open raw data file"):
        _export_onto_dat(session, dat)
    assert {path: path.stat().st_size for path in (dat, cache)} == sizes
    assert h5py.is_hdf5(cache) and other_dat.read_bytes() == b"raw bytes"
    # A new file with a .dat suffix, or a plain existing text file, is fine
    assert session.export_csv(tmp_path / "new.dat").n_rows == 18
    (tmp_path / "old.csv").write_text("old")
    assert session.export_csv(tmp_path / "old.csv").n_rows == 18


def _export_onto_dat(session: UVSession, dat: Path) -> None:
    """Export Both into a folder whose ``<stem>.tec`` is the open raw file (renamed .tec)."""
    target = dat.parent / "tecdir"
    target.mkdir()
    (target / f"{session.export_stem}.tec").symlink_to(dat)
    session.export_outputs(target)


def test_export_errors_name_the_target(session: UVSession, tmp_path: Path) -> None:
    target = tmp_path / "missing" / "x.tec"
    with pytest.raises(OSError) as info:
        session.export_tec(target)
    message = str(info.value)
    assert message.startswith(f"Cannot write {target}: ") and ".tmp" not in message
    folder = tmp_path / "ro"
    folder.mkdir()
    folder.chmod(0o500)
    try:
        if os.access(folder, os.W_OK):  # root ignores permissions
            pytest.skip("the directory is writable anyway")
        with pytest.raises(OSError) as info:
            session.export_outputs(folder)
        message = str(info.value)
        assert message == f"Cannot write to the directory {folder}: Permission denied"
    finally:
        folder.chmod(0o700)
