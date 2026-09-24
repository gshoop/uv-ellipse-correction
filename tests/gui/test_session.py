"""Tests for the GUI data layer ``uvcorr.gui.session`` (no widgets needed)."""

from __future__ import annotations

import logging
import math
import os
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
    copy_files,
    store_batch_results,
)
from uvcorr import cache as cache_module
from uvcorr.analysis import OPTIONS_BATCH, OPTIONS_OVERRIDE, ChannelKey
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
    inspect_raw,
    is_cache_file,
    load_cache,
    load_raw,
)
from uvcorr.options import (
    FLAG_EXTREME_AXIS_RATIO,
    FLAG_HIGH_REJECTION,
    STATUS_FIT_FAILED,
    STATUS_OK,
    STATUS_TOO_FEW_EVENTS,
    FitOptions,
)


@pytest.fixture(scope="module")
def _results_master(
    tmp_path_factory: pytest.TempPathFactory, _ring_files_master: RingFiles
) -> RingFiles:
    dat, cache = copy_files(
        _ring_files_master.dat, _ring_files_master.cache, tmp_path_factory.mktemp("results")
    )
    store_batch_results(cache)
    return RingFiles(dat, cache)


@pytest.fixture
def results_files(tmp_path: Path, _results_master: RingFiles) -> RingFiles:
    """A private copy of the ring ``.dat`` and its cache with stored batch results."""
    dat, cache = copy_files(_results_master.dat, _results_master.cache, tmp_path / "res")
    return RingFiles(dat, cache)


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
