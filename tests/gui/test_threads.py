"""Tests for the GUI worker threads (``uvcorr.gui.threads``)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from tests.conftest import RingFiles
from tests.gui.ring_cache import OUTLIERS
from uvcorr.analysis import AnalysisError, WorkerCrashedError
from uvcorr.cache import UVCache, UVCacheError
from uvcorr.gui.session import OpenedFile, UVSession, load_cache
from uvcorr.gui.threads import (
    WORKER_CRASH_MESSAGE,
    BoardGridThread,
    CacheBuildThread,
    CacheOpenThread,
    ChannelDetailThread,
    FitAllThread,
    RefitThread,
    RevertThread,
    error_text,
    log_failure,
)
from uvcorr.options import FitOptions

pytestmark = pytest.mark.gui


def test_cache_build_thread_builds(qtbot: QtBot, ring_files: RingFiles) -> None:
    ring_files.cache.unlink()
    thread = CacheBuildThread(ring_files.dat)
    progress: list[float] = []
    thread.progress.connect(progress.append)
    with qtbot.waitSignal(thread.finished, timeout=20_000) as blocker:
        thread.start()
    thread.wait()
    opened = blocker.args[0]
    assert isinstance(opened, OpenedFile) and opened.built
    assert progress and progress[-1] == 1.0


def test_cache_build_thread_stop(qtbot: QtBot, ring_files: RingFiles) -> None:
    ring_files.cache.unlink()
    thread = CacheBuildThread(ring_files.dat)
    thread.stop()  # honoured at the first parser batch
    assert thread.stop_requested
    with qtbot.waitSignal(thread.stopped, timeout=20_000):
        thread.start()
    thread.wait()
    assert not ring_files.cache.exists()


def test_cache_open_thread_error(qtbot: QtBot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    thread = CacheOpenThread(tmp_path / "missing.uv.h5")
    with qtbot.waitSignal(thread.error, timeout=20_000) as blocker:
        thread.start()
    thread.wait()
    assert "not found" in blocker.args[0]


def test_fit_all_thread(qtbot: QtBot, ring_files: RingFiles) -> None:
    session = UVSession()
    session.install(load_cache(ring_files.cache))
    thread = FitAllThread(session, FitOptions(), workers=1)
    progress: list[tuple[int, int]] = []
    thread.progress.connect(lambda done, total: progress.append((done, total)))
    with qtbot.waitSignal(thread.finished, timeout=20_000) as blocker:
        thread.start()
    thread.wait()
    views = session.apply_batch(blocker.args[0])
    assert len(views) == 18 and progress[-1][0] == progress[-1][1]


def test_fit_all_thread_stop_and_error(qtbot: QtBot, ring_files: RingFiles) -> None:
    session = UVSession()
    session.install(load_cache(ring_files.cache))
    thread = FitAllThread(session, FitOptions(), workers=1)
    thread.stop()
    with qtbot.waitSignal(thread.stopped, timeout=20_000):
        thread.start()
    thread.wait()
    assert session.stored is None and not session.batch_running

    unopened = FitAllThread(UVSession(), FitOptions(), workers=1)
    with qtbot.waitSignal(unopened.error, timeout=20_000) as blocker:
        unopened.start()
    unopened.wait()
    assert blocker.args == ["No file is open"]


def test_channel_detail_thread(qtbot: QtBot, ring_files: RingFiles) -> None:
    session = UVSession()
    session.install(load_cache(ring_files.cache))
    key = session.data_channels[0]
    thread = ChannelDetailThread(session, session.detail_request(key), generation=7)
    with qtbot.waitSignal(thread.done, timeout=20_000) as blocker:
        thread.start()
    thread.wait()
    assert blocker.args[0] == 7 and blocker.args[1].key == key


def test_non_raw_file_is_a_user_error_without_traceback(
    qtbot: QtBot, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    junk = tmp_path / "junk.dat"
    junk.write_bytes(b"\x00" * 4096)
    thread = CacheBuildThread(junk)
    with caplog.at_level(logging.INFO, logger="uvcorr.gui.threads"):
        with qtbot.waitSignal(thread.error, timeout=20_000) as blocker:
            thread.start()
        thread.wait()
    assert blocker.args[0] == f"no valid frames in {junk}: is this a raw .dat file?"
    records = [r for r in caplog.records if r.name == "uvcorr.gui.threads"]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert records[0].exc_info is None and "no valid frames" in records[0].getMessage()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["junk.dat"]


def test_log_failure(caplog: pytest.LogCaptureFixture) -> None:
    def failing(exc: Exception) -> None:
        try:
            raise exc
        except Exception as caught:
            log_failure("Doing it", caught)

    with caplog.at_level(logging.INFO, logger="uvcorr.gui.threads"):
        failing(UVCacheError("the cache is busy"))  # written for the user: no traceback
        failing(RuntimeError("a bug"))  # unexpected: traceback
    user, bug = caplog.records
    assert user.levelno == logging.WARNING and user.exc_info is None
    assert user.getMessage() == "Doing it failed: the cache is busy"
    assert bug.levelno == logging.ERROR and bug.exc_info is not None
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="uvcorr.gui.threads"):
        failing(UVCacheError("the cache is busy"))  # -vv: the traceback is added
    assert caplog.records[0].levelno == logging.WARNING and caplog.records[0].exc_info


def test_error_text() -> None:
    assert error_text(UVCacheError("cache busy")) == "cache busy"
    assert error_text(KeyError("No cached events")) == "No cached events"
    assert error_text(RuntimeError("boom")) == "RuntimeError: boom"
    assert error_text(MemoryError()) == "MemoryError"
    assert error_text(AnalysisError("board failed", 1, 15)) == "board failed"
    # One GUI-worded message, not the analysis's CLI advice
    assert error_text(WorkerCrashedError()) == WORKER_CRASH_MESSAGE
    assert "--workers 1 (workers=1)" not in WORKER_CRASH_MESSAGE


def test_refit_thread(qtbot: QtBot, results_files: RingFiles) -> None:
    session = UVSession()
    session.install(load_cache(results_files.cache))
    request = session.refit_request(FitOptions(robust=False), board=(1, 15))
    thread = RefitThread(session, request)
    progress: list[tuple[int, int]] = []
    thread.progress.connect(lambda done, total: progress.append((done, total)))
    with qtbot.waitSignal(thread.finished, timeout=20_000) as blocker:
        thread.start()
    thread.wait()
    changed = session.apply_refit(blocker.args[0])
    assert OUTLIERS in changed and len(changed) == 6 and progress[-1] == (6, 6)

    stopped = RefitThread(session, session.refit_request(FitOptions(), board=(1, 15)))
    stopped.stop()
    with qtbot.waitSignal(stopped.stopped, timeout=20_000):
        stopped.start()
    stopped.wait()
    assert len(session.override_keys(1, 15)) == 6 and not session.batch_running


def test_revert_thread(qtbot: QtBot, results_files: RingFiles) -> None:
    session = UVSession()
    session.install(load_cache(results_files.cache))
    session.apply_refit(
        session.run_refit(session.refit_request(FitOptions(robust=False), board=(1, 15)))
    )
    request = session.revert_request(board=(1, 15))
    assert request is not None
    thread = RevertThread(session, request)
    with qtbot.waitSignal(thread.finished, timeout=20_000) as blocker:
        thread.start()
    thread.wait()
    assert len(session.apply_revert(blocker.args[0])) == 6 and session.override_keys() == []


def test_channel_detail_thread_always_ends_with_views_or_failed(
    qtbot: QtBot, ring_files: RingFiles, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error in the second stage ends the load with ``failed`` (the window's queue goes on)."""
    session = UVSession()
    session.install(load_cache(ring_files.cache))
    key = session.data_channels[0]

    def broken(_self: ChannelDetailThread, _detail: object) -> None:
        raise RuntimeError("views broke")

    monkeypatch.setattr(ChannelDetailThread, "_compute_views", broken)
    thread = ChannelDetailThread(session, session.detail_request(key), generation=3)
    signals: list[str] = []
    thread.done.connect(lambda *_: signals.append("done"))
    thread.views.connect(lambda *_: signals.append("views"))
    with qtbot.waitSignal(thread.failed, timeout=20_000) as blocker:
        thread.start()
    thread.wait()
    qtbot.waitUntil(lambda: "done" in signals, timeout=5_000)
    assert blocker.args == [3, "RuntimeError: views broke"] and "views" not in signals
    # Skipped views: views(None) is the last signal
    thread = ChannelDetailThread(session, session.detail_request(key), generation=4)
    thread.skip_views()
    with qtbot.waitSignal(thread.views, timeout=20_000) as blocker:
        thread.start()
    thread.wait()
    assert blocker.args == [4, None]


def test_board_grid_thread_reads_the_cache_it_was_given(
    qtbot: QtBot, ring_files: RingFiles
) -> None:
    """The cache is a snapshot from the GUI thread, not the session's current file."""
    session = UVSession()  # no file open: the thread must not ask the session for one
    thread = BoardGridThread(session, UVCache(ring_files.cache), 1, 15, {}, generation=2)
    with qtbot.waitSignal(thread.done, timeout=20_000) as blocker:
        thread.start()
    thread.wait()
    generation, data = blocker.args
    assert generation == 2 and (data.node, data.board) == (1, 15) and data.n_events > 0
