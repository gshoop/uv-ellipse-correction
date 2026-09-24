"""Tests for the GUI worker threads (``uvcorr.gui.threads``)."""

from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from tests.conftest import RingFiles
from uvcorr.analysis import AnalysisError, WorkerCrashedError
from uvcorr.cache import UVCacheError
from uvcorr.gui.session import OpenedFile, UVSession, load_cache
from uvcorr.gui.threads import (
    WORKER_CRASH_MESSAGE,
    CacheBuildThread,
    CacheOpenThread,
    ChannelDetailThread,
    FitAllThread,
    error_text,
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


def test_error_text() -> None:
    assert error_text(UVCacheError("cache busy")) == "cache busy"
    assert error_text(KeyError("No cached events")) == "No cached events"
    assert error_text(RuntimeError("boom")) == "RuntimeError: boom"
    assert error_text(MemoryError()) == "MemoryError"
    assert error_text(AnalysisError("board failed", 1, 15)) == "board failed"
    # One GUI-worded message, not the analysis's CLI advice
    assert error_text(WorkerCrashedError()) == WORKER_CRASH_MESSAGE
    assert "--workers 1 (workers=1)" not in WORKER_CRASH_MESSAGE
