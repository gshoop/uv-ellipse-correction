"""Tests for the ``uvcorr build-cache`` subcommand."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import shutil
import signal
import subprocess
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import h5py
import pytest

from tests.conftest import SyntheticFile
from tests.synthetic_dat import Frame, Hit, write_dat
from uvcorr import cache, cli
from uvcorr.cache import CacheBuildCancelled, UVCache, default_cache_path

HoldOpen = Callable[[Path], AbstractContextManager[subprocess.Popen[str]]]


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_build_then_reuse_then_force(
    synthetic_file: SyntheticFile, capsys: pytest.CaptureFixture[str]
) -> None:
    dat = synthetic_file.path
    cache_path = default_cache_path(dat)
    n_kept = len(synthetic_file.active_events)

    code, out, err = run(["build-cache", str(dat)], capsys)
    assert code == cli.EXIT_OK
    assert UVCache(cache_path).is_valid_for(dat)
    assert "Building UV cache" in err and "no cache yet" in err
    assert "parsing: 100%" in err
    assert f"UV cache: {cache_path}" in out
    assert "built in" in out
    assert f"events kept:   {n_kept:,} on " in out
    assert "nodes 1-2,5,10, boards 15-16,29-30" in out
    assert "from node 0" in out and "inactive channels" in out
    mtime = cache_path.stat().st_mtime_ns

    code, out, err = run(["build-cache", str(dat)], capsys)
    assert code == cli.EXIT_OK
    assert "reused" in out
    assert err == ""
    assert cache_path.stat().st_mtime_ns == mtime

    code, out, err = run(["build-cache", str(dat), "--force"], capsys)
    assert code == cli.EXIT_OK
    assert "--force" in err
    assert "built in" in out
    assert UVCache(cache_path).is_valid_for(dat)


def test_stale_cache_is_rebuilt(
    synthetic_file: SyntheticFile, capsys: pytest.CaptureFixture[str]
) -> None:
    dat = synthetic_file.path
    assert run(["build-cache", str(dat)], capsys)[0] == cli.EXIT_OK
    st = dat.stat()
    os.utime(dat, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    code, out, err = run(["build-cache", str(dat)], capsys)
    assert code == cli.EXIT_OK
    assert "not a valid cache" in err
    assert "built in" in out


def test_custom_cache_path(
    synthetic_file: SyntheticFile, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "caches" / "run.uv.h5"
    code, out, _ = run(["build-cache", str(synthetic_file.path), "--cache", str(target)], capsys)
    assert code == cli.EXIT_OK
    assert target.is_file()
    assert not default_cache_path(synthetic_file.path).exists()
    assert str(target) in out


def test_missing_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run(["build-cache", str(tmp_path / "nope.dat")], capsys)
    assert code == cli.EXIT_USAGE
    assert "not found" in err
    assert out == ""
    assert not (tmp_path / "nope.dat.uv.h5").exists()


def test_insufficient_disk_space(
    synthetic_file: SyntheticFile,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _p: shutil._ntuple_diskusage(10**9, 10**9, 0),  # type: ignore[attr-defined]
    )
    code, out, err = run(["build-cache", str(synthetic_file.path)], capsys)
    assert code == cli.EXIT_ERROR
    assert "error:" in err and "disk space" in err
    assert out == ""
    assert not default_cache_path(synthetic_file.path).exists()


def test_cancelled_build(
    synthetic_file: SyntheticFile,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cancelled(*_args: Any, **_kwargs: Any) -> None:
        raise CacheBuildCancelled("stopped")

    monkeypatch.setattr(UVCache, "build_from_dat", cancelled)
    code, out, err = run(["build-cache", str(synthetic_file.path)], capsys)
    assert code == cli.EXIT_INTERRUPTED
    assert "cancelled" in err
    assert out == ""


def test_ctrl_c_stops_build_cleanly(
    synthetic_file: SyntheticFile,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A SIGINT during the build sets the stop flag; the build is cancelled at
    # the next check and leaves neither a cache nor a temporary file.
    real_call = cli._ProgressPrinter.__call__

    def progress_with_sigint(self: Any, fraction: float) -> None:
        signal.raise_signal(signal.SIGINT)
        real_call(self, fraction)

    monkeypatch.setattr(cli._ProgressPrinter, "__call__", progress_with_sigint)
    previous = signal.getsignal(signal.SIGINT)
    code, out, err = run(["build-cache", str(synthetic_file.path)], capsys)
    assert code == cli.EXIT_INTERRUPTED
    assert "stopping the build" in err and "cancelled" in err
    cache = UVCache(default_cache_path(synthetic_file.path))
    assert not cache.exists() and cache.tmp_files() == []
    assert signal.getsignal(signal.SIGINT) is previous


def test_sigint_handler_never_raises_until_restored(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Repeated Ctrl-C only sets the stop flag: no KeyboardInterrupt can land in
    # the build's cleanup. The previous handler is back once the block exits.
    stop = threading.Event()
    previous = signal.getsignal(signal.SIGINT)
    with cli._sigint_sets(stop):
        for _ in range(3):
            signal.raise_signal(signal.SIGINT)
        assert stop.is_set()
    assert signal.getsignal(signal.SIGINT) is previous
    err = capsys.readouterr().err
    assert err.count("stopping the build") == 3 and err.count("still stopping") == 2


def test_progress_printer_non_tty_prints_every_ten_percent() -> None:
    stream = io.StringIO()
    printer = cli._ProgressPrinter("parsing", stream)
    for i in range(101):
        printer(i / 100)
    printer(1.0)
    printer.close()
    lines = stream.getvalue().splitlines()
    assert [line.split(":")[1].split("%")[0].strip() for line in lines] == [
        str(p) for p in range(0, 101, 10)
    ]


def test_ranges() -> None:
    assert cli._ranges([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == "1-10"
    assert cli._ranges([1, 3, 4, 5, 9]) == "1,3-5,9"
    assert cli._ranges([15]) == "15"


@pytest.mark.parametrize("which", ["dat", "notes", "directory"])
def test_unsafe_cache_paths_are_refused(
    synthetic_file: SyntheticFile,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    which: str,
) -> None:
    dat = synthetic_file.path
    dat_sha = hashlib.sha256(dat.read_bytes()).hexdigest()
    if which == "dat":
        target, message = dat, "raw data file itself"
    elif which == "notes":
        target, message = tmp_path / "notes.txt", "not a uvcorr UV cache"
        target.write_text("important notes\n")
    else:
        target, message = tmp_path / "somedir", "is a directory"
        target.mkdir()
    code, out, err = run(["build-cache", str(dat), "--cache", str(target)], capsys)
    assert code == cli.EXIT_ERROR
    assert "error:" in err and message in err
    assert "parsing" not in err  # refused before parsing
    assert out == ""
    assert hashlib.sha256(dat.read_bytes()).hexdigest() == dat_sha
    if which == "notes":
        assert target.read_text() == "important notes\n"


def test_busy_cache(
    synthetic_file: SyntheticFile,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    hold_h5_open: HoldOpen,
) -> None:
    dat = synthetic_file.path
    assert run(["build-cache", str(dat)], capsys)[0] == cli.EXIT_OK
    monkeypatch.setattr(cache, "LOCK_RETRY_SECONDS", 0.2)
    with hold_h5_open(default_cache_path(dat)):
        for argv in (["build-cache", str(dat)], ["build-cache", str(dat), "--force"]):
            code, out, err = run(argv, capsys)
            assert code == cli.EXIT_ERROR
            assert "in use by another process" in err
            assert out == ""
    assert UVCache(default_cache_path(dat)).has_results()  # not rebuilt


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_read_only_cache_directory(
    synthetic_file: SyntheticFile, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    readonly = tmp_path / "ro"
    readonly.mkdir()
    readonly.chmod(0o555)
    try:
        code, _, err = run(
            ["build-cache", str(synthetic_file.path), "--cache", str(readonly / "c.uv.h5")],
            capsys,
        )
    finally:
        readonly.chmod(0o755)
    assert code == cli.EXIT_ERROR
    assert "No write permission" in err and "--cache PATH" in err


def test_rebuild_warns_once_about_discarded_results(
    synthetic_file: SyntheticFile,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    dat = synthetic_file.path
    assert run(["build-cache", str(dat)], capsys)[0] == cli.EXIT_OK
    with h5py.File(default_cache_path(dat), "a") as h5f:
        h5f.create_group("results/current")
    with caplog.at_level(logging.WARNING):
        code, _, err = run(["build-cache", str(dat), "--force"], capsys)
    assert code == cli.EXIT_OK
    assert err.count("rebuilding the cache discards them") == 1
    # The library's own log record is filtered out while the CLI reports it.
    assert not [r for r in caplog.records if "/results" in r.getMessage()]
    assert not UVCache(default_cache_path(dat)).has_results()


def test_file_without_frames_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dat = tmp_path / "notes.dat"
    dat.write_bytes(b"not a raw acquisition\n" * 100)
    code, out, err = run(["build-cache", str(dat)], capsys)
    assert code == cli.EXIT_OK
    assert err.count("no valid frames") == 1 and "raw .dat" in err
    assert "events kept:   0 on 0 boards (none)" in out
    code, _, err = run(["build-cache", str(dat)], capsys)  # also when reused
    assert code == cli.EXIT_OK and "no valid frames" in err


def test_summary_reports_uv_zero_events(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dat = write_dat(tmp_path / "z.dat", [Frame(1, 15, 0, 1, (Hit(5, 1, 0, 0), Hit(6, 1, 2, 3)))])
    code, out, _ = run(["build-cache", str(dat)], capsys)
    assert code == cli.EXIT_OK
    assert "1 of them with U = V = 0" in out
