"""Tests for the ``uvcorr build-cache`` and ``uvcorr process`` subcommands."""

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

from tests.conftest import RING_BOARDS, RING_CHANNELS, RingFiles, SyntheticFile
from tests.synthetic_dat import Frame, Hit, write_dat
from uvcorr import cache, cli
from uvcorr.analysis import ChannelKey, analyze_channel
from uvcorr.cache import CacheBuildCancelled, UVCache, default_cache_path
from uvcorr.io.summary_csv import read_summary_csv
from uvcorr.io.tec import read_tec
from uvcorr.options import FLAG_EXTREME_AXIS_RATIO, FLAGS, FitOptions

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


# ----------------------------------------------------------------------
# uvcorr process
# ----------------------------------------------------------------------


def run_process(
    ring: RingFiles, out: Path, capsys: pytest.CaptureFixture[str], *extra: str
) -> tuple[int, str, str]:
    return run(["process", str(ring.dat), "--output-dir", str(out), *extra], capsys)


def test_process_end_to_end(
    ring_files: RingFiles, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out" / "nested"
    code, stdout, stderr = run_process(ring_files, out, capsys, "--workers", "2")
    assert code == cli.EXIT_OK, stderr
    assert "Fitting 3 boards with 2 worker(s), options: defaults" in stderr
    assert "fitting: 100%" in stderr
    assert "Building UV cache" not in stderr  # the fixture's cache is valid

    n_channels = len(RING_BOARDS) * len(RING_CHANNELS)
    assert f"({ring_files.cache.name}" not in stdout
    assert f"UV cache: {ring_files.cache} (reused)" in stdout
    assert f"Analysis: {n_channels} channels" in stdout
    assert "status:     ok 12, too_few_events 3, fit_failed 3" in stdout
    assert "overrides:  0 applied" in stdout
    assert "Time: cache" in stdout and "analysis" in stdout

    stored = UVCache(ring_files.cache).load_results()
    assert stored is not None and stored.options == FitOptions()
    assert len(stored.results) == n_channels
    for flag in FLAGS:
        n_flag = sum(1 for r in stored.results if flag in r.flags)
        assert (f"{flag} {n_flag}" in stdout) == (n_flag > 0)
    assert sum(1 for r in stored.results if FLAG_EXTREME_AXIS_RATIO in r.flags) >= 3

    tec_path = out / "rings.tec"
    csv_path = out / "radial_summary.csv"
    assert sorted(p.name for p in out.iterdir()) == ["radial_summary.csv", "rings.tec"]
    entries = read_tec(tec_path)
    ok = [r for r in stored.results if r.ok]
    assert list(entries) == [r.key for r in ok]
    assert f"({len(ok)} channel blocks" in stdout
    rows = read_summary_csv(csv_path)
    assert [r.key for r in rows] == [r.key for r in stored.results]
    assert [r.status for r in rows] == [r.status for r in stored.results]
    for row, result in zip(rows, stored.results):
        assert row.flags == result.flags and row.n_events == result.n_events
        if result.ok:
            assert result.centerU is not None and result.post_sigma is not None
            assert row.centerU == pytest.approx(result.centerU, rel=1e-8)
            assert entries[row.key].radius_std == pytest.approx(result.post_sigma, rel=1e-5)


def test_process_reuses_the_cache_and_builds_when_needed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], _ring_files_master: RingFiles
) -> None:
    dat = tmp_path / "fresh.dat"
    shutil.copy2(_ring_files_master.dat, dat)
    custom = tmp_path / "caches" / "custom.uv.h5"
    ring = RingFiles(dat, custom)
    args = ("--workers", "1", "--cache", str(custom))
    code, stdout, stderr = run_process(ring, tmp_path / "out", capsys, *args)
    assert code == cli.EXIT_OK, stderr
    assert "Building UV cache" in stderr and f"UV cache: {custom} (built)" in stdout
    assert not default_cache_path(dat).exists()
    code, stdout, stderr = run_process(ring, tmp_path / "out", capsys, *args)
    assert code == cli.EXIT_OK
    assert "Building UV cache" not in stderr and "(reused)" in stdout
    assert "Fitting 3 boards with 1 worker(s)" in stderr
    assert UVCache(custom).load_results() is not None


def test_process_options_reach_fit_options(
    ring_files: RingFiles, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, stdout, stderr = run_process(
        ring_files,
        tmp_path / "out",
        capsys,
        "--workers",
        "1",
        "--no-robust",
        "--clip-k",
        "3.5",
        "--max-iter",
        "2",
        "--geometric",
        "--min-events",
        "30",
    )
    assert code == cli.EXIT_OK, stderr
    expected = FitOptions(min_events=30, robust=False, clip_k=3.5, max_iter=2, geometric=True)
    uv = UVCache(ring_files.cache)
    stored = uv.load_results()
    assert stored is not None and stored.options == expected
    assert "min_events=30, robust=False, clip_k=3.5, max_iter=2, geometric=True" in stdout
    # min_events=30 lets the 40-event channels be fitted
    assert all(r.status != "too_few_events" for r in stored.results)
    for result in stored.results[:6]:
        assert result == analyze_channel(result.key, *uv.channel_data(*result.key), expected)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--min-events", "5"], "must be >= 6"),
        (["--min-events", "0"], "must be >= 1"),
        (["--clip-k", "0"], "must be a finite number > 0"),
        (["--workers", "0"], "must be >= 1"),
        (["--max-iter", "x"], "expected an integer"),
    ],
)
def test_process_rejects_bad_options(
    ring_files: RingFiles,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    message: str,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["process", str(ring_files.dat), "--output-dir", str(tmp_path), *args])
    assert excinfo.value.code == cli.EXIT_USAGE
    assert message in capsys.readouterr().err


def test_process_missing_input_and_bad_output_dir(
    ring_files: RingFiles, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(["process", str(tmp_path / "nope.dat"), "--output-dir", "o"], capsys)
    assert code == cli.EXIT_USAGE and "not found" in err
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x")
    code, _, err = run_process(ring_files, not_a_dir, capsys)
    assert code == cli.EXIT_USAGE and "not a directory" in err
    assert UVCache(ring_files.cache).load_results() is None


def test_process_applies_and_discards_overrides(
    ring_files: RingFiles, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    assert run_process(ring_files, out, capsys, "--workers", "1")[0] == cli.EXIT_OK
    uv = UVCache(ring_files.cache)
    key = ChannelKey(1, 15, 0, 12)
    u, v = uv.channel_data(*key)
    refit_options = FitOptions(robust=False)
    uv.save_override(analyze_channel(key, u, v, refit_options), refit_options)

    code, stdout, _ = run_process(ring_files, out, capsys, "--workers", "1")
    assert code == cli.EXIT_OK and "overrides:  1 applied" in stdout
    rows = {r.key: r for r in read_summary_csv(out / "radial_summary.csv")}
    stored = uv.load_results()
    assert stored is not None and list(stored.overrides) == [key]
    batch_row = next(r for r in stored.results if r.key == key)
    override_row = stored.overrides[key].result
    assert override_row.centerU != batch_row.centerU
    assert rows[key].options_source == "override"
    assert rows[key].centerU == pytest.approx(override_row.centerU, rel=1e-8)
    assert rows[ChannelKey(1, 15, 0, 5)].options_source == "batch"

    code, stdout, _ = run_process(ring_files, out, capsys, "--workers", "1", "--discard-overrides")
    assert code == cli.EXIT_OK and "discarded (--discard-overrides)" in stdout
    rows = {r.key: r for r in read_summary_csv(out / "radial_summary.csv")}
    stored = uv.load_results()
    assert stored is not None and stored.overrides == {}
    assert rows[key].options_source == "batch"
    assert rows[key].centerU == pytest.approx(batch_row.centerU, rel=1e-8)


def test_process_ctrl_c_stops_the_analysis(
    ring_files: RingFiles,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_call = cli._ProgressPrinter.__call__

    def progress_with_sigint(self: Any, fraction: float) -> None:
        if self._label == "fitting":
            signal.raise_signal(signal.SIGINT)
        real_call(self, fraction)

    monkeypatch.setattr(cli._ProgressPrinter, "__call__", progress_with_sigint)
    previous = signal.getsignal(signal.SIGINT)
    out = tmp_path / "out"
    code, stdout, stderr = run_process(ring_files, out, capsys, "--workers", "2")
    assert code == cli.EXIT_INTERRUPTED
    assert "stopping the analysis" in stderr and "analysis cancelled" in stderr
    assert stdout == ""
    assert list(out.iterdir()) == []  # created (and write-tested) up front, nothing written
    assert UVCache(ring_files.cache).load_results() is None
    assert signal.getsignal(signal.SIGINT) is previous


def test_process_analysis_error(
    ring_files: RingFiles, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with h5py.File(ring_files.cache, "r+") as h5f:
        del h5f["events/node_4/board_29/v"]
    code, stdout, stderr = run_process(ring_files, tmp_path / "out", capsys, "--workers", "2")
    assert code == cli.EXIT_ERROR
    assert "error: Analysis of node 4 board 29 failed" in stderr
    assert stdout == ""
    assert list((tmp_path / "out").iterdir()) == []
    assert UVCache(ring_files.cache).load_results() is None


def test_process_read_only_cache_still_writes_the_outputs(
    ring_files: RingFiles, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    uv = UVCache(ring_files.cache)
    # A stored override is applied even though the cache cannot be written
    uv.save_results([], FitOptions())
    key = ChannelKey(1, 15, 0, 12)
    refit_options = FitOptions(robust=False)
    uv.save_override(analyze_channel(key, *uv.channel_data(*key), refit_options), refit_options)
    before = uv.load_results()
    ring_files.cache.chmod(0o444)
    try:
        code, stdout, stderr = run_process(ring_files, tmp_path / "out", capsys, "--workers", "1")
    finally:
        ring_files.cache.chmod(0o644)
    assert code == cli.EXIT_OK, stderr
    assert "is read-only: the results will not be stored" in stderr
    assert "results NOT stored in the cache" in stdout and "overrides:  1 applied" in stdout
    rows = {r.key: r for r in read_summary_csv(tmp_path / "out" / "radial_summary.csv")}
    assert len(rows) == len(RING_BOARDS) * len(RING_CHANNELS)
    assert rows[key].options_source == "override"
    assert uv.load_results() == before  # untouched


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_process_unwritable_output_dir_fails_before_the_analysis(
    ring_files: RingFiles, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "ro"
    out.mkdir()
    out.chmod(0o555)
    try:
        code, stdout, stderr = run_process(ring_files, out, capsys, "--workers", "1")
    finally:
        out.chmod(0o755)
    assert code == cli.EXIT_ERROR
    assert "cannot write to the output directory" in stderr
    assert "fitting" not in stderr and "Fitting" not in stderr and stdout == ""
    assert UVCache(ring_files.cache).load_results() is None


def test_process_output_file_is_a_directory(
    ring_files: RingFiles, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    (out / "rings.tec").mkdir(parents=True)
    code, _, stderr = run_process(ring_files, out, capsys, "--workers", "1")
    assert code == cli.EXIT_ERROR and "rings.tec is a directory" in stderr
    assert "Fitting" not in stderr
    assert UVCache(ring_files.cache).load_results() is None


def test_process_store_failure_keeps_the_outputs(
    ring_files: RingFiles,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def busy(*_args: Any, **_kwargs: Any) -> None:
        raise cache.CacheBusyError("in use by another process")

    monkeypatch.setattr(UVCache, "save_results", busy)
    out = tmp_path / "out"
    code, stdout, stderr = run_process(ring_files, out, capsys, "--workers", "1")
    assert code == cli.EXIT_ERROR
    assert "output files were written, but storing the results in the cache failed" in stderr
    assert "results NOT stored" in stdout
    assert sorted(p.name for p in out.iterdir()) == ["radial_summary.csv", "rings.tec"]
    assert len(read_tec(out / "rings.tec")) == 12


def test_process_unreadable_stored_results(
    ring_files: RingFiles, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    UVCache(ring_files.cache).save_results([], FitOptions())
    with h5py.File(ring_files.cache, "r+") as h5f:
        h5f["results/current"].attrs["options_json"] = "not json"
    code, _, stderr = run_process(ring_files, tmp_path / "out", capsys, "--workers", "1")
    assert code == cli.EXIT_ERROR and "unreadable" in stderr and "--discard-overrides" in stderr
    assert "Fitting" not in stderr  # detected before the analysis
    code, _, stderr = run_process(
        ring_files, tmp_path / "out", capsys, "--workers", "1", "--discard-overrides"
    )
    assert code == cli.EXIT_OK, stderr
    stored = UVCache(ring_files.cache).load_results()
    assert stored is not None and len(stored.results) == 18


def test_process_busy_cache(
    ring_files: RingFiles,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    hold_h5_open: HoldOpen,
) -> None:
    monkeypatch.setattr(cache, "LOCK_RETRY_SECONDS", 0.2)
    with hold_h5_open(ring_files.cache):
        code, _, err = run_process(ring_files, tmp_path / "out", capsys, "--workers", "1")
    assert code == cli.EXIT_ERROR and "in use by another process" in err


def test_process_help_defaults_come_from_fit_options(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def help_text() -> str:
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["process", "--help"])
        assert excinfo.value.code == 0
        return " ".join(capsys.readouterr().out.split())

    text = help_text()
    defaults = FitOptions()
    assert f"(default: {defaults.min_events})" in text
    assert f"(default: {defaults.clip_k:g})" in text
    assert f"(default: {defaults.max_iter})" in text

    def other_defaults(**given: Any) -> FitOptions:
        return FitOptions(**{"min_events": 77, "clip_k": 2.5, "max_iter": 9, **given})

    monkeypatch.setattr(cli, "FitOptions", other_defaults)
    text = help_text()
    assert "(default: 77)" in text and "(default: 2.5)" in text and "(default: 9)" in text
