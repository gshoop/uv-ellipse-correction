"""Tests for uvcorr.cache: the synthetic writer, the build, validity and read access."""

from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import logging
import os
import pickle
import shutil
import subprocess
import threading
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, NamedTuple

import h5py
import numpy as np
import pytest
from adc2kev.parser import PacketParser

from tests.conftest import SYNTHETIC_BOARDS, SYNTHETIC_NODES, SyntheticFile
from tests.synthetic_dat import Frame, Hit, crc8, encode_frame, expected_events, write_dat
from uvcorr import __version__, cache
from uvcorr.cache import (
    EVENT_FIELDS,
    UV_CACHE_VERSION,
    BuildSettings,
    CacheBuildCancelled,
    CacheBuildError,
    CacheBusyError,
    InsufficientDiskSpaceError,
    UVCache,
    compute_source_hash,
    default_cache_path,
    estimate_cache_bytes,
    open_or_build,
)
from uvcorr.channels import active_channel_mask

# Small thresholds so a ~12k-event file goes through many batches and every
# flushing path (global budget, per-board limit, partial chunks).
BUDGET_SETTINGS = BuildSettings(
    batch_events=500, board_flush_events=10**9, max_buffered_events=1_500, chunk_events=256
)
BOARD_FLUSH_SETTINGS = BuildSettings(
    batch_events=700, board_flush_events=200, max_buffered_events=10**9, chunk_events=100
)
UNCOMPRESSED_SETTINGS = BuildSettings(batch_events=1_000, compression=None, shuffle=False)
GZIP_SETTINGS = BuildSettings(batch_events=1_000, compression="gzip", compression_opts=4)

METADATA_ATTRS = {
    # plan 6.1
    "uv_cache_version",
    "uvcorr_version",
    "source_path",
    "source_size",
    "source_mtime",
    "source_hash",
    "parser_frames",
    "parser_events",
    "parser_dropped",
    "n_events_kept",
    "n_events_inactive",
    "build_seconds",
    "created_at",
    # extras
    "n_events_node0",
    "n_events_uv_zero",
    "n_boards",
    "parser_invalid_headers",
    "parser_invalid_lengths",
    "parser_bytes_read",
    "chunk_events",
    "compression",
    "shuffle",
}


class ParsedColumns(NamedTuple):
    node: np.ndarray
    board: np.ndarray
    rena: np.ndarray
    channel: np.ndarray
    pha: np.ndarray
    u: np.ndarray
    v: np.ndarray


def parse_all(path: Path, batch_events: int = 1000) -> tuple[ParsedColumns, Any]:
    parser = PacketParser(path)
    batches = list(parser.iter_event_arrays(batch_events=batch_events))
    columns = ParsedColumns(
        *(
            np.concatenate([getattr(b, name) for b in batches])
            for name in (
                "node_num",
                "board_num",
                "rena_num",
                "channel_num",
                "pha",
                "u",
                "v",
            )
        )
    )
    return columns, parser.get_statistics()


def assert_cache_matches(uv: UVCache, synthetic: SyntheticFile) -> None:
    """Every board of the cache holds exactly the expected active events, in file order."""
    active = synthetic.active_events
    expected_boards = sorted({(int(n), int(b)) for n, b in zip(active["node"], active["board"])})
    assert uv.boards() == expected_boards
    counts = uv.board_event_counts()
    for node, board in expected_boards:
        sel = active[(active["node"] == node) & (active["board"] == board)]
        data = uv.load_board(node, board)
        assert (data.node, data.board) == (node, board)
        assert data.n_events == counts[(node, board)] == len(sel)
        for name in EVENT_FIELDS:
            stored = getattr(data, name)
            assert stored.dtype == cache.EVENT_DTYPES[name]
            np.testing.assert_array_equal(stored, sel[name].astype(stored.dtype))
    assert sum(counts.values()) == len(active)


# ----------------------------------------------------------------------
# Synthetic writer
# ----------------------------------------------------------------------


class TestSyntheticDat:
    def test_file_parses_to_exactly_the_intended_events(
        self, synthetic_file: SyntheticFile
    ) -> None:
        columns, stats = parse_all(synthetic_file.path)
        expected = synthetic_file.events
        assert stats.total_frames == len(synthetic_file.frames)
        assert stats.dropped_frames == 0
        assert stats.invalid_headers == stats.invalid_lengths == 0
        assert stats.total_events == len(expected)
        for name in ParsedColumns._fields:
            np.testing.assert_array_equal(getattr(columns, name), expected[name])

    def test_covers_node0_and_inactive_channels(self, synthetic_file: SyntheticFile) -> None:
        events = synthetic_file.events
        assert set(events["node"].tolist()) == set(SYNTHETIC_NODES)
        assert set(events["board"].tolist()) == set(SYNTHETIC_BOARDS)
        inactive = events[~active_channel_mask(events["rena"], events["channel"])]
        pairs = {(int(r), int(c)) for r, c in zip(inactive["rena"], inactive["channel"])}
        # R0 ch 0-3, R1 ch 0-6 and channels above 28 on both RENAs
        assert {(0, c) for c in range(4)} <= pairs
        assert {(1, c) for c in range(7)} <= pairs
        assert {(0, 35), (1, 29)} <= pairs

    def test_frame_length_and_crc_nibbles(self) -> None:
        frame = Frame(1, 16, 1, 12345, (Hit(9, 4095, 0, 2048), Hit(3, 1, 2, 3)))
        raw = encode_frame(frame)
        assert len(raw) == 19 + 6 * 2
        assert raw[0] == 0xC8 and raw[-1] == 0xFF
        assert raw[3] == (16 << 1) | 1
        assert (raw[-3] << 4) | raw[-2] == crc8(raw[:-3])

    def test_corrupted_crc_is_dropped(self, tmp_path: Path) -> None:
        good = Frame(1, 15, 0, 10, (Hit(5, 100, 200, 300),))
        raw = bytearray(encode_frame(good))
        raw[-2] ^= 0x01  # flip a CRC bit
        path = tmp_path / "bad.dat"
        path.write_bytes(bytes(raw) + encode_frame(good))
        columns, stats = parse_all(path)
        assert stats.dropped_frames == 1
        assert len(columns.u) == 1


# ----------------------------------------------------------------------
# Paths and hashing
# ----------------------------------------------------------------------


def test_default_cache_path_is_sibling() -> None:
    assert default_cache_path("/data/run_120628.dat") == Path("/data/run_120628.dat.uv.h5")
    assert default_cache_path(Path("rel/x.dat")) == Path("rel/x.dat.uv.h5")


def test_compute_source_hash_is_sha256_of_first_mib(tmp_path: Path) -> None:
    head = bytes(range(256)) * 4096  # exactly 1 MiB
    path = tmp_path / "f.bin"
    path.write_bytes(head + b"tail that is not hashed")
    assert compute_source_hash(path) == hashlib.sha256(head).hexdigest()
    small = tmp_path / "small.bin"
    small.write_bytes(b"abc")
    assert compute_source_hash(small) == hashlib.sha256(b"abc").hexdigest()


def test_compute_source_hash_matches_adc2kev(tmp_path: Path) -> None:
    # Reimplementation check against adc2kev's private helper (test-only use).
    from adc2kev.cache.hdf5_cache import CalibrationCache

    path = tmp_path / "f.bin"
    path.write_bytes(np.random.default_rng(0).bytes(3 * 1024 * 1024 + 17))
    assert compute_source_hash(path) == CalibrationCache._compute_file_hash(path)


def test_estimate_cache_bytes() -> None:
    assert cache.BYTES_PER_EVENT == 8
    assert estimate_cache_bytes(0) == 0
    assert estimate_cache_bytes(3_478_589_791) == 1_739_294_896  # ~1.7 GB (plan 6.1)


# ----------------------------------------------------------------------
# Build and round trip
# ----------------------------------------------------------------------


class TestBuild:
    @pytest.mark.parametrize(
        "settings",
        [None, BUDGET_SETTINGS, BOARD_FLUSH_SETTINGS, UNCOMPRESSED_SETTINGS, GZIP_SETTINGS],
        ids=["default", "budget", "board-flush", "uncompressed", "gzip"],
    )
    def test_round_trip_is_exact(
        self, synthetic_file: SyntheticFile, settings: BuildSettings | None
    ) -> None:
        uv = UVCache(default_cache_path(synthetic_file.path))
        stats = uv.build_from_dat(synthetic_file.path, settings=settings)
        assert uv.last_build == stats
        assert_cache_matches(uv, synthetic_file)

        events = synthetic_file.events
        active = active_channel_mask(events["rena"], events["channel"])
        node0 = events["node"] == 0
        assert stats.n_events_parsed == len(events)
        assert stats.n_events_kept == len(synthetic_file.active_events)
        assert stats.n_events_node0 == int(node0.sum()) > 0
        assert stats.n_events_inactive == int((~active & ~node0).sum()) > 0
        assert stats.n_boards == len(uv.boards())
        assert stats.parser_frames == len(synthetic_file.frames)
        assert stats.parser_dropped == 0
        assert stats.cache_bytes == uv.path.stat().st_size
        assert uv.tmp_files() == []

    def test_metadata(self, synthetic_file: SyntheticFile) -> None:
        uv = UVCache(default_cache_path(synthetic_file.path))
        stats = uv.build_from_dat(synthetic_file.path)
        meta = uv.metadata()
        assert set(meta) >= METADATA_ATTRS
        st = synthetic_file.path.stat()
        assert meta["uv_cache_version"] == UV_CACHE_VERSION
        assert meta["uvcorr_version"] == __version__
        assert meta["source_path"] == str(synthetic_file.path.absolute())
        assert meta["source_size"] == st.st_size
        assert meta["source_mtime"] == st.st_mtime
        assert meta["source_hash"] == compute_source_hash(synthetic_file.path)
        assert meta["parser_frames"] == len(synthetic_file.frames)
        assert meta["parser_events"] == len(synthetic_file.events)
        assert meta["parser_dropped"] == 0
        assert meta["n_events_kept"] == stats.n_events_kept
        assert meta["n_events_inactive"] == stats.n_events_inactive
        assert meta["n_events_node0"] == stats.n_events_node0
        assert (
            meta["n_events_kept"] + meta["n_events_inactive"] + meta["n_events_node0"]
            == meta["parser_events"]
        )
        assert meta["n_boards"] == stats.n_boards
        assert meta["build_seconds"] == pytest.approx(stats.build_seconds)
        assert isinstance(meta["created_at"], str) and "T" in meta["created_at"]
        assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())

    def test_layout_only_metadata_and_events(self, synthetic_file: SyntheticFile) -> None:
        uv = UVCache(default_cache_path(synthetic_file.path))
        uv.build_from_dat(synthetic_file.path)
        with h5py.File(uv.path, "r") as h5f:
            assert set(h5f.keys()) == {"metadata", "events"}
            group = h5f["events/node_1/board_16"]
            assert set(group.keys()) == set(EVENT_FIELDS)
            assert group.attrs["n_events"] == group["u"].shape[0]
            assert group["u"].chunks is not None and group["u"].maxshape == (None,)
            assert group["u"].compression == "lzf"

    def test_global_budget_bounds_buffered_events(self, synthetic_file: SyntheticFile) -> None:
        uv = UVCache(default_cache_path(synthetic_file.path))
        stats = uv.build_from_dat(synthetic_file.path, settings=BUDGET_SETTINGS)
        s = BUDGET_SETTINGS
        # Several flush passes over all boards happened during the parse ...
        assert stats.n_flushes > 3 * stats.n_boards
        # ... and the buffers never held more than the budget plus one batch.
        assert stats.max_buffered_events <= s.max_buffered_events + s.batch_events
        assert stats.max_buffered_events > s.max_buffered_events
        assert_cache_matches(uv, synthetic_file)

    def test_per_board_limit_flushes(self, synthetic_file: SyntheticFile) -> None:
        uv = UVCache(default_cache_path(synthetic_file.path))
        stats = uv.build_from_dat(synthetic_file.path, settings=BOARD_FLUSH_SETTINGS)
        assert stats.n_flushes > 2 * stats.n_boards
        assert_cache_matches(uv, synthetic_file)

    def test_defaults_hold_everything_until_the_end_on_a_small_file(
        self, synthetic_file: SyntheticFile
    ) -> None:
        uv = UVCache(default_cache_path(synthetic_file.path))
        stats = uv.build_from_dat(synthetic_file.path)
        assert stats.n_flushes == stats.n_boards  # one final flush per board
        assert stats.max_buffered_events == stats.n_events_kept

    def test_custom_cache_location_is_created(
        self, synthetic_file: SyntheticFile, tmp_path: Path
    ) -> None:
        uv = UVCache(tmp_path / "sub" / "dir" / "c.h5")
        uv.build_from_dat(synthetic_file.path)
        assert uv.exists() and uv.is_valid_for(synthetic_file.path)
        assert not default_cache_path(synthetic_file.path).exists()

    def test_empty_file(self, tmp_path: Path) -> None:
        dat = tmp_path / "empty.dat"
        dat.write_bytes(b"")
        uv = UVCache(default_cache_path(dat))
        progress: list[float] = []
        stats = uv.build_from_dat(dat, progress_cb=progress.append)
        assert stats.n_events_parsed == stats.n_boards == 0
        assert uv.boards() == []
        assert uv.is_valid_for(dat)
        assert progress == [1.0]

    def test_node0_and_inactive_only(self, tmp_path: Path) -> None:
        frames = [
            Frame(0, 16, 0, 1, (Hit(10, 1, 2, 3),)),  # node 0, active channel
            Frame(3, 16, 0, 2, (Hit(2, 1, 2, 3), Hit(30, 4, 5, 6))),  # inactive
            Frame(3, 17, 1, 3, (Hit(6, 1, 2, 3),)),  # inactive
        ]
        dat = write_dat(tmp_path / "x.dat", frames)
        stats = UVCache(default_cache_path(dat)).build_from_dat(dat)
        assert (stats.n_events_kept, stats.n_events_inactive, stats.n_events_node0) == (0, 3, 1)
        assert stats.n_boards == 0

    def test_file_order_within_board_across_batches(self, tmp_path: Path) -> None:
        # One board, interleaved with others, split over many tiny batches.
        frames = [
            Frame(1 + (i % 3), 20, i % 2, i, (Hit(4 + i % 20, i % 4096, i, 4095 - i),))
            for i in range(900)
        ]
        dat = write_dat(tmp_path / "order.dat", frames)
        uv = UVCache(default_cache_path(dat))
        uv.build_from_dat(dat, settings=BuildSettings(batch_events=7, max_buffered_events=50))
        for node in (1, 2, 3):
            data = uv.load_board(node, 20)
            expected_u = [
                i for i in range(900) if 1 + i % 3 == node and (i % 2 == 0 or i % 20 >= 3)
            ]
            assert data.u.tolist() == expected_u

    def test_foreign_tmp_files_are_left_alone(self, synthetic_file: SyntheticFile) -> None:
        # A tmp file of another (e.g. killed) build is never touched: each build
        # removes only its own.
        uv = UVCache(default_cache_path(synthetic_file.path))
        leftover = uv.path.with_name(f"{uv.path.stem}.0123abcd{uv.path.suffix}.tmp")
        leftover.write_bytes(b"from a killed build")
        assert uv.tmp_files() == [leftover]
        uv.build_from_dat(synthetic_file.path)
        assert uv.tmp_files() == [leftover]
        assert leftover.read_bytes() == b"from a killed build"
        assert uv.is_valid_for(synthetic_file.path)

    def test_tmp_file_name_and_permissions(self, synthetic_file: SyntheticFile) -> None:
        uv = UVCache(default_cache_path(synthetic_file.path))
        seen: list[Path] = []
        uv.build_from_dat(synthetic_file.path, progress_cb=lambda _f: seen.extend(uv.tmp_files()))
        assert seen
        name = seen[0].name
        assert name.startswith("synthetic.dat.uv.") and name.endswith(".h5.tmp")
        assert fnmatch.fnmatch(name, "*.h5.tmp")  # covered by .gitignore
        # Default permissions (not mkstemp's owner-only 0o600).
        umask = os.umask(0)
        os.umask(umask)
        assert uv.path.stat().st_mode & 0o777 == 0o666 & ~umask

    def test_missing_source(self, tmp_path: Path) -> None:
        uv = UVCache(tmp_path / "c.uv.h5")
        with pytest.raises(FileNotFoundError):
            uv.build_from_dat(tmp_path / "missing.dat")
        with pytest.raises(FileNotFoundError):
            open_or_build(tmp_path / "missing.dat")
        assert not uv.exists()

    def test_settings_validation(self) -> None:
        with pytest.raises(ValueError, match="chunk_events"):
            BuildSettings(chunk_events=0)
        with pytest.raises(ValueError, match="compression"):
            BuildSettings(compression="zstd")


# ----------------------------------------------------------------------
# Progress, cancellation, failures, disk space
# ----------------------------------------------------------------------


class TestProgressAndCancel:
    def test_progress_is_monotonic_and_ends_at_one(
        self, synthetic_file: SyntheticFile, tmp_path: Path
    ) -> None:
        # The parser reads 1 MB chunks: repeat the synthetic frames to ~3 MB so
        # batches report several distinct byte positions.
        raw = synthetic_file.path.read_bytes()
        big = tmp_path / "big.dat"
        big.write_bytes(raw * (3_000_000 // len(raw) + 1))
        values: list[float] = []
        uv = UVCache(default_cache_path(big))
        stats = uv.build_from_dat(
            big, progress_cb=values.append, settings=BuildSettings(batch_events=5_000)
        )
        assert stats.n_events_parsed > 200_000
        assert len(values) > 10
        assert all(0.0 <= v <= 1.0 for v in values)
        assert all(b >= a for a, b in zip(values, values[1:]))
        assert values[-1] == 1.0
        assert values[0] < 0.5
        # The parse ends at PARSE_PROGRESS_SHARE; 1.0 comes only after the rename.
        assert max(values[:-1]) == pytest.approx(cache.PARSE_PROGRESS_SHARE)
        assert len(set(values)) >= 3

    def test_stop_flag_callable_cancels_cleanly(self, synthetic_file: SyntheticFile) -> None:
        calls = Counter[str]()

        def stop() -> bool:
            calls["n"] += 1
            return calls["n"] > 3

        uv = UVCache(default_cache_path(synthetic_file.path))
        with pytest.raises(CacheBuildCancelled):
            uv.build_from_dat(synthetic_file.path, stop_flag=stop, settings=BUDGET_SETTINGS)
        assert calls["n"] == 4
        assert not uv.exists()
        assert uv.tmp_files() == []
        assert uv.last_build is None

    def test_stop_flag_event_cancels(self, synthetic_file: SyntheticFile) -> None:
        stop = threading.Event()
        stop.set()
        uv = UVCache(default_cache_path(synthetic_file.path))
        with pytest.raises(CacheBuildCancelled):
            open_or_build(synthetic_file.path, stop_flag=stop)
        assert not uv.exists() and uv.tmp_files() == []

    def test_stop_after_last_batch_still_cancels(self, synthetic_file: SyntheticFile) -> None:
        # One batch only: the flag is set during it and caught after the loop.
        stop = threading.Event()
        uv = UVCache(default_cache_path(synthetic_file.path))
        with pytest.raises(CacheBuildCancelled):
            uv.build_from_dat(
                synthetic_file.path, progress_cb=lambda _f: stop.set(), stop_flag=stop
            )
        assert not uv.exists() and uv.tmp_files() == []

    def test_cancel_keeps_previous_cache(self, synthetic_file: SyntheticFile) -> None:
        uv = UVCache(default_cache_path(synthetic_file.path))
        uv.build_from_dat(synthetic_file.path)
        before = uv.path.read_bytes()
        with pytest.raises(CacheBuildCancelled):
            uv.build_from_dat(synthetic_file.path, stop_flag=lambda: True)
        assert uv.path.read_bytes() == before
        assert uv.tmp_files() == []
        assert uv.is_valid_for(synthetic_file.path)

    def test_bad_stop_flag_type(self, synthetic_file: SyntheticFile) -> None:
        uv = UVCache(default_cache_path(synthetic_file.path))
        with pytest.raises(TypeError):
            uv.build_from_dat(synthetic_file.path, stop_flag=True)  # type: ignore[arg-type]

    def test_failure_leaves_no_tmp(
        self, synthetic_file: SyntheticFile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_split = cache._split_batch
        seen = Counter[str]()

        def failing_split(batch: Any, counts: Any) -> Any:
            seen["n"] += 1
            if seen["n"] == 3:
                raise RuntimeError("simulated parser failure")
            return real_split(batch, counts)

        monkeypatch.setattr(cache, "_split_batch", failing_split)
        uv = UVCache(default_cache_path(synthetic_file.path))
        with pytest.raises(CacheBuildError, match="simulated parser failure") as excinfo:
            uv.build_from_dat(synthetic_file.path, settings=BUDGET_SETTINGS)
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert not uv.exists() and uv.tmp_files() == []

    def test_keyboard_interrupt_leaves_no_tmp(self, synthetic_file: SyntheticFile) -> None:
        def interrupt(_fraction: float) -> None:
            assert len(uv.tmp_files()) == 1  # the build writes to its tmp file
            raise KeyboardInterrupt

        uv = UVCache(default_cache_path(synthetic_file.path))
        with pytest.raises(KeyboardInterrupt):
            uv.build_from_dat(synthetic_file.path, progress_cb=interrupt)
        assert not uv.exists() and uv.tmp_files() == []

    def test_source_modified_during_build(self, synthetic_file: SyntheticFile) -> None:
        def touch(_fraction: float) -> None:
            os.utime(synthetic_file.path, ns=(1, 1_000_000_000))

        uv = UVCache(default_cache_path(synthetic_file.path))
        with pytest.raises(CacheBuildError, match="changed"):
            uv.build_from_dat(synthetic_file.path, progress_cb=touch)
        assert not uv.exists() and uv.tmp_files() == []

    def test_insufficient_disk_space(
        self, synthetic_file: SyntheticFile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Path] = []

        def tiny_disk(path: Any) -> Any:
            seen.append(Path(path))
            return shutil._ntuple_diskusage(10**9, 10**9 - 1000, 1000)  # type: ignore[attr-defined]

        monkeypatch.setattr(shutil, "disk_usage", tiny_disk)
        uv = UVCache(default_cache_path(synthetic_file.path))
        with pytest.raises(InsufficientDiskSpaceError, match="disk space") as excinfo:
            open_or_build(synthetic_file.path)
        assert isinstance(excinfo.value, CacheBuildError)
        assert seen == [uv.path.parent]
        assert not uv.exists() and uv.tmp_files() == []

    def test_disk_space_requirement_scales_with_dat_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dat_size = 3_478_589_791
        free = estimate_cache_bytes(dat_size)  # below estimate + headroom
        monkeypatch.setattr(
            shutil,
            "disk_usage",
            lambda _p: shutil._ntuple_diskusage(10**12, 0, free),  # type: ignore[attr-defined]
        )
        with pytest.raises(InsufficientDiskSpaceError, match=r"1\.8\d GB"):
            cache.check_disk_space(tmp_path / "x.uv.h5", dat_size)
        monkeypatch.setattr(
            shutil,
            "disk_usage",
            lambda _p: shutil._ntuple_diskusage(10**12, 0, 2 * free),  # type: ignore[attr-defined]
        )
        cache.check_disk_space(tmp_path / "x.uv.h5", dat_size)


# ----------------------------------------------------------------------
# Validity and open_or_build
# ----------------------------------------------------------------------


@pytest.fixture
def built(synthetic_file: SyntheticFile) -> UVCache:
    uv = UVCache(default_cache_path(synthetic_file.path))
    uv.build_from_dat(synthetic_file.path)
    return uv


class TestValidity:
    def test_valid_after_build(self, built: UVCache, synthetic_file: SyntheticFile) -> None:
        assert built.is_valid_for(synthetic_file.path)
        assert built.is_valid_for(str(synthetic_file.path))

    def test_missing_cache_or_source(self, synthetic_file: SyntheticFile, tmp_path: Path) -> None:
        assert not UVCache(tmp_path / "none.uv.h5").is_valid_for(synthetic_file.path)
        built = UVCache(default_cache_path(synthetic_file.path))
        built.build_from_dat(synthetic_file.path)
        assert not built.is_valid_for(tmp_path / "missing.dat")

    def test_size_change(self, built: UVCache, synthetic_file: SyntheticFile) -> None:
        st = synthetic_file.path.stat()
        with open(synthetic_file.path, "ab") as f:
            f.write(b"\x00")
        os.utime(synthetic_file.path, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert not built.is_valid_for(synthetic_file.path)

    def test_mtime_change(self, built: UVCache, synthetic_file: SyntheticFile) -> None:
        st = synthetic_file.path.stat()
        os.utime(synthetic_file.path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        assert not built.is_valid_for(synthetic_file.path)

    def test_content_change_same_size_and_mtime(
        self, built: UVCache, synthetic_file: SyntheticFile
    ) -> None:
        st = synthetic_file.path.stat()
        data = bytearray(synthetic_file.path.read_bytes())
        data[20] ^= 0x01
        synthetic_file.path.write_bytes(bytes(data))
        os.utime(synthetic_file.path, ns=(st.st_atime_ns, st.st_mtime_ns))
        new = synthetic_file.path.stat()
        assert (new.st_size, new.st_mtime) == (st.st_size, st.st_mtime)
        assert not built.is_valid_for(synthetic_file.path)

    def test_version_mismatch(self, built: UVCache, synthetic_file: SyntheticFile) -> None:
        with h5py.File(built.path, "a") as h5f:
            h5f["metadata"].attrs["uv_cache_version"] = "0.9.0"
        assert not built.is_valid_for(synthetic_file.path)

    def test_uvcorr_version_is_not_checked(
        self, built: UVCache, synthetic_file: SyntheticFile
    ) -> None:
        with h5py.File(built.path, "a") as h5f:
            h5f["metadata"].attrs["uvcorr_version"] = "0.0.1"
        assert built.is_valid_for(synthetic_file.path)

    def test_not_a_cache(self, synthetic_file: SyntheticFile) -> None:
        path = default_cache_path(synthetic_file.path)
        path.write_bytes(b"not an HDF5 file")
        assert not UVCache(path).is_valid_for(synthetic_file.path)
        with h5py.File(path, "w") as h5f:
            h5f.create_group("events")
        assert not UVCache(path).is_valid_for(synthetic_file.path)

    def test_open_or_build_builds_then_reuses(self, synthetic_file: SyntheticFile) -> None:
        progress: list[float] = []
        first = open_or_build(synthetic_file.path, progress_cb=progress.append)
        assert first.path == default_cache_path(synthetic_file.path)
        assert first.last_build is not None
        assert progress[-1] == 1.0
        mtime = first.path.stat().st_mtime_ns

        progress.clear()
        second = open_or_build(synthetic_file.path, progress_cb=progress.append)
        assert second.last_build is None
        assert progress == [1.0]
        assert second.path.stat().st_mtime_ns == mtime

        forced = open_or_build(synthetic_file.path, force=True)
        assert forced.last_build is not None
        assert_cache_matches(forced, synthetic_file)

    def test_open_or_build_rebuilds_when_stale(self, synthetic_file: SyntheticFile) -> None:
        first = open_or_build(synthetic_file.path)
        old_mtime = first.metadata()["source_mtime"]
        st = synthetic_file.path.stat()
        os.utime(synthetic_file.path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        assert not first.is_valid_for(synthetic_file.path)
        again = open_or_build(synthetic_file.path)
        assert again.last_build is not None
        assert again.is_valid_for(synthetic_file.path)
        assert again.metadata()["source_mtime"] == synthetic_file.path.stat().st_mtime != old_mtime
        assert_cache_matches(again, synthetic_file)

    def test_open_or_build_custom_path(self, synthetic_file: SyntheticFile, tmp_path: Path) -> None:
        uv = open_or_build(synthetic_file.path, cache_path=tmp_path / "other.h5")
        assert uv.path == tmp_path / "other.h5"
        assert uv.is_valid_for(synthetic_file.path)
        assert not default_cache_path(synthetic_file.path).exists()


# ----------------------------------------------------------------------
# Read access
# ----------------------------------------------------------------------


class TestReadAccess:
    def test_channels_and_channel_data(self, built: UVCache, synthetic_file: SyntheticFile) -> None:
        active = synthetic_file.active_events
        for node, board in built.boards():
            on_board = active[(active["node"] == node) & (active["board"] == board)]
            expected = Counter(zip(on_board["rena"].tolist(), on_board["channel"].tolist()))
            listed = built.channels(node, board)
            assert listed == sorted((r, c, n) for (r, c), n in expected.items())
            data = built.load_board(node, board)
            assert data.channels() == listed
            for rena, channel, n in listed:
                sel = on_board[(on_board["rena"] == rena) & (on_board["channel"] == channel)]
                u, v = built.channel_data(node, board, rena, channel)
                assert len(u) == len(v) == n
                np.testing.assert_array_equal(u, sel["u"])
                np.testing.assert_array_equal(v, sel["v"])
                bu, bv = data.channel_data(rena, channel)
                np.testing.assert_array_equal(bu, u)
                np.testing.assert_array_equal(bv, v)

    def test_channel_without_events_is_empty(self, built: UVCache) -> None:
        node, board = built.boards()[0]
        u, v = built.channel_data(node, board, 0, 2)  # inactive: never stored
        assert u.shape == v.shape == (0,)
        assert u.dtype == np.int16

    def test_unknown_board(self, built: UVCache) -> None:
        with pytest.raises(KeyError, match="node 9 board 3"):
            built.load_board(9, 3)
        with pytest.raises(KeyError):
            built.channel_data(9, 3, 0, 4)
        with pytest.raises(KeyError):
            built.channels(0, 16)  # node 0 is never stored

    def test_boards_sorted_and_counts(self, built: UVCache, synthetic_file: SyntheticFile) -> None:
        boards = built.boards()
        assert boards == sorted(boards)
        assert all(node != 0 for node, _ in boards)
        counts = built.board_event_counts()
        assert list(counts) == boards
        assert sum(counts.values()) == built.metadata()["n_events_kept"]

    def test_no_handle_left_open(self, built: UVCache) -> None:
        node, board = built.boards()[0]
        built.metadata()
        built.load_board(node, board)
        built.channels(node, board)
        built.channel_data(node, board, 0, 4)
        # HDF5 refuses to reopen a file for writing that this process still has open.
        with h5py.File(built.path, "a") as h5f:
            h5f.require_group("results")
        assert built.boards()  # an extra group does not disturb the readers

    def test_picklable(self, built: UVCache) -> None:
        clone = pickle.loads(pickle.dumps(built))
        assert clone.path == built.path
        assert clone.boards() == built.boards()
        assert repr(clone) == f"UVCache({str(built.path)!r})"

    def test_expected_events_helper_orders_channels(self) -> None:
        frames = [Frame(1, 15, 0, 5, (Hit(20, 1, 2, 3), Hit(5, 4, 5, 6)))]
        assert expected_events(frames)["channel"].tolist() == [5, 20]


def test_board_uv_helpers_on_hand_built_arrays() -> None:
    data = cache.BoardUV(
        node=1,
        board=15,
        rena=np.array([1, 0, 1], dtype=np.int8),
        channel=np.array([7, 4, 7], dtype=np.int8),
        u=np.array([10, 20, 30], dtype=np.int16),
        v=np.array([11, 21, 31], dtype=np.int16),
        pha=np.array([1, 2, 3], dtype=np.int16),
    )
    assert data.n_events == 3
    assert data.channels() == [(0, 4, 1), (1, 7, 2)]
    u, v = data.channel_data(1, 7)
    assert u.tolist() == [10, 30] and v.tolist() == [11, 31]
    assert data.channel_mask(0, 4).tolist() == [False, True, False]
    empty = cache.BoardUV(1, 15, *(np.zeros(0, dtype=d) for d in cache.EVENT_DTYPES.values()))
    assert empty.channels() == [] and empty.n_events == 0


# ----------------------------------------------------------------------
# Review fixes: target safety, concurrency, locks, directory errors,
# /results warnings, dtype checks, frame-less input, U = V = 0 count
# ----------------------------------------------------------------------

HoldOpen = Callable[[Path], AbstractContextManager[subprocess.Popen[str]]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_parse(_fraction: float) -> None:
    raise AssertionError("the build must fail before parsing")


class TestTargetSafety:
    def test_cache_path_equal_to_dat_is_refused(self, synthetic_file: SyntheticFile) -> None:
        dat = synthetic_file.path
        before = sha256(dat)
        with pytest.raises(CacheBuildError, match="raw data file itself"):
            UVCache(dat).build_from_dat(dat, progress_cb=no_parse)
        with pytest.raises(CacheBuildError, match="raw data file itself"):
            open_or_build(dat, cache_path=dat, progress_cb=no_parse)
        with pytest.raises(CacheBuildError, match="raw data file itself"):
            open_or_build(dat, cache_path=str(dat), force=True)
        assert sha256(dat) == before
        assert UVCache(dat).tmp_files() == []

    @pytest.mark.parametrize("kind", ["symlink", "hardlink", "dotdot"])
    def test_other_names_of_the_dat_are_refused(
        self, synthetic_file: SyntheticFile, tmp_path: Path, kind: str
    ) -> None:
        dat = synthetic_file.path
        before = sha256(dat)
        if kind == "symlink":
            target = tmp_path / "link.uv.h5"
            target.symlink_to(dat)
        elif kind == "hardlink":
            target = tmp_path / "hard.uv.h5"
            os.link(dat, target)
        else:
            (tmp_path / "sub").mkdir()
            target = tmp_path / "sub" / ".." / dat.name
        with pytest.raises(CacheBuildError, match="raw data file itself"):
            UVCache(target).build_from_dat(dat, progress_cb=no_parse)
        assert sha256(dat) == before

    def test_directory_is_refused_before_parsing(
        self, synthetic_file: SyntheticFile, tmp_path: Path
    ) -> None:
        target = tmp_path / "a_directory"
        target.mkdir()
        with pytest.raises(CacheBuildError, match="is a directory"):
            UVCache(target).build_from_dat(synthetic_file.path, progress_cb=no_parse)
        assert target.is_dir()

    @pytest.mark.parametrize("content", [b"important notes\n", b""], ids=["text", "empty"])
    def test_non_hdf5_file_is_never_replaced(
        self, synthetic_file: SyntheticFile, tmp_path: Path, content: bytes
    ) -> None:
        target = tmp_path / "notes.txt"
        target.write_bytes(content)
        with pytest.raises(CacheBuildError, match="not a uvcorr UV cache"):
            open_or_build(synthetic_file.path, cache_path=target, progress_cb=no_parse)
        assert target.read_bytes() == content

    def test_foreign_hdf5_file_is_never_replaced(
        self, synthetic_file: SyntheticFile, tmp_path: Path
    ) -> None:
        target = tmp_path / "calibration.h5"
        with h5py.File(target, "w") as h5f:
            h5f.create_group("metadata").attrs["cache_version"] = "2.0"
            h5f.create_dataset("fits", data=np.arange(10))
        before = sha256(target)
        with pytest.raises(CacheBuildError, match="not a uvcorr UV cache"):
            open_or_build(synthetic_file.path, cache_path=target, force=True)
        assert sha256(target) == before

    def test_uv_cache_of_another_file_is_replaced(
        self, synthetic_file: SyntheticFile, tmp_path: Path
    ) -> None:
        other = write_dat(tmp_path / "other.dat", [Frame(1, 15, 0, 1, (Hit(5, 1, 2, 3),))])
        target = tmp_path / "shared.uv.h5"
        open_or_build(other, cache_path=target)
        uv = open_or_build(synthetic_file.path, cache_path=target)
        assert uv.last_build is not None
        assert uv.is_valid_for(synthetic_file.path) and not uv.is_valid_for(other)


class TestConcurrentBuilds:
    def test_builds_use_separate_tmp_files(self, synthetic_file: SyntheticFile) -> None:
        # Build B runs to completion while build A is in progress (from A's
        # progress callback): neither touches the other's temporary file.
        path = default_cache_path(synthetic_file.path)
        a, b = UVCache(path), UVCache(path)
        seen_during_b: list[list[Path]] = []

        def run_b_once(_fraction: float) -> None:
            if b.last_build is None:
                b.build_from_dat(
                    synthetic_file.path,
                    progress_cb=lambda _f: seen_during_b.append(b.tmp_files()),
                    settings=BUDGET_SETTINGS,
                )

        a.build_from_dat(synthetic_file.path, progress_cb=run_b_once, settings=BUDGET_SETTINGS)
        assert a.last_build is not None and b.last_build is not None
        assert max(len(files) for files in seen_during_b) == 2  # A's and B's
        assert a.tmp_files() == []
        assert a.is_valid_for(synthetic_file.path)
        assert_cache_matches(a, synthetic_file)


class TestLocks:
    @pytest.fixture(autouse=True)
    def short_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cache, "LOCK_RETRY_SECONDS", 0.2)

    def test_busy_cache_is_not_invalid(
        self, built: UVCache, synthetic_file: SyntheticFile, hold_h5_open: HoldOpen
    ) -> None:
        with hold_h5_open(built.path):
            with pytest.raises(CacheBusyError, match="in use by another process"):
                built.is_valid_for(synthetic_file.path)
            with pytest.raises(CacheBusyError):
                open_or_build(synthetic_file.path)
            with pytest.raises(CacheBusyError):
                built.metadata()
            with pytest.raises(CacheBusyError):
                built.has_results()
        # Not rebuilt: the other process's /results survived.
        assert built.has_results()
        assert built.is_valid_for(synthetic_file.path)

    def test_build_refuses_a_busy_cache_before_parsing(
        self, built: UVCache, synthetic_file: SyntheticFile, hold_h5_open: HoldOpen
    ) -> None:
        with hold_h5_open(built.path), pytest.raises(CacheBusyError):
            built.build_from_dat(synthetic_file.path, progress_cb=no_parse)
        assert built.tmp_files() == []
        assert built.has_results()

    def test_busy_at_rename_time_keeps_the_other_process_data(
        self, built: UVCache, synthetic_file: SyntheticFile, hold_h5_open: HoldOpen
    ) -> None:
        holders: list[Any] = []

        def lock_during_build(_fraction: float) -> None:
            if not holders:
                context = hold_h5_open(built.path)
                holders.append((context, context.__enter__()))

        try:
            with pytest.raises(CacheBusyError):
                built.build_from_dat(synthetic_file.path, progress_cb=lock_during_build)
        finally:
            for context, _proc in holders:
                context.__exit__(None, None, None)
        assert built.tmp_files() == []
        assert built.has_results()

    def test_lock_released_within_the_retry_window(
        self,
        built: UVCache,
        synthetic_file: SyntheticFile,
        hold_h5_open: HoldOpen,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(cache, "LOCK_RETRY_SECONDS", 20.0)
        with hold_h5_open(built.path) as proc:
            assert proc.stdin is not None
            threading.Timer(0.3, proc.stdin.close).start()
            assert built.is_valid_for(synthetic_file.path)

    def test_is_lock_error(self) -> None:
        assert cache._is_lock_error(BlockingIOError(11, "Unable to open file"))
        assert cache._is_lock_error(
            OSError(
                "unable to lock file, errno = 11, error message = 'Resource temporarily unavailable'"
            )
        )
        assert not cache._is_lock_error(OSError("Unable to open file (file signature not found)"))
        assert not cache._is_lock_error(
            OSError(38, "unable to lock file, errno = 38, 'Function not implemented'")
        )


class TestValidityOfForeignFiles:
    def test_array_valued_attrs(self, built: UVCache, synthetic_file: SyntheticFile) -> None:
        with h5py.File(built.path, "a") as h5f:
            h5f["metadata"].attrs["uv_cache_version"] = np.array([1, 0, 0])
        assert not built.is_valid_for(synthetic_file.path)
        with h5py.File(built.path, "a") as h5f:
            h5f["metadata"].attrs["uv_cache_version"] = UV_CACHE_VERSION
            h5f["metadata"].attrs["source_size"] = np.array([1, 2])
        assert not built.is_valid_for(synthetic_file.path)

    def test_metadata_is_a_dataset(self, synthetic_file: SyntheticFile) -> None:
        path = default_cache_path(synthetic_file.path)
        with h5py.File(path, "w") as h5f:
            h5f.create_dataset("metadata", data=np.zeros(3))
            h5f.create_group("events")
        assert not UVCache(path).is_valid_for(synthetic_file.path)

    def test_has_results(self, built: UVCache, tmp_path: Path) -> None:
        assert not built.has_results()
        with h5py.File(built.path, "a") as h5f:
            h5f.create_group("results")
        assert built.has_results()
        assert not UVCache(tmp_path / "missing.h5").has_results()
        (tmp_path / "text.h5").write_text("not hdf5")
        assert not UVCache(tmp_path / "text.h5").has_results()


class TestDirectoryErrors:
    def test_parent_is_a_file(self, synthetic_file: SyntheticFile, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        with pytest.raises(CacheBuildError, match="Cannot create the cache directory"):
            UVCache(blocker / "sub" / "c.uv.h5").build_from_dat(
                synthetic_file.path, progress_cb=no_parse
            )

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
    def test_read_only_directory(self, synthetic_file: SyntheticFile, tmp_path: Path) -> None:
        readonly = tmp_path / "ro"
        readonly.mkdir()
        readonly.chmod(0o555)
        try:
            with pytest.raises(CacheBuildError, match=r"No write permission.*--cache PATH"):
                UVCache(readonly / "c.uv.h5").build_from_dat(
                    synthetic_file.path, progress_cb=no_parse
                )
            assert list(readonly.iterdir()) == []
        finally:
            readonly.chmod(0o755)

    def test_disk_usage_failure(
        self, synthetic_file: SyntheticFile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(_path: Any) -> Any:
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(shutil, "disk_usage", broken)
        with pytest.raises(CacheBuildError, match="free space"):
            UVCache(default_cache_path(synthetic_file.path)).build_from_dat(
                synthetic_file.path, progress_cb=no_parse
            )


class TestUserWarnings:
    def test_rebuild_warns_about_discarded_results(
        self, built: UVCache, synthetic_file: SyntheticFile, caplog: pytest.LogCaptureFixture
    ) -> None:
        with h5py.File(built.path, "a") as h5f:
            h5f.create_group("results/current")
        with caplog.at_level(logging.WARNING, logger="uvcorr.cache"):
            open_or_build(synthetic_file.path, force=True)
        records = [r for r in caplog.records if "/results" in r.getMessage()]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert getattr(records[0], cache.USER_WARNING) is True
        assert not built.has_results()

    def test_rebuild_without_results_does_not_warn(
        self, built: UVCache, synthetic_file: SyntheticFile, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="uvcorr.cache"):
            open_or_build(synthetic_file.path, force=True)
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_file_without_frames(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        dat = tmp_path / "notes.dat"
        dat.write_bytes(b"not a raw acquisition\n" * 100)
        with caplog.at_level(logging.WARNING, logger="uvcorr.cache"):
            stats = open_or_build(dat).last_build
        assert stats is not None and stats.parser_frames == stats.n_events_parsed == 0
        records = [r for r in caplog.records if "No valid frames" in r.getMessage()]
        assert len(records) == 1 and getattr(records[0], cache.USER_WARNING) is True


class TestBuildChecks:
    def test_unexpected_batch_dtype(self, synthetic_file: SyntheticFile) -> None:
        batch = next(PacketParser(synthetic_file.path).iter_event_arrays())
        cache._split_batch(batch, cache._Counts())  # the real dtypes pass
        wide = dataclasses.replace(batch, u=batch.u.astype(np.int32) + 40_000)
        with pytest.raises(CacheBuildError, match=r"'u' has dtype int32, expected int16"):
            cache._split_batch(wide, cache._Counts())

    def test_bad_dtype_fails_the_build_cleanly(
        self, synthetic_file: SyntheticFile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_iter = PacketParser.iter_event_arrays

        def widened(self: PacketParser, *args: Any, **kwargs: Any) -> Any:
            for batch in real_iter(self, *args, **kwargs):
                yield dataclasses.replace(batch, pha=batch.pha.astype(np.int64))

        monkeypatch.setattr(PacketParser, "iter_event_arrays", widened)
        uv = UVCache(default_cache_path(synthetic_file.path))
        with pytest.raises(CacheBuildError, match="'pha' has dtype int64"):
            uv.build_from_dat(synthetic_file.path)
        assert not uv.exists() and uv.tmp_files() == []

    def test_uv_zero_count(self, tmp_path: Path) -> None:
        frames = [
            Frame(1, 15, 0, 1, (Hit(5, 100, 0, 0), Hit(6, 100, 0, 7), Hit(7, 100, 9, 0))),
            Frame(2, 16, 1, 2, (Hit(7, 0, 0, 0), Hit(20, 5, 1, 1))),
            Frame(2, 16, 1, 3, (Hit(3, 0, 0, 0),)),  # inactive channel: dropped
            Frame(0, 16, 0, 4, (Hit(10, 0, 0, 0),)),  # node 0: dropped
        ]
        dat = write_dat(tmp_path / "zeros.dat", frames)
        uv = UVCache(default_cache_path(dat))
        stats = uv.build_from_dat(dat)
        assert stats.n_events_kept == 5
        assert stats.n_events_uv_zero == 2
        assert uv.metadata()["n_events_uv_zero"] == 2
        assert uv.channel_data(2, 16, 1, 7)[0].tolist() == [0]
