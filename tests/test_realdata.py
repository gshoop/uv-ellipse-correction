"""Checks of the UV cache of the real test acquisition (plan section 10.1).

Marker ``realdata`` (``make test-realdata`` / ``pytest -m realdata``). The
tests skip cleanly when the test ``.dat`` file or its sibling cache is absent;
build the cache first with ``uvcorr build-cache <file>``. Set
``UVCORR_TEST_DAT`` to run them against another acquisition (the exact event
counts are then not checked).
"""

from __future__ import annotations

import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from adc2kev.parser import PacketParser

from uvcorr.cache import UVCache, default_cache_path
from uvcorr.channels import is_active_channel

pytestmark = pytest.mark.realdata

REFERENCE_DAT = Path.home() / "adc2kev-test-data/full-system/sources/ge/data_20260910_120628.dat"
TEST_DAT = Path(os.environ.get("UVCORR_TEST_DAT", str(REFERENCE_DAT)))
IS_REFERENCE = TEST_DAT == REFERENCE_DAT

# Measured on the reference file (plan section 11, re-verified in phase 1 by
# an independent parse): no node-0 events; 5 of the 160 node/board
# combinations have no events on active channels.
REFERENCE_COUNTS = {
    "parser_frames": 106_063_959,
    "parser_events": 208_492_906,
    "parser_dropped": 23,
    "n_events_kept": 208_267_950,
    "n_events_inactive": 224_956,
    "n_events_node0": 0,
    "n_boards": 155,
}


@pytest.fixture(scope="module")
def real_dat() -> Path:
    if not TEST_DAT.is_file():
        pytest.skip(f"test acquisition not found: {TEST_DAT}")
    return TEST_DAT


@pytest.fixture(scope="module")
def real_cache() -> UVCache:
    cache = UVCache(default_cache_path(TEST_DAT))
    if not cache.exists():
        pytest.skip(f"UV cache not found: {cache.path} (run: uvcorr build-cache {TEST_DAT})")
    return cache


def test_cache_is_valid_for_the_file(real_dat: Path, real_cache: UVCache) -> None:
    assert real_cache.is_valid_for(real_dat)


def test_board_counts_sum_to_metadata(real_cache: UVCache) -> None:
    meta = real_cache.metadata()
    counts = real_cache.board_event_counts()
    assert sum(counts.values()) == meta["n_events_kept"]
    assert len(counts) == meta["n_boards"]
    assert (
        meta["n_events_kept"] + meta["n_events_inactive"] + meta["n_events_node0"]
        == meta["parser_events"]
    )


def test_boards_are_nodes_1_to_10_and_boards_15_to_30(real_cache: UVCache) -> None:
    boards = real_cache.boards()
    assert set(boards) <= set(product(range(1, 11), range(15, 31)))
    assert {node for node, _ in boards} == set(range(1, 11))
    assert {board for _, board in boards} == set(range(15, 31))


@pytest.mark.skipif(not IS_REFERENCE, reason="exact counts are known for the reference file")
def test_reference_counts(real_cache: UVCache) -> None:
    meta = real_cache.metadata()
    assert {name: meta[name] for name in REFERENCE_COUNTS} == REFERENCE_COUNTS
    # AND-mode data: every event has a U/V sample (0 measured by a build of the
    # final phase 1 code). Caches built before the attr existed lack it.
    assert meta.get("n_events_uv_zero", 0) == 0


def test_every_stored_channel_is_active(real_cache: UVCache) -> None:
    counts = real_cache.board_event_counts()
    for (node, board), n_board in counts.items():
        channels = real_cache.channels(node, board)
        inactive = [(r, c) for r, c, _ in channels if not is_active_channel(r, c)]
        assert inactive == [], f"node {node} board {board}"
        assert len(channels) <= 47
        assert sum(n for _, _, n in channels) == n_board


def test_load_board_is_fast(real_cache: UVCache) -> None:
    counts = real_cache.board_event_counts()
    (node, board), n = sorted(counts.items(), key=lambda kv: kv[1])[len(counts) // 2]
    start = time.perf_counter()
    data = real_cache.load_board(node, board)
    elapsed = time.perf_counter() - start
    assert data.n_events == n
    assert elapsed < 2.0  # ~20 ms measured for a 1.1M-event board


@pytest.mark.slow
def test_independent_reparse_matches_cache(real_dat: Path, real_cache: UVCache) -> None:
    """Re-parse the whole file, count active events per board, compare with the cache.

    Uses explicit range comparisons (not ``uvcorr.channels``) and also checks one
    median-sized board event for event, in file order.
    """
    counts = real_cache.board_event_counts()
    probe, _ = sorted(counts.items(), key=lambda kv: kv[1])[len(counts) // 2]

    per_board = np.zeros((256, 64), dtype=np.int64)
    probe_u: list[np.ndarray] = []
    probe_v: list[np.ndarray] = []
    probe_ch: list[np.ndarray] = []
    total = 0
    for batch in PacketParser(real_dat).iter_event_arrays(batch_events=2_000_000):
        total += batch.n_events
        r, c = batch.rena_num, batch.channel_num
        active = ((r == 0) & (c >= 4) & (c <= 28)) | ((r == 1) & (c >= 7) & (c <= 28))
        active &= batch.node_num != 0
        np.add.at(per_board, (batch.node_num[active], batch.board_num[active]), 1)
        sel = active & (batch.node_num == probe[0]) & (batch.board_num == probe[1])
        probe_u.append(batch.u[sel])
        probe_v.append(batch.v[sel])
        probe_ch.append(r[sel].astype(np.int16) * 64 + c[sel])

    reparsed = {(int(n), int(b)): int(per_board[n, b]) for n, b in zip(*np.nonzero(per_board))}
    meta = real_cache.metadata()
    assert total == meta["parser_events"]
    assert sum(reparsed.values()) == meta["n_events_kept"]
    assert reparsed == counts

    data = real_cache.load_board(*probe)
    np.testing.assert_array_equal(data.u, np.concatenate(probe_u))
    np.testing.assert_array_equal(data.v, np.concatenate(probe_v))
    np.testing.assert_array_equal(
        data.rena.astype(np.int16) * 64 + data.channel, np.concatenate(probe_ch)
    )
