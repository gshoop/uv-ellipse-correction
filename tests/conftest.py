"""Pytest configuration and shared fixtures for uvcorr tests.

GUI tests (pytest-qt, ``qt_api = "pyqt6"`` in ``pyproject.toml``) run on Qt's
``offscreen`` platform by default so the suite works headless. Export
``QT_QPA_PLATFORM`` (e.g. ``xcb``) before running pytest to watch them on a
real display instead.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from tests.synthetic_dat import Frame, Hit, expected_events, random_frames, write_dat

# Must happen before the first QApplication is created (pytest-qt creates it
# lazily in the ``qapp``/``qtbot`` fixtures).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@dataclass(frozen=True)
class SyntheticFile:
    """A synthetic ``.dat`` file and the events the parser must decode from it."""

    path: Path
    frames: list[Frame]
    events: npt.NDArray[np.void]

    @property
    def active_events(self) -> npt.NDArray[np.void]:
        """Events the UV cache keeps (active channel, node != 0), in file order."""
        from uvcorr.channels import active_channel_mask

        keep = active_channel_mask(self.events["rena"], self.events["channel"]) & (
            self.events["node"] != 0
        )
        return self.events[keep]


# Nodes 0 (dropped), 1, 2, 5 and 10; boards of both parities; all 36 channels
# of both RENAs, so inactive channels (R0 0-3 and 29-35, R1 0-6 and 29-35) occur.
SYNTHETIC_NODES = (0, 1, 2, 5, 10)
SYNTHETIC_BOARDS = (15, 16, 29, 30)


@pytest.fixture(scope="session")
def synthetic_frames() -> list[Frame]:
    """4000 random AND-mode frames (about 12k events) over several nodes and boards."""
    return random_frames(4000, seed=7, nodes=SYNTHETIC_NODES, boards=SYNTHETIC_BOARDS, max_hits=5)


@pytest.fixture
def synthetic_file(tmp_path: Path, synthetic_frames: list[Frame]) -> SyntheticFile:
    """The synthetic frames written to ``<tmp>/synthetic.dat``."""
    path = write_dat(tmp_path / "synthetic.dat", synthetic_frames)
    return SyntheticFile(path, synthetic_frames, expected_events(synthetic_frames))


_HOLD_SCRIPT = """
import sys, h5py
f = h5py.File(sys.argv[1], "a")
f.require_group("results/current").attrs["saved"] = "precious override"
f.flush()
print("ready", flush=True)
sys.stdin.read()  # hold the file (and its HDF5 lock) until the parent closes stdin
f.close()
"""


@contextmanager
def _hold_h5_open(path: Path) -> Iterator[subprocess.Popen[str]]:
    """Keep ``path`` open for writing in another process (adds ``/results/current``).

    HDF5 file locks only conflict between processes, so a child process holds
    the file; it releases it when the block exits (or earlier, when the caller
    closes ``proc.stdin``).
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLD_SCRIPT, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
        yield proc
    finally:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
        proc.wait(timeout=30)


@pytest.fixture
def hold_h5_open() -> Callable[[Path], AbstractContextManager[subprocess.Popen[str]]]:
    """Factory: ``with hold_h5_open(path): ...`` keeps ``path`` locked by another process."""
    return _hold_h5_open


# ---------------------------------------------------------------------------
# Synthetic ring data (analysis tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RingChannel:
    """Recipe of one synthetic channel: points on an ellipse with radial noise.

    ``kind`` selects the shape: ``"ellipse"`` (Gaussian radial noise
    ``noise``, plus ``outlier_frac`` uniform outliers over a 1400 ADC box),
    ``"flat"`` (radial noise uniform in +-``noise``: a flat-topped radial
    distribution, not Gaussian) or ``"line"`` (exactly collinear points: no
    ellipse can be fitted).
    """

    rena: int
    channel: int
    n: int
    kind: str = "ellipse"
    cx: float = 2000.0
    cy: float = 2050.0
    a: float = 600.0
    b: float = 585.0
    phi: float = 0.3
    noise: float = 6.0
    outlier_frac: float = 0.0


# Per board (both parities): a clean ring, a ring with 12 % outliers
# (high_rejection), an eccentric ring (extreme_axis_ratio), a channel below
# min_events (too_few_events) and a collinear channel (fit_failed). The
# cathode on RENA 1 channel 25 (even boards) / RENA 0 channel 5 (odd boards)
# is covered by the clean ring of each parity.
RING_CHANNELS: tuple[RingChannel, ...] = (
    RingChannel(rena=0, channel=5, n=700),
    RingChannel(rena=0, channel=12, n=500, outlier_frac=0.12),
    RingChannel(rena=1, channel=9, n=400, a=620.0, b=250.0, phi=-1.2, noise=3.0),
    RingChannel(rena=1, channel=25, n=600, a=610.0, b=600.0, phi=1.4),
    RingChannel(rena=1, channel=28, n=40),
    RingChannel(rena=0, channel=20, n=150, kind="line"),
)
RING_BOARDS: tuple[tuple[int, int], ...] = ((1, 15), (1, 16), (4, 29))


def ring_points(
    spec: RingChannel, seed: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Integer (U, V) points of a :class:`RingChannel` (12-bit, file order)."""
    rng = np.random.default_rng(seed)
    if spec.kind == "line":
        # Integer steps on the diagonal: exactly collinear after rounding
        t = rng.integers(-300, 301, spec.n).astype(np.float64)
        u, v = spec.cx + t, spec.cy + t
    else:
        t = rng.uniform(0.0, 2.0 * math.pi, spec.n)
        ea, eb = spec.a * np.cos(t), spec.b * np.sin(t)
        u = spec.cx + ea * math.cos(spec.phi) - eb * math.sin(spec.phi)
        v = spec.cy + ea * math.sin(spec.phi) + eb * math.cos(spec.phi)
        r = np.hypot(u - spec.cx, v - spec.cy)
        if spec.kind == "flat":
            dr = rng.uniform(-spec.noise, spec.noise, spec.n)
        else:
            dr = rng.normal(0.0, spec.noise, spec.n)
        scale = 1.0 + dr / r
        u = spec.cx + (u - spec.cx) * scale
        v = spec.cy + (v - spec.cy) * scale
        n_out = int(round(spec.outlier_frac * spec.n))
        if n_out:
            idx = rng.choice(spec.n, n_out, replace=False)
            u[idx] = rng.uniform(spec.cx - 700.0, spec.cx + 700.0, n_out)
            v[idx] = rng.uniform(spec.cy - 700.0, spec.cy + 700.0, n_out)
    ui = np.clip(np.rint(u), 0, 4095).astype(np.int64)
    vi = np.clip(np.rint(v), 0, 4095).astype(np.int64)
    return ui, vi


def ring_frames(
    boards: tuple[tuple[int, int], ...] = RING_BOARDS,
    channels: tuple[RingChannel, ...] = RING_CHANNELS,
    seed: int = 11,
) -> list[Frame]:
    """One-hit frames carrying the ring channels of every board, interleaved.

    Also adds a few events on an inactive channel (RENA 0 channel 2), which
    the cache drops.
    """
    frames: list[Frame] = []
    timestamp = 1000
    for b_index, (node, board) in enumerate(boards):
        per_channel = []
        for c_index, spec in enumerate(channels):
            u, v = ring_points(spec, seed + 100 * b_index + c_index)
            per_channel.append([(spec.rena, spec.channel, int(x), int(y)) for x, y in zip(u, v)])
        per_channel.append([(0, 2, 2000, 2000)] * 5)
        # Round-robin over the channels, as events arrive in a real file
        longest = max(len(events) for events in per_channel)
        for i in range(longest):
            for events in per_channel:
                if i < len(events):
                    rena, channel, u_val, v_val = events[i]
                    frames.append(
                        Frame(node, board, rena, timestamp, (Hit(channel, 100, u_val, v_val),))
                    )
                    timestamp += 3
    return frames


@dataclass(frozen=True)
class RingFiles:
    """A synthetic ring ``.dat`` file and its (valid) UV cache."""

    dat: Path
    cache: Path


@pytest.fixture(scope="session")
def _ring_files_master(tmp_path_factory: pytest.TempPathFactory) -> RingFiles:
    from uvcorr.cache import open_or_build

    directory = tmp_path_factory.mktemp("rings")
    dat = write_dat(directory / "rings.dat", ring_frames())
    cache = open_or_build(dat)
    return RingFiles(dat, cache.path)


@pytest.fixture
def ring_files(tmp_path: Path, _ring_files_master: RingFiles) -> RingFiles:
    """A private copy of the synthetic ring ``.dat`` and its valid cache (no results)."""
    dat = tmp_path / _ring_files_master.dat.name
    cache = tmp_path / _ring_files_master.cache.name
    shutil.copy2(_ring_files_master.dat, dat)  # copy2 keeps the mtime: the cache stays valid
    shutil.copy2(_ring_files_master.cache, cache)
    return RingFiles(dat, cache)
