"""Pytest configuration and shared fixtures for uvcorr tests.

GUI tests (pytest-qt, ``qt_api = "pyqt6"`` in ``pyproject.toml``) run on Qt's
``offscreen`` platform by default so the suite works headless. Export
``QT_QPA_PLATFORM`` (e.g. ``xcb``) before running pytest to watch them on a
real display instead.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from tests.synthetic_dat import Frame, expected_events, random_frames, write_dat

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
