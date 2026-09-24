"""Synthetic ring caches for the GUI tests (helpers, not fixtures).

The ring ``.dat`` of ``tests/conftest.py`` has three boards, (1, 15), (1, 16)
and (4, 29), each with six channels covering every display case:

- R0 Ch05: a clean ring (700 events; a cathode on odd boards),
- R0 Ch12: a ring with 12 % outliers (``high_rejection``, rejected points),
- R1 Ch09: an eccentric ring (``extreme_axis_ratio``),
- R1 Ch25: a near-circle (a cathode on even boards),
- R1 Ch28: 40 events (``too_few_events``),
- R0 Ch20: collinear points (``fit_failed``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from uvcorr.analysis import ChannelKey, analyze_all
from uvcorr.cache import UVCache
from uvcorr.options import FitOptions

RING_BOARDS = ((1, 15), (1, 16), (4, 29))
CHANNELS_PER_BOARD = 6

CLEAN = ChannelKey(1, 15, 0, 5)
OUTLIERS = ChannelKey(1, 15, 0, 12)
ECCENTRIC = ChannelKey(1, 15, 1, 9)
TOO_FEW = ChannelKey(1, 15, 1, 28)
FIT_FAILED = ChannelKey(1, 15, 0, 20)


def copy_files(dat: Path, cache: Path, directory: Path) -> tuple[Path, Path]:
    """Copy a ``.dat`` and its cache into ``directory`` (``copy2`` keeps the cache valid)."""
    directory.mkdir(parents=True, exist_ok=True)
    dat_copy = directory / dat.name
    cache_copy = directory / cache.name
    shutil.copy2(dat, dat_copy)
    shutil.copy2(cache, cache_copy)
    return dat_copy, cache_copy


def store_batch_results(cache_path: Path, options: FitOptions | None = None) -> None:
    """Fit every channel in-process and store the results in the cache."""
    opts = options if options is not None else FitOptions()
    cache = UVCache(cache_path)
    cache.save_results(analyze_all(cache, opts, workers=1), opts)
