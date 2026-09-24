#!/usr/bin/env python3
"""Compare C++ RadialAnalysis output with uvcorr ``--no-robust`` (plan section 10.3).

Inputs:

- ``CPP_DIR``: the ``.uvd`` directory given to ``RadialAnalysis --csv-only``
  (written by ``scripts/dump_uvd.py``), or its ``RadialAnalysis_output``
  subdirectory. The ``.uvd`` file names give the channels the C++ was asked to
  analyse; ``radial_summary.csv`` and the ``.tec`` give its results.
- The uvcorr side, either
  ``--uvcorr-dir DIR`` with the ``radial_summary.csv`` (and ``.tec``) written by
  ``uvcorr process --no-robust --output-dir DIR``, or ``--cache PATH``, which
  runs ``analyze_board(cache, node, board, FitOptions(robust=False,
  min_events=--min-events))`` on the boards of the ``.uvd`` files (read-only).

What is compared, per channel and in aggregate:

- **Channel sets**: the ``.uvd`` input, the channels the C++ wrote a CSV row for
  (it skips a channel when ``findCenter`` or the pre-correction Gaussian fit
  fails), those with an ellipse (``.tec`` block), and uvcorr's status.
- **Event counts**: C++ ``num_events`` against uvcorr ``n_events`` (must be
  equal: both read the same events).
- **Ellipse parameters** (``--rtol``, default 1e-6, plan 10.3): the centre (C++
  only writes it to the ``.tec``, with 6 significant digits, so the tolerance
  adds half a unit of the printed digit), the semi-axes (CSV, 6 decimals) and
  phi **modulo pi**. The C++ phi is in [-pi/2, pi) (the axis swap adds pi/2 and
  its ``while (phi > PI)`` loop only trims a ~3e-8 rad sliver), uvcorr's in
  (-pi/2, pi/2]. Phi of a near-circle is ill-conditioned: a relative error eps
  of the conic moves it by about ``eps a^2 / (a^2 - b^2)``, so phi is checked
  with that tolerance. ``phi_disp = |dphi| (a^2 - b^2) / a^2`` is the matching
  dimensionless displacement of the ellipse.
- **Pedestal blobs**: channels that uvcorr flags ``broad_ring`` (not a thin
  ring) are summarised on their own line and their disagreements are not
  failures, unless ``--strict-blobs``. On blobs of radius ~10 ADC the C++ fit on
  raw coordinates (U, V ~ 2000) is numerically unreliable (docs/ALGORITHM.md).
- **Radial, residual and phase statistics**: differences uvcorr - C++. They
  are expected (plan 5.3, D8): pre radii are about the fitted centre instead of
  ``findCenter``; the Gaussian fit is a binned Poisson ML fit with
  Freedman-Diaconis bins and an iterated range instead of ROOT's chi2 fit over
  mean +- 3 RMS of a 200-bin histogram; skewness and kurtosis are unbinned
  instead of ROOT's binned ``GetSkewness``/``GetKurtosis``.

``--decompose`` (needs ``--cache`` for the raw points) splits those
differences. It re-implements the C++ statistics in Python (``findCenter``,
the ``TH1D`` fill and statistics, ROOT's chi2 fit range and empty-bin rule,
``GetSkewness``/``GetKurtosis``, ``residualHistogram``), reports on how many
channels that emulation reproduces the C++, and then changes one ingredient
at a time towards uvcorr:

- pre radii: C++ emulation (findCenter centre, ROOT-style fit), then the fitted
  centre, then the Poisson ML fit on the same histogram and range; the rest of
  the difference is uvcorr's binning and iterated range;
- post radii: C++ emulation on uvcorr's corrected points (the ellipses agree
  to ~1e-6), then the Poisson ML fit; skewness and kurtosis also unbinned
  without the maximum (which ROOT puts in the overflow bin);
- residuals: C++ emulation.

The ``.tec`` file is the only ``*.tec`` in the output directory, or ``--tec``.

Output: a comparison CSV (``--out``, one row per channel) and a summary on
stdout. Exit code 0; 1 if an ellipse parameter of a ring channel (or of any
channel with ``--strict-blobs``) or an event count disagrees; 2 for bad
arguments or unreadable inputs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from uvcorr.analysis import ChannelKey, ChannelResult, FitOptions, analyze_board
from uvcorr.cache import UVCache
from uvcorr.channels import electrode_label, is_active_channel, polarity_name
from uvcorr.ellipse import (
    EllipseParams,
    correct,
    corrected_residual,
    radii_about_center,
    residual_to_ellipse,
)
from uvcorr.io.summary_csv import SUMMARY_CSV_NAME, read_summary_csv
from uvcorr.io.tec import TecEntry, read_tec
from uvcorr.metrics import poisson_deviance
from uvcorr.options import FLAG_BROAD_RING

FloatArray = npt.NDArray[np.float64]

OUTPUT_SUBDIR = "RadialAnalysis_output"
PHASE_REF_FREQ_HZ = 490e3
DEFAULT_RTOL = 1e-6

# The C++ CSV writes doubles with ``fixed << setprecision(6)``: 6 decimals.
CPP_CSV_HALF_QUANTUM = 0.5e-6
# uvd_common.h uses PI = 3.1415926 when it swaps the axes (phi += PI/2) and when it
# normalises phi (phi -= PI): each costs up to (pi - 3.1415926) = 5.4e-8.
CPP_PI_ERROR = 2 * (math.pi - 3.1415926)
PHI_ABS_TOL = CPP_CSV_HALF_QUANTUM + CPP_PI_ERROR

# ROOT / RadialAnalysis constants (main_radial.cpp, uvd_common.h)
RADIAL_NUMBINS = 200
CENTER_NUMBINS = 100
RESIDUAL_NUMBINS = 150
RESIDUAL_RANGE_STD = 6.0
FIT_RANGE_RMS = 3.0
FWHM_FACTOR = 2.35482


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def phi_diff_mod_pi(phi_a: float, phi_b: float) -> float:
    """``phi_a - phi_b`` modulo pi, in [-pi/2, pi/2] (ellipses are symmetric under pi).

    Example:
        >>> abs(phi_diff_mod_pi(3.09506, 3.09506 - math.pi)) < 1e-12
        True
    """
    return math.remainder(phi_a - phi_b, math.pi)


def phi_tolerance(a: float, b: float, rtol: float) -> float:
    """Tolerance on phi for a relative accuracy ``rtol`` of the conic (see the module docstring).

    ``PHI_ABS_TOL + rtol * a^2 / (a^2 - b^2)``; infinite for a circle.
    """
    aniso = a * a - b * b
    if not aniso > 0.0:
        return math.inf
    return PHI_ABS_TOL + rtol * a * a / aniso


def printed_half_quantum(value: float, digits: int = 6) -> float:
    """Half a unit in the last place of ``value`` printed with ``%g`` (``digits`` significant)."""
    if value == 0.0 or not math.isfinite(value):
        return 0.0
    exponent = math.floor(math.log10(abs(float(format(value, f".{digits}g")))))
    return 0.5 * 10.0 ** (exponent - digits + 1)


def rel_diff(new: float | None, ref: float | None) -> float | None:
    """``(new - ref) / |ref|``; None if either is missing or ``ref`` is 0."""
    if new is None or ref is None or ref == 0.0:
        return None
    return (new - ref) / abs(ref)


def _float_or_none(text: str) -> float | None:
    text = text.strip()
    if text == "":
        return None
    value = float(text)
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------------------
# C++ output
# ---------------------------------------------------------------------------

CPP_INT_COLUMNS = frozenset({"node", "board", "rena", "channel", "num_events"})

_dump_uvd_module: ModuleType | None = None


def _dump_uvd() -> ModuleType:
    """``scripts/dump_uvd.py``, loaded from next to this file (``scripts/`` is no package)."""
    global _dump_uvd_module
    if _dump_uvd_module is None:
        path = Path(__file__).resolve().with_name("dump_uvd.py")
        spec = importlib.util.spec_from_file_location("_compare_radial_dump_uvd", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _dump_uvd_module = module
    return _dump_uvd_module


def parse_uvd_filename(name: str) -> ChannelKey:
    """Channel of a ``.uvd`` file name (``dump_uvd.parse_uvd_filename``, the one copy).

    Raises:
        ValueError: If the name does not have that form.
    """
    return ChannelKey(*_dump_uvd().parse_uvd_filename(name))


@dataclass
class CppOutput:
    """What one RadialAnalysis run produced.

    Attributes:
        uvd_dir: The ``.uvd`` input directory.
        inputs: Channels with a ``.uvd`` file (what the C++ was asked to analyse).
        rows: ``radial_summary.csv`` rows by channel (floats or None; the int
            columns as int).
        tec: ``.tec`` blocks by channel (channels with an ellipse and a
            successful post-correction Gaussian fit).
        tec_path: The ``.tec`` file read (None if there was none).
    """

    uvd_dir: Path
    inputs: set[ChannelKey]
    rows: dict[ChannelKey, dict[str, Any]]
    tec: dict[ChannelKey, TecEntry]
    tec_path: Path | None = None


def read_cpp_csv(path: str | Path) -> dict[ChannelKey, dict[str, Any]]:
    """Read a RadialAnalysis ``radial_summary.csv`` (old schema, with psi and shear columns).

    Returns:
        ``{key: {column: value}}``; empty cells are None, ``node``, ``board``,
        ``rena``, ``channel`` and ``num_events`` are int, the rest float.

    Raises:
        ValueError: If the header lacks the identity columns or a row is malformed.
    """
    rows: dict[ChannelKey, dict[str, Any]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = [name.strip() for name in next(reader, [])]
        missing = [c for c in ("node", "board", "rena", "channel", "ell_a") if c not in header]
        if missing:
            raise ValueError(f"{path}: not a RadialAnalysis radial_summary.csv (missing {missing})")
        for lineno, cells in enumerate(reader, start=2):
            if not cells or all(not c.strip() for c in cells):
                continue
            if len(cells) != len(header):
                raise ValueError(
                    f"{path} line {lineno}: {len(cells)} cells, expected {len(header)}"
                )
            row: dict[str, Any] = {}
            for name, cell in zip(header, cells):
                if name in CPP_INT_COLUMNS:
                    row[name] = int(cell) if cell.strip() else None
                else:
                    row[name] = _float_or_none(cell)
            key = ChannelKey(row["node"], row["board"], row["rena"], row["channel"])
            if key in rows:
                raise ValueError(f"{path} line {lineno}: second row for {key}")
            rows[key] = row
    return rows


def read_cpp_output(path: str | Path, tec: str | Path | None = None) -> CppOutput:
    """Read a RadialAnalysis run from its ``.uvd`` directory or its output directory.

    Args:
        path: The ``.uvd`` directory or its ``RadialAnalysis_output``.
        tec: The ``.tec`` file; default: the only ``*.tec`` in the output
            directory (none is allowed: no channel then has a ``.tec`` block).

    Raises:
        FileNotFoundError: If ``radial_summary.csv`` (or the given ``tec``) is missing.
        ValueError: If the output directory has several ``.tec`` files and
            ``tec`` is not given, or a file is malformed.
    """
    directory = Path(path)
    if directory.name == OUTPUT_SUBDIR:
        uvd_dir, out_dir = directory.parent, directory
    else:
        uvd_dir, out_dir = directory, directory / OUTPUT_SUBDIR
    csv_path = out_dir / "radial_summary.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"RadialAnalysis CSV not found: {csv_path}")
    inputs: set[ChannelKey] = set()
    for uvd in uvd_dir.glob("*.uvd"):
        inputs.add(parse_uvd_filename(uvd.name))
    tec_path: Path | None
    if tec is not None:
        tec_path = Path(tec)
        if not tec_path.is_file():
            raise FileNotFoundError(f".tec file not found: {tec_path}")
    else:
        tec_files = sorted(out_dir.glob("*.tec"))
        if len(tec_files) > 1:
            names = ", ".join(p.name for p in tec_files)
            raise ValueError(f"several .tec files in {out_dir} ({names}); choose one with --tec")
        tec_path = tec_files[0] if tec_files else None
    tec_entries = dict(read_tec(tec_path)) if tec_path is not None else {}
    return CppOutput(uvd_dir, inputs, read_cpp_csv(csv_path), tec_entries, tec_path)


# ---------------------------------------------------------------------------
# uvcorr side
# ---------------------------------------------------------------------------


def read_uvcorr_dir(path: str | Path) -> dict[ChannelKey, ChannelResult]:
    """Read uvcorr's ``radial_summary.csv`` from an output directory (or the file itself)."""
    p = Path(path)
    csv_path = p / SUMMARY_CSV_NAME if p.is_dir() else p
    return {result.key: result for result in read_summary_csv(csv_path)}


def compute_uvcorr(
    cache: UVCache, boards: Iterable[tuple[int, int]], options: FitOptions
) -> dict[ChannelKey, ChannelResult]:
    """Analyse the given boards of a cache on the fly (read-only)."""
    results: dict[ChannelKey, ChannelResult] = {}
    for node, board in sorted(set(boards)):
        for result in analyze_board(cache, node, board, options):
            results[result.key] = result
    return results


# ---------------------------------------------------------------------------
# Emulation of the C++ statistics (for --decompose)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RootHist:
    """A ``TH1D`` filled with values: bin counts and the in-range (unbinned) statistics.

    ROOT puts ``x < xmin`` in the underflow and ``x >= xmax`` in the overflow
    bin (so a histogram over [min, max] of the values loses the maximum);
    neither enters the statistics (``fTsumw``...), so ``GetMean`` and
    ``GetStdDev`` are the unbinned mean and population std of the in-range
    values.
    """

    lo: float
    hi: float
    counts: FloatArray
    mean: float
    std: float

    @property
    def nbins(self) -> int:
        return int(self.counts.size)

    @property
    def centers(self) -> FloatArray:
        width = (self.hi - self.lo) / self.nbins
        centers: FloatArray = self.lo + (np.arange(self.nbins) + 0.5) * width
        return centers


def root_hist(values: npt.ArrayLike, nbins: int, lo: float, hi: float) -> RootHist | None:
    """Fill a ROOT-like ``TH1D(nbins, lo, hi)``; None if the range is empty."""
    x = np.asarray(values, dtype=np.float64)
    if not hi > lo:
        return None
    with np.errstate(all="ignore"):
        idx = np.floor(nbins * (x - lo) / (hi - lo))
    # TAxis::FindBin tests x < xmin and !(x < xmax) before the arithmetic, so x == xmax is
    # always overflow even when nbins * (x - lo) / (hi - lo) rounds to just below nbins.
    inside = (x >= lo) & (x < hi) & (idx < nbins)
    counts = np.bincount(idx[inside].astype(np.int64), minlength=nbins).astype(np.float64)
    xin = x[inside]
    if xin.size == 0:
        return RootHist(lo, hi, counts, math.nan, math.nan)
    mean = float(np.mean(xin))
    var = float(np.mean(xin * xin)) - mean * mean
    return RootHist(lo, hi, counts, mean, math.sqrt(max(var, 0.0)))


@dataclass(frozen=True)
class RootFit:
    """Result of the emulated ``fitRadialHist`` (NaN fields if not ok)."""

    ok: bool
    mean: float
    sigma: float
    chi2ndf: float
    skewness: float
    kurtosis: float

    @property
    def fwhm(self) -> float:
        return FWHM_FACTOR * self.sigma


def root_binned_shape(hist: RootHist) -> tuple[float, float]:
    """ROOT ``GetSkewness``/``GetKurtosis``: binned moments about the unbinned mean/std."""
    w = hist.counts
    np_ = float(np.sum(w))
    if np_ <= 0 or not hist.std > 0:
        return math.nan, math.nan
    d = hist.centers - hist.mean
    skew = float(np.sum(w * d**3)) / (np_ * hist.std**3)
    kurt = float(np.sum(w * d**4)) / (np_ * hist.std**4) - 3.0
    return skew, kurt


def root_fit_range(hist: RootHist) -> npt.NDArray[np.int64]:
    """0-based bins of ``Fit(..., mean - 3 RMS, mean + 3 RMS)`` (ROOT ``ExamineRange``).

    The bins containing the end points, clamped to the axis, minus an end bin
    whose centre is outside the range (empty bins included).
    """
    xlo = hist.mean - FIT_RANGE_RMS * hist.std
    xhi = hist.mean + FIT_RANGE_RMS * hist.std
    centers = hist.centers

    def find_bin(x: float) -> int:  # 0-based; -1 underflow, nbins overflow
        if x < hist.lo:
            return -1
        if x >= hist.hi:
            return hist.nbins
        return int((hist.nbins * (x - hist.lo)) / (hist.hi - hist.lo))

    first = max(find_bin(xlo), 0)
    last = min(find_bin(xhi), hist.nbins - 1)
    if first < last:
        if centers[first] < xlo:
            first += 1
        if centers[last] > xhi:
            last -= 1
    return np.arange(first, last + 1, dtype=np.int64)


def _start_values(x: FloatArray, y: FloatArray, width: float) -> FloatArray:
    """Gaussian start values from the moments of the fitted bins (like ROOT's H1InitGaus)."""
    w = y / np.sum(y)
    mu0 = float(np.sum(w * x))
    sig0 = math.sqrt(max(float(np.sum(w * (x - mu0) ** 2)), (0.5 * width) ** 2))
    return np.array([float(np.max(y)), mu0, sig0])


def root_gaus_fit(hist: RootHist) -> RootFit:
    """Emulate ``fitRadialHist``: ``h->Fit("gaus", "QS", "", mean - 3 RMS, mean + 3 RMS)``.

    ROOT's default chi2 fit (Neyman): the bins of :func:`root_fit_range`,
    empty bins skipped, errors ``sqrt(n)``, the Gaussian evaluated at the bin
    centre, NDF = points - 3. The minimum is found with Levenberg-Marquardt
    instead of Minuit; both converge to the same chi2 minimum well within its
    statistical error (Minuit may stop elsewhere on a handful of events).
    """
    nan = RootFit(False, math.nan, math.nan, math.nan, math.nan, math.nan)
    if not (math.isfinite(hist.mean) and math.isfinite(hist.std)):
        return nan
    skew, kurt = root_binned_shape(hist)
    sel = root_fit_range(hist)
    sel = sel[hist.counts[sel] > 0]
    ndf = sel.size - 3
    if ndf <= 0:
        return nan
    x = hist.centers[sel]
    y = hist.counts[sel]
    err = np.sqrt(y)
    p0 = _start_values(x, y, (hist.hi - hist.lo) / hist.nbins)

    def resid(p: FloatArray) -> FloatArray:
        z = (x - p[1]) / p[2]
        out: FloatArray = (y - p[0] * np.exp(-0.5 * z * z)) / err
        return out

    try:
        sol = least_squares(resid, p0, method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    except (ValueError, np.linalg.LinAlgError):
        return nan
    if not sol.success or not np.all(np.isfinite(sol.x)):
        return nan
    chi2 = float(np.sum(sol.fun**2))
    return RootFit(True, float(sol.x[1]), abs(float(sol.x[2])), chi2 / ndf, skew, kurt)


def ml_fit_root_range(hist: RootHist) -> tuple[float, float]:
    """Poisson ML (Baker-Cousins) Gaussian fit on the C++ histogram and fit range.

    Same bins and range as :func:`root_gaus_fit`, but uvcorr's estimator
    (empty bins included). Isolates the estimator (Neyman chi2 vs Poisson ML)
    from the binning and range choices.

    Returns:
        ``(mean, sigma)``; NaNs if the fit fails.
    """
    if not (math.isfinite(hist.mean) and math.isfinite(hist.std)):
        return math.nan, math.nan
    sel = root_fit_range(hist)
    x = hist.centers[sel]
    y = hist.counts[sel]
    if sel.size < 4 or np.count_nonzero(y) < 3:
        return math.nan, math.nan
    p0 = _start_values(x, y, (hist.hi - hist.lo) / hist.nbins)
    p0[2] = math.log(p0[2])

    def resid(p: FloatArray) -> FloatArray:
        z = (x - p[1]) / math.exp(p[2])
        mu = np.maximum(p[0] * np.exp(-0.5 * z * z), 1e-300)
        dev = poisson_deviance(y, mu)
        out: FloatArray = np.sign(y - mu) * np.sqrt(dev)
        return out

    try:
        sol = least_squares(resid, p0, method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    except (ValueError, np.linalg.LinAlgError):
        return math.nan, math.nan
    if not sol.success or not np.all(np.isfinite(sol.x)):
        return math.nan, math.nan
    return float(sol.x[1]), math.exp(float(sol.x[2]))


def root_radial_fit(radii: npt.ArrayLike) -> RootFit:
    """Emulate the C++ radial statistics: a 200-bin ``TH1D`` over [min R, max R], then the fit."""
    r = np.asarray(radii, dtype=np.float64)
    if r.size == 0:
        return RootFit(False, math.nan, math.nan, math.nan, math.nan, math.nan)
    hist = root_hist(r, RADIAL_NUMBINS, float(np.min(r)), float(np.max(r)))
    if hist is None:
        return RootFit(False, math.nan, math.nan, math.nan, math.nan, math.nan)
    return root_gaus_fit(hist)


def cpp_residual_stats(res: npt.ArrayLike) -> tuple[float, float, bool]:
    """Emulate ``residualHistogram``: 150 bins over mean +- 6 std, the fit, else the sample stats.

    Returns:
        ``(mean, sigma, fit_ok)``.
    """
    x = np.asarray(res, dtype=np.float64)
    if x.size == 0:
        return 0.0, 0.0, False
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    if s <= 0:
        s = 1.0
    hist = root_hist(x, RESIDUAL_NUMBINS, m - RESIDUAL_RANGE_STD * s, m + RESIDUAL_RANGE_STD * s)
    if hist is None:
        return m, s, False
    fit = root_gaus_fit(hist)
    if fit.ok:
        return fit.mean, fit.sigma, True
    return m, s, False


def cpp_find_center(u: npt.ArrayLike, v: npt.ArrayLike) -> tuple[float, float] | None:
    """Emulate ``findCenter`` (uvd_common.h): midpoint of the two peaks of U and of V.

    Each coordinate is histogrammed in 100 bins over [min, max] (the maximum
    lands in the overflow bin); the peak is the first highest bin in the lower
    and in the upper half of the bin centres. Returns None where the C++ fails
    (a half without entries).
    """

    def peaks(values: FloatArray) -> float | None:
        lo, hi = float(np.min(values)), float(np.max(values))
        hist = root_hist(values, CENTER_NUMBINS, lo, hi)
        if hist is None:
            return None
        centers = hist.centers
        middle = 0.5 * (centers[-1] + centers[0])
        lower = centers <= middle
        result = []
        for part in (lower, ~lower):
            counts = hist.counts[part]
            if counts.size == 0 or float(np.max(counts)) <= 0:
                return None
            result.append(float(centers[part][int(np.argmax(counts))]))
        return 0.5 * (result[0] + result[1])

    uu = np.asarray(u, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    if uu.size == 0:
        return None
    cu = peaks(uu)
    cv = peaks(vv)
    if cu is None or cv is None:
        return None
    return cu, cv


def _unbinned_shape(values: FloatArray) -> tuple[float, float]:
    """Biased skewness and excess kurtosis (as ``uvcorr.metrics.radial_stats``)."""
    if values.size < 2:
        return math.nan, math.nan
    d = values - np.mean(values)
    m2 = float(np.mean(d * d))
    if not m2 > 0:
        return math.nan, math.nan
    return float(np.mean(d**3)) / m2**1.5, float(np.mean(d**4)) / m2**2 - 3.0


def decompose_channel(u: FloatArray, v: FloatArray, params: EllipseParams) -> dict[str, float]:
    """The ``--decompose`` quantities of one channel (see the module docstring).

    Pre radii, one ingredient changed per step (each key ``<step>_pre_<stat>``):
    ``emul`` (C++ emulation: findCenter centre, 200-bin ROOT chi2 fit),
    ``centre`` (the same fit about the fitted centre), ``ml`` (Poisson ML fit
    on the same C++ histogram and range, fitted centre). The last step to
    uvcorr's value is its binning and range (FD bins, [0.5, 99.5] percentiles,
    iterated mu +- 2 sigma). Post radii: ``emul_post_*`` and ``ml_post_*``, plus
    the shape ``emul`` (binned, maximum in the overflow bin) and
    ``nomax_post_skewness``/``_kurtosis`` (unbinned without the maximum).
    Residuals: ``emul_rawres_*`` and ``emul_corrres_*``. Also
    ``center_offset``, the findCenter-to-fitted-centre distance (ADC). NaN
    when unavailable.
    """
    nan = math.nan
    out: dict[str, float] = {}
    center = cpp_find_center(u, v)
    fitted_r = radii_about_center(u, v, params)
    fitted_hist = root_hist(
        fitted_r, RADIAL_NUMBINS, float(np.min(fitted_r)), float(np.max(fitted_r))
    )
    nofit = RootFit(False, nan, nan, nan, nan, nan)
    centre = root_gaus_fit(fitted_hist) if fitted_hist is not None else nofit
    ml_mean, ml_sigma = ml_fit_root_range(fitted_hist) if fitted_hist is not None else (nan, nan)
    if center is not None:
        cu, cv = center
        out["center_offset"] = math.hypot(cu - params.cx, cv - params.cy)
        du = u - cu
        du[du == 0] = -1e-6  # shiftToOrigin: U == 0 gets 1e-6 * (rand() % 2 - 1), 0 or -1e-6
        emul = root_radial_fit(np.hypot(du, v - cv))
    else:
        out["center_offset"] = nan
        emul = nofit
    for name in ("mean", "sigma", "chi2ndf", "skewness", "kurtosis"):
        out[f"emul_pre_{name}"] = getattr(emul, name)
        out[f"centre_pre_{name}"] = getattr(centre, name)
    out["ml_pre_mean"], out["ml_pre_sigma"] = ml_mean, ml_sigma

    u_corr, v_corr = correct(u, v, params)
    post_r = np.hypot(u_corr, v_corr)
    post_hist = root_hist(post_r, RADIAL_NUMBINS, float(np.min(post_r)), float(np.max(post_r)))
    post = root_gaus_fit(post_hist) if post_hist is not None else nofit
    for name in ("mean", "sigma", "chi2ndf", "skewness", "kurtosis"):
        out[f"emul_post_{name}"] = getattr(post, name)
    out["ml_post_mean"], out["ml_post_sigma"] = (
        ml_fit_root_range(post_hist) if post_hist is not None else (nan, nan)
    )
    nomax = np.delete(post_r, int(np.argmax(post_r))) if post_r.size else post_r
    out["nomax_post_skewness"], out["nomax_post_kurtosis"] = _unbinned_shape(nomax)
    raw_m, raw_s, _ = cpp_residual_stats(residual_to_ellipse(u, v, params))
    corr_m, corr_s, _ = cpp_residual_stats(corrected_residual(u_corr, v_corr, params))
    out["emul_rawres_mean"], out["emul_rawres_sigma"] = raw_m, raw_s
    out["emul_corrres_mean"], out["emul_corrres_sigma"] = corr_m, corr_s
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

# (label, C++ CSV column, uvcorr ChannelResult field, difference kind)
# kind "rel": (uv - cpp) / |cpp|; "abs": uv - cpp; "ratio": uv / cpp.
STAT_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("pre_mean", "pre_mean", "pre_mean", "rel"),
    ("pre_sigma", "pre_sigma", "pre_sigma", "rel"),
    ("pre_fwhm", "pre_fwhm", "pre_fwhm", "rel"),
    ("pre_chi2ndf", "pre_chi2ndf", "pre_chi2ndf", "ratio"),
    ("pre_skewness", "pre_skewness", "pre_skewness", "abs"),
    ("pre_kurtosis", "pre_kurtosis", "pre_kurtosis", "abs"),
    ("post_mean", "ell_mean", "post_mean", "rel"),
    ("post_sigma", "ell_sigma", "post_sigma", "rel"),
    ("post_fwhm", "ell_fwhm", "post_fwhm", "rel"),
    ("post_chi2ndf", "ell_chi2ndf", "post_chi2ndf", "ratio"),
    ("post_skewness", "ell_skewness", "post_skewness", "abs"),
    ("post_kurtosis", "ell_kurtosis", "post_kurtosis", "abs"),
    ("rawfit_res_mean", "rawfit_res_mean", "rawfit_res_mean", "abs"),
    ("rawfit_res_sigma", "rawfit_res_sigma", "rawfit_res_sigma", "rel"),
    ("corr_res_mean", "corr_res_mean", "corr_res_mean", "abs"),
    ("corr_res_sigma", "corr_res_sigma", "corr_res_sigma", "rel"),
    ("phase_mean_gap_rad", "phase_mean_gap_rad", "phase_mean_gap_rad", "rel"),
    ("phase_max_gap_rad", "phase_max_gap_rad", "phase_max_gap_rad", "rel"),
    ("phase_max_gap_ns", "phase_max_gap_ns", "phase_max_gap_ns", "rel"),
    ("phase_ks", "phase_ks", "phase_ks", "abs"),
    ("timing_jitter_ns", "timing_jitter_ns", "timing_jitter_ns", "rel"),
)

# Decomposition steps printed per statistic, in order: (label, --decompose key). Each
# step changes one ingredient from the C++ (emulated) towards uvcorr, whose value is the
# statistic's own line.
_PRE_CHAIN = (
    ("C++ emulation", "emul_pre_{}"),
    ("+ fitted centre", "centre_pre_{}"),
    ("+ Poisson ML (C++ bins, range)", "ml_pre_{}"),
)
_POST_CHAIN = (
    ("C++ emulation", "emul_post_{}"),
    ("+ Poisson ML (C++ bins, range)", "ml_post_{}"),
)
DECOMP_STEPS: dict[str, tuple[tuple[str, str], ...]] = {
    "pre_mean": tuple((label, key.format("mean")) for label, key in _PRE_CHAIN),
    "pre_sigma": tuple((label, key.format("sigma")) for label, key in _PRE_CHAIN),
    "pre_chi2ndf": tuple((label, key.format("chi2ndf")) for label, key in _PRE_CHAIN[:2]),
    "pre_skewness": tuple((label, key.format("skewness")) for label, key in _PRE_CHAIN[:2]),
    "pre_kurtosis": tuple((label, key.format("kurtosis")) for label, key in _PRE_CHAIN[:2]),
    "post_mean": tuple((label, key.format("mean")) for label, key in _POST_CHAIN),
    "post_sigma": tuple((label, key.format("sigma")) for label, key in _POST_CHAIN),
    "post_chi2ndf": (("C++ emulation", "emul_post_chi2ndf"),),
    "post_skewness": (
        ("C++ emulation", "emul_post_skewness"),
        ("unbinned, maximum excluded", "nomax_post_skewness"),
    ),
    "post_kurtosis": (
        ("C++ emulation", "emul_post_kurtosis"),
        ("unbinned, maximum excluded", "nomax_post_kurtosis"),
    ),
    "rawfit_res_mean": (("C++ emulation", "emul_rawres_mean"),),
    "rawfit_res_sigma": (("C++ emulation", "emul_rawres_sigma"),),
    "corr_res_mean": (("C++ emulation", "emul_corrres_mean"),),
    "corr_res_sigma": (("C++ emulation", "emul_corrres_sigma"),),
}

EMULATION_CHECK = ("pre_mean", "pre_sigma", "post_mean", "post_sigma")
"""Statistics whose emulation-vs-C++ agreement is summarised (``--decompose``)."""
EMULATION_RTOL = 1e-3


def diff(kind: str, new: float | None, ref: float | None) -> float | None:
    """Difference of two values of a statistic by kind (``rel``, ``abs`` or ``ratio``)."""
    if new is None or ref is None or not (math.isfinite(new) and math.isfinite(ref)):
        return None
    if kind == "abs":
        return new - ref
    if kind == "ratio":
        return new / ref if ref != 0 else None
    return rel_diff(new, ref)


def cpp_value(row: Mapping[str, Any], column: str) -> float | None:
    """A C++ CSV value; ``timing_jitter_ns`` is derived as RadialAnalysis' summary does."""
    if column == "timing_jitter_ns":
        sigma, mean = row.get("ell_sigma"), row.get("ell_mean")
        if sigma is None or mean is None or mean == 0:
            return None
        return float(sigma) / (2.0 * math.pi * PHASE_REF_FREQ_HZ * float(mean)) * 1e9
    value = row.get(column)
    return None if value is None else float(value)


@dataclass
class ParamCheck:
    """Agreement of one ellipse parameter of one channel."""

    name: str
    cpp: float
    uv: float
    delta: float
    tol: float

    @property
    def ok(self) -> bool:
        return abs(self.delta) <= self.tol

    @property
    def rel(self) -> float:
        return self.delta / abs(self.cpp) if self.cpp != 0 else math.nan


def check_params(
    row: Mapping[str, Any] | None,
    tec: TecEntry | None,
    uv: ChannelResult,
    rtol: float,
) -> list[ParamCheck]:
    """Compare the ellipse of one channel (both sides ok); see the module docstring."""
    checks: list[ParamCheck] = []
    p = uv.params
    if p is None:
        return checks
    if tec is not None:
        for name, cpp, value in (
            ("centerU", tec.params.cx, p.cx),
            ("centerV", tec.params.cy, p.cy),
            ("radius", tec.radius, p.target_radius),
        ):
            tol = printed_half_quantum(cpp) + rtol * abs(cpp)
            checks.append(ParamCheck(name, cpp, value, value - cpp, tol))
    if row is not None and row.get("ell_a") is not None and row.get("ell_b") is not None:
        for name, column, value in (("semiMajor", "ell_a", p.a), ("semiMinor", "ell_b", p.b)):
            cpp = float(row[column])
            tol = CPP_CSV_HALF_QUANTUM + rtol * abs(cpp)
            checks.append(ParamCheck(name, cpp, value, value - cpp, tol))
        if row.get("ell_phi") is not None:
            cpp_phi = float(row["ell_phi"])
            dphi = phi_diff_mod_pi(p.phi, cpp_phi)
            checks.append(ParamCheck("phi", cpp_phi, p.phi, dphi, phi_tolerance(p.a, p.b, rtol)))
    elif tec is not None:  # no CSV ellipse columns: fall back to the 6-digit .tec values
        for name, cpp, value in (
            ("semiMajor", tec.params.a, p.a),
            ("semiMinor", tec.params.b, p.b),
        ):
            tol = printed_half_quantum(cpp) + rtol * abs(cpp)
            checks.append(ParamCheck(name, cpp, value, value - cpp, tol))
        dphi = phi_diff_mod_pi(p.phi, tec.params.phi)
        tol = printed_half_quantum(tec.params.phi) + phi_tolerance(p.a, p.b, rtol)
        checks.append(ParamCheck("phi", tec.params.phi, p.phi, dphi, tol))
    return checks


PARAM_NAMES = ("centerU", "centerV", "semiMajor", "semiMinor", "radius", "phi")
"""The ellipse parameters compared (``radius`` is the ``.tec`` sqrt(ab))."""


@dataclass
class Comparison:
    """Everything compare() found; ``rows`` is the per-channel comparison table.

    Ellipse checks of channels uvcorr flags ``broad_ring`` (pedestal blobs and
    other non-rings) are kept apart in ``blob_checks``/``blob_failures``; they
    are also in ``param_failures`` only when ``strict_blobs`` is set.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    param_failures: list[tuple[ChannelKey, ParamCheck]] = field(default_factory=list)
    count_mismatches: list[tuple[ChannelKey, int | None, int]] = field(default_factory=list)
    param_checks: dict[str, list[ParamCheck]] = field(default_factory=dict)
    blob_checks: dict[str, list[ParamCheck]] = field(default_factory=dict)
    blob_failures: list[tuple[ChannelKey, ParamCheck]] = field(default_factory=list)
    blob_channels: set[ChannelKey] = field(default_factory=set)
    strict_blobs: bool = False


def _status_of(result: ChannelResult | None) -> str:
    return "missing" if result is None else result.status


def compare(
    cpp: CppOutput,
    uv: Mapping[ChannelKey, ChannelResult],
    rtol: float = DEFAULT_RTOL,
    decomposition: Mapping[ChannelKey, Mapping[str, float]] | None = None,
    *,
    strict_blobs: bool = False,
) -> Comparison:
    """Compare the C++ run with the uvcorr results, channel by channel.

    The channel set is the C++ input (``.uvd`` files) plus any channel with a
    C++ row or ``.tec`` block; uvcorr results of other channels are ignored.
    Ellipse disagreements on ``broad_ring`` channels count as failures only
    with ``strict_blobs`` (see :class:`Comparison`).
    """
    result = Comparison(strict_blobs=strict_blobs)
    keys = sorted(set(cpp.inputs) | set(cpp.rows) | set(cpp.tec))
    for key in keys:
        row = cpp.rows.get(key)
        tec = cpp.tec.get(key)
        res = uv.get(key)
        active = is_active_channel(key.rena, key.channel)
        out: dict[str, Any] = {
            "node": key.node,
            "board": key.board,
            "rena": key.rena,
            "channel": key.channel,
            "polarity": polarity_name(key.board, key.rena, key.channel) if active else "",
            "electrode": electrode_label(key.board, key.rena, key.channel) if active else "",
            "cpp_input": key in cpp.inputs,
            "cpp_row": row is not None,
            "cpp_ellipse": row is not None and row.get("ell_a") is not None,
            "cpp_tec": tec is not None,
            "uv_status": _status_of(res),
            "uv_flags": res.flags_text if res is not None else "",
            "uv_broad_ring": res is not None and FLAG_BROAD_RING in res.flags,
            "n_events_cpp": row.get("num_events") if row is not None else None,
            "n_events_uv": res.n_events if res is not None else None,
        }
        if row is not None and res is not None and row.get("num_events") != res.n_events:
            result.count_mismatches.append((key, row.get("num_events"), res.n_events))
        if res is not None and res.ok and (tec is not None or out["cpp_ellipse"]):
            blob = bool(out["uv_broad_ring"])
            if blob:
                result.blob_channels.add(key)
            for check in check_params(row, tec, res, rtol):
                checks = result.blob_checks if blob else result.param_checks
                checks.setdefault(check.name, []).append(check)
                out[f"{check.name}_cpp"] = check.cpp
                out[f"{check.name}_uv"] = check.uv
                out[f"{check.name}_delta"] = check.delta
                out[f"{check.name}_tol"] = check.tol
                out[f"{check.name}_ok"] = check.ok
                if check.name == "phi" and res.semiMajor and res.semiMinor is not None:
                    a, b = res.semiMajor, res.semiMinor
                    out["phi_disp"] = abs(check.delta) * (a * a - b * b) / (a * a)
                if not check.ok:
                    if blob:
                        result.blob_failures.append((key, check))
                    if not blob or strict_blobs:
                        result.param_failures.append((key, check))
        if row is not None and res is not None and res.ok:
            for label, cpp_column, uv_field, kind in STAT_PAIRS:
                c = cpp_value(row, cpp_column)
                u = getattr(res, uv_field)
                out[f"{label}_cpp"] = c
                out[f"{label}_uv"] = u
                out[f"{label}_{kind}"] = diff(kind, u, c)
        if decomposition is not None and key in decomposition:
            out.update({f"dec_{k}": v for k, v in decomposition[key].items()})
        result.rows.append(out)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _quantiles(values: Sequence[float]) -> tuple[float, float, float, float]:
    """(p5, median, p95, max |value|) of finite values; NaNs if none."""
    arr = np.array([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    p5, med, p95 = (float(q) for q in np.percentile(arr, [5, 50, 95]))
    return p5, med, p95, float(np.max(np.abs(arr)))


def write_comparison_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the per-channel comparison table (union of the row keys, first-seen order)."""
    columns: list[str] = []
    for row in rows:
        for name in row:
            if name not in columns:
                columns.append(name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow(_cell(row.get(name)) for name in columns)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return "" if not math.isfinite(value) else format(value, ".9g")
    return str(value)


def _fmt(value: float, kind: str = "rel") -> str:
    if not math.isfinite(value):
        return "-"
    if kind == "ratio":
        return f"{value:.3f}"
    if kind == "rel":
        return f"{value * 100:+.3f}%" if abs(value) >= 1e-4 else f"{value:+.1e}"
    return f"{value:+.4f}"


def print_summary(
    cpp: CppOutput,
    comp: Comparison,
    rtol: float,
    out: Callable[[str], None] = print,
) -> None:
    """Print the concise comparison summary."""
    rows = comp.rows
    n_inputs = len(cpp.inputs)
    n_rows = sum(1 for r in rows if r["cpp_row"])
    n_ell = sum(1 for r in rows if r["cpp_ellipse"])
    n_tec = sum(1 for r in rows if r["cpp_tec"])
    out(f"C++ run: {cpp.uvd_dir}")
    out(
        f"  channels: {n_inputs} .uvd inputs, {n_rows} CSV rows, {n_ell} with an ellipse, "
        f"{n_tec} .tec blocks"
    )
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["uv_status"]] = statuses.get(r["uv_status"], 0) + 1
    out("  uvcorr status of those channels: " + ", ".join(f"{k} {v}" for k, v in statuses.items()))
    skipped = [r for r in rows if r["cpp_input"] and not r["cpp_row"]]
    for r in skipped:
        out(
            f"  C++ skipped node {r['node']} board {r['board']} rena {r['rena']} "
            f"channel {r['channel']} (n={r['n_events_uv']}): uvcorr {r['uv_status']}"
            + (f" [{r['uv_flags']}]" if r["uv_flags"] else "")
        )
    no_ell = [r for r in rows if r["cpp_row"] and not r["cpp_tec"]]
    for r in no_ell:
        out(
            f"  C++ row without .tec block: node {r['node']} board {r['board']} rena {r['rena']} "
            f"channel {r['channel']}: uvcorr {r['uv_status']}"
        )
    cpp_only = [r for r in rows if r["cpp_tec"] and r["uv_status"] != "ok"]
    uv_only = [r for r in rows if r["uv_status"] == "ok" and not r["cpp_tec"]]
    if cpp_only:
        by_status: dict[str, int] = {}
        for r in cpp_only:
            by_status[r["uv_status"]] = by_status.get(r["uv_status"], 0) + 1
        out(
            f"  C++ ellipse but uvcorr not ok: {len(cpp_only)} ("
            + ", ".join(f"{k} {v}" for k, v in by_status.items())
            + ")"
        )
    if uv_only:
        out(f"  uvcorr ok but no C++ ellipse: {len(uv_only)}")
    if comp.count_mismatches:
        out(f"  EVENT COUNT MISMATCH on {len(comp.count_mismatches)} channels, e.g.")
        for key, c, u in comp.count_mismatches[:5]:
            out(f"    {key}: C++ {c}, uvcorr {u}")
    else:
        n_both = sum(1 for r in rows if r["cpp_row"] and r["n_events_uv"] is not None)
        out(f"  event counts: identical on all {n_both} channels with a C++ row")

    out("")
    n_ring = sum(1 for r in rows if _has_param_check(r) and not r.get("uv_broad_ring"))
    out(
        f"Ellipse parameters of the {n_ring} ring channels (both sides ok, not broad_ring; "
        f"tolerance rtol={rtol:g}, see --help):"
    )
    out(
        f"  {'param':<10} {'n':>4} {'median |rel|':>13} {'max |rel|':>11} {'max |d|/tol':>12}  fail"
    )
    ring_rows = [r for r in rows if not r.get("uv_broad_ring")]
    for name in PARAM_NAMES:
        checks = [c for _, c in _all_checks(comp.param_checks) if c.name == name]
        if not checks:
            continue
        if name == "phi":
            rels = [abs(r["phi_disp"]) for r in ring_rows if r.get("phi_disp") is not None]
            label = "phi*"
        else:
            rels = [abs(c.rel) for c in checks]
            label = name
        ratio = max((abs(c.delta) / c.tol for c in checks if c.tol > 0), default=0.0)
        n_fail = sum(1 for c in checks if not c.ok)
        out(
            f"  {label:<10} {len(checks):>4} {float(np.median(rels)):>13.2e} "
            f"{float(np.max(rels)):>11.2e} {ratio:>12.3f}  {n_fail}"
        )
    phi_checks = [c for _, c in _all_checks(comp.param_checks) if c.name == "phi"]
    if phi_checks:
        dphis = [abs(c.delta) for c in phi_checks]
        out(
            f"  * phi: |dphi mod pi| median {float(np.median(dphis)):.2e} rad, max "
            f"{float(np.max(dphis)):.2e} rad; the table shows |dphi| (a^2-b^2)/a^2"
        )
    if comp.blob_channels:
        failing = {key for key, _ in comp.blob_failures}
        axes = [abs(c.rel) for _, c in comp.blob_failures if c.name in ("semiMajor", "semiMinor")]
        centre = [abs(c.delta) for _, c in comp.blob_failures if c.name in ("centerU", "centerV")]
        worst = (
            f"; worst |rel| of a semi-axis {max(axes):.2g}, worst centre |d| {max(centre):.3g} ADC"
            if axes and centre
            else ""
        )
        how = "counted as failures (--strict-blobs)" if comp.strict_blobs else "not failures"
        out(
            f"  broad_ring channels (blobs, non-rings): {len(comp.blob_channels)} compared, "
            f"{len(failing)} disagree beyond tolerance{worst} ({how})"
        )
    if comp.param_failures:
        n_channels = len({key for key, _ in comp.param_failures})
        out(
            f"  DISAGREEMENTS beyond tolerance: {len(comp.param_failures)} on {n_channels} channels"
        )
        for key, check in comp.param_failures[:20]:
            out(
                f"    {key} {check.name}: C++ {check.cpp:.9g}, uvcorr {check.uv:.9g}, "
                f"delta {check.delta:.3g} > tol {check.tol:.3g}"
            )
    else:
        out("  all ring channels within tolerance")

    out("")
    stat_rows = [r for r in rows if "pre_mean_cpp" in r]
    has_dec = any(any(k.startswith("dec_") for k in r) for r in stat_rows)
    faithful = [r for r in stat_rows if _emulation_faithful(r)]
    out(f"Statistics, uvcorr vs C++ ({len(stat_rows)} channels with both; p5 / median / p95).")
    out(
        "  'print': channels with |uvcorr - C++| <= 5e-7, the C++ CSV's print step " "(6 decimals)."
    )
    if has_dec:
        out("  " + _emulation_summary(stat_rows, faithful))
    out(f"  {'statistic':<18} {'kind':<6} {'p5':>10} {'median':>10} {'p95':>10} {'print':>6}")
    for label, _, _, kind in STAT_PAIRS:
        values = [r.get(f"{label}_{kind}") for r in stat_rows]
        p5, med, p95, _ = _quantiles([v for v in values if v is not None])
        n_print = sum(
            1
            for r in stat_rows
            if r.get(f"{label}_uv") is not None
            and r.get(f"{label}_cpp") is not None
            and abs(r[f"{label}_uv"] - r[f"{label}_cpp"]) <= CPP_CSV_HALF_QUANTUM * 1.001
        )
        out(
            f"  {label:<18} {kind:<6} {_fmt(p5, kind):>10} {_fmt(med, kind):>10} "
            f"{_fmt(p95, kind):>10} {n_print:>6}"
        )
        steps = DECOMP_STEPS.get(label, ()) if has_dec else ()
        for step_label, dec_key in steps:
            emul = [diff(kind, r.get(f"dec_{dec_key}"), r.get(f"{label}_cpp")) for r in faithful]
            p5, med, p95, _ = _quantiles([v for v in emul if v is not None])
            out(
                f"    {step_label:<32} {_fmt(p5, kind):>10} {_fmt(med, kind):>10} "
                f"{_fmt(p95, kind):>10}"
            )
        if steps:
            final = [r.get(f"{label}_{kind}") for r in faithful]
            p5, med, p95, _ = _quantiles([v for v in final if v is not None])
            out(
                f"    {'= uvcorr':<32} {_fmt(p5, kind):>10} {_fmt(med, kind):>10} "
                f"{_fmt(p95, kind):>10}"
            )
    if has_dec:
        offsets = [r.get("dec_center_offset") for r in stat_rows]
        p5, med, p95, mx = _quantiles([v for v in offsets if v is not None])
        out(
            f"  findCenter to fitted centre: median {med:.2f} ADC, p95 {p95:.2f} ADC, "
            f"max {mx:.2f} ADC"
        )


def _emulation_summary(
    stat_rows: Sequence[Mapping[str, Any]], faithful: Sequence[Mapping[str, Any]]
) -> str:
    """One line on how well the ``--decompose`` emulation reproduces the C++."""
    others = [r for r in stat_rows if r not in faithful]
    text = (
        f"Decomposition: the Python emulation of the C++ reproduces its pre/post mean and "
        f"sigma within {EMULATION_RTOL:g} on {len(faithful)} of {len(stat_rows)} channels"
    )
    if others:
        failed = [r for r in others if not _emulation_finite(r)]
        counts = [int(r["n_events_uv"]) for r in others if r.get("n_events_uv") is not None]
        span = f", {min(counts)}-{max(counts)} events" if counts else ""
        text += (
            f" (of the other {len(others)}{span}, the emulated fit failed on {len(failed)} and "
            f"gave a different result on {len(others) - len(failed)})"
        )
    return (
        text + ". The indented steps (value vs C++, those channels only) change one "
        "ingredient at a time from the C++ towards uvcorr."
    )


def _emulation_finite(row: Mapping[str, Any]) -> bool:
    """Whether every emulated pre/post mean and sigma of a row is available."""
    for label in EMULATION_CHECK:
        step = "pre" if label.startswith("pre") else "post"
        value = row.get(f"dec_emul_{step}_{label.split('_', 1)[1]}")
        if value is None or not math.isfinite(value):
            return False
    return True


def _has_param_check(row: Mapping[str, Any]) -> bool:
    """Whether a comparison row holds at least one ellipse-parameter check."""
    return any(f"{name}_ok" in row for name in PARAM_NAMES)


def _all_checks(
    checks: Mapping[str, Sequence[ParamCheck]],
) -> list[tuple[str, ParamCheck]]:
    """Flatten ``{name: [check, ...]}`` into ``(name, check)`` pairs."""
    return [(name, check) for name, items in checks.items() for check in items]


def _emulation_faithful(row: Mapping[str, Any]) -> bool:
    """Whether the ``--decompose`` emulation reproduces the C++ pre/post mean and sigma."""
    for label in EMULATION_CHECK:
        step = "pre" if label.startswith("pre") else "post"
        emul = row.get(f"dec_emul_{step}_{label.split('_', 1)[1]}")
        cpp = row.get(f"{label}_cpp")
        d = rel_diff(emul, cpp)
        if d is None or not abs(d) <= EMULATION_RTOL:
            return False
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare RadialAnalysis (C++) output with uvcorr --no-robust.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "cpp_dir", type=Path, help="The .uvd directory given to RadialAnalysis (or its output dir)."
    )
    source = parser.add_argument_group("uvcorr side (at least one)")
    source.add_argument(
        "--uvcorr-dir", type=Path, default=None, help="Output directory of uvcorr process."
    )
    source.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="UV cache: analyse the boards on the fly (unless --uvcorr-dir) and read raw points "
        "for --decompose.",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=FitOptions().min_events,
        help="min_events of the on-the-fly fit (default: the FitOptions default; 6 fits every "
        "channel the C++ fits).",
    )
    parser.add_argument(
        "--tec",
        type=Path,
        default=None,
        help="The C++ .tec file (default: the only *.tec in RadialAnalysis_output).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help=f"Relative tolerance of the ellipse parameters (default: {DEFAULT_RTOL:g}).",
    )
    parser.add_argument(
        "--strict-blobs",
        action="store_true",
        help="Count ellipse disagreements on broad_ring channels (blobs) as failures.",
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Emulate the C++ statistics on the raw points to split the differences (needs --cache).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write the comparison CSV here.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the exit code."""
    args = _build_parser().parse_args(argv)
    if args.uvcorr_dir is None and args.cache is None:
        print("error: give --uvcorr-dir or --cache", file=sys.stderr)
        return 2
    if args.decompose and args.cache is None:
        print("error: --decompose needs --cache (the raw points)", file=sys.stderr)
        return 2
    try:
        cpp = read_cpp_output(args.cpp_dir, args.tec)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    boards = sorted({(k.node, k.board) for k in cpp.inputs | set(cpp.rows)})
    cache = UVCache(args.cache) if args.cache is not None else None
    if args.uvcorr_dir is not None:
        uv = read_uvcorr_dir(args.uvcorr_dir)
        source = f"{args.uvcorr_dir}"
    else:
        assert cache is not None
        options = FitOptions(robust=False, min_events=args.min_events)
        uv = compute_uvcorr(cache, boards, options)
        source = f"{cache.path} on the fly (robust=False, min_events={args.min_events})"
    print(f"uvcorr: {source}")

    decomposition: dict[ChannelKey, dict[str, float]] | None = None
    if args.decompose:
        assert cache is not None
        decomposition = {}
        for node, board in boards:
            data = cache.load_board(node, board)
            for rena, channel, _ in data.channels():
                key = ChannelKey(node, board, rena, channel)
                res = uv.get(key)
                if res is None or not res.ok or res.params is None or key not in cpp.rows:
                    continue
                u, v = data.channel_data(rena, channel)
                decomposition[key] = decompose_channel(
                    u.astype(np.float64), v.astype(np.float64), res.params
                )

    comp = compare(cpp, uv, args.rtol, decomposition, strict_blobs=args.strict_blobs)
    print_summary(cpp, comp, args.rtol)
    if args.out is not None:
        write_comparison_csv(args.out, comp.rows)
        print(f"\nComparison CSV: {args.out} ({len(comp.rows)} rows)")
    return 1 if comp.param_failures or comp.count_mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
