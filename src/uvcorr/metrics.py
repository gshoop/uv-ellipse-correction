"""Radial, residual and phase metrics for pre/post-correction data.

Implements plan sections 4.4 and 5.3:

- ``gauss_fit_histogram`` / ``gauss_stats``: the Gaussian routine shared by the
  radial and residual statistics (method below).
- ``radial_stats``: the Gaussian stats plus unbinned sample, robust and shape
  statistics of a set of radii. ``radial_fit_detail`` returns exactly what it
  fitted (histogram and curve), for plotting; ``radial_stats_and_detail``
  returns both from one fit.
- ``residual_stats``: the Gaussian stats of residuals (the raw residual to the
  ellipse or the corrected residual, see ``uvcorr.ellipse``);
  ``residual_fit_detail`` is its plotting counterpart.
- ``phase_stats``: sorted-phase gap and Kolmogorov-Smirnov uniformity metrics
  of the corrected points (plan 4.4).
- ``timing_jitter_ns``: the radial-width timing jitter proxy.

Gaussian routine:

1. Histogram the finite values inside the [0.5, 99.5] percentiles. The bin
   count is the Freedman-Diaconis one, clamped to 50-400. An optional
   ``min_bin_width`` floor (1 ADC for the radial and residual helpers) stops
   bins from resolving the int16 lattice at large N, which inflates chi2/ndf.
   The floor wins over the 50-bin minimum, but never below 20 bins.
2. Fit ``A exp(-(x - mu)^2 / (2 sigma^2))`` to the bin counts by binned
   Poisson maximum likelihood. This minimises the Baker-Cousins deviance
   ``2 sum(mu_i - n_i + n_i ln(n_i / mu_i))``, which, unlike a Neyman chi2 with
   ``sigma_i = sqrt(n_i)``, does not bias sigma low at small N. The optimiser
   works in standardised coordinates with tight tolerances (1e-12), so the
   result is translation- and scale-equivariant. The first fit covers
   ``median +- 3 * 1.4826 * MAD``. It is then refitted up to 3 more times over
   ``mu +- 2 sigma`` of the previous fit, stopping early when the range
   selects the same bins again.
3. A fit is usable if the optimiser converged, all parameters are finite,
   ``0.5 * bin width <= sigma <= 0.5 * histogram range``, ``mu`` lies inside
   the histogram range, and the range has at least 5 bins of which at least 3
   are non-empty. If a refit is unusable the previous usable fit is kept. If
   no fit is usable the routine reports the sample mean/std with ``ok=False``.
   The shape of the distribution is not a criterion (plan 5.3): flat-topped or
   double-horned radii still get their best Gaussian, and the unbinned
   skewness and kurtosis are reported alongside.
4. ``chi2ndf`` is the Baker-Cousins deviance over the fitted bins divided by
   (bins in range - 3).

None of these functions raise or warn for degenerate data (empty, one value,
all-equal values, NaN, extreme magnitudes). They report ``ok=False`` (or None
where documented) instead. They never touch the process-wide warning filters:
floating-point warnings are silenced with ``np.errstate``, which is
thread-local, so the routines are safe in worker threads (the GUI fits in
QThreads).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

FWHM_FACTOR = 2.35482
"""FWHM of a Gaussian in units of sigma (2 sqrt(2 ln 2)), as in RadialAnalysis."""

MAD_TO_SIGMA = 1.4826
"""Scale factor from the median absolute deviation to a Gaussian sigma."""

MIN_PHASE_POINTS = 8
"""Phase metrics need at least this many points (as in RadialAnalysis)."""

HIST_PERCENTILES = (0.5, 99.5)
"""Values outside these percentiles are left out of the histogram."""

MIN_BINS = 50
MAX_BINS = 400
"""Clamp for the Freedman-Diaconis bin count."""

MIN_BINS_WITH_FLOOR = 20
"""Fewest bins when a ``min_bin_width`` floor pushes the count below ``MIN_BINS``."""

ADC_MIN_BIN_WIDTH = 1.0
"""Bin-width floor used by ``radial_stats`` and ``residual_stats`` (values in ADC units)."""

INITIAL_RANGE_SIGMAS = 3.0
"""The first fit covers ``median +- INITIAL_RANGE_SIGMAS * robust sigma``."""

REFIT_RANGE_SIGMAS = 2.0
REFIT_MAX = 3
"""Up to ``REFIT_MAX`` refits over ``mu +- REFIT_RANGE_SIGMAS * sigma`` of the previous fit."""

MIN_FIT_BINS = 5
"""A fit range needs at least this many bins (ndf = bins - 3 >= 2)."""

MIN_NONEMPTY_FIT_BINS = 3
"""... of which at least this many must be non-empty (a Gaussian has 3 parameters)."""

MIN_SIGMA_OF_BIN = 0.5
"""A fitted sigma below this fraction of the bin width is a spike, not a resolved peak."""

MIN_GAUSS_SAMPLES = 50
"""Fewer values than this skip the histogram fit and report sample statistics."""

MAX_SIGMA_OF_RANGE = 0.5
"""A fitted sigma above this fraction of the histogram range is not a peak (unusable fit).
For a Gaussian the [0.5, 99.5] percentile range is about 5.2 sigma."""

_MIN_RANGE_REL = 1e-9
"""The histogram range must be at least this fraction of the value magnitude."""

_SERIES_U = 1e-4
"""Below this |n/mu - 1| the Poisson deviance uses its Taylor series (no cancellation)."""

_LOG_CLIP = 700.0
"""Exponents are clipped to +-this inside the fit model to stay finite."""

_FIT_TOL = 1e-12
"""ftol, xtol and gtol of the Gaussian fit (in standardised coordinates)."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussStats:
    """Gaussian-fit statistics of a set of values.

    Attributes:
        mean: Fitted mean, or the sample mean if the fit failed (NaN if there
            are no finite values).
        sigma: Fitted sigma, or the sample standard deviation (ddof=1; 0 for a
            single value) if the fit failed.
        fwhm: ``FWHM_FACTOR * sigma``.
        chi2ndf: Baker-Cousins (Poisson likelihood) chi-square per degree of
            freedom, ndf = bins in range - 3; NaN if the fit failed.
        ok: True if a usable Gaussian fit was found.
    """

    mean: float
    sigma: float
    fwhm: float
    chi2ndf: float
    ok: bool


@dataclass(frozen=True)
class RadialStats:
    """Radial distribution statistics: the Gaussian fit plus unbinned statistics.

    Attributes:
        mean: Gaussian mean (sample mean if ``ok`` is False).
        sigma: Gaussian sigma (sample std if ``ok`` is False).
        fwhm: ``FWHM_FACTOR * sigma``.
        chi2ndf: Chi-square per degree of freedom (NaN if ``ok`` is False).
        ok: True if the Gaussian fit succeeded.
        sample_mean: Unbinned mean of the finite values.
        sample_std: Unbinned standard deviation (ddof=1; 0 for one value).
        robust_sigma: ``1.4826 * MAD``.
        skewness: Biased sample skewness, ``m3 / m2**1.5`` (what
            ``scipy.stats.skew`` returns); NaN if undefined.
        kurtosis: Biased excess kurtosis, ``m4 / m2**2 - 3`` (what
            ``scipy.stats.kurtosis(fisher=True)`` returns); NaN if undefined.
    """

    mean: float
    sigma: float
    fwhm: float
    chi2ndf: float
    ok: bool
    sample_mean: float
    sample_std: float
    robust_sigma: float
    skewness: float
    kurtosis: float

    @property
    def gauss(self) -> GaussStats:
        """The Gaussian part as a ``GaussStats``."""
        return GaussStats(self.mean, self.sigma, self.fwhm, self.chi2ndf, self.ok)


@dataclass(frozen=True)
class PhaseStats:
    """Sorted-phase metrics of corrected points (plan 4.4).

    Attributes:
        mean_gap_rad: Mean gap between consecutive phases, ``2 pi / N``.
        max_gap_rad: Largest gap, including the wrap-around gap.
        max_gap_ns: ``max_gap_rad / (2 pi f) * 1e9`` for reference frequency f.
        ks: Kolmogorov-Smirnov D statistic against the uniform phase CDF.
    """

    mean_gap_rad: float
    max_gap_rad: float
    max_gap_ns: float
    ks: float


@dataclass(frozen=True, eq=False)
class GaussFitDetail:
    """Everything the Gaussian routine fitted, for plotting (e.g. the GUI radial tab).

    Attributes:
        stats: The resulting statistics.
        edges: Histogram bin edges (length ``len(counts) + 1``); empty if no
            histogram could be built.
        counts: Histogram counts.
        amplitude: Fitted Gaussian amplitude in counts per bin (NaN if the fit
            failed).
        fit_range: ``(lo, hi)`` range of the accepted fit; bins whose centres
            lie inside it were fitted. ``(nan, nan)`` if the fit failed.
    """

    stats: GaussStats
    edges: FloatArray
    counts: IntArray
    amplitude: float
    fit_range: tuple[float, float]

    @property
    def centers(self) -> FloatArray:
        """Bin centres."""
        centers: FloatArray = 0.5 * (self.edges[:-1] + self.edges[1:])
        return centers

    def curve(self, x: npt.ArrayLike) -> FloatArray:
        """Evaluate the fitted Gaussian (counts per bin) at ``x``; NaN if the fit failed."""
        xx = np.asarray(x, dtype=np.float64)
        if not self.stats.ok:
            return np.full(xx.shape, np.nan)
        return gaussian(xx, self.amplitude, self.stats.mean, self.stats.sigma)


# ---------------------------------------------------------------------------
# Gaussian routine
# ---------------------------------------------------------------------------


def gaussian(x: npt.ArrayLike, amplitude: float, mean: float, sigma: float) -> FloatArray:
    """Evaluate ``amplitude * exp(-(x - mean)^2 / (2 sigma^2))``.

    Args:
        x: Abscissae.
        amplitude: Peak height.
        mean: Centre.
        sigma: Standard deviation (its sign is irrelevant).

    Returns:
        The Gaussian values.
    """
    xx = np.asarray(x, dtype=np.float64)
    with np.errstate(all="ignore"):
        z = (xx - mean) / sigma
        out: FloatArray = amplitude * np.exp(-0.5 * z * z)
    return out


def poisson_deviance(counts: npt.ArrayLike, expected: npt.ArrayLike) -> FloatArray:
    """Per-bin Baker-Cousins Poisson deviance ``2 (mu - n + n ln(n / mu))``.

    The term is ``2 mu`` for an empty bin. Near ``n = mu`` it is evaluated by
    its Taylor series in ``u = n / mu - 1``, so it stays accurate and never
    negative.

    Args:
        counts: Observed counts ``n`` (>= 0).
        expected: Expected counts ``mu`` (> 0).

    Returns:
        The deviance terms; their sum is the likelihood-ratio chi-square.
    """
    n = np.asarray(counts, dtype=np.float64)
    mu = np.asarray(expected, dtype=np.float64)
    with np.errstate(all="ignore"):
        dev, _ = _deviance_and_slope(n, mu)
    return dev


def _deviance_and_slope(n: FloatArray, mu: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Deviance terms ``D`` and ``g = |n - mu| / sqrt(mu D)`` (g -> 1 as n -> mu).

    ``g`` gives the derivative of the signed root deviance
    ``r = sign(n - mu) sqrt(D)``: ``dr/dmu = -g / sqrt(mu)``.
    """
    u = (n - mu) / mu
    small = np.abs(u) < _SERIES_U
    # h(u) = (1 + u) ln(1 + u) - u = u^2/2 - u^3/6 + u^4/12 - ...
    h_series = u * u * (0.5 - u / 6.0 + u * u / 12.0)
    # n ln(n / mu) = n log1p(u): no rounding of n / mu inside the log (0 for n = 0).
    n_log = n * np.log1p(np.where(n > 0.0, u, 0.0))
    dev = np.where(small, 2.0 * mu * h_series, 2.0 * (mu - n + n_log))
    dev = np.maximum(dev, 0.0)
    g_general = np.abs(n - mu) / np.sqrt(mu * dev)
    g_series = 1.0 / np.sqrt(1.0 - u / 3.0 + u * u / 6.0)
    g = np.where(small, g_series, g_general)
    g = np.where(np.isfinite(g), g, 0.0)
    return dev, g


def _fit_gaussian(
    x: FloatArray, counts: FloatArray, p0: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    """Binned Poisson maximum-likelihood fit of a Gaussian to ``counts`` at centres ``x``.

    Minimises the Baker-Cousins deviance with ``scipy.optimize.least_squares``
    (Levenberg-Marquardt, analytic Jacobian, tolerances 1e-12) on the signed
    root-deviance residuals. It works in coordinates standardised by the start
    values, ``z = (x - mean0) / sigma0``, with parameters
    ``(ln A, (mu - mean0) / sigma0, ln(sigma / sigma0))``: all of order 1, so
    the tolerances do not depend on the offset or scale of the data, and the
    amplitude and width stay positive without bounds.

    Args:
        x: Bin centres of the fit range.
        counts: Bin counts (float).
        p0: Starting ``(amplitude, mean, sigma)``, amplitude and sigma > 0.

    Returns:
        ``(amplitude, mean, sigma)``, or None if the optimiser fails.
    """
    amp0, mu0, sig0 = p0
    if not (amp0 > 0.0 and sig0 > 0.0 and math.isfinite(mu0) and math.isfinite(sig0)):
        return None
    xs = (x - mu0) / sig0
    theta0 = np.array([math.log(amp0), 0.0, 0.0], dtype=np.float64)

    def model(theta: FloatArray) -> tuple[FloatArray, FloatArray, float]:
        log_amp, mean, log_sig = (float(t) for t in theta)
        sig = math.exp(min(max(log_sig, -_LOG_CLIP), _LOG_CLIP))
        z = (xs - mean) / sig
        expected = np.exp(np.clip(log_amp - 0.5 * z * z, -_LOG_CLIP, _LOG_CLIP))
        return expected, z, sig

    def fun(theta: FloatArray) -> FloatArray:
        expected, _, _ = model(theta)
        dev, _ = _deviance_and_slope(counts, expected)
        res: FloatArray = np.sign(counts - expected) * np.sqrt(dev)
        return res

    def jac(theta: FloatArray) -> FloatArray:
        expected, z, sig = model(theta)
        _, g = _deviance_and_slope(counts, expected)
        scale = -g * np.sqrt(expected)  # dr/dmu_i * mu_i
        return np.column_stack((scale, scale * z / sig, scale * z * z))

    try:
        with np.errstate(all="ignore"):
            sol = least_squares(
                fun, theta0, jac=jac, method="lm", ftol=_FIT_TOL, xtol=_FIT_TOL, gtol=_FIT_TOL
            )
    except (ValueError, KeyError, np.linalg.LinAlgError):
        return None
    theta = np.asarray(sol.x, dtype=np.float64)
    if not bool(sol.success) or not np.all(np.isfinite(theta)):
        return None
    log_amp, mean, log_sig = (float(t) for t in theta)
    if abs(log_amp) >= _LOG_CLIP or abs(log_sig) >= _LOG_CLIP:
        return None
    return math.exp(log_amp), mu0 + sig0 * mean, sig0 * math.exp(log_sig)


def _finite_values(values: npt.ArrayLike) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = np.isfinite(arr)
    return arr if bool(np.all(finite)) else arr[finite]


def _sample_mean_std(vals: FloatArray) -> tuple[float, float]:
    """Sample mean and std (ddof=1; 0 for one value; NaN for none)."""
    if vals.size == 0:
        return math.nan, math.nan
    with np.errstate(all="ignore"):
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return mean, std


def _median_and_robust_sigma(vals: FloatArray) -> tuple[float, float]:
    if vals.size == 0:
        return math.nan, math.nan
    with np.errstate(all="ignore"):
        med = float(np.median(vals))
        return med, MAD_TO_SIGMA * float(np.median(np.abs(vals - med)))


def _shape_stats(vals: FloatArray) -> tuple[float, float]:
    """Biased skewness ``m3 / m2**1.5`` and excess kurtosis ``m4 / m2**2 - 3``.

    Central moments about the sample mean, as ``scipy.stats.skew`` and
    ``scipy.stats.kurtosis(fisher=True)`` compute them. NaN when undefined
    (fewer than 2 values, zero spread, or overflow).
    """
    if vals.size < 2:
        return math.nan, math.nan
    with np.errstate(all="ignore"):
        d = vals - np.mean(vals)
        d2 = d * d
        m2 = np.mean(d2)
        skew = float(np.mean(d2 * d) / (m2 * np.sqrt(m2)))
        kurt = float(np.mean(d2 * d2) / (m2 * m2) - 3.0)
    if not m2 > 0.0:
        return math.nan, math.nan
    return (
        skew if math.isfinite(skew) else math.nan,
        kurt if math.isfinite(kurt) else math.nan,
    )


def _fallback(mean: float, std: float) -> GaussStats:
    return GaussStats(mean=mean, sigma=std, fwhm=FWHM_FACTOR * std, chi2ndf=math.nan, ok=False)


def _bin_count(
    inside: FloatArray, lo: float, hi: float, iqr: float, min_bin_width: float | None
) -> int:
    """Freedman-Diaconis bin count, clamped, with the optional width floor."""
    width_range = hi - lo
    with np.errstate(all="ignore"):
        fd_width = 2.0 * iqr / inside.size ** (1.0 / 3.0)
        ratio = width_range / fd_width if fd_width > 0.0 else math.inf
    n_bins = (
        int(np.clip(math.ceil(ratio), MIN_BINS, MAX_BINS)) if math.isfinite(ratio) else MIN_BINS
    )
    if min_bin_width is not None and min_bin_width > 0.0:
        max_bins = width_range / min_bin_width
        if math.isfinite(max_bins) and n_bins > max_bins:
            n_bins = max(int(max_bins), MIN_BINS_WITH_FLOOR)
    return n_bins


def _gauss_fit(
    vals: FloatArray,
    median: float,
    robust_sigma: float,
    sample_mean: float,
    sample_std: float,
    min_bin_width: float | None,
) -> GaussFitDetail:
    """Gaussian routine on finite values with precomputed robust/sample statistics."""
    fallback = _fallback(sample_mean, sample_std)
    no_hist = GaussFitDetail(
        fallback,
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.int64),
        math.nan,
        (math.nan, math.nan),
    )
    if vals.size < MIN_GAUSS_SAMPLES:
        return no_hist

    with np.errstate(all="ignore"):
        lo, q25, q75, hi = (
            float(q)
            for q in np.percentile(vals, [HIST_PERCENTILES[0], 25.0, 75.0, HIST_PERCENTILES[1]])
        )
        width_range = hi - lo
        magnitude = max(abs(lo), abs(hi))
    if not (math.isfinite(width_range) and width_range > _MIN_RANGE_REL * magnitude):
        return no_hist  # all equal, or not resolvable in float64
    inside = vals[(vals >= lo) & (vals <= hi)]
    n_bins = _bin_count(inside, lo, hi, q75 - q25, min_bin_width)
    if not math.isfinite(n_bins / width_range):
        return no_hist
    counts_raw, edges_raw = np.histogram(inside, bins=n_bins, range=(lo, hi))
    counts = counts_raw.astype(np.int64)
    edges = edges_raw.astype(np.float64)
    no_fit = GaussFitDetail(fallback, edges, counts, math.nan, (math.nan, math.nan))

    scale = robust_sigma if robust_sigma > 0.0 else sample_std
    if not (math.isfinite(scale) and scale > 0.0 and math.isfinite(median)):
        return no_fit

    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = width_range / n_bins
    y_all = counts.astype(np.float64)
    fit_lo = median - INITIAL_RANGE_SIGMAS * scale
    fit_hi = median + INITIAL_RANGE_SIGMAS * scale
    best: tuple[float, float, float, float, tuple[float, float]] | None = None
    p0: tuple[float, float, float] | None = None
    prev_sel: npt.NDArray[np.bool_] | None = None
    with np.errstate(all="ignore"):
        for _ in range(1 + REFIT_MAX):
            sel = (centers >= fit_lo) & (centers <= fit_hi)
            if prev_sel is not None and np.array_equal(sel, prev_sel):
                break
            n_sel = int(np.count_nonzero(sel))
            if n_sel < MIN_FIT_BINS:
                break
            x, y = centers[sel], y_all[sel]
            if int(np.count_nonzero(y)) < MIN_NONEMPTY_FIT_BINS:
                break
            if p0 is None:
                p0 = (max(float(np.max(y)), 1.0), median, scale)
            popt = _fit_gaussian(x, y, p0)
            if popt is None:
                break
            amp, mu, sig = popt
            if not (
                math.isfinite(amp)
                and math.isfinite(mu)
                and math.isfinite(sig)
                and amp > 0.0
                and MIN_SIGMA_OF_BIN * bin_width <= sig <= MAX_SIGMA_OF_RANGE * width_range
                and lo <= mu <= hi
            ):
                break
            chi2 = float(np.sum(poisson_deviance(y, gaussian(x, amp, mu, sig))))
            if not math.isfinite(chi2):
                break
            best = (amp, mu, sig, chi2 / (n_sel - 3), (fit_lo, fit_hi))
            prev_sel = sel
            p0 = (amp, mu, sig)
            fit_lo = mu - REFIT_RANGE_SIGMAS * sig
            fit_hi = mu + REFIT_RANGE_SIGMAS * sig

    if best is None:
        return no_fit
    amp, mu, sig, chi2ndf, fit_range = best
    gstats = GaussStats(mean=mu, sigma=sig, fwhm=FWHM_FACTOR * sig, chi2ndf=chi2ndf, ok=True)
    return GaussFitDetail(gstats, edges, counts, amp, fit_range)


def gauss_fit_histogram(
    values: npt.ArrayLike, min_bin_width: float | None = None
) -> GaussFitDetail:
    """Run the Gaussian routine and return the histogram and fitted curve with the stats.

    Non-finite values are ignored. See the module docstring for the method.

    Args:
        values: The values (e.g. radii or residuals).
        min_bin_width: Optional lower limit on the bin width (in the units of
            ``values``). None means no floor. See the module docstring for how
            it interacts with the bin-count clamp.

    Returns:
        The histogram, the fitted curve parameters and the statistics. Never
        raises or warns.
    """
    vals = _finite_values(values)
    median, robust_sigma = _median_and_robust_sigma(vals)
    sample_mean, sample_std = _sample_mean_std(vals)
    return _gauss_fit(vals, median, robust_sigma, sample_mean, sample_std, min_bin_width)


def gauss_stats(values: npt.ArrayLike, min_bin_width: float | None = None) -> GaussStats:
    """Gaussian statistics of ``values`` (``gauss_fit_histogram(values, ...).stats``).

    Args:
        values: The values.
        min_bin_width: Optional bin-width floor, as in ``gauss_fit_histogram``.

    Returns:
        The statistics; ``ok=False`` with the sample mean/std if no usable
        fit was found.
    """
    return gauss_fit_histogram(values, min_bin_width).stats


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------


def radial_fit_detail(
    r: npt.ArrayLike, min_bin_width: float | None = ADC_MIN_BIN_WIDTH
) -> GaussFitDetail:
    """The Gaussian fit behind ``radial_stats``, with its histogram, for plotting.

    ``radial_stats`` calls this function, so the plotted histogram and curve are
    exactly what was fitted.

    Args:
        r: Radii. Non-finite values are ignored.
        min_bin_width: Bin-width floor; the default of 1 ADC suits int16 U/V data.

    Returns:
        The histogram, fitted curve and Gaussian statistics.
    """
    return gauss_fit_histogram(r, min_bin_width)


def residual_fit_detail(
    res: npt.ArrayLike, min_bin_width: float | None = ADC_MIN_BIN_WIDTH
) -> GaussFitDetail:
    """The Gaussian fit behind ``residual_stats``, with its histogram, for plotting.

    Args:
        res: Residuals. Non-finite values are ignored.
        min_bin_width: Bin-width floor; the default of 1 ADC suits int16 U/V data.

    Returns:
        The histogram, fitted curve and Gaussian statistics.
    """
    return gauss_fit_histogram(res, min_bin_width)


def radial_stats_and_detail(
    r: npt.ArrayLike, min_bin_width: float | None = ADC_MIN_BIN_WIDTH
) -> tuple[RadialStats, GaussFitDetail]:
    """``radial_stats`` and the ``radial_fit_detail`` behind it, from one Gaussian fit.

    ``radial_stats`` is this function's first element, so a plot of the
    returned histogram and curve always shows exactly the fit whose numbers
    are reported (the GUI radial tab uses it to fit only once).

    Args:
        r: Radii. Non-finite values are ignored.
        min_bin_width: Bin-width floor for the Gaussian routine; the default
            of 1 ADC suits int16 U/V data.

    Returns:
        ``(stats, detail)`` with ``stats.gauss == detail.stats``. Never raises
        or warns.
    """
    vals = _finite_values(r)
    _, robust_sigma = _median_and_robust_sigma(vals)
    sample_mean, sample_std = _sample_mean_std(vals)
    skew, kurt = _shape_stats(vals)
    detail = radial_fit_detail(vals, min_bin_width)
    g = detail.stats
    stats = RadialStats(
        mean=g.mean,
        sigma=g.sigma,
        fwhm=g.fwhm,
        chi2ndf=g.chi2ndf,
        ok=g.ok,
        sample_mean=sample_mean,
        sample_std=sample_std,
        robust_sigma=robust_sigma,
        skewness=skew,
        kurtosis=kurt,
    )
    return stats, detail


def radial_stats(r: npt.ArrayLike, min_bin_width: float | None = ADC_MIN_BIN_WIDTH) -> RadialStats:
    """Statistics of a set of radii (pre- or post-correction, plan 5.3).

    Args:
        r: Radii. Non-finite values are ignored.
        min_bin_width: Bin-width floor for the Gaussian routine; the default
            of 1 ADC suits int16 U/V data.

    Returns:
        The Gaussian stats (from ``radial_fit_detail``) plus the unbinned
        sample mean/std, robust sigma, skewness and excess kurtosis. Never
        raises or warns. (:func:`radial_stats_and_detail` also returns the
        fitted histogram.)
    """
    return radial_stats_and_detail(r, min_bin_width)[0]


def residual_stats(
    res: npt.ArrayLike, min_bin_width: float | None = ADC_MIN_BIN_WIDTH
) -> GaussStats:
    """Gaussian statistics of residuals (plan 4.4 definitions, plan 5.3 routine).

    Use it on ``uvcorr.ellipse.residual_to_ellipse`` (the raw residual to the
    fitted ellipse) or ``uvcorr.ellipse.corrected_residual``
    (``sqrt(U'^2 + V'^2) - sqrt(ab)``).

    Args:
        res: Residuals. Non-finite values are ignored.
        min_bin_width: Bin-width floor for the Gaussian routine; the default
            of 1 ADC suits int16 U/V data.

    Returns:
        The statistics (sample mean/std with ``ok=False`` if the fit failed),
        i.e. ``residual_fit_detail(res, min_bin_width).stats``.
    """
    return residual_fit_detail(res, min_bin_width).stats


def _check_ref_freq(ref_freq_hz: float) -> None:
    if not (math.isfinite(ref_freq_hz) and ref_freq_hz > 0.0):
        raise ValueError(f"ref_freq_hz must be finite and > 0, got {ref_freq_hz}")


_TWO_PI = 2.0 * math.pi
_BELOW_TWO_PI = math.nextafter(_TWO_PI, 0.0)


def phase_stats(
    u_corr: npt.ArrayLike, v_corr: npt.ArrayLike, ref_freq_hz: float = 490e3
) -> PhaseStats | None:
    """Sorted-phase gap and uniformity metrics of corrected points (plan 4.4).

    The phases ``atan2(V', U')`` are wrapped to [0, 2 pi) and sorted. A tiny
    negative angle that rounds up to 2 pi on wrapping becomes the largest float
    below 2 pi, so it stays at the top of the CDF. The gaps are the consecutive
    differences plus the wrap-around gap ``ph[0] + 2 pi - ph[N-1]``. The KS
    statistic is ``max_j max((j+1)/N - F_j, F_j - j/N)`` with
    ``F_j = ph[j] / (2 pi)``.

    Args:
        u_corr: Corrected U' (centred on the origin).
        v_corr: Corrected V'.
        ref_freq_hz: Reference frequency converting the max gap to ns.

    Returns:
        The phase metrics, or None if there are fewer than
        ``MIN_PHASE_POINTS`` finite points.

    Raises:
        ValueError: If ``ref_freq_hz`` is not finite and positive.
    """
    _check_ref_freq(ref_freq_hz)
    x = np.asarray(u_corr, dtype=np.float64).ravel()
    y = np.asarray(v_corr, dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    if not bool(np.all(finite)):
        x, y = x[finite], y[finite]
    n = int(x.size)
    if n < MIN_PHASE_POINTS:
        return None
    ph = np.arctan2(y, x)
    ph = np.where(ph < 0.0, ph + _TWO_PI, ph)
    ph = np.minimum(ph, _BELOW_TWO_PI)
    ph.sort()
    max_gap = max(float(np.max(np.diff(ph))), float(ph[0] + _TWO_PI - ph[-1]))
    cdf = ph / _TWO_PI
    j = np.arange(n, dtype=np.float64)
    ks = max(float(np.max((j + 1.0) / n - cdf)), float(np.max(cdf - j / n)))
    return PhaseStats(
        mean_gap_rad=_TWO_PI / n,
        max_gap_rad=max_gap,
        max_gap_ns=max_gap / (_TWO_PI * ref_freq_hz) * 1e9,
        ks=ks,
    )


def timing_jitter_ns(post_sigma: float, post_mean: float, ref_freq_hz: float = 490e3) -> float:
    """Timing-jitter proxy ``sigma / (2 pi f mean) * 1e9`` in ns (plan 5.3).

    Args:
        post_sigma: Post-correction radial sigma.
        post_mean: Post-correction mean radius.
        ref_freq_hz: Reference frequency f in Hz.

    Returns:
        The jitter in ns; NaN if the inputs are non-finite or ``post_mean`` is 0.

    Raises:
        ValueError: If ``ref_freq_hz`` is not finite and positive.
    """
    _check_ref_freq(ref_freq_hz)
    if not (math.isfinite(post_sigma) and math.isfinite(post_mean)) or post_mean == 0.0:
        return math.nan
    return post_sigma / (2.0 * math.pi * ref_freq_hz * post_mean) * 1e9
