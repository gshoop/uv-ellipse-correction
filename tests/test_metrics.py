"""Tests for uvcorr.metrics (plan sections 4.4, 5.3 and 10.2).

Tolerances for the Gaussian routine (binned Poisson maximum likelihood) on pure
Gaussian samples: the fitted mean and sigma must be within ``6 * sigma / sqrt(N)``
of the truth. In a Monte Carlo (2000 seeds per N at N <= 1000, 400 above) the
mean of sigma_fit / sigma was 0.9998 at N = 200, 0.9986 at N = 1000 and
1.0004 at N = 20000; the old Neyman fit gave 0.902 and 0.973. Single-fit
z-scores (units of ``sigma / sqrt(N)``) have a standard deviation of about 1.3,
because the fit uses only the central +-2 sigma. chi2/ndf (Baker-Cousins,
ndf ~ 35-90) averaged 1.02-1.08 at N <= 1000 and 0.97-1.02 above.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats as sp_stats
from scipy.optimize import minimize

from uvcorr import metrics
from uvcorr.ellipse import EllipseParams, correct, ellipse_points, fit_ellipse, radii_about_center
from uvcorr.metrics import (
    FWHM_FACTOR,
    MAD_TO_SIGMA,
    GaussStats,
    PhaseStats,
    gauss_fit_histogram,
    gauss_stats,
    phase_stats,
    poisson_deviance,
    radial_stats,
    residual_stats,
    timing_jitter_ns,
)

# Any warning (numpy RuntimeWarning, scipy OptimizeWarning, ...) fails the test.
pytestmark = pytest.mark.filterwarnings("error")

TWO_PI = 2.0 * math.pi
FloatArray = npt.NDArray[np.float64]


def _ring(
    ratio: float, n: int, *, seed: int = 0, noise: float = 5.0, int16: bool = True
) -> tuple[FloatArray, FloatArray, EllipseParams]:
    """A synthetic channel: ellipse a = 430, b = ratio * a, Gaussian radial noise."""
    rng = np.random.default_rng(seed)
    p = EllipseParams(2000.3, 2000.7, 430.0, 430.0 * ratio, 0.4)
    x, y = ellipse_points(p, rng.uniform(0.0, TWO_PI, n))
    dx, dy = x - p.cx, y - p.cy
    k = 1.0 + rng.normal(0.0, noise, n) / np.hypot(dx, dy)
    u, v = p.cx + dx * k, p.cy + dy * k
    if int16:
        u, v = np.round(u), np.round(v)
    return u, v, p


# ---------------------------------------------------------------------------
# Gaussian routine
# ---------------------------------------------------------------------------


class TestGaussRoutine:
    @pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
    def test_pure_gaussian(self, n: int) -> None:
        mu, sigma = 600.0, 5.0
        x = np.random.default_rng(n).normal(mu, sigma, n)
        g = gauss_stats(x)
        assert g.ok
        tol = 6.0 * sigma / math.sqrt(n)
        assert g.mean == pytest.approx(mu, abs=tol)
        assert g.sigma == pytest.approx(sigma, abs=tol)
        assert g.fwhm == pytest.approx(FWHM_FACTOR * g.sigma, rel=1e-15)
        assert 0.4 < g.chi2ndf < 2.0

    @pytest.mark.parametrize(("n", "seeds"), [(200, 300), (1000, 150)])
    def test_small_n_sigma_unbiased(self, n: int, seeds: int) -> None:
        # The Neyman chi2 (sigma_i = sqrt(n_i)) read sigma 10 % low at N = 200 and
        # 2.7 % low at N = 1000; the Poisson likelihood fit must be within 2 %.
        ratios, ok = [], 0
        for s in range(seeds):
            g = gauss_stats(np.random.default_rng(7000 + s).normal(430.0, 7.5, n))
            if g.ok:
                ok += 1
                ratios.append(g.sigma / 7.5)
        assert ok >= 0.97 * seeds
        assert np.mean(ratios) == pytest.approx(1.0, abs=0.02)

    def test_chi2ndf_near_one_on_average(self) -> None:
        values = [
            gauss_stats(np.random.default_rng(100 + s).normal(0.0, 1.0, 10_000)).chi2ndf
            for s in range(20)
        ]
        # Single-fit chi2/ndf scatter is ~sqrt(2/ndf) ~ 0.2; the mean of 20 is good to ~0.05.
        assert np.mean(values) == pytest.approx(1.0, abs=0.15)

    def test_chi2ndf_is_baker_cousins_over_fit_bins(self) -> None:
        x = np.random.default_rng(1).normal(600.0, 5.0, 30_000)
        d = gauss_fit_histogram(x)
        lo, hi = d.fit_range
        sel = (d.centers >= lo) & (d.centers <= hi)
        n_sel = int(sel.sum())
        dev = poisson_deviance(d.counts[sel], d.curve(d.centers[sel]))
        assert d.stats.chi2ndf == pytest.approx(float(dev.sum()) / (n_sel - 3), rel=1e-9)

    def test_poisson_likelihood_conserves_counts(self) -> None:
        # With a free amplitude, the Poisson ML optimum has sum(mu_i) = sum(n_i) over the
        # fitted bins (d deviance / d ln A = 2 sum(mu_i - n_i) = 0). A Neyman or Pearson
        # chi2 fit does not satisfy this.
        x = np.random.default_rng(2).normal(0.0, 1.0, 800)
        d = gauss_fit_histogram(x)
        assert d.stats.ok
        sel = (d.centers >= d.fit_range[0]) & (d.centers <= d.fit_range[1])
        assert float(d.curve(d.centers[sel]).sum()) == pytest.approx(
            float(d.counts[sel].sum()), rel=1e-6
        )

    def test_fit_matches_direct_deviance_minimisation(self) -> None:
        rng = np.random.default_rng(3)
        centers = np.linspace(-3.0, 3.0, 40)
        counts = rng.poisson(30.0 * np.exp(-0.5 * (centers / 1.2) ** 2)).astype(np.float64)
        got = metrics._fit_gaussian(centers, counts, (25.0, 0.1, 1.0))
        assert got is not None

        def total(theta: FloatArray) -> float:
            return float(poisson_deviance(counts, metrics.gaussian(centers, *theta)).sum())

        ref = minimize(
            total,
            x0=[25.0, 0.1, 1.0],
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000},
        )
        np.testing.assert_allclose(got, ref.x, rtol=1e-5, atol=1e-6)
        assert total(np.array(got)) <= ref.fun + 1e-9

    @pytest.mark.parametrize("p0", [(25.0, 0.1, 1.0), (40.0, -0.5, 2.0), (10.0, 0.4, 0.6)])
    def test_fit_reaches_the_likelihood_optimum(self, p0: tuple[float, float, float]) -> None:
        # At the Poisson ML optimum the deviance gradient vanishes. With the 1e-12
        # tolerances |grad| ~ 4e-6 (deviance ~ 39); scipy's default tolerances (or 1e-8)
        # stop ~1e-5 short of the optimum and leave |grad| ~ 1e-3.
        rng = np.random.default_rng(3)
        centers = np.linspace(-3.0, 3.0, 40)
        counts = rng.poisson(30.0 * np.exp(-0.5 * (centers / 1.2) ** 2)).astype(np.float64)
        got = metrics._fit_gaussian(centers, counts, p0)
        assert got is not None
        amp, mu, sig = got
        z = (centers - mu) / sig
        expected = amp * np.exp(-0.5 * z * z)
        w = 2.0 * (1.0 - counts / expected) * expected
        grad = np.array([np.sum(w), np.sum(w * z / sig), np.sum(w * z * z)])  # ln A, mu, ln sig
        assert np.max(np.abs(grad)) < 3e-5

    def test_range_iteration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # First fit over median +- 3 robust sigma, then over mu +- 2 sigma of the previous
        # fit; the reported fit_range is that of the last accepted fit.
        real = metrics._fit_gaussian
        log: list[tuple[FloatArray, tuple[float, float, float] | None]] = []

        def logging_fit(x: FloatArray, counts: FloatArray, p0: Any) -> Any:
            out = real(x, counts, p0)
            log.append((x.copy(), out))
            return out

        monkeypatch.setattr(metrics, "_fit_gaussian", logging_fit)
        vals = np.random.default_rng(4).normal(600.0, 5.0, 100_000)
        d = gauss_fit_histogram(vals)
        centers = d.centers
        med = float(np.median(vals))
        rsig = MAD_TO_SIGMA * float(np.median(np.abs(vals - med)))
        assert 2 <= len(log) <= 4
        first = (centers >= med - 3.0 * rsig) & (centers <= med + 3.0 * rsig)
        np.testing.assert_array_equal(log[0][0], centers[first])
        for (_, prev), (x, _) in zip(log[:-1], log[1:]):
            assert prev is not None
            _, mu, sig = prev
            expected = (centers >= mu - 2.0 * sig) & (centers <= mu + 2.0 * sig)
            np.testing.assert_array_equal(x, centers[expected])
        last = log[-1][1]
        assert last is not None
        amp, mu, sig = last
        assert (d.amplitude, d.stats.mean, d.stats.sigma) == (amp, mu, sig)
        if len(log) >= 2:
            prev = log[-2][1]
            assert prev is not None
            assert d.fit_range == (prev[1] - 2.0 * prev[2], prev[1] + 2.0 * prev[2])

    def test_refit_count_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A fit whose mean drifts every time never converges: exactly 1 + REFIT_MAX fits.
        real = metrics._fit_gaussian
        calls: list[int] = []

        def drifting_fit(x: FloatArray, counts: FloatArray, p0: Any) -> Any:
            out = real(x, counts, p0)
            calls.append(1)
            assert out is not None
            return out[0], out[1] + 0.3 * len(calls) * out[2], out[2]

        monkeypatch.setattr(metrics, "_fit_gaussian", drifting_fit)
        d = gauss_fit_histogram(np.random.default_rng(5).normal(0.0, 1.0, 50_000))
        assert d.stats.ok
        assert len(calls) == 4  # 1 initial fit + REFIT_MAX = 3 refits (plan 5.3)

    @pytest.mark.parametrize("n", [200, 2_000, 20_000])
    def test_shift_and_scale_equivariant(self, n: int) -> None:
        # The fit runs in standardised coordinates with 1e-12 tolerances, so an offset of
        # 1e6 or a rescaling changes nothing beyond round-off (the default LM tolerances
        # left sigma 4.5e-4 off at offset 1e6). What remains is where the optimiser stops
        # inside its 1e-12 tolerance: over 450 seeds the worst differences were 6e-9 (sigma)
        # and 1.1e-8 (mean), medians ~1e-11.
        x = np.random.default_rng(n).normal(0.0, 1.0, n)
        g0 = gauss_stats(x)
        assert g0.ok
        tol = 5e-8
        for offset, scale in ((1e6, 1.0), (600.0, 5.0), (0.0, 1e-3)):
            g1 = gauss_stats(offset + scale * x)
            assert g1.ok
            assert g1.sigma == pytest.approx(scale * g0.sigma, rel=tol)
            assert abs((g1.mean - offset) / scale - g0.mean) < tol * g0.sigma
            assert g1.chi2ndf == pytest.approx(g0.chi2ndf, rel=1e-6)

    def test_robust_to_background(self) -> None:
        # A 5 % flat background barely moves a fit restricted to the core.
        rng = np.random.default_rng(2)
        x = np.r_[rng.normal(600.0, 5.0, 50_000), rng.uniform(400.0, 800.0, 2_500)]
        g = gauss_stats(x)
        assert g.ok
        assert g.mean == pytest.approx(600.0, abs=0.2)
        assert g.sigma == pytest.approx(5.0, rel=0.05)

    def test_detail(self) -> None:
        n = 100_000
        x = np.random.default_rng(3).normal(600.0, 5.0, n)
        d = gauss_fit_histogram(x)
        assert d.stats == gauss_stats(x)
        assert d.edges.shape == (d.counts.shape[0] + 1,)
        assert metrics.MIN_BINS <= d.counts.size <= metrics.MAX_BINS
        lo, hi = np.percentile(x, metrics.HIST_PERCENTILES)
        assert d.edges[0] == pytest.approx(lo) and d.edges[-1] == pytest.approx(hi)
        assert int(d.counts.sum()) == int(np.count_nonzero((x >= lo) & (x <= hi)))
        assert d.counts.dtype == np.int64
        assert d.fit_range[0] < d.stats.mean < d.fit_range[1]
        # The curve is the fitted model in counts per bin.
        peak = int(np.argmax(d.curve(d.centers)))
        assert d.curve(d.centers)[peak] == pytest.approx(d.amplitude, rel=1e-3)
        assert abs(d.counts[peak] - d.amplitude) < 5.0 * math.sqrt(d.amplitude)
        np.testing.assert_allclose(d.centers, 0.5 * (d.edges[:-1] + d.edges[1:]))

    def test_bin_count_freedman_diaconis_and_clamp(self) -> None:
        rng = np.random.default_rng(4)
        # N = 1e3: FD gives ~20 bins -> clamped up to 50.
        assert gauss_fit_histogram(rng.normal(0, 1, 1_000)).counts.size == metrics.MIN_BINS
        # N = 1e6: FD gives ~190 bins over +-2.58 sigma.
        n_bins = gauss_fit_histogram(rng.normal(0, 1, 1_000_000)).counts.size
        assert 150 < n_bins < 230
        # Heavy tails (Cauchy): FD asks for >1000 bins -> clamped down to 400.
        assert gauss_fit_histogram(rng.standard_cauchy(200_000)).counts.size == metrics.MAX_BINS

    def test_min_bin_width_floor(self) -> None:
        rng = np.random.default_rng(6)
        x = rng.normal(430.0, 7.5, 1_000_000)
        # No floor: FD bins ~0.2 wide.
        assert gauss_fit_histogram(x).edges[1] - gauss_fit_histogram(x).edges[0] < 0.3
        # 1-unit floor: fewer, wider bins (38 over the ~38.7-unit range), below MIN_BINS.
        d = gauss_fit_histogram(x, min_bin_width=1.0)
        width = d.edges[1] - d.edges[0]
        assert width >= 1.0 and d.counts.size < metrics.MIN_BINS
        assert d.counts.size == int((d.edges[-1] - d.edges[0]) // 1.0)
        assert d.stats.ok and d.stats.sigma == pytest.approx(7.5, rel=0.01)
        # A narrow distribution: the floor would leave ~10 bins -> the 20-bin minimum wins.
        d = gauss_fit_histogram(rng.normal(0.0, 2.0, 1_000_000), min_bin_width=1.0)
        assert d.counts.size == 20
        # A floor below the FD width changes nothing.
        y = rng.normal(0.0, 1.0, 1_000)
        assert gauss_fit_histogram(y, min_bin_width=1e-3).stats == gauss_fit_histogram(y).stats
        # FD asks for 53 bins, between 1x and 2x the 38 that 1-unit bins allow: the floor wins.
        z = rng.normal(0.0, 7.5, 20_000)
        n_fd = gauss_fit_histogram(z).counts.size
        d = gauss_fit_histogram(z, min_bin_width=1.0)
        n_floor = int((d.edges[-1] - d.edges[0]) // 1.0)
        assert n_floor < n_fd < 2 * n_floor
        assert d.counts.size == n_floor and d.edges[1] - d.edges[0] >= 1.0

    def test_min_bin_width_tames_int16_lattice(self) -> None:
        # Radii of int16-rounded points (about a non-integer centre) have lattice structure
        # finer than 1 ADC. With FD bins ~0.2 ADC wide chi2/ndf blows up at N = 1e6 (~9);
        # 1-ADC bins remove most of it. The remaining excess (~2-3, 1.85-3.05 over seeds)
        # is intrinsic lattice-count structure: a dithered (non-lattice) control gives ~1.
        # Medians over 3 seeds keep the assertion statistically robust.
        wide, narrow = [], []
        for seed in (9, 10, 11):
            u, v, p = _ring(1.0, 1_000_000, seed=seed, noise=7.5)
            r = radii_about_center(u, v, p)
            w = gauss_fit_histogram(r, min_bin_width=1.0).stats
            nw = gauss_fit_histogram(r).stats
            assert w.ok and nw.ok
            assert w.sigma == pytest.approx(math.sqrt(7.5**2 + 1 / 12), rel=0.01)
            assert w.chi2ndf < nw.chi2ndf
            wide.append(w.chi2ndf)
            narrow.append(nw.chi2ndf)
        assert np.median(narrow) > 5.0
        assert np.median(wide) < 0.5 * np.median(narrow)
        assert np.median(wide) < 5.0

    def test_flat_topped_distribution_is_fitted(self) -> None:
        # Real post-correction radii are flat-topped (trimmed excess kurtosis median -0.44,
        # down to -1.3). The Gaussian's ok depends only on the usability rules, not on shape
        # (plan 5.3): this sample (excess kurtosis ~ -0.9) gets its best Gaussian.
        rng = np.random.default_rng(0)
        r = 430.0 + rng.uniform(-13.0, 13.0, 20_000) + rng.normal(0.0, 3.0, 20_000)
        rs = radial_stats(r)
        assert rs.ok
        assert rs.sigma == pytest.approx(9.6, abs=0.35)  # 9.39-9.81 over 60 seeds
        assert rs.kurtosis < -0.8  # still reported as a statistic
        assert not hasattr(metrics, "MIN_EXCESS_KURTOSIS")

    @pytest.mark.parametrize("n", [1_000, 20_000, 200_000])
    @pytest.mark.parametrize("ratio", [0.85, 0.9])
    def test_pre_radii_of_elliptical_ring(self, ratio: float, n: int) -> None:
        # Double-horned pre radii (spread between b and a): whatever the fit decides, nothing
        # raises and the reported numbers are finite; the post radii are a clean Gaussian.
        u, v, _ = _ring(ratio, n, seed=n)
        fit = fit_ellipse(u, v)
        assert fit.params is not None
        pre = radial_stats(radii_about_center(u, v, fit.params))
        assert math.isfinite(pre.mean) and math.isfinite(pre.sigma) and pre.sigma > 0
        assert math.isfinite(pre.kurtosis) and pre.kurtosis < -0.8
        assert math.isfinite(pre.chi2ndf) if pre.ok else math.isnan(pre.chi2ndf)
        uc, vc = correct(u, v, fit.params)
        post = radial_stats(np.hypot(uc, vc))
        assert post.ok and post.sigma == pytest.approx(5.0, rel=0.15)

    def test_sigma_range_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A nearly flat sample (N(0, 5) truncated to +-3) fits a Gaussian wider than half the
        # histogram range: unusable. Without the rule it would be "ok" with sigma ~5.
        rng = np.random.default_rng(6)
        w = rng.normal(0.0, 5.0, 400_000)
        w = w[np.abs(w) < 3.0][:50_000]
        g = gauss_stats(w)
        assert not g.ok and g.sigma == pytest.approx(float(np.std(w, ddof=1)))
        monkeypatch.setattr(metrics, "MAX_SIGMA_OF_RANGE", 100.0)
        loose = gauss_stats(w)
        assert loose.ok and loose.sigma > 0.5 * (w.max() - w.min())

    def test_one_populated_bin_is_not_a_fit(self) -> None:
        # Lattice data with MAD = 0: nearly every value is 430 (429 and 431 each ~0.6 %).
        # The initial range (+-3 sample std) holds one populated bin; a fit would give
        # sigma ~0.01 with chi2/ndf ~0. At least 3 non-empty bins are required.
        x = np.round(np.random.default_rng(0).normal(430.0, 0.2, 100_000))
        d = gauss_fit_histogram(x, min_bin_width=1.0)
        assert not d.stats.ok
        assert d.stats.mean == pytest.approx(float(np.mean(x)))
        assert d.stats.sigma == pytest.approx(float(np.std(x, ddof=1)))
        assert math.isnan(d.stats.chi2ndf) and math.isnan(d.amplitude)
        assert d.counts.size > 0  # the histogram is still there for plotting

    def test_fewer_than_three_populated_bins_skip_the_fit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []
        real = metrics._fit_gaussian

        def spy(x: FloatArray, counts: FloatArray, p0: Any) -> Any:
            calls.append(int(np.count_nonzero(counts)))
            return real(x, counts, p0)

        monkeypatch.setattr(metrics, "_fit_gaussian", spy)
        two_values = np.r_[np.full(500, 430.0), np.full(500, 431.0)]
        assert not gauss_stats(two_values).ok
        assert calls == []
        assert gauss_stats(np.random.default_rng(1).normal(0, 1, 1000)).ok
        assert calls and min(calls) >= 3

    def test_sigma_below_half_a_bin_is_not_a_fit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A spike (90 % of the values at exactly 430) on a flat pedestal: refits over
        # mu +- 2 sigma home in on the spike and want sigma < half a bin width. Such a fit
        # is unusable, so the previous usable fit (sigma ~0.8) is kept.
        rng = np.random.default_rng(5)
        y = np.r_[
            np.full(90_000, 430.0), rng.uniform(420, 429.5, 5000), rng.uniform(430.5, 440, 5000)
        ]
        d = gauss_fit_histogram(y)
        width = d.edges[1] - d.edges[0]
        assert d.stats.ok and d.stats.sigma >= 0.5 * width
        monkeypatch.setattr(metrics, "MIN_SIGMA_OF_BIN", 0.0)
        spiky = gauss_fit_histogram(y).stats
        assert spiky.ok and spiky.sigma < 0.5 * width

    def test_curve_nan_when_failed(self) -> None:
        d = gauss_fit_histogram([1.0, 1.0, 1.0])
        assert not d.stats.ok
        assert np.isnan(d.curve([0.0, 1.0])).all()
        assert d.edges.size == 0 and d.counts.size == 0
        assert all(math.isnan(v) for v in d.fit_range)

    def test_plan_constants(self) -> None:
        # Values fixed by plan 5.3 and the phase 2 review; tests elsewhere use literals.
        assert metrics.ADC_MIN_BIN_WIDTH == 1.0
        assert metrics.MIN_BINS_WITH_FLOOR == 20
        assert metrics.MIN_NONEMPTY_FIT_BINS == 3
        assert metrics.MIN_SIGMA_OF_BIN == 0.5
        assert metrics.MAX_SIGMA_OF_RANGE == 0.5
        assert (metrics.MIN_BINS, metrics.MAX_BINS) == (50, 400)
        assert metrics.HIST_PERCENTILES == (0.5, 99.5)
        assert (metrics.INITIAL_RANGE_SIGMAS, metrics.REFIT_RANGE_SIGMAS) == (3.0, 2.0)
        assert metrics.REFIT_MAX == 3

    def test_min_samples(self) -> None:
        rng = np.random.default_rng(9)
        few = gauss_fit_histogram(rng.normal(0, 1, metrics.MIN_GAUSS_SAMPLES - 1))
        assert not few.stats.ok and few.counts.size == 0
        enough = gauss_fit_histogram(rng.normal(0, 1, metrics.MIN_GAUSS_SAMPLES))
        assert enough.counts.size > 0
        assert metrics.MIN_GAUSS_SAMPLES == 50

    def test_gaussian_function(self) -> None:
        assert metrics.gaussian(2.0, 3.0, 2.0, 0.5) == pytest.approx(3.0)
        assert metrics.gaussian(2.5, 3.0, 2.0, 0.5) == pytest.approx(3.0 * math.exp(-0.5))

    def test_poisson_deviance(self) -> None:
        mu = np.array([2.0, 5.0, 5.0, 100.0, 7.0])
        n = np.array([0.0, 5.0, 8.0, 90.0, 3.0])
        expected = 2.0 * (mu - n + np.where(n > 0, n * np.log(np.where(n > 0, n, 1) / mu), 0))
        np.testing.assert_allclose(poisson_deviance(n, mu), expected, rtol=1e-12)
        assert poisson_deviance([0.0], [2.5])[0] == 5.0
        assert poisson_deviance([4.0], [4.0])[0] == 0.0
        # Continuous and accurate across the Taylor-series switch (|n/mu - 1| = 1e-4) and
        # away from it, where the direct formula is exact to ~1e-13.
        mu_c = 1e6
        for u in (0.99e-4, 1.01e-4, -0.99e-4, -1.01e-4):
            n_c = mu_c * (1.0 + u)
            exact = 2.0 * mu_c * ((1 + u) * math.log1p(u) - u)
            assert poisson_deviance([n_c], [mu_c])[0] == pytest.approx(exact, rel=1e-6)
        for u in (0.05, -0.05, 0.3, 2.0):
            exact = 2.0 * 50.0 * ((1 + u) * math.log1p(u) - u)
            assert poisson_deviance([50.0 * (1 + u)], [50.0])[0] == pytest.approx(exact, rel=1e-10)
        # Just above the switch the general formula must not round n / mu inside the log
        # (that cost ~1.5e-8 relative), and the switch must stay low enough that the
        # truncated series is exact (at |u| = 5e-3 the u^4 series would be off by ~1e-8).
        # Reference: the h(u) series to u^7 (error ~u^6).
        for u in (1.01e-4, -1.01e-4, 3e-4, -3e-4, 5e-3, -5e-3):
            h = u * u * (1 / 2 - u / 6 + u**2 / 12 - u**3 / 20 + u**4 / 30 - u**5 / 42)
            for mu_c in (57.3, 1234.5, 1e6):
                got = poisson_deviance([mu_c * (1.0 + u)], [mu_c])[0]
                assert got == pytest.approx(2.0 * mu_c * h, rel=1e-10)

    @pytest.mark.parametrize(
        "values",
        [
            [],
            [5.0],
            [5.0, 6.0],
            [3.0] * 60,
            [math.nan] * 20,
            [math.inf, -math.inf, math.nan],
            list(range(20)),
        ],
        ids=["empty", "one", "two", "all-equal", "all-nan", "all-nonfinite", "twenty"],
    )
    def test_degenerate_inputs(self, values: list[float]) -> None:
        g = gauss_stats(values)
        assert not g.ok
        assert math.isnan(g.chi2ndf)
        finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
        if finite.size == 0:
            assert math.isnan(g.mean) and math.isnan(g.sigma)
        else:
            assert g.mean == pytest.approx(float(finite.mean()))
            expected_std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
            assert g.sigma == pytest.approx(expected_std)
            assert g.fwhm == pytest.approx(FWHM_FACTOR * expected_std)

    @pytest.mark.parametrize(
        "values",
        [
            np.random.default_rng(10).normal(0, 1, 1000) * 1e300,
            np.random.default_rng(11).normal(0, 1, 1000) * 1e-300,
            np.random.default_rng(12).normal(0, 1, 1000) * 5e-324,
            1e16 + np.random.default_rng(13).normal(0, 1, 1000),
            np.r_[np.full(500, 1e308), np.full(500, -1e308)],
        ],
        ids=["huge", "tiny", "subnormal", "1e16-offset", "near-overflow"],
    )
    def test_extreme_magnitudes_do_not_raise(self, values: FloatArray) -> None:
        for fn in (gauss_fit_histogram, radial_stats, residual_stats):
            fn(values)
        rs = radial_stats(values)
        assert isinstance(rs.ok, bool)

    def test_not_a_peak(self) -> None:
        rng = np.random.default_rng(5)
        assert not gauss_stats(rng.uniform(0.0, 1.0, 10_000)).ok
        assert not gauss_stats(rng.integers(0, 3, 10_000)).ok

    def test_nan_values_ignored(self) -> None:
        x = np.random.default_rng(6).normal(0.0, 1.0, 5000)
        assert gauss_stats(np.r_[x, math.nan, math.inf]) == gauss_stats(x)

    def test_fit_detail_helpers_match_stats(self) -> None:
        # The GUI plots radial_fit_detail / residual_fit_detail: exactly what the stats fit.
        rng = np.random.default_rng(12)
        r = np.round(rng.normal(430.0, 7.5, 300_000)) + 0.3
        rs = radial_stats(r)
        det = metrics.radial_fit_detail(r)
        assert rs.gauss == det.stats and rs.sigma == det.stats.sigma
        assert det.edges[1] - det.edges[0] >= 1.0  # the 1-ADC floor is applied
        np.testing.assert_array_equal(det.edges, gauss_fit_histogram(r, 1.0).edges)
        res = rng.normal(0.2, 4.0, 300_000)
        rdet = metrics.residual_fit_detail(res)
        assert residual_stats(res) == rdet.stats
        assert rdet.edges[1] - rdet.edges[0] >= 1.0
        assert metrics.residual_fit_detail(res, None).stats == residual_stats(res, None)

    def test_residual_stats_is_gauss_routine_with_adc_floor(self) -> None:
        res = np.random.default_rng(7).normal(0.3, 4.0, 200_000)
        got = residual_stats(res)
        assert got == gauss_stats(res, min_bin_width=1.0)
        assert got != gauss_stats(res)  # FD bins would be ~0.2 wide here
        assert residual_stats(res, min_bin_width=None) == gauss_stats(res)
        assert isinstance(residual_stats([]), GaussStats)


# ---------------------------------------------------------------------------
# Radial statistics
# ---------------------------------------------------------------------------


class TestRadialStats:
    def test_unbinned_statistics_match_scipy(self) -> None:
        # A skewed, peaked sample so skewness and kurtosis are clearly non-zero.
        r = 560.0 + 3.0 * np.random.default_rng(8).gamma(4.0, 2.0, 50_000)
        rs = radial_stats(r)
        assert rs.skewness == pytest.approx(float(sp_stats.skew(r)), rel=1e-10)
        assert rs.kurtosis == pytest.approx(float(sp_stats.kurtosis(r, fisher=True)), rel=1e-10)
        assert rs.skewness > 0.5 and rs.kurtosis > 0.5
        assert rs.sample_mean == pytest.approx(float(np.mean(r)))
        assert rs.sample_std == pytest.approx(float(np.std(r, ddof=1)))
        med = np.median(r)
        assert rs.robust_sigma == pytest.approx(MAD_TO_SIGMA * float(np.median(np.abs(r - med))))

    def test_gaussian_part_matches_routine(self) -> None:
        r = np.random.default_rng(9).normal(592.0, 5.0, 30_000)
        rs = radial_stats(r)
        assert rs.ok
        assert rs.gauss == gauss_stats(r, min_bin_width=1.0)
        assert radial_stats(r, min_bin_width=None).gauss == gauss_stats(r)
        assert rs.mean == pytest.approx(592.0, abs=6 * 5.0 / math.sqrt(30_000))
        assert rs.robust_sigma == pytest.approx(5.0, rel=0.03)
        assert abs(rs.skewness) < 0.1 and abs(rs.kurtosis) < 0.1

    def test_nan_ignored(self) -> None:
        r = np.random.default_rng(10).normal(600.0, 5.0, 3000)
        assert radial_stats(np.r_[math.nan, r, math.inf]) == radial_stats(r)

    @pytest.mark.parametrize(
        "values",
        [[], [600.0], [600.0, 601.0], [600.0] * 60, [math.nan] * 5],
        ids=["empty", "one", "two", "all-equal", "all-nan"],
    )
    def test_degenerate(self, values: list[float]) -> None:
        rs = radial_stats(values)
        assert not rs.ok
        assert math.isnan(rs.chi2ndf)
        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            assert math.isnan(rs.sample_mean) and math.isnan(rs.robust_sigma)
        else:
            assert rs.sample_mean == pytest.approx(float(np.mean(finite)))
            assert rs.mean == rs.sample_mean and rs.sigma == rs.sample_std
        if len(set(finite)) < 2:
            assert math.isnan(rs.skewness) and math.isnan(rs.kurtosis)


# ---------------------------------------------------------------------------
# Phase statistics
# ---------------------------------------------------------------------------


def _points_at_degrees(
    degrees: list[float], radius: float = 600.0
) -> tuple[FloatArray, FloatArray]:
    rad = np.deg2rad(np.asarray(degrees, dtype=np.float64))
    return radius * np.cos(rad), radius * np.sin(rad)


class TestPhaseStats:
    def test_hand_computed(self) -> None:
        # Sorted phases 10, 50, 90, 120, 180, 200, 270, 350 deg (given unsorted, with
        # angles past 180 deg that atan2 returns negative). Gaps 40, 40, 30, 60, 20,
        # 70, 80 and wrap-around 20 deg: max 80 deg. KS: the largest term is
        # (5+1)/8 - 200/360 = 7/36.
        u, v = _points_at_degrees([270, 10, 120, 350, 50, 200, 90, 180])
        ps = phase_stats(u, v, 490e3)
        assert ps is not None
        assert ps.mean_gap_rad == pytest.approx(math.pi / 4, rel=1e-15)
        assert ps.max_gap_rad == pytest.approx(math.radians(80), rel=1e-12)
        assert ps.max_gap_ns == pytest.approx((80 / 360) / 490e3 * 1e9, rel=1e-12)
        assert ps.ks == pytest.approx(7 / 36, rel=1e-12)

    def test_hand_computed_wraparound_gap_is_max(self) -> None:
        # Phases 40, 60, 100, 150, 180, 230, 260, 300 deg: the largest gap is the
        # wrap-around one, 40 + 360 - 300 = 100 deg. KS = 8/8 - 300/360 = 1/6.
        u, v = _points_at_degrees([40, 60, 100, 150, 180, 230, 260, 300], radius=3.0)
        ps = phase_stats(u, v, 500e3)
        assert ps is not None
        assert ps.max_gap_rad == pytest.approx(math.radians(100), rel=1e-12)
        assert ps.max_gap_ns == pytest.approx((100 / 360) / 500e3 * 1e9, rel=1e-12)
        assert ps.ks == pytest.approx(1 / 6, rel=1e-12)
        assert ps.mean_gap_rad == pytest.approx(TWO_PI / 8)

    def test_tiny_negative_angle_stays_at_top_of_cdf(self) -> None:
        # atan2(-1e-17, 1) + 2 pi rounds to exactly 2 pi. It must become the largest phase
        # below 2 pi (as in the C++, KS = 7/8), not 0 (KS would be 1) nor 2 pi itself.
        u = np.ones(8)
        v = np.r_[np.zeros(7), -1e-17]
        ps = phase_stats(u, v)
        assert ps is not None
        assert ps.ks == 0.875
        assert ps.max_gap_rad == math.nextafter(TWO_PI, 0.0)

    def test_equally_spaced(self) -> None:
        n = 1000
        deg = (np.arange(n) + 0.5) * 360.0 / n
        ps = phase_stats(*_points_at_degrees(list(deg)))
        assert ps is not None
        assert ps.max_gap_rad == pytest.approx(TWO_PI / n, rel=1e-9)
        assert ps.ks == pytest.approx(0.5 / n, rel=1e-6)

    def test_uniform_random(self) -> None:
        n = 20_000
        rng = np.random.default_rng(11)
        ph = rng.uniform(0.0, TWO_PI, n)
        r = rng.normal(600.0, 5.0, n)  # radii do not matter
        ps = phase_stats(r * np.cos(ph), r * np.sin(ph))
        assert ps is not None
        # KS: P(D > 1.63 / sqrt(N)) ~ 1 %.
        assert ps.ks < 1.63 / math.sqrt(n)
        # Largest of N uniform spacings: ~ (2 pi / N) * (ln N + 0.577).
        expected = TWO_PI / n * (math.log(n) + 0.5772)
        assert 0.5 * expected < ps.max_gap_rad < 2.0 * expected
        assert ps.max_gap_ns == pytest.approx(ps.max_gap_rad / (TWO_PI * 490e3) * 1e9)

    def test_clustered_phases(self) -> None:
        u, v = _points_at_degrees(list(np.linspace(0.0, 90.0, 100)))
        ps = phase_stats(u, v)
        assert ps is not None
        assert ps.max_gap_rad == pytest.approx(math.radians(270), rel=1e-9)
        assert ps.ks == pytest.approx(0.75, abs=0.01)

    def test_all_same_angle(self) -> None:
        ps = phase_stats(np.ones(10), np.zeros(10))
        assert isinstance(ps, PhaseStats)
        assert (ps.mean_gap_rad, ps.max_gap_rad, ps.ks) == (TWO_PI / 10, TWO_PI, 1.0)
        assert ps.max_gap_ns == pytest.approx(1e9 / 490e3, rel=1e-15)

    def test_too_few_points(self) -> None:
        u, v = _points_at_degrees([0, 45, 90, 135, 180, 225, 270])
        assert phase_stats(u, v) is None
        assert phase_stats([], []) is None

    def test_non_finite_points_dropped(self) -> None:
        u, v = _points_at_degrees([10, 50, 90, 120, 180, 200, 270, 350])
        ps = phase_stats(np.r_[u, math.nan, 1.0], np.r_[v, 0.0, math.inf])
        assert ps == phase_stats(u, v)
        assert phase_stats(np.r_[u[:7], math.nan], np.r_[v[:7], 0.0]) is None

    def test_invalid_frequency(self) -> None:
        u, v = _points_at_degrees(list(range(0, 360, 30)))
        for bad in (0.0, -1.0, math.nan, math.inf):
            with pytest.raises(ValueError, match="ref_freq_hz"):
                phase_stats(u, v, bad)


def test_thread_safe() -> None:
    # No process-wide warning filters are touched (only thread-local np.errstate), so fits
    # running concurrently in threads (the GUI's QThreads) give the serial results.
    from concurrent.futures import ThreadPoolExecutor

    def job(seed: int) -> tuple[Any, ...]:
        u, v, _ = _ring(0.95, 20_000, seed=seed)
        fit = fit_ellipse(u, v)
        assert fit.params is not None
        uc, vc = correct(u, v, fit.params)
        return (
            fit.params,
            radial_stats(radii_about_center(u, v, fit.params)),
            radial_stats(np.hypot(uc, vc)),
            phase_stats(uc, vc),
        )

    seeds = list(range(8))
    serial = [job(s) for s in seeds]
    with ThreadPoolExecutor(max_workers=4) as pool:
        threaded = list(pool.map(job, seeds))
    assert threaded == serial


class TestTimingJitter:
    def test_value(self) -> None:
        assert timing_jitter_ns(5.0, 600.0, 490e3) == pytest.approx(
            5.0 / (TWO_PI * 490e3 * 600.0) * 1e9
        )
        assert timing_jitter_ns(5.0, 600.0) == timing_jitter_ns(5.0, 600.0, 490e3)

    @pytest.mark.parametrize(
        ("sigma", "mean"), [(5.0, 0.0), (math.nan, 600.0), (5.0, math.nan), (math.inf, 1.0)]
    )
    def test_degenerate(self, sigma: float, mean: float) -> None:
        assert math.isnan(timing_jitter_ns(sigma, mean, 490e3))

    def test_invalid_frequency(self) -> None:
        with pytest.raises(ValueError, match="ref_freq_hz"):
            timing_jitter_ns(5.0, 600.0, 0.0)
