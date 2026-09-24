"""Tests for uvcorr.ellipse (plan section 10.2).

Synthetic channels mimic the real data: centre ~(2000, 2000), radius ~600,
b/a between 0.7 and 1, N from 1e3 to 1e5, Gaussian radial noise, optional
uniform outliers over the bounding box and optional int16 rounding (the cache
stores U/V as int16).

Parameter tolerances (``_param_tol``): ``10 * sigma / sqrt(N) + 2 * sigma**2 / R``
in ADC for the centre and semi-axes, and that divided by ``a - b`` (radians) for
phi. ``sigma / sqrt(N)`` is the natural standard-error scale of the fitted
parameters; a Monte Carlo over 40 seeds for every configuration used here
(sigma 3 and 8, 0-10 % outliers, robust fit) gave a largest error of 7.8 such
units, so 10 leaves margin while still catching real regressions (the plain fit
with 5 % outliers is off by 10-1000 units). The ``sigma**2 / R`` term covers the
known O(sigma^2 / R) bias of algebraic ellipse fits, which dominates at large N.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from uvcorr import ellipse
from uvcorr.ellipse import (
    EllipseParams,
    correct,
    corrected_residual,
    ellipse_points,
    fit_ellipse,
    fit_ellipse_direct,
    radii_about_center,
    residual_to_ellipse,
    uncorrect,
)
from uvcorr.options import (
    FLAG_BROAD_RING,
    FLAG_CENTER_OUTSIDE_DATA,
    FLAG_EXTREME_AXIS_RATIO,
    FLAG_GEOMETRIC_REFIT_FAILED,
    FLAG_HIGH_REJECTION,
    FLAG_ROBUST_REFIT_FAILED,
    FLAGS,
    STATUS_FIT_FAILED,
    STATUS_OK,
    STATUS_TOO_FEW_EVENTS,
    FitOptions,
)

# Any warning (numpy RuntimeWarning, scipy OptimizeWarning, ...) fails the test.
pytestmark = pytest.mark.filterwarnings("error")

FloatArray = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def make_channel(
    p: EllipseParams,
    n: int,
    noise: float,
    *,
    seed: int = 0,
    outlier_frac: float = 0.0,
    int16: bool = False,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[np.bool_]]:
    """Points on ellipse ``p`` at uniform parametric angles with Gaussian radial noise.

    The noise moves each point along the ray from the centre, so its radial
    residual to the true ellipse is exactly the noise. ``outlier_frac`` of the
    points are replaced by points uniform over the bounding box of the clean
    data. Returns (u, v, is_outlier).
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 2.0 * np.pi, n)
    x, y = ellipse_points(p, t)
    dx, dy = x - p.cx, y - p.cy
    scale = 1.0 + rng.normal(0.0, noise, n) / np.hypot(dx, dy)
    x = p.cx + dx * scale
    y = p.cy + dy * scale
    is_outlier = np.zeros(n, dtype=np.bool_)
    n_out = int(round(outlier_frac * n))
    if n_out:
        lo_x, hi_x, lo_y, hi_y = x.min(), x.max(), y.min(), y.max()
        idx = rng.choice(n, n_out, replace=False)
        x[idx] = rng.uniform(lo_x, hi_x, n_out)
        y[idx] = rng.uniform(lo_y, hi_y, n_out)
        is_outlier[idx] = True
    if int16:
        return np.round(x).astype(np.int16), np.round(y).astype(np.int16), is_outlier
    return x, y, is_outlier


def _param_tol(noise: float, n: int, p: EllipseParams) -> float:
    """Tolerance in ADC for the centre and semi-axes (see the module docstring)."""
    sig = math.hypot(noise, 0.5)  # int16 rounding adds a little noise
    return 10.0 * sig / math.sqrt(n) + 2.0 * sig * sig / p.target_radius


def _phi_diff(phi1: float, phi2: float) -> float:
    """Axis-angle difference modulo pi, in [-pi/2, pi/2]."""
    return math.remainder(phi1 - phi2, math.pi)


def _assert_canonical(p: EllipseParams) -> None:
    assert p.a >= p.b > 0.0
    assert -math.pi / 2 < p.phi <= math.pi / 2


def _assert_close(got: EllipseParams, true: EllipseParams, tol: float) -> None:
    _assert_canonical(got)
    assert abs(got.cx - true.cx) < tol
    assert abs(got.cy - true.cy) < tol
    assert abs(got.a - true.a) < tol
    assert abs(got.b - true.b) < tol
    # phi is meaningful only for a clearly non-circular ellipse.
    if true.b / true.a < 0.99:
        assert abs(_phi_diff(got.phi, true.phi)) < tol / (true.a - true.b)


def _assert_failed_or_flagged(fit: ellipse.EllipseFit) -> None:
    """A non-ring input must come back either failed or with a warning flag."""
    assert fit.status in (STATUS_FIT_FAILED, STATUS_OK)
    if fit.ok:
        assert {FLAG_CENTER_OUTSIDE_DATA, FLAG_EXTREME_AXIS_RATIO, FLAG_BROAD_RING} & set(fit.flags)


def _param_error(got: EllipseParams, true: EllipseParams) -> float:
    """Largest centre/axis error in ADC."""
    return max(
        abs(got.cx - true.cx), abs(got.cy - true.cy), abs(got.a - true.a), abs(got.b - true.b)
    )


def _patch_residuals(
    monkeypatch: pytest.MonkeyPatch, provider: Callable[[int, int], FloatArray]
) -> list[int]:
    """Replace ``residual_to_ellipse`` inside ``fit_ellipse`` by ``provider(call, n)``.

    Every call of the robust loop (and the final ring-quality check) gets the
    array ``provider`` returns for its 0-based call index and point count.
    Returns the list of call indices, for counting.
    """
    calls: list[int] = []

    def fake(u: Any, v: Any, p: EllipseParams) -> FloatArray:
        calls.append(len(calls))
        return np.array(provider(len(calls) - 1, int(np.size(u))), dtype=np.float64)

    monkeypatch.setattr(ellipse, "residual_to_ellipse", fake)
    return calls


@dataclass(frozen=True)
class Case:
    name: str
    params: EllipseParams
    n: int
    noise: float


CASES = [
    Case("tilted", EllipseParams(2000.0, 2000.0, 620.0, 560.0, 0.4), 1_000, 5.0),
    Case("phi+1.55", EllipseParams(2030.0, 1980.0, 640.0, 480.0, 1.55), 10_000, 4.0),
    Case("phi-1.55", EllipseParams(1990.0, 2010.0, 640.0, 480.0, -1.55), 100_000, 6.0),
    Case("phi=pi/2", EllipseParams(2000.0, 2000.0, 650.0, 460.0, math.pi / 2), 10_000, 5.0),
    Case("phi=-0.9", EllipseParams(2010.0, 2040.0, 600.0, 420.0, -0.9), 3_000, 3.0),
    Case("near-circle", EllipseParams(2000.0, 2000.0, 600.0, 597.0, 0.3), 10_000, 5.0),
]
CASE_IDS = [c.name for c in CASES]


# ---------------------------------------------------------------------------
# EllipseParams
# ---------------------------------------------------------------------------


class TestEllipseParams:
    def test_derived(self) -> None:
        p = EllipseParams(1.0, 2.0, 8.0, 2.0, 0.1)
        assert p.target_radius == 4.0
        assert p.axis_ratio == 0.25

    @pytest.mark.parametrize(
        ("phi", "expected"),
        [
            (0.3, 0.3),
            (math.pi / 2, math.pi / 2),
            (-math.pi / 2, math.pi / 2),
            (math.pi, 0.0),
            (2.0, 2.0 - math.pi),
            (-2.0, math.pi - 2.0),
            (3 * math.pi / 2, math.pi / 2),
            (7.0, 7.0 - 2 * math.pi),
        ],
    )
    def test_canonical_phi_wrap(self, phi: float, expected: float) -> None:
        got = EllipseParams(0.0, 0.0, 3.0, 2.0, phi).canonical()
        assert -math.pi / 2 < got.phi <= math.pi / 2
        assert got.phi == pytest.approx(expected, abs=1e-12)

    def test_canonical_exact_half_pi(self) -> None:
        assert EllipseParams(0.0, 0.0, 3.0, 2.0, math.pi / 2).canonical().phi == math.pi / 2
        assert EllipseParams(0.0, 0.0, 3.0, 2.0, -math.pi / 2).canonical().phi == math.pi / 2

    def test_canonical_swaps_axes(self) -> None:
        got = EllipseParams(5.0, 6.0, 2.0, 3.0, -0.2).canonical()
        assert (got.cx, got.cy, got.a, got.b) == (5.0, 6.0, 3.0, 2.0)
        assert got.phi == pytest.approx(-0.2 + math.pi / 2)
        got = EllipseParams(0.0, 0.0, 2.0, 3.0, 1.2).canonical()
        assert got.phi == pytest.approx(1.2 + math.pi / 2 - math.pi)

    def test_canonical_abs_axes(self) -> None:
        got = EllipseParams(0.0, 0.0, -3.0, -2.0, 0.1).canonical()
        assert (got.a, got.b) == (3.0, 2.0)

    def test_canonical_describes_same_ellipse(self) -> None:
        p = EllipseParams(2000.0, 2000.0, 400.0, 600.0, 2.9)
        q = p.canonical()
        t = np.linspace(0.0, 2.0 * np.pi, 50)
        x, y = ellipse_points(p, t)
        np.testing.assert_allclose(residual_to_ellipse(x, y, q), 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Direct algebraic fit
# ---------------------------------------------------------------------------


class TestFitEllipseDirect:
    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_noiseless_exact(self, case: Case) -> None:
        t = np.random.default_rng(1).uniform(0.0, 2.0 * np.pi, 500)
        x, y = ellipse_points(case.params, t)
        got = fit_ellipse_direct(x, y)
        assert got is not None
        _assert_canonical(got)
        true = case.params
        assert got.cx == pytest.approx(true.cx, abs=1e-8)
        assert got.cy == pytest.approx(true.cy, abs=1e-8)
        assert got.a == pytest.approx(true.a, rel=1e-10)
        assert got.b == pytest.approx(true.b, rel=1e-10)
        assert abs(_phi_diff(got.phi, true.phi)) < 1e-8

    def test_exact_half_pi_is_canonical(self) -> None:
        # a along V: the fit must report phi = +-pi/2 in canonical form, i.e. close to
        # pi/2 or just above -pi/2, never outside (-pi/2, pi/2].
        p = EllipseParams(2000.0, 2000.0, 650.0, 460.0, math.pi / 2)
        x, y = ellipse_points(p, np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False))
        got = fit_ellipse_direct(x, y)
        assert got is not None
        _assert_canonical(got)
        assert abs(_phi_diff(got.phi, math.pi / 2)) < 1e-9
        assert got.a == pytest.approx(650.0) and got.b == pytest.approx(460.0)

    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    @pytest.mark.parametrize("int16", [False, True], ids=["float", "int16"])
    def test_noisy(self, case: Case, int16: bool) -> None:
        u, v, _ = make_channel(case.params, case.n, case.noise, seed=3, int16=int16)
        got = fit_ellipse_direct(u, v)
        assert got is not None
        _assert_close(got, case.params, _param_tol(case.noise, case.n, case.params))

    def test_int16_same_as_float(self) -> None:
        p = CASES[1].params
        u, v, _ = make_channel(p, 5000, 4.0, int16=True)
        assert u.dtype == np.int16
        assert fit_ellipse_direct(u, v) == fit_ellipse_direct(
            u.astype(np.float64), v.astype(np.float64)
        )

    def test_translation_equivariance(self) -> None:
        # The same shape at raw magnitude (U ~ 2000) and at the origin: normalisation
        # makes the fit independent of the offset to round-off level.
        shape = EllipseParams(0.0, 0.0, 640.0, 500.0, 0.7)
        u0, v0, _ = make_channel(shape, 20_000, 5.0, seed=11)
        u0 = np.asarray(u0, dtype=np.float64)
        v0 = np.asarray(v0, dtype=np.float64)
        at_origin = fit_ellipse_direct(u0, v0)
        shifted = fit_ellipse_direct(u0 + 2000.0, v0 + 2050.0)
        assert at_origin is not None and shifted is not None
        scale = at_origin.a
        assert abs((shifted.cx - 2000.0) - at_origin.cx) < 1e-9 * scale
        assert abs((shifted.cy - 2050.0) - at_origin.cy) < 1e-9 * scale
        assert shifted.a == pytest.approx(at_origin.a, rel=1e-9)
        assert shifted.b == pytest.approx(at_origin.b, rel=1e-9)
        assert abs(_phi_diff(shifted.phi, at_origin.phi)) < 1e-9

    def test_scale_equivariance(self) -> None:
        shape = EllipseParams(2000.0, 2000.0, 640.0, 500.0, -0.3)
        u, v, _ = make_channel(shape, 5000, 5.0, seed=12)
        base = fit_ellipse_direct(u, v)
        scaled = fit_ellipse_direct(3.0 * np.asarray(u), 3.0 * np.asarray(v))
        assert base is not None and scaled is not None
        assert scaled.cx == pytest.approx(3.0 * base.cx, rel=1e-9)
        assert scaled.a == pytest.approx(3.0 * base.a, rel=1e-9)
        assert scaled.b == pytest.approx(3.0 * base.b, rel=1e-9)
        assert abs(_phi_diff(scaled.phi, base.phi)) < 1e-9

    def test_six_points_exact(self) -> None:
        p = EllipseParams(2000.0, 2000.0, 600.0, 450.0, 0.5)
        x, y = ellipse_points(p, np.array([0.1, 1.0, 2.2, 3.1, 4.4, 5.5]))
        got = fit_ellipse_direct(x, y)
        assert got is not None
        assert got.a == pytest.approx(600.0, rel=1e-8)
        assert got.b == pytest.approx(450.0, rel=1e-8)

    @pytest.mark.parametrize("n", [0, 1, 2, 5])
    def test_too_few_points(self, n: int) -> None:
        x, y = ellipse_points(CASES[0].params, np.linspace(0, 6, n))
        assert fit_ellipse_direct(x, y) is None

    def test_collinear(self) -> None:
        x = np.linspace(1500.0, 2500.0, 200)
        assert fit_ellipse_direct(x, 0.5 * x + 700.0) is None
        xi = np.arange(1500, 1700, dtype=np.int16)
        assert fit_ellipse_direct(xi, (2 * xi - 1000).astype(np.int16)) is None
        assert fit_ellipse_direct(x, np.full_like(x, 2000.0)) is None  # horizontal
        assert fit_ellipse_direct(np.full_like(x, 2000.0), x) is None  # vertical

    def test_identical_points(self) -> None:
        assert fit_ellipse_direct(np.full(50, 2000.0), np.full(50, 1990.0)) is None

    def test_two_distinct_points_repeated(self) -> None:
        x = np.tile([1900.0, 2100.0], 50)
        y = np.tile([2000.0, 2050.0], 50)
        assert fit_ellipse_direct(x, y) is None

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite(self, bad: float) -> None:
        u, v, _ = make_channel(CASES[0].params, 500, 5.0)
        u = np.asarray(u, dtype=np.float64)
        u[10] = bad
        assert fit_ellipse_direct(u, v) is None

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            fit_ellipse_direct(np.zeros(10), np.zeros(11))
        with pytest.raises(ValueError, match="1-D"):
            fit_ellipse_direct(np.zeros((10, 2)), np.zeros((10, 2)))


# ---------------------------------------------------------------------------
# Full per-channel fit
# ---------------------------------------------------------------------------


class TestFitEllipse:
    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    @pytest.mark.parametrize("outlier_frac", [0.0, 0.01, 0.05, 0.10])
    @pytest.mark.parametrize("int16", [False, True], ids=["float", "int16"])
    def test_recovers_parameters(self, case: Case, outlier_frac: float, int16: bool) -> None:
        u, v, _ = make_channel(
            case.params, case.n, case.noise, seed=5, outlier_frac=outlier_frac, int16=int16
        )
        fit = fit_ellipse(u, v)
        assert fit.status == STATUS_OK and fit.ok
        assert fit.params is not None
        _assert_close(fit.params, case.params, _param_tol(case.noise, case.n, case.params))
        assert fit.n_events == case.n
        assert fit.n_used == int(fit.mask_used.sum())
        assert fit.n_used + fit.n_rejected == case.n
        assert fit.mask_used.dtype == np.bool_ and fit.mask_used.shape == (case.n,)

    @pytest.mark.parametrize("case", CASES[:4], ids=CASE_IDS[:4])
    @pytest.mark.parametrize("outlier_frac", [0.05, 0.10])
    def test_robust_beats_plain(self, case: Case, outlier_frac: float) -> None:
        u, v, is_outlier = make_channel(
            case.params, case.n, case.noise, seed=7, outlier_frac=outlier_frac, int16=True
        )
        plain = fit_ellipse(u, v, FitOptions(robust=False))
        robust = fit_ellipse(u, v)
        assert plain.params is not None and robust.params is not None
        tol = _param_tol(case.noise, case.n, case.params)
        plain_err = _param_error(plain.params, case.params)
        robust_err = _param_error(robust.params, case.params)
        # Outliers really bias the plain fit, and the robust fit removes most of it.
        assert plain_err > 2.0 * tol
        assert robust_err < tol
        assert robust_err < plain_err / 5.0
        # Outliers near the ring cannot be told apart, but every outlier farther than
        # 2 * clip_k robust sigmas from the true ellipse must go (the clip is at ~4-5 sigma),
        # and hardly any ring point may.
        rejected = ~robust.mask_used
        sig = math.hypot(case.noise, 0.5)
        far = is_outlier & (np.abs(residual_to_ellipse(u, v, case.params)) > 8.0 * sig)
        assert far.sum() > 0.5 * is_outlier.sum()
        assert rejected[far].all()
        assert np.count_nonzero(rejected & is_outlier) > 0.6 * np.count_nonzero(is_outlier)
        assert np.count_nonzero(rejected & ~is_outlier) < 0.002 * case.n + 2

    def test_no_outliers_rejects_almost_nothing(self) -> None:
        case = CASES[1]
        u, v, _ = make_channel(case.params, case.n, case.noise, seed=8, int16=True)
        fit = fit_ellipse(u, v)
        plain = fit_ellipse_direct(u, v)
        assert fit.params is not None and plain is not None
        # 4-sigma clipping of Gaussian noise rejects ~6e-5 of the points.
        assert fit.n_rejected <= 0.001 * case.n
        assert fit.flags == ()
        assert _param_error(fit.params, plain) < 0.2 * _param_tol(case.noise, case.n, case.params)

    def test_robust_off(self) -> None:
        u, v, _ = make_channel(CASES[0].params, 2000, 5.0, outlier_frac=0.05)
        fit = fit_ellipse(u, v, FitOptions(robust=False))
        assert fit.ok and fit.n_iter == 0 and fit.n_rejected == 0
        assert fit.mask_used.all()
        assert fit.params == fit_ellipse_direct(u, v)

    @pytest.mark.parametrize("max_iter", [1, 3, 7])
    def test_max_iter_limits_iterations(
        self, monkeypatch: pytest.MonkeyPatch, max_iter: int
    ) -> None:
        # Call k rejects k+1 points, so the kept set changes on every iteration and only
        # max_iter can stop the loop.
        def provider(call: int, n: int) -> FloatArray:
            res = np.linspace(-1.0, 1.0, n)
            res[: call + 1] = 1000.0
            return res

        u, v, _ = make_channel(CASES[0].params, 2000, 5.0)
        _patch_residuals(monkeypatch, provider)
        fit = fit_ellipse(u, v, FitOptions(max_iter=max_iter))
        assert fit.ok and fit.n_iter == max_iter
        assert fit.n_rejected == max_iter  # the set of the last accepted refit
        assert not fit.mask_used[:max_iter].any() and fit.mask_used[max_iter:].all()

    @pytest.mark.parametrize(
        ("clip_k", "rejected"),
        [(4.0, [1003, 1004]), (3.95, [1003, 1004]), (3.8, [1001, 1002, 1003, 1004])],
    )
    def test_exact_kept_set(
        self, monkeypatch: pytest.MonkeyPatch, clip_k: float, rejected: list[int]
    ) -> None:
        # Residuals with median 100 (so the median must be subtracted) and MAD exactly 1
        # (robust sigma s = 1.4826): 500 x -1, 500 x +1, one 0, then +-3.9 s and +-4.1 s.
        s = 1.4826
        base = np.r_[
            np.full(500, -1.0), np.full(500, 1.0), 0.0, 3.9 * s, -3.9 * s, 4.1 * s, -4.1 * s
        ]
        n = base.size
        u, v, _ = make_channel(CASES[0].params, n, 5.0)
        _patch_residuals(monkeypatch, lambda call, size: 100.0 + base)
        fit = fit_ellipse(u, v, FitOptions(clip_k=clip_k))
        expected = np.ones(n, dtype=np.bool_)
        expected[rejected] = False
        np.testing.assert_array_equal(fit.mask_used, expected)
        assert fit.n_iter == 1 and fit.n_rejected == len(rejected)

    @pytest.mark.parametrize(("n_out", "flagged"), [(50, False), (51, True)])
    def test_high_rejection_threshold_is_exclusive(
        self, monkeypatch: pytest.MonkeyPatch, n_out: int, flagged: bool
    ) -> None:
        # 1000 events: exactly 5 % rejected is not "more than 5 %"; 5 % + 1 event is.
        def provider(call: int, n: int) -> FloatArray:
            res = np.linspace(-1.0, 1.0, n)
            res[n - n_out :] = 1000.0
            return res

        u, v, _ = make_channel(CASES[0].params, 1000, 5.0)
        _patch_residuals(monkeypatch, provider)
        fit = fit_ellipse(u, v, FitOptions(high_rejection_frac=0.05, broad_ring_frac=1.0))
        assert fit.n_rejected == n_out
        assert (FLAG_HIGH_REJECTION in fit.flags) is flagged

    def test_noiseless_robust_keeps_everything(self) -> None:
        # MAD ~ round-off: the scale floor must stop the clipping of good points.
        p = CASES[3].params
        x, y = ellipse_points(p, np.random.default_rng(2).uniform(0, 2 * np.pi, 2000))
        fit = fit_ellipse(x, y)
        assert fit.ok and fit.n_rejected == 0 and fit.n_iter == 0
        assert fit.params is not None
        assert fit.params.a == pytest.approx(p.a, rel=1e-10)
        assert fit.params.b == pytest.approx(p.b, rel=1e-10)

    def test_noiseless_with_gross_outliers(self) -> None:
        # More than half the residuals are ~0 (MAD ~ 0): exactly the outliers go.
        p = CASES[0].params
        x, y = ellipse_points(p, np.random.default_rng(4).uniform(0, 2 * np.pi, 1000))
        x = np.r_[x, [2000.0, 2100.0, 1700.0, 2500.0, 2050.0]]
        y = np.r_[y, [2000.0, 1900.0, 2300.0, 2100.0, 1650.0]]
        fit = fit_ellipse(x, y)
        assert fit.ok and fit.params is not None
        assert fit.n_rejected == 5
        assert fit.mask_used[:1000].all() and not fit.mask_used[1000:].any()
        assert fit.params.a == pytest.approx(p.a, rel=1e-9)
        assert fit.params.cx == pytest.approx(p.cx, abs=1e-7)

    def test_zero_mad_does_not_reject_everything(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force every residual to exactly 0 (MAD = 0).
        monkeypatch.setattr(
            ellipse, "residual_to_ellipse", lambda u, v, p: np.zeros(np.shape(u), dtype=np.float64)
        )
        u, v, _ = make_channel(CASES[0].params, 1000, 5.0)
        fit = fit_ellipse(u, v)
        assert fit.ok and fit.n_rejected == 0 and fit.mask_used.all()

    def test_robust_refit_failure_keeps_previous(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = ellipse._solve_conic
        calls = {"n": 0}

        def failing_after_first(scatter: FloatArray) -> EllipseParams | None:
            calls["n"] += 1
            return real(scatter) if calls["n"] == 1 else None

        monkeypatch.setattr(ellipse, "_solve_conic", failing_after_first)
        # Nothing is pre-clipped here (3 points 60 ADC outside the ring are well inside the
        # pre-clip window but beyond the 4-sigma residual clip), so the kept first fit is
        # the plain fit of all points.
        p = CASES[0].params
        u, v, _ = make_channel(p, 2000, 5.0)
        x3, y3 = ellipse_points(EllipseParams(p.cx, p.cy, p.a + 60, p.b + 60, p.phi), [0, 2, 4])
        u[:3], v[:3] = x3, y3
        assert ellipse._preclip_mask(u, v) is None
        fit = fit_ellipse(u, v)
        assert fit.ok and FLAG_ROBUST_REFIT_FAILED in fit.flags
        assert fit.n_iter == 0 and fit.mask_used.all()
        monkeypatch.undo()
        assert fit.params == fit_ellipse_direct(u, v)
        # With uniform outliers the pre-clip seeds the first fit; that fit is kept.
        calls["n"] = 0
        monkeypatch.setattr(ellipse, "_solve_conic", failing_after_first)
        u, v, _ = make_channel(CASES[0].params, 2000, 5.0, outlier_frac=0.05)
        fit = fit_ellipse(u, v)
        monkeypatch.undo()
        assert fit.ok and FLAG_ROBUST_REFIT_FAILED in fit.flags and fit.n_iter == 0
        assert 0 < fit.n_rejected < 100  # the pre-clipped far outliers
        again = fit_ellipse_direct(u[fit.mask_used], v[fit.mask_used])
        assert fit.params is not None and again is not None
        assert _param_error(fit.params, again) < 1e-9 * again.a

    def test_geometric_refinement(self) -> None:
        case = CASES[1]
        u, v, _ = make_channel(case.params, case.n, case.noise, seed=9, outlier_frac=0.05)
        alg = fit_ellipse(u, v)
        geo = fit_ellipse(u, v, FitOptions(geometric=True))
        assert alg.params is not None and geo.params is not None
        assert geo.ok and FLAG_GEOMETRIC_REFIT_FAILED not in geo.flags
        _assert_close(geo.params, case.params, _param_tol(case.noise, case.n, case.params))
        np.testing.assert_array_equal(geo.mask_used, alg.mask_used)
        # It minimises the radial residuals of the kept points, so it cannot be worse.
        uk = np.asarray(u, dtype=np.float64)[geo.mask_used]
        vk = np.asarray(v, dtype=np.float64)[geo.mask_used]
        ss_geo = float(np.sum(residual_to_ellipse(uk, vk, geo.params) ** 2))
        ss_alg = float(np.sum(residual_to_ellipse(uk, vk, alg.params) ** 2))
        assert ss_geo <= ss_alg * (1.0 + 1e-9)

    def test_geometric_noiseless_exact(self) -> None:
        p = CASES[3].params
        x, y = ellipse_points(p, np.random.default_rng(6).uniform(0, 2 * np.pi, 1000))
        fit = fit_ellipse(x, y, FitOptions(geometric=True))
        assert fit.params is not None
        _assert_canonical(fit.params)
        assert fit.params.a == pytest.approx(p.a, rel=1e-9)
        assert abs(_phi_diff(fit.params.phi, p.phi)) < 1e-9

    def test_geometric_uses_analytic_jacobian(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = ellipse.least_squares
        seen: dict[str, Any] = {}

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(ellipse, "least_squares", spy)
        u, v, _ = make_channel(CASES[0].params, 2000, 5.0)
        assert fit_ellipse(u, v, FitOptions(geometric=True)).ok
        assert callable(seen.get("jac")) and seen.get("method") == "lm"

    def test_geometric_failure_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("optimizer exploded")

        monkeypatch.setattr(ellipse, "least_squares", boom)
        u, v, _ = make_channel(CASES[0].params, 2000, 5.0)
        fit = fit_ellipse(u, v, FitOptions(geometric=True))
        assert fit.ok and FLAG_GEOMETRIC_REFIT_FAILED in fit.flags
        assert fit.params == fit_ellipse(u, v).params

    def test_kept_scatter_matches_direct_sum(self) -> None:
        rng = np.random.default_rng(15)
        x, y = rng.normal(0.0, 1.0, 3000), rng.normal(0.0, 1.0, 3000)
        design = ellipse._design_matrix(x, y)
        scatter_all = design @ design.T
        for kept in (rng.random(3000) > 0.05, rng.random(3000) > 0.7):
            n_kept = int(kept.sum())
            direct = design[:, kept] @ design[:, kept].T
            got = ellipse._kept_scatter(design, scatter_all, kept, n_kept)
            np.testing.assert_allclose(got, direct, rtol=1e-11, atol=1e-9)
        # Few rejected: exactly the subtraction S_all - D_r D_r^T (no 6xN copy).
        kept = rng.random(3000) > 0.05
        rej = design[:, ~kept]
        got = ellipse._kept_scatter(design, scatter_all, kept, int(kept.sum()))
        np.testing.assert_array_equal(got, scatter_all - rej @ rej.T)
        # Gross far-away outliers carry most of the trace: summed directly, exactly.
        x[:5] = 1e3
        design = ellipse._design_matrix(x, y)
        kept = np.ones(3000, dtype=np.bool_)
        kept[:5] = False
        got = ellipse._kept_scatter(design, design @ design.T, kept, 2995)
        sub = design[:, kept]
        np.testing.assert_array_equal(got, sub @ sub.T)

    def test_robust_fit_equals_direct_fit_of_kept_points(self) -> None:
        u, v, _ = make_channel(CASES[1].params, 20_000, 4.0, outlier_frac=0.05, int16=True)
        fit = fit_ellipse(u, v)
        again = fit_ellipse_direct(u[fit.mask_used], v[fit.mask_used])
        assert fit.params is not None and again is not None
        assert _param_error(fit.params, again) < 1e-9 * fit.params.a
        assert abs(_phi_diff(fit.params.phi, again.phi)) < 1e-9

    def test_radial_jacobian_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(10)
        x, y = rng.normal(0.0, 1.0, 40), rng.normal(0.0, 1.0, 40)
        theta = np.array([0.1, -0.05, 1.1, 0.8, 0.3])
        jac = ellipse._radial_jacobian(x, y, theta)
        h = 1e-6
        for i in range(5):
            tp, tm = theta.copy(), theta.copy()
            tp[i] += h
            tm[i] -= h
            num = (
                residual_to_ellipse(x, y, EllipseParams(*tp))
                - residual_to_ellipse(x, y, EllipseParams(*tm))
            ) / (2 * h)
            np.testing.assert_allclose(jac[:, i], num, atol=1e-7)

    # -- statuses -------------------------------------------------------------

    def test_too_few_events(self) -> None:
        u, v, _ = make_channel(CASES[0].params, 99, 5.0)
        fit = fit_ellipse(u, v)
        assert fit.status == STATUS_TOO_FEW_EVENTS and not fit.ok
        assert fit.params is None and fit.n_used == 0 and fit.n_rejected == 0
        assert fit.n_events == 99 and fit.mask_used.shape == (99,) and not fit.mask_used.any()
        assert fit_ellipse(u, v, FitOptions(min_events=99)).ok

    def test_empty_input(self) -> None:
        fit = fit_ellipse(np.empty(0, np.int16), np.empty(0, np.int16), FitOptions(min_events=1))
        assert fit.status == STATUS_TOO_FEW_EVENTS and fit.mask_used.shape == (0,)

    @pytest.mark.parametrize("n", [1, 5])
    def test_fewer_than_six_points_fail(self, n: int) -> None:
        x, y = ellipse_points(CASES[0].params, np.linspace(0, 6, n))
        fit = fit_ellipse(x, y, FitOptions(min_events=1))
        assert fit.status == STATUS_FIT_FAILED and fit.params is None
        assert fit.flags == () and not fit.mask_used.any()

    def test_collinear_fails(self) -> None:
        x = np.arange(1500, 1800, dtype=np.int16)
        fit = fit_ellipse(x, (2 * x - 1000).astype(np.int16), FitOptions(min_events=6))
        assert fit.status == STATUS_FIT_FAILED

    def test_quantised_line(self) -> None:
        # Integer points along y = x/2 lie on two parallel lines, a degenerate conic.
        # The ellipse-specific fit returns a needle ellipse enclosing the band
        # (observed b/a ~ 1e-4); like the hyperbola arc, it must come back either
        # failed or flagged, never raise.
        x = np.arange(1500, 1800, dtype=np.int16)
        fit = fit_ellipse(x, (x // 2 + 1000).astype(np.int16), FitOptions(min_events=6))
        assert fit.status in (STATUS_FIT_FAILED, STATUS_OK)
        if fit.ok:
            assert FLAG_EXTREME_AXIS_RATIO in fit.flags

    def test_identical_points_fail(self) -> None:
        fit = fit_ellipse(np.full(200, 2000, np.int16), np.full(200, 2000, np.int16))
        assert fit.status == STATUS_FIT_FAILED

    @pytest.mark.parametrize("noise", [0.0, 2.0, 10.0, 30.0])
    def test_hyperbola_like_arc(self, noise: float) -> None:
        # The Fitzgibbon fit is ellipse-specific: on a hyperbola branch it returns the
        # best ellipse, whose centre lies far outside the arc (observed: ~740 ADC to
        # the right of the vertex, noise-free). Either a failure or a flagged fit is
        # acceptable; with more noise the fit degrades into an elongated or broad one.
        t = np.linspace(-1.0, 1.0, 300)
        rng = np.random.default_rng(13)
        x = 2000.0 + 300.0 * np.cosh(t) + rng.normal(0.0, noise, t.size)
        y = 2000.0 + 400.0 * np.sinh(t) + rng.normal(0.0, noise, t.size)
        fit = fit_ellipse(x, y, FitOptions(min_events=6))
        _assert_failed_or_flagged(fit)
        direct = fit_ellipse_direct(x, y)
        assert (direct is None) == (fit.status == STATUS_FIT_FAILED)

    def test_gaussian_blob(self) -> None:
        # A dead or pedestal-only channel: a 2-D blob, not a ring.
        rng = np.random.default_rng(14)
        fit = fit_ellipse(rng.normal(2000, 50, 5000), rng.normal(2000, 50, 5000))
        _assert_failed_or_flagged(fit)
        if fit.ok:
            assert FLAG_BROAD_RING in fit.flags

    @pytest.mark.parametrize("n_distinct", [3, 4])
    def test_few_distinct_points_repeated(self, n_distinct: int) -> None:
        # With fewer than 5 distinct points a whole pencil of conics passes through all of
        # them exactly; the solution is not unique.
        xs = np.array([1900, 2100, 2000, 2050], dtype=np.int16)[:n_distinct]
        ys = np.array([2000, 2050, 2600, 1800], dtype=np.int16)[:n_distinct]
        u, v = np.tile(xs, 200), np.tile(ys, 200)
        fit = fit_ellipse(u, v)
        _assert_failed_or_flagged(fit)
        assert fit.status == STATUS_FIT_FAILED
        assert fit_ellipse_direct(u, v) is None

    def test_rank_check_rejects_pencil(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 4 distinct points in general position: without the rank check the eigenvector is
        # round-off noise and a bogus ellipse (a ~313, b ~80) comes back.
        xs = np.tile(np.array([2136, 2011, 1769, 1807], dtype=np.int16), 200)
        ys = np.tile(np.array([1540, 1575, 1516, 1675], dtype=np.int16), 200)
        assert fit_ellipse_direct(xs, ys) is None
        assert fit_ellipse(xs, ys).status == STATUS_FIT_FAILED
        monkeypatch.setattr(ellipse, "_SR_RANK_TOL", 0.0)
        assert fit_ellipse_direct(xs, ys) is not None

    def test_rank_check_keeps_noise_free_short_arc(self) -> None:
        # A noise-free 0.1 rad (5.7 degree) arc is a legitimate, unique fit. Its middle
        # eigenvalue / trace is ~1e-8, a factor 100 from the 1e-10 threshold either way.
        p = EllipseParams(2000.0, 2000.0, 600.0, 550.0, 0.3)
        x, y = ellipse_points(p, np.random.default_rng(1).uniform(0.0, 0.1, 3000))
        got = fit_ellipse_direct(x, y)
        assert got is not None
        assert got.a == pytest.approx(600.0, abs=0.05) and got.b == pytest.approx(550.0, abs=0.05)

    def test_five_distinct_points_repeated_is_unique(self) -> None:
        p = CASES[0].params
        x, y = ellipse_points(p, np.array([0.3, 1.5, 2.6, 3.9, 5.1]))
        fit = fit_ellipse(np.tile(x, 100), np.tile(y, 100))
        assert fit.ok and fit.params is not None
        assert fit.params.a == pytest.approx(p.a, rel=1e-8)
        assert fit.params.b == pytest.approx(p.b, rel=1e-8)

    def test_non_finite_points_are_excluded(self) -> None:
        u, v, _ = make_channel(CASES[0].params, 1000, 5.0)
        u = np.asarray(u, dtype=np.float64)
        u[[3, 500]] = [math.nan, math.inf]
        fit = fit_ellipse(u, v)
        assert fit.ok and not fit.mask_used[3] and not fit.mask_used[500]
        assert fit.n_rejected >= 2 and fit.n_used + fit.n_rejected == 1000
        clean = fit_ellipse(np.delete(u, [3, 500]), np.delete(np.asarray(v), [3, 500]))
        assert fit.params == clean.params

    # -- flags ----------------------------------------------------------------

    def test_high_rejection_flag(self) -> None:
        p = CASES[0].params
        u, v, _ = make_channel(p, 5000, 5.0, outlier_frac=0.10, int16=True)
        fit = fit_ellipse(u, v)
        assert fit.n_rejected > 0.05 * 5000
        assert fit.flags == (FLAG_HIGH_REJECTION,)
        lax = fit_ellipse(u, v, FitOptions(high_rejection_frac=0.2))
        assert FLAG_HIGH_REJECTION not in lax.flags

    def test_extreme_axis_ratio_flag(self) -> None:
        p = EllipseParams(2000.0, 2000.0, 600.0, 250.0, 0.2)
        u, v, _ = make_channel(p, 3000, 4.0, int16=True)
        fit = fit_ellipse(u, v)
        assert fit.ok and fit.flags == (FLAG_EXTREME_AXIS_RATIO,)
        assert fit_ellipse(u, v, FitOptions(extreme_axis_ratio=0.4)).flags == ()

    def test_center_outside_data_flag(self) -> None:
        p = EllipseParams(2000.0, 2000.0, 600.0, 500.0, 0.3)
        x, y = ellipse_points(p, np.linspace(-0.5, 0.5, 400))  # a 57-degree arc
        fit = fit_ellipse(x, y)
        assert fit.ok and fit.params is not None
        assert fit.params.a == pytest.approx(600.0, rel=1e-8)
        assert fit.flags == (FLAG_CENTER_OUTSIDE_DATA,)

    def test_broad_ring_flag(self) -> None:
        # A real-like ring (sigma 5 on R ~ 590): robust spread / sqrt(ab) ~ 0.0085.
        u, v, _ = make_channel(CASES[0].params, 5000, 5.0, int16=True)
        assert FLAG_BROAD_RING not in fit_ellipse(u, v).flags
        assert FLAG_BROAD_RING not in fit_ellipse(u, v, FitOptions(broad_ring_frac=0.02)).flags
        assert FLAG_BROAD_RING in fit_ellipse(u, v, FitOptions(broad_ring_frac=0.005)).flags
        # A thick ring: sigma 90 on R ~ 590 gives ~0.15 > 0.1.
        u, v, _ = make_channel(CASES[0].params, 5000, 90.0)
        assert FLAG_BROAD_RING in fit_ellipse(u, v).flags

    def test_broad_ring_uses_robust_spread(self) -> None:
        # Robust fit off, so every point is kept: 3 % of the events on a circle of 3R around
        # the centre. Their residuals (~2R) would dominate a standard deviation (~0.35 R),
        # but the MAD-based spread still sees a thin ring.
        p = CASES[0].params
        u, v, _ = make_channel(p, 10_000, 5.0)
        k = 300
        t = np.random.default_rng(3).uniform(0.0, 2 * np.pi, k)
        u[:k] = p.cx + 3.0 * p.target_radius * np.cos(t)
        v[:k] = p.cy + 3.0 * p.target_radius * np.sin(t)
        fit = fit_ellipse(u, v, FitOptions(robust=False))
        assert fit.ok and fit.mask_used.all()
        assert FLAG_BROAD_RING not in fit.flags
        res = residual_to_ellipse(u, v, fit.params)
        assert np.std(res) > 0.2 * fit.params.target_radius  # a std would have flagged it

    def test_broad_ring_uses_kept_points_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Patched residuals: 60 % inliers spread uniformly over [-1, 1] (kept; robust spread
        # 1.4826 * 0.5 = 0.74) and 40 % at 1000 (rejected). Over all points the MAD-based
        # spread would be 1.4826 * 1.333 = 1.98. A threshold between the two separates
        # "kept points" from "all points".
        n = 2000
        n_in = 1200

        def provider(call: int, size: int) -> FloatArray:
            res = np.full(size, 1000.0)
            res[:n_in] = np.linspace(-1.0, 1.0, n_in)
            return res

        u, v, _ = make_channel(CASES[0].params, n, 5.0)
        uf, vf = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
        s = math.sqrt(float(np.mean((uf - uf.mean()) ** 2 + (vf - vf.mean()) ** 2)))
        _patch_residuals(monkeypatch, provider)
        probe = fit_ellipse(u, v)
        assert probe.params is not None and probe.n_used == n_in
        r_n = probe.params.target_radius / s  # the fitted radius in the normalised frame
        between = FitOptions(broad_ring_frac=1.36 / r_n)
        assert FLAG_BROAD_RING not in fit_ellipse(u, v, between).flags
        below = FitOptions(broad_ring_frac=0.6 / r_n)
        assert FLAG_BROAD_RING in fit_ellipse(u, v, below).flags

    def test_far_cluster_does_not_capture_the_fit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 6 % of the events at (0, 0) with the ring at (2000, 2000), R ~ 430: without the
        # pre-clip the first algebraic fit is captured and the robust loop never recovers.
        p = EllipseParams(2000.0, 2000.0, 440.0, 420.0, 0.3)
        u, v, _ = make_channel(p, 20_000, 5.0, int16=True)
        u[:1200] = 0
        v[:1200] = 0
        fit = fit_ellipse(u, v)
        assert fit.ok and fit.params is not None
        _assert_close(fit.params, p, _param_tol(5.0, 18_800, p))
        assert not fit.mask_used[:1200].any()
        assert FLAG_HIGH_REJECTION in fit.flags
        monkeypatch.setattr(ellipse, "_preclip_mask", lambda x, y: None)
        captured = fit_ellipse(u, v)
        assert captured.params is None or _param_error(captured.params, p) > 10.0

    def test_int16_corner_cluster(self) -> None:
        p = EllipseParams(2000.0, 2000.0, 440.0, 420.0, 0.3)
        u, v, _ = make_channel(p, 20_000, 5.0, int16=True)
        u[:100] = 32767
        v[:100] = 32767
        fit = fit_ellipse(u, v)
        assert fit.ok and fit.params is not None
        _assert_close(fit.params, p, _param_tol(5.0, 19_900, p))
        assert fit.flags == ()

    def test_preclip_leaves_rings_and_arcs_alone(self) -> None:
        # Clean rings (noisy or noise-free, circular or not) and partial arcs: nothing to cut.
        for p in (CASES[0].params, CASES[5].params, EllipseParams(0.0, 0.0, 1.0, 1.0, 0.0)):
            x, y = ellipse_points(p, np.random.default_rng(4).uniform(0, 2 * np.pi, 5000))
            assert ellipse._preclip_mask(x, y) is None
            u, v, _ = make_channel(p, 5000, 0.02 * p.a, int16=p.a > 100)
            assert ellipse._preclip_mask(np.asarray(u, float), np.asarray(v, float)) is None
        p = EllipseParams(2000.0, 2000.0, 600.0, 500.0, 0.3)
        for arc in (0.1, 1.0, 3.0):
            x, y = ellipse_points(p, np.random.default_rng(5).uniform(0, arc, 3000))
            assert ellipse._preclip_mask(x, y) is None

    def test_preclip_mad_floor(self) -> None:
        # 60 % of the points exactly on the unit circle (at the 4 axis points, so the median
        # point is the centre and MAD(r) = 0) and 40 % within 1e-4 of it: the floor on the
        # MAD (1e-3 of the median radius) keeps the pre-clip from cutting the 40 %.
        pts = np.tile(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]), (1500, 1))
        rng = np.random.default_rng(6)
        t = rng.uniform(0.0, 2 * np.pi, 4000)
        r = 1.0 + rng.normal(0.0, 1e-4, 4000)
        x = np.r_[pts[:, 0], r * np.cos(t)]
        y = np.r_[pts[:, 1], r * np.sin(t)]
        assert ellipse._preclip_mask(x, y) is None

    def test_preclipped_points_can_come_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The pre-clip only seeds the first fit: a point it dropped that lies on the fitted
        # ellipse is kept by the robust iteration.
        p = CASES[0].params
        u, v, _ = make_channel(p, 3000, 5.0)
        seeded = np.ones(3000, dtype=np.bool_)
        seeded[:40] = False  # pretend the pre-clip dropped 40 good ring points
        monkeypatch.setattr(ellipse, "_preclip_mask", lambda x, y: seeded)
        fit = fit_ellipse(u, v)
        assert fit.ok and fit.mask_used[:40].all()

    def test_flags_are_canonical(self) -> None:
        p = EllipseParams(2000.0, 2000.0, 600.0, 250.0, 0.2)
        u, v, _ = make_channel(p, 3000, 4.0, outlier_frac=0.10)
        fit = fit_ellipse(u, v)
        assert set(fit.flags) == {FLAG_HIGH_REJECTION, FLAG_EXTREME_AXIS_RATIO}
        assert list(fit.flags) == [f for f in FLAGS if f in fit.flags]


# ---------------------------------------------------------------------------
# Correction and per-point quantities
# ---------------------------------------------------------------------------

CORRECTION_PARAMS = [
    EllipseParams(2000.0, 2000.0, 620.0, 560.0, 0.4),
    EllipseParams(2030.0, 1980.0, 640.0, 450.0, -1.55),
    EllipseParams(1990.0, 2010.0, 650.0, 460.0, math.pi / 2),
    EllipseParams(2000.0, 2000.0, 400.0, 600.0, 2.9),  # not canonical
]


class TestEigenvectorSelection:
    """``_select_ellipse_eigvec``: the smallest valid eigenvalue wins, in any order."""

    # Columns: two vectors satisfy 4AC - B^2 > 0, (1, 0, 1) and (1, 0, 2); (0, 1, 0) does not.
    VECS = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 2.0], [0.0, 1.0, 0.0]]).T

    @staticmethod
    def _select(vals: list[complex], order: tuple[int, ...]) -> FloatArray | None:
        eigval = np.array([vals[i] for i in order], dtype=np.complex128)
        eigvec = TestEigenvectorSelection.VECS[:, list(order)].astype(np.complex128)
        return ellipse._select_ellipse_eigvec(eigval, eigvec)

    @pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
    def test_smallest_valid_eigenvalue(self, order: tuple[int, ...]) -> None:
        got = self._select([2.0, 1.0, 0.5], order)  # (0, 1, 0) has the smallest but is invalid
        np.testing.assert_array_equal(got, [1.0, 0.0, 2.0])
        got = self._select([1.0, 2.0, 0.5], order)
        np.testing.assert_array_equal(got, [1.0, 0.0, 1.0])

    @pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
    def test_roundoff_negative_accepted_real_negative_rejected(
        self, order: tuple[int, ...]
    ) -> None:
        np.testing.assert_array_equal(self._select([2.0, -1e-12, 5.0], order), [1.0, 0.0, 2.0])
        np.testing.assert_array_equal(self._select([2.0, -1.0, 5.0], order), [1.0, 0.0, 1.0])

    def test_complex_pairs_skipped_and_none(self) -> None:
        got = self._select([2.0, 1.0 + 1e-3j, 5.0], (0, 1, 2))
        np.testing.assert_array_equal(got, [1.0, 0.0, 1.0])
        assert self._select([-2.0, -1.0, 5.0], (0, 1, 2)) is None


class TestCorrection:
    @pytest.mark.parametrize("p", CORRECTION_PARAMS)
    def test_round_trip(self, p: EllipseParams) -> None:
        rng = np.random.default_rng(20)
        u = rng.uniform(1200, 2800, 1000)
        v = rng.uniform(1200, 2800, 1000)
        uc, vc = correct(u, v, p)
        ub, vb = uncorrect(uc, vc, p)
        np.testing.assert_allclose(ub, u, rtol=0, atol=1e-9)
        np.testing.assert_allclose(vb, v, rtol=0, atol=1e-9)

    def test_round_trip_int16(self) -> None:
        p = CORRECTION_PARAMS[0]
        u, v, _ = make_channel(p, 500, 5.0, int16=True)
        ub, vb = uncorrect(*correct(u, v, p), p)
        np.testing.assert_allclose(ub, u.astype(np.float64), atol=1e-9)
        np.testing.assert_allclose(vb, v.astype(np.float64), atol=1e-9)

    @pytest.mark.parametrize("p", CORRECTION_PARAMS)
    def test_noiseless_maps_to_circle(self, p: EllipseParams) -> None:
        x, y = ellipse_points(p, np.linspace(0, 2 * np.pi, 720, endpoint=False))
        uc, vc = correct(x, y, p)
        np.testing.assert_allclose(np.hypot(uc, vc), p.target_radius, rtol=1e-12)
        np.testing.assert_allclose(corrected_residual(uc, vc, p), 0.0, atol=1e-9)
        # Evenly spaced parametric angles: the corrected points are centred on the origin.
        assert abs(uc.mean()) < 1e-9 and abs(vc.mean()) < 1e-9

    def test_center_maps_to_origin_and_axes_scale(self) -> None:
        p = CORRECTION_PARAMS[1]
        uc, vc = correct(np.array([p.cx]), np.array([p.cy]), p)
        assert abs(uc[0]) < 1e-12 and abs(vc[0]) < 1e-12
        # The end of the a axis goes to distance sqrt(ab) along the same direction.
        end_u = p.cx + p.a * math.cos(p.phi)
        end_v = p.cy + p.a * math.sin(p.phi)
        uc, vc = correct(np.array([end_u]), np.array([end_v]), p)
        assert math.hypot(uc[0], vc[0]) == pytest.approx(p.target_radius)
        assert math.atan2(vc[0], uc[0]) == pytest.approx(p.phi)

    def test_matches_cpp_formula(self) -> None:
        # Plan 4.3, step by step, as in RadialAnalysis applyEllipseCorrection.
        p = CORRECTION_PARAMS[1]
        rng = np.random.default_rng(21)
        u, v = rng.uniform(1300, 2700, 200), rng.uniform(1300, 2700, 200)
        c, s = math.cos(p.phi), math.sin(p.phi)
        du, dv = u - p.cx, v - p.cy
        ur = (du * c + dv * s) * (math.sqrt(p.a * p.b) / p.a)
        vr = (-du * s + dv * c) * (math.sqrt(p.a * p.b) / p.b)
        uc, vc = correct(u, v, p)
        np.testing.assert_allclose(uc, ur * c - vr * s, rtol=0, atol=1e-10)
        np.testing.assert_allclose(vc, ur * s + vr * c, rtol=0, atol=1e-10)

    def test_parametrisation_independent(self) -> None:
        p = CORRECTION_PARAMS[3]
        u, v, _ = make_channel(p.canonical(), 300, 5.0)
        a1 = correct(u, v, p)
        a2 = correct(u, v, p.canonical())
        np.testing.assert_allclose(a1[0], a2[0], atol=1e-9)
        np.testing.assert_allclose(a1[1], a2[1], atol=1e-9)

    @pytest.mark.parametrize("case", CASES[:3], ids=CASE_IDS[:3])
    def test_noisy_corrected_mean_radius(self, case: Case) -> None:
        u, v, _ = make_channel(case.params, case.n, case.noise, seed=22, int16=True)
        fit = fit_ellipse(u, v)
        assert fit.params is not None
        uc, vc = correct(u, v, fit.params)
        mean_r = float(np.mean(np.hypot(uc, vc)))
        tol = 5.0 * case.noise / math.sqrt(case.n) + case.noise**2 / case.params.target_radius
        assert mean_r == pytest.approx(case.params.target_radius, abs=tol)
        assert mean_r == pytest.approx(fit.params.target_radius, abs=tol)


class TestResiduals:
    def test_zero_on_ellipse(self) -> None:
        p = CORRECTION_PARAMS[1]
        x, y = ellipse_points(p, np.linspace(0, 2 * np.pi, 360))
        np.testing.assert_allclose(residual_to_ellipse(x, y, p), 0.0, atol=1e-9)

    def test_known_radial_offset(self) -> None:
        p = CORRECTION_PARAMS[0]
        x, y = ellipse_points(p, np.linspace(0, 2 * np.pi, 360))
        dx, dy = x - p.cx, y - p.cy
        r = np.hypot(dx, dy)
        for offset in (-7.5, 3.0):
            xs, ys = p.cx + dx * (1 + offset / r), p.cy + dy * (1 + offset / r)
            np.testing.assert_allclose(residual_to_ellipse(xs, ys, p), offset, atol=1e-9)

    def test_matches_trig_formula(self) -> None:
        # Plan 4.4 with the explicit angle, as in main_radial.cpp.
        p = CORRECTION_PARAMS[1]
        rng = np.random.default_rng(23)
        u, v = rng.uniform(1300, 2700, 500), rng.uniform(1300, 2700, 500)
        c, s = math.cos(p.phi), math.sin(p.phi)
        du, dv = u - p.cx, v - p.cy
        ua, va = du * c + dv * s, -du * s + dv * c
        th = np.arctan2(va, ua)
        r_ell = p.a * p.b / np.sqrt((p.b * np.cos(th)) ** 2 + (p.a * np.sin(th)) ** 2)
        expected = np.sqrt(ua**2 + va**2) - r_ell
        np.testing.assert_allclose(residual_to_ellipse(u, v, p), expected, rtol=0, atol=1e-9)

    def test_point_at_center(self) -> None:
        p = CORRECTION_PARAMS[0]
        res = residual_to_ellipse(np.array([p.cx, p.cx + 1.0]), np.array([p.cy, p.cy]), p)
        assert res[0] == -p.a
        assert np.isfinite(res[1])

    def test_broadcast_and_scalar_like(self) -> None:
        p = CORRECTION_PARAMS[0]
        res = residual_to_ellipse(np.array([[p.cx + 700.0]]), np.array([p.cy]), p)
        assert res.shape == (1, 1)

    def test_radii_about_center(self) -> None:
        p = CORRECTION_PARAMS[0]
        u = np.array([p.cx + 3.0, p.cx])
        v = np.array([p.cy + 4.0, p.cy - 2.0])
        np.testing.assert_allclose(radii_about_center(u, v, p), [5.0, 2.0])

    @pytest.mark.parametrize(
        "helper", [correct, uncorrect, residual_to_ellipse, radii_about_center, corrected_residual]
    )
    def test_scalar_inputs(self, helper: Callable[..., Any]) -> None:
        p = CORRECTION_PARAMS[1]
        scalar = helper(2000.0, 2000.0, p)
        vector = helper(np.array([2000.0]), np.array([2000.0]), p)
        for s_out, v_out in zip(
            scalar if isinstance(scalar, tuple) else (scalar,),
            vector if isinstance(vector, tuple) else (vector,),
        ):
            assert isinstance(s_out, np.ndarray) and s_out.shape == ()
            assert s_out.dtype == np.float64
            assert float(s_out) == float(v_out[0])
        numpy_scalar = helper(np.float64(2000.0), np.int16(2000), p)
        assert np.shape(numpy_scalar[0] if isinstance(numpy_scalar, tuple) else numpy_scalar) == ()

    def test_ellipse_points_scalar(self) -> None:
        p = CORRECTION_PARAMS[0]
        x, y = ellipse_points(p, 0.7)
        assert x.shape == () and y.shape == ()
        xs, ys = ellipse_points(p, np.array([0.7]))
        assert float(x) == float(xs[0]) and float(y) == float(ys[0])

    @pytest.mark.parametrize(
        "helper", [correct, uncorrect, residual_to_ellipse, radii_about_center, corrected_residual]
    )
    def test_infinite_inputs_are_quiet(self, helper: Callable[..., Any]) -> None:
        # Module-level filterwarnings("error") turns any RuntimeWarning into a failure.
        p = CORRECTION_PARAMS[0]
        out = helper(
            np.array([np.inf, -np.inf, np.nan, 2000.0]), np.array([1.0, np.inf, 0, 2100]), p
        )
        for arr in out if isinstance(out, tuple) else (out,):
            assert not np.isfinite(arr[:3]).any()
            assert np.isfinite(arr[3])
        ellipse_points(p, np.array([np.inf, np.nan]))

    def test_ellipse_points_on_implicit_curve(self) -> None:
        p = CORRECTION_PARAMS[1]
        x, y = ellipse_points(p, np.linspace(0, 2 * np.pi, 100))
        c, s = math.cos(p.phi), math.sin(p.phi)
        ua = (x - p.cx) * c + (y - p.cy) * s
        va = -(x - p.cx) * s + (y - p.cy) * c
        np.testing.assert_allclose((ua / p.a) ** 2 + (va / p.b) ** 2, 1.0, rtol=1e-12)
