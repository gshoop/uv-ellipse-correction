"""Ellipse fitting and the ellipse-to-circle correction.

Implements plan sections 4.2-4.4, 5.1 and 5.2:

- ``fit_ellipse_direct``: the direct least-squares ellipse fit of Fitzgibbon,
  Pilu and Fisher (1999) in the numerically stable reduced form of Halir and
  Flusser (1998), on normalised coordinates.
- ``fit_ellipse``: the per-channel fit. Adds the event-count check, the robust
  clip-and-refit iteration, the optional geometric refinement, the canonical
  form and the warning flags.
- ``correct`` / ``uncorrect``: the RadialAnalysis ellipse-to-circle transform
  (plan 4.3) and its inverse.
- ``residual_to_ellipse``, ``corrected_residual`` and ``radii_about_center``:
  the per-point quantities the metrics are computed from (plan 4.4 and 5.3).

Normalisation: every fit works on ``x = (U - mean U) / s`` and
``y = (V - mean V) / s``, where ``s`` is the RMS distance of the points from
their mean. On raw coordinates (U ~ 2000, R ~ 600) the 6x6 scatter matrix mixes
entries around 1e13 with entries around N and is badly conditioned. The fit is
exactly equivariant under this similarity transform (the algebraic residuals
are unchanged and the constraint scales by ``s**4``), so the normalisation is
undone on the *geometric* parameters: ``cx = mean U + s * cx_n``,
``cy = mean V + s * cy_n``, ``a = s * a_n``, ``b = s * b_n`` and ``phi``
unchanged. This is mathematically identical to transforming the conic
coefficients back and converting in raw coordinates, but avoids the large
intermediate values that conversion would reintroduce.

Performance: each fit builds one 6xN design matrix and one 6x6 scatter
matrix. The robust iteration reuses the design matrix: it keeps the
normalisation of the full point set, which is exact by the equivariance above.
It gets the kept points' scatter matrix by subtracting the few rejected
columns' scatter from the full one (see ``_kept_scatter``). Everything is
vectorised and never touches the process-wide warning filters (floating-point
warnings are silenced with the thread-local ``np.errstate``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from uvcorr.metrics import MAD_TO_SIGMA
from uvcorr.options import (
    FLAG_BROAD_RING,
    FLAG_CENTER_OUTSIDE_DATA,
    FLAG_EXTREME_AXIS_RATIO,
    FLAG_GEOMETRIC_REFIT_FAILED,
    FLAG_HIGH_REJECTION,
    FLAG_ROBUST_REFIT_FAILED,
    STATUS_FIT_FAILED,
    STATUS_OK,
    STATUS_TOO_FEW_EVENTS,
    FitOptions,
    order_flags,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

MIN_FIT_POINTS = 6
"""The algebraic fit needs at least this many points (a conic has 5 degrees of freedom)."""

# Inverse of the Fitzgibbon constraint matrix C1 = [[0, 0, 2], [0, -1, 0], [2, 0, 0]]
# (the constraint 4AC - B^2 = 1 restricted to the quadratic coefficients).
_C1_INV: FloatArray = np.array([[0.0, 0.0, 0.5], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]])

# S3 = D2^T D2 on normalised coordinates is N * [[cov_xx, cov_xy, 0], [cov_xy, cov_yy, 0],
# [0, 0, 1]] with cov_xx + cov_yy = 1, so its condition number is 1 / (smallest principal
# variance) for any sane point cloud (about 100 at b/a = 0.1). Collinear points make it
# singular up to round-off (condition ~1e16); the threshold separates the two.
_S3_MAX_COND = 1e10
# Minimum |B^2 - 4AC| and |A'|, |C'| for a unit-norm (A, B, C), as in the C++ (b/a ~ 1e-6).
_DISC_MIN = 1e-12
_AXIS_COEFF_MIN = 1e-12
# Eigen-output with an imaginary part above this (relative) size is not trusted.
_EIG_IMAG_TOL = 1e-8
# A constraint-satisfying eigenvalue may be negative by this much (relative to the largest
# eigenvalue magnitude) from round-off: noise-free points give an exact eigenvalue of 0.
_EIG_NEG_TOL = 1e-8
# The reduced scatter matrix S1 - S2 S3^-1 S2^T must have rank >= 2 (at most one exact conic
# through the data): its middle eigenvalue must exceed this fraction of trace(S). With fewer
# than 5 distinct points (e.g. 3 or 4 points repeated) a whole pencil of conics fits exactly,
# the ratio sits at round-off (~1e-17) and the eigenvectors are noise. Noisy data are far
# above the threshold: full rings ~4e-2, noisy arcs of 2-120 degrees ~2e-3 to 9e-2 at
# 5 ADC noise, and down to ~5e-5 for 60-degree arcs with little noise. Only noise-free short
# arcs come near it (0.035 rad: 1.4e-10, accepted; 0.01 rad: 2e-12, rejected).
_SR_RANK_TOL = 1e-10
PRECLIP_K = 10.0
"""Robust pre-clip width, in robust sigmas of the radius about the median point (see
``fit_ellipse``)."""
# The pre-clip's robust sigma is floored at this fraction of the median radius, so a
# noise-free circle (MAD ~ 0) is never pre-clipped.
_PRECLIP_MAD_FLOOR_REL = 1e-3
# The pre-clip's medians are taken on a strided subsample of at most this many points.
_PRECLIP_SAMPLE = 65536
# Floor on the robust residual scale, relative to the fitted radius, so that points on a
# noise-free ellipse (MAD ~ round-off, or exactly 0) are never clipped.
_ROBUST_SCALE_FLOOR_REL = 1e-9


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def _wrap_half_turn(phi: float) -> float:
    """Wrap an axis angle into (-pi/2, pi/2] (an ellipse is symmetric under pi)."""
    wrapped = math.remainder(phi, math.pi)  # exact, in [-pi/2, pi/2]
    if wrapped <= -math.pi / 2:
        wrapped += math.pi
    return wrapped


@dataclass(frozen=True)
class EllipseParams:
    """Geometric ellipse parameters in (U, V) coordinates.

    The ellipse is the set of points ``(cx, cy) + R(phi) (a cos t, b sin t)``,
    where ``R(phi)`` is the rotation by ``phi``. Parameters returned by the
    fits are in canonical form (see ``canonical``); a directly constructed
    instance is stored as given.

    Attributes:
        cx: Centre U coordinate.
        cy: Centre V coordinate.
        a: Semi-axis along the direction ``phi`` (the semi-major axis in
            canonical form).
        b: Semi-axis perpendicular to it (the semi-minor axis in canonical
            form).
        phi: Angle of the ``a`` axis from the U axis, in radians.
    """

    cx: float
    cy: float
    a: float
    b: float
    phi: float

    @property
    def target_radius(self) -> float:
        """Radius of the corrected circle, the geometric mean sqrt(a*b)."""
        return math.sqrt(self.a * self.b)

    @property
    def axis_ratio(self) -> float:
        """Axis ratio b/a (in (0, 1] in canonical form)."""
        return self.b / self.a if self.a != 0 else math.nan

    def canonical(self) -> EllipseParams:
        """Return the same ellipse in canonical form.

        Canonical form has ``a >= b > 0`` (absolute values are taken) and
        ``phi`` in (-pi/2, pi/2]. When ``a < b`` the axes are swapped and pi/2 is
        added to ``phi`` before wrapping.

        Returns:
            The canonical parameters.
        """
        a, b, phi = abs(self.a), abs(self.b), self.phi
        if a < b:
            a, b = b, a
            phi += math.pi / 2
        return EllipseParams(
            cx=float(self.cx), cy=float(self.cy), a=float(a), b=float(b), phi=_wrap_half_turn(phi)
        )


@dataclass(frozen=True, eq=False)
class EllipseFit:
    """Result of ``fit_ellipse`` for one channel.

    Attributes:
        status: One of ``STATUS_OK``, ``STATUS_TOO_FEW_EVENTS`` or
            ``STATUS_FIT_FAILED`` (``uvcorr.options``).
        params: Canonical ellipse parameters; None unless ``status`` is ok.
        mask_used: Boolean array over the input points, True for the points
            the final fit used (the kept set of the robust iteration). All
            False unless ``status`` is ok. Non-finite input points are never
            used.
        n_events: Number of input points.
        n_used: Number of points the final fit used (``mask_used.sum()``);
            0 unless ``status`` is ok.
        n_rejected: ``n_events - n_used`` when ``status`` is ok (robust
            rejections plus any non-finite points), else 0.
        n_iter: Number of accepted robust clip-and-refit iterations (0 with
            ``robust`` off or when the first clipping pass rejected nothing).
        flags: Warning flags in canonical order (``uvcorr.options.FLAGS``).
    """

    status: str
    params: EllipseParams | None
    mask_used: BoolArray
    n_events: int
    n_used: int
    n_rejected: int
    n_iter: int
    flags: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True if the fit succeeded (``status == "ok"``)."""
        return self.status == STATUS_OK


# ---------------------------------------------------------------------------
# Algebraic fit internals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Frame:
    """Normalising similarity transform ``x = (U - mu) / s``, ``y = (V - mv) / s``."""

    mu: float
    mv: float
    s: float

    def to_raw(self, p: EllipseParams) -> EllipseParams:
        """Map parameters fitted in the normalised frame back to raw coordinates."""
        return EllipseParams(
            cx=self.mu + self.s * p.cx,
            cy=self.mv + self.s * p.cy,
            a=self.s * p.a,
            b=self.s * p.b,
            phi=p.phi,
        )


def _make_frame(x: FloatArray, y: FloatArray) -> _Frame | None:
    """Return the normalising frame of finite points, or None if they coincide."""
    mu = float(np.mean(x))
    mv = float(np.mean(y))
    dx = x - mu
    dy = y - mv
    s = math.sqrt(float(np.mean(dx * dx + dy * dy)))
    if not (math.isfinite(mu) and math.isfinite(mv) and math.isfinite(s)) or s <= 0.0:
        return None
    return _Frame(mu=mu, mv=mv, s=s)


def _design_matrix(xn: FloatArray, yn: FloatArray) -> FloatArray:
    """Return the transposed design matrix, rows [x^2, xy, y^2, x, y, 1] (6 x N)."""
    design = np.empty((6, xn.size), dtype=np.float64)
    np.multiply(xn, xn, out=design[0])
    np.multiply(xn, yn, out=design[1])
    np.multiply(yn, yn, out=design[2])
    design[3] = xn
    design[4] = yn
    design[5] = 1.0
    return design


def _select_ellipse_eigvec(
    eigval: npt.NDArray[np.complex128] | FloatArray,
    eigvec: npt.NDArray[np.complex128] | FloatArray,
) -> FloatArray | None:
    """Pick the eigenvector (column) that satisfies the ellipse constraint 4AC - B^2 > 0.

    In exact arithmetic exactly one eigenvector does, and its eigenvalue is the only
    non-negative one. If round-off lets several through, the one with the smallest
    eigenvalue (non-negative up to round-off) wins. Eigenpairs with a non-negligible
    imaginary part are skipped.

    Args:
        eigval: The 3 eigenvalues.
        eigvec: The eigenvectors as columns (3x3).

    Returns:
        The real eigenvector (A, B, C), or None if none qualifies.
    """
    eig_scale = float(np.max(np.abs(eigval)))
    if not math.isfinite(eig_scale):
        return None
    best: FloatArray | None = None
    best_val = math.inf
    for i in range(3):
        val = complex(eigval[i])
        vec = eigvec[:, i]
        vec_scale = float(np.max(np.abs(vec)))
        if vec_scale == 0.0 or abs(val.imag) > _EIG_IMAG_TOL * max(eig_scale, 1e-300):
            continue
        if float(np.max(np.abs(np.imag(vec)))) > _EIG_IMAG_TOL * vec_scale:
            continue
        cand = np.real(vec).astype(np.float64)
        if 4.0 * cand[0] * cand[2] - cand[1] * cand[1] <= 0.0:
            continue
        if val.real < -_EIG_NEG_TOL * eig_scale:
            continue
        if val.real < best_val:
            best, best_val = cand, val.real
    return best


def _solve_conic(scatter: FloatArray) -> EllipseParams | None:
    """Solve the reduced Fitzgibbon problem for a 6x6 scatter matrix.

    Args:
        scatter: ``D D^T`` for the transposed design matrix ``D`` of
            normalised points.

    Returns:
        The ellipse in the same (normalised) coordinates, not canonicalised,
        or None if there is no valid ellipse.
    """
    if not np.all(np.isfinite(scatter)):
        return None
    s1 = scatter[:3, :3]
    s2 = scatter[:3, 3:]
    s3 = scatter[3:, 3:]
    try:
        if not np.linalg.cond(s3) < _S3_MAX_COND:  # also catches inf/NaN
            return None
        t_mat = -np.linalg.solve(s3, s2.T)
        reduced = s1 + s2 @ t_mat
        sr_eig = np.linalg.eigvalsh(0.5 * (reduced + reduced.T))  # ascending
        if not sr_eig[1] > _SR_RANK_TOL * float(np.trace(scatter)):
            return None  # not unique: a pencil of conics fits the data exactly
        eigval, eigvec = np.linalg.eig(_C1_INV @ reduced)
    except (np.linalg.LinAlgError, ValueError):
        return None

    best = _select_ellipse_eigvec(eigval, eigvec)
    if best is None:
        return None

    a1 = best / float(np.linalg.norm(best))
    a2 = t_mat @ a1
    coef_a, coef_b, coef_c = (float(c) for c in a1)
    coef_d, coef_e, coef_f = (float(c) for c in a2)

    disc = coef_b * coef_b - 4.0 * coef_a * coef_c
    if abs(disc) < _DISC_MIN:
        return None
    cx = (2.0 * coef_c * coef_d - coef_b * coef_e) / disc
    cy = (2.0 * coef_a * coef_e - coef_b * coef_d) / disc
    phi = 0.5 * math.atan2(coef_b, coef_a - coef_c)
    f_c = (
        coef_a * cx * cx + coef_b * cx * cy + coef_c * cy * cy + coef_d * cx + coef_e * cy + coef_f
    )
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    a_p = coef_a * cos_p * cos_p + coef_b * cos_p * sin_p + coef_c * sin_p * sin_p
    c_p = coef_a * sin_p * sin_p - coef_b * cos_p * sin_p + coef_c * cos_p * cos_p
    if abs(a_p) < _AXIS_COEFF_MIN or abs(c_p) < _AXIS_COEFF_MIN:
        return None
    a_sq = -f_c / a_p
    b_sq = -f_c / c_p
    if not (math.isfinite(a_sq) and math.isfinite(b_sq)) or a_sq <= 0.0 or b_sq <= 0.0:
        return None
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return None
    return EllipseParams(cx=cx, cy=cy, a=math.sqrt(a_sq), b=math.sqrt(b_sq), phi=phi)


def _as_float_1d(u: npt.ArrayLike, v: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Convert a (U, V) pair to float64 1-D arrays of equal length."""
    x = np.asarray(u, dtype=np.float64)
    y = np.asarray(v, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(
            f"u and v must be 1-D arrays of equal length, got shapes {x.shape} and {y.shape}"
        )
    return x, y


def fit_ellipse_direct(u: npt.ArrayLike, v: npt.ArrayLike) -> EllipseParams | None:
    """Fit an ellipse to all points by the direct algebraic method.

    Fitzgibbon's ellipse-specific least-squares fit (constraint
    ``4AC - B^2 = 1``) in the Halir-Flusser reduced form, on normalised
    coordinates (see the module docstring). No robust clipping.

    Args:
        u: U coordinates (any real dtype, e.g. the int16 cache columns).
        v: V coordinates, same length as ``u``.

    Returns:
        The canonical ellipse parameters, or None if no ellipse can be fitted:
        fewer than 6 points, any non-finite value, coincident or collinear
        points (singular ``S3``), a non-unique solution (the reduced scatter
        matrix has rank < 2, e.g. fewer than 5 distinct points), no
        eigenvector satisfying the constraint, a degenerate discriminant, or
        non-positive squared semi-axes. Never warns.

    Raises:
        ValueError: If ``u`` and ``v`` are not 1-D arrays of equal length.
    """
    x, y = _as_float_1d(u, v)
    if x.size < MIN_FIT_POINTS:
        return None
    with np.errstate(all="ignore"):
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            return None
        frame = _make_frame(x, y)
        if frame is None:
            return None
        design = _design_matrix((x - frame.mu) / frame.s, (y - frame.mv) / frame.s)
        fitted = _solve_conic(design @ design.T)
    if fitted is None:
        return None
    return frame.to_raw(fitted).canonical()


# ---------------------------------------------------------------------------
# Geometric refinement
# ---------------------------------------------------------------------------


def _radial_jacobian(x: FloatArray, y: FloatArray, theta: FloatArray) -> FloatArray:
    """Jacobian of ``residual_to_ellipse`` with respect to ``theta = (cx, cy, a, b, phi)``.

    With ``(ua, va)`` the point in the ellipse-aligned frame, ``r = |(ua, va)|``
    and ``q = sqrt(b^2 ua^2 + a^2 va^2)``, the residual is ``res = r - ab r / q``.
    Its partial derivatives are, with ``g = 1 - ab / q``::

        d res / d ua  = (ua / r) g + a b^3 r ua / q^3
        d res / d va  = (va / r) g + a^3 b r va / q^3
        d res / d cx  = -cos(phi) d res/d ua + sin(phi) d res/d va
        d res / d cy  = -sin(phi) d res/d ua - cos(phi) d res/d va
        d res / d a   = -b^3 r ua^2 / q^3
        d res / d b   = -a^3 r va^2 / q^3
        d res / d phi = ab (b^2 - a^2) r ua va / q^3

    Rows of points exactly at the centre are zero.
    """
    cx, cy, a, b, phi = (float(t) for t in theta)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    du = x - cx
    dv = y - cy
    ua = du * cos_p + dv * sin_p
    va = dv * cos_p - du * sin_p
    r = np.sqrt(du * du + dv * dv)
    q = np.sqrt(b * b * ua * ua + a * a * va * va)
    at_center = q == 0.0
    has_center = bool(np.any(at_center))
    if has_center:
        r = np.where(at_center, 1.0, r)
        q = np.where(at_center, 1.0, q)
    ab = a * b
    g = 1.0 - ab / q
    rq3 = r / (q * q * q)
    d_ua = ua / r * g + ab * b * b * rq3 * ua
    d_va = va / r * g + ab * a * a * rq3 * va
    jac = np.empty((x.size, 5), dtype=np.float64)
    jac[:, 0] = -d_ua * cos_p + d_va * sin_p
    jac[:, 1] = -d_ua * sin_p - d_va * cos_p
    jac[:, 2] = -(b**3) * rq3 * ua * ua
    jac[:, 3] = -(a**3) * rq3 * va * va
    jac[:, 4] = ab * (b * b - a * a) * rq3 * ua * va
    if has_center:
        jac[at_center] = 0.0
    return jac


def _geometric_refine(x: FloatArray, y: FloatArray, start: EllipseParams) -> EllipseParams | None:
    """Minimise the radial residuals over (cx, cy, a, b, phi), starting from ``start``.

    Levenberg-Marquardt with the analytic Jacobian. Works in the caller's
    (normalised) coordinates, where all five parameters are of order 1.

    Returns:
        The refined (not canonicalised) parameters, or None if the optimiser
        raises, does not converge, or gives an invalid ellipse.
    """
    theta0 = np.array([start.cx, start.cy, start.a, start.b, start.phi], dtype=np.float64)

    def fun(theta: FloatArray) -> FloatArray:
        return residual_to_ellipse(x, y, EllipseParams(*(float(t) for t in theta)))

    def jac(theta: FloatArray) -> FloatArray:
        return _radial_jacobian(x, y, theta)

    try:
        with np.errstate(all="ignore"):
            sol = least_squares(fun, theta0, jac=jac, method="lm")
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None
    theta = np.asarray(sol.x, dtype=np.float64)
    if not bool(sol.success) or not np.all(np.isfinite(theta)):
        return None
    cx, cy, a, b, phi = (float(t) for t in theta)
    if a * b <= 0.0:  # flipping the sign of one axis is not the same ellipse
        return None
    return EllipseParams(cx=cx, cy=cy, a=abs(a), b=abs(b), phi=phi)


# ---------------------------------------------------------------------------
# Per-channel fit
# ---------------------------------------------------------------------------


def _kept_scatter(
    design: FloatArray, scatter_all: FloatArray, kept: BoolArray, n_kept: int
) -> FloatArray:
    """Scatter matrix ``D_k D_k^T`` of the kept columns of the design matrix.

    When fewer points are rejected than kept (the usual case), it is computed
    as ``S_all - D_r D_r^T`` from the few rejected columns, which avoids copying
    the kept 6xN block. The subtraction is used only if the rejected points
    carry at most half of ``trace(S_all)``, so cancellation costs at most about
    one bit. Otherwise, e.g. with gross far-away outliers, the kept columns are
    summed directly.
    """
    if kept.size - n_kept < n_kept:
        rejected = np.compress(~kept, design, axis=1)
        scatter_rej = rejected @ rejected.T
        if float(np.trace(scatter_rej)) <= 0.5 * float(np.trace(scatter_all)):
            result: FloatArray = scatter_all - scatter_rej
            return result
    sub = np.compress(kept, design, axis=1)
    return sub @ sub.T


def _preclip_mask(x: FloatArray, y: FloatArray) -> BoolArray | None:
    """Coarse outlier cut that seeds the robust fit: keep points near the median radius.

    Radii are taken about the coordinate-wise median point. A point is kept if
    ``|r - median(r)| <= PRECLIP_K * 1.4826 * MAD(r)``, with the MAD floored at
    ``1e-3 * median(r)``. For a ring (full or partial) the window is many
    times its radial spread, so only far-away points go, e.g. a compact
    cluster of garbage events that would otherwise capture the first algebraic
    fit. The medians use a strided subsample of at most 65536 points.

    Args:
        x: Normalised U coordinates (finite).
        y: Normalised V coordinates.

    Returns:
        The keep mask, or None if it keeps every point or fewer than
        ``MIN_FIT_POINTS``.
    """
    n = x.size
    step = max(1, -(-n // _PRECLIP_SAMPLE))
    mx = float(np.median(x[::step]))
    my = float(np.median(y[::step]))
    dx = x - mx
    dy = y - my
    r2 = dx * dx + dy * dy
    r_sample = np.sqrt(r2[::step])
    med_r = float(np.median(r_sample))
    mad_r = max(float(np.median(np.abs(r_sample - med_r))), _PRECLIP_MAD_FLOOR_REL * med_r)
    half_width = PRECLIP_K * MAD_TO_SIGMA * mad_r
    lo = max(med_r - half_width, 0.0)
    hi = med_r + half_width
    keep = (r2 >= lo * lo) & (r2 <= hi * hi)
    n_keep = int(np.count_nonzero(keep))
    if n_keep == n or n_keep < MIN_FIT_POINTS:
        return None
    return keep


def _failed_fit(status: str, n_events: int) -> EllipseFit:
    return EllipseFit(
        status=status,
        params=None,
        mask_used=np.zeros(n_events, dtype=np.bool_),
        n_events=n_events,
        n_used=0,
        n_rejected=0,
        n_iter=0,
        flags=(),
    )


def fit_ellipse(
    u: npt.ArrayLike, v: npt.ArrayLike, options: FitOptions | None = None
) -> EllipseFit:
    """Fit one channel's ellipse with the full plan 5.1 procedure.

    1. Fewer than ``options.min_events`` finite points: ``too_few_events``.
    2. Direct algebraic fit of all finite points (``fit_ellipse_direct``);
       failure: ``fit_failed``. This includes data through which a whole
       family of conics passes exactly (fewer than 5 distinct points).
    3. If ``options.robust``: first a coarse pre-clip seeds the fit. Points
       whose distance from the coordinate-wise median point is more than
       ``PRECLIP_K`` (10) robust sigmas from the median distance are left out
       of the first fit, and the first fit uses the rest. If that fit fails,
       all points are used. This stops a compact far cluster (e.g. 5 % of
       events at (0, 0), or a few at an int16 corner) from capturing the
       algebraic fit. Pre-clipped points count as rejected, but they are not
       special afterwards: every iteration re-evaluates all points, so a
       pre-clipped point that lies on the fitted ellipse comes back.
       Then compute every point's radial residual to the current ellipse
       (plan 4.4), keep the points with
       ``|res - median| <= clip_k * 1.4826 * MAD`` and refit; stop when the
       kept set no longer changes or after ``max_iter`` refits. A failed
       refit (or fewer than 6 kept points) keeps the previous fit and sets
       ``robust_refit_failed``. The residual scale has a tiny floor (1e-9 of
       the fitted radius) so noise-free data (MAD ~ 0) is never clipped.
    4. If ``options.geometric``: refine by least squares on the radial
       residuals of the kept points; on failure keep the algebraic fit and
       set ``geometric_refit_failed``.
    5. Canonical form and the warning flags ``high_rejection``,
       ``extreme_axis_ratio``, ``center_outside_data`` (centre outside the
       bounding box of all finite points) and ``broad_ring``
       (``1.4826 * MAD`` of the kept points' residuals to the final ellipse
       above ``broad_ring_frac * sqrt(ab)``: not a thin ring, e.g. a blob).

    Non-finite points are excluded up front and count as rejected.

    Args:
        u: U coordinates (any real dtype).
        v: V coordinates, same length as ``u``.
        options: Fit options; defaults to ``FitOptions()``.

    Returns:
        The fit result. Never raises for degenerate data and never warns.

    Raises:
        ValueError: If ``u`` and ``v`` are not 1-D arrays of equal length.
    """
    opts = options if options is not None else FitOptions()
    x, y = _as_float_1d(u, v)
    n_events = int(x.size)
    finite = np.isfinite(x) & np.isfinite(y)
    all_finite = bool(np.all(finite))
    if not all_finite:
        x, y = x[finite], y[finite]
    n_finite = int(x.size)
    if n_finite < opts.min_events:
        return _failed_fit(STATUS_TOO_FEW_EVENTS, n_events)
    if n_finite < MIN_FIT_POINTS:
        return _failed_fit(STATUS_FIT_FAILED, n_events)

    flags: set[str] = set()
    n_iter = 0
    with np.errstate(all="ignore"):
        frame = _make_frame(x, y)
        if frame is None:
            return _failed_fit(STATUS_FIT_FAILED, n_events)
        xn = (x - frame.mu) / frame.s
        yn = (y - frame.mv) / frame.s
        design = _design_matrix(xn, yn)
        scatter_all = design @ design.T
        kept = np.ones(n_finite, dtype=np.bool_)
        params_n: EllipseParams | None = None
        if opts.robust:
            preclip = _preclip_mask(xn, yn)
            if preclip is not None:
                n_pre = int(np.count_nonzero(preclip))
                params_n = _solve_conic(_kept_scatter(design, scatter_all, preclip, n_pre))
                if params_n is not None:
                    kept = preclip
        if params_n is None:
            params_n = _solve_conic(scatter_all)
        if params_n is None:
            return _failed_fit(STATUS_FIT_FAILED, n_events)

        if opts.robust:
            for _ in range(opts.max_iter):
                dev = residual_to_ellipse(xn, yn, params_n)
                med = float(np.median(dev))
                dev -= med
                np.abs(dev, out=dev)
                mad = float(np.median(dev))
                scale = max(MAD_TO_SIGMA * mad, _ROBUST_SCALE_FLOOR_REL * params_n.target_radius)
                new_kept = dev <= opts.clip_k * scale
                if np.array_equal(new_kept, kept):
                    break
                n_new_kept = int(np.count_nonzero(new_kept))
                if n_new_kept < MIN_FIT_POINTS:
                    flags.add(FLAG_ROBUST_REFIT_FAILED)
                    break
                refit = _solve_conic(_kept_scatter(design, scatter_all, new_kept, n_new_kept))
                if refit is None:
                    flags.add(FLAG_ROBUST_REFIT_FAILED)
                    break
                params_n, kept = refit, new_kept
                n_iter += 1

        if opts.geometric:
            all_kept = bool(np.all(kept))
            refined = _geometric_refine(
                xn if all_kept else xn[kept], yn if all_kept else yn[kept], params_n
            )
            if refined is None:
                flags.add(FLAG_GEOMETRIC_REFIT_FAILED)
            else:
                params_n = refined

    params = frame.to_raw(params_n).canonical()
    if not all(math.isfinite(val) for val in (params.cx, params.cy, params.a, params.b)):
        return _failed_fit(STATUS_FIT_FAILED, n_events)

    # Ring quality: robust spread of the kept points' residuals relative to the radius
    # (scale-invariant, so computed in the normalised frame).
    with np.errstate(all="ignore"):
        res_kept = residual_to_ellipse(xn, yn, params_n)
        if not bool(np.all(kept)):
            res_kept = res_kept[kept]
        res_med = float(np.median(res_kept))
        spread = MAD_TO_SIGMA * float(np.median(np.abs(res_kept - res_med)))
    if not spread <= opts.broad_ring_frac * params_n.target_radius:  # NaN counts as broad
        flags.add(FLAG_BROAD_RING)

    if all_finite:
        mask_used = kept
    else:
        mask_used = np.zeros(n_events, dtype=np.bool_)
        mask_used[finite] = kept
    n_used = int(np.count_nonzero(kept))
    n_rejected = n_events - n_used

    if n_rejected > opts.high_rejection_frac * n_events:
        flags.add(FLAG_HIGH_REJECTION)
    if params.axis_ratio < opts.extreme_axis_ratio:
        flags.add(FLAG_EXTREME_AXIS_RATIO)
    if not (
        float(np.min(x)) <= params.cx <= float(np.max(x))
        and float(np.min(y)) <= params.cy <= float(np.max(y))
    ):
        flags.add(FLAG_CENTER_OUTSIDE_DATA)

    return EllipseFit(
        status=STATUS_OK,
        params=params,
        mask_used=mask_used,
        n_events=n_events,
        n_used=n_used,
        n_rejected=n_rejected,
        n_iter=n_iter,
        flags=order_flags(flags),
    )


# ---------------------------------------------------------------------------
# Correction and per-point quantities
# ---------------------------------------------------------------------------


def _as_float_pair(u: npt.ArrayLike, v: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Broadcast a (U, V) pair to float64 arrays of a common shape (0-d allowed)."""
    x, y = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
    return x, y


def correct(u: npt.ArrayLike, v: npt.ArrayLike, p: EllipseParams) -> tuple[FloatArray, FloatArray]:
    """Map points on the ellipse ``p`` to the circle of radius sqrt(ab) (plan 4.3).

    Translate to the fitted centre, rotate by ``-phi`` into the ellipse frame,
    scale the axes by ``R/a`` and ``R/b`` (``R = sqrt(ab)``), and rotate back.
    The output is centred on the origin. This is exactly RadialAnalysis'
    ``applyEllipseCorrection``, which the ``.tec`` consumers also apply.

    Args:
        u: U coordinates (any shape, including scalars; broadcast with ``v``).
        v: V coordinates.
        p: Ellipse parameters (canonical or not).

    Returns:
        The corrected ``(U', V')`` as float64 arrays of the broadcast shape.
        Non-finite inputs give non-finite outputs without warnings.
    """
    x, y = _as_float_pair(u, v)
    cos_p, sin_p = math.cos(p.phi), math.sin(p.phi)
    target = p.target_radius
    with np.errstate(invalid="ignore", over="ignore"):
        du = x - p.cx
        dv = y - p.cy
        u_r = (du * cos_p + dv * sin_p) * (target / p.a)
        v_r = (dv * cos_p - du * sin_p) * (target / p.b)
        return (
            np.asarray(u_r * cos_p - v_r * sin_p, dtype=np.float64),
            np.asarray(u_r * sin_p + v_r * cos_p, dtype=np.float64),
        )


def uncorrect(
    u_corr: npt.ArrayLike, v_corr: npt.ArrayLike, p: EllipseParams
) -> tuple[FloatArray, FloatArray]:
    """Invert ``correct``: map corrected, origin-centred points back to raw (U, V).

    Args:
        u_corr: Corrected U' coordinates (any shape, including scalars).
        v_corr: Corrected V' coordinates.
        p: The ellipse parameters ``correct`` was called with.

    Returns:
        The raw ``(U, V)`` as float64 arrays of the broadcast shape.
    """
    x, y = _as_float_pair(u_corr, v_corr)
    cos_p, sin_p = math.cos(p.phi), math.sin(p.phi)
    target = p.target_radius
    with np.errstate(invalid="ignore", over="ignore"):
        u_r = (x * cos_p + y * sin_p) * (p.a / target)
        v_r = (y * cos_p - x * sin_p) * (p.b / target)
        return (
            np.asarray(u_r * cos_p - v_r * sin_p + p.cx, dtype=np.float64),
            np.asarray(u_r * sin_p + v_r * cos_p + p.cy, dtype=np.float64),
        )


def residual_to_ellipse(u: npt.ArrayLike, v: npt.ArrayLike, p: EllipseParams) -> FloatArray:
    """Radial residual of raw points to the ellipse ``p`` (plan 4.4).

    In the ellipse-aligned frame about the centre, with polar angle
    ``theta``, the residual is ``|(u_a, v_a)| - r_ell(theta)`` where
    ``r_ell = ab / sqrt((b cos theta)^2 + (a sin theta)^2)``. It is evaluated
    without trigonometry as ``r - ab r / sqrt(b^2 u_a^2 + a^2 v_a^2)``; a
    point exactly at the centre gets ``-a`` (the C++ ``atan2(0, 0) = 0``
    convention).

    Args:
        u: U coordinates (any shape, including scalars; broadcast with ``v``).
        v: V coordinates.
        p: Ellipse parameters with positive semi-axes.

    Returns:
        The residuals, a float64 array of the broadcast shape. Non-finite
        inputs give non-finite residuals without warnings.
    """
    x, y = _as_float_pair(u, v)
    shape = x.shape
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    cos_p, sin_p = math.cos(p.phi), math.sin(p.phi)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # In-place arithmetic: this runs once per robust iteration on up to ~1M points.
        du = x - p.cx
        dv = y - p.cy
        ua = du * cos_p
        ua += dv * sin_p
        va = dv * cos_p
        va -= du * sin_p
        du *= du
        dv *= dv
        du += dv
        r = np.sqrt(du, out=du)
        ua *= ua
        ua *= p.b * p.b
        va *= va
        va *= p.a * p.a
        ua += va
        q = np.sqrt(ua, out=ua)
        at_center = q == 0.0
        res: FloatArray = np.divide(r, q, out=q)
        res *= -(p.a * p.b)
        res += r
    if np.any(at_center):
        res[at_center] = -p.a
    return res.reshape(shape)


def radii_about_center(u: npt.ArrayLike, v: npt.ArrayLike, p: EllipseParams) -> FloatArray:
    """Distance of raw points from the fitted centre: the pre-correction radii (plan 5.3).

    Args:
        u: U coordinates (any shape, including scalars).
        v: V coordinates.
        p: Ellipse parameters (only the centre is used).

    Returns:
        ``sqrt((U - cx)^2 + (V - cy)^2)`` as a float64 array.
    """
    x, y = _as_float_pair(u, v)
    with np.errstate(invalid="ignore", over="ignore"):
        return np.asarray(np.hypot(x - p.cx, y - p.cy), dtype=np.float64)


def corrected_residual(
    u_corr: npt.ArrayLike, v_corr: npt.ArrayLike, p: EllipseParams
) -> FloatArray:
    """Residual of corrected points to the target circle: ``sqrt(U'^2 + V'^2) - sqrt(ab)``.

    Args:
        u_corr: Corrected U' coordinates (output of ``correct``; any shape).
        v_corr: Corrected V' coordinates.
        p: The ellipse parameters used for the correction.

    Returns:
        The residuals (plan 4.4) as a float64 array.
    """
    x, y = _as_float_pair(u_corr, v_corr)
    with np.errstate(invalid="ignore", over="ignore"):
        return np.asarray(np.hypot(x, y) - p.target_radius, dtype=np.float64)


def ellipse_points(p: EllipseParams, t: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Points on the ellipse at parametric angles ``t`` (for drawing and tests).

    Args:
        p: Ellipse parameters.
        t: Parametric angles in radians (any shape, including scalars).

    Returns:
        ``(U, V)`` of ``(cx, cy) + R(phi) (a cos t, b sin t)`` as float64 arrays.
    """
    tt = np.asarray(t, dtype=np.float64)
    cos_p, sin_p = math.cos(p.phi), math.sin(p.phi)
    with np.errstate(invalid="ignore", over="ignore"):
        ea = p.a * np.cos(tt)
        eb = p.b * np.sin(tt)
        return (
            np.asarray(p.cx + ea * cos_p - eb * sin_p, dtype=np.float64),
            np.asarray(p.cy + ea * sin_p + eb * cos_p, dtype=np.float64),
        )
