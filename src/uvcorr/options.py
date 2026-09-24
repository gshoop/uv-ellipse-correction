"""Fit options and the status/flag vocabulary shared by the whole package.

``FitOptions`` configures the per-channel ellipse fit and the metrics (plan
sections 5.1 and 7). It lives here rather than in ``uvcorr.analysis`` (where the
plan first put it) because ``uvcorr.ellipse.fit_ellipse`` needs it and
``uvcorr.analysis`` imports ``uvcorr.ellipse``. ``uvcorr.analysis`` re-exports
it (phase 3).

Two flags are additions to the plan 5.1 list: ``FLAG_BROAD_RING`` (the points
do not form a thin ring, e.g. a dead or pedestal-only channel) and
``FLAG_GEOMETRIC_REFIT_FAILED`` (the optional geometric refinement failed and
the algebraic fit was kept).

The status and flag strings below are the single source of truth for the
``status`` and ``flags`` CSV columns (plan 5.1 and 6.2).
"""

from __future__ import annotations

import json
import logging
import math
import numbers
from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel fit status (exactly one per channel).
# ---------------------------------------------------------------------------

STATUS_OK = "ok"
"""The ellipse fit succeeded; parameters and metrics are available."""

STATUS_TOO_FEW_EVENTS = "too_few_events"
"""The channel has fewer than ``FitOptions.min_events`` usable events."""

STATUS_FIT_FAILED = "fit_failed"
"""The algebraic fit found no ellipse (degenerate data, no ellipse eigenvector, ...)."""

STATUSES: tuple[str, ...] = (STATUS_OK, STATUS_TOO_FEW_EVENTS, STATUS_FIT_FAILED)
"""All valid statuses."""

# ---------------------------------------------------------------------------
# Warning flags (zero or more per channel; the status stays ``ok``).
# ---------------------------------------------------------------------------

FLAG_HIGH_REJECTION = "high_rejection"
"""The robust iteration rejected more than ``FitOptions.high_rejection_frac`` of the events."""

FLAG_EXTREME_AXIS_RATIO = "extreme_axis_ratio"
"""The fitted axis ratio b/a is below ``FitOptions.extreme_axis_ratio``."""

FLAG_CENTER_OUTSIDE_DATA = "center_outside_data"
"""The fitted centre lies outside the bounding box of the channel's (U, V) points."""

FLAG_BROAD_RING = "broad_ring"
"""The kept points do not form a thin ring: their robust radial spread
``1.4826 * MAD(residuals)`` exceeds ``FitOptions.broad_ring_frac * sqrt(ab)``."""

FLAG_ROBUST_REFIT_FAILED = "robust_refit_failed"
"""A robust refit failed; the previous successful fit was kept."""

FLAG_GEOMETRIC_REFIT_FAILED = "geometric_refit_failed"
"""The optional geometric refinement failed; the algebraic fit was kept (not in plan 5.1)."""

FLAG_GAUSS_FIT_FAILED_PRE = "gauss_fit_failed_pre"
"""The Gaussian fit to the pre-correction radii failed; sample statistics are reported."""

FLAG_GAUSS_FIT_FAILED_POST = "gauss_fit_failed_post"
"""The Gaussian fit to the post-correction radii failed; sample statistics are reported."""

FLAGS: tuple[str, ...] = (
    FLAG_HIGH_REJECTION,
    FLAG_EXTREME_AXIS_RATIO,
    FLAG_CENTER_OUTSIDE_DATA,
    FLAG_BROAD_RING,
    FLAG_ROBUST_REFIT_FAILED,
    FLAG_GEOMETRIC_REFIT_FAILED,
    FLAG_GAUSS_FIT_FAILED_PRE,
    FLAG_GAUSS_FIT_FAILED_POST,
)
"""All valid flags, in the canonical order used when a channel carries several."""

FLAG_SEPARATOR = ";"
"""Separator between flags in the CSV ``flags`` column."""


def order_flags(flags: Iterable[str]) -> tuple[str, ...]:
    """Return flags de-duplicated and in the canonical order of ``FLAGS``.

    Args:
        flags: Flag strings, each one of ``FLAGS``.

    Returns:
        The distinct flags, ordered as in ``FLAGS``.

    Raises:
        ValueError: If a flag is not one of ``FLAGS``.
    """
    present = set(flags)
    unknown = present - set(FLAGS)
    if unknown:
        raise ValueError(f"Unknown flag(s): {sorted(unknown)}")
    return tuple(flag for flag in FLAGS if flag in present)


# ---------------------------------------------------------------------------
# Fit options
# ---------------------------------------------------------------------------

_INT_FIELDS = frozenset({"min_events", "max_iter"})
_BOOL_FIELDS = frozenset({"robust", "geometric"})
_FLOAT_FIELDS = frozenset(
    {"clip_k", "phase_ref_freq_hz", "high_rejection_frac", "extreme_axis_ratio", "broad_ring_frac"}
)


@dataclass(frozen=True)
class FitOptions:
    """Options for the per-channel ellipse fit and its metrics.

    Instances are immutable and validated on construction. Two instances with
    the same values compare equal, so ``options == FitOptions()`` (or
    ``options.is_default()``) tells whether a fit used non-default options.

    Attributes:
        min_events: Minimum number of (finite) events a channel needs to be
            fitted; fewer gives status ``too_few_events``. At least 1 (the
            algebraic fit itself needs 6 points and reports ``fit_failed``
            below that).
        robust: Run the robust clip-and-refit iteration (plan 5.1.3).
        clip_k: Clipping threshold in robust standard deviations: a point is
            kept when ``|res - median(res)| <= clip_k * 1.4826 * MAD``.
            Finite and positive.
        max_iter: Maximum number of robust clip-and-refit iterations. At
            least 1; ignored when ``robust`` is False.
        geometric: Refine the algebraic ellipse by a geometric least-squares
            fit of the radial residuals of the kept points (plan 5.1.4).
        phase_ref_freq_hz: U/V quadrature reference frequency in Hz, used to
            convert phase gaps and radial widths to time. Finite and positive.
        high_rejection_frac: Rejected-event fraction above which the
            ``high_rejection`` flag is set. In [0, 1].
        extreme_axis_ratio: Axis ratio b/a below which the
            ``extreme_axis_ratio`` flag is set. In [0, 1].
        broad_ring_frac: Robust radial spread of the kept points
            (``1.4826 * MAD`` of their residuals to the ellipse), relative to
            ``sqrt(ab)``, above which the ``broad_ring`` flag is set. Finite
            and positive. Real rings are about 0.017.

    Example:
        >>> opts = FitOptions(robust=False)
        >>> opts.is_default()
        False
        >>> FitOptions.from_json(opts.to_json()) == opts
        True
    """

    min_events: int = 100
    robust: bool = True
    clip_k: float = 4.0
    max_iter: int = 5
    geometric: bool = False
    phase_ref_freq_hz: float = 490e3
    high_rejection_frac: float = 0.05
    extreme_axis_ratio: float = 0.5
    broad_ring_frac: float = 0.1

    def __post_init__(self) -> None:
        """Validate and normalise the field types.

        Integers are accepted for float fields (and stored as float); numpy
        integer, float and bool scalars are converted to the Python types.
        Booleans are rejected for numeric fields.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If a field is out of range.
        """
        for name in _INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"FitOptions.{name} must be an int, got {value!r}")
            object.__setattr__(self, name, int(value))
        for name in _FLOAT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"FitOptions.{name} must be a number, got {value!r}")
            object.__setattr__(self, name, float(value))
        for name in _BOOL_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"FitOptions.{name} must be a bool, got {value!r}")
            object.__setattr__(self, name, bool(value))

        if self.min_events < 1:
            raise ValueError(f"FitOptions.min_events must be >= 1, got {self.min_events}")
        if self.max_iter < 1:
            raise ValueError(f"FitOptions.max_iter must be >= 1, got {self.max_iter}")
        if not (math.isfinite(self.clip_k) and self.clip_k > 0):
            raise ValueError(f"FitOptions.clip_k must be finite and > 0, got {self.clip_k}")
        if not (math.isfinite(self.phase_ref_freq_hz) and self.phase_ref_freq_hz > 0):
            raise ValueError(
                "FitOptions.phase_ref_freq_hz must be finite and > 0, "
                f"got {self.phase_ref_freq_hz}"
            )
        for name in ("high_rejection_frac", "extreme_axis_ratio"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:  # also rejects NaN
                raise ValueError(f"FitOptions.{name} must be in [0, 1], got {value}")
        if not (math.isfinite(self.broad_ring_frac) and self.broad_ring_frac > 0):
            raise ValueError(
                f"FitOptions.broad_ring_frac must be finite and > 0, got {self.broad_ring_frac}"
            )

    def is_default(self) -> bool:
        """Return True if every option has its default value."""
        return self == FitOptions()

    def to_dict(self) -> dict[str, Any]:
        """Return the options as a plain dict in field-declaration order."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self) -> str:
        """Serialise the options to a compact JSON object.

        Keys appear in field-declaration order, so the same options always give
        the same string (it is stored in the cache as ``options_json``).

        Returns:
            The JSON text.
        """
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, strict: bool = True) -> FitOptions:
        """Build options from a dict such as the one ``to_dict`` returns.

        Missing keys take their default values, so options saved before a new
        option was added still load. Unknown keys usually mean a typo or options
        written by a newer uvcorr. With ``strict`` (the default, for user input)
        they are an error. With ``strict=False`` (for loading stored results)
        they are ignored and a warning is logged, so a newer cache still opens.

        Args:
            data: Mapping of option name to value.
            strict: Raise on unknown keys instead of ignoring them.

        Returns:
            The validated options.

        Raises:
            ValueError: If ``data`` has unknown keys (``strict`` only) or a value
                is out of range.
            TypeError: If ``data`` is not a dict or a value has the wrong type.
        """
        if not isinstance(data, dict):
            raise TypeError(f"FitOptions data must be a dict, got {type(data).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            if strict:
                raise ValueError(f"Unknown FitOptions key(s): {unknown}")
            logger.warning("Ignoring unknown FitOptions key(s): %s", unknown)
            data = {key: value for key, value in data.items() if key in known}
        return cls(**data)

    @classmethod
    def from_json(cls, text: str, *, strict: bool = True) -> FitOptions:
        """Parse options from JSON produced by ``to_json``.

        See ``from_dict`` for the handling of missing and unknown keys.

        Args:
            text: A JSON object.
            strict: Raise on unknown keys instead of ignoring them.

        Returns:
            The validated options.

        Raises:
            ValueError: If the text is not valid JSON, not an object, has
                unknown keys (``strict`` only) or an out-of-range value.
            TypeError: If a value has the wrong type.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid FitOptions JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("FitOptions JSON must be an object")
        return cls.from_dict(data, strict=strict)
