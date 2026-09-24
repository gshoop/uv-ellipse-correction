"""Tests for uvcorr.options: FitOptions validation/JSON and the status/flag vocabulary."""

from __future__ import annotations

import dataclasses
import json
import logging
import math

import numpy as np
import pytest

from uvcorr import options
from uvcorr.options import FitOptions

DEFAULTS = {
    "min_events": 100,
    "robust": True,
    "clip_k": 4.0,
    "max_iter": 5,
    "geometric": False,
    "phase_ref_freq_hz": 490e3,
    "high_rejection_frac": 0.05,
    "extreme_axis_ratio": 0.5,
    "broad_ring_frac": 0.1,
}


class TestVocabulary:
    def test_status_strings(self) -> None:
        assert options.STATUS_OK == "ok"
        assert options.STATUS_TOO_FEW_EVENTS == "too_few_events"
        assert options.STATUS_FIT_FAILED == "fit_failed"
        assert options.STATUSES == ("ok", "too_few_events", "fit_failed")

    def test_plan_flag_strings(self) -> None:
        plan_flags = {
            "high_rejection",
            "extreme_axis_ratio",
            "center_outside_data",
            "robust_refit_failed",
            "broad_ring",
            "gauss_fit_failed_pre",
            "gauss_fit_failed_post",
        }
        assert plan_flags <= set(options.FLAGS)
        assert options.FLAG_HIGH_REJECTION == "high_rejection"
        assert options.FLAG_EXTREME_AXIS_RATIO == "extreme_axis_ratio"
        assert options.FLAG_CENTER_OUTSIDE_DATA == "center_outside_data"
        assert options.FLAG_ROBUST_REFIT_FAILED == "robust_refit_failed"
        assert options.FLAG_GAUSS_FIT_FAILED_PRE == "gauss_fit_failed_pre"
        assert options.FLAG_GAUSS_FIT_FAILED_POST == "gauss_fit_failed_post"
        assert options.FLAG_GEOMETRIC_REFIT_FAILED == "geometric_refit_failed"
        assert options.FLAG_BROAD_RING == "broad_ring"
        assert len(set(options.FLAGS)) == len(options.FLAGS)
        assert all(options.FLAG_SEPARATOR not in flag for flag in options.FLAGS)

    def test_order_flags(self) -> None:
        got = options.order_flags(
            ["gauss_fit_failed_post", "high_rejection", "center_outside_data", "high_rejection"]
        )
        assert got == ("high_rejection", "center_outside_data", "gauss_fit_failed_post")
        assert options.order_flags(set()) == ()

    def test_order_flags_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown flag"):
            options.order_flags(["high_rejection", "bogus"])


class TestFitOptionsDefaults:
    def test_defaults(self) -> None:
        opts = FitOptions()
        assert opts.to_dict() == DEFAULTS
        assert opts.is_default()
        assert opts == FitOptions()

    def test_non_default(self) -> None:
        for name, value in [
            ("min_events", 50),
            ("robust", False),
            ("clip_k", 3.0),
            ("max_iter", 10),
            ("geometric", True),
            ("phase_ref_freq_hz", 500e3),
            ("high_rejection_frac", 0.1),
            ("extreme_axis_ratio", 0.3),
            ("broad_ring_frac", 0.2),
        ]:
            opts = FitOptions(**{name: value})
            assert not opts.is_default(), name
            assert opts != FitOptions()

    def test_frozen_and_hashable(self) -> None:
        opts = FitOptions()
        with pytest.raises(dataclasses.FrozenInstanceError):
            opts.clip_k = 3.0  # type: ignore[misc]
        assert hash(opts) == hash(FitOptions())


class TestFitOptionsValidation:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("min_events", 0),
            ("min_events", -5),
            ("max_iter", 0),
            ("clip_k", 0.0),
            ("clip_k", -1.0),
            ("clip_k", math.nan),
            ("clip_k", math.inf),
            ("phase_ref_freq_hz", 0.0),
            ("phase_ref_freq_hz", -490e3),
            ("phase_ref_freq_hz", math.nan),
            ("phase_ref_freq_hz", math.inf),
            ("high_rejection_frac", -0.01),
            ("high_rejection_frac", 1.5),
            ("high_rejection_frac", math.nan),
            ("extreme_axis_ratio", -0.1),
            ("extreme_axis_ratio", 1.01),
            ("broad_ring_frac", 0.0),
            ("broad_ring_frac", -0.1),
            ("broad_ring_frac", math.nan),
            ("broad_ring_frac", math.inf),
        ],
    )
    def test_out_of_range(self, name: str, value: float) -> None:
        with pytest.raises(ValueError, match=name):
            FitOptions(**{name: value})

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("min_events", 100.0),
            ("min_events", True),
            ("min_events", "100"),
            ("max_iter", 2.5),
            ("clip_k", True),
            ("clip_k", "4"),
            ("robust", 1),
            ("robust", "yes"),
            ("geometric", 0),
            ("phase_ref_freq_hz", None),
            ("broad_ring_frac", "0.1"),
        ],
    )
    def test_wrong_type(self, name: str, value: object) -> None:
        with pytest.raises(TypeError, match=name):
            FitOptions(**{name: value})

    def test_boundaries_accepted(self) -> None:
        FitOptions(min_events=1, max_iter=1, high_rejection_frac=0.0, extreme_axis_ratio=1.0)
        FitOptions(high_rejection_frac=1.0, extreme_axis_ratio=0.0)
        FitOptions(broad_ring_frac=5.0)  # > 1 effectively disables the flag

    def test_numeric_coercion(self) -> None:
        opts = FitOptions(
            min_events=np.int64(50),
            clip_k=3,
            phase_ref_freq_hz=np.float32(490e3),
            robust=np.bool_(False),
        )
        assert type(opts.min_events) is int and opts.min_events == 50
        assert type(opts.clip_k) is float and opts.clip_k == 3.0
        assert type(opts.phase_ref_freq_hz) is float
        assert opts.robust is False
        # An int given for a float field compares (and serialises) like the float.
        assert FitOptions(clip_k=4) == FitOptions()
        assert FitOptions(clip_k=4).to_json() == FitOptions().to_json()


class TestFitOptionsJson:
    def test_round_trip_default(self) -> None:
        opts = FitOptions()
        assert FitOptions.from_json(opts.to_json()) == opts

    def test_round_trip_non_default(self) -> None:
        opts = FitOptions(
            min_events=20,
            robust=False,
            clip_k=2.5,
            max_iter=9,
            geometric=True,
            phase_ref_freq_hz=123456.789,
            high_rejection_frac=0.125,
            extreme_axis_ratio=0.3,
            broad_ring_frac=0.25,
        )
        text = opts.to_json()
        back = FitOptions.from_json(text)
        assert back == opts
        assert back.to_json() == text

    def test_key_order_is_field_order(self) -> None:
        keys = list(json.loads(FitOptions().to_json()))
        assert keys == [f.name for f in dataclasses.fields(FitOptions)]
        assert keys == list(DEFAULTS)

    def test_stable_text(self) -> None:
        assert FitOptions().to_json() == (
            '{"min_events":100,"robust":true,"clip_k":4.0,"max_iter":5,"geometric":false,'
            '"phase_ref_freq_hz":490000.0,"high_rejection_frac":0.05,"extreme_axis_ratio":0.5,'
            '"broad_ring_frac":0.1}'
        )

    def test_missing_keys_take_defaults(self) -> None:
        assert FitOptions.from_json('{"robust": false}') == FitOptions(robust=False)
        assert FitOptions.from_json("{}") == FitOptions()

    def test_unknown_key_is_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown FitOptions key"):
            FitOptions.from_json('{"robust": false, "clipk": 3}')
        with pytest.raises(ValueError, match="Unknown FitOptions key"):
            FitOptions.from_json('{"robust": false, "clipk": 3}', strict=True)
        with pytest.raises(ValueError, match="Unknown FitOptions key"):
            FitOptions.from_dict({"clipk": 3})

    def test_unknown_key_lenient(self, caplog: pytest.LogCaptureFixture) -> None:
        # Loading stored results written by a newer uvcorr: ignore and log.
        text = '{"robust": false, "new_option": 7, "another": "x"}'
        with caplog.at_level(logging.WARNING, logger="uvcorr.options"):
            opts = FitOptions.from_json(text, strict=False)
        assert opts == FitOptions(robust=False)
        assert "new_option" in caplog.text and "another" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="uvcorr.options"):
            assert FitOptions.from_dict({"max_iter": 2, "zzz": 1}, strict=False) == FitOptions(
                max_iter=2
            )
        assert "zzz" in caplog.text
        # Known keys are still validated in lenient mode.
        with pytest.raises(ValueError, match="clip_k"):
            FitOptions.from_json('{"clip_k": -1, "zzz": 1}', strict=False)

    def test_lenient_without_unknown_keys_logs_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="uvcorr.options"):
            assert FitOptions.from_json(FitOptions().to_json(), strict=False) == FitOptions()
        assert caplog.text == ""

    @pytest.mark.parametrize("text", ["not json", "[1, 2]", '"x"', ""])
    def test_invalid_json(self, text: str) -> None:
        with pytest.raises(ValueError):
            FitOptions.from_json(text)

    def test_invalid_values_in_json(self) -> None:
        with pytest.raises(ValueError, match="clip_k"):
            FitOptions.from_json('{"clip_k": NaN}')
        with pytest.raises(TypeError, match="min_events"):
            FitOptions.from_json('{"min_events": 1.5}')

    def test_from_dict(self) -> None:
        assert FitOptions.from_dict({"max_iter": 3}) == FitOptions(max_iter=3)
        with pytest.raises(TypeError):
            FitOptions.from_dict([("max_iter", 3)])  # type: ignore[arg-type]
