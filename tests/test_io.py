"""Tests for uvcorr.io: the RadialAnalysis ``.tec`` file and ``radial_summary.csv``."""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path
from typing import Any

import pytest

from uvcorr.analysis import CSV_COLUMNS, RESULT_COLUMNS, ChannelKey, ChannelResult
from uvcorr.channels import electrode_label, is_active_channel, polarity_name
from uvcorr.io import tec
from uvcorr.io._atomic import atomic_write_text, atomic_write_texts
from uvcorr.io.export import prepare_output_dir, write_outputs
from uvcorr.io.summary_csv import (
    format_cell,
    format_summary_csv,
    parse_cell,
    read_summary_csv,
    write_summary_csv,
)
from uvcorr.io.tec import (
    TecFormatError,
    format_tec,
    format_tec_double,
    parse_tec_blocks,
    read_tec,
    write_tec,
)
from uvcorr.options import FLAG_EXTREME_AXIS_RATIO, FLAG_GAUSS_FIT_FAILED_PRE

HOME = Path.home()
# .tec files written by RadialAnalysis (ellipse format) ...
RADIAL_ANALYSIS_TEC = [
    HOME / "time-calibration-data/full-system-data/calibration/data_20250625_092749UVdata_ECv2.tec",
    HOME / "tp-processing/tp-data/145mV_300ns/data_20260528_170144UVdata.tec",
]
# ... and by the legacy shear-method EllipseCorrection (centerU, centerV, psi, radius)
LEGACY_TEC = [
    HOME / "DataProcessing/EllipseCorrection/EllipseCorrection_output/"
    "data_node10_20250415_151923UVdata.tec",
    HOME / "QE_Work/data_20240503_103208UVdata.tec",
    HOME / "rilpetlib/scripts/tests/data/data_20240412_094819UVdata.tec",
]


def make_result(key: tuple[int, int, int, int], status: str = "ok", **values: Any) -> ChannelResult:
    node, board, rena, channel = key
    if is_active_channel(rena, channel):
        polarity, electrode = polarity_name(board, rena, channel), electrode_label(
            board, rena, channel
        )
    else:  # old RadialAnalysis files also cover RENA 1 channels 4-6
        polarity, electrode = "anode", "-"
    base: dict[str, Any] = {
        "node": node,
        "board": board,
        "rena": rena,
        "channel": channel,
        "polarity": polarity,
        "electrode": electrode,
        "status": status,
        "flags": (),
        "n_events": 1000,
    }
    base.update(values)
    return ChannelResult(**base)


def ok_result(key: tuple[int, int, int, int], **values: Any) -> ChannelResult:
    params: dict[str, Any] = {
        "n_used": 990,
        "n_rejected": 10,
        "centerU": 2030.4123456,
        "centerV": 2036.8912345,
        "semiMajor": 695.58412345,
        "semiMinor": 670.07098765,
        "phi": 0.0312345678,
        "post_mean": 682.5,
        "post_sigma": 9.295574,
    }
    params["target_radius"] = math.sqrt(params["semiMajor"] * params["semiMinor"])
    params["axis_ratio"] = params["semiMinor"] / params["semiMajor"]
    params.update(values)
    return make_result(key, **params)


# ---------------------------------------------------------------------------
# .tec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (2036.0, "2036"),
        (2022.3649, "2022.36"),
        (695.58412345, "695.584"),
        (3.1156399999, "3.11564"),
        (-2.99377e-05, "-2.99377e-05"),
        (1e-4, "0.0001"),
        (1e-5, "1e-05"),
        (123456.4, "123456"),
        (1234567.0, "1.23457e+06"),
        (999999.5, "1e+06"),
        (0.5, "0.5"),
        (0.0, "0"),
        (-0.0, "-0"),
        (-1.0, "-1"),
        (1.0000005, "1"),
    ],
)
def test_format_tec_double_matches_cpp_ostream(value: float, text: str) -> None:
    # Expected strings are what `std::cout << value` prints (default precision 6);
    # the whole mapping was checked against a compiled C++ program.
    assert format_tec_double(value) == text


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_format_tec_double_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        format_tec_double(value)


EXPECTED_TEC = (
    "channel{\n"
    "\tnode=1\n"
    "\tboard=15\n"
    "\trena=0\n"
    "\tchannel=4\n"
    "\tcenterU=2030.41\n"
    "\tcenterV=2036.89\n"
    "\tsemiMajor=695.584\n"
    "\tsemiMinor=670.071\n"
    "\tphi=0.0312346\n"
    "\tradius=682.708\n"
    "\tradiusStd=9.29557\n"
    "}\n"
    "channel{\n"
    "\tnode=1\n"
    "\tboard=15\n"
    "\trena=1\n"
    "\tchannel=28\n"
    "\tcenterU=2036\n"
    "\tcenterV=-0\n"
    "\tsemiMajor=420.809\n"
    "\tsemiMinor=410.282\n"
    "\tphi=-2.99377e-05\n"
    "\tradius=415.512\n"
    "\tradiusStd=12.4171\n"
    "}\n"
)


def expected_tec_results() -> list[ChannelResult]:
    return [
        ok_result(
            (1, 15, 1, 28),
            centerU=2036.0,
            centerV=-0.0,
            semiMajor=420.80912,
            semiMinor=410.28187,
            phi=-2.99377e-05,
            target_radius=415.5121,
            post_sigma=12.41712,
        ),
        make_result((1, 15, 0, 7), status="too_few_events", n_events=12),
        ok_result((1, 15, 0, 4)),
        make_result((1, 14, 0, 4), status="fit_failed"),
    ]


def test_tec_exact_bytes() -> None:
    # The RadialAnalysis layout: no header, tab-indented key=value lines, "}\n" after
    # every block, no blank lines; ok channels only, sorted by key.
    assert format_tec(expected_tec_results()) == EXPECTED_TEC


def test_write_tec_file_bytes(tmp_path: Path) -> None:
    path = write_tec(tmp_path / "run.tec", expected_tec_results())
    assert path.read_bytes() == EXPECTED_TEC.encode("ascii")
    assert write_tec(tmp_path / "empty.tec", [make_result((1, 15, 0, 4), "fit_failed")])
    assert (tmp_path / "empty.tec").read_bytes() == b""
    assert read_tec(tmp_path / "empty.tec") == {}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["empty.tec", "run.tec"]  # no temps


def test_tec_radius_defaults_to_sqrt_ab() -> None:
    result = ok_result((1, 15, 0, 4), target_radius=None)
    text = format_tec([result])
    assert "\tradius=682.708\n" in text


def test_tec_rejects_ok_rows_without_values() -> None:
    with pytest.raises(ValueError, match="node 1 board 15 rena 0 channel 4"):
        format_tec([ok_result((1, 15, 0, 4), post_sigma=None)])
    with pytest.raises(ValueError):
        format_tec([ok_result((1, 15, 0, 4), phi=None)])


def test_tec_round_trip_is_exact_to_written_precision(tmp_path: Path) -> None:
    results = expected_tec_results()
    path = write_tec(tmp_path / "run.tec", results)
    entries = read_tec(path)
    ok = sorted((r for r in results if r.ok), key=lambda r: r.key)
    assert list(entries) == [r.key for r in ok]
    for result in ok:
        entry = entries[result.key]
        assert entry.key == result.key and isinstance(entry.key, ChannelKey)

        def rounded(value: float | None) -> float:
            assert value is not None
            return float(format(value, "g"))

        assert entry.params.cx == rounded(result.centerU)
        assert entry.params.cy == rounded(result.centerV)
        assert entry.params.a == rounded(result.semiMajor)
        assert entry.params.b == rounded(result.semiMinor)
        assert entry.params.phi == rounded(result.phi)
        assert entry.radius == rounded(result.target_radius)
        assert entry.radius_std == rounded(result.post_sigma)
    # Writing the parsed values again gives the same bytes
    rewritten = [
        dataclasses.replace(
            r,
            centerU=entries[r.key].params.cx,
            centerV=entries[r.key].params.cy,
            semiMajor=entries[r.key].params.a,
            semiMinor=entries[r.key].params.b,
            phi=entries[r.key].params.phi,
            target_radius=entries[r.key].radius,
            post_sigma=entries[r.key].radius_std,
        )
        for r in ok
    ]
    assert format_tec(rewritten) == EXPECTED_TEC


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("node=1\n", "expected 'channel{'"),
        ("channel{\n\tnode=1\n", "not closed"),
        ("channel{\n\tnode=1\n\tnode=2\n}\n", "repeated"),
        ("channel{\n\tnode\n}\n", "key=value"),
        ("channel{\n\t=3\n}\n", "key=value"),
    ],
)
def test_parse_tec_blocks_errors(text: str, message: str) -> None:
    with pytest.raises(TecFormatError, match=message):
        parse_tec_blocks(text)


def test_parse_tec_blocks_is_lenient_on_whitespace() -> None:
    blocks = parse_tec_blocks("\n  channel{ \r\n\tnode = 1\r\n psi=0.5\n}\n\n")
    assert blocks == [{"node": "1", "psi": "0.5"}]


def test_read_tec_errors(tmp_path: Path) -> None:
    block = EXPECTED_TEC.split("}\n")[0] + "}\n"
    path = tmp_path / "bad.tec"
    path.write_text(block + block)
    with pytest.raises(TecFormatError, match="second block"):
        read_tec(path)
    path.write_text(block.replace("\tradiusStd=9.29557\n", ""))
    with pytest.raises(TecFormatError, match="missing keys \\['radiusStd'\\]"):
        read_tec(path)
    path.write_text(block.replace("}", "\textra=1\n}"))
    with pytest.raises(TecFormatError, match="unknown keys \\['extra'\\]"):
        read_tec(path)
    path.write_text(block.replace("centerU=2030.41", "centerU=abc"))
    with pytest.raises(TecFormatError, match="block 0"):
        read_tec(path)
    path.write_text("channel{\n\tnode=1\n\tboard=15\n\trena=0\n\tchannel=4\n\tpsi=0.1\n}\n")
    with pytest.raises(TecFormatError, match="legacy shear-method"):
        read_tec(path)


def _existing(paths: list[Path]) -> list[Path]:
    return [p for p in paths if p.is_file()]


@pytest.mark.parametrize("path", RADIAL_ANALYSIS_TEC, ids=lambda p: p.name)
def test_read_real_radial_analysis_tec(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"{path} not found")
    entries = read_tec(path)
    assert len(entries) > 1000
    for entry in entries.values():
        p = entry.params
        assert all(math.isfinite(x) for x in (p.cx, p.cy, p.a, p.b, p.phi, entry.radius))
        assert entry.radius == pytest.approx(math.sqrt(p.a * p.b), rel=2e-5)
        assert -math.pi <= p.phi <= math.pi  # the C++ writes (-pi, pi]
        assert entry.radius_std > 0
    # Our writer reproduces the C++ file byte for byte from its parsed values
    results = [
        ok_result(
            key,
            centerU=e.params.cx,
            centerV=e.params.cy,
            semiMajor=e.params.a,
            semiMinor=e.params.b,
            phi=e.params.phi,
            target_radius=e.radius,
            post_sigma=e.radius_std,
        )
        for key, e in entries.items()
    ]
    assert list(entries) == sorted(entries)  # RadialAnalysis loop order = key order
    assert format_tec(results).encode("ascii") == path.read_bytes()


@pytest.mark.parametrize("path", LEGACY_TEC, ids=lambda p: p.name)
def test_legacy_shear_tec_files(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"{path} not found")
    blocks = parse_tec_blocks(path.read_text())
    assert blocks and set(blocks[0]) == {
        "node",
        "board",
        "rena",
        "channel",
        "centerU",
        "centerV",
        "psi",
        "radius",
    }
    with pytest.raises(TecFormatError, match="legacy shear-method"):
        read_tec(path)


# ---------------------------------------------------------------------------
# radial_summary.csv
# ---------------------------------------------------------------------------

PLAN_HEADER = (
    "node,board,rena,channel,polarity,electrode,status,flags,n_events,n_used,n_rejected,"
    "centerU,centerV,semiMajor,semiMinor,phi,axis_ratio,target_radius,"
    "pre_mean,pre_sigma,pre_fwhm,pre_chi2ndf,pre_skewness,pre_kurtosis,pre_robust_sigma,"
    "post_mean,post_sigma,post_fwhm,post_chi2ndf,post_skewness,post_kurtosis,post_robust_sigma,"
    "rawfit_res_mean,rawfit_res_sigma,corr_res_mean,corr_res_sigma,"
    "phase_mean_gap_rad,phase_max_gap_rad,phase_max_gap_ns,phase_ks,timing_jitter_ns,options_source"
)


def full_result() -> ChannelResult:
    values = {c.name: 0.1 * (i + 1) / 3 for i, c in enumerate(RESULT_COLUMNS)}
    return ok_result(
        (2, 16, 1, 25),
        flags=(FLAG_GAUSS_FIT_FAILED_PRE, FLAG_EXTREME_AXIS_RATIO),
        n_events=123456,
        n_used=123000,
        n_rejected=456,
        centerU=2030.4123456789,
        centerV=-0.000123456789123,
        semiMajor=695.58412345678,
        semiMinor=1e-7,
        phi=-1.5707963267948966,
        axis_ratio=0.963456789,
        target_radius=682.70812345,
        pre_mean=1234567.89,
        pre_chi2ndf=None,
        options_source="override",
        **{
            name: value
            for name, value in values.items()
            if name.startswith(
                ("pre_s", "pre_f", "pre_k", "pre_r", "post_", "raw", "corr", "phase")
            )
            or name == "timing_jitter_ns"
        },
    )


EXPECTED_CSV = (
    PLAN_HEADER + "\n"
    "1,15,0,7,cathode,C08,too_few_events,,12,,,"
    ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,batch\n"
    "2,16,1,25,cathode,C06,ok,extreme_axis_ratio;gauss_fit_failed_pre,123456,123000,456,"
    "2030.41235,-0.000123456789,695.584123,1e-07,-1.57079633,0.963457,682.708,"
    "1.23457e+06,0.666667,0.7,,0.766667,0.8,0.833333,"
    "0.866667,0.9,0.933333,0.966667,1,1.03333,1.06667,"
    "1.1,1.13333,1.16667,1.2,"
    "1.23333,1.26667,1.3,1.33333,1.36667,override\n"
)


def test_summary_csv_golden() -> None:
    results = [full_result(), make_result((1, 15, 0, 7), status="too_few_events", n_events=12)]
    text = format_summary_csv(results)
    assert text.splitlines()[0] == PLAN_HEADER == ",".join(CSV_COLUMNS)
    assert text == EXPECTED_CSV


def test_summary_csv_round_trip(tmp_path: Path) -> None:
    results = [
        full_result(),
        make_result((1, 15, 0, 7), status="too_few_events", n_events=12),
        ok_result((1, 15, 0, 4), pre_skewness=-0.0),
    ]
    path = write_summary_csv(tmp_path / "radial_summary.csv", results)
    back = read_summary_csv(path)
    assert [r.key for r in back] == sorted(r.key for r in results)
    by_key = {r.key: r for r in results}
    for loaded in back:
        original = by_key[loaded.key]
        for column in RESULT_COLUMNS:
            expected = getattr(original, column.name)
            if expected is not None and column.kind in ("float", "float_precise"):
                fmt = ".9g" if column.kind == "float_precise" else ".6g"
                expected = float(format(expected, fmt))
            assert getattr(loaded, column.name) == expected, column.name
    # Formatting what was read gives the same text
    assert format_summary_csv(back) == path.read_text()


def test_read_summary_csv_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("node,board\n1,15\n")
    with pytest.raises(ValueError, match="unexpected header"):
        read_summary_csv(path)
    good = format_summary_csv([make_result((1, 15, 0, 7), status="too_few_events")])
    path.write_text(good + "1,15\n")
    with pytest.raises(ValueError, match="line 3"):
        read_summary_csv(path)
    path.write_text(good.replace("too_few_events", "unknown_status"))
    with pytest.raises(ValueError, match="line 2"):
        read_summary_csv(path)


@pytest.mark.parametrize(
    ("kind", "value", "text"),
    [
        ("float", None, ""),
        ("float", 1.0, "1"),
        ("float", 2022.36491234, "2022.36"),
        ("float", 1e-7, "1e-07"),
        ("float", -0.0, "-0"),
        ("float_precise", 2022.36491234, "2022.36491"),
        ("float_precise", 1.5707963267948966, "1.57079633"),
        ("int", 17, "17"),
        ("count", None, ""),
        ("count", 0, "0"),
        ("flags", (), ""),
        ("flags", ("a", "b"), "a;b"),
        ("str", "C03", "C03"),
    ],
)
def test_format_and_parse_cell(kind: str, value: Any, text: str) -> None:
    assert format_cell(kind, value) == text
    parsed = parse_cell(kind, text)
    if kind in ("float", "float_precise") and value is not None:
        assert parsed == float(text)
    else:
        assert parsed == value


def test_atomic_write_keeps_the_old_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "out.txt"
    atomic_write_text(path, "old\n")

    def fail(*_args: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(path, "new\n")
    assert path.read_text() == "old\n"
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]


def test_tec_module_exports() -> None:
    assert tec.TEC_KEYS[4:] == (
        "centerU",
        "centerV",
        "semiMajor",
        "semiMinor",
        "phi",
        "radius",
        "radiusStd",
    )


# ---------------------------------------------------------------------------
# Atomic writes and the paired export
# ---------------------------------------------------------------------------


def test_atomic_write_texts_all_or_nothing(tmp_path: Path) -> None:
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    atomic_write_texts([(first, "a1\n", "utf-8"), (second, "b1\n", "ascii")])
    assert (first.read_text(), second.read_text()) == ("a1\n", "b1\n")
    # The second file fails to encode: neither file changes, no temporaries remain
    with pytest.raises(UnicodeEncodeError):
        atomic_write_texts([(first, "a2\n", "utf-8"), (second, "b2 µ\n", "ascii")])
    assert (first.read_text(), second.read_text()) == ("a1\n", "b1\n")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt", "b.txt"]
    # A missing directory for the second file: same
    with pytest.raises(OSError):
        atomic_write_texts([(first, "a3\n", "utf-8"), (tmp_path / "no" / "c.txt", "c", "utf-8")])
    assert first.read_text() == "a1\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt", "b.txt"]


def test_atomic_write_fsyncs_before_renaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(
        os, "replace", lambda a, b: (events.append("replace"), real_replace(a, b))[1]
    )
    atomic_write_texts([(tmp_path / "a", "1", "utf-8"), (tmp_path / "b", "2", "utf-8")])
    assert events[:4] == ["fsync", "fsync", "replace", "replace"]


def test_prepare_output_dir(tmp_path: Path) -> None:
    out = tmp_path / "x" / "y"
    tec_path, csv_path = prepare_output_dir(out, "run")
    assert out.is_dir() and (tec_path, csv_path) == (out / "run.tec", out / "radial_summary.csv")
    assert list(out.iterdir()) == []  # the write test leaves nothing behind
    (out / "run.tec").mkdir()
    with pytest.raises(IsADirectoryError, match="run.tec is a directory"):
        prepare_output_dir(out, "run")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(OSError, match="cannot create the output directory"):
        prepare_output_dir(blocker / "sub", "run")


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_prepare_output_dir_unwritable(tmp_path: Path) -> None:
    out = tmp_path / "ro"
    out.mkdir()
    out.chmod(0o555)
    try:
        with pytest.raises(OSError, match="cannot write to the output directory"):
            prepare_output_dir(out, "run")
    finally:
        out.chmod(0o755)


def test_write_outputs(tmp_path: Path) -> None:
    results = expected_tec_results()
    tec_path, csv_path = write_outputs(tmp_path, "run", results)
    assert tec_path.read_text() == format_tec(results) == EXPECTED_TEC
    assert csv_path.read_text() == format_summary_csv(results)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["radial_summary.csv", "run.tec"]
