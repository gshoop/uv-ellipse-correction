"""Tests for the C++ cross-check scripts: ``scripts/dump_uvd.py`` and ``scripts/compare_radial.py``.

``scripts/`` is not a package; the scripts are loaded from their files. Nothing
here runs the C++ RadialAnalysis binary: its output is written by hand in the
exact RadialAnalysis format instead.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from tests.conftest import RING_BOARDS, RING_CHANNELS, RingFiles
from uvcorr.analysis import ChannelKey, ChannelResult, FitOptions, analyze_board
from uvcorr.cache import UVCache
from uvcorr.channels import electrode_label, polarity_name
from uvcorr.ellipse import EllipseParams, ellipse_points
from uvcorr.io.summary_csv import write_summary_csv
from uvcorr.io.tec import format_tec_double

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# radial_summary.csv header line written by RadialAnalysis (main_radial.cpp)
CPP_HEADER_LINE = (
    "node,board,rena,channel,psi,"
    "pre_mean,pre_sigma,pre_fwhm,pre_chi2ndf,pre_skewness,pre_kurtosis,"
    "post_mean,post_sigma,post_fwhm,post_chi2ndf,post_skewness,post_kurtosis,"
    "ell_a,ell_b,ell_phi,"
    "ell_mean,ell_sigma,ell_fwhm,ell_chi2ndf,ell_skewness,ell_kurtosis,"
    "num_events,"
    "rawfit_res_mean,rawfit_res_sigma,corr_res_mean,corr_res_sigma,"
    "phase_mean_gap_rad,phase_max_gap_rad,phase_max_gap_ns,phase_ks"
)
CPP_HEADER = CPP_HEADER_LINE.split(",")


def _load(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module (registered so dataclasses resolve)."""
    module_name = f"_uvcorr_script_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dump_uvd() -> ModuleType:
    return _load("dump_uvd")


@pytest.fixture(scope="module")
def compare_radial() -> ModuleType:
    return _load("compare_radial")


# ---------------------------------------------------------------------------
# dump_uvd.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "name"),
    [
        ((1, 15, 0, 4), "node1board15rena00channel04.uvd"),
        ((10, 9, 1, 28), "node10board09rena01channel28.uvd"),
        ((3, 30, 1, 7), "node3board30rena01channel07.uvd"),
    ],
)
def test_uvd_filename_matches_build_uvd_filename(
    dump_uvd: ModuleType, key: tuple[int, int, int, int], name: str
) -> None:
    assert dump_uvd.uvd_filename(*key) == name
    assert dump_uvd.parse_uvd_filename(name) == key


def test_parse_uvd_filename_rejects_other_names(dump_uvd: ModuleType) -> None:
    for name in ("radial_summary.csv", "node1board15rena00.uvd", "nodeXboard15rena00channel04.uvd"):
        with pytest.raises(ValueError, match="not a .uvd"):
            dump_uvd.parse_uvd_filename(name)


def test_format_uvd_exact_text(dump_uvd: ModuleType) -> None:
    u = np.array([2520, 1, 4095], dtype=np.int16)
    v = np.array([2486, 0, 17], dtype=np.int16)
    assert dump_uvd.format_uvd(u, v) == "U Values V Values\n2520 2486\n1 0\n4095 17\n"
    assert dump_uvd.format_uvd(u[:0], v[:0]) == "U Values V Values\n"
    with pytest.raises(ValueError, match="equal length"):
        dump_uvd.format_uvd(u, v[:2])
    with pytest.raises(ValueError, match="integer"):
        dump_uvd.format_uvd(u.astype(float), v.astype(float))


@pytest.mark.parametrize(
    ("text", "pairs"),
    [
        # loadUVD discards line 1 whatever it is (here an event: extractData writes no header)
        ("2520 2486\n1 2\n3\t4\n\n5 6\n", [(1, 2), (3, 4), (5, 6)]),
        # no trailing whitespace: the last pair is read at EOF and popped
        ("U Values V Values\n1 2\n3 4", [(1, 2)]),
        ("U Values V Values\n1 2\n3 4 \n", [(1, 2), (3, 4)]),
        # an unpaired trailing value never counts
        ("U Values V Values\n1 2\n3\n", [(1, 2)]),
        ("U Values V Values\n1 2\n3", [(1, 2)]),
        ("U Values V Values\n", []),
        ("U Values V Values", []),
    ],
)
def test_read_uvd_follows_load_uvd(
    dump_uvd: ModuleType, tmp_path: Path, text: str, pairs: list[tuple[int, int]]
) -> None:
    path = tmp_path / "x.uvd"
    path.write_text(text, encoding="ascii")
    u, v = dump_uvd.read_uvd(path)
    assert list(zip(u.tolist(), v.tolist())) == [(float(a), float(b)) for a, b in pairs]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1:15,1:16", [(1, 15), (1, 16)]), (" 8:24 , 1:15,8:24", [(8, 24), (1, 15)])],
)
def test_parse_boards(dump_uvd: ModuleType, text: str, expected: list[tuple[int, int]]) -> None:
    assert dump_uvd.parse_boards(text) == expected


def test_parse_channels(dump_uvd: ModuleType) -> None:
    assert dump_uvd.parse_channels("0:4-6,1:28") == {(0, 4), (0, 5), (0, 6), (1, 28)}
    for bad in ("", "0", "0:x", "0:9-4", "a:1"):
        with pytest.raises(ValueError):
            dump_uvd.parse_channels(bad)
    for bad in ("", "15", "1:x"):
        with pytest.raises(ValueError):
            dump_uvd.parse_boards(bad)


def test_dump_writes_every_channel_exactly(
    dump_uvd: ModuleType, ring_files: RingFiles, tmp_path: Path
) -> None:
    out = tmp_path / "uvd"
    assert (
        dump_uvd.main(
            ["--cache", str(ring_files.cache), "--boards", "1:15,4:29", "--out", str(out)]
        )
        == 0
    )
    cache = UVCache(ring_files.cache)
    expected = {
        dump_uvd.uvd_filename(node, board, rena, channel): (node, board, rena, channel)
        for node, board in ((1, 15), (4, 29))
        for rena, channel, _ in cache.channels(node, board)
    }
    assert sorted(p.name for p in out.iterdir()) == sorted(expected)
    assert len(expected) == 2 * len(RING_CHANNELS)
    for name, (node, board, rena, channel) in expected.items():
        u, v = cache.channel_data(node, board, rena, channel)
        raw = (out / name).read_bytes()
        lines = raw.decode("ascii").split("\n")
        # header, one "u v" line per event in file order, and a final newline (empty tail)
        assert lines[0] == "U Values V Values"
        assert lines[-1] == ""
        assert lines[1:-1] == [f"{a} {b}" for a, b in zip(u.tolist(), v.tolist())]
        assert b"\r" not in raw and b"\t" not in raw
        # RadialAnalysis' loadUVD reads back every event
        ru, rv = dump_uvd.read_uvd(out / name)
        np.testing.assert_array_equal(ru, u)
        np.testing.assert_array_equal(rv, v)


def test_dump_channel_filter_and_dat_argument(
    dump_uvd: ModuleType, ring_files: RingFiles, tmp_path: Path
) -> None:
    out = tmp_path / "some"
    # A .dat argument uses its default cache <dat>.uv.h5 (the fixture's valid cache)
    args = [str(ring_files.dat), "--boards", "1:16", "--channels", "1:25-28,0:5", "--out", str(out)]
    assert dump_uvd.main(args) == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "node1board16rena00channel05.uvd",
        "node1board16rena01channel25.uvd",
        "node1board16rena01channel28.uvd",
    ]


def test_dump_refuses_a_non_empty_directory(
    dump_uvd: ModuleType, ring_files: RingFiles, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "uvd"
    base = ["--cache", str(ring_files.cache), "--out", str(out)]
    assert dump_uvd.main([*base, "--boards", "1:15"]) == 0
    stale = out / "node9board30rena00channel04.uvd"  # a stale file RadialAnalysis would read
    stale.write_text("U Values V Values\n1 2\n", encoding="ascii")
    (out / "RadialAnalysis_output").mkdir()
    before = sorted(p.name for p in out.iterdir())
    capsys.readouterr()
    assert dump_uvd.main([*base, "--boards", "1:16"]) == 2
    assert "not empty" in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == before  # untouched
    assert dump_uvd.main([*base, "--boards", "1:16", "--overwrite"]) == 0
    names = sorted(p.name for p in out.iterdir())
    assert "RadialAnalysis_output" in names and stale.name not in names
    assert all(n.startswith("node1board16") for n in names if n.endswith(".uvd"))
    assert sum(n.endswith(".uvd") for n in names) == len(RING_CHANNELS)


def test_dump_errors(
    dump_uvd: ModuleType, ring_files: RingFiles, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = str(tmp_path / "x")
    assert dump_uvd.main(["--cache", str(ring_files.cache), "--boards", "9:20", "--out", out]) == 1
    # nothing matches: an error, not "Wrote 0 .uvd files"
    capsys.readouterr()
    args = ["--cache", str(ring_files.cache), "--boards", "1:15", "--channels", "0:4", "--out", out]
    assert dump_uvd.main(args) == 1
    assert "no channel" in capsys.readouterr().err
    assert not Path(out).exists() or not any(Path(out).iterdir())
    assert dump_uvd.main(["--cache", str(tmp_path / "missing.h5"), "--out", out]) == 2
    stray = tmp_path / "stray.dat"
    stray.write_bytes(b"\x00" * 16)
    assert dump_uvd.main([str(stray), "--out", out]) == 2  # no valid cache for it
    assert dump_uvd.main([str(ring_files.dat), "--cache", str(ring_files.cache), "--out", out]) == 2


# ---------------------------------------------------------------------------
# compare_radial.py: helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (3.09506, 3.09506 - math.pi, 0.0),  # C++ (-pi, pi] vs uvcorr (-pi/2, pi/2]
        (-0.0465, 3.09506, -0.0465 - 3.09506 + math.pi),
        (math.pi / 2, -math.pi / 2 + 1e-9, -1e-9),  # across the wrap of (-pi/2, pi/2]
        (0.3, 0.3 + 2 * math.pi, 0.0),
        (0.1, -0.1, 0.2),
    ],
)
def test_phi_diff_mod_pi(compare_radial: ModuleType, a: float, b: float, expected: float) -> None:
    got = compare_radial.phi_diff_mod_pi(a, b)
    assert abs(got) <= math.pi / 2
    assert got == pytest.approx(expected, abs=1e-12)


def test_phi_tolerance_grows_for_near_circles(compare_radial: ModuleType) -> None:
    tol = compare_radial.phi_tolerance
    assert tol(1.0, 1.0, 1e-5) == math.inf
    base = compare_radial.PHI_ABS_TOL
    assert tol(2.0, 1.0, 1e-5) == pytest.approx(base + 1e-5 * 4.0 / 3.0)
    assert tol(600.0, 580.0, 1e-5) > tol(600.0, 300.0, 1e-5)


@pytest.mark.parametrize(
    ("value", "half"),
    [(2030.41, 0.005), (2036.0, 0.005), (717.906, 0.0005), (3.09506, 5e-6), (-0.0465, 5e-8)],
)
def test_printed_half_quantum(compare_radial: ModuleType, value: float, half: float) -> None:
    assert compare_radial.printed_half_quantum(value) == pytest.approx(half)


def test_read_cpp_csv(compare_radial: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "radial_summary.csv"
    row_ok = ["1", "15", "0", "4", "-0.003957"] + ["1.5"] * 12 + ["717.905517", "691.088197"]
    row_ok += ["3.095061"] + ["2.0"] * 6 + ["105223"] + ["0.1"] * 8
    row_no_ell = ["1", "15", "0", "5", "0.000000"] + ["1.5"] * 6 + [""] * 6 + [""] * 9
    row_no_ell += ["42"] + [""] * 8
    path.write_text(
        ",".join(CPP_HEADER) + "\n" + ",".join(row_ok) + "\n" + ",".join(row_no_ell) + "\n",
        encoding="ascii",
    )
    rows = compare_radial.read_cpp_csv(path)
    assert list(rows) == [ChannelKey(1, 15, 0, 4), ChannelKey(1, 15, 0, 5)]
    ok = rows[ChannelKey(1, 15, 0, 4)]
    assert ok["ell_a"] == 717.905517 and ok["ell_phi"] == 3.095061 and ok["num_events"] == 105223
    assert isinstance(ok["num_events"], int)
    no_ell = rows[ChannelKey(1, 15, 0, 5)]
    assert no_ell["ell_a"] is None and no_ell["post_mean"] is None and no_ell["pre_mean"] == 1.5
    assert no_ell["num_events"] == 42

    bad = tmp_path / "bad.csv"
    bad.write_text("node,board,rena\n1,15,0\n", encoding="ascii")
    with pytest.raises(ValueError, match="not a RadialAnalysis"):
        compare_radial.read_cpp_csv(bad)


def test_root_hist_puts_the_maximum_in_the_overflow(compare_radial: ModuleType) -> None:
    values = np.array([0.0, 0.5, 1.0, 9.99, 10.0])
    hist = compare_radial.root_hist(values, 10, 0.0, 10.0)
    assert hist.counts.sum() == 4 and hist.counts[0] == 2 and hist.counts[9] == 1
    assert hist.mean == pytest.approx(np.mean(values[:4]))  # 10.0 is not in the statistics
    assert hist.std == pytest.approx(np.std(values[:4]))


def test_cpp_find_center_on_a_ring(compare_radial: ModuleType) -> None:
    p = EllipseParams(cx=2030.0, cy=2040.0, a=600.0, b=580.0, phi=0.4)
    t = np.random.default_rng(5).uniform(0.0, 2 * math.pi, 20_000)
    u, v = ellipse_points(p, t)
    center = compare_radial.cpp_find_center(np.rint(u), np.rint(v))
    assert center is not None
    assert (
        abs(center[0] - p.cx) < 15 and abs(center[1] - p.cy) < 15
    )  # within a 100-bin histogram bin
    assert compare_radial.cpp_find_center(np.array([1.0, 1.0]), np.array([2.0, 2.0])) is None


# ---------------------------------------------------------------------------
# compare_radial.py: comparison on hand-made C++ output
# ---------------------------------------------------------------------------


def _uv_result(key: ChannelKey, params: EllipseParams, n: int, **extra: Any) -> ChannelResult:
    values: dict[str, Any] = {
        "node": key.node,
        "board": key.board,
        "rena": key.rena,
        "channel": key.channel,
        "polarity": polarity_name(key.board, key.rena, key.channel),
        "electrode": electrode_label(key.board, key.rena, key.channel),
        "status": "ok",
        "n_events": n,
        "n_used": n,
        "n_rejected": 0,
        "centerU": params.cx,
        "centerV": params.cy,
        "semiMajor": params.a,
        "semiMinor": params.b,
        "phi": params.phi,
        "axis_ratio": params.axis_ratio,
        "target_radius": params.target_radius,
        "pre_mean": 700.0,
        "pre_sigma": 9.0,
        "post_mean": 704.0,
        "post_sigma": 7.0,
    }
    values.update(extra)
    return ChannelResult(**values)


def _cpp_row(
    key: ChannelKey, params: EllipseParams, n: int, phi_offset: float = 0.0
) -> dict[str, str]:
    row = dict.fromkeys(CPP_HEADER, "")
    row.update(
        node=str(key.node),
        board=str(key.board),
        rena=str(key.rena),
        channel=str(key.channel),
        psi="0.000000",
        pre_mean="703.000000",
        pre_sigma="10.000000",
        ell_a=f"{params.a:.6f}",
        ell_b=f"{params.b:.6f}",
        ell_phi=f"{params.phi + phi_offset:.6f}",
        ell_mean="704.000000",
        ell_sigma="6.500000",
        num_events=str(n),
    )
    return row


def _tec_block(key: ChannelKey, params: EllipseParams, phi_offset: float = 0.0) -> str:
    lines = [f"\tnode={key.node}", f"\tboard={key.board}", f"\trena={key.rena}"]
    lines.append(f"\tchannel={key.channel}")
    for name, value in (
        ("centerU", params.cx),
        ("centerV", params.cy),
        ("semiMajor", params.a),
        ("semiMinor", params.b),
        ("phi", params.phi + phi_offset),
        ("radius", params.target_radius),
        ("radiusStd", 6.5),
    ):
        lines.append(f"\t{name}={format_tec_double(value)}")
    return "channel{\n" + "\n".join(lines) + "\n}\n"


def _write_cpp_run(
    directory: Path,
    inputs: list[ChannelKey],
    rows: list[dict[str, str]],
    tec_blocks: list[str],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for key in inputs:
        name = f"node{key.node}board{key.board:02d}rena0{key.rena}channel{key.channel:02d}.uvd"
        (directory / name).write_text("U Values V Values\n", encoding="ascii")
    out = directory / "RadialAnalysis_output"
    out.mkdir()
    with open(out / "radial_summary.csv", "w", encoding="ascii", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CPP_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (out / f"{directory.name}.tec").write_text("".join(tec_blocks), encoding="ascii")
    return directory


def test_compare_hand_made_run(compare_radial: ModuleType, tmp_path: Path) -> None:
    good = ChannelKey(1, 15, 0, 4)  # agrees; C++ phi is uvcorr's + pi
    off = ChannelKey(1, 15, 0, 5)  # C++ centre 0.1 ADC away: beyond the tolerance
    skipped = ChannelKey(1, 15, 0, 6)  # .uvd input, no C++ row (e.g. its pre fit failed)
    few = ChannelKey(1, 15, 0, 7)  # C++ fitted it, uvcorr has too few events
    blob = ChannelKey(1, 15, 0, 8)  # a pedestal blob (broad_ring): C++ axis 30 % off
    p = EllipseParams(cx=2028.3912, cy=2035.7481, a=717.905517, b=691.088197, phi=-0.046532)
    p_off = EllipseParams(p.cx + 0.1, p.cy, p.a, p.b, p.phi)
    p_blob = EllipseParams(cx=2067.14, cy=2077.8, a=18.8, b=6.6, phi=-0.77)
    p_blob_cpp = EllipseParams(cx=2067.29, cy=2077.66, a=24.4, b=6.7, phi=2.37)
    run = _write_cpp_run(
        tmp_path / "uvd",
        [good, off, skipped, few, blob],
        [
            _cpp_row(good, p, 1000, math.pi),
            _cpp_row(off, p, 1000),
            _cpp_row(few, p, 50, math.pi),
            _cpp_row(blob, p_blob_cpp, 700),
        ],
        [
            _tec_block(good, p, math.pi),
            _tec_block(off, p_off),
            _tec_block(few, p),
            _tec_block(blob, p_blob_cpp),
        ],
    )
    cpp = compare_radial.read_cpp_output(run / "RadialAnalysis_output")  # the output dir works too
    assert cpp.inputs == {good, off, skipped, few, blob}
    assert set(cpp.rows) == {good, off, few, blob} and set(cpp.tec) == {good, off, few, blob}

    identity = {"polarity": "anode", "electrode": "A01"}
    uv = {
        good: _uv_result(good, p, 1000),
        off: _uv_result(off, p, 1001),  # also a count mismatch
        skipped: _uv_result(skipped, p, 900),
        few: ChannelResult(
            node=1, board=15, rena=0, channel=7, status="too_few_events", n_events=50, **identity
        ),
        blob: _uv_result(blob, p_blob, 700, flags=("broad_ring",)),
    }
    comp = compare_radial.compare(cpp, uv, rtol=1e-5)
    by_key = {ChannelKey(r["node"], r["board"], r["rena"], r["channel"]): r for r in comp.rows}
    assert set(by_key) == {good, off, skipped, few, blob}

    row = by_key[good]
    assert row["cpp_row"] and row["cpp_tec"] and row["uv_status"] == "ok"
    assert row["phi_ok"] and abs(row["phi_delta"]) < 1e-6  # equal modulo pi
    assert row["centerU_ok"] and row["semiMajor_ok"] and row["radius_ok"]
    assert row["post_sigma_rel"] == pytest.approx(7.0 / 6.5 - 1.0)
    assert row["pre_mean_rel"] == pytest.approx(700.0 / 703.0 - 1.0)
    assert row["timing_jitter_ns_cpp"] == pytest.approx(6.5 / (2 * math.pi * 490e3 * 704.0) * 1e9)

    assert [(k, c.name) for k, c in comp.param_failures] == [(off, "centerU")]
    # the blob disagrees on every parameter, but is only reported, not a failure
    assert comp.blob_channels == {blob} and by_key[blob]["uv_broad_ring"]
    assert {c.name for k, c in comp.blob_failures} >= {"centerU", "semiMajor", "phi"}
    assert all(k == blob for k, _ in comp.blob_failures)
    strict = compare_radial.compare(cpp, uv, rtol=1e-5, strict_blobs=True)
    assert {k for k, _ in strict.param_failures} == {off, blob}
    assert comp.count_mismatches == [(off, 1000, 1001)]
    assert not by_key[skipped]["cpp_row"] and by_key[skipped]["uv_status"] == "ok"
    assert by_key[few]["cpp_tec"] and by_key[few]["uv_status"] == "too_few_events"
    assert "centerU_cpp" not in by_key[few]  # no parameter comparison without a uvcorr fit

    lines: list[str] = []
    compare_radial.print_summary(cpp, comp, 1e-5, out=lines.append)
    text = "\n".join(lines)
    assert "C++ skipped node 1 board 15 rena 0 channel 6" in text
    assert "DISAGREEMENTS beyond tolerance: 1 on 1 channels" in text
    assert "EVENT COUNT MISMATCH" in text
    assert "Ellipse parameters of the 2 ring channels" in text
    assert "broad_ring channels (blobs, non-rings): 1 compared, 1 disagree" in text
    assert "(not failures)" in text

    out_csv = tmp_path / "comparison.csv"
    compare_radial.write_comparison_csv(out_csv, comp.rows)
    with open(out_csv, encoding="utf-8", newline="") as f:
        table = list(csv.DictReader(f))
    assert len(table) == 5 and table[0]["phi_ok"] == "1"


def test_compare_cli_blobs_and_tec_choice(compare_radial: ModuleType, tmp_path: Path) -> None:
    ring, blob = ChannelKey(1, 16, 0, 5), ChannelKey(1, 16, 0, 6)
    p = EllipseParams(cx=2030.1234, cy=2040.5678, a=600.25, b=580.5, phi=0.4)
    p_blob = EllipseParams(cx=2067.14, cy=2077.8, a=18.8, b=6.6, phi=-0.77)
    p_blob_cpp = EllipseParams(cx=2067.29, cy=2077.66, a=24.4, b=6.7, phi=2.37)
    run = _write_cpp_run(
        tmp_path / "uvd",
        [ring, blob],
        [_cpp_row(ring, p, 1000), _cpp_row(blob, p_blob_cpp, 700)],
        [_tec_block(ring, p), _tec_block(blob, p_blob_cpp)],
    )
    uv_dir = tmp_path / "uvcorr"
    uv_dir.mkdir()
    write_summary_csv(
        uv_dir / "radial_summary.csv",
        [_uv_result(ring, p, 1000), _uv_result(blob, p_blob, 700, flags=("broad_ring",))],
    )
    base = [str(run), "--uvcorr-dir", str(uv_dir)]
    assert compare_radial.main(base) == 0  # the blob alone does not fail the run
    assert compare_radial.main([*base, "--strict-blobs"]) == 1

    # two .tec files: refused unless one is chosen
    out = run / "RadialAnalysis_output"
    (out / "other.tec").write_text(_tec_block(ring, p), encoding="ascii")
    assert compare_radial.main(base) == 2
    with pytest.raises(ValueError, match="several .tec"):
        compare_radial.read_cpp_output(run)
    assert compare_radial.main([*base, "--tec", str(out / "uvd.tec")]) == 0
    assert compare_radial.main([*base, "--tec", str(out / "missing.tec")]) == 2


def test_parse_uvd_filename_is_shared(compare_radial: ModuleType, dump_uvd: ModuleType) -> None:
    name = dump_uvd.uvd_filename(10, 9, 1, 28)
    assert compare_radial.parse_uvd_filename(name) == ChannelKey(10, 9, 1, 28)
    with pytest.raises(ValueError, match="not a .uvd"):
        compare_radial.parse_uvd_filename("radial_summary.csv")


def _pinned_values() -> list[float]:
    """A deterministic, RNG-free sample: Irwin-Hall(3) from low-discrepancy sequences, 3 outliers.

    Only IEEE multiplication and fmod are used, so the values are identical on every platform.
    """
    a1, a2, a3 = 0.6180339887498949, 0.7548776662466927, 0.5698402909980532
    values = [
        400.0 + 10.0 * ((i * a1) % 1.0 + (i * a2) % 1.0 + (i * a3) % 1.0) for i in range(1, 3001)
    ]
    return [*values, 350.0, 470.0, 480.0]


def test_emulation_matches_root(compare_radial: ModuleType) -> None:
    # Reference: ROOT 6.26/10 via PyROOT on the same values. TH1D("h", "", 200, min, max), Fill,
    # h->Fit("gaus", "QS", "", mean - 3 RMS, mean + 3 RMS) (NDF 40), GetSkewness/GetKurtosis;
    # and residualHistogram's TH1D(150, m - 6 s, m + 6 s) with its fit. The maximum (480)
    # lands in the overflow bin. Minuit converges to ~1e-6 relative.
    values = _pinned_values()
    fit = compare_radial.root_radial_fit(values)
    assert fit.ok
    assert fit.mean == pytest.approx(415.0035264532038, abs=1e-4)
    assert fit.sigma == pytest.approx(5.075799258572229, rel=1e-5)
    assert fit.chi2ndf == pytest.approx(0.40305425109530474, rel=1e-6)
    assert fit.skewness == pytest.approx(-0.21137508427908294, rel=1e-9)
    assert fit.kurtosis == pytest.approx(11.016747843441602, rel=1e-9)
    hist = compare_radial.root_hist(values, 200, min(values), max(values))
    assert hist.mean == pytest.approx(414.99628545928, rel=1e-9)  # ROOT GetMean (sum order)
    assert hist.std == pytest.approx(5.229103578244744, rel=1e-9)  # ROOT GetRMS
    mean, sigma, ok = compare_radial.cpp_residual_stats(values)
    assert ok
    assert mean == pytest.approx(415.01374703564244, abs=1e-4)
    assert sigma == pytest.approx(5.073083603934352, rel=1e-5)


def test_compare_cli_end_to_end_on_the_ring_cache(
    compare_radial: ModuleType, dump_uvd: ModuleType, ring_files: RingFiles, tmp_path: Path
) -> None:
    # Fake a RadialAnalysis run from uvcorr's own --no-robust results (C++ layout, 6 decimals,
    # phi shifted by pi), then compare it on the fly against the cache.
    node, board = RING_BOARDS[0]
    run = tmp_path / "uvd"
    assert (
        dump_uvd.main(
            ["--cache", str(ring_files.cache), "--boards", f"{node}:{board}", "--out", str(run)]
        )
        == 0
    )
    results = analyze_board(ring_files.cache, node, board, FitOptions(robust=False))
    fitted = [(r.key, r.params, r.n_events) for r in results if r.ok and r.params is not None]
    rows = [_cpp_row(key, p, n, math.pi) for key, p, n in fitted]
    blocks = [_tec_block(key, p, math.pi) for key, p, _ in fitted]
    out = run / "RadialAnalysis_output"
    out.mkdir()
    with open(out / "radial_summary.csv", "w", encoding="ascii", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CPP_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (out / "uvd.tec").write_text("".join(blocks), encoding="ascii")

    comparison = tmp_path / "comparison.csv"
    args = [str(run), "--cache", str(ring_files.cache), "--decompose", "--out", str(comparison)]
    assert compare_radial.main(args) == 0
    with open(comparison, encoding="utf-8", newline="") as f:
        table = list(csv.DictReader(f))
    assert len(table) == len(RING_CHANNELS)
    compared = [r for r in table if r.get("centerU_ok")]
    assert len(compared) == len(rows) and all(r["phi_ok"] == "1" for r in compared)
    assert all(r["dec_emul_post_sigma"] for r in compared)  # the decomposition ran

    # A C++ centre 1 ADC off is reported and fails the run
    key, p, _ = fitted[0]
    blocks[0] = _tec_block(key, EllipseParams(p.cx + 1.0, p.cy, p.a, p.b, p.phi), math.pi)
    (out / "uvd.tec").write_text("".join(blocks), encoding="ascii")
    assert compare_radial.main([str(run), "--cache", str(ring_files.cache)]) == 1


def test_compare_cli_argument_errors(compare_radial: ModuleType, tmp_path: Path) -> None:
    assert compare_radial.main([str(tmp_path)]) == 2  # no uvcorr side
    assert compare_radial.main([str(tmp_path), "--uvcorr-dir", str(tmp_path), "--decompose"]) == 2
    assert compare_radial.main([str(tmp_path), "--uvcorr-dir", str(tmp_path)]) == 2  # no C++ CSV
