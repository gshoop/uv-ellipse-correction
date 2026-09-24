"""Smoke tests for the phase 0 package skeleton and CLI wiring."""

from __future__ import annotations

import importlib
import re

import pytest

import uvcorr
from uvcorr import cli
from uvcorr.gui import main as gui_main

SKELETON_MODULES = [
    "uvcorr.channels",
    "uvcorr.cache",
    "uvcorr.ellipse",
    "uvcorr.metrics",
    "uvcorr.analysis",
    "uvcorr.io",
    "uvcorr.io.tec",
    "uvcorr.io.summary_csv",
    "uvcorr.cli",
    "uvcorr.gui",
    "uvcorr.gui.main",
]


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", uvcorr.__version__)


@pytest.mark.parametrize("name", SKELETON_MODULES)
def test_modules_import(name: str) -> None:
    importlib.import_module(name)


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"uvcorr {uvcorr.__version__}"


def test_cli_requires_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2
    assert "required" in capsys.readouterr().err


def test_cli_process_not_implemented(capsys: pytest.CaptureFixture[str]) -> None:
    # build-cache is implemented since phase 1 (tests/test_cli.py).
    assert cli.main(["process", "data.dat", "--output-dir", "out"]) == cli.EXIT_NOT_IMPLEMENTED
    assert "not implemented yet" in capsys.readouterr().err


def test_gui_entry_not_implemented(capsys: pytest.CaptureFixture[str]) -> None:
    assert gui_main.main() == 1
    assert "not implemented yet" in capsys.readouterr().err
