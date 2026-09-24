# uvcorr

UV ellipse correction for the RENA-3 fine-timing phase of the dual-panel CZT PET system, run
directly on raw `.dat` acquisition files. Each channel records its fine-timing phase as a
quadrature pair (U, V) that should lie on a circle but in practice lies on a tilted, offset
ellipse. `uvcorr` parses the raw file once into a sibling HDF5 UV cache (`<name>.dat.uv.h5`),
fits a robust least-squares ellipse to every active channel, maps it to a circle, and exports the
RadialAnalysis-format `<name>.tec` plus a `radial_summary.csv` of per-channel radial, residual and
phase metrics. A PyQt6/pyqtgraph GUI (`uvcorr-gui`) shows each channel's before/after U-V scatter
and diagnostics, and can re-fit channels with adjusted options. It is a separate package built on
[adc2kev](../adc2kev-python) (parser and detector geometry), like `specview`.

> Status: phase 0 (project skeleton). The CLI subcommands and the GUI are not implemented yet.
> See [`docs/planning/UV_ELLIPSE_CORRECTION_PLAN.md`](docs/planning/UV_ELLIPSE_CORRECTION_PLAN.md).

## Install

Requires Python 3.10 and an adc2kev checkout at `~/adc2kev-python` (not on PyPI).

```bash
python3.10 -m venv venv
venv/bin/pip install --upgrade pip setuptools wheel Cython
venv/bin/pip install -e ~/adc2kev-python    # builds adc2kev's Cython parser extension
venv/bin/pip install -e ".[dev]"
```

or, equivalently, `make venv install-dev` (use `make ADC2KEV=/path/to/adc2kev-python ...` for
another checkout).

Check that the fast Cython parser is available (without it, parsing falls back to pure Python and
is about 100x slower):

```bash
venv/bin/python -c "import adc2kev.parser.packet_parser as p; assert p.CYTHON_AVAILABLE and p.CYTHON_ARRAYS_AVAILABLE"
```

or `make cython-check`. If it fails, run `python setup.py build_ext --inplace` in
`~/adc2kev-python`.

## CLI usage

_To be written (phases 1 and 3)._ Planned:

```bash
uvcorr build-cache data.dat [--cache PATH] [--force]
uvcorr process data.dat --output-dir out/ [--cache PATH] [--workers N] \
    [--min-events 100] [--no-robust] [--clip-k 4] [--max-iter 5] [--geometric]
```

## GUI usage

_To be written (phases 4-6)._ Planned: `uvcorr-gui [file]`.

## Development

The Makefile uses `./venv` automatically when it exists.

| Target | Runs |
|--------|------|
| `make format` / `make format-check` | `black src/ tests/` (`--check`) |
| `make lint` | `ruff check src/ tests/` |
| `make type-check` | `mypy src/` |
| `make test` | `pytest -m "not realdata"` |
| `make test-realdata` | `pytest -m realdata` (needs the real test data) |
| `make dev-check` | format, lint, type-check, test |
| `make check` | format-check, lint, type-check, test (no file changes) |

GUI tests use pytest-qt and run on Qt's `offscreen` platform by default; set `QT_QPA_PLATFORM`
to override.
