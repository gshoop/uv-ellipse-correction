# uvcorr

UV ellipse correction of the RENA-3 fine-timing phase of the dual-panel CZT PET system, run
directly on raw `.dat` acquisition files.

Each RENA-3 channel records its fine-timing phase as a quadrature pair (U, V). The points of a
channel should lie on a circle, but they lie on a tilted, offset ellipse. `uvcorr` parses the raw
file once into a sibling HDF5 UV cache, fits a robust least-squares ellipse to every active
channel, maps it to a circle, and writes the correction as a RadialAnalysis-format `<name>.tec`
plus a `radial_summary.csv` of per-channel radial, residual and phase metrics. The GUI
(`uvcorr-gui`) shows each channel's before/after U-V scatter and diagnostics, re-fits channels or
boards with other options, and exports the results.

- **Relation to RadialAnalysis** (C++/ROOT, `~/DataProcessing/EllipseCorrection`). `uvcorr`
  replaces the chain `extractData -t` → per-channel `.uvd` files → `RadialAnalysis`. The ellipse
  fit (Fitzgibbon, Halir-Flusser reduced form) and the correction (`applyEllipseCorrection`) are
  the same, and the `.tec` is written byte for byte in the RadialAnalysis layout. It also computes
  all the RadialAnalysis radial, residual and phase metrics. It drops the legacy shear/ψ method.
  The fit runs on normalised coordinates and is robust by default, and the radial statistics use
  a different Gaussian estimator. [`docs/ALGORITHM.md`](docs/ALGORITHM.md) lists every difference
  and the C++ cross-check on the full test file.
- **Relation to adc2kev** (`~/adc2kev-python`). `uvcorr` is a separate package that depends on
  adc2kev, as `specview` does. It uses adc2kev's `PacketParser` (Cython), `ElectrodeMap` and
  detector geometry, and follows its GUI conventions. The System Map is a fork of the adc2kev
  2.2.7 map.

Status: version 0.1.0. Phases 0-6 of the plan
([`docs/planning/UV_ELLIPSE_CORRECTION_PLAN.md`](docs/planning/UV_ELLIPSE_CORRECTION_PLAN.md))
are done.

```
raw .dat
    │
    ▼
[adc2kev PacketParser] ────────► streamed, Cython; inactive channels and node 0 dropped
    │
    ▼
[UV cache <dat>.uv.h5] ────────► /events/node_N/board_B/: rena, channel, u, v, pha
    │                              (uvcorr build-cache: once, ~21 s for 3.5 GB)
    ▼
[analysis, one task per board] ► per channel: robust ellipse fit, correction,
    │                              radial / residual / phase metrics (uvcorr process, ~16 s)
    ├──► <stem>.tec                RadialAnalysis layout, ok channels
    ├──► radial_summary.csv        every channel
    └──► /results/current          in the cache
            │
            ▼
[uvcorr-gui] ──────────────────► browse, re-fit (stored as overrides in the cache), export
```

---

## Install

| Requirement | Note |
|-------------|------|
| Python 3.10 | `python3.10` |
| System packages (Debian/Ubuntu) | `python3.10-venv`, `python3.10-dev` and `build-essential` (a C compiler and the Python headers for adc2kev's Cython build); for the GUI on X11/WSLg, `libxcb-cursor0` (Qt ≥ 6.5) |
| adc2kev ≥ 2.2.7 | checkout at `~/adc2kev-python`, installed editable (not on PyPI) |
| adc2kev's Cython parser extension | about 100× faster than the pure-Python fallback |
| numpy < 2, scipy, h5py, PyQt6, pyqtgraph, threadpoolctl | installed with `uvcorr` |

On Ubuntu: `sudo apt install python3.10-venv python3.10-dev build-essential libxcb-cursor0`. The
GUI needs a display (`DISPLAY` set, e.g. by WSLg); without one, `QT_QPA_PLATFORM=offscreen` runs
it headless (the GUI tests do this by default).

```bash
cd ~/uv-ellipse-correction
python3.10 -m venv venv
venv/bin/pip install --upgrade pip setuptools wheel Cython
venv/bin/pip install -e ~/adc2kev-python    # builds adc2kev's Cython parser extension
venv/bin/pip install -e ".[dev]"            # uvcorr, its dependencies and the dev tools
```

`make venv install-dev` does the same. Use `make ADC2KEV=/path/to/adc2kev-python install-dev`
for another adc2kev checkout. `.[dev]` adds pytest, pytest-cov, pytest-qt, black, ruff and mypy.
Use `pip install -e .` without them.

**Cython check.** Without the extension, parsing falls back to pure Python.

```bash
venv/bin/python -c "import adc2kev.parser.packet_parser as p; assert p.CYTHON_AVAILABLE and p.CYTHON_ARRAYS_AVAILABLE"
make cython-check                           # the same, prints "adc2kev Cython parser: OK"
```

If it fails, build the extension inside `~/adc2kev-python` with this venv's Python (there may
be no `python` on `PATH`): `cd ~/adc2kev-python && ~/uv-ellipse-correction/venv/bin/python
setup.py build_ext --inplace`.

**threadpoolctl** is a runtime dependency. The analysis runs one worker process per board. Each
worker limits its BLAS/OpenMP pools to one thread with `threadpoolctl`, because numpy's OpenBLAS
would otherwise start one thread per core in every worker. That is 15-18 % faster on the test
file. No environment variable is set.

**Fallback: adc2kev's venv.** If a fresh venv causes trouble, install `uvcorr` into
`~/adc2kev-python/venv`. It already has adc2kev (editable, with the extension), threadpoolctl and
compatible versions of all the dev tools, so pip adds only `uvcorr`:

```bash
~/adc2kev-python/venv/bin/pip install -e ".[dev]"
make VENV=$HOME/adc2kev-python/venv check   # the Makefile uses $(VENV)/bin/python
```

## Quick start

```bash
source venv/bin/activate                         # or call venv/bin/uvcorr etc.
uvcorr build-cache data.dat                      # once: writes data.dat.uv.h5 next to the .dat
uvcorr process data.dat --output-dir out/        # out/data.tec, out/radial_summary.csv
uvcorr-gui data.dat                              # or: uvcorr-gui data.dat.uv.h5
```

`process` builds the cache itself when it is missing or stale, so `build-cache` is optional. In
the GUI, *Process > Fit All* does what `process` does. Its results are stored in the cache, and
*File > Export Both* writes the two files.

---

## CLI reference

```
uvcorr [--version] [-v | -vv] {build-cache,process} ...
```

`-v` logs at INFO, `-vv` at DEBUG (to stderr). Summaries and the census go to **stdout**.
Progress, warnings and errors go to **stderr**. On a terminal the progress line is redrawn in
place at every 1 %. On a pipe or file a new line is written every 10 %.

### `uvcorr build-cache`

```
uvcorr build-cache DAT [--cache PATH] [--force]
```

| Option | Default | Meaning |
|--------|---------|---------|
| `DAT` | (required) | Raw `.dat` acquisition file |
| `--cache PATH` | `<DAT>.uv.h5` | Cache location, e.g. when the data directory is read-only |
| `--force` | off | Rebuild even if a valid cache exists |

A valid cache is reused (the check takes about 1 ms). Otherwise the cache is built (see
[The UV cache](#the-uv-cache)). Real run on the test file (paths shortened, progress lines
elided):

```
$ uvcorr build-cache data_20260910_120628.dat
Building UV cache data_20260910_120628.dat.uv.h5 (no cache yet)
parsing:   0% (  0.5 s)
parsing:  10% (  2.1 s)
...
WARNING adc2kev.parser.packet_parser: Parser finished with 1 bytes remaining in buffer
parsing: 100% ( 20.6 s)
UV cache: data_20260910_120628.dat.uv.h5
  status:        built in 20.6 s
  events kept:   208,267,950 on 155 boards (nodes 1-10, boards 15-30)
  dropped:       224,956 on inactive channels, 0 from node 0
  parser:        106,063,959 frames, 208,492,906 events, 23 dropped frames
  cache size:    1.26 GB
  time:          20.6 s
```

On a valid cache the status line reads `reused (valid for the .dat; built <time>)`. The adc2kev
warning about 1 remaining byte is emitted for this file and is harmless. A file in which the
parser finds no valid frame (not raw data, or an empty file) is an error: `error: no valid
frames in <file>: is this a raw .dat file?` (exit code 1). No cache is written, and a previous
cache is left untouched. The same holds for `process` and the GUI. A raw file whose frames all
carry inactive channels gives a valid, empty cache.

### `uvcorr process`

```
uvcorr process DAT --output-dir DIR [--cache PATH] [--workers N] [--min-events N]
               [--no-robust] [--clip-k K] [--max-iter N] [--geometric] [--discard-overrides]
```

| Option | Default | Meaning |
|--------|---------|---------|
| `DAT` | (required) | Raw `.dat` file. With a valid cache only its size, mtime and first MiB are read |
| `--output-dir DIR` | (required) | Output directory, created if needed |
| `--cache PATH` | `<DAT>.uv.h5` | Cache location (built if missing or stale) |
| `--workers N` | `min(8, usable CPUs)` | Worker processes, capped at the number of boards. `1` runs in-process |
| `--min-events N` | `100` | Channels with fewer events get `too_few_events`. Must be ≥ 6, the minimum for the fit |
| `--no-robust` | robust on | Disable the MAD-clipped refit (plain algebraic fit) |
| `--clip-k K` | `4` | Robust clip: keep \|res − median\| ≤ K · 1.4826 · MAD |
| `--max-iter N` | `5` | Maximum robust clip-and-refit iterations |
| `--geometric` | off | Refine the ellipse by least squares on the radial residuals (about 4× slower, no material change on real rings) |
| `--discard-overrides` | keep | Ignore the per-channel GUI overrides stored in the cache and delete them |

The defaults come from `uvcorr.options.FitOptions`, and `--help` prints them. Options not
given keep their `FitOptions` defaults. Four more `FitOptions` fields have no CLI option. Set
them through the Python API:

| Field | Default | Used for |
|-------|---------|----------|
| `phase_ref_freq_hz` | 490e3 | Phase gaps and jitter in ns |
| `high_rejection_frac` | 0.05 | `high_rejection` flag |
| `extreme_axis_ratio` | 0.5 | `extreme_axis_ratio` flag |
| `broad_ring_frac` | 0.05 | `broad_ring` flag (retuned from 0.1, [ALGORITHM 7.6](docs/ALGORITHM.md#76-broad_ring_frac-retuned-01--005)) |

What `process` does, in order:

1. It creates and write-tests the output directory, so a long run cannot end in a permission error.
2. It reuses the cache or builds it. A stale cache that stores results is rebuilt with a warning,
   and its results and overrides are lost.
3. It reads the stored overrides, unless `--discard-overrides` is given.
4. It fits every active channel (one task per board, largest boards first).
5. It writes `<DAT stem>.tec` and `radial_summary.csv` from the batch results with the overrides
   applied. Both files are replaced atomically.
6. It stores the batch results in the cache as `/results/current`. The existing overrides are
   kept unless `--discard-overrides`. A read-only cache skips this step with a warning, and the
   output files are still written.
7. It prints the census and the timings.

An override fitted with options that fit exactly like the new batch options is reproduced by the
batch, so it is neither applied (step 5) nor kept (step 6). The rule is the GUI's Fit All rule
(`uvcorr.options.same_fit`): all nine `FitOptions` fields are compared, except that with robust
off `clip_k` and `max_iter` are ignored on both sides. An override whose stored options include
a field this uvcorr does not know (written by a newer version) is always kept. The census then
adds e.g. `2 overrides now match the batch options and were dropped` (on a read-only cache: `…
were not applied`).

**Every run replaces the stored batch.** A run with non-default options (e.g. `--no-robust` for
a test) makes those options the cache's batch, which the GUI then shows, and permanently drops
the overrides that those options reproduce (e.g. every robust-off override). To try options
without touching the stored results, give the run its own scratch cache (`--cache
/tmp/data.uv.h5`, built in about 21 s for the test file), or use a read-only tool such as
`compare_radial.py --cache`.

Real run on the test file, cache reused (paths shortened):

```
$ uvcorr process data_20260910_120628.dat --output-dir out/
Fitting 155 boards with 8 worker(s), options: defaults
fitting:   0% (  0.0 s)
fitting:  10% (  3.0 s)
...
fitting: 100% ( 15.6 s)
UV cache: data_20260910_120628.dat.uv.h5 (reused)
Analysis: 6,557 channels (208,267,950 events) on 155 boards, 8 worker(s), options: defaults
  status:     ok 6,115, too_few_events 442, fit_failed 0
  flags:      high_rejection 128, extreme_axis_ratio 9, broad_ring 106, gauss_fit_failed_pre 1,093, gauss_fit_failed_post 15
              (1,299 channels with at least one flag)
  overrides:  0 applied
Outputs:
  out/data_20260910_120628.tec  (6,115 channel blocks, 1.04 MB)
  out/radial_summary.csv  (6,557 rows, 2.01 MB)
  results stored in the cache (/results/current)
Time: cache 0.0 s, analysis 15.8 s, write 0.1 s, store 0.0 s, total 15.9 s
```

Non-default options are listed instead of `defaults`, e.g. `options: min_events=50,
robust=False`. The census counts the exported (merged) rows. `overrides:` gives the number
applied, or `discarded (--discard-overrides)`, and the number dropped because the batch
reproduces them.

### Exit codes (both subcommands)

| Code | When |
|-----:|------|
| 0 | Success |
| 1 | An error: failed build or analysis, a file without any valid frame (not raw data), unusable cache path, cache in use by another process, too little disk space, unreadable stored results (`process --discard-overrides` skips reading them), or the outputs were written but storing the results failed |
| 2 | Missing `.dat`, bad arguments (e.g. `--min-events 5`, missing `--output-dir`), or `--output-dir` is an existing file |
| 130 | Stopped with Ctrl-C |

Ctrl-C sets a stop flag, and further Ctrl-Cs are ignored until the cleanup is done. A build stops
within one parser batch, removes its temporary file and leaves any previous cache untouched. An
analysis stops its workers at their next channel, and nothing is written or stored.

---

## The UV cache

`<dat path>.uv.h5` (e.g. `data_20260910_120628.dat.uv.h5`), or any path given with `--cache`.
The GUI's *Open Raw* always uses the default path. Open a cache stored elsewhere with *Open Cache*.

```
/metadata                      source path/size/mtime/hash, parser totals, events kept/dropped,
                               build time, layout version (uv_cache_version 1.0.0)
/events/node_N/board_B/        rena, channel (int8), u, v, pha (int16), file order; lzf + shuffle,
                               65,536-event chunks; attr n_events
/results/current/              the last batch run: table (one row per channel, every CSV column),
                               attrs options_json, created_at, uvcorr_version
/results/current/overrides/    per-channel GUI re-fits: same columns plus options_json per row
```

- **Contents.** Every event of an active channel is kept: RENA 0 channels 4-28 and RENA 1
  channels 7-28, anodes and cathodes. There is no PHA or coincidence gating. Events on other
  channels and from node 0 are dropped. `pha` is stored for a future gate but is not used.
- **Validity.** The layout version, the `.dat` size, its mtime and the SHA-256 of its first MiB
  must all match. Any mismatch makes the file stale, and it is rebuilt. A cache that records no
  valid frame (which only an earlier uvcorr wrote, for a file that is not raw data) is never
  valid either, so its rebuild reports the `no valid frames` error. Copying the `.dat`
  without keeping its mtime (use `cp -p` or `rsync -t`) invalidates the cache.
- **Build.** The file is streamed in 2M-event parser batches into per-board buffers, and memory
  stays bounded (peak RSS 0.9 GB on the test file). The build writes a temporary file next to
  the cache (`data.dat.uv.<8 hex digits>.h5.tmp`) and renames it only on success, so an
  interrupted or failed build never leaves a cache that looks valid. Before parsing, the build
  checks the free space: about half the `.dat` size plus 64 MiB. It refuses to replace the
  `.dat` itself, a directory, any file that is not a UV cache, or a cache another process has
  open.
- **Test file.** 3.48 GB `.dat` → 1.26 GB cache (1.67 GB uncompressed). The build takes 20.6-23 s
  (the parse alone 11-16 s). Loading a median board (1.1M events) takes 16-25 ms, the largest
  (5.3M) about 0.1 s.
- **Results.** `uvcorr process` and the GUI's Fit All write `/results/current`, and the GUI's
  re-fits write the overrides. Every write builds the new group or table completely, then swaps
  it in, so an error leaves the previous results intact. This does not protect against a crash
  inside an HDF5 call, because HDF5 is not journaled; the events can always be rebuilt from the
  `.dat`, but the results cannot. HDF5 does not reclaim the space of replaced tables (2.5 MB
  each for the full system), so after many saves `h5repack` compacts the file. The test cache
  is 1.29 GB after the development runs, against 1.26 GB freshly built.
- **Rebuilding discards `/results`**, both the batch results and the overrides. The CLI warns and
  goes on (`build-cache --force`, or a stale cache). The GUI asks first. Export anything you
  need before rebuilding.
- **Concurrency.** No HDF5 handle is kept open between calls. When another process has the file
  open for writing, an access retries for 1 s and then fails with *"The UV cache … is in use by
  another process (HDF5 file lock); try again when it has finished"*. A busy cache is never
  mistaken for an invalid one, so it is never rebuilt because of the lock. The GUI refuses to
  re-fit or revert on results that another process has replaced since it loaded them, even
  when the operation would write nothing
  ([docs/GUI.md](docs/GUI.md#12-other-processes-busy-cache-and-results-changed-on-disk)).

---

## Outputs

Method, statuses and flags: [`docs/ALGORITHM.md`](docs/ALGORITHM.md) (sections 3-5).

### `<stem>.tec`

`<stem>` is the `.dat` stem. The layout is exactly that of RadialAnalysis: one block per `ok`
channel, in (node, board, rena, channel) order, tab-indented `key=value` lines, no header, no
blank lines, and the file ends with `}\n`. Doubles use C++ `ostream` default formatting (`%g`, 6
significant digits). From `out/data_20260910_120628.tec`:

```
channel{
	node=1
	board=15
	rena=0
	channel=4
	centerU=2028.39
	centerV=2035.75
	semiMajor=717.905
	semiMinor=691.088
	phi=-0.0465024
	radius=704.369
	radiusStd=8.1154
}
```

| Key | Value |
|-----|-------|
| `centerU`, `centerV` | Fitted ellipse centre (ADC) |
| `semiMajor`, `semiMinor` | a ≥ b (ADC) |
| `phi` | Angle of the semi-major axis from +U towards +V, in **(−π/2, π/2]** (rad) |
| `radius` | √(ab), the radius of the corrected circle |
| `radiusStd` | Gaussian σ of the corrected radii. When that fit fails (`gauss_fit_failed_post`, 15 channels on the test file) it is the sample standard deviation |

**φ convention.** RadialAnalysis writes φ in [−π/2, π): 27 % and 33 % of the blocks in its two
`.tec` files on disk lie outside (−π/2, π/2]. φ and φ + π describe the same ellipse and give the
same correction, and the downstream consumers accept either (plan D10). The correction a
consumer applies is RadialAnalysis' `applyEllipseCorrection`:

```
u = U − centerU,  v = V − centerV
u_r = ( u cosφ + v sinφ) · radius / semiMajor
v_r = (−u sinφ + v cosφ) · radius / semiMinor
U′ = u_r cosφ − v_r sinφ,  V′ = u_r sinφ + v_r cosφ     (centred on the origin)
```

**Channel set.** RadialAnalysis also fits R1 ch 4-6, when extractData wrote files for them;
uvcorr has no blocks for those channels. Channels whose pre-correction Gaussian fails are kept
(the C++ drops them). Channels below `min_events` get no block. On the test file uvcorr writes
6,115 blocks: 10 blocks the C++ does not write, and it lacks 240 C++ blocks, all channels with
6-99 events ([ALGORITHM 5.3, 6.5](docs/ALGORITHM.md#53-tec-d10)).

### `radial_summary.csv`

There is one row per active channel with events (6,557 on the test file), sorted by (node,
board, rena, channel). Failed channels also get a row, with `status` set and the unavailable
cells left empty. Floats use `%.6g`; the centre, the axes and φ use `%.9g`. This is a new schema
(D9): there are no shear columns, and it is not column-compatible with the RadialAnalysis CSV.

| Columns | Meaning |
|---------|---------|
| `node`, `board`, `rena`, `channel` | Channel address |
| `polarity`, `electrode` | `anode`/`cathode`; `ElectrodeMap` label (`A01`-`A39`, `C01`-`C08`) |
| `status`, `flags` | `ok`, `too_few_events` or `fit_failed`; `;`-separated warning flags (below) |
| `n_events`, `n_used`, `n_rejected` | Events; events in the final (robust) fit; `n_events − n_used` |
| `centerU`, `centerV`, `semiMajor`, `semiMinor`, `phi` | The ellipse, as in the `.tec` but with 9 digits |
| `axis_ratio`, `target_radius` | b/a; √(ab) |
| `pre_mean`, `pre_sigma`, `pre_fwhm`, `pre_chi2ndf` | Gaussian fit to the radii of the raw points about the **fitted** centre: μ, σ, 2.35482σ, Baker-Cousins deviance per degree of freedom. If the fit fails: the sample mean and std (FWHM from that std), χ²/ndf empty |
| `pre_skewness`, `pre_kurtosis`, `pre_robust_sigma` | Unbinned skewness, excess kurtosis and 1.4826·MAD of the same radii |
| `post_*` (the same 7) | The same for the corrected radii \|(U′, V′)\| |
| `rawfit_res_mean`, `rawfit_res_sigma` | Gaussian stats of the radial residual to the fitted ellipse |
| `corr_res_mean`, `corr_res_sigma` | Residual to the target circle; equal to `post_mean − target_radius` and `post_sigma` by construction |
| `phase_mean_gap_rad`, `phase_max_gap_rad`, `phase_max_gap_ns`, `phase_ks` | Sorted corrected phase atan2(V′, U′): mean gap 2π/N, largest gap (rad; ns at 490 kHz), KS distance from uniform. Only for N ≥ 8 |
| `timing_jitter_ns` | post σ / (2π · 490 kHz · post μ) |
| `options_source` | `batch` or `override` (a GUI re-fit with its own options) |

All metrics use every finite event of the channel, including the points the robust fit
rejected: they describe what a `.tec` consumer gets.

| Status | Meaning | Test file |
|--------|---------|----------:|
| `ok` | Ellipse fitted; all columns filled | 6,115 |
| `too_few_events` | Fewer than `min_events` events | 442 |
| `fit_failed` | No ellipse fits the points (fewer than 6 points, collinear points, degenerate conic) | 0 |

| Flag (status stays `ok`) | Condition | Test file |
|--------------------------|-----------|----------:|
| `high_rejection` | The robust fit rejected > 5 % of the events (usually a second, b/a ≈ 0.30 ellipse) | 128 |
| `extreme_axis_ratio` | b/a < 0.5 | 9 |
| `broad_ring` | Robust radial spread of the kept points > 0.05 √(ab): a blob or other non-ring | 106 |
| `center_outside_data` | Fitted centre outside the bounding box of the points | 0 |
| `robust_refit_failed` | A robust refit failed; the previous fit was kept | 0 |
| `geometric_refit_failed` | `--geometric` refinement failed; the algebraic fit was kept | 0 |
| `gauss_fit_failed_pre` | Pre-correction Gaussian not usable (resolved double-horned radii). **Informational** | 1,093 |
| `gauss_fit_failed_post` | Post-correction Gaussian not usable; sample stats reported (also `radiusStd`) | 15 |

1,299 channels carry a flag, and 241 carry one other than `gauss_fit_failed_pre`
([ALGORITHM 5.2, 7](docs/ALGORITHM.md#52-flags-zero-or-more-status-stays-ok)).

---

## GUI

```bash
uvcorr-gui [FILE] [--workers N] [-v | -vv]
```

`FILE` is a raw `.dat` (its cache is reused or built) or a `.uv.h5` cache. `--workers` sets the
Fit All worker processes (default `min(8, usable CPUs)`). A **control band** above the tabs
(Prev/Next, node/board/channel selectors, the channel's status, the fit options and the fit
buttons) sits over four tabs: **Scatter** (raw and corrected U-V side by side, with overlay and
density modes), **Radial** (pre and post radius histograms with their Gaussian fits), **Radius vs
angle** (mean radius ± σ in 5° bins) and **Board grid** (the 47 channels of a board). The
**Fit Inspector** dock lists every result field, the flags and the options used. The **System
Map** dock shows both panels at electrode granularity, coloured by status or by a metric. Opening
a cache with stored results shows them at once, without a refit. Fit Channel / Fit Board re-fit
with other options and store the results as per-channel overrides in the cache, and the exports
write the batch results with the overrides applied.

![uvcorr-gui with the full-system cache: Scatter tab, Fit Inspector and System Map](docs/images/main_window.png)

*The largest channel of the test file (N2 B16 R0 Ch27, 946,025 events): raw points with the
fitted ellipse and its axes (left), corrected points with the target circle √(ab) (right), 58
rejected points in grey.*

The GUI user guide is [docs/GUI.md](docs/GUI.md).

---

## Performance (test file)

`data_20260910_120628.dat`: Ge-68, full system, 3.48 GB, 208,267,950 events on active channels,
155 boards, 6,557 channels. Linux (WSL2), 24 cores, 15 GB RAM.

| Step | Time | Notes |
|------|-----:|-------|
| Parse only (`iter_event_arrays`, Cython) | 11-16 s | depends on the run |
| `uvcorr build-cache` | 20.6-23 s | 1.26 GB cache, peak RSS 0.9 GB |
| Cache validity check (reuse) | ~1 ms | size, mtime, hash of 1 MiB |
| `uvcorr process`, cache reused, 8 workers | 16-17 s | 15.8 s analysis / 15.9 s total in the run above; target was < 2 min |
| same, `--geometric` | 72 s | against 18.5 s for the default run in the same conditions |
| Analysis (`analyze_all`, the fitting step of `process` and Fit All) with 1 / 4 / 8 / 12 / 16 workers | 109 / 29 / 16 / 12 / 11 s | peak PSS of the process tree 0.5 / 1.1 / 1.8 / 2.4 / 3.0 GB, about 0.17 GB per worker. 2026-09-24, no other jobs on the machine, cache file in the page cache ([ALGORITHM 8](docs/ALGORITHM.md#8-performance-24-cores-wsl2)) |
| RadialAnalysis `--csv-only`, all 6,557 `.uvd` files | 1 min 54 s | single-threaded, `.uvd` export not included |
| GUI: open the cache with stored results | 0.3 s | without stored results every board is scanned for its channels (~2 s) |
| GUI: select a channel, Scatter drawn | about 0.15 s (80k-105k events) to 0.37-0.41 s (421k-946k) | offscreen, cache read from disk |
| GUI: select a channel, all tabs drawn | 0.16-0.58 s | same channels |
| GUI: Board grid of one board | 0.18-0.23 s | 2.2M-5.3M events |

The GUI's Fit All runs the same analysis as `process` in a `spawn` process pool.

---

## Cross-check against RadialAnalysis

The two scripts in `scripts/` compare uvcorr with the C++ on exactly the same events. ROOT is
needed only to run RadialAnalysis itself (`source ~/root-install/bin/thisroot.sh`). The full
procedure and the results are in [ALGORITHM 6](docs/ALGORITHM.md#6-cross-check-against-the-c-radialanalysis).

```bash
# 1. .uvd files from the cache (default boards: node 1 boards 15, 16; node 8 board 24)
venv/bin/python scripts/dump_uvd.py --cache data.dat.uv.h5 --out uvd
venv/bin/python scripts/dump_uvd.py data.dat --boards 1:15,3:25 --channels 0:4-28 --out uvd

# 2. RadialAnalysis on them
source ~/root-install/bin/thisroot.sh
~/DataProcessing/EllipseCorrection/RadialAnalysis --csv-only $PWD/uvd

# 3a. compare, fitting uvcorr on the fly (robust off; min_events 6 fits every channel the C++ fits)
venv/bin/python scripts/compare_radial.py uvd --cache data.dat.uv.h5 --min-events 6 \
    --decompose --out comparison.csv
# 3b. or against a uvcorr run, on a scratch cache: --no-robust would otherwise replace the
#     stored batch of data.dat.uv.h5 and drop its robust-off overrides
uvcorr process data.dat --no-robust --cache /tmp/data.uv.h5 --output-dir X
venv/bin/python scripts/compare_radial.py uvd --uvcorr-dir X
```

| Script | Options | Exit codes |
|--------|---------|------------|
| `dump_uvd.py [DAT]` | `--cache PATH` (instead of `<DAT>.uv.h5`, which must be valid; without either, the test file), `--boards N:B,...` (default `1:15,1:16,8:24`), `--channels R:C or R:LO-HI,...`, `--out DIR` (required; must be empty or new), `--overwrite` (delete the `.uvd` files already there) | 0; 1 nothing written; 2 bad arguments, missing or invalid cache, non-empty `--out` |
| `compare_radial.py CPP_DIR` | `--uvcorr-dir DIR` and/or `--cache PATH` (at least one), `--min-events N`, `--tec FILE` (if several `.tec`), `--rtol` (default 1e-6), `--strict-blobs`, `--decompose` (needs `--cache`), `--out CSV` | 0; 1 an ellipse parameter of a ring channel or an event count disagrees; 2 bad arguments or unreadable inputs |

The files are named `node{N}board{BB:02d}rena0{R}channel{CC:02d}.uvd`. Each starts with the
header line `U Values V Values`, has one `u v` line per event and ends with a newline. RadialAnalysis'
`loadUVD` discards line 1 and the value read at EOF, so this layout makes it read every event.
extractData's `.uvd` files have no header, and RadialAnalysis loses their first event.
`compare_radial.py` reports channels flagged `broad_ring` (pedestal blobs, where the C++
raw-coordinate fit is numerically wrong) on their own line and does not count them as failures
unless `--strict-blobs` is given. `--decompose` re-implements the C++ statistics in Python to
attribute each difference in σ to its cause.

---

## Development

The Makefile uses `./venv` when it exists (override with `make VENV=/path/to/venv ...`), otherwise
the `python3` on `PATH`.

| Target | Runs |
|--------|------|
| `make check` | format-check, lint, type-check, test: the phase gate, changes no files |
| `make dev-check` | format (rewrites files), lint, type-check, test |
| `make test` | `pytest -m "not realdata"` |
| `make test-fast` | `pytest -m "not slow and not realdata"` |
| `make test-realdata` | `pytest -m realdata` |
| `make test-cov` | tests with coverage (terminal, `htmlcov/`, `coverage.xml`) |
| `make lint` | `ruff check src/ tests/ scripts/` |
| `make format` / `make format-check` | `black src/ tests/ scripts/` (`--check`) |
| `make type-check` | `mypy src/ scripts/` |
| `make venv`, `make install`, `make install-dev`, `make cython-check` | environment (see [Install](#install)) |
| `make clean` | build artefacts and tool caches |

The tooling follows adc2kev: black and ruff with line length 100, mypy with adc2kev's strict flag
set, and pytest with `--strict-markers`.

| Marker | Tests |
|--------|-------|
| `gui` | pytest-qt tests of the widgets and the main window (`tests/gui/`). They run on Qt's `offscreen` platform unless `QT_QPA_PLATFORM` is set. `QSettings` is redirected to a per-test directory |
| `realdata` | `tests/test_realdata.py`: the cache of the test file (counts, validity, board loading) and the full-file census. Skipped when the `.dat` or its cache is absent |
| `slow` | the independent re-parse of the test file, the analysis of node 1 board 15 and the full-file census (all three also `realdata`) |
| `integration` | declared; no test uses it yet |

**Real test data**: `~/adc2kev-test-data/full-system/sources/ge/data_20260910_120628.dat` with
its sibling cache `data_20260910_120628.dat.uv.h5`. Build it first with `uvcorr build-cache`. Set
`UVCORR_TEST_DAT=/path/to/other.dat` to run the realdata tests on another acquisition; the exact
counts are then not checked. Do not modify or move the reference file.

Synthetic data for the tests is written by `tests/synthetic_dat.py` (valid AND-mode 0xC8 frames
with CRC) and the ring helpers in `tests/conftest.py`: clean, outlier, eccentric,
too-few-events and collinear channels.

### Project layout

```
src/uvcorr/
├── options.py         FitOptions (defaults, JSON), status and flag names
├── channels.py        active channels, electrode labels, polarity (adc2kev ElectrodeMap)
├── cache.py           UVCache: build, validity, board loading, /results and overrides
├── ellipse.py         fit_ellipse (algebraic, robust, geometric), correct / uncorrect, residuals
├── metrics.py         radial Gaussian (Poisson ML), unbinned stats, residual and phase metrics
├── analysis.py        ChannelResult (= CSV columns), analyze_channel / board / all (process pool)
├── cli.py             uvcorr build-cache | process
├── io/                tec.py, summary_csv.py, export.py, _atomic.py (atomic writes)
└── gui/               main.py (entry), window.py, session.py (data layer), threads.py,
                       controls.py, scatter.py, radial.py, angle.py, board_grid.py,
                       inspector.py, system_map.py + _system_map_model.py (adc2kev fork),
                       map_colors.py, _layout.py, _flow_layout.py, _tab_data.py, _contrast.py
scripts/               dump_uvd.py, compare_radial.py (C++ cross-check)
tests/                 unit, CLI and script tests; gui/ (pytest-qt); test_realdata.py
docs/                  ALGORITHM.md, GUI.md, images/, planning/UV_ELLIPSE_CORRECTION_PLAN.md
```

## Documentation

- [docs/ALGORITHM.md](docs/ALGORITHM.md): the method, its differences from RadialAnalysis, the C++
  cross-check, the census of the test file and the retuning of the defaults.
- [docs/GUI.md](docs/GUI.md): the GUI user guide.
- [docs/planning/UV_ELLIPSE_CORRECTION_PLAN.md](docs/planning/UV_ELLIPSE_CORRECTION_PLAN.md): the
  design decisions (D1-D11), formats and phases.
