# UV Ellipse Correction from Raw Data: Plan

**Status:** Phases 0-5 done (2026-09-24): skeleton; UV cache and `uvcorr build-cache`; ellipse fit and metrics; analysis, `.tec`/CSV export, `uvcorr process`, C++ cross-check and census (see `docs/ALGORITHM.md`); GUI with System Map, Scatter, Radial, Radius vs angle and Board grid tabs, Fit Inspector, channel/board re-fits stored as overrides, and export (screenshots in `docs/images/`). Next: phase 6 (docs).
**Repository:** `/home/swuupii/uv-ellipse-correction` (git, branch `main`)
**Package name:** `uvcorr` (confirmed). Console scripts: `uvcorr` (CLI), `uvcorr-gui`.

This document is self-contained. An implementer with no prior context should be able to carry out
the whole plan from it. Read sections 1-4 first, then work through the phases in section 12.
Section 13 lists every external file referenced.

---

## 1. Problem

The detector is a dual-panel CZT PET system read out by RENA-3 ASICs. Each channel records a
fine-timing phase as a quadrature pair (U, V). Plotted as a scatter, a channel's (U, V) points should
fall on a circle. In practice they fall on a tilted, offset **ellipse**. The ellipse correction fits
a least-squares ellipse to each channel's points and maps it to a circle.

The existing tool, **RadialAnalysis** (C++/ROOT, `~/DataProcessing/EllipseCorrection`), does this. Its
input is a folder of per-channel `.uvd` text files, which a separate program (`extractData -t`)
produces from an unpacked text dump of the raw data. We want to:

1. Run the ellipse correction **directly on a raw `.dat` acquisition file**.
2. Provide a **GUI** that shows each channel's before/after U-V scatter and supporting diagnostics,
   lets the user re-fit channels with different options, and exports the results.

The design follows **adc2kev-python** (`~/adc2kev-python`): the user's Python rewrite of the energy
calibration software, with a PyQt6/pyqtgraph GUI. It also follows the companion viewer **specview**
(`~/adc2kev-python/specview`), which is a separate package built on the adc2kev library. That is the
same relationship `uvcorr` will have.

**Test data:** `~/adc2kev-test-data/full-system/sources/ge/data_20260910_120628.dat` (3.48 GB, Ge-68
source, full system). Do not modify or move it. Write caches for it into a scratch or output
location, or next to it only if the directory is writable. By default the cache goes next to the
`.dat` file (section 6.1).

## 2. Decisions (all confirmed by the user)

| # | Decision |
|---|----------|
| D1 | Port the ellipse fit and correction plus **all** RadialAnalysis metrics: pre/post radial stats, residuals, and phase gaps/KS. **Drop** the legacy shear/ψ method (`findCenter`, `computePsi`, `quadFinder`). |
| D2 | Fit on **all events of each active channel**, anodes and cathodes alike. Active means RENA 0 channels 4-28 and RENA 1 channels 7-28. Events on other channels are dropped. No PHA or coincidence gating by default. |
| D3 | Deliverables are **`<name>.tec` + `radial_summary.csv`**. **Out of scope:** exporting corrected per-event U/V, rewriting a corrected `.dat`, and batch PDF/PNG figure export. |
| D4 | A **separate package** in this repo that **depends on adc2kev** (installed editable from `~/adc2kev-python`). Reuse its parser and geometry; don't copy them. |
| D5 | The GUI views are:<br>- Before/after scatter: side-by-side linked panels, an overlay toggle, and a density-heatmap toggle<br>- Radial histograms<br>- Radius vs angle<br>- System Map<br>- Board grid overview |
| D6 | A **sibling HDF5 UV cache** (`<name>.uv.h5`), built once and loaded per board on demand. |
| D7 | The GUI **views results and can re-fit** a channel, board or everything with adjusted options, then exports `.tec`/`.csv`. No manual point curation or accept/reject flags. |
| D8 | **Best method, no parity requirement.** The numbers may differ from the C++ where the method improves on it. The C++ binary is a sanity cross-check, not a golden reference. |
| D9 | **New CSV schema** (section 6.2). There are no shear columns and it isn't column-compatible with the old `radial_summary.csv`. |
| D10 | The **`.tec` format stays byte-for-byte in the RadialAnalysis layout** (section 6.3). The user confirmed that downstream consumers need only the existing fields and accept φ in (−π/2, π/2]. |
| D11 | Initialise a **git repo** in this directory in phase 0. |

## 3. Environment

| Item | Value |
|------|-------|
| OS | Linux (WSL2), 24 cores, 15 GB RAM |
| Python | 3.10 (`python3.10`) |
| adc2kev | `~/adc2kev-python`, version 2.2.7, installed editable in `~/adc2kev-python/venv`. Its Cython parser extension is built (`CYTHON_AVAILABLE` and `CYTHON_ARRAYS_AVAILABLE` are both `True`). |
| adc2kev pins | `numpy>=1.24,<2.0`, `scipy`, `h5py`, `PyQt6`, `pyqtgraph`. The adc2kev venv has numpy 1.26.4, scipy 1.15.3, h5py 3.15.1, PyQt6 6.10.2, pyqtgraph 0.14.0, pytest 9.0.2, pytest-qt 4.5.0, mypy 1.19.1, ruff 0.14.14, black 25.12.0, Cython 3.2.8. |
| ROOT | `~/root-install` (`source ~/root-install/bin/thisroot.sh`). Only needed for the C++ cross-check. |
| RadialAnalysis binary | `~/DataProcessing/EllipseCorrection/RadialAnalysis`, already built and linking fine |

**Venv for this project (phase 0):** create `./venv` with `python3.10 -m venv venv`. Then run
`pip install --upgrade pip setuptools wheel Cython` and
`pip install -e ~/adc2kev-python` (this builds the Cython extension). Finally run `pip install -e ".[dev]"`.
Verify with:

```bash
python -c "import adc2kev.parser.packet_parser as p; assert p.CYTHON_AVAILABLE and p.CYTHON_ARRAYS_AVAILABLE"
```

If the arrays extension is missing, run `python setup.py build_ext --inplace` in `~/adc2kev-python`.
Without the extension, parsing falls back to pure Python and becomes about 100x slower. Alternatively,
develop inside `~/adc2kev-python/venv` if installing into a fresh venv causes trouble.

## 4. Reference: the C++ RadialAnalysis algorithm

Source files: `~/DataProcessing/EllipseCorrection/uvd_common.h` (522 lines, the algorithms) and
`main_radial.cpp` (2400 lines, mostly ROOT plotting; the per-channel pipeline is in `main()` at lines
1835-2325).

### 4.1 Per-channel pipeline (C++)

1. Load (U, V) from `.uvd`.
2. `findCenter` (histogram peaks). Shift to the origin and compute pre radii. Fit a Gaussian to the pre radial histogram (*we replace the centre, see 5.3*).
3. Shear ψ correction (*dropped*).
4. `fitEllipse` on the **raw, unshifted** points, then `applyEllipseCorrection`, then compute the corrected radii and fit a Gaussian.
5. Residuals and phase metrics on the corrected data (only if N ≥ 8).
6. Write a CSV row, plus a `.tec` block if the ellipse succeeded.

### 4.2 `fitEllipse` (Fitzgibbon 1999 in the Halir-Flusser reduced form)

Conic: A x² + B xy + C y² + D x + E y + F = 0. Constraint: 4AC − B² = 1. Needs N ≥ 6.

```
D1 = [x², xy, y²] (N×3),  D2 = [x, y, 1] (N×3)
S1 = D1ᵀD1, S2 = D1ᵀD2, S3 = D2ᵀD2
T  = −S3⁻¹ S2ᵀ                      (C++: explicit inverse, fails if |det S3| < 1e-12)
M  = C1⁻¹ (S1 + S2 T),  C1⁻¹ = [[0,0,½],[0,−1,0],[½,0,0]]
a1 = eigenvector of M with 4·a1[0]·a1[2] − a1[1]² > 0;  a2 = T a1
(A,B,C) = a1, (D,E,F) = a2
```

Conic to geometric parameters:

```
disc = B² − 4AC                     (fails if |disc| < 1e-12)
cx = (2CD − BE)/disc,  cy = (2AE − BD)/disc
φ  = ½·atan2(B, A − C)
Fc = A cx² + B cx cy + C cy² + D cx + E cy + F
A' = A cos²φ + B cosφ sinφ + C sin²φ,  C' = A sin²φ − B cosφ sinφ + C cos²φ
a = √(−Fc/A'),  b = √(−Fc/C')         (fails if either argument is ≤ 0)
if a < b: swap and add π/2 to φ
```

The C++ then normalises φ with loops that step by π but compare against ±π, so φ can end up anywhere
in (−π, π]. This is harmless because an ellipse is symmetric under a π rotation, but we canonicalise
instead (5.1).

### 4.3 `applyEllipseCorrection` (this must also be what the `.tec` consumers apply)

```
u = U − cx, v = V − cy
u_r =  u cosφ + v sinφ,   v_r = −u sinφ + v cosφ
R = √(ab);  u_r *= R/a;  v_r *= R/b
U' = u_r cosφ − v_r sinφ,  V' = u_r sinφ + v_r cosφ      (output centred on the origin)
```

### 4.4 Metrics (C++)

- **Radial Gaussian** (`fitRadialHist`): a 200-bin `TH1D` over [min R, max R], fitted with
  `Fit("gaus","QS")` over histogram mean ± 3·RMS. The outputs are μ, σ, FWHM = 2.35482σ, χ²/ndf, and the
  histogram's binned `GetSkewness`/`GetKurtosis` (ROOT kurtosis is the excess kurtosis). A failed fit
  skips the whole channel (pre) or marks it not corrected (post).
- **Raw residual to the ellipse:** in the ellipse-aligned frame (u_a, v_a) about (cx, cy),
  θ = atan2(v_a, u_a), r_ell = ab / √((b cosθ)² + (a sinθ)²), and res = √(u_a² + v_a²) − r_ell.
- **Corrected residual:** res = √(U'² + V'²) − √(ab).
- **Residual stats:** 150 bins over mean ± 6·std, then a Gaussian fit; fall back to the sample mean/std.
- **Phase:** p = atan2(V', U') wrapped to [0, 2π) and sorted. The gaps are the consecutive differences plus the
  wrap-around gap ph[0] + 2π − ph[N−1].
  - mean gap = 2π/N; max gap; max gap in ns = max_gap / (2π · 490 kHz) · 1e9.
  - KS D = max over j of max((j+1)/N − F_j, F_j − j/N), with F_j = ph[j]/(2π).
- **System-summary jitter proxy:** σ_R / (2π · 490 kHz · R̄) in ns.

### 4.5 Channel loops and cathodes (C++)

The loops cover nodes 1-10, boards 0-30, RENA 0-1, and channels 4-28 (a missing file is skipped). In
`main_radial.cpp:279`, a channel is a cathode on even boards for ch 25-28 (both RENAs), and on odd boards for
RENA 0 ch 4-7 or RENA 1 ch 7-10. In `uvcorr`, take the polarity from the parsed event (`polarity`:
0 = cathode, 1 = anode) or from `ElectrodeMap.is_cathode`.

## 5. Algorithms in `uvcorr` (best method, D8)

### 5.1 Ellipse fit (`uvcorr.ellipse`)

1. **Normalise:** x = (U − mean U)/s and y = (V − mean V)/s, where s is the RMS distance from the mean.
   On raw coordinates (U ≈ 2000, R ≈ 600) the scatter matrix mixes values around 1e13 with values around N,
   which makes it badly conditioned.
2. Use the same reduced-eigenproblem fit as in 4.2, but compute T with `np.linalg.solve(S3, S2.T)`
   instead of an inverse. Pick the eigenvector satisfying the constraint (if several do, take the one
   with the smallest positive eigenvalue). Undo the normalisation on the conic coefficients, then convert to (cx, cy, a, b, φ)
   with the formulas in 4.2.
3. **Robust iteration (default on):**
   1. Compute each point's radial residual to the current ellipse (formula in 4.4).
   2. Keep the points with |res − median| ≤ k · 1.4826 · MAD (default k = 4.0), and refit.
   3. Stop when the kept set no longer changes or after `max_iter` iterations (default 5).
   4. Record `n_used` and `n_rejected`.
   5. If the refit fails, keep the previous successful fit and set a flag.
4. **Geometric refinement (option, default off):** `scipy.optimize.least_squares` on the
   parameters (cx, cy, a, b, φ), minimising the radial residuals of the kept points. It starts
   from the algebraic solution.
5. **Canonical form:** a ≥ b, and φ wrapped into (−π/2, π/2].
6. **Status:** `ok`, `too_few_events` (n < `min_events`, default 100), or `fit_failed` (no
   ellipse eigenvector, degenerate disc, or non-positive a²/b²). **Warning flags** (the status stays `ok`):
   `high_rejection` (rejected fraction > 5 %), `extreme_axis_ratio` (b/a < 0.5), `center_outside_data`
   (centre outside the data's bounding box), `robust_refit_failed`, and `gauss_fit_failed_pre/post`.
   All thresholds live in `FitOptions`, and the census in phase 3 retunes them.

### 5.2 Correction

The transform in 4.3, vectorised as `correct(u, v, params) -> (u', v')`, plus an inverse
`uncorrect` used for tests.

### 5.3 Metrics (`uvcorr.metrics`)

- **Pre radii** are measured about the **fitted ellipse centre** (not `findCenter`), so pre vs post
  isolates the ellipticity.
- **Radial Gaussian:**
  - Histogram the radii inside the [0.5, 99.5] percentiles with Freedman-Diaconis bins
    (clamped to 50-400).
  - Fit a Gaussian with `scipy.optimize.curve_fit` (Poisson weights, empty bins given σ = 1). Start the
    fit range at median ± 3·(1.4826·MAD), then iterate it to μ ± 2σ, up to 3 times.
  - Report μ, σ, FWHM = 2.35482σ, and χ²/ndf. If the fit fails, flag it and report the sample
    statistics.
- **Also report:** the unbinned mean/std, robust σ (1.4826·MAD), and unbinned skewness plus excess kurtosis
  (`scipy.stats.skew`, `scipy.stats.kurtosis(fisher=True)`).
- **Residual stats and phase metrics:** exactly the definitions in 4.4. Residual stats use the same
  Gaussian routine as the radial fit. Phase metrics are computed only if N ≥ 8.
- `timing_jitter_ns` = post σ / (2π · `phase_ref_freq_hz` · post μ) · 1e9, with
  `phase_ref_freq_hz` = 490e3.

## 6. Data formats

### 6.1 UV cache `<name>.uv.h5` (default path: `<dat path>.uv.h5`, e.g. `data_…_120628.dat.uv.h5`; overridable with `--cache`)

```
/metadata                      attrs: uv_cache_version="1.0.0", uvcorr_version, source_path, source_size,
                               source_mtime, source_hash (SHA-256 of the first 1 MiB, same as adc2kev's
                               CalibrationCache._compute_file_hash — reimplement, it is private),
                               parser_frames, parser_events, parser_dropped, n_events_kept,
                               n_events_inactive, build_seconds, created_at
/events/node_{N}/board_{B}/    datasets, file order, chunked (e.g. 262144) + gzip/lzf:
    rena (int8), channel (int8), u (int16), v (int16), pha (int16)
    attr n_events
/results/current/              the last analysis run
    attrs: options_json (FitOptions), created_at, uvcorr_version
    table                      structured/compound dataset: one row per channel (all CSV fields)
/results/current/overrides/    per-channel re-fits from the GUI: table with the same dtype plus options_json per row
```

- **Build:** stream `PacketParser(dat).iter_event_arrays(batch_events=2_000_000)`. Drop inactive channels
  and node 0, then group each batch by (node, board) (use `np.argsort` on a board key, or boolean
  masks) and append to per-board buffers. Flush a buffer to resizable HDF5 datasets when it exceeds a
  threshold (e.g. 4M events), and flush everything at the end. Memory stays bounded. Report progress as
  `batch.bytes_read / file size`. Write to `<path>.tmp` and rename on success, so an interrupted build
  never leaves a cache that looks valid.
- **Validity** (pattern: `adc2kev/cache/diagnostic_cache.py:131` `is_valid_for`): the version, size,
  mtime and hash must all match. Otherwise rebuild.
- **Load:** `load_board(node, board)` returns the arrays. `channel_data(node, board, rena, channel)` selects
  with a mask. A board has about 1M events, so this is trivial.
- Keep `pha` so a PHA gate can be added later without rebuilding (it isn't used now, D2).
- Expected size: about 208M events × 8 B ≈ 1.7 GB uncompressed; measure it in phase 1.

### 6.2 `radial_summary.csv` (new schema, D9)

There is one row per active channel with any events, sorted by (node, board, rena, channel). Failed channels still get a
row: `status` is set and the unavailable metrics are left empty. Floats are written with `%.6g`, or more
digits for centre, axes and φ (`%.9g`). Columns in order:

```
node,board,rena,channel,polarity,electrode,status,flags,n_events,n_used,n_rejected,
centerU,centerV,semiMajor,semiMinor,phi,axis_ratio,target_radius,
pre_mean,pre_sigma,pre_fwhm,pre_chi2ndf,pre_skewness,pre_kurtosis,pre_robust_sigma,
post_mean,post_sigma,post_fwhm,post_chi2ndf,post_skewness,post_kurtosis,post_robust_sigma,
rawfit_res_mean,rawfit_res_sigma,corr_res_mean,corr_res_sigma,
phase_mean_gap_rad,phase_max_gap_rad,phase_max_gap_ns,phase_ks,timing_jitter_ns,options_source
```

- `polarity` is `anode` or `cathode`.
- `electrode` is the `ElectrodeMap` label (e.g. `A17`, `C03`).
- `flags` is semicolon-separated.
- `axis_ratio` = b/a.
- `options_source` is `batch` or `override`.

### 6.3 `<name>.tec` (unchanged format, D10)

Written only for `status == ok` channels, in the same order as the CSV. `<name>` is the `.dat`
stem. The layout is exact: tab indentation, one `key=value` per line, and values written with
default-precision C++-style `ostream` formatting (6 significant digits, `%g`).

```
channel{
	node=<int>
	board=<int>
	rena=<int>
	channel=<int>
	centerU=<double>
	centerV=<double>
	semiMajor=<double>
	semiMinor=<double>
	phi=<double>
	radius=<double>        # sqrt(semiMajor*semiMinor)
	radiusStd=<double>     # post-correction Gaussian sigma
}
```

Write 6 significant digits to match the existing files exactly (for example, `centerU=2022.36` in the old
`.tec` files). Also implement `read_tec` so a round-trip test can check the result.

## 7. Package layout and APIs

```
pyproject.toml         setuptools, requires-python >=3.10. Deps: adc2kev>=2.2.7, numpy>=1.24,<2,
                       scipy, h5py, PyQt6, pyqtgraph. dev extras: pytest, pytest-cov, pytest-qt,
                       black, ruff, mypy. Copy adc2kev's black/ruff/mypy/pytest config (line length 100,
                       mypy strict, markers slow/integration/gui/realdata).
src/uvcorr/
├── __init__.py        __version__
├── channels.py        is_active_channel(rena, ch); ACTIVE_CHANNELS; electrode_label(); is_cathode()
│                      (wraps adc2kev.tools.electrode_map.ElectrodeMap.default() and adc2kev.tools.geometry)
├── cache.py           UVCache(path): build_from_dat(dat, progress_cb, stop_flag), is_valid_for(dat),
│                      boards() -> list[(node, board)], load_board(node, board) -> BoardUV,
│                      channel_data(...) -> (u, v), channels(node, board) -> list[(rena, ch, n)],
│                      save_results(results, options), load_results(), save_override(...)
├── ellipse.py         EllipseParams(cx, cy, a, b, phi) [frozen dataclass; target_radius property],
│                      fit_ellipse_direct(u, v) -> EllipseParams | None,
│                      fit_ellipse(u, v, options) -> EllipseFit(params, mask_used, n_iter, flags),
│                      correct(u, v, p), uncorrect(u, v, p), residual_to_ellipse(u, v, p)
├── metrics.py         GaussStats(mean, sigma, fwhm, chi2ndf, ok), RadialStats(... + skew, kurt,
│                      robust_sigma, sample_mean, sample_std), radial_stats(r), residual_stats(res),
│                      PhaseStats, phase_stats(u_corr, v_corr, ref_freq_hz)
├── analysis.py        FitOptions (frozen dataclass: min_events=100, robust=True, clip_k=4.0,
│                      max_iter=5, geometric=False, phase_ref_freq_hz=490e3, warn thresholds; to_json/from_json),
│                      ChannelKey-like (node, board, rena, channel), ChannelResult (all CSV fields),
│                      analyze_channel(key, u, v, options) -> ChannelResult,
│                      analyze_board(cache_path, node, board, options) -> list[ChannelResult],
│                      analyze_all(cache, options, workers, progress_cb, stop_flag) (ProcessPoolExecutor,
│                      one task per board; workers default min(8, cpu_count))
├── io/tec.py          write_tec(path, results), read_tec(path) -> dict[key, EllipseParams + radius, radiusStd]
├── io/summary_csv.py  write_summary_csv(path, results)
├── cli.py             argparse. Subcommands: build-cache, process (section 8)
└── gui/               section 9
tests/                 test_ellipse.py, test_metrics.py, test_cache.py, test_io.py, test_analysis.py,
                       test_cli.py, gui/test_*.py (pytest-qt, marker gui), test_realdata.py (marker realdata,
                       skips if the test .dat or its cache is absent)
scripts/               dump_uvd.py and compare_radial.py (the C++ cross-check, section 10.3)
docs/                  planning/ (this file), README, ALGORITHM.md (differences from the C++), GUI.md
```

For synthetic `.dat` test files, write valid AND-mode (0xC8) frames. The frame layout and CRC are in
`~/adc2kev-python/docs/technical/PACKET_FORMAT_SPEC.md`. Check whether adc2kev's tests already have a
frame-builder helper you can reuse (`grep -rn "0xC8" ~/adc2kev-python/tests`).

## 8. CLI

```bash
uvcorr build-cache data.dat [--cache PATH] [--force]
uvcorr process data.dat --output-dir out/ [--cache PATH] [--workers N]
        [--min-events 100] [--no-robust] [--clip-k 4] [--max-iter 5] [--geometric]
# builds or reuses the cache, writes out/<stem>.tec and out/radial_summary.csv,
# stores the results in the cache (/results/current), and prints a status census and timing
```

Target: the full test file processes in **under 2 minutes** from an existing cache.

## 9. GUI (`uvcorr-gui [file]`)

Base it on the adc2kev GUI patterns: a `QMainWindow`, docks with object names, `QSettings` persistence,
a View menu with dock toggles and Reset Layout, and `QThread` workers with `progress`/`finished`/`error`/`stopped`
signals plus Stop (see `~/adc2kev-python/src/adc2kev/gui/main.py`, classes `ProcessingThread` through
`CacheBuildThread`, lines 64-480, and `docs/implementation/GUI_LAYOUT.md`).

| Area | Content |
|------|---------|
| Menu / toolbar | **File:** Open Raw (.dat; builds or reuses the cache in `CacheBuildThread` with a progress bar), Open Cache, Export .tec, Export CSV, Exit. **Process:** Fit All. **Toolbar:** Open, Fit All, Stop. **View:** dock toggles, Reset Layout. |
| Control band | Channel label (node/board/RENA/channel/electrode) and status; fit options (Robust checkbox, clip k, max iter, Geometric refine, min events); Fit Channel / Fit Board / Fit All buttons. A fit run with non-default options on a channel or board is saved as an override (6.1). |
| Central tabs | **Scatter:** two linked panels with locked 1:1 aspect ratio. The left panel shows raw (U, V) with the fitted ellipse, centre marker and axes drawn, and rejected points in a muted colour. The right panel shows the corrected (U', V') data with the target circle r = √(ab). An *Overlay* toggle shows raw-minus-centre and corrected points on one centred axis pair. A *Density* toggle shows 2D histograms (`pg.ImageItem`, log colour) instead of points. A point cap (spin box, default 50k) uses a deterministic random subsample and reports "showing N of M"; see specview `gui/main.py:1054`.<br>**Radial:** pre and post radius histograms with the Gaussian curve and a μ/σ/FWHM/χ² text box.<br>**Radius vs angle:** pre and post mean ± σ in 72 bins of 5°.<br>**Board grid:** the selected board's 47 channels as small plots (layout: RENA 0 then RENA 1 by channel, or physical strip order). It has a before/after toggle and common axes; cathodes are outlined, and a click opens the channel. |
| Right dock | **Fit Inspector:** all ChannelResult fields, the flags, and the options used (batch or override). |
| Bottom dock | **System Map:** a fork of adc2kev's `gui/system_map.py` (1301 lines) and `gui/_system_map_model.py` (505 lines), documented in `docs/implementation/SYSTEM_MAP.md`. Generalise `MapCell` so it carries a fill `QColor` and a tooltip instead of the calibration-only `CellStatus` enum. The colour mode is either *Status* (ok / flagged / failed / too few / no data) or a *Metric* (post σ, jitter ns, KS D, axis ratio, rejected fraction) using a sequential colormap with a colour bar and percentile-clipped limits. Keep the click, keyboard stepping, and board strip behaviour. |
| Persistence | On open, load `/results/current` and its overrides if they are present, so no refit is needed. |

The GUI needs a data/session layer (like specview's `session.py`) that owns the cache, the current results and
the overrides. The widgets talk only to that layer.

## 10. Validation

1. **Cache:** on the test file, the sum of `n_events` over boards equals the number of parsed events on
   active channels. The totals measured on 2026-09-24 were 208,492,906 events, 224,956 of them on inactive
   channels, so 208,267,950 active is expected (verify; node 0 or other anomalies would shift this). A
   synthetic `.dat` file round-trips exactly. A stale cache is detected when the size or mtime changes.
   - **Result (phase 1, 2026-09-24):** 208,267,950 kept and 224,956 inactive (R0 ch 1-3, R1 ch 6), 0 on
     node 0, confirmed by an independent re-parse that also matched all 155 boards event for event.
2. **Synthetic ellipses:**
   - Known cx, cy, a, b, φ (including φ near ±π/2 and near-circles with b/a > 0.99), with Gaussian radial
     noise and 0-10 % injected uniform outliers. The recovered parameters match within tolerance, and the
     robust fit beats the plain fit when outliers are present.
   - `correct` followed by `uncorrect` gives back the original points. The corrected points have a mean radius of about √(ab).
   - Fewer than 6 points, collinear points, and a hyperbola-like arc return `fit_failed` without raising.
3. **Cross-check against the C++ (`scripts/`):**
   - `dump_uvd.py` writes `.uvd` files from the cache for a few boards (e.g. node 1 boards 15 and 16, plus
     one low-count board).
     - File name: `node{N}board{BB:02d}rena0{R}channel{CC:02d}.uvd`, e.g. `node1board15rena00channel04.uvd`.
     - Contents: a first header line `U Values V Values`, then one `u v` line per event, with a trailing newline.
   - Run `source ~/root-install/bin/thisroot.sh && ~/DataProcessing/EllipseCorrection/RadialAnalysis --csv-only <dir>`.
     The output goes to `<dir>/RadialAnalysis_output/radial_summary.csv` and `<dirname>.tec`.
   - `compare_radial.py` compares that output with `uvcorr process --no-robust` on the same channels. The ellipse
     parameters should agree closely (the centre and axes to about 1e-6 relative, and φ modulo π). Explain any
     differences in the radial stats by the method changes in 5.3, and record them in `docs/ALGORITHM.md`.
4. **Full test file:** record the cache build time and size, the processing time, and the status/flag census.
   Also look at the distributions of pre vs post σ and of the rejected fraction per channel class
   (anode/cathode, count decile). Retune the defaults in `FitOptions` if the census says so, and record the
   evidence in `docs/ALGORITHM.md`.
   - **Result (phase 3):** `uvcorr process` takes ≈ 17 s from the cache (8 workers, peak PSS ≈ 1.9 GB). Status:
     6,115 ok, 442 too_few_events, 0 fit_failed. Flags: gauss_fit_failed_pre 1,093 (informational: resolved
     double-horned pre radii), high_rejection 128, broad_ring 106, gauss_fit_failed_post 15,
     extreme_axis_ratio 9. Only `broad_ring_frac` was retuned (0.1 → 0.05). A second, concentric U/V
     population (b/a ≈ 0.30, φ ≈ −44°) appears in ≈ 900 channels; the robust fit rejects it.
5. **GUI:** pytest-qt smoke tests on a small synthetic cache (open, select a channel through the map,
   switch tabs, re-fit a channel with robust off, export, and check that the files exist). Then a
   manual spot check on the test file, with screenshots in `docs/images/`.

## 11. Measured facts about the test file (2026-09-24, adc2kev Cython parser)

| Quantity | Value |
|----------|-------|
| Size / parse time | 3,478,589,791 B / 15.8 s (`iter_event_arrays(batch_events=2_000_000)`) |
| Frames / events / dropped | 106,063,959 / 208,492,906 / 23 |
| Nodes / boards | 1-10 / 15-30; **155 boards with data** (not 160). No events on (node, board) (2,18), (6,24), (7,15), (8,22), (9,20); 1-5 events on (8,16), (8,21), (9,18), (7,20), (4,20), (6,20). Median board 1.08M events, max (3,25) 5.31M |
| Channels with data | 6,609: 6,557 active, 52 inactive (R0 ch 1-3, R1 ch 6, 224,956 events) |
| Events per channel | min 1, p5 29, p10 451, median 17,646, p90 51,550, max 946,025; 442 active channels have < 100 |
| U / V (5M-event sample) | U 1266-3187, V 1312-2752, median about 2043 for both; no U = V = 0 (AND mode) |
| UV cache (phase 1) | lzf + shuffle, 65,536-event chunks: build 22-23 s wall (parse alone ≈ 11-12 s on this run), peak RSS ≈ 0.9 GB, **1.26 GB** on disk (1.67 GB uncompressed). Alternatives measured: none 15.8 s / 1.71 GB, gzip-1 + shuffle 35 s / 0.98 GB (≈1.5× slower reads). Reuse check 1.4 ms; `load_board` 16-25 ms for a median board, ≈ 0.1 s for the largest |

The `EventBatch` columns (from `adc2kev.parser.measurement_event`) are:
- `trigger_num` int64, `node_num` uint8, `board_num` uint8, `rena_num` uint8, `channel_num` uint8
- `polarity` uint8 (0 = cathode, 1 = anode), `timestamp` int64, `pha` int16, `u` int16, `v` int16
- `n_events`, `bytes_read`

Parser totals come from `parser.get_statistics()`.

## 12. Phases

Commit at the end of every phase (conventional message, on a branch if preferred). Each phase must
leave `ruff check`, `black --check`, `mypy src/` and `pytest -m "not realdata"` passing.

| Phase | Scope | Done when |
|-------|-------|-----------|
| 0 | `git init`, `.gitignore` (venv, caches, `*.h5`, build, htmlcov, `.scratch`), venv setup (section 3), `pyproject.toml`, package skeleton, README stub, Makefile targets like adc2kev's (`format`, `lint`, `type-check`, `test`, `dev-check`) | Install works; the Cython check passes; the tooling passes on the skeleton |
| 1 | `channels.py`, `cache.py`, `uvcorr build-cache` | 10.1 passes; build time and size on the test file recorded here |
| 2 | `ellipse.py`, `metrics.py` | 10.2 passes |
| 3 | `analysis.py`, `io/`, `uvcorr process`, `scripts/` cross-check, census | 10.3 and 10.4 done; `out/` for the test file produced; `docs/ALGORITHM.md` written |
| 4 | GUI shell: session layer, main window, open raw/cache with a threaded build, System Map fork, Scatter tab, Fit Inspector, Fit All thread | Any channel's before/after scatter can be browsed from the map |
| 5 | Radial, Radius vs angle and Board grid tabs, re-fit controls, overrides, persistence, export | 10.5 done |
| 6 | README (install, CLI, GUI), `docs/GUI.md` with screenshots, update this plan's status | Reviewed by the user |

## 13. Risks and open items

- **Robust thresholds:** the defaults are guesses until the phase 3 census. Cathodes or low-count channels
  may need different handling.
- **Plotting performance:** channels have up to about 1M points. pyqtgraph scatter plots slow down above about
  100k points, so always apply the point cap or use density mode.
- **System Map fork:** about 1.8k lines. Keep the structure close to adc2kev so fixes can be carried across,
  and note the source version (adc2kev 2.2.7) in the module docstring.
- **adc2kev coupling:** use only public APIs (`PacketParser`, `EventBatch`, `ElectrodeMap`,
  `tools.geometry`). Reimplement private helpers such as `_compute_file_hash`.
- **Disk space:** the cache for the test file is about 1-2 GB. Check free space before building and
  fail with a clear message.

## 14. Reference files

| Path | Why |
|------|-----|
| `~/DataProcessing/EllipseCorrection/uvd_common.h` | `fitEllipse` (line 374), `applyEllipseCorrection` (line 495) |
| `~/DataProcessing/EllipseCorrection/main_radial.cpp` | `fitRadialHist` (124), `isCathode` (279), residual and phase metrics (2159-2238), CSV/.tec writing (1797, 2286), channel loop (1835) |
| `~/DataProcessing/EllipseCorrection/ellipse_correction_procedure.md` | `.tec` format and how consumers apply it |
| `~/DataProcessing/EllipseCorrection/README.md`, `CLAUDE.md` | RadialAnalysis options and build |
| `~/DataProcessing/extractData/data_kimia_edit/main.cpp` | Current `.uvd` producer (`exportUVD`, line 131: all events, no filtering) |
| `~/adc2kev-python/README.md`, `docs/INDEX.md` | adc2kev overview and doc index |
| `~/adc2kev-python/docs/technical/PACKET_FORMAT_SPEC.md`, `HARDWARE_CONSTRAINTS.md` | Packet format, valid channel ranges |
| `~/adc2kev-python/src/adc2kev/parser/packet_parser.py` | `PacketParser.iter_event_arrays` (line 501) |
| `~/adc2kev-python/src/adc2kev/parser/measurement_event.py` | `EventBatch` |
| `~/adc2kev-python/src/adc2kev/cache/diagnostic_cache.py` | Single-source cache and validity pattern |
| `~/adc2kev-python/src/adc2kev/tools/electrode_map.py`, `tools/geometry.py` | Electrode labels, cathode test, panels, active boards 15-30 |
| `~/adc2kev-python/src/adc2kev/gui/main.py`, `gui/system_map.py`, `gui/_system_map_model.py`, `gui/_flow_layout.py` | GUI patterns and the System Map to fork |
| `~/adc2kev-python/docs/implementation/GUI_LAYOUT.md`, `SYSTEM_MAP.md` | GUI layout and map behaviour |
| `~/adc2kev-python/specview/` (`README.md`, `src/specview/session.py`, `gui/main.py`) | Model for a separate package on adc2kev, the session layer, and the scatter point cap |
| `~/adc2kev-python/pyproject.toml` | Tooling configuration to mirror |
