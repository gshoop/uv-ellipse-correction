# uvcorr: method, differences from RadialAnalysis, validation

This document describes what `uvcorr` computes for each channel, and where and why it differs from the
C++/ROOT **RadialAnalysis** (`~/DataProcessing/EllipseCorrection`, `main_radial.cpp` and
`uvd_common.h`). Decisions D1-D11 and the section numbers refer to
`docs/planning/UV_ELLIPSE_CORRECTION_PLAN.md`. Sections 6 and 7 hold the evidence: the C++ cross-check
and the census of the full test acquisition (`data_20260910_120628.dat`: Ge-68, full system,
208,267,950 events on active channels, 6,557 channels).

## 1. Summary of differences

| Step | RadialAnalysis (C++) | uvcorr | Effect on this file |
|------|----------------------|--------|---------------------|
| Input | `.uvd` text files per channel; loops R0 and R1 ch 4-28 | UV cache from the raw `.dat`; R0 ch 4-28, R1 ch 7-28 (D2) | R1 ch 6 (28 channels, 45,777 events) no longer fitted |
| Conic fit | Halir-Flusser on raw coordinates, explicit `S3⁻¹` | Same algorithm on normalised coordinates, `solve`, rank and eigenvalue checks | Agree to print precision on all 6,301 channels that are not pedestal blobs. On the 44 blobs (R ≈ 10 ADC) the C++ fit is numerically wrong by up to 39 % (6.2) |
| Outliers | none | far-outlier pre-clip + MAD-clipped refit (`robust`, default on) | Centre moves > 0.26 ADC on 10 % of channels (7.3) |
| φ | [−π/2, π), `PI = 3.1415926` | (−π/2, π/2], exact | Same ellipse (D10) |
| Pre radii | about `findCenter` (histogram peaks) | about the fitted centre | findCenter is off by 2 ADC (median), up to 540 ADC at low N |
| Radial Gaussian | ROOT χ² (Neyman), 200 bins over [min, max], range mean ± 3 RMS | binned Poisson ML (Baker-Cousins), FD bins 50-400 with 1-ADC floor, percentile range, iterated μ ± 2σ | Shipped `.tec` radiusStd vs C++: median ×1.05, ×1.25 on flat-topped rings (estimator), ×0.54 on thin rings with the second population (robust fit), ×0.55 at 100-999 events (6.5) |
| Failed Gaussian | pre: channel skipped; post: no `.tec` block | status stays `ok`; flag; sample mean/std reported (also as radiusStd) | 10 `.tec` blocks only uvcorr writes (6.5) |
| Skewness, kurtosis | binned, largest value excluded (ROOT overflow bin) | unbinned, all events | Differs only for heavy tails (6.4) |
| Residual stats | 150 bins over mean ± 6 std, ROOT fit | the same Gaussian routine as the radii | `corr_res_*` = post stats by construction (4.5) |
| Phase metrics | as in plan 4.4 | identical definitions, `nextafter` clamp | Agree to print precision |
| Output | old CSV schema, `.tec` | new CSV schema (D9), `.tec` byte format unchanged (D10) | |

## 2. Channel selection

uvcorr fits every event of each **active** channel: RENA 0 channels 4-28 and RENA 1 channels 7-28, anodes
and cathodes alike, with no PHA or coincidence gating (D2). The cache drops everything else at build time.
On this file that is 224,956 events: R0 ch 1-3 (179,179) and R1 ch 6 (45,777 events in 28 channels, 21
of them with ≥ 100 events).

RadialAnalysis loops channels 4-28 on **both** RENAs (`for(int ch=4; ch<CHNNUM; ch++)`). It therefore fits
R1 ch 4-6 whenever the extractData output has a file for them. Both RadialAnalysis `.tec` files on
disk contain R1 ch 6 blocks: 10 in `data_20250625_092749UVdata_ECv2.tec` and 8 in
`data_20260528_170144UVdata.tec`. uvcorr never writes them.

A channel is a cathode on even boards for ch 25-28 (both RENAs), and on odd boards for R0 ch 4-7 or R1
ch 7-10 (`isCathode`, `main_radial.cpp:279`). uvcorr takes polarity and electrode labels from
adc2kev's `ElectrodeMap`, which agrees.

## 3. Ellipse fit (`uvcorr.ellipse`)

### 3.1 Algebraic fit

The fit is the Fitzgibbon direct least-squares ellipse fit (constraint 4AC − B² = 1) in the Halir-Flusser
reduced form, as in `fitEllipse`, with four numerical changes:

1. **Normalisation.** The fit runs on x = (U − Ū)/s and y = (V − V̄)/s, where s is the RMS distance from the mean.
   On raw coordinates (U ≈ 2000, R ≈ 430-700) the 6×6 scatter matrix mixes entries of ~N·10¹³ with
   entries of ~N, and the C++ determinant test `|det S3| < 1e-12` is meaningless at that scale. The
   fit is exactly equivariant under this similarity, so uvcorr undoes it on the geometric
   parameters: cx = Ū + s·cx′, a = s·a′, and φ unchanged. It never goes back through raw-scale conic coefficients.
   The raw-coordinate fit is accurate enough on real rings (R ≈ 430 ADC). On small rings at U ≈ 2000, i.e. pedestal
   blobs of R ≈ 10 ADC, it fails: the full-file C++ fit is off by up to 39 % in the semi-axes on 44 such channels (6.2).
2. **Solve instead of an explicit inverse.** T = −S3⁻¹S2ᵀ is computed with `np.linalg.solve`. The fit fails
   if cond(S3) ≥ 1e10. On normalised data S3 = N·[[cov, 0], [0, 1]], so its condition is 1/(smallest principal
   variance): about 100 at b/a = 0.1, and about 10¹⁶ for collinear points.
3. **Rank check.** The reduced scatter matrix S1 + S2T must have rank ≥ 2 (its middle eigenvalue must be
   greater than 1e-10·trace S). With fewer than 5 distinct points a whole pencil of conics fits exactly
   and the eigenvectors are noise, so this is reported as `fit_failed`.
4. **Eigenvector choice.** The C++ takes the *first* eigenvector with 4AC − B² > 0 in `TMatrixDEigen`
   order. uvcorr skips eigenpairs with a non-negligible imaginary part and eigenvalues below −1e-8 of
   the largest. Of the constraint-satisfying ones it takes the smallest non-negative eigenvalue, which in
   exact arithmetic is the unique solution.

Conversion to (cx, cy, a, b, φ) uses the plan 4.2 formulas with the same degeneracy thresholds
(|disc| < 1e-12, |A′|, |C′| < 1e-12, a², b² ≤ 0, applied to a unit-norm (A, B, C)). Phase 2 checked the
fit against a 60-digit port of the C++ `fitEllipse` (agreement ~1e-14).

**Canonical form:** a ≥ b (axes swapped and π/2 added to φ if needed), and φ wrapped into **(−π/2, π/2]**
with `math.remainder(φ, π)` (exact). The C++ swap and its normalisation loops use `PI = 3.1415926`,
which adds up to ~1e-7 rad. They leave φ in [−π/2, π): the swap adds π/2 and `while (phi > PI)` only trims a ~3e-8 rad
sliver.

### 3.2 Robust iteration (`robust=True`, the default)

1. **Far-outlier pre-clip.** This seeds the first fit. It takes radii about the coordinate-wise median point and keeps points
   with |r − median r| ≤ 10 × 1.4826·MAD(r). The MAD is floored at 1e-3·median r, and the medians use a strided
   subsample of ≤ 65,536 points. For a ring the window is many times its width, so only compact far clusters
   go (e.g. events at an int16 corner, which would otherwise capture the algebraic fit). If this first
   fit fails, all points are used.
2. **Clip and refit.** For every point it computes the radial residual to the current ellipse (4.5). It keeps
   |res − median| ≤ `clip_k` × 1.4826·MAD (the scale is floored at 1e-9·√(ab), so noise-free data is
   never clipped) and refits. It stops when the kept set repeats or after `max_iter` refits. Every
   iteration re-evaluates **all** points, so a pre-clipped point on the ellipse comes back.
3. A failed refit (or fewer than 6 kept points) keeps the previous fit and sets `robust_refit_failed`.
4. `n_used` is the size of the final kept set. `n_rejected = n_events − n_used`.

### 3.3 Geometric refinement (`geometric=True`, default off)

Levenberg-Marquardt (`scipy.optimize.least_squares`, analytic Jacobian) minimises the radial residuals
of the kept points over (cx, cy, a, b, φ). It works in the normalised frame and starts from the algebraic solution. A failure keeps
the algebraic fit and sets `geometric_refit_failed`. On real data it changes nothing material (7.8),
so it stays off.

### 3.4 Correction

The correction is exactly `applyEllipseCorrection` (plan 4.3): translate to (cx, cy), rotate by −φ, scale by √(ab)/a and
√(ab)/b, and rotate back. The output is centred on the origin. `.tec` consumers apply the same transform.

## 4. Metrics (`uvcorr.metrics`, `uvcorr.analysis`)

### 4.1 Pre radii about the fitted centre

Pre-correction radii are measured about the **fitted** ellipse centre, not about `findCenter`, which
takes the midpoints of the two highest histogram bins of U and of V (100 bins over [min, max]).
Pre vs post then isolates the ellipticity. findCenter is a histogram-peak estimate. In the full-file cross-check
it is 2.0 ADC (median) from the fitted centre, 12 ADC at p95 and 540 ADC at worst. All 161 offsets above 100 ADC are on
channels with ≤ 176 events. Between 20 and 100 ADC there are also 21 channels with ≥ 1,000 events, all of them `broad_ring` (6.4).

### 4.2 Gaussian routine (radii and residuals)

1. **Histogram.** Values inside the [0.5, 99.5] percentiles. The bin count is Freedman-Diaconis, clamped to
   **50-400**, with a **1-ADC minimum bin width** (`ADC_MIN_BIN_WIDTH`) so bins never resolve the int16
   lattice, which at large N inflates χ²/ndf. The floor wins over the 50-bin minimum, down to 20 bins.
2. **Fit.** A Gaussian A·exp(−(x−μ)²/2σ²) is fitted to the counts by **binned Poisson maximum likelihood**, which minimises the
   **Baker-Cousins** deviance 2Σ(μᵢ − nᵢ + nᵢ ln(nᵢ/μᵢ)) with empty bins included. Unlike a Neyman χ²
   with σᵢ = √nᵢ and empty bins skipped (ROOT's default), it does not bias σ at small counts; see 6.4 for the size
   of that bias on real data. The optimiser works in standardised coordinates with tolerance 1e-12, so the
   result is translation- and scale-equivariant.
3. **Range.** The first fit covers median ± 3 × 1.4826·MAD. It is then refitted up to 3 times over μ ± 2σ of
   the previous fit, stopping when the same bins are selected again.
4. **Usability.** A fit is usable if it converged, all parameters are finite, 0.5·bin width ≤ σ ≤ 0.5·histogram
   range, μ lies inside the range, and the range has ≥ 5 bins with ≥ 3 non-empty. An unusable refit
   keeps the previous usable fit. If none is usable, `ok=False` and the sample mean/std (ddof 1) are reported.
5. With fewer than **`MIN_GAUSS_SAMPLES` = 50** values there is no histogram fit: `ok=False` and sample statistics are reported.
6. χ²/ndf is the Baker-Cousins deviance over the fitted bins divided by (bins − 3).

**No shape criterion.** A kurtosis guard (reject the fit when the excess kurtosis is too negative) was
tried in phase 2 and removed (`MIN_EXCESS_KURTOSIS` no longer exists). Real post-correction radii are
flat-topped. Over the 6,009 non-broad channels with ≥ 100 events, the excess kurtosis of the post radii
inside the [0.5, 99.5] percentiles has a median of **−0.45**, p25 −0.99 and p5 −1.08, and is negative for 87 % of
channels. A guard would fail most channels, including all the broad rings of nodes 5, 6 and 8-10. The fit's `ok`
depends only on the usability rules, and the unbinned shape statistics are reported alongside.

On flat-topped rings the Gaussian σ depends on the estimator. For the 2,646 channels with post robust σ ≥ 5 ADC,
the uvcorr σ is 1.025 × the robust σ (1.4826·MAD) and 1.23 × the sample std. For the 2,791 thin rings
(robust σ < 3 ADC) all three agree to 2 %.

### 4.3 Unbinned statistics

Also reported: the unbinned mean and std, the robust σ = 1.4826·MAD, and the **unbinned** biased skewness
m₃/m₂^1.5 and excess kurtosis m₄/m₂² − 3 (what `scipy.stats.skew`/`kurtosis(fisher=True)` return).
ROOT's `GetSkewness`/`GetKurtosis` are binned (bin centres about the unbinned mean, normalised by the
unbinned std) and exclude the maximum value (6.4).

### 4.4 Metrics over all events

Radial, residual and phase metrics use **all** finite events of the channel, including the ones the robust
fit rejected. The correction is applied to every event downstream, so the metrics describe what the
`.tec` consumers get. The Gaussian routine is itself robust to the rejected outliers, and
`n_rejected` reports how many there were.

### 4.5 Residuals

- Raw residual to the ellipse (plan 4.4) in the ellipse-aligned frame (uₐ, vₐ): r − r_ell(θ) with
  r_ell = ab/√((b cos θ)² + (a sin θ)²). It is evaluated without trigonometry as r − ab·r/√(b²uₐ² + a²vₐ²). A point
  exactly at the centre gets −a, as the C++ `atan2(0, 0) = 0` gives.
- Corrected residual: √(U′² + V′²) − √(ab), which is the post radius minus a constant. Because the
  Gaussian routine is translation-equivariant, **`corr_res_sigma` is identical to `post_sigma`** and
  `corr_res_mean = post_mean − target_radius` by construction. This is verified on all 6,115 fitted channels (σ identical,
  means within CSV rounding). The columns are kept for schema compatibility. In the C++ they differ
  only because the two histograms differ (200 bins over [min, max] vs 150 bins over mean ± 6 std).

### 4.6 Phase metrics and jitter proxy

- p = atan2(V′, U′) is wrapped to [0, 2π) and sorted. A tiny negative angle that rounds up to exactly 2π on
  wrapping is clamped to `nextafter(2π, 0)`, so it stays at the top of the CDF (the C++ would put 2π there).
  The gaps include the wrap-around gap ph[0] + 2π − ph[N−1]. The metrics are the mean gap 2π/N, the max gap in rad and
  ns (f = 490 kHz), and KS D = maxⱼ max((j+1)/N − Fⱼ, Fⱼ − j/N). They need N ≥ 8. The definitions are those of
  plan 4.4, and phase 2 found them bit-identical to the C++.
- `timing_jitter_ns` = post σ / (2π f · post μ) · 1e9. The C++ computes this only for its system summary
  plot (from `ell_sigma`/`ell_mean`). uvcorr writes it per channel.

## 5. Status, flags and output files

### 5.1 Status (exactly one per channel)

| Status | Meaning | This file |
|--------|---------|-----------|
| `ok` | The ellipse fit succeeded; all parameters and metrics are available | 6,115 |
| `too_few_events` | Fewer than `min_events` (100) finite events | 442 |
| `fit_failed` | No ellipse: < 6 points, collinear/coincident points (cond(S3) ≥ 1e10), a pencil of conics (rank < 2), no constraint-satisfying eigenvector, a degenerate discriminant or axis coefficient, or non-positive a², b² | 0 |

### 5.2 Flags (zero or more; status stays `ok`)

| Flag | Condition (`FitOptions` field, default) | This file | Meaning on real data (7.5-7.7) |
|------|------------------------------------------|-----------|--------------------------------|
| `high_rejection` | n_rejected > `high_rejection_frac` (0.05) × n_events | 128 | A second population on a thin b/a ≈ 0.30 ellipse, which the correction does not handle |
| `extreme_axis_ratio` | b/a < `extreme_axis_ratio` (0.5) | 9 | 8 channels wholly in the b/a ≈ 0.30 mode, 1 degenerate line |
| `broad_ring` | 1.4826·MAD of the kept points' residuals > `broad_ring_frac` (**0.05**, was 0.1) × √(ab) | 106 | Not a thin ring: pedestal blobs, filled boxes, mixed-mode fits |
| `center_outside_data` | fitted centre outside the bounding box of the points | 0 | e.g. a short arc |
| `robust_refit_failed` | a robust refit failed; the previous fit kept | 0 | (4 channels below 20 events with `min_events=6`) |
| `geometric_refit_failed` | geometric refinement failed; the algebraic fit kept | 0 | only with `geometric=True` |
| `gauss_fit_failed_pre` | pre-radii Gaussian not usable; sample stats reported | 1,093 | Resolved double-horned pre radii, **informational** |
| `gauss_fit_failed_post` | post-radii Gaussian not usable; sample stats reported (also the `.tec` `radiusStd`) | 15 | 13 of 15 also `broad_ring` |

1,299 channels carry at least one flag. Only 241 carry a flag other than `gauss_fit_failed_pre`.

### 5.3 `.tec` (D10)

The `.tec` is byte-for-byte in the RadialAnalysis layout: a `channel{` line, then tab-indented `key=value` lines in the order
node, board, rena, channel, centerU, centerV, semiMajor, semiMinor, phi, radius (= √(ab)) and radiusStd (= post
Gaussian σ), then `}`. Blocks follow each other with no blank lines, and the file ends with `}\n`. Doubles are written as C++ `ostream`
default formatting (`%g`, 6 significant digits; Python `format(x, "g")` gives identical bytes, as verified against
a compiled `cout` on 30,000 values). Only `status == ok` channels get a block, in (node, board, rena, channel) order.

Compatibility notes:
- **φ range.** uvcorr writes φ ∈ (−π/2, π/2]. RadialAnalysis writes φ ∈ [−π/2, π) (3.1): its two `.tec` files on disk
  span −1.57027 to 3.14155, and 1,617 of 5,981 blocks in `data_20250625_092749UVdata_ECv2.tec` and 2,108 of 6,476 in
  `data_20260528_170144UVdata.tec` lie outside (−π/2, π/2]. An ellipse is invariant under φ → φ + π, and so is the correction, since the rotation and back-rotation
  cancel. The user confirmed consumers accept this (D10).
- **Channel set.** There are no R1 ch 4-6 blocks (section 2). Channels whose pre Gaussian fails are *not* dropped
  (the C++ drops them). Channels below `min_events` are dropped (the C++ fits anything with ≥ 6 events that passes its pre fit). On the
  full file uvcorr writes 10 blocks the C++ does not, and the C++ writes 240 (6-99 events) that uvcorr does not (6.5).
- **radiusStd** changes, for three reasons (6.5): the Gaussian estimator, the robust ellipse, and the C++ σ inflation at low
  counts. Over the 6,105 common blocks the ratio
  uvcorr/C++ has median 1.05. It is 1.25 on flat-topped rings (estimator) and 1.00 on clean thin rings. On thin rings with a
  second population it is 0.54, because uvcorr's σ describes the main ring and the C++ σ is inflated by the second
  population. At 100-999 events it is 0.55. A channel whose post Gaussian fails still gets a block, with radiusStd = the sample std
  (15 blocks).

`radial_summary.csv` uses the new schema (D9, plan 6.2): `%.6g` floats, `%.9g` for centre, axes and φ, and empty
cells for unavailable values.

## 6. Cross-check against the C++ RadialAnalysis

### 6.1 Setup

```bash
venv/bin/python scripts/dump_uvd.py --cache <cache> --out uvd          # default: node 1 boards 15, 16; node 8 board 24
source ~/root-install/bin/thisroot.sh && ~/DataProcessing/EllipseCorrection/RadialAnalysis --csv-only $PWD/uvd
venv/bin/python scripts/compare_radial.py uvd --cache <cache> --min-events 6 --decompose --out comparison.csv
# or, on a scratch cache (a --no-robust run replaces the stored batch and drops the robust-off overrides):
# uvcorr process data.dat --no-robust --cache /tmp/data.uv.h5 --output-dir X; compare_radial.py uvd --uvcorr-dir X
```

`compare_radial.py --cache` only reads the cache. A `uvcorr process --no-robust` run on the real cache would make the
robust-off fit its stored batch and drop every robust-off override, hence the scratch cache (built in about 21 s).

There are two runs. The first covers three boards: node 1 boards 15 and 16 (plan 10.3) and node 8 board 24, a low-count board whose
47 channels hold 1 to 19,516 events. It has 135 channels and 2.77M events, and RadialAnalysis `--csv-only` took 2.1 s.
The second is the **full file** (`dump_uvd.py --boards` with every board): 6,557 channels and 208M events. There RadialAnalysis took
1 min 54 s and `compare_radial.py --decompose` 2 min 52 s. The numbers below are from the full-file run
unless stated; the three-board run gives the same picture. uvcorr runs on the fly with `robust=False` and
`min_events=6`, which fits every channel the C++ can. Section 6.5 compares the shipped defaults (`robust=True`,
`min_events=100`) instead.

**`.uvd` format.** Each file has the header line `U Values V Values` and one `"%d %d\n"` line per event, so it ends with a newline. The
C++ `loadUVD` *always* discards line 1. It then reads pairs until EOF and pops the last pair: that pair is garbage when the file ends with
whitespace, and a real event when it does not. Two consequences:
- The current extractData `exportUVD` writes **no header**, so RadialAnalysis silently loses the first event of
  every channel it reads from extractData output. Only the old Python DAQ exporter wrote this header.
- Without the trailing newline, the last event would be lost instead.

With `dump_uvd.py` files, the C++ `num_events` equals uvcorr's `n_events` on **all 6,353** channels with a C++ row.

### 6.2 Ellipse parameters

The C++ writes the centre only to the `.tec`, with 6 significant digits (±0.005 ADC). It writes a, b and φ to the CSV with 6 decimals.
The tolerance is rtol = **1e-6** (plan 10.3) plus half the printed digit. For φ it is 6e-7 rad + rtol·a²/(a² − b²), since φ of a
near-circle is ill-conditioned. `compare_radial.py` reports channels that uvcorr flags `broad_ring` on a separate line, and does not
count their disagreements as failures unless `--strict-blobs` is given.

Full file, 6,223 ring channels (both sides `ok`, not `broad_ring`):

| Parameter | median \|rel\| | max \|rel\| | max \|Δ\|/tol | Limited by |
|-----------|------------:|---------:|------------:|------------|
| centerU | 1.2e-6 | 2.5e-6 | 0.71 | `.tec` rounding (±0.005 ADC) |
| centerV | 1.2e-6 | 2.5e-6 | 0.71 | `.tec` rounding |
| semiMajor | 5.6e-10 | 9.4e-8 | 0.09 | CSV 6 decimals |
| semiMinor | 5.7e-10 | 3.2e-8 | 0.03 | CSV 6 decimals |
| radius (`.tec`) | 5.6e-7 | 1.2e-6 | 0.55 | `.tec` rounding |
| φ mod π | \|Δφ\| 2.5e-7 rad | 5.3e-7 rad | 0.20 | CSV 6 decimals + `PI = 3.1415926` |

**Ellipses agree to print precision on all 6,301 channels that are not pedestal blobs.** That is the 6,223 ring channels plus 78 of
the 122 `broad_ring` channels. The comparison is limited by the C++ print precision. Phase statistics agree to the print step
(max gap on 6,255, KS on 6,245 of 6,347 channels; max gap in ns within 3e-7 relative).

**The 44 pedestal blobs disagree** (6.6): node 3 board 15 R1 ch 7-28 and node 6 board 21 R1 ch 7-28. These are
compact blobs of fitted radius √(ab) ≈ 8-16 ADC around U, V ≈ 2070, and all of them are `broad_ring` in this comparison. In the default run
43 are `ok` + `broad_ring` and one has 29 events, so it is `too_few_events`. The C++ semi-major axis is off by up to 39 %,
the semi-minor by up to 32 %, and the centre by up to 1.6 ADC in one coordinate (2.0 ADC in distance).
A float64 fit on integer-shifted coordinates reproduces uvcorr exactly. For example, node 3 board 15 R1 ch 9 has a = 18.799 in both, against
21.904 in the C++, and numpy's explicit-inverse fit on raw coordinates gives a third answer (or none). On normal
rings all three agree (node 1 board 15 R0 ch 4: a = 717.9055).

### 6.3 Channel sets (full file)

| | Count |
|---|---:|
| `.uvd` inputs | 6,557 |
| C++ CSV rows | 6,353 |
| C++ skipped, findCenter failed (a half of the U or V histogram empty) | 173, all ≤ 5 events; uvcorr `too_few_events` |
| C++ skipped, pre Gaussian failed | 31: 21 with < 6 events, and 10 that uvcorr fits (e.g. node 1 board 16 R0 ch 20, 1,457 events, mixed-mode; node 8 board 24 R1 ch 21, 102 events, a clean thin ring; two of the b/a ≈ 0.30 channels) |
| C++ rows without an ellipse (`fitEllipse` failed: it needs 6 points) | 6, all with 5 events |
| C++ rows with an ellipse but no `.tec` block (post Gaussian failed) | 2 (node 1 board 17 R0 ch 23, b/a 0.30; node 10 board 23 R0 ch 8) |
| C++ `.tec` blocks | 6,345 |

### 6.4 Radial and residual statistics (`robust=False`, full file)

Differences uvcorr − C++ over the 6,347 channels where both sides have statistics (p5 / median / p95): rel = relative, abs = absolute,
ratio = uvcorr/C++.

| Statistic | kind | p5 | median | p95 |
|-----------|------|---:|-------:|----:|
| pre μ | rel | −0.40 % | +0.07 % | +0.89 % |
| pre σ | rel | −80 % | +20 % | +83 % |
| pre χ²/ndf | ratio | 0.34 | 2.0 | 5.5 |
| pre skewness / kurtosis | abs | −0.36 / −0.46 | 0.00 / −0.01 | +0.29 / +4.6 |
| post μ | rel | −0.13 % | 0.00 % | +0.09 % |
| post σ | rel | −52 % | +9.6 % | +46 % |
| post χ²/ndf | ratio | 0.41 | 1.7 | 3.4 |
| post skewness / kurtosis | abs | 0.00 / −1.4 | +0.002 / +0.002 | +0.41 / +0.73 |
| raw / corrected residual σ | rel | −12 % / −11 % | +11 % / +11 % | +51 % / +51 % |
| timing jitter (ns) | rel | −52 % | +9.6 % | +46 % |
| phase mean gap, max gap, KS | | agree to the C++ print step | | |

**Decomposition.** `compare_radial.py --decompose` re-implements the C++ statistics in Python: the
`findCenter` emulation, `TH1D` filling (the maximum lands in the overflow bin), ROOT's fit-range rule (`ExamineRange`), and
Neyman χ² with empty bins skipped, `GetSkewness`/`GetKurtosis` and `residualHistogram`. ROOT itself (via PyROOT),
fitting the emulated radii, reproduces the C++ CSV values to all printed digits. A unit test pins the emulation to PyROOT
reference numbers. The emulation reproduces the C++ pre/post μ and σ within 1e-3 on **5,994 of 6,347** channels (median 1e-8).
Of the other 353 (6 to 252,599 events), the emulated Levenberg-Marquardt fit fails on 189 where Minuit returned something, and
gives a different result on 164. Most of the 353 are low-count channels (median 67 events, 309 below 1,000) or `broad_ring` (72),
where the χ² surface is flat or multi-modal. On the 5,994 channels, one ingredient is
changed at a time (median [p5, p95] vs C++):

| Step | pre σ | post σ |
|------|------:|-------:|
| C++ emulation | 0 [±3e-6] | 0 [±3e-6] |
| + fitted centre instead of findCenter | −3.1 % [−15 %, +2.8 %] | n/a |
| + Poisson ML on the same 200 bins and range | +12 % [−6.6 %, +48 %] | +5.3 % [+0.1 %, +45 %] |
| = uvcorr: + FD bins, percentile range, iterated μ ± 2σ | +21 % [−3.4 %, +82 %] | +11 % [−4.5 %, +46 %] |

By population, post σ uvcorr/C++ is:

| Channels | n | median [p5, p95] | ML on C++ bins / C++ | median σ C++ → uvcorr (robust σ) | Main cause |
|----------|--:|------------------|----:|-----------------------------------|------------|
| < 100 events | 240 | 0.16 [0.11, 0.44] | 0.24 | 25.8 → 4.4 ADC | Neyman χ² on a sparse 200-bin histogram (mostly 0/1 counts, empty bins skipped) inflates σ |
| 100-999 events | 290 | 0.67 [0.25, 1.08] | 0.69 | 6.4 → 4.3 (4.1) | the same, weaker |
| ≥ 1000, thin rings (robust σ < 3 ADC) | 2,259 | 1.01 [0.95, 1.06] | 1.02 | 1.47 → 1.48 (1.48) | none: both fits describe a Gaussian peak |
| ≥ 1000, flat-topped rings (≥ 3 ADC) | 3,475 | 1.26 [1.06, 1.51] | 1.11 | 6.2 → 7.9 (7.9) | the Gaussian σ of a box is estimator-dependent: ×1.11 from Neyman vs ML (Neyman weights the low tail bins), the rest from fitting the core (μ ± 2σ) instead of mean ± 3 RMS |

The other statistics follow:
- **Pre σ, μ, skewness, kurtosis.** findCenter is 2.0 ADC from the fitted centre (median), 12 ADC at p95 and 540 ADC at worst. The large offsets are
  on low-count channels (all 161 above 100 ADC have ≤ 176 events), where the C++ pre "fit" returns σ of hundreds of ADC with χ²/ndf ≈ 0.1-0.6. At high N pre radii are double-horned
  (arcsine-like, half-width (a − b)/2 ≈ 7 ADC), and any Gaussian σ is a convention.
- **Skewness and kurtosis.** The emulation reproduces the C++ exactly. Unbinned statistics *without the maximum*
  agree with the C++ on most channels (median difference 2e-4 in kurtosis, 1e-4 in skewness). So the large differences come from
  ROOT dropping the largest radius (it lands in the overflow bin), where one far outlier dominates m₄, and from
  binning on heavy-tailed channels. Examples: node 1 board 16 R0 ch 16 has post kurtosis 0.30 in the C++ and 2,375 unbinned
  (0.18 with the one maximum removed), and node 8 board 24 R1 ch 14 has 0.44 vs 280 (0.50). Kurtosis is not a robust statistic, and both columns are only indicative.
- **Residual σ.** The residual σ follows the post σ: the C++ fits 150 bins over mean ± 6 std, uvcorr uses the radial routine.
- **χ²/ndf.** The values are not comparable: the C++ uses Neyman χ² over its range, uvcorr the Baker-Cousins deviance over narrower 1-ADC bins.

### 6.5 `radiusStd` in the shipped `.tec` vs the C++ `.tec`

This is the comparison downstream users care about. It uses `out/data_20260910_120628.tec` (defaults: robust fit, `min_events=100`) against the full-file
RadialAnalysis `.tec`, over the **6,105** channels with a block in both. radiusStd uvcorr/C++:

| Channels | n | median [p5, p95] |
|----------|--:|------------------|
| all common blocks | 6,105 | **1.05 [0.44, 1.43]**: 43 % up by more than 10 %, 15 % down by more than 10 % |
| ≥ 1000 events, thin rings (post robust σ < 3 ADC), not `broad_ring` | 2,656 | 1.00 [0.37, 1.05] |
| … of which reject ≤ 0.5 % of events | 2,030 | 1.01 [0.95, 1.05] |
| … of which reject > 0.5 % (the second population, 7.5) | 626 | 0.54 [0.20, 0.89] |
| ≥ 1000 events, flat-topped rings (≥ 3 ADC), not `broad_ring` | 3,086 | 1.25 [1.03, 1.47] |
| 100-999 events | 290 | 0.55 [0.10, 1.04] |
| `broad_ring` | 102 | 1.00 [0.13, 1.98] |

There are three causes:
1. **Estimator on flat-topped rings** (6.4): uvcorr's σ is about 1.25× the C++ value, and close to the robust σ.
2. **Robust fit on second-population channels.** 359 of the 2,656 thin rings are below ×0.6, and every one of them rejects more than 0.5 % of its events.
   Their median radiusStd is 1.41 ADC in uvcorr, 4.91 ADC with robust off, and 3.57 ADC in the C++. The robust ellipse locks onto the main
   ring, so uvcorr's σ describes the main ring. The C++ (plain) ellipse is pulled by the b/a ≈ 0.30 population, which broadens
   its corrected radii. The drop comes from the fit, not from the Gaussian estimator.
3. **Low counts.** At 100-999 events the C++ Neyman χ² inflates σ (6.4), and the robust fit adds to the drop.

Blocks written by one side only:
- **10 only in uvcorr.** For 9 of them the C++ skipped the channel because its pre Gaussian failed: node 1 board 16 R0 ch 20, node 3 board 29 R1 ch 24, node 4 board 15
  R0 ch 27, node 4 board 25 R0 ch 24, node 7 board 18 R1 ch 20, node 8 board 18 R1 ch 26, node 8 board 24 R1 ch 21, node 9 board 29 R0 ch 4 and node 10 board 20 R0 ch 12.
  For the 10th (node 1 board 17 R0 ch 23, b/a 0.30) the C++ post Gaussian failed, and the C++ writes no block then.
- **240 only in the C++.** These are the channels with 6-99 events, which are `too_few_events` in uvcorr.
- A channel whose post Gaussian fails in uvcorr still gets a block, with **radiusStd = sample std**. That is 15 blocks (`gauss_fit_failed_post`,
  13 of them also `broad_ring`). 14 of them also have a C++ block, where ROOT's fit returned a σ. Node 9 board 29 R0 ch 4 is in uvcorr only.

### 6.6 C++ issues found (not uvcorr bugs; for the record)

1. `loadUVD` drops the first event of every extractData `.uvd` file, which has no header (6.1).
2. **The raw-coordinate fit is numerically wrong on small rings.** On the 44 pedestal blobs (R ≈ 8-16 ADC at U, V ≈ 2070) the semi-axes are off
   by up to 39 % and the centre by up to 1.6 ADC (6.2). This is the conditioning problem of section 3.1: the scatter matrix mixes
   x⁴ ≈ 2·10¹³ with 1. A fit on normalised (or just integer-shifted) coordinates is exact.
3. The ROOT histogram over [min R, max R] puts the maximum in the overflow bin. It is excluded from the fit, mean/RMS,
   skewness and kurtosis.
4. `PI = 3.1415926` in the axis swap and φ normalisation (≤ 1.1e-7 rad), and φ is left in [−π/2, π). The swap adds π/2 and
   `while (phi > PI)` only trims a ~3e-8 rad sliver. The old `.tec` files span −1.57027 to 3.14155.
5. Neyman χ² with √n errors and empty bins skipped biases σ upwards on sparse histograms: ×6 (median) below 100 events and
   ×1.5 at 100-999 events relative to uvcorr. A failed *pre* fit drops the whole channel, even when the ellipse fit is fine (31 channels,
   10 of them fittable, e.g. node 8 board 24 R1 ch 21).
6. findCenter is unreliable at low N (errors up to 540 ADC), which makes the C++ pre statistics meaningless there.

## 7. Census of the full test file and retuning

All numbers are from `uvcorr process` with default options (after the retune in 7.6) on the full file,
unless stated. Scratch analyses that re-fit with other options used the same cache.

### 7.1 Status and flags

Status: 6,557 channels (208,267,950 events on 155 boards). `ok` 6,115 (anodes 5,139 of 5,492; cathodes 976 of
1,065), `too_few_events` 442, `fit_failed` 0.

| Flag | All | Anodes | Cathodes |
|------|----:|-------:|---------:|
| gauss_fit_failed_pre | 1,093 | 869 | 224 |
| high_rejection | 128 | 113 | 15 |
| broad_ring (0.05) | 106 | 87 | 19 |
| gauss_fit_failed_post | 15 | 12 | 3 |
| extreme_axis_ratio | 9 | 5 | 4 |
| center_outside_data, robust_refit_failed | 0 | 0 | 0 |

### 7.2 Distributions (`ok` channels)

| Quantity | p5 | p25 | median | p75 | p95 | max |
|----------|---:|----:|-------:|----:|----:|----:|
| pre σ (ADC) | 5.3 | 6.7 | 8.7 | 10.6 | 13.1 | 212 |
| post σ (ADC) | 1.27 | 1.50 | 3.9 | 8.5 | 11.5 | 165 |
| pre robust σ | 5.7 | 7.1 | 8.0 | 9.3 | 11.8 | 282 |
| post robust σ | 1.27 | 1.51 | 3.8 | 8.4 | 10.9 | 114 |
| post / pre robust σ | 0.16 | 0.21 | 0.52 | 0.92 | 1.02 | 5.2 |
| timing jitter (ns) | 0.95 | 1.12 | 2.9 | 6.3 | 8.8 | 134 |
| axis ratio b/a | 0.960 | 0.964 | 0.966 | 0.969 | 0.973 | 0.981 |
| phase KS D: anodes / cathodes | 0.0045 / 0.0099 | | 0.011 / 0.033 | | 0.059 / 0.127 | 0.28 |
| max phase gap (ns) | 1.4 | 1.8 | 2.2 | 2.9 | 16 | 224 |
| target radius √(ab) (ADC) | 416 | 426 | 432 | 439 | 657 | 705 |
| rejected fraction | 0 | 0 | 0 | 3.1e-4 | 0.027 | 0.29 |

- **Ellipticity** is nearly the same everywhere, b/a = 0.966 ± 0.004, with a clean gap down to the b/a ≈ 0.30 channels (7.5).
- **Post σ is bimodal by node.** The median post robust σ is 1.55-1.80 ADC on nodes 1-4, 2.6 on node 7, and 5.3-8.1 on nodes
  5, 6, 8, 9 and 10. The broad rings are flat-topped (4.2). The 2,646 channels with post robust σ ≥ 5 ADC have robust σ
  5.5-11.3 ADC (p5-p95, median 8.7), which is a box of full width 4·MAD = robust σ / 0.371 ≈ 15-30 ADC (median 23 ADC). The correction removes the
  ellipticity: post/pre robust σ is 0.21 on thin rings and ≈ 0.9-1.0 where the ring width dominates.

**Rejected fraction per class** (`ok` channels; count deciles over all `ok` channels; median / p90 / max / number above 5 %,
2 significant figures):

| Decile (events) | Anodes: n | median | p90 | max | > 5 % | Cathodes: n | median | p90 | max | > 5 % |
|-----------------|--:|---:|---:|---:|--:|--:|---:|---:|---:|--:|
| D1 100-4,649 | 525 | 0 | 0.0089 | 0.20 | 22 | 87 | 0 | 0.0085 | 0.11 | 3 |
| D2 -8,735 | 594 | 0 | 0.0077 | 0.14 | 5 | 17 | 0 | 0.00037 | 0.00083 | 0 |
| D3 -11,726 | 596 | 0 | 0.010 | 0.072 | 1 | 16 | 5.4e-5 | 0.038 | 0.090 | 2 |
| D4 -15,127 | 591 | 0 | 0.0080 | 0.11 | 16 | 20 | 0 | 0.00040 | 0.0074 | 0 |
| D5 -19,343 | 586 | 0 | 0.014 | 0.29 | 16 | 25 | 0 | 0.015 | 0.090 | 1 |
| D6 -23,695 | 588 | 8.5e-5 | 0.021 | 0.12 | 18 | 24 | 0 | 9.4e-5 | 0.0033 | 0 |
| D7 -28,312 | 571 | 7.4e-5 | 0.020 | 0.097 | 10 | 40 | 0 | 0.0020 | 0.11 | 1 |
| D8 -35,769 | 569 | 6.1e-5 | 0.024 | 0.085 | 13 | 43 | 0 | 0.00067 | 0.036 | 0 |
| D9 -56,739 | 484 | 1.1e-4 | 0.027 | 0.082 | 11 | 127 | 1.9e-5 | 0.0026 | 0.14 | 1 |
| D10 -946,025 | 35 | 6.4e-5 | 0.0081 | 0.057 | 1 | 577 | 3.5e-5 | 0.0032 | 0.25 | 7 |

There is no trend with count and no anode/cathode difference that would call for class-specific thresholds. Half the channels
reject nothing (3,097 exactly zero). The distribution of the rejected fraction is continuous: 1,830 channels reject more than 0 but under 0.1 %, 214 reject 0.1-0.5 %, 212 reject 0.5-1 %,
305 at 1-2 %, 329 at 2-5 %, 96 at 5-10 % and 32 at ≥ 10 %.

### 7.3 Robust iteration: `clip_k` (4) and `max_iter` (5)

Refits needed with the defaults (6,115 channels): 0: 3,121; 1: 2,146; 2: 751; 3: 68; 4: 5; 5 (= `max_iter`): 24. Of those
24, 22 had not converged. With `max_iter=20`, 19 still cycle. Going from 5 to 20 iterations moves their centre by 0.001 ADC (median),
0.14 ADC at p95 and 1.2 ADC at worst.

| clip_k | mean rejected | median | p99 | channels rejecting nothing | high_rejection | reach max_iter | centre shift vs k=4, p95 / max |
|-------:|------:|------:|----:|----:|----:|----:|----:|
| 3.0 | 0.64 % | 0.18 % | 8.4 % | 38 % | 141 | 172 | 0.025 / 12 ADC |
| 3.5 | 0.49 % | 0.02 % | 7.8 % | 44 % | 130 | 58 | 0.008 / 9.4 ADC |
| **4.0** | 0.46 % | 0 | 7.7 % | 51 % | 128 | 24 | 0 |
| 5.0 | 0.43 % | 0 | 7.4 % | 70 % | 121 | 6 | 0.007 / 15 ADC |
| 6.0 | 0.41 % | 0 | 7.1 % | 74 % | 110 | 7 | 0.009 / 15 ADC |

The fit is insensitive to k between 3.5 and 6: the centre moves < 0.01 ADC at p95, and the maxima are on the mixed-mode
channels. k = 3 costs iterations without changing the ellipses. **Keep k = 4 and max_iter = 5.**

Robust vs plain fit (k = 4 vs `robust=False`): the centre moves 0 ADC (median), 0.26 ADC at p90, 2.2 ADC at p99 and 15 ADC at worst.
The axes move up to 9.6 ADC at p99 and 55 ADC at worst. The large shifts are on channels with the second population (7.5)
or on non-rings. **Keep robust on.**

### 7.4 `min_events` (100)

The 442 channels below 100 events break down as 200 with < 6 events, 99 with 6-19, 55 with 20-49 and 88 with 50-99. Refitted with
`min_events=6`, every one of the 242 channels with ≥ 6 events fits. The axis ratio is sane (0.94-0.99) for 79 of 99 channels at 6-19 events, 54 of 55 at 20-49 and
86 of 88 at 50-99. Below 50 events the Gaussian routine never fits, so `gauss_fit_failed_pre/post` is set by construction.

To measure accuracy, 240 unflagged channels with ≥ 5,000 events were subsampled (40 random subsets per size) and each subset
ellipse was applied to the channel's kept events, compared with the full-channel fit. The rms phase error is after removing the constant offset, converted to ns at 490 kHz.
Values are medians [p95]:

| n | flagged | centre error (ADC) | rms phase error (ns) | thin rings | broad rings | post robust σ ratio |
|--:|--:|--:|--:|--:|--:|--:|
| 10 | 9.1 % | 2.9 [16] | 2.3 [10] | 0.94 [4.3] | 3.6 [12] | 1.19 [2.5] |
| 15 | 7.9 % | 1.9 [7.6] | 1.6 [5.3] | 0.66 [1.7] | 2.5 [5.8] | 1.09 [1.5] |
| 20 | 1.2 % | 1.5 [5.9] | 1.25 [4.0] | 0.54 [1.25] | 2.1 [4.5] | 1.05 [1.27] |
| 30 | 1.7 % | 1.1 [4.4] | 0.96 [3.0] | 0.42 [0.91] | 1.6 [3.3] | 1.03 [1.16] |
| 50 | 1.4 % | 0.84 [3.4] | 0.73 [2.2] | 0.31 [0.67] | 1.2 [2.5] | 1.02 [1.09] |
| 100 | 0.4 % | 0.58 [2.2] | 0.50 [1.6] | 0.21 [0.47] | 0.82 [1.7] | 1.01 [1.05] |
| 200 | 0.2 % | 0.41 [1.6] | 0.35 [1.1] | 0.15 [0.32] | 0.58 [1.2] | 1.00 [1.03] |
| 1000 | 0.1 % | 0.17 [0.69] | 0.15 [0.47] | 0.07 [0.14] | 0.25 [0.52] | 1.00 [1.01] |

Fits stay sane down to about 20 events: false flags jump from about 1 % to 8 % below 20. At 50 events the ellipse-induced phase error (0.7 ns
median) is small next to the channels' own jitter (median 2.9 ns; about 1.1 ns thin, 6-8 ns broad). So 100 is
conservative, and **50 would be defensible**: it matches `MIN_GAUSS_SAMPLES`, so every `ok` channel still gets a Gaussian fit, and it would add 88
channels with about 6.6k events. **Not changed.** 100 is not wrong, the gain is 88 low-count channels (which channels get a `.tec` block
is a policy choice), and the census and the C++ comparisons in this document assume 100. The default is defined only in
`FitOptions`: the CLI help and the GUI's control band read it from there, and `tests/test_options.py` pins it. Use
`--min-events 50`, or 20 at the lowest, for more coverage.

### 7.5 `high_rejection_frac` (0.05) and `extreme_axis_ratio` (0.5): the second ellipse

Scatter plots of the flagged channels show that the rejected events are not random. Most sit on a **second, thin
ellipse**, concentric with the main ring.

**Selection rule.** Take the `ok` channels with ≥ 100 events and ≥ 20 rejected events (1,214 channels). Fit the rejected events
with `fit_ellipse` (defaults, robust, `min_events=20`); all 1,214 fits succeed. Accept a channel if the secondary fit has
0.25 ≤ b/a ≤ 0.35 and uses ≥ 50 % of the rejected events. There is no φ or centre window. This accepts **910** channels,
and 901 of them also pass a window of φ −47° to −41° with the centre within 10 ADC of the main one. The count depends on the rule: a
*plain* (non-robust) secondary fit with b/a 0.25-0.35, φ −47° to −41° and centre within 10 ADC accepts 784 (review). The
accepted population has:
- b/a = 0.30 (median 0.302), φ = −44° (p1-p99 −44.6° to −43.5°), a ≈ 581 ADC and b ≈ 175 ADC (median);
- a centre within 1.6 ADC (median; p99 5.8) of the main centre;
- 0.24-8.6 % of the channel's events (p5-p95), and up to 29 %.

The share of channels with this signature grows with the rejected fraction: 134 of 212 at 0.5-1 % rejected, 230 of 305 at 1-2 %, 314 of 329 at 2-5 %, 89 of 96
at 5-10 % and 32 of 32 above 10 %. Interpretation: for a pair A cos ωt, A cos(ωt + δ) the Lissajous ellipse has semi-axes
√2A sin(δ/2) and √2A cos(δ/2) on the diagonals. The main ring is δ = 90° (radius A ≈ 430). a = 581 then gives
δ ≈ 146°, and cot(73°) = 0.31 matches the measured b/a. The second population is therefore *consistent with* a fraction of
events recorded with a quadrature offset of about 146° instead of 90° (an inference from the geometry, not a
hardware measurement). Whatever its origin, a single-ellipse correction mis-corrects these events.

- `high_rejection` (128 channels): 121 of them show the second ellipse. The remaining 7 are sparse or blob
  channels (141-723 events) with a few random outliers; 2 of them are also `broad_ring`. The main-ring fit is
  correct in every channel inspected. The flag is genuine, but the phenomenon is continuous down to below 1 %,
  so 5 % is a policy cut. The rejected fraction is available as a map metric. **Keep 0.05.**
- `extreme_axis_ratio` (9 channels): 8 channels at b/a = 0.298-0.304 are *entirely* in the b/a ≈ 0.30 mode. They are
  good thin rings, so the correction is valid for them, and they sit on 7 boards on 4 nodes. The 9th (node 10
  board 20 R0 ch 12, b/a 0.006) is a degenerate line and also `broad_ring`. The b/a distribution is empty between 0.304 and 0.503 (a blob), and every normal channel is
  ≥ 0.952. **Keep 0.5.** A value in the middle of the gap (0.4) flags the same channels, if margin is wanted.

### 7.6 `broad_ring_frac`: **retuned 0.1 → 0.05**

The flag's own metric, the robust spread of the kept points' residuals / √(ab), has a clean empty gap:

| spread / √(ab) | < 0.01 | 0.01-0.02 | 0.02-0.0288 | **(0.0288, 0.0759)** | 0.0759-0.1 | 0.1-0.2 | 0.2-1 | > 1 |
|----------------|-------:|----------:|------------:|---------------------:|-----------:|--------:|------:|----:|
| channels | 3,216 | 1,575 | 1,218 | **0** | 8 | 21 | 76 | 1 |

The widest ring is 0.02877 (node 9 board 30 R1 ch 7), and the narrowest non-ring is 0.07592 (node 6 board 21 R1 ch 15).

Scatter plots confirm what the flagged channels are:
- pedestal-only blobs (e.g. node 6 board 21 R1, node 3 board 15 R1). Their fitted √(ab) is 8-12 ADC (8-16 ADC with robust off), so they
  span about 20-25 ADC in the scatter plots;
- uniformly filled rectangles (node 8 board 26 R1);
- sparse scatter;
- mixed-mode channels where the b/a ≈ 0.30 population is large, so the fit lands between the two ellipses (e.g. node 6 board 26 R1 ch 24, node 9 board 29 R0 ch 18 and 22, node 2 board 23 R0 ch 27).

At 0.1 the 8 channels in [0.0759, 0.1) were **missed**, and all 8 are non-rings: 6 blobs on node 6 board 21 R1, a mixed-mode fit
(node 10 board 15 R1 ch 12) and a sparse scatter (node 8 board 24 R1 ch 26). 0.05 sits in the middle of the gap (0.0288, 0.0759), with the
widest normal ring 1.7× below it and the narrowest non-ring 1.5× above.

| | broad_ring | channels with any flag | with a non-informational flag |
|---|---:|---:|---:|
| before (0.1) | 98 | 1,294 | 236 |
| after (0.05) | 106 | 1,299 | 241 |

(3 of the 8 new ones were already `high_rejection`.) Among the `broad_ring` channels, 27 also carry
`gauss_fit_failed_pre`, 13 `gauss_fit_failed_post` and 1 `extreme_axis_ratio`.

### 7.7 Informational flags for the GUI (System Map, Status mode)

- **`gauss_fit_failed_pre`: informational.** It fires on 1,093 channels (17.9 % of `ok`), driven by resolved double horns in
  the pre radii, i.e. by (a − b)/(2 σ_post,robust), where σ_post,robust = 1.4826·MAD of the post radii (all 6,115 `ok` channels):

  | (a−b)/(2 σ_post,robust) | < 0.5 | 0.5-1 | 1-1.5 | 1.5-2 | 2-3 | 3-5 | ≥ 5 |
  |---|--:|--:|--:|--:|--:|--:|--:|
  | channels | 24 | 1,741 | 913 | 401 | 402 | 1,342 | 1,292 |
  | pre fit failed | 0 % | 1.3 % | 0.5 % | 0.2 % | 2.2 % | 10 % | 71 % |

  These are the *best* channels (thin rings, nodes 1-4). The pre statistics are descriptive only and the correction is unaffected.
- **All other flags: warnings.** `broad_ring` means the fit should not be trusted. `high_rejection` and `extreme_axis_ratio`
  mean the fit is fine but part or all of the data is in the b/a ≈ 0.30 mode. `gauss_fit_failed_post` (13 of 15 with `broad_ring`;
  the other two are 143 and 147-event flat rings) means `radiusStd` is a sample std. `center_outside_data`,
  `robust_refit_failed` and `geometric_refit_failed` do not occur with the defaults.
- With this split, Status mode shows 241 flagged channels instead of 1,299.

### 7.8 Geometric refinement (`--geometric`)

Compared with the algebraic robust fit, on the 5,874 channels without warning flags:

| | median | p95 | max |
|---|---:|---:|---:|
| centre shift (ADC) | 0.012 | 0.040 | 0.15 |
| √(ab) change (ADC) | −0.015 | −0.002 (p5 −0.086) | 0.12 |
| φ displacement (ADC at the ring) | 0.009 | 0.052 | 0.69 |
| post robust σ ratio | 1.000 | 1.004 | 1.035 |
| KS change | 7e-7 | 3e-5 | 0.0017 |

Larger changes happen only on non-rings. There, 18 `broad_ring` fits move to b/a < 0.5 (`extreme_axis_ratio` 9 → 27) and
`gauss_fit_failed_post` goes 15 → 7. The run took 72 s instead of 18.5 s in the same conditions (×3.9). **Keep it off.** It is immaterial on
real rings.

## 8. Performance (24 cores, WSL2)

| Step | Time | Notes |
|------|-----:|-------|
| Cache build (`uvcorr build-cache`) | 20.6-23 s | 3.48 GB `.dat` → 1.26 GB cache (lzf + shuffle), peak RSS 0.9 GB; the parse alone takes 11-16 s, depending on the run |
| `uvcorr process`, cache reused, 8 workers | 16 s quiet (worker scaling below); **17.3 s** analysis / 17.5 s total in the final run | The final run shared the machine with other jobs (load 4-6) |
| same with `--geometric` | 72 s | Measured with 18.5 s for the default run in the same conditions |
| RadialAnalysis `--csv-only`, 135 channels, 2.8 M events | 2.1 s | For scale |
| RadialAnalysis `--csv-only`, full file (6,557 `.uvd` files, 208 M events) | 1 min 54 s | Single-threaded, excluding the `.uvd` export |

Worker scaling of the analysis, i.e. the fitting step of `uvcorr process` and of the GUI's Fit All (`analyze_all` with the
default options, peak PSS of the whole process tree sampled every 50 ms): 1 / 4 / 8 / 12 / 16 workers take
109 / 29 / 16 / 12 / 11 s and use 0.5 / 1.1 / 1.8 / 2.4 / 3.0 GB, about 0.17 GB per worker. Measured on 2026-09-24 with one
fresh process per run, the cache file in the page cache and no other jobs on the machine (load average 0.6 before the
runs). An earlier run the same day was 5-16 % slower at 1-4 workers and within 4 % at 8-16 (114 / 33 / 16 / 12.5 / 11 s,
0.5 / 1.1 / 1.9 / 2.5 / 3.1 GB). Workers run with single-threaded BLAS, which is 15-18 % faster than OpenBLAS's default of
one thread per core. The default is min(8, usable CPUs).

## 9. Reproducing

- Cross-check: section 6.1. `compare_radial.py` exits 1 if an ellipse parameter of a ring channel or an event count
  disagrees. Add `--strict-blobs` to also fail on `broad_ring` channels, and `--tec` if the output directory holds several `.tec` files.
  `dump_uvd.py` refuses a non-empty output directory unless `--overwrite` is given (RadialAnalysis reads every `.uvd` file there).
- Census: `uvcorr process <dat> --output-dir out/` prints the status/flag census. The distributions, rejected-fraction
  table and subsampling study above were computed from `radial_summary.csv` and by re-fitting with
  `uvcorr.ellipse.fit_ellipse` under the stated options.
