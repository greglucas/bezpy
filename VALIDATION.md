# Method-comparison results

**What this is:** the evidence base for bezpy's method defaults.
Every option where past practice and a proposed default diverge is scored
against synthetic truth by `benchmarks/validation_harness.py` (built on
`bezpy.validation`), and where a synthetic result needs confirming against
real observatory/MT data, by `benchmarks/validation_harness_real.py`
(sections marked "real-storm"). This file records the current results;
re-run the relevant harness and update this file when the underlying code
changes.

**The synthetic truth** (`bezpy.validation.synthetic_scenario`): a two-layer
conductivity structure varying smoothly in space (surface resistivity,
basement contrast, and layer thickness all vary), driven by a spatially
uniform band-limited source (120 s – 8 h). Under uniform B the true E at any
point is an exact frequency-domain product with the local layered impedance,
and the true V along a line is a dense (1 km) local quadrature of that
field — no method assumptions in the reference.

Last run: 2026-07-09, seed 0, ntimes = 8192, dt = 60 s, 5×5 site grid,
three lines (~80 / ~500 / ~900 km).

## D6 — impedance band-edge handling (`extrapolate=`)

Empirical (band-limited) impedances built by sampling the true Z at 31
periods, E computed through the production path, scored against full-band
truth:

| impedance band | `extrapolate="zeros"` (legacy) | `extrapolate="asymptotic"` |
| --- | --- | --- |
| 300–7,200 s | 0.648 | **0.030** |
| 600–3,600 s | 0.819 | **0.066** |

(relative RMS E error at the sites)

**Reading:** the hard band-stop discards the out-of-band content the source
actually carries — here most of the signal variance — while the anchored
sqrt(f) continuation recovers it to a few percent even though the layered
truth does not follow the half-space asymptote exactly. This is the
short-line error mechanism identified in Kelbert & Lucas (2020) Table 3.
The real-data band (7–30,000 s vs a 1-minute Nyquist
of 120 s) is truncated far less severely than this stress test, so the
production-scale effect is smaller but the direction is unambiguous.

### D6 real-storm confirmation (2026-07-18)

The reference-storm comparison this section called for:
`benchmarks/validation_harness_real.py`, real EMTF sites in the docs'
example CONUS bbox (-110, 35, -100, 45; 342 sites), 5-station observatory
network (BOU/FRD/TUC/NEW/FRN), peak |E| per site under each option for the
2024-05 Gannon and 2003-10 Halloween storms:

| storm | n sites | ratio median | ratio p90 | ratio max | ratio min |
| --- | --- | --- | --- | --- | --- |
| 2024-05 Gannon | 342 | 1.009 | 1.061 | 7.01 | 0.876 |
| 2003-10 Halloween | 342 | 1.022 | 1.176 | 5.86 | 0.870 |

(ratio = peak |E| under `"asymptotic"` / peak |E| under `"zeros"`; no
non-finite values in either storm)

**Reading:** the real-data band is indeed truncated far less severely than
the synthetic stress test, as expected — the *typical* site barely moves
(median ~1-2%). But the distribution has real tails in both directions, and
they are explainable rather than numerical noise: the max-ratio sites in
both storms belong to a single local survey (`NCP19*`) whose transfer
functions were measured only out to ~1,190 s, well short of the network's
typical ~5-hour long-period limit — exactly the truncated-band regime D6's
synthetic test targets, and `"asymptotic"` recovers the storm-scale power
`"zeros"` was hard-discarding there. The min-ratio sites (`"asymptotic"`
~10-13% *below* `"zeros"`) have ordinary or long bands, so the small
decrease there is the expected phase-interference effect of a physically
continued (not just larger) out-of-band tail, not an artifact. No site in
either storm produced a NaN/Inf or a runaway ratio unexplained by its band.

**Default:** changed to `extrapolate="asymptotic"` (was `"zeros"`) --
`Site3d.calcZ`, `SiteCollection.convolve_fft`, `calc_geoelectric_field`, and
`TelluricResponse` all default to it now. The synthetic evidence was already
unambiguous on direction; this real-storm run confirms the production-scale
effect is small for a typically-banded site, large and correctly-directed
for a narrow-banded one, and never unstable. `extrapolate="zeros"` remains
available by name to reproduce pre-flip numbers. See `CHANGELOG.md` for the
result-changing note.

## D7 — E-interpolation scheme for line voltages (`how=`, `resample_km=`)

Voltages via each interpolation scheme vs the exact quadrature:

| how | resample_km | rel RMS voltage error |
| --- | --- | --- |
| nn | — | 0.040 |
| nn | 5 km | 0.050 |
| nn | 1 km | 0.049 |
| delaunay | — | 0.108 |
| delaunay | 5 km | 0.039 |
| delaunay | 1 km | **0.037** |

**Reading (revised 2026-07-09):** an earlier run of this table showed
nearest-neighbor beating Delaunay by ~3×. Most of that gap was a
longstanding bug, present since the original package: the gnomonic
projection used to build the Delaunay weights divided the line points by
cos(c) but not the sites, skewing the barycentric weights everywhere except
the projection center (fixed in `TransmissionLine.set_delaunay_weights`;
regression test `test_tl.py::test_delaunay_vertex_at_site_gets_full_weight`).
With the projection consistent, Delaunay blending plus build-time
resampling is the best scheme measured (0.037), and resampling now behaves
as theory predicts — it converges the quadrature toward the true line
integral instead of toward a bug. Note this bug was live in the published
delaunay-based workflows, so exact reproduction of paper voltages requires
the *old* weights; the difference is at the few-percent level.

**Default:** `how="delaunay"` retained, now with evidence rather than
tradition; pair it with `resample_km≈5` when building kernels. The
`secs+local-z` scheme (interpolate B, assign Z locally) may still improve
on it across sharp conductivity contrasts and lands with the multi-input
kernels.

### D7 extension (2026-07-12): accuracy vs truth + real-storm divergence

The table above aggregates a few long lines over a coarse (5×5) grid, where
every scheme scores well. A Minnesota Gannon-storm case study found
`delaunay` and `nn` line voltages differing by ~19% median (200% p99) on
real 3-D impedances — far above the "few percent" folklore — because
neighboring MT sites' peak |E| can differ 20× at ~43 km spacing. Two
questions follow: (a) *which scheme is right* when they disagree, and
(b) *when* do they diverge in practice. The synthetic truth answers (a); the
storm answers (b).

**Stress-test design** (`synth_interp.py`, seed 1, ntimes 4096, MN-scale
domain 44–49 °N × 89–97 °W): site density is swept 4×4 → 11×11 (median
spacing 185 → 56 km) and conductivity roughness `r` (full wavelengths of the
layered structure across the domain) is swept 1 → 4, so the ratio of site
spacing to field wavelength — the quantity that actually governs
interpolation error — is scanned from well-resolved to badly undersampled.
Every scheme is scored against the exact layered-Z field at ~80 held-out
interior points and along random lines binned by length.

**Held-out E interpolation, relative RMS error vs exact truth** (the "which
is right" arbiter — points, not integrated over a line):

| roughness | spacing | `delaunay` (gnomonic) | `delaunay` (plate-carrée) | `nn` |
| --- | --- | --- | --- | --- |
| r=1 (resolved) | 56 km | **0.045** | 0.101 | 0.183 |
| r=1 | 79 km | **0.129** | 0.222 | 0.300 |
| r=1 | 111 km | **0.237** | 0.362 | 0.357 |
| r=1 | 185 km | **0.453** | 0.502 | 0.517 |
| r=2 | 56 km | **0.262** | 0.373 | 0.460 |
| r=2 | 111 km | **0.539** | 0.554 | 0.749 |
| r=4 (undersampled) | 56 km | 0.572 | 0.576 | **0.511** |
| r=4 | 185 km | **0.784** | 0.784 | 0.844 |

Gnomonic-Delaunay is the most accurate scheme in **9 of 12** density×roughness
cells (plate-carrée 2, nn 1). Its worst-case (max relative) error is also the
smallest almost everywhere; `nn`'s max error runs 2–8× larger, the mechanism
behind the real-storm 200% p99. Two regimes:

* **Resolved / mildly rough** (spacing ≲ field wavelength): Delaunay blending
  is decisively better than nearest-site — up to ~4× lower error at tight
  spacing — and the margin over `nn` *grows as coverage thins*, exactly the
  hard case. This is the operational regime for real MT surveys.
* **Severely undersampled** (r=4, spacing ≫ wavelength): every scheme carries
  50–85% error and the choice is a wash (the one `nn` "win" lives here). No
  interpolation rescues a field sampled below its Nyquist — that is a
  data-density problem, not a scheme problem.

**Integrated line voltages, relative RMS error vs exact quadrature** (r=1,
averaged over site spacings; random lines per length bin):

| line length | `delaunay` (gnomonic) | `delaunay` (plate-carrée) | `nn` |
| --- | --- | --- | --- |
| short <20 km | **0.230** | 0.278 | 0.373 |
| ~40 km | **0.286** | 0.327 | 0.488 |
| ~120 km | **0.368** | 0.458 | 0.551 |
| ~300 km | 0.662 | **0.606** | 0.797 |

`nn` is the worst scheme in every length bin; the Delaunay family wins **29 of
32** length×density×roughness voltage cells (gnomonic 18, plate-carrée 11, nn
3). The Delaunay-over-`nn` margin again widens with spacing (voltage error
0.42 vs 0.68 at 185 km vs 0.35 vs 0.43 at 56 km). Very long lines average
pointwise error down and the two projections converge (plate-carrée edges
gnomonic only there).

**Projection (gnomonic vs plate-carrée).** The synthetic truth resolves the
known diagonal-flip sensitivity: gnomonic is the better-scoring projection for
held-out E in every resolved cell (often ~2× lower error) and wins most
voltage cells; the two projections only tie deep in the undersampled regime
where both are hopeless. They *disagree* with each other by a median 21% (p90
36%, max 47%) in synthetic line voltage — a real, truth-relevant gap, not a
rounding difference. **Gnomonic is the correct default**, confirmed against
truth, not just convention.

**Real-storm divergence** (`real_storm.py`, May 2024 Gannon, MN line set,
263 long-period MT sites after dropping 91 screening-flagged "bad" codes,
North-American SECS B; 1140 lines above 0.5 V):

| slice | `delaunay` vs `nn` peak-\|V\| (p50 / p90 / p99) |
| --- | --- |
| all lines | 18.8% / 58.3% / 166% |
| length 0–20 km | 22.7% / 61.5% / 200% |
| length 20–50 km | 14.9% / 51.8% / 86% |
| length 50–100 km | 14.2% / 48.7% / 65% |
| length 100–200 km | 6.5% / 56.9% / 63% |
| local spacing 0–20 km | 11.0% / 31.8% / 54% |
| local spacing 20–35 km | 20.8% / 60.5% / 113% |
| local spacing 35–50 km | 23.2% / 61.6% / 264% |
| local spacing 50–80 km | 39.3% / 88.5% / 152% |

Divergence is largest for **short lines** (they are near-point samples, so
voltage is set by which single site is picked) and for **sparse local
coverage** (median divergence climbs 11% → 39% as the nearest-site distance
grows 20 → 80 km) — the same spacing dependence the synthetic truth shows,
and the regime where the truth says Delaunay is the more accurate of the two.
Gnomonic-vs-plate-carrée peak-|V| differences on the same lines are 14.4% /
59.6% / 339% (p50/p90/p99): the projection choice moves voltages as much as
the interpolation scheme does.

**Hazard stakes.** Switching `delaunay → nn` shifts per-line peak voltage by a
signed median +5.8% (mean |shift| 27.6%); on the 810 hazard-relevant lines
(fitted V100 > 5 kV) the median peak-|V| move is 16.2%, and since the
100-year voltage is a monotone transform of the per-storm peaks, V100 carries
a comparable fractional shift — up to ~60% on individual lines. Choosing the
less accurate scheme (`nn`) therefore biases the reported 100-year hazard by
roughly its divergence, in the direction the truth says is wrong.

**Verdict / default (unchanged, now with truth behind it):** keep
`how="delaunay"` with the **gnomonic** projection. The synthetic truth ranks
gnomonic-Delaunay the most accurate scheme in every resolved regime for both
held-out E and line voltages, `nn` the least accurate, and plate-carrée a
measurably worse triangulation; the real-storm divergence tracks the same
spacing/length dependence. The one caveat is honest uncertainty, not a
different default: where MT coverage is much coarser than the field's spatial
scale (very sparse networks over strong 3-D structure), *all* schemes carry
50%+ error and the voltage should be reported with matching uncertainty rather
than trusted to any interpolation. Evidence: `synth_interp_Efield.csv`,
`synth_interp_voltage.csv`, `synth_interp_error_vs_length.png`,
`real_storm_divergence.csv` in `~/.cache/bezpy/hazard/`.

## 3.1 — SECS fitting method (`method=`, `robust=`)

pysecs 0.4 unblocked the temporal Kalman filter/smoother and robust IRLS
fitting. Scored by `bezpy.validation.score_fitting_methods`: a smooth
SEC-source truth at a 6×6 network, observations corrupted with ~40% white
noise, a 20-sample station dropout, and a persistent single-station
baseline jump; held-out prediction RMSE in nT (240 steps at 60 s,
tau = 600 s, gate 4σ):

| method | held-out RMSE (nT) |
| --- | --- |
| lstsq (default) | 1.315 |
| lstsq robust=huber | 1.309 |
| lstsq robust=bisquare | 1.341 |
| kalman | **0.946** |
| kalman gated (4σ) | 1.110 |

**Reading:** temporal coupling is worth ~30% here — the AR(1) prior
averages the noise across time steps the way independent snapshot fits
cannot, and it rides through the dropout. Robust IRLS is a wash on this
scenario: the corruption budget is dominated by dense white noise (which
IRLS rightly leaves alone), not by the one outlying station. Innovation
gating does not help against a *persistent* baseline jump — after the
first few steps the jump is absorbed into the forecast and stops looking
like an innovation; gating is for transient spikes.

### 3.1 real-storm measurement (2026-07-18)

The noise regime called for above: `benchmarks/validation_harness_real.py`,
leave-one-out over the same 5-station CONUS network (BOU/FRD/TUC/NEW/FRN)
on the 2024-05 Gannon and 2003-10 Halloween storms, B linearly detrended
per-window first (SECS models the disturbance field only, not each
station's very different core-field baseline -- comparing against raw B
would just measure that offset):

| storm | method | RMSE (nT) | MAE (nT) | correlation |
| --- | --- | --- | --- | --- |
| 2024-05 Gannon | lstsq (default) | 91.562 | 57.611 | 0.643 |
| 2024-05 Gannon | lstsq robust=huber | 94.907 | 60.324 | 0.567 |
| 2024-05 Gannon | kalman tau=3600 | 91.470 | 57.570 | 0.644 |
| 2003-10 Halloween | lstsq (default) | 72.561 | 45.435 | 0.605 |
| 2003-10 Halloween | lstsq robust=huber | 78.324 | 49.076 | 0.537 |
| 2003-10 Halloween | kalman tau=3600 | 71.951 | 45.266 | 0.611 |

epsilon/tau tuning curves (`SECSInterpolator.loo_score`, mean RMSE across
observatories): **epsilon is flat** across 0.01-0.2 on both storms (this
5-station network is sparse enough that the regularization strength in that
range never binds); **tau helps monotonically but marginally**, 91.56 nT
(tau=300s) -> 91.45 nT (tau=7200s) on Gannon, 72.12 -> 71.93 nT on
Halloween.

**Reading:** real observatory data is far cleaner than the synthetic stress
test (which added ~40% white noise plus a dropout plus a baseline jump by
construction) -- Kalman's real-storm win over lstsq is **<1%** here, not the
~30% the synthetic scenario showed. That scenario measured Kalman's *upper
bound* under heavy corruption, not what a real 1-minute CONUS network
delivers. `robust=huber` is not a wash on real data the way it was on
synthetic: it makes both storms measurably *worse* (+3.7% and +8.0% RMSE) --
real storm excursions are large, genuine signal that IRLS partly downweights
as if it were an outlier, exactly the failure mode the synthetic scenario's
single-outlying-station design could not expose.

**Default:** `method="lstsq"` retained -- now confirmed rather than assumed:
on real data the memory/compute cost of `method="kalman"` buys a real but
small win. The "use kalman for storm case studies" recommendation is
**tempered**, not withdrawn: reach for it on genuinely sparse/gappy networks
where the synthetic scenario's stress applies, tune `tau` with
`loo_score` on your own network rather than assuming a large win, and do
**not** reach for `robust=` in a storm case study -- it costs real skill
here. `smoother=False` remains the causal real-time recommendation
(unaffected by this measurement).

## 1.1 — Network GIC (Lehtinen-Pirjola solve)

`bezpy.tl.network.Network` (roadmap item 1.1): a nodal-admittance solve of
the DC-equivalent network (transmission lines + grounded transformer
windings + substation grounding) for the current through each transformer's
neutral -- the quantity NERC TPL-007 assessments need, one step past line
voltages. Composes with the existing Level-1 voltage kernel via
`gic_matrix`/`apply_gic_matrix`, the same build-once/apply-per-chunk
factorization as `voltage_matrix`.

**The oracle.** Horton, Boteler, Overbye, Pirjola & Dugan (2012)'s EPRI
21-bus GIC test system is the field's standard published benchmark, but the
paper itself does not tabulate machine-readable numbers. bezpy ships the
topology as package data (`Network.epri21_benchmark()`), sourced from
`lanl-ansi/PowerModelsGMD.jl`'s `test/data/matpower/epri21.m` (an
open-source, actively maintained GMD-modeling package built on the same
benchmark) rather than re-deriving it from the paper's figures. That
project also ships a separate, independently PowerWorld-computed
verification file for this exact topology
(`test/data/pw_csv/verification/EpriCase_Mar2012_GICXformer_1Vkm.csv`),
which is what bezpy's acceptance tests check against
(`tests/test_network.py`).

**Reconciling the two.** A first attempt at matching bezpy's solved
transformer currents against that verification data showed a mismatch --
but a *consistent* one: every single-grounded-winding (GSU) transformer's
current came out almost exactly 3x the verification value (2.995-2.996x
across three independent transformer pairs, bus6/bus12/bus17). This is not
a bug in either implementation: `Network` solves the single-line-equivalent
circuit (all three phases lumped into one DC-resistance/EMF pair per
branch, the standard simplification and the one the shipped topology data
itself uses), while NERC TPL-007 "effective GIC" is conventionally reported
*per phase*. Dividing bezpy's total by 3 matches the published per-phase
values to <0.2% for every GSU transformer -- strong, externally-verified
evidence the nodal-admittance solve (sign convention, KCL, Norton current
injection) is correct. The two-winding (`"wye-wye"`) and autotransformer
(`"auto"`) cases were not matched the same way, since NERC's "effective"
value for those combines hi/lo (or series/common) currents through a
turns-ratio-weighted formula (`effective_gic`, `I_eff = |a*I_high +
b*I_low|`) that was not independently reverse-engineered with the same
confidence; those are instead validated through exact internal invariants
that hold regardless of any external convention: KCL residual at machine
precision, identical parallel branches carrying exactly equal current, and
a transformer with both windings ungrounded (delta or ungrounded-wye)
carrying exactly zero current.

**What `Network` reports.** Given this, `Network.neutral_currents` (and
therefore `gic_matrix`) returns the **total single-line-equivalent
current**, not a per-phase value -- consistent with every other bezpy
electrical quantity being reported literally (`Ex`/`Ey` in mV/km, `V` in
volts) rather than pre-scaled by an implicit convention. Divide by 3 (or
apply your own transformer's actual per-phase model) if you need NERC's
per-phase reporting convention to compare against other tools.

**Coverage.** `tests/test_network.py`: a hand-verifiable two-substation
series-loop network (exact Ohm's-law match); the EPRI-21 benchmark's KCL
residual, symmetric-parallel-branch equality, isolated-winding zero-current
check, and the GSU/3 cross-check above; and
`test_gic_matrix_matches_direct_solve`, the linear-kernel-factorization
invariant every bezpy kernel relies on (`apply_gic_matrix(E, K)` must
exactly reproduce composing `voltage_matrix` and `Network.solve` directly).
`tests/test_pipeline.py::test_scan_network_reduces_gic` checks the `scan(...,
network=)` integration is chunk-invariant (D4) like `E` and `V`.

## D1 / D2 / D4 / D5 — covered by comparison tests

These decisions are exercised directly in the test suite rather than the
harness tables:

- **D1** gap masking vs interpolate-then-drop: `test_secs.py::test_gap_masking_beats_drop`
  (masking strictly better on synthetic gappy data; gap-free samples
  bit-identical to a full fit).
- **D2** linear detrend + Tukey taper vs bare median:
  `test_mag_processing.py::test_d2_linear_taper_beats_median_on_leakage`
  (<1% of the leaked broadband power on a trending record).
- **D4** continuous scan vs storm-window scan:
  `test_pipeline.py::test_scan_finds_index_invisible_event` (a local burst
  invisible to the index-selected window is caught by the continuous scan);
  chunk-size sensitivity decays with pad length
  (`test_scan_chunk_invariance`).
- **D5** GPD tail vs lognormal body fit:
  `test_stats.py::test_methods_diverge_on_heavy_tail` plus profile-CI
  coverage on simulation.

## Variometer despiking (`bezpy.mag.despike`, `despike_obs=`)

FDSN-distributed USGS/EarthScope variometer feeds carry instrumental
artifacts a boxcar minute-decimation passes straight through: isolated wild
samples, multi-minute saturation rails (values pinned near ±10^5 nT), and DC
baseline steps (instrument re-zeroing, and the ~1 Hz `LF`->10 Hz `BF` channel
switch around Dec-2024). Physically-impossible 1-minute rates (|dB/dt| up to
~3×10^5 nT/min, vs a real mid-latitude ceiling of a few hundred) appear at
essentially every station. Left in place they inject broadband high-frequency
energy — exactly the band that dominates the induced E field.

**Effect on hazard E (per-site max|E|, CONUS long-period sites, ratio of
network *with* variometers to observatory-only baseline A).** Adding *raw*
variometers should barely move E (they add stations to a consistent field);
in the clean LF era it does, but the glitchy BF era inflates it badly.
Despiking (iterated Hampel, `window="31min"`, `nsig=6`, `sig_floor=3` nT)
returns the ratio to ~1.0 — the variometers behave like the LF-era data again
— while leaving the clean LF era essentially untouched:

| Storm (era) | raw C/A median | raw p90 | raw max | despiked median | despiked p90 | despiked max |
|---|---|---|---|---|---|---|
| 2024-08 (LF control) | 0.88 | 1.23 | 4.20 | 1.04 | 1.67 | 3.42 |
| 2025-11 (BF) | 1.29 | 7.13 | 27.6 | 1.03 | 1.43 | 2.42 |
| 2026-01 (BF) | 1.37 | 6.50 | 47.5 | 0.91 | 1.14 | 1.65 |
| 2026-07 (BF) | 2.31 | 11.8 | 55.1 | 0.92 | 1.50 | 2.47 |

Validated against the co-located CMO observatory (College, AK): despiking
preserves the dB/dt correlation with the clean observatory (0.819) and the
real storm peak, confirming it removes artifacts, not signal. A pure DC step
(no accompanying spike) is intentionally *not* flagged — the variometer
baseline is arbitrary and removed per-window by `prepare`'s detrend; only an
analysis window straddling a step is affected, and the fix there is to mask
around it or split the window. Unit coverage:
`test_mag_processing.py::test_despike_*`,
`test_pipeline.py::test_despike_obs_recovers_field_from_spiked_observatory`.

## Still to add

- ~~Reference-storm comparison on real data (May 2024 or Halloween 2003)
  under all method combinations~~ -- done 2026-07-18, see the "D6 real-storm
  confirmation" and "3.1 real-storm measurement" subsections above
  (`benchmarks/validation_harness_real.py`).
- ~~`epsilon` and `tau` tuning curves via `SECSInterpolator.loo_score` on a
  real storm~~ -- done 2026-07-18, same run (3.1 real-storm measurement).
- Not measured: comparison against measured E (e.g. USGS geoelectric
  monitoring at BOU) -- no reader exists yet for that product, so the
  real-storm runs above use leave-one-out and cross-option comparison
  instead of an independent E ground truth. Worth adding alongside other
  `bezpy.data` source work (roadmap 3.2) if the product has a stable API.
- D7's `secs+local-z` scheme and the multi-input kernel work it depends on
  (roadmap 2.1) still only has the synthetic D7-extension evidence above.
