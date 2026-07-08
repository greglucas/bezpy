# Method-comparison results

**What this is:** the evidence base for bezpy's method defaults (PLAN.md §2).
Every option where past practice and a proposed default diverge is scored
against synthetic truth by `benchmarks/validation_harness.py` (built on
`bezpy.validation`); this file records the current results. Re-run the
harness and update this file when the relevant code changes.

**The synthetic truth** (`bezpy.validation.synthetic_scenario`): a two-layer
conductivity structure varying smoothly in space (surface resistivity,
basement contrast, and layer thickness all vary), driven by a spatially
uniform band-limited source (120 s – 8 h). Under uniform B the true E at any
point is an exact frequency-domain product with the local layered impedance,
and the true V along a line is a dense (1 km) local quadrature of that
field — no method assumptions in the reference.

Last run: 2026-07-08, seed 0, ntimes = 8192, dt = 60 s, 5×5 site grid,
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
short-line error mechanism identified in Kelbert & Lucas (2020) Table 3
(ARCHITECTURE.md §8). The real-data band (7–30,000 s vs a 1-minute Nyquist
of 120 s) is truncated far less severely than this stress test, so the
production-scale effect is smaller but the direction is unambiguous.

**Default:** unchanged (`"zeros"`) until a reference-storm comparison on
real EMTF impedances is added; the harness evidence supports switching for
dB/dt-dominated applications. Use `extrapolate="asymptotic"` explicitly.

## D7 — E-interpolation scheme for line voltages (`how=`, `resample_km=`)

Voltages via each interpolation scheme vs the exact quadrature:

| how | resample_km | rel RMS voltage error |
| --- | --- | --- |
| nn | — | **0.040** |
| nn | 5 km | 0.050 |
| nn | 1 km | 0.049 |
| delaunay | — | 0.154 |
| delaunay | 5 km | 0.135 |
| delaunay | 1 km | 0.135 |

**Reading:** on this smoothly-varying structure, nearest-neighbor (local
assignment) beats Delaunay blending of E by ~3× — consistent with
THEORY.md §2's argument that E is a *local* response and blending it across
conductivity contrasts is the wrong smoothness assumption. Build-time
resampling converges each scheme to its own ideal (it is not guaranteed to
reduce error against truth — for `nn` the coarse quadrature was accidentally
closer here).

**Default:** `how="delaunay"` retained for now (single synthetic scenario;
`nn` produces discontinuous E maps and the paper workflows are
delaunay-based), but this table is the standing challenge to it. The
`secs+local-z` scheme (interpolate B, assign Z locally) is the anticipated
winner and lands with the multi-input kernels.

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

## Still to add

- Reference-storm comparison on real data (May 2024 or Halloween 2003)
  under all method combinations, with a persisted results table.
- `epsilon` tuning curves via `SECSInterpolator.loo_score` on a real storm.
- Kalman vs lstsq SECS fitting, when pysecs ships KalmanSECS.
