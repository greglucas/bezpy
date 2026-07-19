# Roadmap

Future directions for bezpy, written after the 0.5.0 rebuild. Each item is
scoped so an implementer can pick it up cold: what to build, where it lives,
how it connects to the existing API, and what "done" means. Priorities run
Tier 1 (highest value) to Tier 4 (project health).

Context an implementer needs first:

- Read `VALIDATION.md` — bezpy's method defaults are evidence-driven, and
  several items below extend that evidence base. New methods land as a named
  option scored against the harness (`benchmarks/validation_harness.py`,
  built on `bezpy.validation`), never as an unmeasured default swap.
- Settled design decisions (do not relitigate): xarray as the core data
  model; no geopandas dependency (Shapely + plain arrays only); the package
  scope is B → E → V hazard analysis — MT transfer-function *estimation*
  from raw time series is out of scope (that is aurora/resistics/mtpy
  territory; readers for their outputs are in scope).
- The architectural idiom throughout is *linear-map factorization*: anything
  linear in E (or B) gets contracted into a precomputed kernel once, then
  applied per chunk as a matmul (`bezpy.tl.kernels` is the template).
- Real-data testbed: a full archive exists in `~/.cache/bezpy` (26
  observatories 1991–present, 5,652 US/CA MT sites, a 75-storm hazard run)
  for anyone working on the validation items.

---

## Tier 1 — highest value

### 1.1 Network GIC: from line voltages to amperes per transformer

**The gap.** The package stops at open-circuit line voltages. The quantity
utilities and regulators actually assess (NERC TPL-007, transformer heating)
is GIC through each transformer neutral, which requires the DC network
solution: the Lehtinen–Pirjola (1985) / equivalent nodal-admittance method
over line resistances, transformer winding resistances, and substation
grounding resistances.

**Why it fits bezpy.** GIC is linear in the geoelectric field, exactly like
voltage. The network solve is one more constant matrix contracted onto the
existing Level-1 kernel: if `V = L E` (per-line voltages from
`voltage_matrix`) and the network response maps line-source voltages to
nodal/transformer currents `I = H V`, then `K = H L` is a per-transformer
GIC kernel with the same build-once / apply-per-chunk economics as
`apply_voltage_matrix`. It drops straight into `pipeline.scan` /
`storm_scan` reducers (a third reduced quantity, `I`, next to `E` and `V`)
and feeds `bezpy.stats.return_level` unchanged.

**Scope.**

- New module `bezpy/network.py` (or `bezpy/tl/network.py`):
  - `Network` class holding substations (id, lat/lon, grounding resistance),
    transformers (substation, winding resistance(s), type: GSU / auto /
    wye-wye, grounded or not), and branches (from/to substation, resistance,
    associated `TransmissionLine` geometry). Constructor from plain
    dicts/DataFrames; a `Network.from_json` for a documented file format.
    No new dependencies — the solve is a small sparse linear system
    (`scipy.sparse.linalg.spsolve`).
  - `network.gic_matrix(lines, n_sites, how="delaunay") -> (K, valid)`:
    composes the L-P solution with `voltage_matrix`, returning the
    (ntransformers × nsites·2) kernel.
  - `apply_gic_matrix(E, K, valid)` mirroring `apply_voltage_matrix`.
  - "Effective GIC" helper (`I_eff = |a·I_high + b·I_low|` per TPL-007) as a
    method, since that is the reported quantity.
- Pipeline integration: `scan(..., network=net)` adds an `I` reducer group
  (`I_max`, `I_max_time`, quantiles, exceedances) alongside `E`/`V`.
- Validation case: ship the Horton et al. (2012) EPRI 20-bus benchmark test
  system as package data with a test asserting the published GIC values per
  bus (that paper tabulates them for a uniform 1 V/km eastward and northward
  field — an exact, citable oracle). This is the acceptance test.
- Tutorial `docs/tutorials/network-gic.md`: benchmark system + a real storm
  B → E → V → GIC end to end.

**Deliberately out of scope:** transformer thermal response models, harmonics,
reactive power loss, AC load flow. Real network data is proprietary — bezpy
ships the format, the solver, and the benchmark, not utility data.

**References:** Lehtinen & Pirjola (1985, Ann. Geophys.); Horton, Boteler,
Overbye, Pirjola & Dugan (2012, IEEE Trans. Power Delivery,
doi:10.1109/TPWRD.2012.2206407); Boteler & Pirjola (2017, Space Weather,
equivalence of the two network formulations).

### 1.2 Uncertainty propagation onto E and V

**The gap.** `VALIDATION.md` (D7 extension) concludes that where MT coverage
is sparse relative to the field's spatial scale, "the voltage should be
reported with matching uncertainty rather than trusted to any
interpolation" — but the pipeline emits point estimates. All the inputs
already exist: `Z_var` on every `Site3d`, prediction variances from pysecs
0.4 in `SECSInterpolator`, and measured scheme divergence (delaunay-vs-nn
spreads in `~/.cache/bezpy/hazard/real_storm_divergence.csv`).

**Phase 1 — ensemble mode (cheap, honest, do this first). DONE (2026-07-19).**

- `calc_geoelectric_field(..., n_realizations=None)`: when set, perturb each
  site's impedance within `Z_var` (complex Gaussian, independent per period
  and component — document the independence assumption) and optionally B
  within the SECS prediction variance, producing a `realization` dimension
  on the output Dataset.
- `scan(..., n_realizations=)`: run reducers per realization; report
  `E_max` / `V_max` spread (e.g. `E_max_std` or per-realization maxima the
  caller can quantile). Memory cost is linear in realizations, so document
  the chunk-size interaction.
- Keep the RNG seedable end to end; a `seed=` kwarg threaded through.

Shipped as `Site.perturbed`/`Site3d.perturbed`, `calc_geoelectric_field(...,
n_realizations=, perturb_b=, seed=)`, and `scan(..., n_realizations=, seed=)`
(realizations fixed for the whole scan, not redrawn per chunk — see
`VALIDATION.md` 1.2 for why that matters). `scan`'s ensemble mode does not
yet support `reduce="exceedances"` (per-realization exceedance counts don't
stack on a shared dimension) or `out_path` resumability; `perturb_b` is
`calc_geoelectric_field`-only. `plot_hazard_map` gained
`relative_uncertainty=`/`uncertainty_limit=` fading. The calibration
acceptance test passes but documents Phase 1's ensemble as mildly
conservative (~0.60-0.65 observed 1σ coverage vs. nominal 0.68, traced to
the production impedance spline's automatic smoothing) — motivation for
Phase 2 below, not a blocker.

**Phase 2 — analytic variance through the linear chain. DONE (2026-07-19),
shipped with a documented calibration caveat.** E at a site is linear in Z
at fixed B, and V is linear in E, so first-order variance propagates in
closed form: `var(V) = Σ |L_j|² var(E_j)` plus the interpolation-scheme
term. Implemented as `calc_geoelectric_field(..., return_var=True)` /
`SiteCollection.convolve_fft(..., return_var=True)` (a numerical Jacobian
of the production `calcZ` pipeline, `Site3d.z_sensitivities`, propagated
through the same FFT convolution as the point estimate) and
`calc_voltages(..., return_var=True)` (the exact `Σ L_j² var(E_j)` step,
no approximation). Scoped to `bezpy.calc_voltages` (Level-1); the
Level-2 `TelluricResponse.calc_voltages(..., return_var=True)` originally
named here was **not** built — it would need the same per-site Jacobian
machinery threaded through `_build_tau`'s multi-site accumulation, a
bigger lift deferred given phase 2's calibration result below; compose
`calc_geoelectric_field(return_var=True)` + `calc_voltages(return_var=True)`
instead (mathematically equivalent under the uniform-B assumption the
module already documents). The delaunay-vs-nn structural-uncertainty term
was **not** auto-wired in either — it's analysis-specific (depends on your
own site geometry), so `calc_voltages`'s docstring documents it as a
manual add-on pointing at the D7 measured magnitude rather than baking in
a number from one specific validation run.

**Acceptance test result — did not hold.** The calibration test
(`bezpy.validation.score_analytic_calibration`, same truth-vs-measured
scenario as phase 1's) shows analytic coverage around **0.32-0.37** against
the nominal 68% — *more* conservative than phase 1's ensemble (~0.60-0.65),
not less, despite phase 2 being the "more principled" first-order method.
Root cause (see `VALIDATION.md` 1.2 phase 2 for the full write-up):
the Jacobian is frozen at the one noisy measured impedance's specific
FITPACK smoothing-spline knot configuration, and that local derivative is
a poor predictor of the gap to the (differently, exactly-interpolated)
truth — phase 1's ensemble avoids this because every Monte Carlo draw
actually refits the spline, sampling across knot configurations. This is
a property of linearizing a knot-adaptive smoothing spline at one point,
verified not to be an implementation bug (internal consistency against
local Monte Carlo redraws is good, ratio 0.75-1.03; an independent
hand-rolled finite-difference check matches exactly; the gap doesn't
shrink as `relative_std` → 0, ruling out a truncation/curvature story).
**Recommendation: prefer `n_realizations` for reported uncertainty;
`return_var` is a cheap, deterministic lower bound, not yet a calibrated
band.** Hazard-map fading (`bezpy.plot.plot_hazard_map`'s
`relative_uncertainty=`/`uncertainty_limit=`) already shipped in phase 1
and works with either method's output unchanged.

**Note:** this is per-site/per-line uncertainty propagation, *not*
observatory covariance infill — that direction was evaluated and rejected
(see memory/decision record: SECS wins, high-frequency band decorrelates).

### 1.3 Finish the validation backlog (gates default switches)

Already listed in `VALIDATION.md` "Still to add"; restated here so it's
scheduled, not just remembered:

- **Reference-storm comparison on real data** (May 2024 Gannon and/or
  Halloween 2003) across the method matrix: `extrapolate` zeros/asymptotic ×
  `interp` secs/uniform × SECS lstsq/kalman × delaunay/nn, scored against
  withheld observatories (`leave_one_out`) and, where available, measured E
  (USGS published geoelectric monitoring at BOU). Persist the results table
  in `VALIDATION.md` like the existing D6/D7 sections.
- **`epsilon` and `tau` tuning curves** via `SECSInterpolator.loo_score` on
  a real storm — the Kalman-vs-lstsq verdict is synthetic-only so far.
- **Decision point:** these results decide whether `extrapolate="asymptotic"`
  becomes the default (the synthetic evidence already points that way; D6)
  and whether `method="kalman"` is recommended-by-default for storm case
  studies. Flip defaults only with the table in hand, and record the change
  in `CHANGELOG.md` under a result-changing heading.

---

## Tier 2 — capability and reach

### 2.1 Multi-input telluric kernels (`secs+local-z`)

Trailed twice in `VALIDATION.md` (D7). Today `TelluricResponse` assumes
uniform B (single observatory, the Kelbert & Lucas 2020 setting). The
extension: fold the SECS interpolation into the kernel so τ becomes
per-line, per-*observatory* — `V(ω) = Σ_k [τ_x^k(ω) B_x^k(ω) + τ_y^k(ω)
B_y^k(ω)]` over observatories k. SECS is linear B_obs → B_site, so this is
another constant-matrix contraction (SECS transfer matrix × local Z ×
length vector); no new math, just plumbing and a storage format bump
(`tau.nc` gains an `obs` dimension; keep reading the old single-input files).

**Constant-matrix contraction: DONE (2026-07-19).** Shipped as
`SECSInterpolator.transfer_matrix(locations)` (`bezpy/mag/secs.py`) -- the
frequency- and time-independent obs→site B map, valid for `method="lstsq"`
fits with a single time-invariant observation-uncertainty pattern (no NaN
gaps, `weight_gaps=False`, `robust=None`; `method="kalman"` and gappy/robust
fits raise a clear `ValueError` instead of silently composing a wrong
answer -- they need `predict()` per sample, not a constant matrix). It
reaches into `pysecs.SECS`'s private `_VWU_patterns`/`_pattern_index`/
`_calc_T` (no public accessor exists in pysecs 0.4); accepted as consistent
with the existing `pysecs>=0.4` pin, with a regression test
(`test_transfer_matrix_matches_predict`) that would fail loudly if pysecs's
internals change. `TelluricResponse` (`bezpy/tl/kernels.py`) gains
`M=`/`secs=` on `__init__`/`from_lines`; `tau_master`, `_build_tau`,
`calc_voltages`, and `to_netcdf`/`from_netcdf` all grew an `obs` axis
additively (the single-input path is unchanged code, gated on `self.M is
None`), and old single-input `.nc` files keep loading exactly as before --
the array's own dimensionality (`tau_master.ndim`) is the format signal, no
version flag needed. Acceptance test passes exactly as specified: with one
observatory (identity `M`), results reproduce the classic single-input path
to rounding error (`test_telluric_multi_obs_identity_matches_single_input`);
an independent check composing `SECSInterpolator.predict()` sample-by-sample
with the *unmodified* single-input kernel also matches the new multi-input
path to rounding error
(`test_telluric_multi_obs_matches_secs_plus_single_input`), proving the
`_build_tau` einsum against ground truth rather than just the identity
shortcut.

**Not built in this pass (follow-on):**

- The `how="secs+local-z"` **Level-1** scheme (SECS-predicted B at arbitrary
  points along a line's own geometry + nearest-site Z lookup) and its
  `score_interpolation_methods`/D7 harness column. This is a genuinely
  separate code path from the Level-2 τ-folding above -- no line-geometry
  quadrature work landed here, only the observatory-to-site kernel
  contraction. Still needed before any `secs+local-z` default claim.
- `pipeline.calc_geoelectric_field`/`scan` integration of the multi-input
  `TelluricResponse` as a fast path -- anticipated by item 2.2's sequencing
  note ("2.2 benefits from 2.1's per-observatory τ"), not done here.
- `return_var=True` on the multi-obs kernel (1.2 phase 2 already deferred
  this for the single-input case).
- A "recompose per-sample" fallback for gappy/Kalman/robust SECS fits that
  don't reduce to a constant matrix -- `transfer_matrix()` raises clearly
  instead of attempting one.

### 2.2 Operational / real-time mode

The causal pieces exist (`SECSInterpolator(smoother=False)` for filtering,
`bezpy.mt.impulse.DTIR` for time-domain convolution, `bezpy.mag.despike`,
`bezpy.data.update_observatory`); what's missing is the composition. There
is a real consumer (the swx-trec live geoelectric map runs on bezpy).

- `bezpy.pipeline.Nowcast` (or `nowcast()`): holds a fitted
  `SECSInterpolator`, a `SiteCollection` (or `TelluricResponse`), and a
  rolling B buffer sized by the impedance pad; an `update(new_b)` method
  appends samples and returns E/V for the newly valid interval only —
  no window recomputation. Internally this is DTIR (causal FIR) rather than
  FFT-over-window; the existing `impulse.py` machinery is the engine.
- CLI entry points via `pyproject.toml [project.scripts]`: `bezpy download`,
  `bezpy update`, `bezpy scan`, `bezpy nowcast` — thin argparse wrappers
  over the existing API so operational users don't write Python. The
  existing `scripts/` examples show the patterns to wrap.
- Acceptance: a test asserting the streaming path matches the batch
  `calc_geoelectric_field` result on the interior of a window (edge samples
  differ by the causal-vs-acausal taper, which should be documented), and a
  docs page on operational deployment.
- Follow-on (separate PR): 1-second data support end to end (the archive
  format already carries `sampling_period`; audit `prepare`/pad assumptions
  for dt ≠ 60 s — much of it is already dt-general).

**Engine + CLI: DONE (2026-07-19).** Shipped as `bezpy.pipeline.Nowcast`
(`bezpy/pipeline.py`) and `bezpy download`/`update`/`scan`/`nowcast`
(`bezpy/cli.py`, `pyproject.toml [project.scripts]`). `Nowcast` composes
`DTIR` (causal FIR) with `SECSInterpolator.transfer_matrix` (roadmap 2.1: a
constant, per-sample obs->site B matrix) -- `update(new_b)` appends to a
bounded rolling buffer and returns only the newly-causally-valid E (and V,
via the static Level-1 `voltage_matrix`/`apply_voltage_matrix`, no new math).
Two design points settled with real evidence, not assumption, since this
item had no precedent in the codebase to copy:

- **The buffer/finalization algorithm was verified against a whole-array
  `DTIR.convolve()` reference to float64 noise (6.6e-16), and the minimum
  buffer length before *any* output can appear is `2*max(nmax,|nmin|)+1`,
  not `nmax+|nmin|`** -- `np.convolve(..., mode="same")` silently returns a
  *different-length* array below that floor, which corrupted an early draft
  of the indexing before this was caught. `Nowcast.warmup_samples` documents
  the corrected value.
- **DTIR reconstruction fidelity depends on how spectrally broadband the
  input is, not just on periods being well-sampled relative to `dt`**:
  measured ~5% relative RMS vs `calc_geoelectric_field` for a 15-tone
  broadband synthetic (storm-like), but 10-50% for the *same* well-sampled
  period band driven by just 1-2 dominant tones, at the same
  `decay_factor` -- the regularized inversion's per-frequency error doesn't
  average away when only a few frequencies carry the signal's energy. Real
  storm B is broadband, which favors real operational use, but a narrowband
  bench test of this class understates error and should not be read as
  general accuracy. Separately (and independently confirmed), periods
  closer than ~10x `dt` are an aliasing failure no `decay_factor`/`Q_choice`
  tuning fixes (measured 50-65% relative RMS at that ratio) --
  `nmin`/`nmax="auto"` warns when a site's band triggers this. See
  `Nowcast`'s docstring "Accuracy" section and `docs/tutorials/operational.md`
  for the full numbers and a worked latency example (~20 hours startup
  transient, ~2 minute steady-state lag, for a representative 900-18000s
  band at `dt=60`\ s).
- SECS interpolation defaults to `method="lstsq"` +
  `SECSInterpolator.transfer_matrix` (one constant matrix, computed once,
  applied as a matmul every `update()` -- zero added latency beyond DTIR's
  own); falls back to a per-call `fit()`/`predict()` over the rolling buffer
  when a chunk has a gap or the caller configured `method="kalman"` (both
  paths tested; verified matching to `rtol=1e-9` on gap-free input).

**Not built in this pass (follow-on):**

- `despike_obs=` support -- despike's centered rolling window needs its own
  future-sample lookahead on the raw observatory stream, a second instance
  of the buffering problem this item solves for DTIR; deferred rather than
  bolted on without the same rigor. Callers run `bezpy.mag.despike`
  themselves per chunk before calling `update()` for now.
- Per-site `(nmin, nmax)` tuning -- one shared horizon across all sites in a
  `Nowcast` (matching `calc_geoelectric_field`'s own `pad="auto"`
  precedent); a `dtirs=` escape hatch allows fully custom per-site `DTIR`s
  for advanced callers willing to share one `(dt, nmin, nmax)`.
- 1-second data support end to end (unchanged from below).

### 2.3 Parallel `scan`

`scan`'s chunks are independent by construction (pad-protected edges make
results chunk-invariant — see `test_pipeline.py::test_scan_chunk_invariance`),
but the loop is sequential. Best effort-to-value item on this list.

- `scan(..., workers=None)`: `concurrent.futures.ProcessPoolExecutor` over
  chunks; each worker returns its block reductions; reducers merge in the
  parent (max/argmax merge trivially; histogram sketches add; exceedance
  lists concatenate). No dask dependency.
- Watch items: `SiteCollection` and its interpolated-impedance cache must be
  picklable or rebuilt per worker (rebuild is fine — cost is per-worker,
  once); resumable `out_path` writes stay in the parent only; keep
  `workers=None` (current sequential behavior) the default.
- Acceptance: bit-identical results vs sequential on a two-year synthetic
  record with `workers=4` (the chunk-invariance property makes this a hard
  equality test, not a tolerance test).

---

## Tier 3 — new domains and data

### 3.1 Pipelines (and rails): distributed-source transmission line theory

Pipe-to-soil potentials are the second-largest GIC application. This is a
real modeling addition, not a rename: buried pipelines are *continuously
grounded* conductors, so the open-circuit line integral in `bezpy.tl` does
not apply. The standard treatment is distributed-source transmission line
(DSTL) theory (Boteler 1997/2013): per-unit-length series impedance and
parallel admittance from pipe steel + coating, characteristic impedance and
propagation constant, E projected along the route as the distributed source.

- New `bezpy/tl/pipeline_dstl.py` (name it to avoid clashing with
  `bezpy.pipeline`): `Pipeline` object (route geometry like
  `TransmissionLine`, plus diameter, wall thickness, coating resistance,
  steel resistivity, burial soil resistivity) and a solver returning
  pipe-to-soil potential along the route. Handle junctions/terminations as
  boundary conditions; a network of pipe segments reuses the nodal machinery
  from item 1.1.
- The DSTL response is again linear in E → same kernel factorization,
  per-node PSP kernels.
- Validation oracle: Boteler (1997) / Pulkkinen et al. (2001) published
  Finnish-pipeline test cases with tabulated PSP profiles.

### 3.2 More regions and MT sources

The `bezpy.data` source architecture (see `_wdc.py`, `_intermagnet.py` for
the plugin pattern) is ready for more sources:

- **AusLAMP** (Australian national MT survey) — transfer functions are
  published as EDI files via AuScope; `read_edi` already exists, so this is
  a search/download/ingest source (`search_mt(source="auslamp")`) plus
  survey-layout mapping into the `<cache>/mt/<survey>/` scheme.
- **MTH5/aurora interop** — a reader mapping an MTH5 transfer-function
  container to `Site3d` (`bezpy.mt.read_mth5`, optional `mth5` extra).
  Reading only; estimation stays out of scope.
- The Fernberg 1-D physiographic-region models already ship in
  `bezpy/mt/data_1d/` and cover the no-MT fallback; a docs example showing
  1-D fallback outside MT coverage would surface a capability that already
  exists.

### 3.3 Hazard-statistics depth

- **Nonstationarity:** exceedance rates are solar-cycle modulated, so
  `years_observed` over partial cycles biases the GPD rate term. Add a
  diagnostic (`bezpy.stats.rate_stability`: exceedance counts per year /
  per cycle phase with a dispersion test) before attempting any
  nonstationary model; the diagnostic alone tells users whether their span
  is long enough. A full nonstationary GPD (rate as a function of cycle
  phase) is research-grade — prototype in `benchmarks/` first.
- **Regional pooling of the GPD shape:** per-site tail fits with a dozen
  exceedances have wild shape estimates; `shape_prior_scale` regularizes
  toward zero by hand. The principled version pools the shape across sites
  (fit a common or spatially smooth ξ, per-site scale/rate) — a
  `return_level_pooled(events_by_site, ...)` API. Validate by simulation:
  coverage of the pooled CI vs per-site MLE on synthetic heavy-tailed data.
- **Declustering sensitivity:** a small utility reporting how
  `find_events(gap=...)` choices move the resulting return level (sweep gap,
  refit, tabulate) — cheap, and reviewers of hazard studies always ask.

---

## Tier 4 — adoption infrastructure

- **conda-forge feedstock** — much of this community installs via conda.
  Straightforward once 0.5.0 is on PyPI (grayskull + staged-recipes PR;
  optional deps map to a single package with extras documented).
- **JOSS paper** — the rebuild plus the validation harness clears JOSS's
  bar and gives users a modern citation target alongside the 2019/2020
  science papers. The paper is ~1000 words; `VALIDATION.md` and the docs
  already contain the substance. Update `CITATION.cff` when accepted.
- **Docs gallery** — the tutorials are prose-with-code; a thumbnail gallery
  page (each tutorial's headline figure, linked) is the highest-leverage
  docs improvement for discoverability. mkdocs-material handles this with a
  simple grid; no new tooling needed.
- **CONTRIBUTING.md + issue templates** — test/lint commands exist in the
  README; add expectations for new methods (must come with a validation
  harness entry — make the `VALIDATION.md` discipline explicit policy).
- **API stability statement** — a short docs section declaring what is
  stable at 1.0 (the top-level pipeline API, data formats on disk) vs
  internal (`_`-prefixed modules), so downstream users (swx-trec) know what
  they can pin against.

---

## Explicit non-goals

Recorded so future contributors don't re-open settled questions:

- **MT transfer-function estimation from raw time series** — aurora /
  resistics / mtpy territory. Interop readers yes, estimation no.
- **Geopandas dependency** — Shapely + arrays cover the need
  (`filter_bbox`/`filter_polygon` exist).
- **Covariance-based observatory gap infill** (MICE-style) — evaluated and
  rejected: SECS handles spatial infill better, and the high-frequency band
  that dominates E decorrelates between observatories.
- **3-D EM forward modeling** — bezpy consumes gridded impedance products
  (`read_gridded_impedance`); producing them is a different package.
- **Transformer thermal/harmonic impact modeling** — item 1.1 stops at GIC
  amperes (and TPL-007 effective GIC), the interface quantity to power
  engineering tools.

## Suggested sequencing

1. **2.3 parallel scan** — small, isolated, immediate payoff for every user.
2. **1.3 validation backlog** — settles deferred defaults before new
   features widen the method matrix.
3. **1.1 network GIC** — the flagship feature; benchmark oracle makes it
   well-bounded.
4. **1.2 uncertainty (phase 1 ensemble)** — rides on whatever 1.3 learned.
5. **2.1 multi-input kernels**, then **2.2 operational mode** (2.2 benefits
   from 2.1's per-observatory τ).
6. Tier 3/4 items as demand appears; conda-forge and JOSS as soon as the
   release is cut.
