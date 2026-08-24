# Changelog

## 0.5.0 (2026-08-23)

The first release of the rebuilt package: bezpy grows from a collection of
per-paper utilities into a full B → E → V → I hazard-analysis pipeline with
an evidence-driven validation story. See `docs/validation.md` for the
method-comparison results behind the defaults.

### Result-changing fixes (read before reproducing published numbers)

- **Delaunay interpolation weights**: the gnomonic projection was applied to
  transmission-line points but not site points when building Delaunay weights
  (line points were divided by cos(c), sites were not), skewing the weights
  since the original package. Voltages computed with `how="delaunay"` change
  by a few percent relative to the published 2019/2020 workflows. This was a
  bug, not a method, so there is **no legacy switch**; use the git history if
  you need the old numbers exactly.
- **EMTF sign convention**: `read_xml` now conjugates impedances from files
  declaring the `exp(-i omega t)` convention, so every `Site3d` carries the
  package's `exp(+i omega t)` convention. Previously such files silently
  produced time-reversed impulse responses.
- **Impedance spline weighting**: periods with non-finite or non-positive `Z_var` are
  now excluded from the interpolating spline. Previously a single NaN
  variance (e.g. from a gridded-product row without an error column) made an
  entire impedance component — and every E field computed from it — NaN.
- **Unknown impedance uncertainty**: `apparent_resistivity(..., Z_var=None)`
  now returns NaN uncertainty arrays instead of fabricating a finite unit
  error. Central apparent resistivity/phase values are unchanged, but code
  that treated the old synthetic error as measured uncertainty must be
  updated.
- **Legacy EDI uncertainty rotation**: independent impedance/tipper variances
  are now rotated with squared tensor coefficients instead of rotating
  standard errors as if they were field values. EDI files declaring the
  opposite Fourier sign are conjugated into bezpy's convention on read.
- **`leave_one_out` band coherence** was computed on an interleaved Bx/By
  sequence and is now computed per component and averaged. Coherence-based
  skill scores from earlier runs are unreliable; RMSE/MAE/correlation scores
  were always correct.
- **`extrapolate` default changed `"zeros"` -> `"asymptotic"`** (not a bug --
  a deferred default settled with evidence; see `docs/validation.md` D6 and its
  "D6 real-storm confirmation"). `Site3d.calcZ`, `SiteCollection.convolve_fft`,
  `calc_geoelectric_field`, and `TelluricResponse` all switch: impedances are
  now continued past the measured period band with the physical half-space
  sqrt(f) asymptote instead of hard-zeroed. E computed from `Site3d`/gridded
  (band-limited) impedances without an explicit `extrapolate=` will change --
  typically a few percent, more for narrow-banded sites/surveys where the
  measured band falls well short of the storm timescale. Pass
  `extrapolate="zeros"` explicitly to reproduce old numbers; `Site1d`
  (physical, always-valid) sites are unaffected.

### New

- `bezpy.tpl007` — NERC TPL-007 compliance quantities from measured
  conductivity: `beta_factor` (the Earth-conductivity scaling factor from a
  3-D impedance, replacing the standard's tabulated 1-D physiographic
  value), `alpha_factor` (eq. 3 or Table 2 — they disagree by up to 0.026),
  `peak_geoelectric_field` (eq. 1/2), `tabulated_beta` (Table 3 lookup, for
  comparison against the computed value), and the `QUEBEC_REFERENCE_MODEL`
  the standard normalizes against. `beta` is computed as a ratio against
  that reference model driven by the same waveform rather than a literal
  `E / 8`, which is exact by the definition of the reference waveform,
  insensitive to rescaling it, and turns a wrong waveform into a warning
  instead of a silently wrong `beta`. NERC's reference waveform file is not
  redistributed; the caller supplies it. Validated against the standard's
  own Table 3 — see `docs/validation.md` item 1.3, **including the residual
  bias that is not yet closed**.
- `bezpy.calc_substation_fields` — interpolate the site geoelectric field
  onto arbitrary point locations (substations), using the same gnomonic
  Delaunay weights as the voltage kernels, so the two agree exactly.
  Locations outside the MT network's convex hull return NaN and are flagged
  in a `within_hull` coordinate rather than silently extrapolated.
- `bezpy.tl.utils.delaunay_weights` — the projection-plus-triangulation step
  shared by `TransmissionLine.set_delaunay_weights` and
  `calc_substation_fields`, so there is one place that both point sets go
  through the same gnomonic projection (the pre-0.5.0 bug was doing it to
  only one of them).
- `bezpy.mag.SECSInterpolator` — fit/predict SECS interpolation (pysecs 0.4:
  Kalman smoothing, robust fits, prediction variances), per-sample gap
  masking instead of interpolate-then-drop, `predict_grid`, `loo_score`.
- `bezpy.mag.combine_observatories` / `prepare` / `trim_pad` — shared B
  preprocessing (linear detrend + Tukey taper over an impedance-derived pad).
- `bezpy.pipeline` — `calc_geoelectric_field`, top-level `calc_voltages`,
  chunked continuous `scan` with streaming reducers (resumable netCDF), and
  `storm_scan` for catalog windows. `scan(..., workers=)` parallelizes
  independent chunks across processes; results (and resumable checkpoints)
  are bit-identical to the sequential default.
- Uncertainty propagation, phase 1: `calc_geoelectric_field(...,
  n_realizations=)` and `scan(..., n_realizations=)` redraw each site's `Z`
  within its reported `Z_var` (`Site3d.perturbed`) and rerun the chain per
  draw, adding a `realization` dimension (`scan` also gains `*_max_mean`/
  `*_max_std` convenience variables). `calc_geoelectric_field(...,
  perturb_b=True)` additionally propagates SECS prediction variance. See
  `docs/validation.md` "1.2 -- Uncertainty propagation, phase 1" for the
  calibration test and its documented mild, smoothing-spline-driven
  undercoverage.
- Uncertainty propagation, phase 2: `calc_geoelectric_field(...,
  return_var=True)` / `SiteCollection.convolve_fft(..., return_var=True)`
  propagate `Z_var` analytically (`Site3d.z_sensitivities`, a numerical
  Jacobian of the production `calcZ` pipeline) instead of by Monte Carlo
  redraw, and `calc_voltages(..., return_var=True)` applies the exact
  `var(V) = sum L_j^2 var(E_j)` step. **Read `docs/validation.md` "1.2 --
  Uncertainty propagation, phase 2" before trusting the numbers**: its
  calibration test shows ~0.32-0.37 coverage@1σ against nominal 0.68 --
  substantially more *anti-conservative* (too narrow/overconfident) than
  phase 1's ensemble because the Jacobian is frozen at one noisy
  realization's smoothing-spline knot configuration. Prefer
  `n_realizations` for reported uncertainty; treat `return_var`'s band as a
  lower bound for now.
- `bezpy.tl.network` — `Network`: a Lehtinen-Pirjola (nodal-admittance) GIC
  solve from transmission lines, grounded transformer windings, and
  substation grounding to per-transformer neutral current, the quantity
  NERC TPL-007 assessments need. `gic_matrix`/`apply_gic_matrix` compose it
  with the Level-1 voltage kernel (same build-once/apply-per-chunk
  factorization as `voltage_matrix`); `effective_gic` implements the
  TPL-007 `|a*I_high + b*I_low|` combination. `scan(..., network=)` adds an
  `I` reducer group next to `E`/`V`. Ships the Horton et al. (2012) EPRI
  21-bus GIC test system as package data
  (`Network.epri21_benchmark()`) with its own cross-tool validation (see
  `docs/validation.md` 1.1).
- `bezpy.storms` — `find_storms` (index catalog) and `find_events`
  (threshold + runs declustering on any series).
- `bezpy.stats.return_level` — GPD/POT with profile-likelihood or bootstrap
  CIs; `lognormal-2019` and `gev` by name. CIs are mandatory in the output.
- `bezpy.validation` — leave-one-out skill scores and a synthetic-truth
  benchmark that arbitrates method defaults; `benchmarks/validation_harness_real.py`
  extends the same discipline to real observatory/MT data (reference-storm
  method comparison, epsilon/tau tuning) where synthetic truth alone can't
  settle a default.
- `bezpy.tl.kernels` — Level-1 voltage functionals and Level-2 power-line
  telluric response functions (Kelbert & Lucas 2020), with netCDF round-trip
  and FFT-grid caching; `TransmissionLine.resampled` great-circle
  densification; `filter_bbox`/`filter_polygon` (no geopandas).
- Multi-input telluric kernels: `SECSInterpolator.transfer_matrix`
  exposes the constant, frequency- and time-independent observatory-to-site B
  map that a `method="lstsq"` fit with a time-invariant uncertainty pattern
  reduces to (raises clearly for `method="kalman"` or a gappy/robust fit,
  which don't reduce to one matrix). `TelluricResponse.from_lines(...,
  secs=)`/`M=` folds it into tau, which gains an `obs` axis, so line voltages
  come from several observatories' B directly (`V(ω) = Σ_k [τ_x^k(ω) Bx^k(ω)
  + τ_y^k(ω) By^k(ω)]`); the single-input path and old single-input `.nc`
  files are unaffected. The Level-1 `secs+local-z` E-interpolation scheme and
  full `pipeline` integration remain future extensions.
- Operational/real-time mode: `bezpy.pipeline.Nowcast` composes
  `bezpy.mt.impulse.DTIR` (causal FIR) with `SECSInterpolator.transfer_matrix`
  into a stateful engine -- `update(new_b)` appends B to a bounded rolling
  buffer and returns only the newly causally-valid E (and V, via the static
  Level-1 voltage kernel). Defaults to a fast, zero-added-latency SECS path
  (`method="lstsq"` + `transfer_matrix`, one constant matrix applied as a
  matmul per call); falls back to a per-call fit over the buffer for gappy
  chunks or `method="kalman"`. `bezpy download`/`update`/`scan`/`nowcast`
  (`bezpy/cli.py`, `pyproject.toml [project.scripts]`) wrap the data,
  scan, and nowcast APIs for operational use without writing Python. See
  `docs/tutorials/operational.md` for the measured
  latency (~20h startup transient + `|nmin|`-sample steady-state lag for a
  representative band) and accuracy (DTIR reconstructs broadband/storm-like
  B to ~5% relative RMS vs the batch FFT path, but narrowband B far worse
  -- 10-50% -- even within an otherwise well-sampled period band) numbers;
  `despike_obs=` is not supported by `Nowcast`.
- `bezpy.data` — observatory archive (USGS/NRCan/INTERMAGNET, provenance-
  and level-aware merge), activity indices (Kp/ap/Dst), EMTF XML archive with
  SPUD download, HIFLD transmission-line download, IAGA ingest.
- `bezpy.data` `source="wdc"` — the BGS World Data Centre archive, for
  definitive minute/hourly records reaching back before the real-time
  services (e.g. Kakioka to the 1910s).
- `bezpy.data.search_mt` / `download_mt_search` — search the SPUD EMTF
  archive by region/survey/quality and download every match in one call.
- `bezpy.mag.despike` — iterated Hampel (rolling median/MAD) masking of
  instrumental spikes, saturation rails, and step artifacts, exposed as
  `despike_obs=` on `calc_geoelectric_field`/`scan`/`storm_scan`; built for
  FDSN variometer inputs.
- `bezpy.mt.screen_impedance` / `screen_sites` — physical-plausibility and
  spatial-outlier screening of MT transfer functions before hazard use.
- `bezpy.stats.return_level(..., shape_prior_scale=)` — optional Gaussian
  shape prior (penalized/MAP GPD fit) that stabilizes small-sample tail
  estimates and their profile CIs.
- `bezpy.mt` — legacy readers (`read_zfile`, `read_edi`), ModEM/gridded
  impedance reader (`read_gridded_impedance`), canonical xarray form for
  site collections (`to_dataset`/`from_netcdf`), `extrapolate="asymptotic"`
  band-edge continuation, per-site fields in `SiteCollection.convolve_fft`.
- `bezpy.plot` — map helpers (cartopy optional); `plot_hazard_map` accepts
  `vmin`/`vmax` (saturating log color limits) or a full `norm` override, and
  `relative_uncertainty=`/`uncertainty_limit=` to fade sites whose ensemble
  spread exceeds a limit.
- Runnable example scripts (`scripts/mt_site_example.py`,
  `scripts/japan_halloween.py`, `scripts/china_may2024.py`) replace the old
  Jupyter notebooks and include an open, checksum-verified China MT dataset;
  the documentation and scripts are included in the source distribution.
- `bezpy.data.download_zenodo_file` downloads named research artifacts using
  live record metadata and refuses to cache content that fails its repository
  checksum; `bezpy.mt.read_zfile_archive` reads a ZIP of EMTF Z-files without
  extracting it.
- Documentation site (mkdocs-material) with tutorials; CI (lint, 3.10–3.14,
  oldest-pins job).

### Fixed

- `prepare` (and therefore `calc_geoelectric_field`) now raises an actionable
  error when the record is shorter than twice the pad instead of silently
  returning an empty result.
- `SiteCollection.convolve_fft` and `TelluricResponse.calc_voltages` reject
  NaN magnetic-field input with guidance instead of silently producing NaN
  everywhere; `convolve_fft` processes sites in memory-bounded blocks.
- `DTIR.calculate(periods=..., Z=...)` works as documented and tolerates
  sites without impedance variances or component-specific missing variances;
  the data-space path uses a pseudo-inverse. Equal-horizon DTIRs share their
  regularization matrix/pseudo-inverse, removing repeated cubic-cost setup in
  multi-site nowcasts.
- Resumable scans fingerprint their scientific configuration and refuse to
  combine incompatible inputs/options; checkpoints and magnetic archive
  updates are written atomically. Scan outputs carry stable `line_id` and
  `transformer_id` coordinates. Histogram-sketch overflow is retained and
  signaled instead of silently biasing approximate quantiles. The checkpoint
  schema and quantile-sketch layout are part of the fingerprint, preventing a
  package upgrade from mixing incompatible reducer state.
- The nowcast CLI tracks consumed magnetic input independently of finalized
  output, removes inclusive archive-boundary duplicates, and bounds startup
  history with `--startup-window`; it also reads the last durable CSV
  timestamp after a process restart. This prevents repeated warmup/output
  samples and avoids loading an entire archive on every process start.
- Invalid MT periods/shapes, non-positive impedance variances, unsafe cache
  paths, malformed line geometries, and non-positive/disconnected network
  resistances now fail early with actionable errors instead of reaching a
  spline/linear solve or silently contaminating a result.
- Downloaded MT, transmission-line, and magnetic cache files are replaced
  atomically. Transmission-line GeoJSON records the service URL, query,
  bounding box, and UTC retrieval time needed to identify the source dataset.
- `scan` builds the voltage kernel once instead of per chunk; `storm_scan`
  skips (with a warning) catalog windows outside the magnetic record;
  exceedance collection warns when a threshold is set too low.
- NIMS system-ID parsing recognizes backbone ("B") units; experimental-unit
  notice is a warning, not a print.
- INTERMAGNET GIN requests are chunked to 92 days of minute data (1 day of
  second data) and halve adaptively on connection resets: the GIN drops
  responses larger than ~10 MB mid-transfer regardless of its documented
  366-day request limit, which made multi-month downloads fail.
- GIN client-error responses (4xx) fail fast with the service's message
  instead of pointlessly retrying with smaller chunks; observatory downloads
  that return an all-NaN payload (levels the service does not actually hold)
  are no longer stored in the archive.
- `leave_one_out` band coherence uses enough Welch segments to be meaningful
  on short windows (a single segment is identically 1).
- `spud` bundle resolution, INTERMAGNET table refresh, pathlib waveform IO,
  `calc_resistivity` naming (deprecated misspelled alias retained).

### Changed

- `bezpy.data` access without the optional dependencies now raises an
  ImportError with the install hint (previously `bezpy.data` was `None`).
- Defaults that may change with more evidence (tracked in `docs/validation.md`):
  SECS fitting stays `method="lstsq"`; band-edge handling is
  `extrapolate="asymptotic"`. The legacy hard band-stop remains available as
  `extrapolate="zeros"`.
