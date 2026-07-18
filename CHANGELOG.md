# Changelog

## 0.5.0 (unreleased)

The first release of the rebuilt package: bezpy grows from a collection of
per-paper utilities into a full B → E → V hazard-analysis pipeline with an
evidence-driven validation story. See `VALIDATION.md` for the
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
- **Impedance spline weighting**: periods with non-finite or zero `Z_var` are
  now excluded from the interpolating spline. Previously a single NaN
  variance (e.g. from a gridded-product row without an error column) made an
  entire impedance component — and every E field computed from it — NaN.
- **`leave_one_out` band coherence** was computed on an interleaved Bx/By
  sequence and is now computed per component and averaged. Coherence-based
  skill scores from earlier runs are unreliable; RMSE/MAE/correlation scores
  were always correct.

### New

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
- `bezpy.storms` — `find_storms` (index catalog) and `find_events`
  (threshold + runs declustering on any series).
- `bezpy.stats.return_level` — GPD/POT with profile-likelihood or bootstrap
  CIs; `lognormal-2019` and `gev` by name. CIs are mandatory in the output.
- `bezpy.validation` — leave-one-out skill scores and a synthetic-truth
  benchmark that arbitrates method defaults.
- `bezpy.tl.kernels` — Level-1 voltage functionals and Level-2 power-line
  telluric response functions (Kelbert & Lucas 2020), with netCDF round-trip
  and FFT-grid caching; `TransmissionLine.resampled` great-circle
  densification; `filter_bbox`/`filter_polygon` (no geopandas).
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
  `vmin`/`vmax` (saturating log color limits) or a full `norm` override.
- Runnable example scripts (`scripts/mt_site_example.py`,
  `scripts/japan_halloween.py`) replace the old Jupyter notebooks.
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
  sites without impedance variances; the data-space path uses a
  pseudo-inverse.
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
- Defaults that may change with more evidence (tracked in `VALIDATION.md`):
  SECS fitting stays `method="lstsq"`; band-edge handling stays
  `extrapolate="zeros"`. Both alternatives are one keyword away.
