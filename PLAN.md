# bezpy work plan

**Status:** Actionable plan — written 2026-07-07, revised same day. Supersedes
the *sequencing* in `ROADMAP.md` (which remains the broader idea inventory);
informed by `THEORY.md` (physics), `ARCHITECTURE.md` (kernel factorization +
prototype), and a review of the maintainer's four paper repositories
(GeoelectricHazardPaper2019, EMP_paper_2020, Anomalies1989,
Modified_GIC_Paper_Kelbert_2020).

**Audience:** an agent picking up one work item. Each item is one branch, one
PR, independently shippable, with tests. Read §2 and §3 first; then take the
next unclaimed item.

**The framing that governs everything below:** the paper workflows are
**evidence of need, not specifications**. They show which blocks every project
re-invents; they do not dictate how those blocks should work. For each block we
implement the *best* method we can defend as the default, keep the legacy
method as an explicitly named option where it is cheap to do so (reproducing
published results has real value), and prove the difference in a comparison
harness rather than asserting it. Priority order when methods conflict:
**correctness > performance > usefulness/ergonomics > fidelity to past
practice.**

---

## 1. The evidence: what every paper re-implements

Common blocks across all four repos (details in git history of this file):

1. `calculate_SECS(B, obs_xy, pred_xy)` — copied nearly verbatim per repo.
2. Observatory prep — xarray/netCDF per observatory; pad storm window ±2 days;
   `interpolate_na`; **drop the whole observatory if any NaN remains**; median
   detrend; trim pad after FFT convolution.
3. Storm cataloging — iterative threshold-and-mask on Dst (< −140 nT) and Kp
   (≥ 8), ±1.5-day windows, merge overlaps.
4. Per-storm scan — loop storms → SECS → **per-site `convolve_fft` in a Python
   loop** → per-line `calc_voltages` loop → keep maxima → CSVs.
5. Hazard stats + maps — lognormal fits of storm maxima (`powerlaw` package) →
   100-year return levels → cartopy maps with repeated boilerplate.

## 2. Where past practice is *not* the way to do it (settled improvements)

These are deliberate departures from the notebooks, to be implemented as
defaults, each with the legacy behavior available by name and a comparison
test demonstrating the difference:

- **D1. Gaps: mask, don't fabricate.** The notebooks interpolate across gaps
  and then drop an observatory entirely if NaNs survive — the 1989 analysis
  loses all of BOU to one gap. Better: keep every sample and feed SECS
  **per-sample infinite variance for missing data** (pysecs supports per-obs
  std), optionally with `generate_var`-style inflation near gap edges. No
  fabricated smooth B, no discarded stations. Legacy: `gaps="interpolate"`
  (with `max_gap`) for single-site, non-SECS paths.
- **D2. Detrend/window for the FFT, not a bare median.** Median subtraction
  controls neither the secular trend nor spectral leakage. Default: linear
  detrend + Tukey taper applied over the pad region; pad length **derived from
  the impedance's longest period** (default ≥ 3× max(T) ≈ 1 day for EMTF's
  30,000 s, not a hardcoded 2 days). Legacy: `detrend="median"`, `pad="2D"`.
- **D3. Batched FFTs always.** The notebooks loop `site.convolve_fft`;
  `SiteCollection` exists and must be the only path the pipeline uses (plus
  `scipy.fft`, `workers=-1`).
- **D4. Continuous scan replaces storm pre-selection for statistics.** Picking
  ~5-day windows around index-selected storms was partly a *performance
  workaround*; it also biases the sample (index-quiet but locally-active
  intervals are silently excluded — regional dB/dt events don't all move Dst).
  With batching + kernels, processing the **entire multi-decade record in
  chunks with streaming reducers** (running maxima, threshold exceedances,
  quantile sketches — never materializing full E) is feasible and unbiased;
  storms fall out as labeled exceedance clusters instead of going in as
  assumptions. Storm-window mode remains for case studies ("give me the 5 days
  around the storm") and legacy reproduction.
- **D5. Tail statistics done properly.** Lognormal fits of storm maxima fit
  the body, not the tail. Default: peaks-over-threshold + GPD with runs
  declustering and profile-likelihood/bootstrap CIs on return levels
  (D4 provides the exceedances directly). Legacy: `method="lognormal-2019"`
  reproducing the published workflow. Report uncertainty always — a return
  level without a CI is not a result.
- **D6. Z band edges: extrapolate physically, don't zero.** `ext=1` outside
  7–30,000 s is a hard band-stop that truncates exactly the short-period
  content that dominates dB/dt-driven E (THEORY.md §6, ARCHITECTURE.md §8 —
  it is also the dominant *short-line* error in the Kelbert & Lucas 2020
  comparison). Add `extrapolate="asymptotic"` (√(iω) high-frequency tail,
  Z→0 at DC) as an option, measure its effect in the harness, then decide the
  default with evidence.
- **D7. E-interpolation is a named, swappable choice — and delaunay-on-E is
  not sacred.** THEORY.md §2 argues interpolate-B/assign-Z-locally is the
  physical scheme; the kernel layer makes every linear scheme cheap. `how=`
  grows options (`"nn"`, `"1d"`, `"delaunay"`, `"secs+local-z"`) and the
  harness (3.7) scores them against synthetic truth instead of tradition.
- **D8. Validation is part of the package, not the paper.** Leave-one-out
  observatory checks appear in every repo as bespoke loops; they become a
  utility with a defined skill score.

Decisions carried over unchanged from the first draft of this plan, still
settled: **xarray core** (pipeline data layer; numpy internals untouched),
**geopandas never**, **pysecs core**, **V/GIC secondary to B→E hazard**,
precompute-and-cache doctrine, no numba/cython.

---

## 3. Work items (one branch each)

Branch naming: `feature/<slug>`. Every item: code + tests + docstring examples.
Do not combine items in one PR. Where an item implements a D-decision, the PR
must include the legacy-vs-default comparison test.

### 3.1 `feature/secs-interpolation` — the missing bridge ✅ DONE

**Status (2026-07-08):** `bezpy.mag.SECSInterpolator` landed with D1 gap
masking (per-sample inf variance; `gaps="drop"` legacy; `weight_gaps` edge
inflation), auto/bbox/explicit grids, `loo_score()` for data-driven epsilon,
xarray + bare-numpy paths, and pattern-sorted fitting (one SVD per unique
variance pattern). `method="kalman"` reserved pending a pysecs release.
Scripts rewritten on top of it. Tests: synthetic SEC-source recovery,
mask-beats-drop (D1 comparison), xarray round trip.

`bezpy.mag.SECSInterpolator` replacing the copy-pasted `calculate_SECS`:

```python
interp = bezpy.mag.SECSInterpolator(grid="auto")   # grid from obs coverage; explicit bbox accepted
interp.fit(b_obs)                                  # xr.Dataset (time, obs); NaNs -> inf variance (D1)
b_pred = interp.predict(latlons)                   # xr.Dataset (time, loc): Bx, By [, prediction var]
```

- Wraps `pysecs.SECS`; paper defaults (110 km shell, epsilon=0.05, Bz
  excluded) as the baseline configuration.
- **D1**: missing samples get infinite variance; `weight_gaps=True` ports
  `generate_var` edge inflation. No pre-interpolation of B.
- `method="lstsq" | "kalman"`: expose `pysecs.KalmanSECS` + robust fitting
  (huber/bisquare) — designed for exactly this gappy storm data; harness (3.7)
  decides if it becomes the default.
- Regularization: `epsilon` tunable with a documented `loo_score()` helper so
  the choice is data-driven, not folklore.
- Tests: synthetic dipole recovery; gap-masking beats interpolate-then-drop on
  synthetic gappy data (the D1 comparison); Kalman ≥ lstsq on noisy input.
- Rewrite `scripts/secs_data.py`/`realtime.py` on top of it.

### 3.2 `feature/preprocessing` — own the boring-but-critical B prep ✅ DONE

**Status (2026-07-08):** `combine_observatories`/`prepare`/`trim_pad` landed
in `bezpy.mag.processing`: union time axis with lat/lon coords (attrs or IAGA
table), uniform-dt validation with actionable errors, D2 defaults (linear
detrend + Tukey taper over the pad, `pad="auto"` = 3 x max impedance period)
with legacy `detrend="median"`/explicit `pad="2D"`, D1 `gaps="mask"` default
with legacy `"interpolate"` + `max_gap`. D2 leakage comparison test included.

`bezpy.mag` functions used by the pipeline, usable standalone:

```python
b = bezpy.mag.combine_observatories({"BOU": ds1, ...})        # -> (time, obs) Dataset
b = bezpy.mag.prepare(b, window=(t0, t1), pad="auto",         # D2: pad from max Z period
                      detrend="linear", taper="tukey",         # D2 defaults
                      gaps="mask")                             # D1 default; "interpolate" legacy
e = bezpy.mag.trim_pad(e)                                      # pad tracked in attrs
```

- Uniform-dt validation with actionable errors (also lands inside
  `convolve_fft` — ROADMAP P5).
- Tests: pad round-trip; leakage of a synthetic long-period signal under
  median-vs-linear+taper (the D2 comparison); gap policies.

### 3.3 `feature/storms` — storm catalogs and index data ✅ DONE

**Status (2026-07-08):** `bezpy.storms.find_storms` (Dst/Kp trigger + window
expansion + overlap merging, legacy-compatible) and `find_events` (threshold
exceedance + runs declustering on any series, the D4/3.6 primitive) landed as
a top-level module; `bezpy.data.fetch_index` fetches Kp/ap (GFZ Potsdam) and
Dst (OMNI2 yearly files) into the pooch cache. Parsers fixture-tested;
catalogs tested on synthetic series incl. Kp-only (index-invisible-to-Dst)
storms and overlap merging.

`bezpy.storms` + `bezpy.data.fetch_index`:

```python
idx = bezpy.data.fetch_index("dst", "1985", "2015")            # + "kp"; pooch-cached
catalog = bezpy.storms.find_storms(idx, dst_threshold=-140, kp_threshold=8,
                                   window="1.5D", min_duration="3H")
catalog = bezpy.storms.find_events(series, threshold=..., decluster="runs")
```

- `find_storms`: the papers' index-based catalog (legacy-compatible, still the
  right tool for case-study selection).
- `find_events`: threshold exceedance + runs declustering on **any** series —
  |dB/dt|, |E|, |V| — the D4-compatible primitive that turns continuous-scan
  output into an event catalog. One implementation shared with 3.6.
- Tests: synthetic series with known events for both; overlap merging;
  boundary behavior.

### 3.4 `feature/pipeline` — nothing → E (→ V) with both scan modes ✅ DONE

**Status (2026-07-08):** `bezpy.pipeline` landed: `calc_geoelectric_field`
(prepare → SECS/uniform → batched `SiteCollection.convolve_fft`, which now
accepts per-site B and threads D6 `extrapolate` — → trim), top-level
`calc_voltages` on Level-1 kernels, `scan` (chunked, streaming max/
exceedances/histogram-sketch quantiles, resumable netCDF, `leave_one_out`),
`storm_scan`, `bezpy.data.load_observatory_group`, and `bezpy.tl.filter_bbox`
/`filter_polygon` (STRtree). Tests: uniform-vs-manual exactness, kernel-vs-
legacy voltage exactness, D4 index-invisible-event comparison, chunk-pad
convergence, resume-after-crash equality.

`bezpy.pipeline` composing 3.1–3.3 (ROADMAP P2):

```python
sites = bezpy.data.load_mt_sites(bbox=..., min_rating=3)
b_obs = bezpy.data.load_observatory_group([...], start, end)
e = bezpy.calc_geoelectric_field(b_obs, sites, interp="secs")   # prepare -> SECS -> SiteCollection FFT -> trim
v = bezpy.calc_voltages(lines, e)                               # Level-1 kernels, never per-line loops

# D4 — the statistics path: chunked continuous scan with streaming reducers
result = bezpy.pipeline.scan(b_obs, sites, lines=None,
                             chunk="1Y", reduce=["max", "exceedances", "quantiles"],
                             threshold=...)                     # -> small xr.Dataset, resumable (netCDF as it goes)

# Case-study path: explicit windows (storm catalog or hand-picked)
maxima = bezpy.pipeline.storm_scan(catalog, b_obs, sites, lines=None)
```

- Streaming reducers never hold full E in memory; chunk boundaries handled
  with pad-overlap so convolution edges never touch the statistics.
- `leave_one_out=obs_code` supported in both scans (D8 hook).
- Two shapely helpers so geopandas isn't needed: `bezpy.tl.filter_bbox` /
  `filter_polygon` (STRtree).
- Tests: end-to-end synthetic; **continuous scan vs storm-window scan on a
  synthetic record with an index-invisible local event** (the D4 comparison);
  chunk-boundary invariance (result identical for chunk="1Y" vs "6M"); resume.

### 3.5 `feature/telluric-kernels` — productionize the prototype ✅ DONE

Prototype validated (ARCHITECTURE.md §9: Level-1 exact, 23× voltage stage; τ
≈1e-8). Remaining: line resampling at build time (`resample_km=`), τ
`to_netcdf/from_netcdf`, FFT-grid τ caching across equal-length chunks, D6
band-edge option threaded through τ construction, docs.

**Status (2026-07-08):** all remaining items landed — `TransmissionLine.resampled`
(great-circle densification) + `resample_km=` in `from_lines`; τ
`to_netcdf`/`from_netcdf` (geometry-free restore); FFT-grid τ cache keyed on
(nfft, dt) with a memory budget; D6 `extrapolate="asymptotic"` implemented in
`Site3d.calcZ` and threaded through both exact and master-grid τ (validated
against an analytic half space); README section.

### 3.6 `feature/hazard-stats` — defensible tail statistics

`bezpy.stats` (scipy-only core):

```python
rl = bezpy.stats.return_level(exceedances, years_observed=31, return_period=100,
                              method="gpd", ci="profile")       # D5 default, CI mandatory in output
rl = bezpy.stats.return_level(storm_maxima, ..., method="lognormal-2019")  # reproduces the paper
```

- GPD/POT with declustering (shared `find_events`), profile-likelihood or
  bootstrap CIs; GEV-on-annual-maxima as the third option.
- Directional hazard noted as follow-on: scalar max |E| hides polarization;
  API leaves room for per-direction return levels (don't build yet).
- Drop the `powerlaw` dependency if scipy reproduces the 2019 numbers;
  otherwise `[stats]` extra.
- Tests: parameter recovery on synthetic GPD/lognormal draws; CI coverage on
  simulation; regression against a published 2019 return level where feasible.

### 3.7 `feature/validation-harness` — where method choices get decided (new, D7/D8)

The apparatus that keeps this plan honest — `benchmarks/` + `bezpy.validation`:

- **Synthetic truth benchmark** (THEORY.md §8): a known conductivity structure
  + synthetic source where true E and V are computable; score every `how=`,
  `gaps=`, `detrend=`, `extrapolate=`, `method=` option against truth. This is
  the arbiter for D1/D2/D6/D7 defaults.
- **Reference-storm comparisons**: one well-covered storm (e.g. May 2024 or
  Halloween 2003) run under all method combinations, with a persisted results
  table so PRs show diffs.
- `bezpy.validation.leave_one_out(b_obs, interp, metric=...)` — the every-paper
  loop as a utility with defined skill scores (RMS, coherence per band).
- Extends the perf benchmarks (`benchmarks/bench_kernels.py` pattern) so
  performance regressions are visible alongside accuracy.
- Deliverable includes a short `VALIDATION.md` summarizing current
  method-comparison results — the evidence base for defaults.

### 3.8 `feature/map-plotting` — optional plotting extra

`bezpy.plot` (cartopy soft-imported): `add_map_features(ax)`,
`plot_vectors(ax, latlons, vec, scale="log")` (magnitude-preserving-angle
scaling from the papers), `plot_lines(ax, lines, values)`,
`plot_hazard_map(ax, latlons, values)`. Thin: axes in, artists out. Smoke
tests on Agg; cartopy tests skipped when absent.

### 3.9 `feature/legacy-mt-io` — Z-file and EDI readers (small)

Port `ZFile`/`EDIFile` from Anomalies1989's `data_readers.py` into
`bezpy.mt.io` as `read_zfile`/`read_edi` returning `Site3d` (needed for
pre-EMTF-era data like 1989). Fixture-file tests.

### 3.10 `chore/data-module-gaps` — finish `bezpy.data` (small)

`fetch_index` (with 3.3); ship/fetch the **1D region polygons** (still
unusable from a fresh install; resolve redistribution in the PR); SPUD bbox
discovery (documented stub); refresh the INTERMAGNET table.

### 3.11 `chore/api-polish` — opportunistic cleanups

`calc_resisitivity`→`calc_resistivity` (deprecated alias), `list_1d_sites()`,
pathlib waveform I/O, `scipy.fft` workers, `Site3d` attrs in `__init__`, type
hints on new API, `CITATION.cff`. Never mixed with feature PRs.

### 3.12 `docs/site` — after 3.4 lands

Sphinx/mkdocs-material + executed tutorials: (1) one storm B→E; (2) storm
catalog + case study; (3) continuous scan → GPD return-level hazard map (the
2019 paper, done the D4/D5 way, in ~40 lines — with a sidebar reproducing the
legacy method); (4) line voltages via kernels (~30 lines); (5) units and
conventions page (§4). CI executes them.

---

## 4. Conventions (enforce in every PR)

**Units:** B in nT, E in mV/km, V in volts, Z in (mV/km)/nT organized
`(xx, xy, yx, yy)`, τ in V/nT. x = North, y = East. `TransmissionLine.pts` is
lon/lat; legacy `site_xys` arguments are (lat, lon) — document loudly; new
xarray APIs use explicit `latitude`/`longitude` coords so the ambiguity never
reaches users.

**xarray:** dims `time`, `site`/`obs`/`line`, `component` (`"x"`,`"y"`);
coords `latitude`, `longitude` on the spatial dim; units in `attrs`; pad/dt
metadata in `attrs`. Pipeline functions accept Dataset or documented bare
numpy and return in kind where sensible.

**Method options:** every place past practice and the new default diverge
(D1–D7) is a keyword with the legacy value **named and documented** (e.g.
`gaps="interpolate"`, `detrend="median"`, `method="lognormal-2019"`), and at
least one test comparing the two. Defaults may only change with harness (3.7)
evidence cited in the PR.

**Code:** vectorize over time and sites; loops over storms/events are fine.
Static things (weights, kernels, Z grids) computed once, cached with explicit
invalidation keys (`SiteCollection._Z_interp` / `Site3d.calcZ` pattern).
Style per ruff config; tests mirror `tests/test_kernels.py` (synthetic
fixtures; exactness assertions for reassociations, tolerances for
approximations).

**Process:** one feature per branch/PR; low-level numpy API stays stable; new
deps justified against §2 in the PR body; update PLAN.md status markers when
an item merges.

## 5. Explicit non-goals (for now)

GIC network solver (revisit on pull; ARCHITECTURE.md Level 3 makes it easy
later); geopandas/pyogrio/cartopy as required deps; absorbing pysecs or MTH5;
rewriting numpy internals to xarray; multi-input (SECS-weighted) τ kernels and
uncertainty propagation — they follow 3.1 + 3.5 and the harness, not precede
them; directional hazard statistics (API room left in 3.6).

## 6. Suggested sequence

```
3.5 kernels merge ──┐              (branch exists; close it out first)
3.1 SECS          ──┼─► 3.4 pipeline ─► 3.6 stats ──► 3.12 docs
3.2 preprocessing ──┤                     ▲
3.3 storms        ──┘                     │
3.7 validation harness ───────────────────┘   (start early; grows with each item)
3.8 plotting / 3.9 legacy io / 3.10 data gaps / 3.11 polish: anytime, parallel
```

3.1–3.3 are mutually independent; 3.4 consumes them. 3.7 starts as soon as any
two method options exist and is a prerequisite for changing any default.
Milestone 0.4 after 3.4 + 3.5 ("5 lines from nothing to hazard numbers");
0.5 after 3.6 + 3.7 + 3.12 (the JOSS-paper point, with a validation story no
comparable package has).
