# bezpy Roadmap / Improvement Plan

> **Note (2026-07-07):** `PLAN.md` now supersedes this document's sequencing and
> priorities — it is the actionable, branch-by-branch work plan, informed by the
> maintainer's real paper workflows. This file remains the broader idea
> inventory; individual P-items are referenced from PLAN.md.

**Status:** Proposal for review — written 2026-07-05 after the packaging/CI modernization pass.
**Audience:** A future agent (or contributor) picking up work items. Each item below is
scoped so it can be evaluated and implemented independently. Nothing here is committed;
items marked "Decision needed" require the maintainer's call before starting.

**Guiding constraints (from the maintainer):** robust, simple, easy to use, performant.
The single biggest theme is lowering the barrier for other researchers: today a user must
already know where to find EMTF transfer functions, IAGA-2002 observatory data, 1D region
polygons, and transmission-line geometries, and must know the correct order of ~8 API
calls to chain B → E → V. Most of the plan is about making that chain automatic.

---

## 1. Current state (what exists today)

| Module | Contents | Maturity |
| --- | --- | --- |
| `bezpy.mt` | `Site1d` (NERC layered-earth Z), `Site3d` (EMTF XML impedance + spline interpolation), `SiteCollection` (batched FFT convolution), `ConductivityModel` (ModEM reader), `DTIR` (time-domain impulse response), IRIS waveform download via `DataLogger`/ObsPy, apparent-resistivity plotting | Solid core, the scientific heart of the package |
| `bezpy.mag` | IAGA-2002 reader/writer, butterworth filtering, polynomial detrend, bundled INTERMAGNET observatory table | Functional but manual — user supplies files |
| `bezpy.tl` | `TransmissionLine`: haversine lengths/dl, nearest-site / 1D-region / Delaunay E-field interpolation, voltage line integral | Solid numerics; awkward construction (expects an object with `.geometry`), user must supply region polygons and site coordinates |
| `scripts/` | Real-time SECS maps, gridding, movie-making (uses `pysecs`, `requests`, `cartopy`) | Useful recipes trapped in ad-hoc scripts with duplicated download code |
| Data bundled | 38 USGS 1D `earth_model_*.txt` profiles, INTERMAGNET observatory list (2017 snapshot) | The 1D **region polygons** needed by `set_1d_regions` are NOT bundled — a real gap |
| Tests/CI | pytest suite, ruff, GitHub Actions across 3.10–3.13 | Recently added, good foundation |

Strengths to preserve: small dependency footprint (numpy/pandas/scipy/shapely), vectorized
numerics, optional extras keeping the base install light.

---

## 2. Priority tiers (recommendation)

- **Tier 1 — do first, highest leverage:** P1 data fetching, P2 high-level pipeline API, P3 pysecs integration, P4 docs site.
- **Tier 2 — robustness and API polish:** P5–P8.
- **Tier 3 — new science capability:** P9 GIC network solver (the one genuinely *missing* piece), P10–P12.

Rationale: nothing in Tier 3 is usable by outsiders until Tier 1 exists. The numerics are
already good; discovery, data access, and chaining are the bottlenecks.

---

## 3. Tier 1 — Out-of-the-box usability

### P1. Automated data acquisition and caching (`bezpy.data`)

The single highest-impact addition. All three input datasets are "tricky to find and get
right" and every user re-solves the same problem. Introduce a `bezpy.data` (or
`bezpy.fetch`) module built on **[pooch](https://github.com/fatiando/pooch)** — the
standard scientific-Python download/cache library (used by scipy, scikit-image, MetPy,
and the Fatiando geophysics stack). Pooch gives us: OS-appropriate cache directory,
hash verification, progress bars, and offline reuse for free. It is a tiny, pure-Python
dependency — worth adding to core or a `data` extra.

Sub-items, roughly in value order:

1. **MT transfer functions (EMTF/SPUD).** ✅ *Partially implemented (v0.3-dev) in
   `bezpy.data.mt`:* `download_mt` / `ingest_mt_xml` / `load_mt_site` /
   `load_mt_sites(bbox=, min_rating=)` / `mt_inventory`, with EMTF XML stored
   hierarchically by survey (`mt/<survey>/<CODE>.xml`) and loaded back to
   `Site3d`. Remaining: programmatic bounding-box **discovery** against the live
   SPUD service (`spud_search_url` is a documented stub — the current SPUD query
   schema needs confirmation before wiring `fetch_mt_sites(bbox=...)`).
   The IRIS/EarthScope SPUD EMTF archive (http://ds.iris.edu/spud/emtf) has an HTTP API
   that can be queried by bounding box and returns the same XML files `read_xml` already
   parses. Cache the XMLs; return parsed `Site3d` objects, optionally filtered by data
   quality `rating`. This removes the biggest "where do I even get Z?" hurdle.

2. **Magnetic observatory data.** ✅ *Implemented (v0.3-dev) in `bezpy.data`:*
   `download_observatory` / `update_observatory` / `load_observatory` /
   `observatory_inventory` / `ingest_iaga`, backed by the **USGS/NRCan** web
   service **and the INTERMAGNET GIN**, with a provenance-aware netCDF archive
   (`magnetic/<source>/<level>/<code>/<code>_<year>.nc`), incremental
   (near-real-time) updates, cross-level best-available merge on load, and
   pooch-based cache management. Remaining: SuperMAG (needs API key) and HAPI
   backends, and evaluating geomagio (below).
   `bezpy.data.fetch_observatory(code, start, end, source="usgs") -> xr.Dataset`
   Sources, in order of implementation ease:
   - USGS geomag web service (`geomag.usgs.gov/ws/data/`) — JSON/IAGA2002, no auth;
     the download code already exists in `scripts/realtime.py` and just needs promoting.
   - NRCan web service (same pattern; the scripts already list the Canadian codes).
   - INTERMAGNET via its web service or HAPI endpoint.
   - SuperMAG API (optional; needs a user API key, but gives worldwide baseline-removed
     data — very attractive for researchers outside North America).
   Return tidy DataFrames indexed by time with X/Y/Z/F columns, i.e. exactly what
   `Site.convolve_fft` and pysecs fitting want.

3. **1D region polygons.** Bundle (or fetch) the physiographic-region polygons that
   `TransmissionLine.set_1d_regions` requires. Today the code defaults to region index 18
   (CP-1) with a comment, but the polygons themselves are not shipped — a user literally
   cannot run the 1D pathway from a fresh install. Small data file, big usability win.

4. **Transmission line geometries.** ✅ *Implemented (v0.3-dev) in `bezpy.data.tl`:*
   `download_transmission_lines(bbox=, min_voltage=)` queries a HIFLD-derived
   ArcGIS feature service (paginated GeoJSON), caches it, and returns
   `TransmissionLine` objects; `load_transmission_lines` / `read_geojson` for
   local files. `TransmissionLine` now also accepts a bare `LineString`/coord
   array and carries a `metadata` dict. Remaining: the default ArcGIS endpoint is
   re-hosted periodically (pass `url=` to override) — confirm the canonical HIFLD
   URL/license; optional `geopandas`/`pyogrio` shapefile loader.

5. **Refresh the bundled INTERMAGNET observatory table** (2017 snapshot) or replace the
   custom `.dat` parsing with a fetch + cache of the live list.

**Decision needed:** pooch as core dependency vs `[data]` extra (recommend core — it's
tiny and Tier 1 features all depend on it).

### P2. High-level pipeline API — "5 lines from nothing to voltages"

Today's chain requires knowing: `read_xml` → `calcZ`/`convolve_fft` on detrended,
gap-free, uniformly sampled B → build `TransmissionLine` from a GeoDataFrame row →
`set_delaunay_weights`/`set_nearest_sites` with correctly ordered lon/lat arrays →
`calc_voltages(E, how=...)`. Each arrow is a place newcomers get stuck (units, array
shapes `(ntimes, nlocs, 2)`, lon/lat vs lat/lon ordering).

Add a thin orchestration layer (new module, e.g. `bezpy.pipeline`) that composes the
existing objects without replacing them:

```python
import bezpy

b = bezpy.data.fetch_observatory("BOU", "2024-05-10", "2024-05-12")
sites = bezpy.data.fetch_mt_sites(bbox=(-110, 35, -100, 45), rating=3)
e = bezpy.calc_geoelectric_field(b, sites)          # handles detrend/gaps/FFT
lines = bezpy.data.fetch_transmission_lines(bbox=...)
v = bezpy.calc_voltages(lines, e, sites, how="delaunay")  # handles weights setup
```

Design notes for the implementer:
- Keep the low-level API untouched and documented; the pipeline is sugar, not a rewrite.
- `calc_geoelectric_field` should do the boring-but-critical preprocessing with loud,
  documented defaults: uniform-sampling check, gap interpolation policy, detrending,
  optional windowing, dt inference from the DataFrame index.
- Return values should carry coordinates/metadata (see P6 on data structures) so the
  next stage can consume them without the user re-supplying site locations.
- When multiple observatories are passed, interpolate B to MT-site locations via pysecs
  (P3) automatically.

### P3. pysecs integration for magnetic-field interpolation

**Recommendation: depend on pysecs; do not absorb it.** Reasons:
- pysecs is numpy-only, tiny, independently useful (satellite/current-mapping users have
  no need for bezpy), and separately cited/DOI'd. Merging it in would couple release
  cycles and dilute both packages' identities. The precedent (scipy stack, Fatiando)
  favors small composable packages.
- bezpy's `scripts/` already import it — the dependency de-facto exists; make it honest.

Concrete work:
1. Add `pysecs` to a `[secs]` extra or core deps (it's numpy-only — core is defensible;
   **decision needed**).
2. New `bezpy.mag.SECSInterpolator` (or function `interpolate_b`): takes observatory
   DataFrames + locations, fits `pysecs.SECS` (or `KalmanSECS` for gappy/noisy storm
   data — its innovation gating and robust `huber`/`bisquare` fitting are exactly what
   multi-observatory storm data needs), predicts B(t) at arbitrary lat/lon (MT sites,
   grid points). This is the missing bridge that turns "I have 5 observatories" into
   "I have B at every MT site", which then feeds `SiteCollection.convolve_fft`.
3. Propagate pysecs prediction variances through to E-field uncertainty where cheap
   (ties into P12).
4. Rewrite `scripts/secs_data.py` / `realtime.py` to use the new module (they become
   ~50-line examples instead of 300-line scripts with duplicated download classes).

### P4. Documentation site

A package aimed at "researchers who don't want to know all the details" lives or dies by
docs. The README + 2 notebooks aren't enough.
- Sphinx (or mkdocs-material) + API reference from docstrings + executed tutorial
  gallery (sphinx-gallery or myst-nb), published via GitHub Pages from CI.
- Tutorials to write, in order: (1) B→E at one 3D site, (2) 1D vs 3D comparison,
  (3) multi-observatory SECS → grid E-field map, (4) transmission-line voltages for a
  storm, (5) DTIR time-domain workflow.
- Convert the two existing notebooks into executed tutorials so they can't silently rot
  (CI executes them).
- A "units and conventions" page: B in nT, E in mV/km, V in volts, Z organized
  `(xx, xy, yx, yy)`, x=North/y=East, pts stored lon/lat — every one of these is
  currently tribal knowledge read from comments.
- Add a `CITATION.cff` (pysecs already has one) and conda-forge feedstock.

---

## 4. Tier 2 — Robustness and API polish

### P5. Input validation and preprocessing hardening

The FFT convolution silently assumes uniform sampling, no NaNs, and mean-zero-ish input.
Real observatory data violates all three. Add:
- Explicit checks with actionable errors in `convolve_fft` (uniform dt, NaN detection).
- A documented preprocessing helper (gap fill policy, detrend, taper/window option) used
  by the P2 pipeline.
- `read_iaga`: the hardcoded `usecols=[0, 1, 3, 4, 5, 6]` breaks on files with more/fewer
  channels; derive columns from the header instead. Handle `|`-terminated headers and
  daylight-savings/timezone edge cases explicitly (IAGA is UTC — assert it).
- `write_iaga_2002`: header is hardcoded (station "Random", source "Greg Lucas", claims
  XYZF while writing Ex/Ey). Parameterize station metadata; build header from arguments.

### P6. Data-structure conventions (decision needed: xarray or documented ndarray)

E fields flow through the code as bare `(ntimes, nlocs, 2)` arrays; site coordinates
travel separately; units are implicit. Two options:
- **Option A (lighter, recommended to start):** keep numpy/pandas, but standardize and
  document shapes/units in one place, and have pipeline functions return DataFrames with
  named columns / MultiIndex (site, component).
- **Option B:** adopt xarray for multi-site/gridded outputs (`E(time, site, component)`
  with lat/lon coords). Natural fit for grid products and netCDF export, and researchers
  in this field increasingly expect it; but it's a new core dependency and a bigger API
  change. Could live only in the pipeline layer (`as_xarray=True`).

### P7. API cleanups (small, safe, do opportunistically)

- `Site3d.calc_resisitivity` → `calc_resistivity` with deprecated alias.
- `TransmissionLine.__init__(line)` requires an object with `.geometry.coords` (a
  GeoDataFrame row). Accept a shapely `LineString` (or bare coord array) directly and
  add `TransmissionLine.from_geodataframe(gdf)` classmethod for the current pattern.
- `Site3d` grows attributes outside `__init__` (`network_code`, `elevation`,
  `declination`, `channel_orientation`, `rating`, ...) set by `read_xml`. Declare them in
  `__init__` (or make `read_xml` a classmethod `Site3d.from_emtf_xml`) so the class is
  introspectable and type-checkable.
- `get_1d_site` raises on unknown names — add a `list_1d_sites()` discovery function.
- `save/load_waveforms(directory + name)` — use `pathlib` joins; accept full paths.
- Type hints on public API (pysecs already did this; mirror its style), `py.typed`.
- `_rotate_waveforms` assumes `channel_orientation` keys exist; guard with a clear error
  for older XMLs without `SiteLayout` (currently would `KeyError`).

### P8. Performance (targeted, not speculative)

Current numerics are already vectorized and cached sensibly. Two cheap wins:
- Use `scipy.fft.rfft(..., workers=-1)` instead of `numpy.fft` in
  `Site.convolve_fft`/`SiteCollection.convolve_fft` — multithreaded FFTs for free on the
  large batched transforms (scipy is already a dependency).
- `SiteCollection` caches `Z_interp` on `(N0, dt)` but recomputes per collection; expose
  `SiteCollection.calcZ(freqs)` for reuse and profile memory for ~1000-site CONUS runs
  (the `(4, nsites, nfreqs)` complex array is the real constraint — document it).
- Add an `asv` or simple pytest-benchmark suite (pysecs has `benchmarks/` — copy the
  pattern) so regressions are visible. Avoid numba/cython until a profile demands it.

---

## 5. Tier 3 — New capabilities

### P9. GIC network solver — the genuinely missing piece

bezpy stops at line voltages. The quantity utilities and most GIC papers actually need is
**transformer/substation GIC (amps)**, via the Lehtinen–Pirjola or nodal-admittance
method using substation grounding resistances and line/transformer resistances. There is
no maintained, general Python package for this (PowerModelsGMD is Julia; most groups have
private MATLAB). A `bezpy.gic` module — network ingest (nodes, lines, resistances),
sparse solve (scipy.sparse, trivially fast), benchmark against the published
Horton/NERC 20-bus GIC test case — would make bezpy the end-to-end open-source GIC
stack and is highly citable. Moderate effort; the math is ~200 lines; the work is data
model + validation.

### P10. Gridded E-field / hazard-map products

Promote the "map" workflow to a first-class API: regular lat/lon grid → SECS-interpolated
B → per-cell E via nearest MT site or 1D region → xarray/netCDF export + optional cartopy
quicklook plot. This is what the swx-trec.com real-time product does; packaging it lets
others reproduce it regionally (Europe, Australia...).

### P11. Extreme-value / hazard statistics

The Lucas et al. (2020) 100-year hazard analysis workflow (peaks-over-threshold,
return-level estimation on |E| or voltages) currently lives outside the package. A small
`bezpy.stats` module (scipy.stats GPD fitting, declustering, return-level CIs) would let
others reproduce hazard maps for their own regions and directly serves NERC TPL-007
benchmark-style studies. Consider leveraging `pyextremes` instead of hand-rolling —
evaluate its maintenance status first.

### P12. Uncertainty propagation

`Z_var` is parsed and used for spline weighting, but E and V come out without error bars.
Frequency-domain propagation (Var(E) from Var(Z)·|B|² per frequency) is cheap and, paired
with pysecs prediction variances (P3), yields "E-field with uncertainty" — rare in this
field and a strong differentiator.

---

## 6. Leverage instead of reinvent — package survey

| Package | What it offers | Recommendation |
| --- | --- | --- |
| **pooch** | Download + cache + hash verification | **Adopt** (P1) |
| **pysecs** (maintainer's own) | SECS B-field interpolation, Kalman smoothing, robust fits, variances | **Adopt as dependency** (P3); keep separate package |
| **geomagio** (USGS) | IAGA2002/Edge/web-service readers, geomag algorithms | **Investigate** for `bezpy.data` observatory fetching before writing our own client; check dependency weight. If too heavy, mimic its web-service endpoints with plain `requests` |
| **mt_metadata / mth5 / mtpy-v2** (MTH5 ecosystem) | Canonical EMTF XML parsing, MT waveform HDF5, full MT processing | **Interop, not dependency**: keep the lightweight built-in XML parser; add optional `Site3d.from_mt_metadata(tf)` adapter so MTH5-world users can enter bezpy. Their dep stack is heavy |
| **ObsPy** | FDSN waveform access | Already an optional extra — keep as-is |
| **hapiclient** | HAPI time-series protocol (INTERMAGNET, CDAWeb) | Investigate as one backend for P1.2 |
| **SuperMAG API** | Global cleaned observatory data (auth required) | Optional P1.2 backend |
| **geopandas / pyogrio** | Shapefile/GeoJSON ingest for transmission lines & region polygons | Optional `[tl]`/`[data]` extra; shapely 2 already core |
| **pyproj / geographiclib** | Exact geodesics | Not needed now — haversine error is negligible vs data uncertainty; revisit only if someone needs cm-accurate line lengths |
| **xarray** | Labeled N-D arrays, netCDF | Decision in P6 |
| **pyextremes** | Extreme-value analysis | Evaluate for P11 |
| **cartopy** | Map plotting | Keep optional (scripts/gallery only) |

Rule of thumb applied: adopt small pure-Python infrastructure (pooch, pysecs); interop
with heavy domain stacks (MTH5, ObsPy) behind optional adapters; never make matplotlib,
cartopy, or geopandas required.

---

## 7. Adjacent fields and extensions worth exploring

Ordered by closeness to current capability:

1. **Pipelines (cathodic protection / pipe-to-soil potentials).** Same E-field input,
   different network model (distributed-source transmission line theory, Boteler &
   Cookson). The pipeline-integrity industry actively studies this; a `bezpy` example or
   module would open a second user community.
2. **Railway signaling and submarine/long telecom cables.** Same line-integral math
   `TransmissionLine` already does; mostly a documentation/example effort.
3. **NERC TPL-007 benchmark studies.** Utilities must compute benchmark GMD E-fields ×
   their network. P9 + P11 make bezpy directly usable for compliance-adjacent research.
4. **Regional ports outside North America** (Europe/UK, Australia, China, South Africa,
   Brazil all have active GIC programs). Mostly unlocked by P1 supporting INTERMAGNET/
   SuperMAG sources and non-US MT archives; a worked non-US example would signal this.
5. **Ionospheric/auroral equivalent currents** — pysecs's home turf; bezpy grid products
   (P10) feed this via the same interpolation layer.
6. **Space-weather forecasting/nowcasting** — the realtime script productized (P10) is a
   nowcast; coupling to forecast B (e.g. from MHD model output) is a natural paper.

---

## 8. Quick wins (do in any order, < 1 day each)

- Ship the 1D region polygons (or a fetcher) — unblocks `set_1d_regions` (P1.3).
- `scipy.fft` with `workers=-1` (P8).
- `calc_resisitivity` typo deprecation, `list_1d_sites()`, pathlib in waveform I/O (P7).
- Parameterize `write_iaga_2002` header (P5).
- `TransmissionLine` accepting shapely `LineString` (P7).
- Refresh `intermagnet_observatories.dat` (P1.5).
- CITATION.cff.

## 9. Suggested sequencing

```
Milestone 0.3: P1.1 (SPUD fetch) + P1.2 (USGS/NRCan fetch) + P1.3 (region polygons)
               + quick wins + CITATION.cff
Milestone 0.4: P3 (pysecs bridge) + P2 (pipeline API) + P5 (validation) + scripts rewrite
Milestone 0.5: P4 (docs site + tutorials) + P6 decision + P7 cleanups  → announce/paper?
Milestone 0.6: P9 (GIC solver) or P10 (grid products) — pick based on user pull
Later:         P11, P12, adjacent-field examples
```

A JOSS (Journal of Open Source Software) paper after Milestone 0.5 would formalize
citability and drive adoption; bezpy is squarely in JOSS scope.

## 10. Open decisions for the maintainer

1. pooch and pysecs: core dependencies or extras? (Recommend both core — tiny footprints.)
2. xarray: adopt (pipeline-layer only?) or stay pandas/numpy? (P6)
3. geomagio as dependency vs plain-requests web-service client? (P1.2)
4. Where do 1D region polygons come from, and are they redistributable? (P1.3)
5. HIFLD transmission-line access status/license in 2026. (P1.4)
6. P9 (GIC network) vs P10 (grid products) first — which community to court?
