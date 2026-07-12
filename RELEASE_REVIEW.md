# bezpy pre-release code review — 2026-07-11

**Scope:** full-package review of the `telluric-kernels` branch (HEAD `2d1b9cf`) ahead
of the upgrade release. Every module in `bezpy/` was read; the math-critical paths
(kernels, impedances, SECS wrapper, statistics, preprocessing) were verified against
analytic results by running code, not by inspection alone. Test suite at review time:
**216 passed, 1 skipped** (cartopy absent).

**Verdict:** the package is in strong shape — the design (kernel factorization,
canonical xarray conventions, provenance-aware data archive, streaming scans,
evidence-driven defaults) is coherent and better than any comparable package in this
space. The math that matters is right. What stands between this branch and a release
is a short list of **six defects** (three of them silent-data-corruption class, all
small fixes), some **memory-scaling guardrails**, and **release mechanics** (version,
changelog, publish workflow). Nothing structural.

---

## 1. Math audit — what was verified and found correct

| Area | Check | Result |
|---|---|---|
| `Site1d.calcZ` (Wait recursion) | Half-space vs analytic `Z = √(iωμρ)·(10⁻³/μ)`; phase +45°; layered recursion (NERC eqs. 5–7); H→B unit factor | exact match |
| `apparent_resistivity` | `ρ_a = T·|Z|²/5` for Z in (mV/km)/nT — the standard practical-unit formula | correct |
| Level-1 kernels | `voltage_functional` is the exact reassociation of `calc_voltages` (nn/1d/delaunay), mV/km·km/1000 → V; out-of-hull → NaN semantics preserved; sparse contraction index order | correct (also exactness-tested in `tests/test_kernels.py`) |
| Level-2 τ (Kelbert & Lucas 2020 Eqs. 10–13) | `τ^N = Σ L_x Z_xx + L_y Z_yx`, `τ^E = Σ L_x Z_xy + L_y Z_yy`; V/nT units; master-grid log-period spline mirrors `Site3d` interpolation; √f band-edge continuation anchored at the correct edges (ascending-period indexing) | correct |
| FFT conventions | rfft/irfft pairing with EMTF's e^{+iωt} impedances is internally consistent (verified: half-space phase sign propagates to a causal response); padding rule `N = 2^(⌊log₂N₀⌋+2)` ≥ 2·N₀ with wrap-around confined to the trimmed pad; single-site vs `SiteCollection` results identical | correct |
| `haversine_dl` / voltage quadrature | (dl_lat, dl_lon) ordering matches (Ex, Ey); signed arc lengths; `resampled()` slerp preserves great-circle length, degenerate segments handled | correct |
| Gnomonic projection in `set_delaunay_weights` | standard forward formulas; both site and line points now projected identically (the 2026-07-09 fix); barycentric weights sum to 1; outside-hull points always yield a negative weight (find_simplex −1 → last simplex) | correct |
| SECS wrapper | inf-variance masking (D1) zero-fills B only where std=∞; `loo_score` folds keep the pole grid fixed; predict reshape handles pysecs squeezing | correct |
| `stats.return_level` | GPD level `u + σ/ξ((rate·T)^ξ − 1)` with the ξ→0 limit; profile-likelihood CI (Coles 4.3.3) reparametrization σ(x_t, ξ) inverted correctly, deviance target `χ²₁(level)/2`; lognormal-2019 `ppf(1 − 1/(rate·T))`; GEV sign convention documented | correct |
| `pipeline` streaming reducers | histogram sketch: 1350 log bins ≈ 1.5% resolution over 10⁻⁴–10⁵; quantile inversion via CDF searchsorted; chunk edges from `pd.period_range` are disjoint at ns resolution (no double counting — verified) | correct |
| `prepare`/`trim_pad` | pad bookkeeping via searchsorted against the actual time axis (handles truncated pads at record edges); Tukey ramps confined to the pad; NaN-aware detrend | correct (but see defect **B2**) |
| `DTIR` | A-matrix `e^{−iωnΔt}`; model-space Gramian `AᵀΣ⁻¹A + λQ` and data-space `AQ⁻¹Aᵀ + λΣ` both correct despite confusing local naming (`sigma` holds Σ⁻¹) | correct (but see defect **B6**) |
| ZFile/EDI rotations | EMTF variance formula `Re(σ_e·σ_s)`; EDI de-rotation `RᵀZR`; rotation-invariance covered by tests | correct for orthogonal channel pairs (see C3) |

The projection fix (Q4), the short-period spline degradation, and the resume guard
from the 2026-07-09/10 review round are all in place and tested.

---

## 2. Defects to fix before release

### A. Silent wrong-answer class (must fix)

**A1. Non-finite / zero `Z_var` poisons the impedance spline → all-NaN E, silently.**
`Site3d.spline_interp` (`bezpy/mt/site.py:199`) builds weights `1/√|Z_var|` using
`good_vals` that only screens NaN **Z**, not NaN or zero **Z_var**. A single NaN
variance makes the whole component's spline NaN at every frequency; a zero makes an
infinite weight. Verified: one NaN column in `Z_var` → `calcZ` returns all-NaN → E is
all-NaN with no warning.
This is reachable through documented behavior: `read_gridded_impedance` sets
`Z_var[i,j] = NaN` for data rows lacking an error column while other rows have one
(`bezpy/mt/modem.py:87`), and accepts `error = 0.0` as-is. `read_xml` already drops
zero/NaN-var periods for exactly this reason (`bezpy/mt/io.py:165`) — the spline
itself should be the guard, not each reader.
*Fix:* in `spline_interp`, fold `np.isfinite(Z_var[i]) & (Z_var[i] > 0)` into
`good_vals` when weights are used (or set those weights to the max finite variance).
One test with mixed-error input.

**A2. `leave_one_out` band coherence is computed on an interleaved Bx/By series.**
`bezpy/validation.py:105-107` passes `observed[good].ravel()` (row-major → Bx(t₀),
By(t₀), Bx(t₁), …) to `scipy.signal.coherence` with `fs=1/dt`. The "series" alternates
physical components at a fictitious double sample rate. Verified: perfect predictions
score **0.73** where per-component coherence is 1.000. Since the harness is the
instrument that decides method defaults (D1–D7), this metric must be right.
*Fix:* compute coherence per component and average (or report both); also note that
gap-splicing (`observed[good]`) biases the spectrum — acceptable if documented.

**A3. `TelluricResponse.calc_voltages` has no NaN-input guard.**
`SiteCollection.convolve_fft` refuses NaN B with an actionable message
(`bezpy/mt/site.py:559`); the τ path (`bezpy/tl/kernels.py:289`) does not. Verified:
one NaN sample → **all** voltages NaN, silently. Same guard, same message.

### B. Loud or path-blocking (fix or document)

**B1. EMTF sign convention is parsed but never applied.**
`read_xml` stores `site.sign_convention = -1` for `exp(-iωt)` files
(`bezpy/mt/io.py:145-146`) and then nothing anywhere conjugates Z. A minority-
convention file yields a time-reversed (acausal) impulse response — phases wrong,
magnitudes right, nothing crashes. The EDI reader at least warns; the Z-file reader
doesn't check at all.
*Fix (minimum):* warn on `-1` like the EDI reader. *Fix (right):* conjugate Z (and
tippers) on read so every `Site3d` is in one convention, and delete the attribute
ambiguity. Add one flipped-convention fixture test.

**B2. Records shorter than 2×pad silently produce an empty result.**
`prepare` without `window=` treats `min+pad … max−pad` as the analysis span
(`bezpy/mag/processing.py:207`). With the default `pad="auto"` = 25 h/side, a 1-day
record → `pad_left = pad_right = ntimes` → `trim_pad` returns **0 samples**. Verified.
`calc_geoelectric_field(b, sites)` on a short record — the most natural first call a
new user makes — returns an empty Dataset with no diagnostic.
*Fix:* raise in `prepare` when the analysis span is empty, naming the record length,
the pad, and the two ways out (`window=`, smaller `pad=`).

**B3. `DTIR.calculate(periods=..., Z=...)` always raises; `Z_var=None` crashes.**
The documented keyword path hits an unconditional `raise` (`bezpy/mt/impulse.py:84-86`
— the `elif` is commented out), and a `Site3d` without variances dies on
`site.Z_var[:, good_locs]` (`impulse.py:83`). Verified both. Either restore the
keyword path + a `Z_var is None` fallback (ones, as `calculate` already intends), or
deprecate the module loudly — it is currently exported as public API.

**B4. `stats.return_level` GEV docstring promises a check that doesn't exist.**
"`years_observed` … is checked against the number of blocks (GEV)"
(`bezpy/stats.py:137-139`) — `years_observed` is not used at all for `method="gev"`.
Implement the check (warn if `len(data) != round(years_observed)`) or fix the docstring.

**B5. Backbone-NIMS branch in `_get_logger_info` is dead code.**
`nim2 = int(nim2str)` (`bezpy/mt/datalogger.py:207`) raises `ValueError` for IDs like
`2611-B4` **before** the `nim2str[0] == 'B'` backbone check at line 213; `read_xml`
swallows the ValueError, so backbone systems silently lose their system response.
Move the prefix handling above the `int()`. (Also: `print()` → `warnings.warn`.)

**B6. `DTIR._calcQinv` comment says pseudo-inverse, code calls `np.linalg.inv`**
(`bezpy/mt/impulse.py:239`) on a matrix the comment itself calls singular. Use
`np.linalg.pinv` or fix the comment after confirming Q₁ is well-conditioned.

### C. Minor (any time)

- **C1.** `get_iaga_observatory` raises `ValueError("No observatory: ", code)` — tuple
  message (`bezpy/mag/utils.py:207`).
- **C2.** EDI `Zerr` is rotated linearly like the tensor (`bezpy/mt/legacy.py:258`);
  variances should rotate with squared coefficients. Today the consequence is only
  that `_to_site3d` discards `Z_var` whenever rotation drives an entry ≤ 0 — lossy
  but never wrong. Worth a comment now, a proper variance rotation later.
- **C3.** `_rotate_waveforms` (`bezpy/mt/site.py:371`) and the ZFile predicted-channel
  rotation use the transpose where the inverse is exact — correct only for orthogonal
  channel pairs (the overwhelmingly common case). Document the assumption.
- **C4.** `set_nearest_sites`/`set_delaunay_weights` docstrings don't state that
  `site_xys` is **(lat, lon)** while `line.pts` is (lon, lat). PLAN.md §4 says
  "document loudly" — do it at the two entry points users actually call.
- **C5.** Gnomonic projection has no guard for points ≥90° from the network centroid
  (cos_c ≤ 0 flips the projection). Fine for CONUS; one `ValueError` would make it
  safe globally.
- **C6.** `storm_scan` raises mid-loop if a catalog window has no B data, unlike
  `scan` which skips record gaps. Skip-and-warn would make index-derived catalogs
  (which often outspan B coverage) usable directly.

---

## 3. Performance review

The broadcasting/caching story is genuinely good: batched rfft over sites, one
`Z_interp` cache keyed on (N₀, dt, extrapolate), τ cached on the FFT grid across
equal-length chunks with an explicit memory budget, pattern-grouped SECS SVDs in
pysecs, STRtree spatial queries, sparse voltage contraction. Three gaps:

**P1. `SiteCollection.convolve_fft` memory has no guardrail and no site-blocking.**
The Z cache is `4·nsites·nfreq·16 B` and the E spectra are `2·nsites·nfreq·16 B`
transient. With the default `scan(chunk="1Y")` at 1-min cadence, nfreq ≈ 1.05 M →
**~100 MB per site** peak. 100 sites ≈ 10 GB; the 0.25° gridded CONUS product
(~10⁴ sites) is out of reach at that chunk size, and the failure mode is an opaque
`MemoryError` (or swap death). The τ path already solves this exact problem with
`line_chunk` + `_TAU_CACHE_MAX_BYTES` — mirror it: process sites in blocks, and/or
have `scan` auto-size `chunk` from `nsites` and a memory budget, and document the
`bytes ≈ 100·nsites·nfreq` rule of thumb in `scan`'s docstring.

**P2. `scan` rebuilds the Level-1 voltage matrix every chunk.**
`calc_voltages` → `voltage_matrix(lines, …)` per chunk (`bezpy/pipeline.py:318`). L is
static across chunks — the whole point of the kernel layer is build-once. Negligible
for coarse lines, real for `resample_km=1` inventories (10⁴ lines × 10³ points ×
`np.add.at` per chunk). Build L once before the chunk loop.

**P3. Power-of-2 FFT padding costs up to 2× over `scipy.fft.next_fast_len`.**
`N = 2^(⌊log₂N₀⌋+2)` lands between 2×N₀ and 4×N₀ depending on N₀. Using
`next_fast_len(N₀ + pad_samples)` would cut FFT time and the P1 memory roughly in
half in the worst case. It changes the frequency grid (results differ at rounding
level), so make it an opt-in (`nfft="fast"`) and flip the default in a minor release
with a changelog note.

Smaller, known (PLAN Q5): the quantile sketch loops sites per chunk (fine at current
scale); `_detrend_linear` loops columns — `np.polyfit` accepts 2-D y, so this is a
two-line vectorization.

Not worth doing: numba/cython (correctly ruled out), xarray-ifying numpy internals.

---

## 4. Design assessment

**Strengths worth protecting.**
- The three-level factorization (E fields → length vectors → τ transfer functions) is
  the right abstraction and is exactness-tested against the naive path.
- One canonical xarray convention (`sites_to_dataset`, `(time, obs)`, units in attrs,
  pad metadata in attrs) applied consistently across mag/mt/pipeline. This is what
  makes the package feel like one library instead of four paper repos.
- Method options named for legacy behavior (`gaps=`, `detrend=`, `method=
  "lognormal-2019"`, `extrapolate=`) with comparison tests, and a validation harness
  that decides defaults. No comparable package has this.
- The data archive (source/level provenance, merge precedence, resumable scans,
  content-driven parsers) is unusually careful engineering for a science package.
- Optional-dependency discipline is right: core is numpy/pandas/scipy/shapely; xarray,
  pysecs, matplotlib, cartopy, requests all soft-imported with actionable errors; CI
  tests the oldest supported pins.

**Gaps / decisions still open.**
- **Duplicated LOO logic.** `SECSInterpolator.loo_score` and `validation.leave_one_out`
  are two implementations of the same fold loop with different metrics. Consolidate
  (have `loo_score` delegate) before they drift.
- **`bezpy.data = None` fallback** (`bezpy/__init__.py:12-15`): a module-level
  `__getattr__` raising the actionable ImportError would beat the current
  `AttributeError: 'NoneType' has no attribute ...` a user gets from `bezpy.data.x`.
- **Q2/Q3 defaults** (kalman, asymptotic) are deliberately parked pending a real-storm
  harness entry — fine to ship, but the release notes should say the defaults may
  change and cite VALIDATION.md.
- **1D region polygons** (PLAN 3.10) remain unshippable; `how="1d"` is thus a
  documented-but-unusable path from a fresh install. Either resolve redistribution or
  state the limitation in the docs where `set_1d_regions` appears.
- **Known Q5 polish** still open and worth batching into this release: `extrapolate=`
  on single-site `Site.convolve_fft`; `TransmissionLine.calc_voltages` returning a
  float for single-time input (inconsistent with everything else; deprecate).
- **`scan` exceedance memory**: the exceedance list is unbounded; a too-low threshold
  on a long scan accumulates silently. Cheap fix: warn when it passes ~10⁶ entries.

---

## 5. Python patterns & ecosystem integration

Good: scikit-learn-style fit/predict on the interpolator; pathlib IO; dataclass with
`__str__` for `ReturnLevel`; deprecation done properly (`calc_resisitivity` alias with
`DeprecationWarning`); pandas/xarray in → same type out; pure-Python wheel; ruff in CI;
`__all__` everywhere; lazy heavy imports.

To bring up to current ecosystem baseline for the release after this one (explicitly
deferred in PLAN 3.11, still the right call not to block on):

1. **Type hints + `py.typed`** on the new public API (pipeline, stats, storms,
   validation, secs, kernels). Docstrings already carry the types; annotations are
   mostly transcription.
2. **CI-executed tutorials** (PLAN 3.12 open item): the docs' code blocks are
   verified-by-hand only. A small cached fixture dataset + `pytest --doctest-glob` or
   jupytext execution would catch API drift the moment it happens.
3. Consider `xr.register_dataset_accessor` only if users ask; the functional API is
   fine and more testable.

---

## 6. Release checklist

Must do (blockers) — **all completed 2026-07-11**:
- [x] Fix A1 (Z_var spline poisoning) + regression test
- [x] Fix A2 (coherence interleaving) + regression test (no VALIDATION.md decision
      cited band coherence, so no tables needed re-running)
- [x] Fix A3 (τ NaN guard)
- [x] Fix B2 (empty-result trap → actionable error)
- [x] B1: impedances conjugated on read for `sign_convention == -1` files
- [x] B3: `DTIR` keyword path restored; `Z_var=None` tolerated
- [x] Version bump to 0.5.0
- [x] `CHANGELOG.md` added, Delaunay-fix note included
- [x] Publish workflow (`.github/workflows/publish.yml`, PyPI trusted publishing)

Should do (small, high value) — **all completed 2026-07-11**:
- [x] B4, B5, B6, C1, C4, C5 (plus exceedance-list warning)
- [x] P1: `convolve_fft` processes sites in memory-bounded blocks
      (`site_chunk=`, ~512 MB workspace budget); sizing formula documented in `scan`
- [x] P2: voltage matrix built once before the `scan` chunk loop
- [x] `loo_score` delegates to `validation.leave_one_out` (one fold loop)
- [x] `__getattr__`-based `bezpy.data` error
- [x] C6 skip-and-warn in `storm_scan`

Defer intentionally (documented): type hints + py.typed, executed tutorials, P3 FFT
length option, 1D polygons redistribution, kalman/asymptotic default flips (await the
reference-storm harness entry), directional hazard statistics.
