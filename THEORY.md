# Theory and numerics of the bezpy geoelectric/GIC pipeline

**Status:** Analysis / design discussion — written 2026-07-06.
**Purpose:** A first-principles look at the interpolation and integration schemes
bezpy uses to go from sparse magnetic and magnetotelluric measurements to
transmission-line voltages, an assessment of whether they are the right schemes,
and concrete alternatives (some standard, some novel) worth considering. Written
for the maintainer and a future implementer; nothing here is committed.

The short version, up front:

1. **The single most impactful change is *what* we interpolate.** bezpy currently
   interpolates the geoelectric field **E** in space. E is the product of a
   spatially *smooth* driver (**B**) and a spatially *rough* local property (the
   MT impedance **Z**). Interpolating the product blends two sites' unrelated
   geology. The physically correct decomposition is: **interpolate B (smooth,
   well-posed), assign Z locally, and form E = Z·B at each line point.** This is
   both more accurate and far more robust to sparse MT coverage. It also makes
   the existing `pysecs` dependency do real physical work.
2. **The line integral `∫E·dl` is the right quantity** — better than an
   endpoint potential difference — *because* the storm-time geoelectric field is
   **not** curl-free. Any scheme that replaces it with a single scalar potential
   silently discards the inductive EMF that actually drives GIC loops. The
   improvement here is not to abandon the line integral but to (a) resolve the
   line adequately and (b) understand and optionally decompose the field.
3. **Robustness at the edges is the weakest current link.** Delaunay
   interpolation returns NaN outside the convex hull, so lines near data gaps or
   the domain boundary silently drop out. Natural-neighbor / SECS-based /
   Gaussian-process interpolation degrade gracefully instead.

---

## 1. The physical chain and its approximations

The pipeline is:

```
B(t) at observatories ──(interp)──▶ B(t) everywhere
                                        │  × Z(ω)  (MT plane-wave approx.)
                                        ▼
                              E(t) at MT sites ──(interp)──▶ E(t) along line
                                                                │  ∫·dl
                                                                ▼
                                                          line voltage V(t)
```

Governing relations, all under the **magnetotelluric (plane-wave) approximation**
— the ionospheric/magnetospheric source is treated as a vertically-incident
plane wave, valid because storm source scales (hundreds to thousands of km) are
large compared with the EM skin depth (tens of km):

- **Surface impedance (local, per frequency):**
  E(ω, r) = Z(ω, r) · B(ω, r) / μ₀ , where Z is the 2×2 complex MT tensor at
  location r. (bezpy folds the μ₀ and unit factors into its stored Z; E in
  mV/km, B in nT.)
- **Faraday's law (why path matters):** ∇ × E = −∂B/∂t. The vertical component
  gives (∂E_y/∂x − ∂E_x/∂y) = −∂B_z/∂t. During storms ∂B_z/∂t ≠ 0, so the
  horizontal E field has non-zero curl and **is not conservative**.
- **Induced EMF on a line:** V = ∫_line E · dl. For a *conservative* field this
  is φ(a) − φ(b) (endpoints only); in general it depends on the path, and the
  EMF around a closed loop equals −dΦ_B/dt ≠ 0.

Each arrow above is an approximation with its own error budget. The rest of this
document walks them.

---

## 2. What we interpolate: E vs. (B, Z) — the central issue

### 2.1 The problem with interpolating E

bezpy computes E at each MT site (via `Site.convolve_fft`, i.e. E = Z·B in the
frequency domain) and then interpolates **E** in space onto the transmission
line points (`TransmissionLine.calc_voltages(how="delaunay"|"nn"|"1d")`).

Decompose the spatial variability of E. At a fixed frequency,
E(r) = Z(r) · B(r). Then

```
∇E ≈ (∇Z)·B + Z·(∇B).
```

- **B is spatially smooth.** Its horizontal scale is the source scale — hundreds
  of km. Over the ~tens-of-km spacing of MT sites, B barely changes.
- **Z is spatially rough.** It is a *local geological* property; adjacent MT
  sites can differ by an order of magnitude and rotate arbitrarily, because
  conductivity structure changes over km.

So the term `(∇Z)·B` dominates: **E's spatial roughness is inherited from Z, not
from B.** Linearly interpolating E between two MT sites is therefore
interpolating between two *geologically unrelated* impedances multiplied by
essentially the same B — a blend with no physical meaning between the sites. In
sparse-coverage regions the interpolated E can be badly wrong in both magnitude
and phase, and the error is invisible (no NaN, just a plausible-looking number).

(One precision, from `ARCHITECTURE.md` §8: when the *same* B is applied at
every site — bezpy's usual single-observatory usage — interpolating E and
interpolating Z with the same weights give mathematically identical voltages.
The critique above therefore has two separable parts: *blending* unrelated
geology is questionable for E and Z alike, and only local assignment avoids
it; while the E-vs-Z distinction proper becomes real once B varies between
sites, i.e. once SECS-interpolated B enters.)

### 2.2 The physically-correct decomposition

Split the interpolation along the natural scale separation:

1. **Interpolate B**, not E, to each line point. This is *well-posed*: B is
   smooth, and there is a physics-based basis for it — **Spherical Elementary
   Current Systems (SECS)**, which is exactly what `pysecs` provides. SECS
   represents the equivalent ionospheric currents and reconstructs B (and its
   gradients) between observatories in a way consistent with ∇·B = 0 and a
   physical source, and extrapolates gracefully beyond the station network.
2. **Assign Z locally** at each line point. Options, in increasing fidelity:
   nearest MT site (piecewise-constant Z), 1D physiographic region model (bezpy
   already has these), or a geologically-informed / covariance-based assignment.
   Critically, Z is treated as a *property of the ground under the point*, not
   something to blend spatially.
3. **Form E = Z(point) · B(point)** at each line point, then integrate.

This is more accurate *and* more robust: the ill-posed step (blending rough Z)
is removed, and the interpolation that remains (B) is the one that is actually
smooth. It also converts `pysecs` from "used only in the realtime plotting
script" into the load-bearing interpolation engine of the whole pipeline
(see the ROADMAP P3 item).

**Recommendation:** add a `how="secs"` (or `how="b_then_z"`) path to
`calc_voltages` / a new pipeline function: interpolate B with pysecs to the line
points, assign Z per point (reuse the existing `nn` / `1d` / delaunay machinery
*for Z*, not for E), compute E locally. Keep the current E-interpolation paths
for backward compatibility and for the case where only E is available.

### 2.3 A subtlety: interpolating the tensor Z itself

If Z is interpolated (rather than nearest-assigned), note that a 2×2 complex MT
tensor is not a set of four independent scalars. Component-wise interpolation can
produce non-physical tensors (violating causality, or the phase-tensor
constraints). Interpolating **phase-tensor invariants** or the tensor in a
rotationally-consistent form is more defensible. This matters only if we move
beyond nearest/region assignment for Z; flagged for completeness.

---

## 3. The line integral, conservativeness, and the potential-field idea

The user asked specifically whether a **potential-field** formulation — define a
scalar electric potential and integrate through it — is a better way to think
about the line integral.

### 3.1 Why the pure potential approach is *not* a free win

A scalar potential φ with E = −∇φ exists **iff** E is curl-free. From §1,
∇ × E = −∂B/∂t, so E is curl-free only when ∂B_z/∂t ≈ 0 — precisely *not* the
case during the geomagnetic storms that matter for GIC. If we forced a potential
(e.g. by least-squares fitting φ to the interpolated E), we would **discard the
rotational part of E**, which is the inductive EMF that drives currents around
network loops. For a single radial line the endpoint difference might be close;
for a *meshed grid* (the real GIC problem, solved by Lehtinen–Pirjola), the loop
EMFs are essential and a pure-potential model gets them wrong.

So `∫E·dl` along the actual path — what bezpy does — is the correct quantity, and
is *better* than the endpoint-potential shortcut common in older engineering
practice. The improvement is not to replace it.

### 3.2 Where a potential/decomposition *does* help: Helmholtz splitting

A genuinely useful and somewhat novel angle: apply a **Helmholtz (curl-free /
divergence-free) decomposition** to the interpolated horizontal E field:

```
E = −∇φ  +  ẑ × ∇ψ
     └cf┘     └─df─┘
```

- The **curl-free part** (−∇φ) is conservative: its line integral is
  path-independent and equals a true potential difference. It is the part that
  behaves like an electrostatic field.
- The **divergence-free part** (ẑ × ∇ψ) carries all the rotational EMF; by
  Faraday its curl ties to ∂B_z/∂t.

Value of doing this:
- **Diagnostics / QC:** the magnitude of the df part quantifies how
  path-dependent (and hence how sensitive to line routing and interpolation) a
  given voltage is. Large df ⇒ trust the result less; endpoint approximations
  are invalid there.
- **Scale-appropriate interpolation:** the two potentials φ, ψ are *scalar* and
  smoother than the vector E; one can interpolate/regularize them separately
  (even enforce C¹ continuity) and recombine — potentially a smoother, more
  robust E field than component-wise vector interpolation.
- **Consistency with the driver:** the df part can be tied directly to the SECS
  reconstruction of ∂B_z/∂t, closing the loop with §2.

This is a research-grade idea, not a quick change, but it directly answers the
"is there a better physics framing?" question: **decompose, don't collapse to a
potential.**

### 3.3 SECS as a unifying representation

`pysecs` already builds divergence-free and curl-free elementary-current systems
for B. The same elementary-system machinery is a natural basis for representing
the surface E field (or its source) on the sphere, giving a single, physically
constrained, sphere-aware interpolation/extrapolation framework for both B and E
— replacing three ad-hoc schemes (nn, 1d, delaunay) with one. Worth prototyping.

---

## 4. Assessment of the current interpolation schemes

`TransmissionLine.calc_voltages` offers three ways to get E onto the line:

| Scheme | Continuity | Physics respected | Sparse-data behavior | Verdict |
| --- | --- | --- | --- | --- |
| `nn` (nearest MT site) | Discontinuous (piecewise const.) | Assigns a real, measured E — no blending artifact | Robust (always defined) but blocky; jumps at Voronoi edges | Fine as a Z-assignment rule (§2), poor as an E-field |
| `1d` (physiographic region) | Discontinuous at region edges | Encodes large-scale conductivity provinces; a physical prior | Robust; ignores 3-D/local MT variation | Good regularized fallback where 3-D MT is absent |
| `delaunay` (barycentric linear) | C⁰ | Linear blend of E components — see §2 caveat | **NaN outside convex hull** — silent dropouts | Best of the three inside the hull; fragile at edges |

Cross-cutting issues:

- **Component-wise vector interpolation.** All three treat E_x and E_y as
  independent scalars. Barycentric linear interpolation of a vector is at least
  frame-consistent, but see §2 for why interpolating E at all is the deeper
  problem.
- **Convex-hull cliff (delaunay).** `fast_interpolate` / `calc_voltages` set
  points with any negative barycentric weight to NaN. A single line vertex
  outside the hull NaNs the whole line's voltage. For CONUS-scale runs with
  irregular MT coverage this quietly removes lines — a correctness-of-coverage
  issue that should at least be *logged*, and ideally handled (below).
- **Projection.** The gnomonic projection before Delaunay is a sound choice
  (great circles → straight lines), better than the cos-latitude scaling
  fallback; good.

### 4.1 Better interpolators for sparse/irregular data

Ranked by effort-to-value:

1. **Natural-neighbor (Sibson).** C¹ almost everywhere, no convex-hull cliff in
   the interior, smoother than barycentric, still local and fast. A strong
   drop-in improvement over `delaunay` for the E- or B-field.
2. **SECS (pysecs).** Physics-based, sphere-aware, graceful extrapolation. Best
   for B (§2). Requires the (B, Z) refactor to shine.
3. **Inverse-distance / Gaussian-process (kriging).** GP with a covariance
   length scale set from the *B* correlation length gives smooth interpolation
   **and** an uncertainty estimate at every point — feeding directly into E-field
   error bars (ROADMAP P12). Heavier, but principled for sparse data.
4. **Keep `nn`/`1d` as robust fallbacks / Z-assignment rules**, not as the E
   interpolant of record.

---

## 5. The integration scheme

`calc_voltages` computes `V = Σ_i E3d_i · dl_i / 1000`, where `dl_i` is the
haversine displacement (decomposed into north/east components) between
consecutive line vertices and `E3d_i` is the interpolated field at those
vertices. Assessment:

- **This is a first-order (rectangle/midpoint-ish) quadrature of `∫E·dl`.** Its
  accuracy is set by how well the vertices sample E's spatial variation along the
  line. Transmission-line geometries from HIFLD have irregular vertex spacing;
  long straight spans between distant vertices are under-sampled if E varies
  across them.
- **`dl` decomposition is correct.** Splitting the great-circle step into
  (north, east) components and dotting with (E_north, E_east) is the right inner
  product; the haversine components are accurate at these scales.
- **Improvements:**
  - **Resample the line to a target segment length** (e.g. ≤ a few km, or a
    fraction of the MT-site spacing) before integrating, so the quadrature
    resolves the E field the interpolant can actually represent. Cheap, and
    removes a hidden dependence on the source data's vertex density.
  - **Trapezoidal rule** (average E at the two endpoints of each segment) instead
    of a single sample per segment — second-order, essentially free.
  - Both matter more once E varies smoothly along the line (i.e. after the §2/§4
    interpolation improvements); with blocky `nn` E they are moot.

---

## 6. Frequency-domain transfer, causality, and band limits

Two numerical-physics issues in the B→E step (`Site3d.spline_interp` /
`convolve_fft`, and the alternative `DTIR`):

- **Causality of interpolated Z.** `spline_interp` interpolates Z across period
  independently in real and imaginary parts (or magnitude/phase). Nothing
  enforces the **Kramers–Kronig / Hilbert-transform** relation between Re Z and
  Im Z, so an interpolated Z can be mildly non-causal, producing small acausal
  ringing in the time-domain E. The `DTIR` (discrete-time impulse response)
  method **does** impose a causal, finite-support response and is the more
  physically-grounded route for time-domain work; this trade-off is worth
  documenting and, ideally, offering a causality-preserving interpolation
  (e.g. fit a causal rational/kernel model to Z).
- **Band-limit extrapolation.** Outside the measured period band `spline_interp`
  uses `ext=1` (zeros) — a hard band-stop. Physically, Z has known asymptotics:
  Z ∝ √(iω) for a uniform half-space at high frequency, and Z → 0 as ω → 0
  (DC). Extrapolating with these asymptotics (rather than zeroing) treats the
  spectral tails of the geoelectric field more faithfully, which matters for the
  short-period content that dominates dB/dt-driven E during substorm onsets.
- **Windowing / detrending** before the FFT (currently the caller's
  responsibility) affects low-frequency leakage; the pipeline should own a
  documented, physically-sensible default (ROADMAP P5).

---

## 7. Robustness with sparse input — summary of levers

Pulling the threads together, in priority order for "robust, simple, performant"
against sparse and irregular data:

1. **Interpolate B (SECS), apply Z locally, form E per point** (§2). Removes the
   ill-posed blend and uses graceful physical extrapolation. Highest impact.
2. **Replace the convex-hull NaN cliff** with natural-neighbor or SECS, and at
   minimum **log dropped/extrapolated line points** instead of silently NaN-ing
   (§4). Turns silent coverage loss into a visible, quantified one.
3. **Resample lines + trapezoidal integration** (§5). Cheap accuracy, removes
   dependence on source vertex density.
4. **Carry uncertainty** end-to-end (GP interpolation variance + Z variance →
   E and V error bars). Sparse data *should* produce wide error bars; making that
   explicit is both more honest and more useful than a bare number.
5. **Causality-aware Z interpolation / DTIR** and **physical band-limit
   extrapolation** (§6) for the time-domain fidelity of the tails.

None of these change the public API destructively; they are new `how=` options,
a new pipeline path, and internal robustness fixes, consistent with keeping the
existing low-level building blocks intact.

---

## 8. Open questions worth a literature pass

- How large is the divergence-free (path-dependent) fraction of E in practice for
  CONUS storms, and how much does line routing therefore matter? (Motivates §3.2.)
- Does a phase-tensor-consistent Z interpolation measurably beat nearest-site
  assignment where MT is dense enough to test?
- Is a full SECS-for-E representation better than SECS-for-B + local Z, or does
  the local-geology roughness of Z defeat any smooth-basis representation of E?
  (Almost certainly the latter — but worth confirming.)
- Benchmark: build a synthetic case with a known 3-D conductivity model where the
  "true" E and line voltages are computable, and score each interpolation/
  integration scheme against it. This is the cleanest way to turn the arguments
  above into numbers.

---

### Relationship to the other planning docs

- `ARCHITECTURE.md` (written 2026-07-07) takes the physics choices argued here
  (§2's interpolate-B/assign-Z-locally, §5's quadrature) and shows they can be
  contracted into precomputed per-line/per-transformer transfer functions —
  generalizing Kelbert & Lucas (2020) — so the "cost of Z at every line point"
  concern disappears at runtime. Read it as the computational half of this doc.
- `ROADMAP.md` P3 (pysecs integration) is the enabling dependency for §2 here;
  this document argues that P3 is not just "add B interpolation" but *the*
  correctness fix for the whole spatial pipeline.
- `ROADMAP.md` P12 (uncertainty propagation) is §7.4 here.
- The GIC network solver (`ROADMAP.md` P9) consumes the per-line `∫E·dl` EMFs
  discussed in §3; the Helmholtz/loop-EMF analysis there is what makes the
  network solve physically meaningful.
