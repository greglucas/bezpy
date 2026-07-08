# Computational architecture of the B → V → GIC chain

**Status:** Analysis / design discussion — written 2026-07-07. Companion to
`THEORY.md` (physics of interpolation/integration) and `ROADMAP.md` (work items).
**Question addressed:** What is the computational cost of evaluating the MT
impedance Z at points *along transmission lines* (rather than only at MT sites)?
Is that overhead hidden under the FFT cost, or is it a real degradation? And in a
greenfield design, what is the ideal factorization of the whole computation, from
observatory B to the quantity people actually care about — GIC in the lines and
transformers?

The short version, up front:

1. **The FFT is not where the time goes.** Measured on this machine, evaluating
   one site's Z (8 spline evaluations) onto a storm-length frequency grid costs
   **5–30× more than the inverse FFT itself**, and the current time-domain
   per-vertex interpolation in `calc_voltages` costs more than either. The
   intuition "Z-at-every-point is expensive" is correct *for a naive design* —
   but the naive design is never necessary.
2. **Every step of the chain is linear and time-invariant (LTI).** B-interpolation,
   Z-multiplication, the line integral, and the Lehtinen–Pirjola network solve are
   all *linear, per-frequency* operations with *static* coefficients. Linear
   operators compose associatively, so the geometry- and geology-dependent parts
   can be contracted into small, precomputed kernels, and the storm-time cost
   collapses to a handful of FFTs — **independent of the number of line points**.
3. **This is exactly the generalization of Kelbert & Lucas (2020).** The
   "power line telluric response function" of that paper is the contraction
   `τ(ω) = ∫ Z(r,ω)·dl` under a uniform-B assumption, and its Eqs. 14–15 already
   carry the contraction through the Lehtinen–Pirjola solve to per-transformer
   system coefficients. Folding the (also linear, also static) B-interpolation
   weights into the same contraction removes the uniform-source assumption —
   the paper's own flagged future work, and per its Table 3 the dominant error
   for long (≳500–1000 km) lines — at **zero additional runtime cost** (see §8).
   The resulting per-transformer **GIC transfer functions** H(ω) are directly
   comparable to empirically estimated B→GIC transfer functions at monitored
   transformers: a falsifiable end-to-end test.
4. **An immediate, code-level corollary:** for *every* interpolation scheme bezpy
   currently offers (`nn`, `1d`, `delaunay`), the line voltage is a fixed sparse
   linear functional of the per-site E fields: `V_l(t) = Σ_s L_ls · E_s(t)`.
   The `(nsites, 2)` "effective length vector" `L_l` per line can be precomputed
   once; `calc_voltages` then never touches individual line vertices at runtime.
   Same answer, orders of magnitude faster, and the vertex count becomes a
   *precompute-time* quadrature knob (resample as finely as you like — it's free
   at runtime).

---

## 1. Where the time actually goes (measurements)

Benchmarked 2026-07-07, single core, numpy FFT, scipy `UnivariateSpline` with
~30 log-spaced knots (i.e. the `Site3d.spline_interp` pattern); machine-dependent,
so treat as ratios, not absolutes. Two representative storm runs, 3 days of data:

| Cost per unit | 1-min cadence (N=2¹⁴, nfreq=8.2k) | 1-s cadence (N=2¹⁹, nfreq=262k) |
| --- | --- | --- |
| `rfft` / `irfft` (one series) | 0.14 ms | 12–18 ms |
| Z spline eval, one site/point (8 splines, full grid) | 4.4 ms | 90 ms |
| `Site1d.calcZ` (layered-earth, analytic) | 9 ms | 478 ms |
| Frequency-domain Z·B multiply (4 complex mults) | 0.08 ms | 4 ms |
| Current `calc_voltages` delaunay path, one 30-vertex line | 22 ms | 1730 ms |

Three conclusions:

- **"It's already taken up by the FFTs" is not right.** Per point, the Z spline
  evaluation onto the full FFT frequency grid is the dominant cost — 5–30× the
  FFT. A naive "evaluate Z and run an FFT pair at every line point" design pays
  ~5 ms/point (minute cadence) or ~110 ms/point (second cadence). For a CONUS
  run with lines resampled at km scale (~10⁵–10⁶ points), that is minutes to
  days of compute, plus prohibitive memory (below). So the worry is justified —
  *for that design*.
- **The current runtime path is already the slowest piece.** The time-domain
  fancy-indexed interpolation in `calc_voltages` (22 ms per 30-vertex line at
  minute cadence) applied to ~10⁵ HIFLD lines is ~40 minutes per storm — and it
  recomputes, at every call, contractions that are static properties of the
  geometry. §3 removes this entirely.
- **Memory is the real wall at 1-s cadence.** The cached `SiteCollection`
  impedance array `(4, nsites, nfreq)` complex128 is 0.8 GB for 1500 sites at
  minute cadence but **25 GB** at 1-s cadence. Any design that stores per-*point*
  spectra multiplies this by npts/nsites. The kernel factorization (§3) stores
  per-*line* (or per-transformer) kernels at MT-style period sampling instead:
  megabytes.

---

## 2. The one structural fact: the whole chain is LTI

Write the full pipeline in the frequency domain. For a line ℓ with quadrature
points rᵢ and displacement vectors dlᵢ:

```
V_ℓ(ω) = Σᵢ dlᵢᵀ · Z(rᵢ, ω) · B(rᵢ, ω)                      (line integral, E = Z·B)
B(rᵢ, ω) = Σₖ W(rᵢ)ₖ · B_obs,k(ω)                            (any linear B interpolation)
GIC_t(ω) = Σ_ℓ A_tℓ · V_ℓ(ω)                                  (Lehtinen–Pirjola / nodal solve)
```

Every coefficient in sight — `dlᵢ`, `Z(rᵢ,ω)`, `W(rᵢ)ₖ`, `A_tℓ` — is **static**:
it depends on geometry, geology, network topology, and the observatory set, but
not on the storm. (SECS included: fitting pole amplitudes per time step is a
fixed linear solve `amps(t) = S⁺ B_obs(t)`, so prediction is a static matrix
`W = T_pred S⁺` as long as the station set and gap mask are fixed. Robust/Kalman
variants break strict linearity — see §6.)

Therefore the map from observatory magnetic spectra to any output — E at a
point, V on a line, GIC in a transformer — is a **linear time-invariant transfer
function**, and matrix associativity lets us contract the static factors in any
order, once, ahead of time:

```
V_ℓ(ω)   = Σₖ K_ℓk(ω) · B_obs,k(ω)      K_ℓk(ω) = Σᵢ dlᵢᵀ Z(rᵢ,ω) W(rᵢ)ₖ     (1×2 complex per line per input)
GIC_t(ω) = Σₖ H_tk(ω) · B_obs,k(ω)      H_tk(ω) = Σ_ℓ A_tℓ K_ℓk(ω)           (1×2 complex per transformer per input)
```

Storm-time cost: `2·n_obs` forward FFTs, one multiply-accumulate over inputs,
and **one inverse FFT per output channel** — no dependence on npts at all. The
question "does evaluating Z at every line point add a lot of computation?"
dissolves: all per-point work happens once, at kernel-build time, on a coarse
period grid (§4).

### Where this sits in the literature

- **Kelbert & Lucas (2020)** [10.1029/2020SW002467] is the `n_obs = 1`,
  uniform-B special case: `τ_ℓ(ω) = Σᵢ dlᵢᵀ Z(rᵢ,ω)` (its Eqs. 12–13), a
  "static, frequency-dependent power line telluric response function ...
  considered part of the power grid system model," and Eqs. 14–15 already
  contract τ through the network into per-transformer coefficients — i.e. the
  paper covers Levels 2 *and* 3 of the table below, under per-transformer
  locally-uniform B. See §8 for what the full text settles.
- The generalization above folds **nonuniform B** (SECS or any linear
  interpolation) into the same static kernel. The paper's own error
  decomposition (its Table 3, discussed in §8) shows the uniform-B assumption
  is negligible for ~100 km lines but essentially *all* of the error for the
  1,000-km coast-parallel line — and the authors explicitly flag relaxing it
  as future work. Folding the interpolation weights in does exactly that.
- Carrying the contraction through the network (`H_tk`) reproduces, from first
  principles, the *empirical* B→GIC transfer functions long estimated at
  monitored transformers (Pulkkinen/Viljanen-style system identification, and
  the New Zealand work of Ingham, Rodger and colleagues). Physics-predicted
  H(ω) vs. empirically-fit H(ω) at instrumented transformers is the cleanest
  end-to-end validation this field has available, frequency by frequency —
  far more diagnostic than comparing GIC time series peaks.

---

## 3. The immediate corollary for bezpy: effective length vectors

You do not need the full kernel machinery to reap most of the runtime win.
Look at what `calc_voltages` computes for each of its three `how` options:

```
V_ℓ(t) = Σᵢ ( Σⱼ wᵢⱼ E_{s(i,j)}(t) ) · dlᵢ          (delaunay: j = 3 vertices; nn/1d: j = 1)
       = Σ_s [ Σ_{i,j : s(i,j)=s} wᵢⱼ dlᵢ ]ᵀ · E_s(t)
       = Σ_s L_ℓs · E_s(t)
```

`L_ℓ` — call it the **effective length vector** — is an `(nsites, 2)` sparse
array (each line touches only a handful of sites/regions): the dl-weighted
footprint of line ℓ on each site's E field. It is a pure function of geometry +
interpolation scheme, computable once per (line set, site set, `how`). Then:

- `V(t) = L · E(t)` — one sparse matmul for the *entire* line inventory, in the
  time domain, no per-vertex work, no per-line loop. Replaces the 22 ms/line
  fancy-indexing with microseconds/line, and reproduces the current results
  **exactly** (it is the same sum, reassociated).
- Line **resampling** (THEORY.md §5) becomes free at runtime: resample to km
  segments when building `L`, and `L` stays the same size (it's per-site, not
  per-vertex). The quadrature-resolution knob no longer has a runtime cost.
- The out-of-hull NaN policy is applied when building `L` (mark the line, or
  renormalize weights) instead of NaN-ing silently at runtime — fixing the
  THEORY.md §4 "silent dropout" issue in the same stroke.
- This is a small, non-breaking change: a
  `TransmissionLine.build_line_functional(...)` / module-level
  `build_voltage_operator(lines, sites, how=...)` returning a scipy sparse
  matrix, with `calc_voltages` kept as-is for compatibility.

The full frequency-domain kernel (§2) is `K_ℓ(ω) = Σ_s Z_s(ω) L_ℓs` — i.e. the
effective length vector *is* the geometric half of the Kelbert & Lucas response
function. Whether to apply `L` in the time domain (after today's per-site FFT
convolution) or fold it with Z into `K(ω)` is purely a question of what you want
to ship and cache; the numbers are identical.

### The decision table

| Design | Per-point Z evals at runtime | FFTs at runtime | When to use |
| --- | --- | --- | --- |
| **Level 0 — today:** E at sites (FFT) → per-vertex time-domain interp → Σ·dl | 0 (but per-vertex interp every call) | 2·nsites | Backward compat only |
| **Level 1 — effective lengths `L`:** `V = L·E_sites(t)` | 0 | 2·nsites | Default. Exact same physics choices as today, 100–1000× faster voltages |
| **Level 2 — line kernels `K(ω)`:** per-line response functions vs obs inputs | 0 (Z folded in at build time) | 2·n_obs + n_lines | Operational/real-time products; shipping grid-model constants (K&L 2020's goal); when nsites ≫ n_lines of interest |
| **Level 3 — GIC transfer functions `H(ω)`:** per-transformer vs obs inputs | 0 | 2·n_obs + n_transformers | End-state: hazard scenarios, real-time GIC nowcast, validation against measured GIC |
| *(anti-pattern)* naive per-point: Z eval + FFT pair at every line point | npts × 8 splines on full grid | 2·npts | Never — strictly dominated by Levels 1–3 |

Note what happened to the original question: evaluating Z "at every point on the
line" (Level 2/3 with per-point Z assignment) costs *less at runtime* than what
bezpy does today, because the per-point work moved to a one-time build on a
coarse period grid.

---

## 4. Why kernels are cheap to build and store

Two properties make the precompute genuinely small:

1. **Z is stored and is smooth on a log-period grid.** EMTF products carry Z at
   ~30–60 log-spaced periods; `spline_interp` upsamples to the FFT grid on
   demand. Do kernel accumulation on a common **master period grid** (~40–60
   log-spaced periods spanning ~1 s–10⁵ s), not on the FFT grid: per-point Z
   evaluation there is microseconds, not the 4–90 ms of §1 (that cost was
   evaluation onto 10³–10⁵× more frequencies than the data support). Kernels
   `K`, `H` inherit Z's smoothness (they are finite sums of Z's with real
   geometric weights), so **store kernels exactly the way Z is stored** —
   complex values at master periods — and spline onto each run's FFT grid at
   runtime, once per output channel, reusing the existing `spline_interp`/cache
   machinery unchanged.
2. **With nearest-site (or 1-D region) Z assignment there are no per-point Z
   evaluations at all.** `K_ℓ(ω) = Σ_s Z_s(ω) L_ℓs` touches only the ~10²–10³
   distinct site impedances, regardless of quadrature resolution. Per-point Z
   evaluation only enters if Z itself is spatially interpolated/gridded (the
   Kelbert 0.5° grid product, or phase-tensor-consistent interpolation per
   THEORY.md §2.3) — and even then only at build time on the master grid.

Storage (complex128, 50 master periods):

| Object | Shape | Size |
| --- | --- | --- |
| `L` effective lengths, 10⁵ lines × ~5 touched sites | sparse (10⁵ × nsites × 2) | ~30 MB |
| `K` line kernels, 10⁵ lines × 10 obs inputs × 2 | dense (10⁵ × 10 × 2 × 50) | 1.6 GB (float32: 0.8) |
| `H` transformer kernels, 2·10⁴ transformers × 10 obs × 2 | dense | 320 MB |
| *(compare)* cached `(4, 1500, nfreq)` site Z at 1-s cadence | dense | 25 GB |

`K`/`H` for a *regional* study or a chosen line subset is megabytes. A netCDF
of `H(transformer, obs_input, component, period)` is a shippable, versioned
artifact: "the grid's GIC response functions, given this MT compilation and
this network model" — exactly the "part of the power grid system model" framing
of Kelbert & Lucas (2020), extended to nonuniform sources.

Rebuild triggers are rare and explicit: new MT compilation, changed line
routing/network topology, changed observatory set. Storm data never triggers a
rebuild.

---

## 5. Greenfield design

If starting from scratch, the architecture that falls out of §2 is a two-stage
system with a frequency-axis operator algebra in the middle:

```
BUILD (slow, rare, cacheable)                      RUN (fast, per storm)
─────────────────────────────                      ─────────────────────
MT compilation ──► Z_s(ω_master)                   B_obs(t) ──rfft──► B_obs(ω)
line geometry ──► resample ──► quadrature dl       GIC_t(t) = irfft( Σₖ H_tk(ω_fft)·B_obs,k(ω) )
obs network  ──► SECS weights W                    V_ℓ(t)  = irfft( Σₖ K_ℓk(ω_fft)·B_obs,k(ω) )
grid model   ──► Lehtinen–Pirjola A                E maps  = per-cell kernels, same machinery
        └────────► contract ──► K(ω), H(ω) ──► netCDF
```

Design rules that keep it clean:

- **Everything between the two FFTs is a per-frequency matrix.** Represent each
  stage (interpolation, impedance, quadrature, network) as an operator with an
  explicit frequency axis; composition is einsum. New physics (e.g. a better Z
  assignment) is a new factor, not a new pipeline.
- **Contract from the output side.** You rarely want all intermediates; ask for
  transformers → build H; ask for line voltages → stop at K; ask for E maps →
  per-cell kernels. Intermediates (E at every point) exist only if requested.
- **Uncertainty rides the same algebra** (ROADMAP P12): Var(K) = Σ |weights|²
  Var(Z) per frequency, propagated by the same contractions; SECS prediction
  variance enters as Var(W)·|B|² terms. Error bars on GIC come out of the same
  einsum with squared moduli — cheap, and unprecedented in operational products.
- **Time-domain remains a first-class citizen at the edges** — preprocessing
  (detrend/gap/window, ROADMAP P5) before the forward FFT; DTIR-style causal
  impulse responses are the inverse-FFT of the same kernels if a streaming
  time-domain implementation is ever needed (a causal FIR of K(ω) gives
  real-time V(t) with O(filter length) per sample — the natural nowcast form).

What still requires per-point, time-domain work (and honestly so):

- **Nonstationarity:** changing observatory availability mid-storm (gap masks
  change W), Kalman/robust SECS fitting, network topology changes (switching).
  Handle by segmenting time into stationary blocks with per-block kernels, or
  fall back to Level 1.
- **Nonlinearity:** transformer saturation / reactive power / harmonics live
  downstream of GIC(t) and consume the time series — the LTI chain still
  produces their input.
- **Research on the interpolation itself** (THEORY.md §8 benchmark): you need
  the full field to study the error of collapsing it.

---

## 6. Physics: is integration the right thing, and what the literature glosses

The end hazard is GIC in transformer windings (heating, harmonics, reactive
loss). The path there necessarily runs through per-line EMFs — the network
solve takes `∫E·dl` per line as its sources, and THEORY.md §3 establishes that
no endpoint-potential shortcut is valid because storm-time E is not curl-free.
So *some* line integration is unavoidable. The architectural point of this
document is that "unavoidable" ≠ "at runtime": the integral is a static linear
functional and belongs in the build stage. Given that, an honest list of what
current practice (ours and the field's) gets subtly wrong or leaves implicit:

1. **The uniform-B assumption is quietly everywhere.** `SiteCollection.convolve_fft`
   takes a single `mag_x, mag_y` — one observatory's field applied to every MT
   site. Kelbert & Lucas (2020) makes the same assumption explicitly per line.
   For CONUS-scale studies driven by one reference observatory this is likely a
   leading error term — and it is *free* to fix inside the kernel factorization
   (fold W in). Nobody should pay 50% error for a term that costs nothing at
   runtime.
2. **Interpolating E across geology** (THEORY.md §2) — the kernel view removes
   the last practical excuse for it: assigning Z locally at line points used to
   look expensive; it is now a build-time detail. `L`/`K` with nearest-site Z is
   both more physical and faster than delaunay-on-E.
3. **Quadrature at HIFLD vertices** — line vertices are digitization artifacts,
   not quadrature nodes. Resampling to uniform km-scale segments is free once it
   happens at build time (§3). Any accuracy study comparing "integration
   methods" without controlling vertex density is partially measuring the
   digitizer.
4. **Zeroing Z outside the measured band** (`ext=1`) — the kernels inherit
   whatever band policy Z has, so the known-asymptotics extrapolation
   (Z ∝ √(iω) high-frequency tail; Z → 0 at DC) from THEORY.md §6 should be
   applied *before* kernel building, or the shipped response functions embed a
   hard band-stop that silently truncates substorm-onset dB/dt response.
5. **Point-sampled galvanic distortion.** A measured Z carries static,
   real-but-local distortion from near-surface structure within ~km of the
   electrodes. The line integral is a spatial average, which is *favorable* —
   small-scale E structure genuinely averages out along a 100-km line — but
   nearest-site assignment extends one point's local distortion across its whole
   Voronoi cell (~70 km for USArray spacing). This is a data-sparsity error no
   scheme removes; the kernel formulation at least makes sensitivity tests cheap
   (rebuild K with distortion-stripped vs raw Z; the runtime is unchanged).
   Worth a literature pass on distortion effects on GIC estimates specifically.
6. **The plane-wave (local-uniform-source) approximation degrades exactly where
   hazard peaks** — subauroral latitudes during expansions, where source scales
   shrink toward the skin depth. Folding SECS-interpolated B into kernels
   captures the *geometric* nonuniformity of B but still applies a plane-wave Z
   pointwise ("local plane-wave approximation"). This is standard practice and
   defensible; it should nonetheless be stated as an assumption of the shipped
   response functions, and it bounds how far north the product should be
   trusted without full induction modeling.
7. **Validation is done on time-series peaks, not transfer functions.** Since
   the physics chain *is* a transfer function, compare it as one: predicted
   H(ω) vs. empirically identified H(ω) at transformers with GIC monitors,
   coherence and phase per frequency band. Peak-GIC scatter plots conflate
   source, Earth, and network errors; the frequency domain separates them.

---

## 7. Concrete next steps (proposed, not committed)

Ordered so each step is useful on its own and none blocks the previous:

1. **Effective length vectors (Level 1).** `build_voltage_operator(lines, sites,
   how=...)` → sparse `L`; verify `L·E == calc_voltages(E, how)` to machine
   precision on the existing test suite; add line resampling as a build option.
   Small, non-breaking, immediately 100–1000× on the voltage stage. (Extends
   ROADMAP P8; enables P2's clean pipeline.)
2. **Line kernels K(ω) at master periods (Level 2).** `K = Σ_s Z_s L_s` stored
   Z-style; a `calc_voltages_fft`-equivalent that goes obs-spectra → V(t) with
   one irfft per line. Reproduces Kelbert & Lucas (2020) as the `n_obs=1` case —
   a good regression target against that paper's published numbers.
3. **The error-decomposition experiment.** On a storm with dense coverage
   (e.g. May 2024, TA footprint): compute V by (a) full per-point integration
   with SECS-interpolated B [truth proxy], (b) uniform-B kernels [K&L 2020],
   (c) SECS-weighted kernels. The (b)−(a) vs (c)−(a) gap measures how much of
   the ≤50% is the uniform-source assumption — the publishable question this
   architecture poses. Line-length stratification (short vs >200 km lines) will
   show where it matters.
4. **GIC transfer functions H(ω) (Level 3)** — lands with ROADMAP P9's network
   solver; validate against the Horton/NERC 20-bus test case analytically, and
   against any accessible measured-GIC dataset as H(ω) (see §6.7).
5. **Uncertainty axis** — Var(Z) → Var(K) → Var(H) through the same
   contractions (ROADMAP P12), so shipped response functions carry error bars.

## 8. Notes from the full text of Kelbert & Lucas (2020)

Read in full 2026-07-07 (author-supplied PDF). What it settles, and what it
leaves open, relative to the design above:

**Construction of τ.** Eqs. 12–13 define
`τ^N = ∫(Z_xx dx + Z_yx dy)`, `τ^E = ∫(Z_xy dx + Z_yy dy)`, so
`V(ω) = τ^N(ω)B_x(ω) + τ^E(ω)B_y(ω)` — τ in V/nT. They were built by
**Delaunay-interpolating the impedance tensor components** (not E) onto lines
discretized at **1-km segments**, and evaluated at the **31 periods
(~7–30,000 s)** of the empirical EMTF impedances. Two design points this
confirms: (a) the master-period-grid storage strategy of §4 is exactly what
the paper does — τ lives at MT-native period sampling, not on any FFT grid;
(b) resampling to km-scale quadrature happens at build time, where it is free.

**An identity worth stating explicitly.** Under uniform B, "interpolate E
then integrate" (bezpy's current `delaunay` path fed one observatory) and
"interpolate Z then integrate" (the paper's τ) are **mathematically
identical** — the same linear functional applied on either side of a multiply
by the same B. So the τ formulation is not a different physics choice from
today's bezpy; it is today's bezpy, reassociated (the prototype in §9 confirms
this to rounding error). `THEORY.md` §2's E-vs-Z interpolation critique only
distinguishes the two once B varies between sites; its real target is
*blending* (of either E or Z) versus local assignment.

**Error decomposition — who owns the ≤50%?** Table 3 compares each method's
RMS difference from the full 3-D reference. The "3-D no mag" column (full
per-point integration but spatially uniform B) isolates the uniform-B error;
"Modified" adds to that only the τ band-limitation to 31 periods. For the
100-km line A1: Modified 0.52 A vs no-mag 0.06 A — there the residual is
dominated by τ's **period sampling/band handling**, not by uniform B. For the
1,000-km line B4: Modified 2.21 A vs no-mag 2.15 A — there uniform B is
essentially the entire error. Consequences: (a) folding B-interpolation
weights into τ (the §2 generalization) fixes the long-line/high-latitude
regime, and the paper explicitly anticipates this ("could be relaxed with
future improvements to magnetic field monitoring and modeling"); (b) for
short lines, the fix is instead **band-edge treatment of τ** — asymptotic
extrapolation rather than the hard 7–30,000 s stop (§6.4) — and denser
master-period sampling, both build-time knobs.

**The paper already does Level 3.** Eqs. 14–15 fold τ through the
Lehtinen–Pirjola matrix `K_ji/R_ki` into per-transformer system coefficients,
with B taken *locally at each transformer* — so B may vary across the grid,
just not over any one transformer's connected lines ("a milder assumption ...
the geomagnetic field is averaged along the lengths of the power lines").
What this document adds beyond the paper is therefore: (i) the exact Level-1
reassociation of bezpy's existing runtime path (§3, new), (ii) per-input
(multi-observatory / SECS) kernels removing the residual uniform-B assumption
(§2), (iii) uncertainty propagation through the same contractions (§5), and
(iv) validation-as-transfer-functions (§6.7), which the paper's framing makes
natural but does not pursue.

**Measured speedup in the paper.** Its Table 1 (eight lines, Halloween-storm
run): voltage stage 1.66 s → 0.027 s (~60×), *plus* skipping the 3.9–4.8 s
E-field stage entirely. Consistent with the §9 prototype numbers at larger
line counts.

**Connection to Weigel & Cilliers (2019).** That work showed
frequency-dependent transformer coefficients fit from *measured GIC* beat
frequency-independent ones; Kelbert & Lucas provide the way to construct those
coefficients from MT data alone, no GIC monitor required. Our H(ω) validation
proposal (§6.7) is the reunion of the two: construct from physics, verify
against the empirical fits wherever monitors exist.

---

## 9. Prototype results (`telluric-kernels` branch, 2026-07-07)

Implemented in `bezpy/tl/kernels.py` (+ `tests/test_kernels.py`,
`benchmarks/bench_kernels.py`), non-breaking, ~230 lines:

- `voltage_functional` / `voltage_matrix` / `apply_voltage_matrix` — Level 1
  effective length vectors for all three existing `how` schemes, applied as a
  sparse matmul. Out-of-hull lines are flagged `valid=False` at build time
  instead of silently NaN-ing at runtime.
- `TelluricResponse` — Level 2 kernels. Exact mode (`periods=None`) evaluates
  τ on the run's FFT grid straight from `site.calcZ` (validation); master-grid
  mode stores τ at ~31 log-spaced periods and spline-interpolates in
  log-period per run, mirroring `Site3d` Z handling, with line-chunked
  evaluation to bound memory. `calc_voltages(mag_x, mag_y, dt)` mirrors
  `Site.convolve_fft` conventions and returns `(ntimes, nlines)` volts.

Test suite (10 tests, all passing; full repo suite 65 passed): Level 1
reproduces `calc_voltages` **exactly** (rtol 1e-12) for `nn`/`1d`/`delaunay`;
exact-mode τ matches the full convolve-then-interpolate pipeline to rtol
~1e-9 (the reassociation identity of §8); master-grid τ agrees with exact τ
to ~1e-8 relative RMS on smooth synthetic impedances.

Benchmark (one 30-day, 1-minute chunk; 100 synthetic MT sites, 200 lines of
10 vertices; single core):

| Path | E-field stage | Voltage stage | Total (warm) | Agreement |
| --- | --- | --- | --- | --- |
| A. current (`convolve_fft` + per-line `calc_voltages`) | 0.8 s | 7.3 s | 8.1 s | — |
| B. Level 1 (`convolve_fft` + sparse `L·E`) | 0.8 s | **0.32 s** | 1.1 s | 2e-16 rel RMS |
| C. Level 2 (`TelluricResponse`, τ at 31 periods) | — | 2.3 s | 2.3 s | 1e-8 rel RMS |

Reading the numbers:

- **Level 1 is the free lunch:** 23× on the voltage stage, 7× end-to-end,
  bit-identical output, zero API disruption. Its advantage grows linearly with
  line count and vertex count (a CONUS-scale 50k-line inventory takes the
  current path from "8 s" to "half an hour *per 30-day chunk*"; Level 1 keeps
  it at ~a minute).
- **Level 2 wins on structure, not (yet) on this benchmark's clock:** with
  only 100 sites the shared E-field stage is cheap, and C's per-line spline
  eval + irfft (~11 ms/line) exceeds B's amortized cost. C pulls ahead when
  (a) sites ≫ lines of interest, (b) cadence is 1-s and the per-site spectral
  cache (25 GB) can't be held — C never forms E — or (c) the τ themselves are
  the product to ship. Headroom: cache τ on the FFT grid across equal-length
  chunks (a historical scan reuses the same grid thousands of times), which
  removes the spline cost from the loop.
- **For the 30-year minute-cadence historical workflow** (the maintainer's
  main use case): year-long chunks at 1-min are N=2²¹ FFTs — fine — and the
  whole scan becomes: FFT the observatory record per chunk, one sparse
  matmul (B) or one irfft per line (C), giving *continuous multi-decade
  voltage/GIC series* on which storm identification and peaks-over-threshold
  statistics (ROADMAP P11) run directly, instead of per-storm reprocessing.

Not yet prototyped (deliberately): line resampling at build time, per-input
(SECS-weighted) kernels — needs ROADMAP P3 — asymptotic band-edge
extrapolation of τ, the Level-3 network contraction, and τ persistence
(netCDF). All are additive on this foundation.

---

### Relationship to the other planning docs

- `THEORY.md` §2 ("interpolate B, assign Z locally") supplies the *physics* of
  what goes inside the kernel; this document supplies the *factorization* that
  makes it free at runtime. The two are one design.
- `ROADMAP.md` P3 (pysecs) enters at kernel-build time as the static W matrix;
  P8 (performance) is largely superseded by Level 1; P9 (network solver) is the
  A matrix of Level 3; P10 (grid products) uses per-cell kernels from the same
  machinery; P12 (uncertainty) rides the contractions.
- Kelbert, A., & Lucas, G. M. (2020). Modified GIC estimation using 3-D Earth
  conductivity. *Space Weather*, 18, e2020SW002467.
  https://doi.org/10.1029/2020SW002467
