# Operational mode: causal, streaming E (and V)

`calc_geoelectric_field`/`scan` are acausal: FFT convolution needs a pad of
*future* samples on both sides of the analysis window, so neither can
produce a result as new B arrives live. `bezpy.pipeline.Nowcast` composes
the same site impedances differently — through `bezpy.mt.impulse.DTIR` (a
causal time-domain impulse response) fed by `SECSInterpolator.transfer_matrix`
(a constant, per-sample observatory-to-site B map) — into a stateful engine:
`update(new_b)` appends samples to a bounded rolling buffer and returns E
(and V) for whatever interval just became causally valid, with no window
recomputation.

## Building one

```python
import bezpy
from bezpy.mag import SECSInterpolator

sites = bezpy.data.load_mt_sites(bbox=(-110, 35, -100, 45), min_rating=3)
lines = [...]  # optional TransmissionLines

nowcast = bezpy.pipeline.Nowcast(
    sites, SECSInterpolator(), dt=60., lines=lines)
```

`nmin`/`nmax` default to `"auto"`, derived from the sites' own period band
(`nmax ~ 2x` the longest period, `nmin ~ 0.1x` the shortest). For a band of
900-18000s at `dt=60`\ s that works out to `nmax=601`, `nmin=-2` —
`warmup_samples = 2*max(nmax, |nmin|) + 1 = 1203` samples (~20 hours).

## Feeding it

```python
result = nowcast.update(new_b)   # same forms as SECSInterpolator.fit's b_obs
if result.sizes["time"]:
    save(result)                 # Ex, Ey [, V], only the newly-valid samples
```

A real poll loop:

```python
last_input_time = None
while True:
    bezpy.data.update_observatory("BOU")
    b_obs = bezpy.data.load_observatory_group(
        ["BOU", "FRD", "TUC"], start=last_input_time
    )
    if last_input_time is not None:
        # Archive intervals are inclusive; Nowcast must receive each sample once.
        b_obs = b_obs.isel(time=b_obs.time > last_input_time)
    if b_obs.sizes["time"]:
        result = nowcast.update(b_obs)
        # Track consumed input even during the start-up period with no output.
        last_input_time = b_obs.time.values[-1]
        if result.sizes["time"]:
            save(result)
    time.sleep(60)
```

`bezpy nowcast` (see below) wraps exactly this loop as a CLI command.

## Latency

Two fixed sources, independent of how often you call `update`:

* A **start-up transient** of `nowcast.warmup_samples` before *any* output
  appears — larger than `nmax` alone because of how `DTIR.convolve`'s
  `mode="same"` convolution is aligned (verified directly, not assumed). For
  the 900-18000s / `dt=60`\ s example above, that is
  ~20 hours.
* Thereafter, a **steady-state delay** of `|nmin|` samples: DTIR is only
  quasi-causal (Kelbert et al. 2017's design uses a small acausal lead), so
  a sample only becomes final once `|nmin|` further samples have arrived —
  2 minutes in the example above.

## Accuracy

Two independent, separately-measured effects:

* Periods close to `dt`'s Nyquist limit are an aliasing failure no
  `decay_factor`/`Q_choice` tuning fixes — restrict the periods passed to
  DTIR (`dtirs=`) or use finer `dt` if `nowcast`'s construction warns about
  this.
* Even entirely within a well-sampled band, DTIR reconstructs **broadband**
  B (many active frequencies, as in a real storm) much better than
  **narrowband** B (one or two dominant tones) — measured ~5% relative RMS
  vs `calc_geoelectric_field` for a 15-tone broadband synthetic, vs
  10-50% for the same band driven by just 1-2 tones. Real storm-time B is
  broadband, so this favors real operational use.

`Nowcast`'s own docstring (`bezpy.pipeline.Nowcast`) has the full accuracy
and latency reference with the measured numbers.

## What's not here

`despike_obs=` is not supported — despike's centered rolling window needs
its own future-sample lookahead on the raw observatory stream, independent
of DTIR's. Run `bezpy.mag.despike` on each chunk yourself before calling
`update` if you need it.

## CLI

```bash
bezpy nowcast --obs BOU FRD TUC --bbox -110 35 -100 45 \
    --dt 60 --out live.csv --poll-interval 60
```

Appends newly-finalized rows to a wide CSV (`time, <site>_Ex, <site>_Ey, ...`)
every poll; `--once` runs a single poll/update/write cycle and exits, for
cron/systemd-timer-driven operation instead of a long-running process.
On startup the command loads the latest three days of archived input so the
DTIR can warm up without reading an entire multi-year archive; change this
with `--startup-window` when the selected impedance band needs a longer
warmup. Inclusive archive boundaries are removed before each update, so a
sample is never applied twice. On process restart the startup interval is
replayed to rebuild in-memory DTIR state, but rows at or before the last
timestamp already in the output CSV are not appended again.
Streaming to a live dashboard is out of scope for this CLI — it is a
reference/cron-friendly sink, not a production integration.
