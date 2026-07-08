# One storm, B to E

Compute geoelectric fields at MT sites for a single storm — the May 2024
(Gannon) storm — from nothing.

!!! note
    This tutorial downloads real data (a few MB of observatory minute data
    and MT transfer functions) into the bezpy cache on first run.

## 1. Get the inputs

```python
import bezpy

# Observatory magnetic fields around the storm (±2 days for context)
for code in ["BOU", "FRD", "TUC", "NEW", "FRN"]:
    bezpy.data.download_observatory(code, "2024-05-08", "2024-05-14")
b_obs = bezpy.data.load_observatory_group(["BOU", "FRD", "TUC", "NEW", "FRN"],
                                          "2024-05-08", "2024-05-14")

# MT sites over the region of interest (EarthScope EMTF transfer functions,
# downloaded previously with bezpy.data.download_mt_spud / download_mt)
sites = bezpy.data.load_mt_sites(bbox=(-110, 35, -100, 45), min_rating=3)
```

## 2. One call

`bezpy.calc_geoelectric_field` runs the full chain — prepare (detrend, taper,
pad), SECS-interpolate B onto each site, batched FFT convolution with each
site's impedance, trim the pad:

```python
e = bezpy.calc_geoelectric_field(
    b_obs, sites,
    interp="secs",                            # each site sees its local field
    window=("2024-05-10", "2024-05-12"),      # the storm; pad added/removed
)
print(e)          # Ex, Ey (time, site) in mV/km, lat/lon coordinates
```

## 3. Or step by step

The one-call form is a composition of pieces you can run (and swap)
yourself:

```python
b = bezpy.mag.prepare(b_obs, window=("2024-05-10", "2024-05-12"),
                      pad="auto", detrend="linear", taper="tukey",
                      gaps="mask")

interp = bezpy.mag.SECSInterpolator(grid="auto")
interp.fit(b)
b_sites = interp.predict(sites)               # Bx, By, Bz at the MT sites

from bezpy.mt.site import SiteCollection
ex, ey = SiteCollection(sites).convolve_fft(
    b_sites["Bx"].values, b_sites["By"].values,
    dt=b.attrs["dt_seconds"])
```

## 4. Look at it

```python
import matplotlib.pyplot as plt

peak = int(abs(e["Ex"]).mean("site").argmax())     # storm peak sample
fig, ax = plt.subplots()
bezpy.plot.plot_vectors(
    ax,
    latlons=list(zip(e["latitude"].values, e["longitude"].values)),
    vectors=list(zip(e["Ex"].values[peak], e["Ey"].values[peak])),
    scale="log")
plt.show()
```

## Notes

- **Gaps**: with `gaps="mask"` (default), missing observatory samples are
  excluded from the SECS fit sample-by-sample. Nothing is interpolated, no
  observatory is dropped.
- **Regularization**: tune `epsilon` on your own network with
  `interp.loo_score(b, epsilon=...)` instead of trusting the default.
- **Band edges**: pass `extrapolate="asymptotic"` to
  `calc_geoelectric_field` to continue the impedances physically past the
  measured band instead of hard band-stopping (see
  [validation results](../validation.md)).
