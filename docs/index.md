# bezpy

**bezpy** analyzes the chain from geomagnetic field variations (**B**) through
Earth conductivity (impedance, **Z**) to geoelectric fields (**E**) and the
voltages they induce on power transmission lines — the quantities behind
geomagnetically induced current (GIC) hazard studies.

The package covers the full workflow:

1. **Get the data** (`bezpy.data`): download and cache observatory magnetic
   fields (USGS / INTERMAGNET, or the century-long BGS World Data Centre
   archive), MT transfer functions searched by region from EarthScope SPUD,
   and transmission line geometries (HIFLD); fetch Dst/Kp activity indices.
2. **Prepare it** (`bezpy.mag`): combine observatories onto one time axis,
   detrend, taper, pad, and handle gaps *without fabricating data*.
3. **Interpolate B** (`bezpy.mag.SECSInterpolator`): spherical elementary
   current systems carry the observatory field to any location, with missing
   samples masked rather than interpolated over.
4. **Compute E and V** (`bezpy.mt`, `bezpy.tl`): batched FFT convolution with
   each site's impedance — individual EMTF XML sites or whole gridded
   impedance products read the same way, and any collection saves/loads as
   one canonical netCDF; line voltages via precomputed kernels (one sparse
   matmul for all lines) or per-line telluric response functions computed
   straight from B.
5. **Scan and catalog** (`bezpy.pipeline`, `bezpy.storms`): process decades of
   data in chunks with streaming reducers, or scan explicit storm windows;
   catalog events by threshold exceedance.
6. **Hazard statistics** (`bezpy.stats`): return levels with confidence
   intervals — GPD tail fits by default, the published lognormal method by
   name.
7. **Validate** (`bezpy.validation`): leave-one-out observatory skill and a
   synthetic-truth benchmark that scores every method option (see
   [Validation results](validation.md)).

## Install

```bash
pip install bezpy
```

Optional features live behind extras so the base install stays lightweight:

| Extra | Enables |
| --- | --- |
| `data` | downloading, caching, and storing input data (`bezpy.data`) |
| `secs` | SECS interpolation of observatory fields (pysecs) |
| `plotting` | matplotlib plotting helpers (`bezpy.plot`, apparent resistivity) |
| `waveforms` | reading/writing downloaded waveforms to HDF5 |
| `iris` | downloading raw waveforms from IRIS/EarthScope (ObsPy) |

```bash
pip install "bezpy[data,secs,plotting]"
```

## Five lines from nothing to hazard numbers

```python
import bezpy

sites = bezpy.data.load_mt_sites(bbox=(-110, 35, -100, 45), min_rating=3)
b_obs = bezpy.data.load_observatory_group(["BOU", "FRD", "TUC"],
                                          "1985-01-01", "2015-12-31")
result = bezpy.pipeline.scan(b_obs, sites, chunk="1Y",
                             reduce=("max", "exceedances"), threshold=100.)
events = bezpy.storms.find_events(...)          # decluster the exceedances
rl = bezpy.stats.return_level(events["peak_value"], years_observed=31)
```

See the tutorials for each stage in detail, and
[Units and conventions](conventions.md) before mixing bezpy with other code.
