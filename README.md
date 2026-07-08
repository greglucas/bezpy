# Bezpy

[![DOI](https://zenodo.org/badge/115665136.svg)](https://zenodo.org/badge/latestdoi/115665136)

Bezpy is an open source library for analysis of geomagnetic (B), geoelectric (E),
and magnetotelluric impedance (Z) data within a geophysical framework.
This library contains routines for calculating the geoelectric field from the geomagnetic field in
multiple different ways.

## Features

- Geomagnetic to geoelectric field calculations
- Integration of the geoelectric field along transmission lines
- Built using established, fast, open source python libraries
    [Pandas](http://www.pandas.pydata.org/),
    [NumPy](http://www.numpy.org/),
    [SciPy](http://www.scipy.org/)

## Examples

Example notebooks can be found in [notebooks/](./notebooks/)

Example scripts for command line use can be found in [scripts/](./scripts/)

## Install

bezpy requires Python 3.10+ and the core scientific stack
(NumPy, pandas, SciPy, Shapely). The easiest method to install bezpy is
directly from PyPI using pip.

```bash
pip install bezpy
```

Optional features live behind extras so the base install stays lightweight:

| Extra | Enables | Install |
| --- | --- | --- |
| `data` | downloading, caching, and storing observatory data (`bezpy.data`; pooch + xarray + netCDF) | `pip install "bezpy[data]"` |
| `secs` | SECS interpolation of observatory magnetic fields (`bezpy.mag.SECSInterpolator`; pysecs) | `pip install "bezpy[secs]"` |
| `plotting` | apparent-resistivity / depth plots (matplotlib) | `pip install "bezpy[plotting]"` |
| `waveforms` | reading/writing downloaded waveforms to HDF5 (PyTables) | `pip install "bezpy[waveforms]"` |
| `iris` | downloading raw waveforms from the IRIS/EarthScope FDSN service (ObsPy) | `pip install "bezpy[iris]"` |
| `scripts` | running the example command-line scripts in `scripts/` | `pip install "bezpy[scripts]"` |

If you want a local install to modify anything in the code, you
can clone the git repository and install locally with these commands.

```bash
git clone https://github.com/greglucas/bezpy
cd bezpy
pip install -e ".[test]"
```

## Downloading and storing data (`bezpy.data`)

The `bezpy.data` subpackage (install with `pip install "bezpy[data]"`) takes
care of finding, downloading, caching, and organizing input data so you don't
have to hunt for files or manage them by hand. It covers all three inputs the
geoelectric/GIC workflow needs: geomagnetic observatory field data,
magnetotelluric (MT) transfer functions, and transmission-line geometries.

### Magnetic-field data

Observatory data can be downloaded from three sources:

- **`usgs`** / **`nrcan`** — the USGS web service (USGS observatories and
  relayed NRCan observatories; selected automatically by observatory code).
- **`intermagnet`** — the INTERMAGNET GIN, the global network of ~150
  observatories (pass `source="intermagnet"`).

Data is stored as **CF-style netCDF, one file per observatory per year**, in a
directory layout that records its *provenance* — both the source and the data
quality **level**:

```
<cache>/magnetic/<source>/<level>/<CODE>/<CODE>_<YEAR>.nc
```

Levels run from `variation` (real-time, available immediately) through
`provisional` and `quasi-definitive` up to `definitive` (final, baseline-
corrected, but published a year or more later). Keeping them in separate files
means slow-to-arrive definitive data never overwrites the real-time series you
already have, and the original un-culled real-time data stays available. On
load, levels are merged into a single **best-available** series: higher-quality
data wins, and lower levels backfill any gaps — including points that
definitive processing may have removed.

```python
import bezpy.data

# Real-time download of a storm interval (level="variation" by default).
bezpy.data.download_observatory("BOU", "2024-05-10", "2024-05-12")

# Later, top the real-time archive up to the present.
bezpy.data.update_observatory("BOU")

# When definitive data is published, download it alongside — it is stored
# separately and does not overwrite the real-time series.
bezpy.data.download_observatory("BOU", "2024-05-10", "2024-05-12",
                                level="definitive")

# The global network via INTERMAGNET.
bezpy.data.download_observatory("ESK", "2024-05-10", "2024-05-12",
                                source="intermagnet", level="definitive")

# Load the best available merge (xarray.Dataset), or a specific level/source.
ds = bezpy.data.load_observatory("BOU")                       # best available
definitive = bezpy.data.load_observatory("BOU", level="definitive")
df = bezpy.data.load_observatory("BOU", as_dataframe=True)

# By default higher quality wins the merge; override to let real-time data
# take precedence (e.g. to recover points definitive processing removed).
raw_first = bezpy.data.load_observatory("BOU", prefer="variation")

# See what you have, by source/level/year, and how complete it is.
print(bezpy.data.observatory_inventory(detailed=True))

# Already have IAGA-2002 files? Convert them into the fast netCDF archive.
bezpy.data.ingest_iaga("bou20240510.min", level="definitive")
```

### Magnetotelluric transfer functions

MT transfer functions (EMTF XML from the IRIS/EarthScope SPUD archive) are
stored hierarchically by **survey** — the field campaign a station belongs to:

```
<cache>/mt/<survey>/<CODE>.xml
```

```python
# Download from a SPUD EMTF page id (ds.iris.edu/spud/emtf/<id>) — bezpy
# resolves the bundle's XML attachment automatically.
bezpy.data.download_mt_spud(13283146)

# ...or from any direct EMTF-XML URL, or organize files you already have.
bezpy.data.download_mt("https://ds.iris.edu/spudservice/data/<xml_id>")
bezpy.data.ingest_mt_xml("TEST01.xml")

# List what you have, and load sites (optionally by survey / bbox / quality).
print(bezpy.data.mt_inventory())
site = bezpy.data.load_mt_site("WVO55")                        # -> bezpy.mt.Site3d
sites = bezpy.data.load_mt_sites(bbox=(-110, 35, -100, 45), min_rating=4)
```

### Transmission line geometries

High-voltage transmission line geometries (from the HIFLD Electric Power
Transmission Lines feature service) can be downloaded for a region and cached as
GeoJSON, returning ready-to-use `bezpy.tl.TransmissionLine` objects:

```python
lines = bezpy.data.download_transmission_lines(
    bbox=(-110, 35, -100, 45),   # west, south, east, north
    min_voltage=345,             # kV
)
lines = bezpy.data.load_transmission_lines()   # from cache
```

The cache location defaults to a per-user directory (via
[pooch](https://www.fatiando.org/pooch/)) and can be overridden with the
`BEZPY_DATA_DIR` environment variable or `bezpy.data.set_cache_dir(...)`.

## From observatory B to hazard numbers (`bezpy.pipeline`)

The pipeline composes the pieces below — data loading, preparation, SECS
interpolation, batched FFT convolution, and voltage kernels — into two scan
modes:

```python
sites = bezpy.data.load_mt_sites(bbox=(-110, 35, -100, 45), min_rating=3)
b_obs = bezpy.data.load_observatory_group(["BOU", "FRD", "TUC"], start, end)

# One storm window, explicitly
e = bezpy.calc_geoelectric_field(b_obs, sites, interp="secs",
                                 window=("2024-05-10", "2024-05-12"))
v = bezpy.calc_voltages(lines, e)      # Level-1 kernels, one sparse matmul

# The statistics path: scan the whole record in chunks with streaming
# reducers -- no storm pre-selection, no index bias, bounded memory.
result = bezpy.pipeline.scan(b_obs, sites, lines=lines, chunk="1Y",
                             reduce=("max", "exceedances", "quantiles"),
                             threshold={"E": 100., "V": 50.},
                             out_path="scan.nc")   # resumable as it goes

# The case-study path: explicit windows from a storm catalog
catalog = bezpy.storms.find_storms(pd.concat([dst, kp], axis=1))
maxima = bezpy.pipeline.storm_scan(catalog, b_obs, sites, lines=lines)
```

Chunk edges are protected by the convolution pad, so scan results are
invariant to the chunk size; exceedances feed `bezpy.storms.find_events`
(event catalogs) and peaks-over-threshold return-level statistics.
`leave_one_out="BOU"` runs any scan with an observatory withheld from the
SECS fit for validation.

## Return levels with uncertainty (`bezpy.stats`)

Hazard tail statistics, scipy-only, with the confidence interval built into
the result — a return level without a CI is not a result:

```python
events = bezpy.storms.find_events(e_magnitude, threshold=100., gap="12h")
rl = bezpy.stats.return_level(events["peak_value"], years_observed=31,
                              return_period=100,
                              method="gpd", ci="profile")   # default: POT + GPD tail
print(rl)   # 100-year return level: 412.3 [318.2, 604.7] (95% CI, gpd)

# The published 2019 workflow, by name, for reproduction/comparison
rl = bezpy.stats.return_level(storm_maxima, years_observed=31,
                              method="lognormal-2019", ci="bootstrap")
```

## Preparing magnetic time series (`bezpy.mag.prepare`)

The FFT-convolution workflow needs its input combined onto one time axis,
uniformly sampled, detrended, tapered, and padded so convolution edge effects
never touch the interval of interest:

```python
b = bezpy.mag.combine_observatories({"BOU": ds_bou, "FRD": ds_frd})
b = bezpy.mag.prepare(b, window=("2024-05-10", "2024-05-12"),
                      pad="auto",         # derived from the impedance band (3 x 30,000 s)
                      detrend="linear",   # legacy: "median"
                      taper="tukey",      # cosine ramps over the pad region only
                      gaps="mask")        # keep NaNs for per-sample weighting;
                                          # legacy: "interpolate" (with max_gap=...)
...                                       # SECS -> impedances -> E
e = bezpy.mag.trim_pad(e)                 # pad bookkeeping travels in attrs
```

## Interpolating magnetic fields between observatories (`bezpy.mag.SECSInterpolator`)

Spherical Elementary Current Systems (SECS) interpolate the observatory
magnetic field to arbitrary locations — e.g. onto MT site locations for
geoelectric field calculations. The interpolator wraps
[pysecs](https://github.com/greglucas/pysecs) with the configuration used in
the published hazard studies as its baseline (110 km divergence-free shell,
`epsilon=0.05`, Bz excluded), and keeps every valid sample: missing data are
masked with infinite variance rather than interpolated over or causing whole
observatories to be dropped.

```python
interp = bezpy.mag.SECSInterpolator(grid="auto")   # SEC poles from obs coverage
interp.fit(b_obs)              # xr.Dataset (time, obs) or ndarray (ntimes, nobs, 3)
b_pred = interp.predict(latlons)                   # Bx, By, Bz at new locations

# Make the regularization choice data-driven instead of folklore:
scores = interp.loo_score(b_obs, epsilon=0.02)     # leave-one-out RMSE per obs
```

## Fast line voltages: kernels and telluric response functions (`bezpy.tl.kernels`)

Every interpolation scheme in `TransmissionLine.calc_voltages` is a *linear*
map from the per-site electric fields to a voltage, so the line geometry can
be contracted out ahead of time instead of re-integrated at every time step.
`bezpy.tl.kernels` provides two levels of this factorization:

**Effective length vectors** reproduce `calc_voltages` exactly (the same sum,
reassociated) with one sparse matrix multiply for all lines at once — about
20× faster on the voltage stage for hundreds of lines:

```python
from bezpy.tl.kernels import voltage_matrix, apply_voltage_matrix

L, valid = voltage_matrix(lines, len(sites), how="delaunay")
V = apply_voltage_matrix(E, L, valid)   # (ntimes, nlines) volts from E in mV/km
```

**Telluric response functions** (Kelbert & Lucas, 2020) fold the MT
impedances into the length vectors, giving per-line transfer functions
`tau^N(ω), tau^E(ω)` so voltages come straight from the magnetic field
spectra with no electric-field intermediate:

```python
from bezpy.tl.kernels import TelluricResponse

tr = TelluricResponse.from_lines(
    lines, sites, how="delaunay",
    periods=periods,              # store tau at MT-native periods (bounded memory)
    resample_km=1.0,              # densify line quadrature at build time
    extrapolate="asymptotic",     # physical band-edge continuation instead of a band-stop
)
V = tr.calc_voltages(mag_x, mag_y, dt=60)   # (ntimes, nlines) volts from B in nT

tr.to_netcdf("tau.nc")                       # tau is a small, shippable data product
tr2 = TelluricResponse.from_netcdf("tau.nc") # ...usable with no geometry or MT sites
```

With `periods=None` the response is evaluated exactly on each run's FFT grid
and reproduces the classic convolve-then-interpolate pipeline to rounding
error. tau evaluated on the FFT grid is cached between equal-length chunks,
so scanning a long record pays the construction cost once.

## Development and testing

The test suite uses [pytest](https://docs.pytest.org/) and
[ruff](https://docs.astral.sh/ruff/) for linting. After installing the `test`
extra as shown above:

```bash
pytest          # run the test suite
ruff check .    # lint the code
```

The documentation site (mkdocs-material) builds with the `docs` extra:

```bash
pip install -e ".[docs,data,secs]"
mkdocs serve    # live-reload preview at http://127.0.0.1:8000
```

Continuous integration runs the tests across Python 3.10–3.13 (and against the
oldest supported dependency versions) on every push and pull request.

## License

The code is released under the MIT license
[described in LICENSE](./LICENSE)

## References

This package has been developed from different publications. Please consider citing the papers
that are relevant to the work you are doing if you are utilizing this code. The culmination of much of the work was contained in our paper
"A 100-year geoelectric hazard analysis for the U.S. high-voltage power grid."

[doi:10.1029/2019SW002329](https://doi.org/10.1029/2019SW002329)

```bibtex
Lucas, G., Love, J. J., Kelbert, A., Bedrosian, P. A., & Rigler, E. J. (2020).
A 100-year geoelectric hazard analysis for the U.S. high-voltage power grid.
Space Weather, 18, e2019SW002329.
https://doi.org/10.1029/2019SW002329
```

### Geoelectric field calculations

[doi:10.1002/2017GL076042](https://doi.org/10.1002/2017GL076042)

```bibtex
Love, J. J., Lucas, G. M., Kelbert, A., & Bedrosian, P. A. (2018).
Geoelectric hazard maps for the Mid‐Atlantic United States:
100 year extreme values and the 1989 magnetic storm.
Geophysical Research Letters, 44, doi:10.1002/2017GL076042.
```

### Transmission line integrations

[doi:10.1002/2017SW001779](https://doi.org/10.1002/2017SW001779)

```bibtex
Lucas, G. M., Love, J. J., & Kelbert, A. (2018). Calculation of voltages
in electric power transmission lines during historic geomagnetic storms:
An investigation using realistic earth impedances. Space Weather, 16,
181–195, doi:10.1002/2017SW001779.
```

### Time domain (DTIR)

[doi:10.1002/2017SW001594](https://doi.org/10.1002/2017SW001594)

```bibtex
Kelbert, A., C. C. Balch, A. Pulkkinen, G. D. Egbert,
J. J. Love, E. J. Rigler, and I. Fujii (2017),
Methodology for time-domain estimation of storm time geoelectric fields
using the 3-D magnetotelluric response tensors,
Space Weather, 15, 874–894, doi:10.1002/2017SW001594.
```

## Earthscope impedance database

[doi:10.17611/DP/EMTF.1](https://doi.org/10.17611/DP/EMTF.1)

```bibtex
Kelbert, A., G.D. Egbert and A. Schultz (2011),
IRIS DMC Data Services Products: EMTF, The Magnetotelluric Transfer Functions,
doi:10.17611/DP/EMTF.1.
```

## Problems/Questions

- [Report an issue using the GitHub issue tracker](http://github.com/greglucas/bezpy/issues)

## Additional Links

- [USGS Geomagnetism Program Home Page](https://geomag.usgs.gov/)
- [Real-time geoelectric field plots using bezpy](https://swx-trec.com/geoelectric)
