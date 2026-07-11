# Gridded impedance products and live E-field maps

Everything downstream of a reader consumes the same thing — a collection
of MT sites — so a gridded impedance product (e.g. the 0.25° CONUS
products in ModEM data format) drops into the exact pipeline used for
individual EMTF XML sites. This is the workflow of a live geoelectric
mapping service in ~25 lines.

## 1. Read the gridded product once, save it canonically

```python
import bezpy

sites = bezpy.mt.read_gridded_impedance(
    "USA_impedance_gridded_45per_0.25x0.25-SP2.dat")
print(sites)                       # SiteCollection(~21000 sites: Site3d)

sites.to_netcdf("conus_impedances.nc")   # the canonical form; no pickles
```

The `.dat` parsing takes a while for a continental grid; the netCDF
round trip does not:

```python
sites = bezpy.mt.SiteCollection.from_netcdf("conus_impedances.nc")
```

## 2. Observatory B → E at every grid site

`SiteCollection` goes anywhere a site list goes. The E-field is computed
*at* the impedance sites — no nearest-neighbor matching between a B grid
and a Z grid, and one batched FFT instead of a per-cell loop:

```python
b_obs = bezpy.data.load_observatory_group(
    ["BOU", "FRD", "FRN", "NEW", "SIT", "TUC"], start, end)

e = bezpy.calc_geoelectric_field(b_obs, sites)   # Ex, Ey (time, site)
```

For a live feed with noisy or dropping stations, fit the temporally
coupled filter instead of independent snapshots:

```python
interp = bezpy.mag.SECSInterpolator(method="kalman", tau=3600.,
                                    smoother=False)   # causal: real-time
e = bezpy.calc_geoelectric_field(b_obs, sites, secs=interp)
```

## 3. Pivot onto the grid and map it

The sites of a gridded product lie on a regular lattice, so the
`(time, site)` result unstacks onto `(time, latitude, longitude)`:

```python
import numpy as np

e_grid = e.set_index(site=["latitude", "longitude"]).unstack("site")
eh = np.hypot(e_grid["Ex"], e_grid["Ey"])        # |E| (time, lat, lon)
peak = eh.max("time")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
bezpy.plot.plot_hazard_map(ax, latlons=np.column_stack(
    [e["latitude"].values, e["longitude"].values]),
    values=eh.max("time").values.ravel())
```

Interpolated B maps come straight off the interpolator:

```python
interp.fit(bezpy.mag.prepare(b_obs))
b_maps = interp.predict_grid(np.arange(23, 51), np.arange(-125, -64))
# Bx/By/Bz (time, latitude, longitude); return_var=True adds variances
```

## Notes

- **Units**: the gridded products declare `[mV/km]/[nT]` — bezpy's native
  impedance convention. The reader warns if a file declares anything else.
- **Errors**: when the product carries the ModEM error column, it becomes
  `Z_var` on each site and weights the impedance interpolation.
- **Cadence products**: for a low-rate archive (the live service keeps
  30-minute peaks), reduce after the physics:
  `eh.resample(time="30min").max()`.
