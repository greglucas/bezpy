# Line voltages via kernels

Voltages across transmission lines, computed the fast way. Every
interpolation scheme in `TransmissionLine.calc_voltages` is a *linear* map
from site electric fields to a voltage, so the geometry can be contracted
out ahead of time and applied as a single sparse matrix multiply — or folded
together with the impedances so voltages come **straight from B**.

## Get lines

```python
import bezpy

lines = bezpy.data.download_transmission_lines(
    bbox=(-110, 35, -100, 45), min_voltage=345)          # HIFLD, cached
lines = bezpy.tl.filter_bbox(lines, (-108, 36, -102, 44))  # tighten the region
```

## Level 1: E → V, all lines at once

```python
e = bezpy.calc_geoelectric_field(b_obs, sites,
                                 window=("2024-05-10", "2024-05-12"))
v = bezpy.calc_voltages(lines, e)      # (time, line) volts, one sparse matmul
```

This reproduces the per-line `calc_voltages` loop *exactly* (same sum,
reassociated) at ~20× the speed for hundreds of lines; lines outside the MT
site coverage come back NaN.

## Level 2: B → V, no E-field intermediate

Per-line telluric response functions τᴺ(ω), τᴱ(ω) (Kelbert & Lucas, 2020)
fold the impedances into the length vectors, so one inverse FFT per line
gives voltages directly from the magnetic field:

```python
from bezpy.tl.kernels import TelluricResponse
import numpy as np

tr = TelluricResponse.from_lines(
    lines, sites, how="delaunay",
    periods=np.logspace(np.log10(7), np.log10(30000), 31),  # MT-native grid
    resample_km=1.0,               # densify the quadrature at build time
    extrapolate="asymptotic",      # physical band-edge continuation
)
V = tr.calc_voltages(mag_x, mag_y, dt=60)     # (ntimes, nlines) volts
```

τ evaluated on the FFT grid is cached between equal-length chunks, so a
multi-decade scan pays the construction cost once.

## τ as a data product

Master-grid responses are small and self-contained — they can be shipped and
used with **no line geometry or MT sites at all**:

```python
tr.to_netcdf("tau.nc")
tr2 = TelluricResponse.from_netcdf("tau.nc")
V = tr2.calc_voltages(mag_x, mag_y, dt=60)
```

## Exactness

With `periods=None` the response is evaluated exactly on each run's FFT grid
and reproduces the classic convolve-then-interpolate pipeline to rounding
error (`tests/test_kernels.py`). The master-grid mode differs only by the τ
period sampling — the same trade the published response functions make.
