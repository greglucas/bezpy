# Units and conventions

These conventions hold everywhere in bezpy. When a legacy interface deviates
(noted below), the docstring says so loudly.

## Units

| Quantity | Symbol | Units |
| --- | --- | --- |
| Magnetic field | B | nT |
| Electric field | E | mV/km |
| Line voltage | V | volts |
| Impedance | Z | (mV/km)/nT |
| Telluric response | τ | V/nT |
| Periods | T | seconds |

## Coordinates

- **x = geographic North, y = geographic East** for all field components
  (`Bx`, `By`, `Ex`, `Ey`) and impedance organization.
- Impedance tensors are organized **(xx, xy, yx, yy)** along the first axis:
  `Ex = Zxx·Bx + Zxy·By`, `Ey = Zyx·Bx + Zyy·By`.
- New xarray APIs use explicit `latitude` / `longitude` coordinates so
  ordering never reaches users.

!!! warning "Two legacy orderings exist"
    - `TransmissionLine.pts` stores vertices in **(longitude, latitude)**
      order (GeoJSON convention).
    - Legacy `site_xys` arguments (e.g. `set_delaunay_weights`) are
      **(latitude, longitude)**.

    Bare-array inputs to new APIs (`SECSInterpolator.predict`, plotting
    helpers) are **(latitude, longitude)** and documented per-function.

## xarray data model

- Dimensions: `time`, plus `obs` (observatories), `site` (MT sites), or
  `line` (transmission lines).
- Data variables: per-component (`Bx`/`By`/`Bz`, `Ex`/`Ey`) with `units` in
  `attrs`.
- Spatial coordinates: `latitude` / `longitude` on the spatial dimension.
- Processing metadata (pad extent, sample interval) travels in `attrs` —
  `bezpy.mag.prepare` writes it, `bezpy.mag.trim_pad` consumes it.

## Canonical site-collection dataset

MT site collections have one canonical xarray form, produced by
`SiteCollection.to_dataset()` / consumed by `SiteCollection.from_dataset()`
(with `to_netcdf` / `from_netcdf` for files). Every reader — EMTF XML,
Z-files, EDI, gridded impedance products — yields sites that round-trip
through it, so a collection saved from one study loads instantly in the
next with no pickles and no reparsing:

| | dims | variables | coords |
| --- | --- | --- | --- |
| 3D (`Site3d`) | `site`, `component`, `period` | `Z_real`, `Z_imag` [, `Z_var`] | `site` names, `latitude`/`longitude` (site), `period` (s, ascending), `component` = `xx, xy, yx, yy` |
| 1D (`Site1d`) | `site`, `layer` | `thickness` (m), `resistivity` (ohm-m) | `site` names, `latitude`/`longitude` (site) |

Sites with different period grids share the sorted union axis with NaN
where a site has no estimate (all downstream interpolation drops NaNs
per-component).

## Method options

Everywhere past practice and the current default diverge, the legacy
behavior is available **by name** and at least one test compares the two:

| Choice | Default | Legacy option |
| --- | --- | --- |
| Gap handling | `gaps="mask"` (infinite variance) | `gaps="interpolate"` (+ `max_gap`), `gaps="drop"` |
| Detrending | `detrend="linear"` + `taper="tukey"` | `detrend="median"`, `taper=None` |
| Convolution pad | `pad="auto"` (3 × longest impedance period) | `pad="2D"` |
| Impedance band edges | `extrapolate="zeros"` | (`"asymptotic"` is the physical option) |
| Tail statistics | `method="gpd"` (POT + profile CI) | `method="lognormal-2019"` |

Defaults change only with evidence from the
[validation harness](validation.md).
