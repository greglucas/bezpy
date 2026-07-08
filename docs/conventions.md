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
