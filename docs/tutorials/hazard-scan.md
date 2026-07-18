# Continuous scan to hazard map

The 2019 geoelectric hazard analysis, done the modern way: scan the **whole
record** instead of pre-selected storms, fit the **tail** instead of the
body, and put a confidence interval on every number. About 40 lines.

## Why not storm windows?

Selecting ~5-day windows around index-picked storms was partly a performance
workaround, and it biases the sample: regional dB/dt events don't all move
Dst, so index-quiet-but-locally-active intervals are silently excluded. The
continuous scan processes every sample in chunks with streaming reducers —
full E series are never held in memory — and storms *fall out* of the result
as exceedance clusters instead of going in as assumptions.

## The scan

```python
import bezpy

sites = bezpy.data.load_mt_sites(bbox=(-110, 35, -100, 45), min_rating=3)
b_obs = bezpy.data.load_observatory_group(["BOU", "FRD", "TUC", "NEW"],
                                          "1985-01-01", "2015-12-31")

result = bezpy.pipeline.scan(
    b_obs, sites,
    chunk="1Y",                                # memory-bounding chunk length
    reduce=("max", "exceedances", "quantiles"),
    threshold={"E": 100.},                     # mV/km exceedance level
    out_path="scan.nc",                        # resumable: re-run to continue
)
```

Chunk edges are protected by the convolution pad, so the result does not
depend on the chunk size. If the run is interrupted, calling `scan` again
with the same `out_path` resumes after the last completed chunk.

Chunks are independent, so a multi-decade scan over many sites can use
`workers=` to process them across processes instead of one at a time
(`scan(..., workers=4)`); results and `out_path` checkpoints are unchanged
by the worker count.

## Return levels with uncertainty

Decluster the exceedances into independent events per site, then fit a
Generalized Pareto tail (peaks-over-threshold):

```python
import numpy as np
import pandas as pd

exceed = pd.Series(
    result["E_exceedance_value"].values,
    index=pd.DatetimeIndex(result["E_exceedance_time"].values))
items = result["E_exceedance_item"].values

levels, low, high = [], [], []
for i, site in enumerate(sites):
    events = bezpy.storms.find_events(exceed[items == i].sort_index(),
                                      threshold=100., gap="12h")
    rl = bezpy.stats.return_level(events["peak_value"], years_observed=31,
                                  return_period=100, method="gpd",
                                  ci="profile")
    levels.append(rl.value); low.append(rl.ci_low); high.append(rl.ci_high)
```

## The map

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(projection=ccrs.LambertConformal())
ax.set_extent((-110, -100, 35, 45))
bezpy.plot.add_map_features(ax)
bezpy.plot.plot_hazard_map(
    ax, list(zip(result["latitude"].values, result["longitude"].values)),
    levels, colorbar_label="100-year |E| (mV/km)")
plt.show()
```

??? info "Reproducing the published 2019 method"
    The paper fit a **lognormal** to per-storm maxima. That method remains
    available by name, so the two can be compared directly:

    ```python
    catalog = bezpy.storms.find_storms(index, dst_threshold=-140,
                                       kp_threshold=8)
    maxima = bezpy.pipeline.storm_scan(catalog, b_obs, sites)
    rl = bezpy.stats.return_level(maxima["E_max"].values[:, i],
                                  years_observed=31,
                                  method="lognormal-2019", ci="bootstrap")
    ```

    On heavy-tailed data the two methods genuinely disagree — the body fit
    is pulled away from the tail. That disagreement is the reason the
    default changed (`tests/test_stats.py::test_methods_diverge_on_heavy_tail`).
