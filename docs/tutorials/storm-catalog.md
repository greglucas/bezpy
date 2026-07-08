# Storm catalogs and case studies

Build a storm catalog from activity indices and compute per-storm maxima —
the classic case-study workflow.

## 1. Fetch activity indices

```python
import pandas as pd
import bezpy

dst = bezpy.data.fetch_index("dst", "1985", "2015")   # hourly, OMNI2
kp = bezpy.data.fetch_index("kp", "1985", "2015")     # 3-hourly, GFZ Potsdam
index = pd.concat([dst, kp], axis=1)
```

## 2. Catalog the storms

Samples where Dst drops below the threshold **or** Kp reaches it trigger a
storm; each trigger contributes a ±window interval and overlaps merge:

```python
catalog = bezpy.storms.find_storms(index, dst_threshold=-140, kp_threshold=8,
                                   window="1.5D")
print(catalog)
#        start                 end                   min_dst  max_kp
# 0      1986-02-05 12:00:00   1986-02-10 03:00:00   -307.0   9.0
# ...
```

## 3. Scan the catalog

`storm_scan` runs the B→E (→V) chain over each window and keeps the maxima:

```python
b_obs = bezpy.data.load_observatory_group(["BOU", "FRD", "TUC"],
                                          "1985-01-01", "2015-12-31")
sites = bezpy.data.load_mt_sites(bbox=(-110, 35, -100, 45), min_rating=3)

maxima = bezpy.pipeline.storm_scan(catalog, b_obs, sites)
print(maxima["E_max"])       # (storm, site) peak |E| in mV/km
```

## Events without an index

Index-based catalogs miss local events: a regional dB/dt burst does not have
to move Dst. `find_events` catalogs threshold exceedances of **any** series
— |E| at a site, a line voltage, |dB/dt| — with runs declustering:

```python
events = bezpy.storms.find_events(e_magnitude_series, threshold=100.,
                                  gap="12h", min_duration=None)
```

This is the primitive that connects the
[continuous scan](hazard-scan.md) to hazard statistics: the scan produces
exceedances everywhere, `find_events` declusters them into independent
events, and `bezpy.stats.return_level` fits the tail.
