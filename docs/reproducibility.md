# Reproducing publications

bezpy supports two related goals which should not be conflated:

1. **Reproduce a published calculation** with the methods and software state
   used by that paper.
2. **Repeat the analysis with current best practices**, including corrected
   interpolation, explicit provenance, uncertainty, and validated defaults.

The current API is the recommended path for new work. Exact historical
reproduction can require an older release because a known Delaunay projection
bug was fixed in 0.5.0; there is deliberately no option that recreates that
bug. See the result-changing section of the
[changelog](https://github.com/greglucas/bezpy/blob/main/CHANGELOG.md).

## Paper-to-code map

| Publication / method | Current bezpy entry point | Worked starting point |
| --- | --- | --- |
| Lucas et al. (2020), 100-year U.S. hazard | `pipeline.scan`, `storms.find_events`, `stats.return_level` | [Continuous hazard scan](tutorials/hazard-scan.md) |
| Love et al. (2018), regional hazard maps | `calc_geoelectric_field`, `plot.plot_hazard_map` | [One storm, B to E](tutorials/storm-to-e.md) |
| Lucas, Love, & Kelbert (2018), line voltages | `calc_voltages`, `tl.voltage_matrix` | [Line voltages](tutorials/line-voltages.md) |
| Kelbert et al. (2017), DTIR | `mt.DTIR`, `pipeline.Nowcast` | [Operational mode](tutorials/operational.md) |
| Kelbert & Lucas (2020), telluric response functions | `tl.TelluricResponse` | [Line voltages](tutorials/line-voltages.md) |
| Horton et al. (2012), EPRI 21-bus benchmark | `tl.Network.epri21_benchmark` | [Network GIC](tutorials/network-gic.md) |

The runnable `scripts/japan_halloween.py` example downloads every input for a
real storm automatically. `scripts/mt_site_example.py` first uses the local
cache and automatically searches/downloads EarthScope SPUD if the requested
region is not cached.

!!! warning "Current limitation"
    The repository does not yet contain a single command that regenerates
    every table and figure from the 2018/2020 hazard papers. The APIs and data
    acquisition pieces are present, but paper-specific line geometries,
    screened site lists, intermediate catalogs, and plotting recipes need to
    be archived together as versioned research compendia. Until those
    artifacts exist, describe a result as a **method reproduction**, not an
    exact computational reproduction.

## Historical compatibility settings

For comparisons with pre-0.5 calculations, state these choices explicitly:

```python
e = bezpy.calc_geoelectric_field(
    b_obs,
    sites,
    extrapolate="zeros",  # historical hard stop outside the measured MT band
    detrend="median",
    taper=None,
    pad="2D",
)

rl = bezpy.stats.return_level(
    storm_maxima,
    years_observed=years,
    method="lognormal-2019",
    ci="bootstrap",
)
```

These settings reproduce named legacy methods, not the corrected pre-0.5
Delaunay implementation. Use the exact release/tag cited by a paper when
bitwise historical behavior matters.

## Minimum provenance to publish

Archive the following beside every result:

- the bezpy version and immutable source revision;
- Python and dependency versions;
- magnetic source, quality level, observatory codes, time span, cadence, and
  the `provenance` attribute returned by `load_observatory`;
- MT product IDs/survey, site list, quality ratings, screening decisions, and
  original EMTF XML (or a persistent reference to it);
- transmission-line geometry source/date and any resampling;
- every non-default method parameter, including SECS grid/regularization, gap
  policy, detrend/taper/pad, impedance extrapolation, interpolation method,
  event threshold/declustering gap, and tail model;
- random seeds and realization/bootstrap counts;
- validation results relevant to the chosen network and region.

Resumable scans protect against changed methods, sites, geometries, and
topology with a configuration fingerprint. They intentionally allow the
magnetic record to grow, so the fingerprint does not hash every B value. If a
provider revises already-scanned magnetic samples, start a new scan rather
than resuming the old checkpoint.

A compact environment record can be made with:

```bash
python -c "import bezpy; print(bezpy.__version__)"
python -m pip freeze > environment.txt
git rev-parse HEAD
```

For archival releases, cite both the software DOI and the method paper(s)
listed in the README. A release-specific DOI is preferable to a badge that
always resolves to the newest version.
