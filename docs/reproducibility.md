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

## Greg Lucas geoelectric publication map

These papers form the bezpy method lineage. “Method reproduction” means that
the present package exposes the calculation; it does not imply that every
historical input and intermediate artifact is public.

| Publication | Principal calculation | Current bezpy starting point | Reproduction assets |
| --- | --- | --- | --- |
| [Love, Lucas, Kelbert, & Bedrosian (2018), Mid-Atlantic](https://doi.org/10.1002/2017GL076042) | 100-year regional E and March 1989 storm | `calc_geoelectric_field`, `stats.return_level` | [One storm, B to E](tutorials/storm-to-e.md), [hazard scan](tutorials/hazard-scan.md) |
| [Lucas, Love, & Kelbert (2018), transmission-line voltages](https://doi.org/10.1002/2017SW001779) | E integrated along realistic lines | `calc_voltages`, `tl.voltage_matrix` | [Line voltages](tutorials/line-voltages.md) |
| [Love, Lucas, Kelbert, & Bedrosian (2018), Pacific Northwest](https://doi.org/10.1029/2018SW001844) | Regional 3-D impedance hazard maps | `calc_geoelectric_field`, `plot.plot_hazard_map` | [One storm, B to E](tutorials/storm-to-e.md) |
| [Love, Lucas, Bedrosian, & Kelbert (2019), Northeast](https://doi.org/10.1029/2018SW002068) | Extreme amplitude and polarization | `stats.return_level`, ensemble E | [Hazard scan](tutorials/hazard-scan.md) |
| [Lucas et al. (2020), U.S. high-voltage grid](https://doi.org/10.1029/2019SW002329) | 100-year CONUS E and line voltage | `pipeline.scan`, `storms.find_events`, `stats.return_level` | [Original paper repository](https://github.com/greglucas/GeoelectricHazardPaper2019), [hazard scan](tutorials/hazard-scan.md) |
| [Kelbert & Lucas (2020), modified GIC](https://doi.org/10.1029/2020SW002467) | Power-line telluric response functions | `tl.TelluricResponse` | [Original paper repository](https://github.com/greglucas/Modified_GIC_Paper_Kelbert_2020), [line voltages](tutorials/line-voltages.md) |
| [Hartinger et al. (2020), simultaneous ULF B/E observations](https://doi.org/10.1029/2020GL089441) | Frequency-dependent observed B→E response | `Site3d.convolve_fft`, validation scores | [Validation guide](validation.md) |
| [Murphy et al. (2021), MT sampling](https://doi.org/10.1029/2020SW002693) | Sensitivity to survey density | `mt.screen_sites`, subsets of `SiteCollection` | [Gridded/site collections](tutorials/gridded-impedance.md) |
| [Love et al. (2021), E3 EMP surface impedance](https://doi.org/10.1029/2021EA001792) | Realistic surface-impedance mapping | `Site3d.calcZ`, `calc_geoelectric_field` | [Original analysis repository](https://github.com/greglucas/EMP_paper_2020) |
| [Love et al. (2022), March 1989 superstorm](https://doi.org/10.1029/2021SW003030) | Time-dependent CONUS B→E and impact comparison | `pipeline.storm_scan`, `calc_geoelectric_field` | [Storm catalogs](tutorials/storm-catalog.md) |
| [Love et al. (2022), submarine cables](https://doi.org/10.3389/fphy.2022.1022475) | Geomagnetic induction along grounded submarine cables | `calc_geoelectric_field`, line kernels for the incident field | [Line voltages](tutorials/line-voltages.md); cable-specific network physics remains paper-specific |
| [Love et al. (2023), March 1940 superstorm](https://doi.org/10.1029/2022SW003379) | Historical B→E and communication/power impacts | `pipeline.storm_scan`, `calc_geoelectric_field` | [Storm catalogs](tutorials/storm-catalog.md) |
| [Bortnik et al. (2024), LiveWire](https://doi.org/10.1029/2024JH000151) | One-hour-ahead geoelectric-field prediction | bezpy data/field preparation around the external ML model | [Operational mode](tutorials/operational.md) |
| [Shi et al. (2025), multi-scale field perturbations](https://doi.org/10.1029/2024SW004046) | Observed local B/E perturbations | `Site3d.convolve_fft`, validation scores | [Validation guide](validation.md) |
| [Love et al. (2025), Mapping a Carrington Storm](https://doi.org/10.1029/2025GL116835) | CONUS extreme-event E mapping | `calc_geoelectric_field`, `plot.plot_hazard_map` | [USGS result data](https://doi.org/10.5066/F7D799CT), [one storm, B to E](tutorials/storm-to-e.md) |
| [Love et al. (2025), May 1921 impacts](https://doi.org/10.1029/2025SW004563) | Historical long-line impacts and impedance context | line-voltage and storm-catalog context | [Line voltages](tutorials/line-voltages.md), [storm catalogs](tutorials/storm-catalog.md) |

Related works that should be cited when they supply the scientific context,
rather than presented as paper-specific bezpy reproductions, include
[Love et al. (2026), *Magnetic Storms and Geoelectric Hazards*](https://doi.org/10.1146/annurev-earth-032524-012356)
and the chapter
[Love, Bedrosian, Kelbert, & Lucas (2019), *Empirical Estimation of Natural
Geoelectric Hazards*](https://doi.org/10.1002/9781119434412.ch6).

For published studies outside the United States, including executable North
China tensors and the access boundary for the 2026 New Zealand study, see
[Regional studies and open data](regional-studies.md).

The runnable `scripts/japan_halloween.py` example downloads every input for a
real storm automatically. `scripts/china_may2024.py` downloads and verifies an
open paper dataset, then reads and screens its 22 tensors.
`scripts/mt_site_example.py` first uses the local cache and automatically
searches/downloads EarthScope SPUD if the requested region is not cached.

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
