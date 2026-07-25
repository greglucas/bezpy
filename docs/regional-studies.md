# Regional studies and open data

This page separates three questions that are easy to blur:

1. Are the magnetic and magnetotelluric inputs available?
2. Can bezpy read them?
3. Are the power-network model and validation measurements available for an
   exact GIC reproduction?

The answer can differ at every stage. The table records availability checked
on **25 July 2026**; repository access conditions can change.

## Availability

| Region and study | MT tensors | Magnetic/electric data | Network/GIC data | Reproduction status |
| --- | --- | --- | --- | --- |
| North Island, Aotearoa New Zealand — [Pratscher et al. (2026)](https://doi.org/10.1029/2026SW004997) | The 56 new long-period sites and earlier surveys can be **requested** through the [GNS/ESNZ MT dataset](https://doi.org/10.21420/0KY8-MX13); they are not an anonymous bezpy download | EYR is public through INTERMAGNET; OAKS is available by request from MANA | Transpower model characteristics and measured GIC are commercially restricted | B→E is reproducible after the MT request is granted; the published transformer-level comparison is not independently reproducible without Transpower permission |
| Southern South Island, New Zealand — [Pratscher et al. (2024)](https://doi.org/10.1029/2024SW003907) | Same request-based GNS/ESNZ catalog | EYR is public; SWP is request-based | Transpower network and measurements are restricted | Same partial-reproduction boundary |
| North China, May 2024 — [Wang et al. (2025)](https://doi.org/10.1029/2025SW004557) | **Open:** 22 full EMTF `.zrr` tensors in [Zenodo record 15107366](https://doi.org/10.5281/zenodo.15107366) | **Open:** four magnetic and 22 measured electric-field station series in the same record | This paper validates E rather than publishing a grid model | The tensors work directly in bezpy; the small tensor-only example below is executable now |
| North China, March 2015 — [Wang et al. (2024)](https://doi.org/10.1029/2023SW003758) | 119 SinoProbe tensors were used, but no persistent public tensor archive was identified | Six INTERMAGNET observatories are public | Line-voltage geometries are not packaged with bezpy | Method reproduction is possible if the SinoProbe tensors and routes are obtained |
| Guangxi, China — [Liu et al. (2025)](https://doi.org/10.1029/2025SW004646) | 143 measured MT sites were used; no persistent public tensor archive was identified | Storm magnetic inputs are described in the paper | The 500 kV grid model is not an open bezpy input | The method maps to bezpy, but the published calculation is not currently self-contained |

“No archive identified” means that the paper and its stated sources were
checked; it is not a claim that the authors will not share the data. Contacting
the authors is the appropriate next step.

## Executable North China tensor example

Install the script/data extras and run:

```bash
python -m pip install -e ".[scripts]"
python scripts/china_may2024.py
```

The script calls `download_zenodo_file`, which obtains the live repository
metadata and verifies the archive checksum, then calls
`read_zfile_archive`, which reads all 22 transfer functions without extracting
the ZIP. Repeated runs use the checksum-valid cache.

The Zenodo Z-files are scientifically usable, but their location headers need
repair before mapping: 19 of the 22 files report the same
`(34.255° N, 105.537° E)` coordinate. The associated
`May2024SuperStormdata.zip` contains per-station coordinates and is the
authoritative spatial metadata. It is about 211 MB and is intentionally not
downloaded by the quick example:

```python
import bezpy.data

timeseries = bezpy.data.download_zenodo_file(
    15107366,
    "May2024SuperStormdata.zip",
)
```

With bezpy's current default impedance screen, the live archive produces
9 `ok`, 12 `review`, and 1 `bad` classifications. These flags are triage, not
automatic exclusions: inspect the apparent-resistivity/phase curves and the
paper's own quality decisions before defining a reproducible site subset.

Do not interpolate or map these tensors using the repeated Z-file coordinates.
First join each electric station to the coordinates in its MATLAB record and
archive that corrected station table beside the analysis.

## New Zealand B→E starting point

After GNS/ESNZ grants access to the requested tensors, bezpy already has the
required readers:

```python
import bezpy
import bezpy.data

# The paper used one-second EYR magnetic data for the May 2024 storm.
bezpy.data.download_observatory(
    "EYR",
    "2024-05-10",
    "2024-05-12",
    source="intermagnet",
    level="definitive",
    sampling_period=1,
)
b = bezpy.data.load_observatory(
    "EYR",
    source="intermagnet",
    level="definitive",
    start="2024-05-10",
    end="2024-05-12",
    as_dataframe=True,
)

# Select the reader matching the supplied GNS files.
site = bezpy.mt.read_edi("granted-site.edi")
# or: site = bezpy.mt.read_zfile("granted-site.zrr")
ex, ey = site.convolve_fft(b["X"], b["Y"], dt=1)
```

The 2026 paper assumes the same magnetic series at every MT site, fits the
measured impedance band, calculates local E, and spatially interpolates it.
For a like-for-like method comparison, keep that uniform-B assumption first.
Then repeat with `calc_geoelectric_field` and several observatories/SECS to
measure the effect of spatially variable B without confusing it with a reader
or convention difference.

An exact GIC comparison additionally needs the Transpower DC network
characteristics, transformer grounding parameters, and measured currents.
Those are explicitly restricted by the paper, so bezpy must not ship a
surrogate under the paper's name. With permission, they map to
`bezpy.tl.Network`, `gic_matrix`, and `apply_gic_matrix`.

## Comparison checklist

For any new region, save a compact manifest containing:

- paper DOI and every data DOI or request date;
- original filenames and repository checksums;
- station identifiers, corrected coordinates, tensor units, channel
  orientation, and Fourier sign;
- magnetic source, data level, cadence, components, and gap policy;
- impedance screening results and period coverage;
- whether B is assumed uniform, spatially interpolated, or SECS-derived;
- interpolation, extrapolation, detrend, taper, and padding settings;
- network provenance and access restrictions;
- central results plus uncertainty and at least one independently observed
  E, voltage, or GIC series when available.

This makes a regional example useful even when commercial infrastructure data
prevents a complete reproduction.
