"""Data acquisition, storage, and inventory for bezpy.

This subpackage downloads the input datasets bezpy needs and manages them in a
local cache so analyses are reproducible and fast. Today it covers geomagnetic
observatory field data; MT transfer functions and transmission-line/region
geometries are planned (see ROADMAP.md).

Magnetic-field data is stored as CF-style netCDF, one file per observatory per
year, which reads far faster than the text-based IAGA-2002 format and supports
incremental near-real-time updates.

Requires the optional ``data`` dependencies::

    pip install "bezpy[data]"

Quick start::

    import bezpy.data

    bezpy.data.download_observatory("BOU", "2024-05-10", "2024-05-12")
    bezpy.data.update_observatory("BOU")            # top up to now
    ds = bezpy.data.load_observatory("BOU")         # xarray.Dataset
    print(bezpy.data.observatory_inventory())       # what do I have?
"""

# Fail early with an actionable message if the optional stack is missing,
# rather than a confusing ImportError deep in a submodule.
try:
    import pooch as _pooch  # noqa: F401
    import xarray as _xarray  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without extras
    raise ImportError(
        "bezpy.data requires optional dependencies (pooch, xarray, and a "
        "netCDF engine). Install them with:\n\n    pip install 'bezpy[data]'\n"
    ) from exc

from . import mt, tl
from ._cache import clear_cache, get_cache_dir, set_cache_dir
from ._levels import LEVELS, canonical_level
from ._webservice import (
    NRCAN_OBSERVATORIES,
    USGS_OBSERVATORIES,
)
from .indices import fetch_index
from .inventory import observatory_inventory, summarize_observatory
from .magnetic import (
    download_observatory,
    fetch_observatory,
    ingest_iaga,
    load_observatory,
    load_observatory_group,
    observatory_metadata,
    observatory_years,
    stored_partitions,
    update_observatory,
)
from .mt import (
    download_mt,
    download_mt_spud,
    ingest_mt_xml,
    load_mt_site,
    load_mt_sites,
    mt_inventory,
    stored_mt_surveys,
)
from .tl import (
    download_transmission_lines,
    load_transmission_lines,
    read_geojson,
)

__all__ = [
    # cache management
    "get_cache_dir", "set_cache_dir", "clear_cache",
    # magnetic field: fetch / store / update / load
    "fetch_observatory", "download_observatory", "update_observatory",
    "load_observatory", "load_observatory_group", "ingest_iaga",
    "observatory_metadata",
    "observatory_years", "stored_partitions",
    # magnetic inventory
    "observatory_inventory", "summarize_observatory",
    # geomagnetic activity indices
    "fetch_index",
    # magnetotelluric transfer functions (EMTF XML)
    "download_mt", "download_mt_spud", "ingest_mt_xml", "load_mt_site",
    "load_mt_sites", "mt_inventory", "stored_mt_surveys", "mt",
    # transmission line geometries
    "download_transmission_lines", "load_transmission_lines", "read_geojson",
    "tl",
    # data levels + observatory network constants
    "LEVELS", "canonical_level", "USGS_OBSERVATORIES", "NRCAN_OBSERVATORIES",
]
