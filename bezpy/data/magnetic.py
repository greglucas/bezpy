"""Download, store, and update geomagnetic observatory data.

This module manages a local archive of magnetic-field time series. Data is
stored as netCDF (one file per observatory per calendar year) under a directory
layout that records its *provenance* -- both the source it came from and its
quality level::

    <cache>/magnetic/<source>/<level>/<CODE>/<CODE>_<YEAR>.nc

For example::

    magnetic/usgs/variation/BOU/BOU_2024.nc          # real-time
    magnetic/usgs/definitive/BOU/BOU_2023.nc         # final, published later
    magnetic/intermagnet/definitive/ESK/ESK_2015.nc  # non-US observatory

Keeping levels in separate files means slow-to-publish ``definitive`` data
never clobbers the ``variation`` (real-time) series you already have, and the
original un-culled real-time data stays available. On load, levels can be
merged into a single "best available" series (higher quality wins, with lower
levels backfilling any gaps -- including points that definitive processing may
have removed).

Sources currently supported:

* ``usgs`` / ``nrcan`` -- the USGS web service (USGS observatories and relayed
  NRCan observatories).
* ``intermagnet`` -- the INTERMAGNET GIN (the global network; pass
  ``source="intermagnet"`` explicitly).

All times are handled in UTC.
"""
__all__ = [
    "observatory_metadata", "fetch_observatory", "download_observatory",
    "update_observatory", "load_observatory", "load_observatory_group",
    "observatory_years", "stored_partitions", "ingest_iaga",
]

import datetime
import functools
import warnings

import numpy as np
import pandas as pd

from . import _intermagnet, _webservice
from ._cache import get_cache_dir
from ._levels import BEST, canonical_level, level_rank
from ._netcdf import dataset_to_frame, frame_to_dataset, merge_datasets, read_dataset, write_dataset
from ._webservice import default_source

DEFAULT_ELEMENTS = ("X", "Y", "Z", "F")

# Provenance tiebreak when two sources provide the same quality level: prefer
# the global/authoritative network, then the national services.
_SOURCE_RANK = {"intermagnet": 3, "usgs": 2, "nrcan": 2, "iaga": 1}


# ----------------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------------
def _obs_dir(code, source, level, create=True):
    """Return the archive directory for one observatory/source/level."""
    level = canonical_level(level)
    return get_cache_dir(f"magnetic/{source}/{level}/{code.upper()}", create=create)


def _year_path(code, source, level, year):
    """Return the netCDF path for one observatory/source/level/year."""
    return _obs_dir(code, source, level) / f"{code.upper()}_{year}.nc"


def _years_in_dir(code_dir, code):
    """Return the sorted years present in an observatory directory."""
    years = []
    for path in code_dir.glob(f"{code.upper()}_*.nc"):
        try:
            years.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(years)


def stored_partitions(code):
    """Return the ``(source, level)`` pairs stored locally for an observatory.

    Ordered from highest to lowest quality (definitive first), which is also the
    precedence used when merging.
    """
    code = code.upper()
    root = get_cache_dir("magnetic", create=False)
    partitions = []
    if root.exists():
        for source_dir in root.iterdir():
            if not source_dir.is_dir():
                continue
            for level_dir in source_dir.iterdir():
                if (level_dir / code).is_dir():
                    partitions.append((source_dir.name, level_dir.name))
    # Sort by (quality, source preference), highest first.
    partitions.sort(key=lambda sl: (_partition_rank(sl[0], sl[1])), reverse=True)
    return partitions


def _partition_rank(source, level):
    """Combined precedence key: quality level first, source preference second."""
    try:
        lvl = level_rank(level)
    except ValueError:
        lvl = -1
    return (lvl, _SOURCE_RANK.get(source, 0))


def observatory_years(code, source=None, level=None):
    """Return the sorted union of years stored for an observatory.

    Optionally restricted to a single ``source`` and/or ``level``.
    """
    code = code.upper()
    years = set()
    for src, lvl in stored_partitions(code):
        if source is not None and src != source:
            continue
        if level is not None and lvl != canonical_level(level):
            continue
        years.update(_years_in_dir(_obs_dir(code, src, lvl, create=False), code))
    return sorted(years)


# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------
def observatory_metadata(code):
    """Return location metadata for an observatory from the bundled IAGA table.

    Falls back to an empty dict (rather than raising) for unknown codes so that
    a download can still proceed using metadata from the response.
    """
    # Imported here to avoid importing the whole mag package at module load.
    from ..mag import get_iaga_observatory

    try:
        info = get_iaga_observatory(code)
    except ValueError:
        return {}
    return {
        "station_name": info.get("name"),
        "latitude": info.get("latitude"),
        "longitude": info.get("longitude"),
        "country": info.get("country"),
    }


# ----------------------------------------------------------------------------
# Fetch (source dispatch)
# ----------------------------------------------------------------------------
def _resolve_source(code, source):
    """Turn ``source="auto"`` into a concrete source for an observatory."""
    return default_source(code) if source in (None, "auto") else source


def _fetch_frame(code, start, end, source, level, elements, sampling_period,
                 base_url, verbose):
    """Dispatch to the correct source client and return a DataFrame."""
    if source in ("usgs", "nrcan"):
        kwargs = {"base_url": base_url} if base_url else {}
        return _webservice.fetch_dataframe(
            code, start, end, elements=elements,
            sampling_period=sampling_period, level=level, verbose=verbose,
            **kwargs)
    if source == "intermagnet":
        kwargs = {"base_url": base_url} if base_url else {}
        return _intermagnet.fetch_dataframe(
            code, start, end, sampling_period=sampling_period, level=level,
            verbose=verbose, **kwargs)
    raise ValueError(
        f"Unknown source {source!r}. Supported: 'usgs', 'nrcan', 'intermagnet' "
        f"(or 'auto').")


def fetch_observatory(code, start, end, source="auto", level="variation",
                      elements=DEFAULT_ELEMENTS, sampling_period=60,
                      base_url=None, verbose=False):
    """Fetch observatory data into memory (no disk write).

    Returns an :class:`xarray.Dataset` indexed by UTC time. Use
    :func:`download_observatory` to fetch *and* persist to the local archive.
    """
    source = _resolve_source(code, source)
    level = canonical_level(level)
    df = _fetch_frame(code, start, end, source, level, elements,
                      sampling_period, base_url, verbose)

    metadata = observatory_metadata(code)
    metadata.update({k: v for k, v in df.attrs.items() if v is not None})
    metadata.update({
        "source": source,
        "level": level,
        "sampling_period": int(sampling_period),
    })
    return frame_to_dataset(df, code, metadata)


# ----------------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------------
def _store_frame(code, df, source, level, metadata):
    """Write a (possibly multi-year) frame into per-year netCDF files.

    Existing files are merged with the new data rather than overwritten, so
    incremental updates accumulate. Returns the list of written paths.
    """
    if df.empty:
        return []

    written = []
    for year, group in df.groupby(df.index.year):
        year_meta = dict(metadata)
        year_meta["year"] = int(year)
        year_meta["last_updated"] = _utcnow().isoformat()
        new_ds = frame_to_dataset(group, code, year_meta)

        path = _year_path(code, source, level, int(year))
        if path.exists():
            existing = read_dataset(path)
            new_ds = merge_datasets(existing, new_ds)
        write_dataset(new_ds, path)
        written.append(path)
    return written


def download_observatory(code, start, end, source="auto", level="variation",
                         elements=DEFAULT_ELEMENTS, sampling_period=60,
                         base_url=None, verbose=False):
    """Download an interval and store it in the local per-year archive.

    Parameters
    ----------
    code : str
        IAGA observatory code.
    start, end : datetime-like
        UTC interval to download (naive timestamps are treated as UTC).
    source : str, optional
        ``"auto"`` (default; USGS or NRCan by observatory), ``"usgs"``,
        ``"nrcan"``, or ``"intermagnet"``.
    level : str, optional
        Data quality level to request and file under. Default ``"variation"``
        (real-time). See :mod:`._levels`.
    elements, sampling_period, base_url, verbose
        Passed through to the source client.

    Returns
    -------
    list of pathlib.Path
        The netCDF files written or updated.
    """
    source = _resolve_source(code, source)
    level = canonical_level(level)
    ds = fetch_observatory(code, start, end, source=source, level=level,
                           elements=elements, sampling_period=sampling_period,
                           base_url=base_url, verbose=verbose)
    df = dataset_to_frame(ds)
    if df.empty:
        warnings.warn(
            f"No {level} data returned for {code} ({source}) "
            f"between {start} and {end}.", stacklevel=2)
        return []
    return _store_frame(code, df, source, level, ds.attrs)


def update_observatory(code, source="auto", level="variation", until=None,
                       sampling_period=60, base_url=None, default_days=3,
                       verbose=False):
    """Bring one source/level of an observatory's archive up to date.

    Fetches from just after the most recent stored sample (for this source and
    level) up to ``until`` (default: now, UTC). If nothing is stored yet, the
    most recent ``default_days`` days are downloaded to seed the archive.

    Returns the list of netCDF files written or updated (empty if current).
    """
    source = _resolve_source(code, source)
    level = canonical_level(level)

    until = _utcnow() if until is None else pd.Timestamp(until)
    if until.tzinfo is not None:
        until = until.tz_convert("UTC").tz_localize(None)

    last = _last_stored_time(code, source, level)
    if last is None:
        start = until - datetime.timedelta(days=default_days)
        if verbose:
            print(f"No local {level} data for {code} ({source}); "
                  f"seeding last {default_days} days.")
    else:
        # Resume one sample after the last stored point to avoid a duplicate.
        start = last + datetime.timedelta(seconds=int(sampling_period))

    if start >= until:
        if verbose:
            print(f"{code} ({source}/{level}) already up to date "
                  f"(last sample {last}).")
        return []

    return download_observatory(code, start, until, source=source, level=level,
                                sampling_period=sampling_period,
                                base_url=base_url, verbose=verbose)


def _last_stored_time(code, source, level):
    """Return the latest stored timestamp for one source/level, or None."""
    years = _years_in_dir(_obs_dir(code, source, level, create=False), code)
    if not years:
        return None
    ds = read_dataset(_year_path(code, source, level, years[-1]))
    return pd.Timestamp(ds["time"].values.max())


# ----------------------------------------------------------------------------
# Load (with cross-level merge)
# ----------------------------------------------------------------------------
def load_observatory(code, level=BEST, source=None, prefer=None, start=None,
                     end=None, as_dataframe=False):
    """Load stored observatory data, merging levels by default.

    Parameters
    ----------
    code : str
        IAGA observatory code.
    level : str, optional
        Which quality level to load. Default ``"best"``: all stored levels are
        merged into one series, with higher-quality data taking precedence and
        lower levels backfilling any gaps. Pass a specific level (e.g.
        ``"definitive"``) to load only that level.
    source : str, optional
        Restrict to a single source (e.g. ``"intermagnet"``). By default all
        stored sources are considered.
    prefer : str, tuple, or list, optional
        Override the default quality precedence when merging. By default higher
        quality wins (definitive over variation); pass ``prefer="variation"`` to
        let real-time data overwrite definitive instead, ``prefer="intermagnet"``
        to prefer a source, or a ``(source, level)`` tuple / list of these for
        finer control. Preferred partitions take precedence in the order given;
        any others follow in normal quality order.
    start, end : datetime-like, optional
        Restrict the returned data to this UTC interval.
    as_dataframe : bool, optional
        Return a :class:`pandas.DataFrame` instead of an
        :class:`xarray.Dataset`.

    Raises
    ------
    FileNotFoundError
        If no matching data is stored.
    """
    want_level = None if level == BEST else canonical_level(level)

    partitions = [
        (src, lvl) for src, lvl in stored_partitions(code)
        if (source is None or src == source)
        and (want_level is None or lvl == want_level)
    ]
    if not partitions:
        raise FileNotFoundError(
            f"No stored data for observatory {code.upper()} matching "
            f"source={source!r}, level={level!r}. "
            f"Download it first with bezpy.data.download_observatory().")

    partitions = _apply_prefer(partitions, prefer)

    # partitions is highest-precedence first, which is the order combine_first()
    # wants (fill from the next partition only where the higher one is missing).
    frames = []
    provenance = []
    merged_meta = {}
    for src, lvl in partitions:
        frame = _load_partition_frame(code, src, lvl, start, end)
        if frame is None or frame.empty:
            continue
        frames.append(frame)
        provenance.append(f"{src}/{lvl}")
        # Keep metadata from the highest-precedence partition seen first.
        for key, value in frame.attrs.items():
            merged_meta.setdefault(key, value)

    if not frames:
        raise FileNotFoundError(
            f"No stored samples for observatory {code.upper()} in the requested "
            f"window.")

    merged = functools.reduce(lambda hi, lo: hi.combine_first(lo), frames)
    merged = merged.sort_index()
    if start is not None or end is not None:
        merged = merged.loc[_slice_mask(merged.index, start, end)]

    merged_meta["iaga_code"] = code.upper()
    merged_meta["provenance"] = ",".join(provenance)
    merged.attrs.update(merged_meta)

    if as_dataframe:
        return merged
    # Rebuild a dataset so the return type matches a single-level load.
    return frame_to_dataset(merged, code, merged_meta)


def load_observatory_group(codes, start=None, end=None, **kwargs):
    """Load several observatories onto one ``(time, obs)`` Dataset.

    A thin composition of :func:`load_observatory` and
    :func:`bezpy.mag.combine_observatories`, ready for the SECS interpolator
    and the pipeline.

    Parameters
    ----------
    codes : sequence of str
        IAGA observatory codes.
    start, end : datetime-like, optional
        Restrict to this UTC interval.
    **kwargs
        Forwarded to :func:`load_observatory` (``level``, ``source``,
        ``prefer``, ...).
    """
    from ..mag import combine_observatories
    return combine_observatories(
        {code: load_observatory(code, start=start, end=end, **kwargs)
         for code in codes})


def _apply_prefer(partitions, prefer):
    """Reorder partitions so preferred ones win the merge.

    ``prefer`` may be a level name, a source name, a ``(source, level)`` tuple,
    or a list of any of these. Matching partitions are moved to the front (in
    the order given); the rest keep their quality order.
    """
    if prefer is None:
        return partitions
    prefs = [prefer] if isinstance(prefer, (str, tuple)) else list(prefer)

    def _matches(part, pref):
        src, lvl = part
        if isinstance(pref, tuple):
            return src == pref[0] and lvl == canonical_level(pref[1])
        try:
            # A recognized level name matches by level...
            return lvl == canonical_level(pref)
        except ValueError:
            # ...otherwise treat it as a source name.
            return src == pref

    preferred = []
    for pref in prefs:
        for part in partitions:
            if part not in preferred and _matches(part, pref):
                preferred.append(part)
    return preferred + [p for p in partitions if p not in preferred]


def _load_partition_frame(code, source, level, start, end):
    """Load and concatenate the year files for one source/level partition."""
    import xarray as xr

    code_dir = _obs_dir(code, source, level, create=False)
    years = _years_in_dir(code_dir, code)
    if start is not None:
        years = [y for y in years if y >= pd.Timestamp(start).year]
    if end is not None:
        years = [y for y in years if y <= pd.Timestamp(end).year]
    if not years:
        return None

    datasets = [read_dataset(_year_path(code, source, level, y)) for y in years]
    ds = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
    return dataset_to_frame(ds.sortby("time"))


def _slice_mask(index, start, end):
    """Boolean mask selecting index values within [start, end]."""
    mask = np.ones(len(index), dtype=bool)
    if start is not None:
        mask &= index >= pd.Timestamp(start)
    if end is not None:
        mask &= index <= pd.Timestamp(end)
    return mask


# ----------------------------------------------------------------------------
# IAGA ingest -- convert existing text files into the fast netCDF archive
# ----------------------------------------------------------------------------
def ingest_iaga(paths, code=None, source="iaga", level="definitive",
                sampling_period=60):
    """Convert existing IAGA-2002 text files into the netCDF archive.

    Lets users who already have IAGA files benefit from the fast, merge-aware
    netCDF store without re-downloading. Components are read via
    :func:`bezpy.mag.read_iaga` (converted to XYZF) and filed under
    ``source``/``level`` like any other data.

    Returns the list of netCDF files written or updated.
    """
    from ..mag import read_iaga

    level = canonical_level(level)
    if isinstance(paths, str):
        paths = [paths]

    written = []
    for path in paths:
        df, header = read_iaga(path, return_header=True)
        obs_code = (code or header.get("iaga code", "")).upper()
        if not obs_code:
            raise ValueError(f"Could not determine observatory code for {path}; "
                             "pass code=...")

        metadata = observatory_metadata(obs_code)
        metadata.update({
            "source": source,
            "level": level,
            "sampling_period": int(sampling_period),
            "station_name": header.get("station name") or metadata.get("station_name"),
        })
        written.extend(_store_frame(obs_code, df, source, level, metadata))
    return written


def _utcnow():
    """Return the current UTC time as a naive pandas Timestamp.

    The archive stores naive UTC timestamps (matching the web service and IAGA
    files), so we strip the timezone for internal comparisons.
    """
    return pd.Timestamp(datetime.datetime.now(datetime.UTC)).tz_localize(None)
