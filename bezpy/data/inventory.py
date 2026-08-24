"""Summaries of what magnetic-field data is stored in the local archive.

These helpers let a user see, at a glance, which observatories, sources, quality
levels, and years they have downloaded, how complete each file is, and how much
disk it uses -- without needing to know the on-disk layout.
"""

__all__ = ["observatory_inventory", "summarize_observatory"]

import numpy as np
import pandas as pd
import xarray as xr

from ._cache import get_cache_dir
from ._netcdf import _select_engine
from .magnetic import _obs_dir, _year_path, _years_in_dir, stored_partitions

# Components we look at when measuring data completeness, in preference order.
_PRIMARY_COMPONENTS = ("X", "Y", "Z", "H", "F")


def _summarize_file(path, source, level, detailed):
    """Return a summary dict for a single archive netCDF file."""
    engine = _select_engine()
    with xr.open_dataset(path, engine=engine) as ds:
        times = ds["time"].values
        n_samples = int(times.size)
        start = pd.Timestamp(times.min()) if n_samples else pd.NaT
        end = pd.Timestamp(times.max()) if n_samples else pd.NaT

        period = ds.attrs.get("sampling_period")
        # Expected number of evenly-spaced samples across the covered span.
        if period and n_samples > 1:
            span = (end - start).total_seconds()
            expected = int(round(span / float(period))) + 1
        else:
            expected = n_samples
        coverage = (n_samples / expected) if expected else np.nan

        row = {
            "code": ds.attrs.get("iaga_code", path.stem.split("_")[0]),
            "source": source,
            "level": level,
            "year": int(ds.attrs.get("year", path.stem.split("_")[-1])),
            "start": start,
            "end": end,
            "n_samples": n_samples,
            "coverage": coverage,
            "components": ",".join(str(v) for v in ds.data_vars),
            "size_mb": path.stat().st_size / 1e6,
            "path": str(path),
        }

        if detailed:
            # Fraction of finite values in the first available primary component.
            row["completeness"] = np.nan
            for comp in _PRIMARY_COMPONENTS:
                if comp in ds.data_vars and n_samples:
                    finite = int(np.isfinite(ds[comp].values).sum())
                    row["completeness"] = finite / n_samples
                    break
    return row


def observatory_inventory(code=None, detailed=False):
    """Summarize the locally stored magnetic-field archive.

    Parameters
    ----------
    code : str, optional
        Restrict the summary to a single observatory. If omitted, every stored
        observatory is included.
    detailed : bool, optional
        Also compute the fraction of non-missing samples per file
        (``completeness``). This reads the field values and so is slower.

    Returns
    -------
    pandas.DataFrame
        One row per stored observatory / source / level / year, sorted by code,
        source, level, then year. Empty (with the expected columns) if nothing
        is stored.
    """
    magnetic_root = get_cache_dir("magnetic", create=False)

    if code is not None:
        codes = [code.upper()]
    elif magnetic_root.exists():
        # Any code directory found under any source/level partition.
        codes = sorted(
            {
                code_dir.name
                for source_dir in magnetic_root.iterdir()
                if source_dir.is_dir()
                for level_dir in source_dir.iterdir()
                if level_dir.is_dir()
                for code_dir in level_dir.iterdir()
                if code_dir.is_dir()
            }
        )
    else:
        codes = []

    rows = []
    for obs_code in codes:
        for source, level in stored_partitions(obs_code):
            code_dir = _obs_dir(obs_code, source, level, create=False)
            for year in _years_in_dir(code_dir, obs_code):
                path = _year_path(obs_code, source, level, year)
                rows.append(_summarize_file(path, source, level, detailed))

    columns = [
        "code",
        "source",
        "level",
        "year",
        "start",
        "end",
        "n_samples",
        "coverage",
        "components",
        "size_mb",
        "path",
    ]
    if detailed:
        columns.insert(9, "completeness")

    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values(["code", "source", "level", "year"]).reset_index(drop=True)
    return df


def summarize_observatory(code):
    """Return a per-source/level/year summary for one observatory (detailed).

    Convenience wrapper around :func:`observatory_inventory` that always
    computes completeness and is scoped to one observatory.
    """
    return observatory_inventory(code=code, detailed=True)
