"""netCDF read/write helpers for magnetic-field time series.

bezpy stores magnetic-field data as CF-style netCDF: one file per observatory
per calendar year. netCDF is a compact, self-describing binary format that
reads far faster than the text-based IAGA-2002 files, and the per-year split
keeps individual files a manageable size (~0.5 M one-minute samples/year).

We support whichever netCDF engine xarray can find, preferring ``h5netcdf``
(pure Python + h5py, no compiled netCDF-C library required).
"""
__all__ = ["frame_to_dataset", "dataset_to_frame", "write_dataset", "read_dataset"]

import numpy as np
import xarray as xr

# Standard geomagnetic components and their metadata, in canonical order.
_COMPONENT_ATTRS = {
    "X": {"long_name": "geographic north component", "units": "nT"},
    "Y": {"long_name": "geographic east component", "units": "nT"},
    "Z": {"long_name": "vertical (downward) component", "units": "nT"},
    "F": {"long_name": "total field intensity", "units": "nT"},
    "H": {"long_name": "horizontal intensity", "units": "nT"},
    "D": {"long_name": "declination", "units": "degree"},
}


def _select_engine():
    """Return the first available netCDF engine, preferring h5netcdf."""
    for engine in ("h5netcdf", "netcdf4"):
        try:
            __import__(engine)
            return engine
        except ImportError:
            continue
    raise ImportError(
        "Writing netCDF requires 'h5netcdf' (recommended) or 'netCDF4'. "
        "Install the bezpy data extras with: pip install 'bezpy[data]'")


def frame_to_dataset(df, code, metadata=None):
    """Build a CF-style :class:`xarray.Dataset` from a magnetic-field frame.

    Parameters
    ----------
    df : pandas.DataFrame
        Data indexed by UTC time with one column per field component.
    code : str
        IAGA observatory code, stored as an attribute and used downstream for
        file naming.
    metadata : dict, optional
        Station/provenance metadata to attach as dataset attributes (e.g.
        ``latitude``, ``longitude``, ``elevation``, ``source``, ``data_type``,
        ``sampling_period``). Values of ``None`` are dropped.
    """
    # Always normalize the time axis to "time" so every archive file (web
    # service, IAGA ingest, ...) has an identical dimension name.
    df = df.rename_axis("time")

    data_vars = {}
    for column in df.columns:
        attrs = _COMPONENT_ATTRS.get(str(column).upper(), {})
        # Store components as float32 to halve file size; nT precision is well
        # within float32's ~7 significant digits.
        data_vars[column] = ("time", df[column].to_numpy(dtype="float32"), attrs)

    ds = xr.Dataset(data_vars=data_vars,
                    coords={"time": df.index.values})

    ds.attrs["iaga_code"] = code.upper()
    if metadata:
        ds.attrs.update({k: v for k, v in metadata.items() if v is not None})
    return ds


def dataset_to_frame(ds):
    """Convert a magnetic-field dataset back to a time-indexed DataFrame."""
    df = ds.to_dataframe()
    # to_dataframe() can attach extra coord columns; keep only data variables.
    df = df[[str(v) for v in ds.data_vars]]
    df.attrs.update(ds.attrs)
    return df


def write_dataset(ds, path, compress=True):
    """Write a dataset to netCDF, choosing an available engine and compression.

    Falls back to uncompressed output if the engine rejects the compression
    settings, so a write never fails purely over compression support.
    """
    engine = _select_engine()
    encoding = {}
    if compress:
        if engine == "h5netcdf":
            opts = {"compression": "gzip", "compression_opts": 4}
        else:  # netcdf4
            opts = {"zlib": True, "complevel": 4}
        encoding = {str(v): opts for v in ds.data_vars}

    try:
        ds.to_netcdf(path, engine=engine, encoding=encoding)
    except (ValueError, TypeError):
        # Some engine/version combinations reject the compression keys; retry
        # without any encoding rather than failing the whole write.
        ds.to_netcdf(path, engine=engine)


def read_dataset(path):
    """Read a magnetic-field netCDF file into an in-memory dataset."""
    engine = _select_engine()
    # load() reads eagerly and closes the file handle, which matters when we
    # immediately merge and overwrite the same path during updates.
    with xr.open_dataset(path, engine=engine) as ds:
        return ds.load()


def merge_datasets(existing, new):
    """Merge two magnetic-field datasets on the time axis.

    Overlapping timestamps take their values from ``new`` (the freshly fetched
    data), matching a "latest download wins" policy for revised observatory data.
    """
    time_name = "time" if "time" in existing.dims else list(existing.dims)[0]
    combined = xr.concat([existing, new], dim=time_name)
    # Keep the last occurrence of any duplicated timestamp (from `new`), sorted.
    _, keep = np.unique(combined[time_name].values[::-1], return_index=True)
    keep = len(combined[time_name]) - 1 - keep
    combined = combined.isel({time_name: np.sort(keep)})
    combined.attrs.update(new.attrs)
    return combined
