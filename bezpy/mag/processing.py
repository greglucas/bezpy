"""Preparation of observatory magnetic time series for spectral processing.

The FFT-convolution pipeline (impedances applied in the frequency domain)
needs its input combined onto one time axis, uniformly sampled, detrended,
tapered, and padded generously enough that convolution edge effects never
touch the interval of interest. These are the shared, boring-but-critical
steps every analysis re-implements; they live here with defensible defaults:

* **Gaps are masked, not fabricated** -- missing samples stay NaN so the SECS
  fit can exclude them per-sample (``bezpy.mag.SECSInterpolator``). The
  legacy fill is available as ``gaps="interpolate"`` for single-site paths.
* **Linear detrend + Tukey taper** instead of a bare median subtraction: the
  median controls neither the secular trend nor spectral leakage; the taper
  is applied over the pad region only, so the analysis window is untouched.
* **Pad length derived from the impedance's longest period** (default
  3 x 30,000 s for EMTF transfer functions) rather than a hardcoded value.
"""

__all__ = ["combine_observatories", "prepare", "trim_pad"]

import numpy as np
import pandas as pd

from .utils import get_iaga_observatory

# Longest period in EMTF empirical transfer functions (seconds)
DEFAULT_MAX_PERIOD = 30000.
# Component-name normalization applied when combining
_RENAME = {"X": "Bx", "Y": "By", "Z": "Bz"}


def _require_xarray():
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "bezpy.mag preprocessing requires xarray; install it with "
            "`pip install xarray` (or the bezpy[data] extra).") from None
    return xr


def _obs_latlon(code, obj):
    """(latitude, longitude) from attrs, falling back to the IAGA table."""
    attrs = getattr(obj, "attrs", {})
    if attrs.get("latitude") is not None and attrs.get("longitude") is not None:
        return float(attrs["latitude"]), float(attrs["longitude"])
    info = get_iaga_observatory(code)
    return info["latitude"], info["longitude"]


def combine_observatories(observatories):
    """Combine per-observatory series into one ``(time, obs)`` Dataset.

    :param observatories: mapping of IAGA code to an :class:`xarray.Dataset`
        (as returned by ``bezpy.data.load_observatory``) or a
        :class:`pandas.DataFrame`, with ``X``/``Y``/``Z`` (or ``Bx``/``By``/
        ``Bz``) components indexed by time.
    :return: Dataset with ``Bx``/``By``/``Bz`` variables dimensioned
        ``(time, obs)`` on the union of all time axes (missing samples NaN),
        with ``latitude``/``longitude`` coordinates on the obs dimension.
        Observatory coordinates come from the input attrs when present, the
        bundled IAGA table otherwise.
    """
    xr = _require_xarray()
    per_obs = []
    latlons = []
    for code, obj in observatories.items():
        ds = obj if hasattr(obj, "data_vars") else xr.Dataset.from_dataframe(
            obj.rename_axis("time"))
        ds = ds.rename({old: new for old, new in _RENAME.items() if old in ds})
        keep = [name for name in ("Bx", "By", "Bz") if name in ds]
        if not keep:
            raise ValueError(f"{code}: no X/Y/Z or Bx/By/Bz variables found")
        latlons.append(_obs_latlon(code, obj))
        per_obs.append(ds[keep].expand_dims(obs=[str(code).upper()]))

    combined = xr.concat(per_obs, dim="obs", join="outer",
                         combine_attrs="drop_conflicts")
    combined = combined.transpose("time", "obs").sortby("time")
    latlons = np.asarray(latlons, dtype=float)
    combined = combined.assign_coords(latitude=("obs", latlons[:, 0]),
                                      longitude=("obs", latlons[:, 1]))
    for name in combined.data_vars:
        combined[name].attrs.setdefault("units", "nT")
    return combined


def _uniform_dt(times):
    """Sample interval in seconds; raises with specifics if non-uniform."""
    if len(times) < 2:
        raise ValueError("need at least two samples to process")
    steps = np.diff(times.astype("datetime64[ns]").astype("int64")) / 1e9
    if steps.min() <= 0:
        raise ValueError("time axis must be strictly increasing "
                         "(duplicate or unsorted timestamps found)")
    if steps.max() != steps.min():
        bad = np.nonzero(steps != steps[0])[0][0]
        raise ValueError(
            f"time axis is not uniformly sampled: step changes from "
            f"{steps[0]:g} s to {steps[bad]:g} s at {times[bad + 1]} -- "
            "resample first (e.g. ds.resample(time=...) leaves gaps as NaN)")
    return float(steps[0])


def _as_seconds(value):
    """Timedelta-like or numeric seconds -> float seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    return pd.Timedelta(value).total_seconds()


def _detrend_linear(values):
    """Remove the least-squares line from each column, ignoring NaNs."""
    x = np.arange(values.shape[0], dtype=float)
    out = values.copy()
    flat = out.reshape(values.shape[0], -1)
    for j in range(flat.shape[1]):
        good = np.isfinite(flat[:, j])
        if good.sum() < 2:
            continue
        slope, intercept = np.polyfit(x[good], flat[good, j], 1)
        flat[:, j] -= slope*x + intercept
    return out


def _long_runs(missing, n_max):
    """Mask of samples inside NaN runs longer than ``n_max`` samples.

    ``missing`` is boolean with time as the first axis.
    """
    out = np.zeros_like(missing)
    flat_in = missing.reshape(missing.shape[0], -1)
    flat_out = out.reshape(out.shape[0], -1)
    for j in range(flat_in.shape[1]):
        edges = np.flatnonzero(np.diff(np.r_[0, flat_in[:, j].astype(int), 0]))
        for start, end in zip(edges[::2], edges[1::2]):
            if end - start > n_max:
                flat_out[start:end, j] = True
    return out


def _pad_taper(ntimes, pad_left, pad_right):
    """Window that is 1 over the analysis span, cosine-ramped over the pads."""
    window = np.ones(ntimes)
    if pad_left:
        ramp = 0.5*(1 - np.cos(np.pi*np.arange(pad_left)/pad_left))
        window[:pad_left] = ramp
    if pad_right:
        ramp = 0.5*(1 - np.cos(np.pi*np.arange(pad_right)/pad_right))
        window[-pad_right:] = ramp[::-1]
    return window


def prepare(b, window=None, pad="auto", max_period=DEFAULT_MAX_PERIOD,
            detrend="linear", taper="tukey", gaps="mask", max_gap=None):
    """Prepare a combined B Dataset for FFT convolution.

    Selects the analysis window plus padding, validates uniform sampling,
    detrends, tapers the pad region, and applies the gap policy. The pad
    extent is recorded in ``attrs`` so :func:`trim_pad` can remove it after
    the convolution.

    :param b: ``(time, obs)`` Dataset from :func:`combine_observatories`
        (any Dataset whose variables share a ``time`` dimension works).
    :param window: optional ``(start, end)`` analysis interval. With a
        window, the returned data covers ``start - pad`` to ``end + pad``
        (as available). Without one, the whole record is used and its outer
        ``pad`` on each side is treated as the pad region.
    :param pad: pad duration per side: ``"auto"`` (default) derives it from
        the impedance band as ``3 * max_period``, otherwise anything
        ``pandas.Timedelta`` accepts (e.g. ``"2D"``, the legacy value) or
        numeric seconds.
    :param max_period: longest impedance period in seconds used by
        ``pad="auto"`` (default 30,000 s, the EMTF limit).
    :param detrend: ``"linear"`` (least-squares line, default), ``"median"``
        (the legacy bare median), or None.
    :param taper: ``"tukey"`` (cosine ramps spanning the pad region, default)
        or None (legacy).
    :param gaps: ``"mask"`` (default) leaves missing samples as NaN for
        downstream per-sample weighting; ``"interpolate"`` (legacy) fills
        them by time interpolation, optionally limited to gaps shorter than
        ``max_gap``.
    :param max_gap: with ``gaps="interpolate"``, the longest gap to fill
        (Timedelta-like); longer gaps stay NaN.
    :return: a new Dataset; ``attrs`` carry ``pad_left``/``pad_right``
        (samples), ``dt_seconds``, and the applied options.
    """
    _require_xarray()
    if detrend not in ("linear", "median", None):
        raise ValueError("detrend must be one of: linear, median, None")
    if taper not in ("tukey", None):
        raise ValueError("taper must be one of: tukey, None")
    if gaps not in ("mask", "interpolate"):
        raise ValueError("gaps must be one of: mask, interpolate")

    pad_seconds = 3.*max_period if isinstance(pad, str) and pad == "auto" \
        else _as_seconds(pad)
    pad_delta = pd.Timedelta(seconds=pad_seconds)

    if window is not None:
        start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        out = b.sel(time=slice(start - pad_delta, end + pad_delta))
        if out.sizes["time"] == 0:
            raise ValueError(f"no data in {start - pad_delta} .. {end + pad_delta}")
    else:
        out = b
        start = pd.Timestamp(b["time"].values.min()) + pad_delta
        end = pd.Timestamp(b["time"].values.max()) - pad_delta
    out = out.copy(deep=True)

    dt = _uniform_dt(out["time"].values)
    times = pd.DatetimeIndex(out["time"].values)
    pad_left = int(np.searchsorted(times.values, np.datetime64(start)))
    pad_right = int(len(times) - np.searchsorted(times.values, np.datetime64(end),
                                                 side="right"))
    if pad_left + pad_right >= len(times):
        record = pd.Timestamp(times[-1]) - pd.Timestamp(times[0])
        raise ValueError(
            f"no samples remain in the analysis window {start} .. {end}: the "
            f"record ({record}, {len(times)} samples) is consumed entirely by "
            f"the pad ({pd.Timedelta(seconds=pad_seconds)} per side) -- "
            "everything would be trimmed away. Pass an explicit "
            "window=(start, end), a shorter pad= (e.g. pad='2h'), or a smaller "
            "max_period.")

    if gaps == "interpolate":
        missing = {name: np.isnan(da.transpose("time", ...).values)
                   for name, da in out.data_vars.items() if "time" in da.dims}
        out = out.interpolate_na(dim="time")
        if max_gap is not None:
            # Restore NaN over runs longer than max_gap (interior gaps only;
            # interpolate_na never extrapolates past the record edges)
            n_max = int(round(_as_seconds(max_gap)/dt))
            for name, was_missing in missing.items():
                da = out[name].transpose("time", ...)
                values = da.values
                values[_long_runs(was_missing, n_max)] = np.nan
                out[name] = (da.dims, values, da.attrs)

    taper_window = _pad_taper(len(times), pad_left, pad_right) \
        if taper == "tukey" else None
    for name, da in out.data_vars.items():
        if "time" not in da.dims:
            continue
        values = da.transpose("time", ...).values
        if detrend == "linear":
            values = _detrend_linear(values)
        elif detrend == "median":
            values = values - np.nanmedian(values, axis=0, keepdims=True)
        if taper_window is not None:
            values = values * taper_window.reshape((-1,) + (1,)*(values.ndim - 1))
        out[name] = (da.transpose("time", ...).dims, values, da.attrs)

    out.attrs.update({"pad_left": pad_left, "pad_right": pad_right,
                      "dt_seconds": dt, "detrend": str(detrend),
                      "taper": str(taper), "gaps": gaps})
    return out


def trim_pad(ds, pad_left=None, pad_right=None):
    """Remove the pad added by :func:`prepare` after the convolution.

    Reads ``pad_left``/``pad_right`` (samples) from ``attrs`` unless given
    explicitly, so it works on any product that carried the attrs through
    (E-field Datasets included).
    """
    left = ds.attrs.get("pad_left", 0) if pad_left is None else pad_left
    right = ds.attrs.get("pad_right", 0) if pad_right is None else pad_right
    out = ds.isel(time=slice(left, len(ds["time"]) - right if right else None))
    out.attrs.pop("pad_left", None)
    out.attrs.pop("pad_right", None)
    return out
