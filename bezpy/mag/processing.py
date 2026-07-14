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

__all__ = ["combine_observatories", "despike", "prepare", "trim_pad"]

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


def _hampel_mask(values, window, nsig, sig_floor, iterations):
    """Iterated Hampel outlier mask over the first axis of a 2-D array.

    ``values`` is ``(time, ncols)``; returns a boolean array of the same shape
    marking outliers. Each pass flags samples deviating from a centered rolling
    median by more than ``nsig`` robust sigmas (``1.4826 * MAD``, floored at
    ``sig_floor``); flagged samples are removed before the next pass, so a run
    of bad samples longer than half the window still erodes from its edges
    inward (a plain single-pass Hampel cannot clean a run wider than the
    window, because the median itself is then contaminated).
    """
    w = int(window)
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    min_periods = w // 2 + 1
    df = pd.DataFrame(np.asarray(values, dtype="float64"))
    bad = np.zeros(df.shape, dtype=bool)
    work = df
    for _ in range(int(iterations)):
        med = work.rolling(w, center=True, min_periods=min_periods).median()
        dev = (work - med).abs()
        mad = dev.rolling(w, center=True, min_periods=min_periods).median()
        sigma = (1.4826 * mad).clip(lower=sig_floor)
        new_bad = (dev > nsig * sigma).to_numpy() & np.isfinite(df.to_numpy())
        new_bad &= ~bad
        if not new_bad.any():
            break
        bad |= new_bad
        work = df.mask(bad)
    return bad


def despike(b, window="31min", nsig=6.0, sig_floor=3.0, iterations=5,
            return_flags=False):
    """Mask instrumental spikes, rails, and steps in magnetic time series.

    Field magnetometers -- especially variometers distributed through
    seismological FDSN feeds -- carry non-physical artifacts that a plain
    boxcar decimation passes straight through: isolated wild samples, multi-
    minute saturation rails (values pinned near an instrument limit), and DC
    baseline steps (re-zeroing, or the ~1 Hz->10 Hz channel switch on the
    USGS/EarthScope stations). All three inject broadband high-frequency
    energy, which is exactly the band that dominates the induced geoelectric
    field, so left in place they inflate SECS-interpolated E by large factors.

    This applies an iterated Hampel (rolling median / MAD) filter along the
    time axis and sets flagged samples to NaN, leaving them for the per-sample
    gap handling downstream (``gaps="mask"`` in :func:`prepare`, infinite
    variance in :class:`~bezpy.mag.SECSInterpolator`). Isolated spikes and
    multi-minute rails are removed outright. Real storm signal is preserved --
    a smoothly evolving large excursion tracks the rolling median and is not
    flagged (validated against co-located observatories in ``bezpy.validation``).

    A *pure* DC step (a persistent re-level with no accompanying spike) is not
    an outlier against a rolling median and is intentionally left in place: the
    variometer baseline is arbitrary and removed per-window by :func:`prepare`'s
    detrend anyway, so a step only matters to an analysis window that straddles
    it. When it does, mask a band around the discontinuity or process the two
    sides as separate windows.

    :param b: :class:`xarray.Dataset` (``Bx``/``By``/``Bz`` or ``X``/``Y``/``Z``
        over a ``time`` dim, optionally with an ``obs``/``site`` dim),
        :class:`pandas.DataFrame` (time-indexed component columns), or a bare
        ``(time,)`` / ``(time, ncols)`` :class:`numpy.ndarray`.
    :param window: Hampel window as a duration (``"31min"``, the default) when
        the time step is known (Dataset/DataFrame), or an integer sample count.
        A duration is converted with the series' own sampling interval.
    :param nsig: robust-sigma threshold; larger flags less (default 6.0).
    :param sig_floor: minimum robust sigma in data units (default 3.0 nT). The
        rolling MAD collapses toward zero over a locally smooth stretch, so
        without a floor the tiny (few-nT) curvature at the crest of a real,
        smoothly-varying storm would trip the threshold; 3 nT keeps such crests
        while leaving the huge (10^3-10^5 nT) artifacts far above it.
    :param iterations: maximum erosion passes (default 5; converges in ~3 even
        for multi-minute rails).
    :param return_flags: if True, also return the boolean outlier mask.
    :return: the despiked input in its own type (NaN where flagged), with
        ``n_despiked`` recorded in ``attrs`` for a Dataset. With
        ``return_flags``, a ``(despiked, flags)`` tuple.
    """
    if hasattr(b, "data_vars"):                                  # xarray Dataset
        out = b.copy(deep=True)
        dt = _uniform_dt(out["time"].values)
        w = _window_samples(window, dt)
        flags = {}
        total = 0
        for name, da in out.data_vars.items():
            if "time" not in da.dims:
                continue
            da_t = da.transpose("time", ...)
            arr = da_t.values.astype("float64")
            flat = arr.reshape(arr.shape[0], -1)
            mask = _hampel_mask(flat, w, nsig, sig_floor, iterations)
            flat[mask] = np.nan
            out[name] = (da_t.dims, flat.reshape(arr.shape), da.attrs)
            flags[name] = mask.reshape(arr.shape)
            total += int(mask.sum())
        out.attrs["n_despiked"] = total
        return (out, flags) if return_flags else out

    if isinstance(b, pd.DataFrame):
        dt = _uniform_dt(b.index.values)
        w = _window_samples(window, dt)
        mask = _hampel_mask(b.to_numpy(), w, nsig, sig_floor, iterations)
        out = b.mask(pd.DataFrame(mask, index=b.index, columns=b.columns))
        return (out, mask) if return_flags else out

    arr = np.asarray(b, dtype="float64")
    flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)
    w = _window_samples(window, None)
    mask = _hampel_mask(flat, w, nsig, sig_floor, iterations)
    out = arr.copy()
    out.reshape(flat.shape)[mask] = np.nan
    mask = mask.reshape(arr.shape)
    return (out, mask) if return_flags else out


def _window_samples(window, dt):
    """Hampel window in samples from a duration (needs ``dt``) or an int."""
    if isinstance(window, (int, np.integer)):
        return int(window)
    if dt is None:
        raise ValueError("window must be an integer sample count when the "
                         "sampling interval is unknown (bare-array input)")
    return max(3, int(round(_as_seconds(window) / dt)))


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
