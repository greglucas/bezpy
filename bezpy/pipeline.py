"""The B -> E (-> V) pipeline: composition of the building blocks.

Two scan modes cover the two ways these calculations are used:

* :func:`scan` -- the statistics path. The **entire record** is processed in
  chunks with streaming reducers (running maxima, threshold exceedances,
  histogram-based quantiles), so multi-decade hazard statistics never require
  pre-selecting storms by a global index -- locally-driven events that leave
  Dst unmoved are counted too, and full E/V series are never materialized.
  Chunk edges are protected by the convolution pad, so results are invariant
  to the chunk size.
* :func:`storm_scan` -- the case-study path: explicit windows from a storm
  catalog (or hand-picked), returning per-window maxima.

The single-window building blocks :func:`calc_geoelectric_field` and
:func:`calc_voltages` are exposed at the top level as
``bezpy.calc_geoelectric_field`` / ``bezpy.calc_voltages``.
"""

__all__ = ["calc_geoelectric_field", "calc_voltages", "scan", "storm_scan"]

import numpy as np
import pandas as pd

from .mag.processing import DEFAULT_MAX_PERIOD, combine_observatories, prepare, trim_pad
from .mag.secs import SECSInterpolator
from .mt.site import SiteCollection
from .tl.kernels import apply_voltage_matrix, voltage_matrix

# Histogram sketch used for streaming quantiles: log-spaced bins covering
# any plausible |E| (mV/km) or |V| (volts) with ~1.5% resolution.
_SKETCH_BINS = np.r_[0., np.logspace(-4, 5, 1350)]


def _require_xarray():
    import xarray as xr
    return xr


def _site_latlons(sites):
    return np.array([[site.latitude, site.longitude] for site in sites])


def _max_site_period(sites):
    periods = [np.max(site.periods) for site in sites
               if getattr(site, "periods", None) is not None]
    return max(periods) if periods else DEFAULT_MAX_PERIOD


def calc_geoelectric_field(b_obs, sites, interp="secs", window=None,
                           pad="auto", detrend="linear", taper="tukey",
                           gaps="mask", extrapolate="zeros", secs=None,
                           leave_one_out=None, collection=None):
    """Geoelectric fields at MT sites from observatory magnetic fields.

    The full chain: :func:`bezpy.mag.prepare` (detrend/taper/pad) ->
    interpolate B onto the sites -> batched FFT convolution with each site's
    impedance -> :func:`bezpy.mag.trim_pad`.

    :param b_obs: ``(time, obs)`` Dataset (see
        ``bezpy.mag.combine_observatories`` /
        ``bezpy.data.load_observatory_group``), or a dict of per-observatory
        Datasets/DataFrames which is combined first
    :param sites: list of MT sites (``Site3d``/``Site1d``) or a
        ``SiteCollection``
    :param interp: how B reaches the sites: ``"secs"`` (default) fits a SECS
        interpolator so each site sees its local field; ``"uniform"`` applies
        the single observatory's field everywhere (requires exactly one
        gap-free observatory; the Kelbert & Lucas 2020 assumption)
    :param window: optional ``(start, end)`` analysis interval; padding is
        added around it and trimmed from the result
    :param pad: pad duration per side; ``"auto"`` derives it from the longest
        site impedance period (see :func:`bezpy.mag.prepare`)
    :param detrend: passed to :func:`bezpy.mag.prepare`
    :param taper: passed to :func:`bezpy.mag.prepare`
    :param gaps: passed to :func:`bezpy.mag.prepare`
    :param extrapolate: impedance band-edge handling (``"zeros"`` or
        ``"asymptotic"``, see ``Site3d.calcZ``)
    :param secs: optionally a configured ``SECSInterpolator`` to use
    :param leave_one_out: observatory code to exclude from the SECS fit
        (validation hook)
    :param collection: optionally a reused ``SiteCollection`` (keeps its
        interpolated-impedance cache warm across equal-length windows);
        takes precedence over a ``SiteCollection`` passed as ``sites``
    :return: Dataset with ``Ex``/``Ey`` (time, site) in mV/km; site
        latitude/longitude coordinates; pad metadata already trimmed
    """
    xr = _require_xarray()
    if hasattr(sites, "sites"):  # a SiteCollection was passed
        if collection is None:
            collection = sites
        sites = sites.sites
    if isinstance(b_obs, dict):
        b_obs = combine_observatories(b_obs)
    prepared = prepare(b_obs, window=window, pad=pad, detrend=detrend,
                       taper=taper, gaps=gaps,
                       max_period=_max_site_period(sites) if pad == "auto"
                       else DEFAULT_MAX_PERIOD)
    dt = prepared.attrs["dt_seconds"]
    latlons = _site_latlons(sites)

    if leave_one_out is not None:
        if leave_one_out not in prepared["obs"].values:
            raise ValueError(f"leave_one_out={leave_one_out!r} is not one of "
                             f"{list(prepared['obs'].values)}")
        fit_input = prepared.drop_sel(obs=leave_one_out)
    else:
        fit_input = prepared

    if interp == "secs":
        interpolator = secs if secs is not None else SECSInterpolator()
        interpolator.fit(fit_input, gaps="mask")
        b_sites = interpolator.predict(latlons)
        mag_x = b_sites["Bx"].values
        mag_y = b_sites["By"].values
    elif interp == "uniform":
        if fit_input.sizes["obs"] != 1:
            raise ValueError(
                f"interp='uniform' needs exactly one observatory, got "
                f"{fit_input.sizes['obs']}; use interp='secs' for networks")
        mag_x = fit_input["Bx"].values[:, 0]
        mag_y = fit_input["By"].values[:, 0]
    else:
        raise ValueError("interp must be one of: secs, uniform")

    collection = collection if collection is not None else SiteCollection(sites)
    ex, ey = collection.convolve_fft(mag_x, mag_y, dt=dt, extrapolate=extrapolate)

    e = xr.Dataset(
        {"Ex": (("time", "site"), ex.T, {"units": "mV/km"}),
         "Ey": (("time", "site"), ey.T, {"units": "mV/km"})},
        coords={"time": prepared["time"].values,
                "site": [site.name for site in sites],
                "latitude": ("site", latlons[:, 0]),
                "longitude": ("site", latlons[:, 1])},
        attrs={key: prepared.attrs[key]
               for key in ("pad_left", "pad_right", "dt_seconds")})
    return trim_pad(e)


def calc_voltages(lines, e, how="delaunay"):
    """Voltages across transmission lines from site electric fields.

    Applies the precomputed Level-1 voltage kernels
    (:mod:`bezpy.tl.kernels`) -- one sparse matmul for all lines, never a
    per-line Python loop. Interpolation weights are built from the E-field
    site locations if a line does not have them yet.

    :param lines: sequence of TransmissionLines
    :param e: ``Ex``/``Ey`` Dataset from :func:`calc_geoelectric_field`
        (or a bare ``(ntimes, nsites, 2)`` array in mV/km)
    :param how: E interpolation scheme (``"delaunay"`` default, ``"nn"``)
    :return: DataArray ``(time, line)`` of voltages in volts (NaN for lines
        outside the site coverage); bare-array input returns an ndarray
    """
    is_dataset = hasattr(e, "data_vars")
    if is_dataset:
        E = np.stack([e["Ex"].values, e["Ey"].values], axis=-1)
        site_xys = np.column_stack([e["latitude"].values, e["longitude"].values])
    else:
        E = np.asarray(e)
        site_xys = None

    needs_weights = any(
        (how == "delaunay" and line.delaunay_vtx is None) or
        (how == "nn" and line.nearest_sites is None) for line in lines)
    if needs_weights:
        if site_xys is None:
            raise ValueError("bare-array input requires the lines to have "
                             "their interpolation weights already set")
        for line in lines:
            if how == "delaunay" and line.delaunay_vtx is None:
                line.set_delaunay_weights(site_xys)
            elif how == "nn" and line.nearest_sites is None:
                line.set_nearest_sites(site_xys)

    L, valid = voltage_matrix(lines, E.shape[1], how=how)
    V = apply_voltage_matrix(E, L, valid)
    if not is_dataset:
        return V
    xr = _require_xarray()
    return xr.DataArray(V, dims=("time", "line"),
                        coords={"time": e["time"].values},
                        attrs={"units": "V"}, name="voltage")


def _chunk_edges(times, chunk):
    """Half-open chunk intervals covering the time axis."""
    start = pd.Timestamp(times.min())
    end = pd.Timestamp(times.max())
    periods = pd.period_range(start, end, freq=chunk)
    return [(p.start_time, p.end_time) for p in periods]


class _Reducers:
    """Streaming reducers over (time, item) magnitude blocks."""

    # Warn once when the exceedance list passes this size -- a threshold set
    # too low on a long scan accumulates unbounded memory silently otherwise.
    _EXCEEDANCE_WARN_AT = 1_000_000

    def __init__(self, n_items, reduce, threshold):
        self.reduce = list(reduce)
        self.threshold = threshold
        self.n = n_items
        self.max = np.full(n_items, -np.inf)
        self.max_time = np.full(n_items, np.datetime64("NaT", "ns"))
        self.counts = np.zeros((n_items, len(_SKETCH_BINS) - 1), dtype=np.int64)
        self.n_samples = 0
        self.exceedances = []  # (time, item index, value) tuples
        self._exceedance_warned = False
        if "exceedances" in self.reduce and threshold is None:
            raise ValueError("reduce='exceedances' requires threshold=")

    def update(self, times, magnitudes):
        """magnitudes: (ntimes, n_items), NaN allowed (skipped)."""
        self.n_samples += magnitudes.shape[0]
        filled = np.where(np.isfinite(magnitudes), magnitudes, -np.inf)
        idx = filled.argmax(axis=0)
        block_max = filled[idx, np.arange(self.n)]
        better = block_max > self.max
        self.max = np.where(better, block_max, self.max)
        self.max_time[better] = times[idx[better]]

        if "quantiles" in self.reduce:
            for j in range(self.n):
                good = np.isfinite(magnitudes[:, j])
                self.counts[j] += np.histogram(magnitudes[good, j],
                                               bins=_SKETCH_BINS)[0]
        if "exceedances" in self.reduce:
            with np.errstate(invalid="ignore"):
                t_idx, item = np.nonzero(magnitudes > self.threshold)
            self.exceedances.extend(zip(times[t_idx], item,
                                        magnitudes[t_idx, item]))
            if (not self._exceedance_warned
                    and len(self.exceedances) > self._EXCEEDANCE_WARN_AT):
                self._exceedance_warned = True
                import warnings
                warnings.warn(
                    f"over {self._EXCEEDANCE_WARN_AT:,} exceedances collected "
                    f"-- threshold={self.threshold} is likely too low for this "
                    "record; memory grows with every exceedance", stacklevel=2)

    def quantiles(self, qs):
        """Approximate quantiles from the accumulated histogram sketch."""
        out = np.full((self.n, len(qs)), np.nan)
        centers = np.sqrt(_SKETCH_BINS[:-1] * np.maximum(_SKETCH_BINS[1:], 1e-30))
        centers[0] = 0.
        for j in range(self.n):
            total = self.counts[j].sum()
            if total == 0:
                continue
            cdf = np.cumsum(self.counts[j]) / total
            out[j] = centers[np.searchsorted(cdf, qs, side="left").clip(
                max=len(centers) - 1)]
        return out


def scan(b_obs, sites, lines=None, chunk="1Y", reduce=("max",),
         threshold=None, quantiles=(0.5, 0.9, 0.99), out_path=None,
         how="delaunay", **field_kwargs):
    """Continuous scan of the whole record with streaming reducers (D4).

    Processes ``b_obs`` chunk by chunk through
    :func:`calc_geoelectric_field` (and :func:`calc_voltages` when lines are
    given), reducing |E| (and |V|) on the fly -- the full series are never
    held in memory. Each chunk is padded and trimmed by the underlying
    machinery, so the result is invariant to the chunk size.

    :param b_obs: ``(time, obs)`` Dataset covering the whole record
    :param sites: list of MT sites (``Site3d``/``Site1d``) or a
        ``SiteCollection``
    :param lines: optional TransmissionLines to reduce voltages for
    :param chunk: chunk length as a pandas period frequency (default one
        year; use e.g. ``"3ME"`` to bound memory for dense site sets).
        Peak memory per chunk is roughly ``50 * nsites * chunk_samples``
        bytes (E fields plus bounded FFT workspace) -- one year of 1-minute
        data is ~25 MB per site, so hundreds of sites want monthly chunks
        and gridded products (10^4 sites) want ~daily ones
    :param reduce: any of ``"max"``, ``"exceedances"``, ``"quantiles"``
    :param threshold: exceedance level for ``"exceedances"``: a number
        applied to both quantities, or ``{"E": mv_per_km, "V": volts}``
    :param quantiles: quantile levels reported for ``"quantiles"``
    :param out_path: write the (small) result Dataset to this netCDF path
        after every chunk, making long scans resumable: if the file already
        exists, completed chunks are skipped and accumulation continues
    :param field_kwargs: forwarded to :func:`calc_geoelectric_field`
        (``interp``, ``extrapolate``, ``leave_one_out``, ...)
    :return: Dataset with per-site (and per-line) reductions: ``E_max``,
        ``E_max_time`` [, ``E_qXX``, exceedance variables on their own
        dimension]. Feed exceedances to ``bezpy.storms.find_events`` for an
        event catalog and to ``bezpy.stats.return_level`` for hazard levels.
    """
    xr = _require_xarray()
    if hasattr(sites, "sites"):  # a SiteCollection was passed
        collection = sites
        sites = collection.sites
    else:
        collection = SiteCollection(sites)
    chunks = _chunk_edges(b_obs["time"].values, chunk)
    thresholds = threshold if isinstance(threshold, dict) \
        else {"E": threshold, "V": threshold}
    site_reducer = _Reducers(len(sites), reduce, thresholds.get("E"))
    line_reducer = _Reducers(len(lines), reduce, thresholds.get("V")) \
        if lines is not None else None

    # The voltage kernel is static across chunks: set interpolation weights
    # and contract the geometry once, then apply per chunk (never rebuild).
    if lines is not None:
        site_xys = _site_latlons(sites)
        for line in lines:
            if how == "delaunay" and line.delaunay_vtx is None:
                line.set_delaunay_weights(site_xys)
            elif how == "nn" and line.nearest_sites is None:
                line.set_nearest_sites(site_xys)
        L, valid = voltage_matrix(lines, len(sites), how=how)

    done_until = None
    if out_path is not None:
        import os
        if os.path.exists(out_path):
            with xr.open_dataset(out_path) as existing:
                existing.load()
            if "scan_done_until" not in existing.attrs:
                raise ValueError(
                    f"{out_path} exists but is not a scan result (no "
                    "scan_done_until attribute); refusing to overwrite it")
            done_until = pd.Timestamp(existing.attrs["scan_done_until"])
            _restore_reducers(existing, site_reducer, line_reducer)

    for c0, c1 in chunks:
        if done_until is not None and c1 <= done_until:
            continue
        try:
            e = calc_geoelectric_field(b_obs, sites, window=(c0, c1),
                                       collection=collection, **field_kwargs)
        except ValueError as exc:
            if "no data in" in str(exc):  # record gap spanning the chunk
                continue
            raise
        times = pd.DatetimeIndex(e["time"].values)

        magnitude = np.hypot(e["Ex"].values, e["Ey"].values)
        site_reducer.update(times.values, magnitude)
        if line_reducer is not None:
            E = np.stack([e["Ex"].values, e["Ey"].values], axis=-1)
            v = apply_voltage_matrix(E, L, valid)
            line_reducer.update(times.values, np.abs(v))

        if out_path is not None:
            result = _scan_result(xr, site_reducer, line_reducer, sites,
                                  quantiles, threshold, done_until=c1)
            result.to_netcdf(out_path)

    return _scan_result(xr, site_reducer, line_reducer, sites, quantiles,
                        threshold, done_until=chunks[-1][1])


def _scan_result(xr, site_reducer, line_reducer, sites, quantiles, threshold,
                 done_until):
    """Assemble the (small) output Dataset from reducer state."""
    latlons = _site_latlons(sites)
    ds = xr.Dataset(coords={"site": [s.name for s in sites],
                            "latitude": ("site", latlons[:, 0]),
                            "longitude": ("site", latlons[:, 1])})
    ds.attrs["scan_done_until"] = str(done_until)
    ds.attrs["n_samples"] = site_reducer.n_samples
    thresholds = threshold if isinstance(threshold, dict) \
        else {"E": threshold, "V": threshold}
    for key, value in thresholds.items():
        if value is not None:
            ds.attrs[f"threshold_{key}"] = value

    def add(reducer, dim, prefix):
        ds[f"{prefix}_max"] = ((dim,), reducer.max)
        ds[f"{prefix}_max_time"] = ((dim,), reducer.max_time)
        # Persist the histogram so resumed scans keep accumulating
        ds[f"{prefix}_sketch"] = ((dim, "bin"), reducer.counts)
        if "quantiles" in reducer.reduce:
            values = reducer.quantiles(np.asarray(quantiles))
            for q, column in zip(quantiles, values.T):
                ds[f"{prefix}_q{int(round(q*100)):02d}"] = ((dim,), column)
        if "exceedances" in reducer.reduce:
            events = reducer.exceedances
            edim = f"{prefix}_exceedance"
            ds[f"{edim}_time"] = ((edim,), np.array([t for t, _, _ in events],
                                                    dtype="datetime64[ns]"))
            ds[f"{edim}_item"] = ((edim,), np.array([i for _, i, _ in events],
                                                    dtype=np.int64))
            ds[f"{edim}_value"] = ((edim,), np.array([v for _, _, v in events]))

    add(site_reducer, "site", "E")
    if line_reducer is not None:
        ds = ds.assign_coords(line=np.arange(line_reducer.n))
        add(line_reducer, "line", "V")
    return ds


def _restore_reducers(existing, site_reducer, line_reducer):
    """Load accumulated reducer state from a partially-completed scan file."""
    for reducer, prefix in ((site_reducer, "E"), (line_reducer, "V")):
        if reducer is None or f"{prefix}_max" not in existing:
            continue
        reducer.max = existing[f"{prefix}_max"].values.copy()
        reducer.max_time = existing[f"{prefix}_max_time"].values.copy()
        reducer.counts = existing[f"{prefix}_sketch"].values.copy()
        reducer.n_samples = int(existing.attrs.get("n_samples", 0))
        edim = f"{prefix}_exceedance"
        if f"{edim}_time" in existing:
            reducer.exceedances = list(zip(existing[f"{edim}_time"].values,
                                           existing[f"{edim}_item"].values,
                                           existing[f"{edim}_value"].values))


def storm_scan(catalog, b_obs, sites, lines=None, how="delaunay",
               **field_kwargs):
    """Per-storm maxima for a catalog of explicit windows (case-study path).

    :param catalog: DataFrame with ``start``/``end`` columns
        (``bezpy.storms.find_storms`` output) or a sequence of
        ``(start, end)`` pairs
    :param b_obs: ``(time, obs)`` Dataset covering the storms
    :param sites: list of MT sites (``Site3d``/``Site1d``) or a
        ``SiteCollection``
    :param lines: optional TransmissionLines
    :param field_kwargs: forwarded to :func:`calc_geoelectric_field`
    :return: Dataset ``(storm, site[, line])`` with ``E_max`` (mV/km),
        ``V_max`` (volts), and the storm windows as coordinates
    """
    xr = _require_xarray()
    if isinstance(catalog, pd.DataFrame):
        windows = list(zip(catalog["start"], catalog["end"]))
    else:
        windows = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in catalog]

    if hasattr(sites, "sites"):  # a SiteCollection was passed
        collection = sites
        sites = collection.sites
    else:
        collection = SiteCollection(sites)
    e_max, v_max = [], []
    for start, end in windows:
        try:
            e = calc_geoelectric_field(b_obs, sites, window=(start, end),
                                       collection=collection, **field_kwargs)
        except ValueError as exc:
            if "no data in" not in str(exc):
                raise
            # Index-derived catalogs often outspan the magnetic record; keep
            # the storm row (NaN) instead of failing the whole scan.
            import warnings
            warnings.warn(f"storm window {start} .. {end} has no magnetic "
                          "data; recording NaN maxima", stacklevel=2)
            e_max.append(np.full(len(sites), np.nan))
            if lines is not None:
                v_max.append(np.full(len(lines), np.nan))
            continue
        magnitude = np.hypot(e["Ex"].values, e["Ey"].values)
        e_max.append(np.nanmax(magnitude, axis=0))
        if lines is not None:
            v = calc_voltages(lines, e, how=how)
            # Lines outside the site coverage are all-NaN; keep them NaN
            # without the all-NaN-slice RuntimeWarning
            av = np.abs(v.values)
            filled = np.where(np.isnan(av), -np.inf, av).max(axis=0)
            v_max.append(np.where(np.isfinite(filled), filled, np.nan))

    latlons = _site_latlons(sites)
    ds = xr.Dataset(
        {"E_max": (("storm", "site"), np.array(e_max), {"units": "mV/km"})},
        coords={"storm": np.arange(len(windows)),
                "storm_start": ("storm", [w[0] for w in windows]),
                "storm_end": ("storm", [w[1] for w in windows]),
                "site": [s.name for s in sites],
                "latitude": ("site", latlons[:, 0]),
                "longitude": ("site", latlons[:, 1])})
    if lines is not None:
        ds["V_max"] = (("storm", "line"), np.array(v_max), {"units": "V"})
    return ds
