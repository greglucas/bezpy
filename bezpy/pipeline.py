"""The B -> E (-> V) pipeline: composition of the building blocks.

Three modes cover the ways these calculations are used:

* :func:`scan` -- the statistics path. The **entire record** is processed in
  chunks with streaming reducers (running maxima, threshold exceedances,
  histogram-based quantiles), so multi-decade hazard statistics never require
  pre-selecting storms by a global index -- locally-driven events that leave
  Dst unmoved are counted too, and full E/V series are never materialized.
  Chunk edges are protected by the convolution pad, so results are invariant
  to the chunk size.
* :func:`storm_scan` -- the case-study path: explicit windows from a storm
  catalog (or hand-picked), returning per-window maxima.
* :class:`Nowcast` -- the operational path: both of the above are acausal
  (FFT convolution needs future samples and a pad on both sides of the
  analysis window), so neither can run as new B arrives in real time.
  ``Nowcast`` composes ``bezpy.mt.impulse.DTIR`` (causal FIR convolution)
  with ``SECSInterpolator.transfer_matrix`` (a constant, per-sample obs->site
  B map) into a stateful engine: ``update(new_b)`` appends samples to a
  bounded rolling buffer and returns E (and V) for whatever interval just
  became causally valid, with no window recomputation.

The single-window building blocks :func:`calc_geoelectric_field` and
:func:`calc_voltages` are exposed at the top level as
``bezpy.calc_geoelectric_field`` / ``bezpy.calc_voltages``.
"""

__all__ = ["calc_geoelectric_field", "calc_voltages", "scan", "storm_scan", "Nowcast"]

import concurrent.futures
import warnings

import numpy as np
import pandas as pd

from .mag.processing import DEFAULT_MAX_PERIOD, combine_observatories, despike, prepare, trim_pad
from .mag.secs import SECSInterpolator, _extract_b
from .mt.impulse import DTIR
from .mt.site import Site3d, SiteCollection
from .tl.kernels import apply_voltage_matrix, voltage_matrix
from .tl.network import apply_gic_matrix, gic_matrix

# Histogram sketch used for streaming quantiles: log-spaced bins covering
# any plausible |E| (mV/km) or |V| (volts) with ~1.5% resolution.
_SKETCH_BINS = np.r_[0., np.logspace(-4, 5, 1350)]


def _require_xarray():
    import xarray as xr
    return xr


def _site_latlons(sites):
    return np.array([[site.latitude, site.longitude] for site in sites])


def _maybe_despike(b_obs, despike_obs):
    """Apply :func:`bezpy.mag.despike` when requested (True or a kwargs dict)."""
    if not despike_obs:
        return b_obs
    return despike(b_obs, **(despike_obs if isinstance(despike_obs, dict) else {}))


def _max_site_period(sites):
    periods = [np.max(site.periods) for site in sites
               if getattr(site, "periods", None) is not None]
    return max(periods) if periods else DEFAULT_MAX_PERIOD


def calc_geoelectric_field(b_obs, sites, interp="secs", window=None,
                           pad="auto", detrend="linear", taper="tukey",
                           gaps="mask", despike_obs=False, extrapolate="asymptotic",
                           secs=None, leave_one_out=None, collection=None,
                           n_realizations=None, perturb_b=False, seed=None,
                           return_var=False):
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
    :param despike_obs: if True, run :func:`bezpy.mag.despike` on the combined
        B before preparing it -- masks instrumental spikes, saturation rails,
        and steps that would otherwise inflate the interpolated E. Recommended
        for FDSN variometer inputs; a no-op cost on clean observatory data
        (nothing is flagged). Pass a dict to override despike keywords.
    :param extrapolate: impedance band-edge handling, ``"asymptotic"``
        (default) or the legacy ``"zeros"`` band-stop (see ``Site3d.calcZ``)
    :param secs: optionally a configured ``SECSInterpolator`` to use
    :param leave_one_out: observatory code to exclude from the SECS fit
        (validation hook)
    :param collection: optionally a reused ``SiteCollection`` (keeps its
        interpolated-impedance cache warm across equal-length windows);
        takes precedence over a ``SiteCollection`` passed as ``sites``.
        Ignored when ``n_realizations`` is set -- each realization builds
        its own perturbed collection, so there is no cache to reuse.
    :param n_realizations: uncertainty propagation (roadmap 1.2 phase 1,
        "ensemble mode"). When set, each site's impedance is redrawn
        ``n_realizations`` times within its ``Z_var`` (independent complex
        Gaussian per period/component; see ``Site3d.perturbed``) and the
        whole chain reran per draw, adding a leading ``realization``
        dimension to ``Ex``/``Ey``. Sites without ``Z_var`` are identical
        across every realization -- their contribution to the ensemble
        spread is honestly zero, not fabricated. Memory and compute cost is
        linear in ``n_realizations`` (a full extra FFT convolution per
        draw), so bound it the same way ``scan``'s ``chunk`` bounds memory
        for long records. Default None: the original single point-estimate
        path (no ``realization`` dimension, no extra cost).
    :param perturb_b: also perturb each site's SECS-interpolated B within
        its prediction variance (``SECSInterpolator.predict(return_var=True)``),
        independent per time/site/component -- a cheap, honest but
        temporally-uncorrelated approximation (real SECS prediction errors
        are correlated in time; document this when interpreting the spread).
        Requires ``interp="secs"`` and ``n_realizations`` set. Default False:
        only impedance uncertainty is propagated.
    :param seed: RNG seed for reproducible realizations (default None: a
        fresh, unseeded draw each call)
    :param return_var: analytic uncertainty propagation (roadmap 1.2 phase
        2 -- the deterministic, "linear chain" counterpart of
        ``n_realizations``'s Monte Carlo draws). Adds ``Ex_var``/``Ey_var``
        ((mV/km)^2) computed via ``SiteCollection.convolve_fft(...,
        return_var=True)``: a first-order propagation of each site's
        ``Z_var`` through the same spline/extrapolation/FFT-convolution
        pipeline used for the point estimate (see
        ``Site3d.z_sensitivities``), treating E as linear in Z at fixed B
        (B uncertainty is not included -- use ``n_realizations`` +
        ``perturb_b`` for that). Mutually exclusive with
        ``n_realizations`` (the ensemble mode already gives an empirical
        variance via ``.std("realization")**2``; combining both is
        redundant). Substantially more expensive than the point estimate
        (roughly one extra FFT convolution per valid impedance
        measurement per site), so this is meant for single-window/case-
        study use, not wired into ``scan``. See ``VALIDATION.md`` 1.2 for
        the calibration test and its measured coverage.
    :return: Dataset with ``Ex``/``Ey`` in mV/km; site latitude/longitude
        coordinates; pad metadata already trimmed. Dims ``(time, site)``, or
        ``(realization, time, site)`` when ``n_realizations`` is set.
        ``Ex_var``/``Ey_var`` (time, site) are added when ``return_var=True``.
    """
    if perturb_b and interp != "secs":
        raise ValueError("perturb_b=True requires interp='secs'")
    if perturb_b and n_realizations is None:
        raise ValueError("perturb_b=True has no effect without n_realizations=")
    if return_var and n_realizations is not None:
        raise ValueError("return_var=True and n_realizations= are mutually "
                         "exclusive (n_realizations already gives an "
                         "empirical variance via .std('realization')**2)")

    xr = _require_xarray()
    if hasattr(sites, "sites"):  # a SiteCollection was passed
        if collection is None:
            collection = sites
        sites = sites.sites
    if isinstance(b_obs, dict):
        b_obs = combine_observatories(b_obs)
    b_obs = _maybe_despike(b_obs, despike_obs)
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

    mag_x_var = mag_y_var = None
    if interp == "secs":
        interpolator = secs if secs is not None else SECSInterpolator()
        interpolator.fit(fit_input, gaps="mask")
        b_sites = interpolator.predict(latlons, return_var=perturb_b)
        mag_x = b_sites["Bx"].values
        mag_y = b_sites["By"].values
        if perturb_b:
            mag_x_var = b_sites["Bx_var"].values
            mag_y_var = b_sites["By_var"].values
    elif interp == "uniform":
        if fit_input.sizes["obs"] != 1:
            raise ValueError(
                f"interp='uniform' needs exactly one observatory, got "
                f"{fit_input.sizes['obs']}; use interp='secs' for networks")
        mag_x = fit_input["Bx"].values[:, 0]
        mag_y = fit_input["By"].values[:, 0]
    else:
        raise ValueError("interp must be one of: secs, uniform")

    coords = {"time": prepared["time"].values,
              "site": [site.name for site in sites],
              "latitude": ("site", latlons[:, 0]),
              "longitude": ("site", latlons[:, 1])}
    ex_var_out = ey_var_out = None
    if n_realizations is None:
        collection = collection if collection is not None else SiteCollection(sites)
        if return_var:
            ex, ey, ex_var, ey_var = collection.convolve_fft(
                mag_x, mag_y, dt=dt, extrapolate=extrapolate, return_var=True)
            ex_var_out, ey_var_out = ex_var.T, ey_var.T
        else:
            ex, ey = collection.convolve_fft(mag_x, mag_y, dt=dt, extrapolate=extrapolate)
        dims = ("time", "site")
        ex_out, ey_out = ex.T, ey.T
    else:
        if n_realizations < 1:
            raise ValueError("n_realizations must be a positive integer")
        rng = np.random.default_rng(seed)
        ntimes = mag_x.shape[0]
        ex_out = np.empty((n_realizations, ntimes, len(sites)))
        ey_out = np.empty((n_realizations, ntimes, len(sites)))
        for r in range(n_realizations):
            realized = SiteCollection([site.perturbed(rng) for site in sites])
            if perturb_b:
                mx = mag_x + rng.normal(size=mag_x.shape) * np.sqrt(np.maximum(mag_x_var, 0.))
                my = mag_y + rng.normal(size=mag_y.shape) * np.sqrt(np.maximum(mag_y_var, 0.))
            else:
                mx, my = mag_x, mag_y
            rex, rey = realized.convolve_fft(mx, my, dt=dt, extrapolate=extrapolate)
            ex_out[r], ey_out[r] = rex.T, rey.T
        dims = ("realization", "time", "site")
        coords["realization"] = np.arange(n_realizations)

    data_vars = {"Ex": (dims, ex_out, {"units": "mV/km"}),
                 "Ey": (dims, ey_out, {"units": "mV/km"})}
    if return_var:
        data_vars["Ex_var"] = (dims, ex_var_out, {"units": "(mV/km)^2"})
        data_vars["Ey_var"] = (dims, ey_var_out, {"units": "(mV/km)^2"})
    e = xr.Dataset(
        data_vars, coords=coords,
        attrs={key: prepared.attrs[key]
               for key in ("pad_left", "pad_right", "dt_seconds")})
    return trim_pad(e)


def calc_voltages(lines, e, how="delaunay", return_var=False):
    """Voltages across transmission lines from site electric fields.

    Applies the precomputed Level-1 voltage kernels
    (:mod:`bezpy.tl.kernels`) -- one sparse matmul for all lines, never a
    per-line Python loop. Interpolation weights are built from the E-field
    site locations if a line does not have them yet.

    :param lines: sequence of TransmissionLines
    :param e: ``Ex``/``Ey`` Dataset from :func:`calc_geoelectric_field`
        (or a bare ``(ntimes, nsites, 2)`` array in mV/km)
    :param how: E interpolation scheme (``"delaunay"`` default, ``"nn"``)
    :param return_var: also propagate ``Ex_var``/``Ey_var`` (from
        ``calc_geoelectric_field(..., return_var=True)``) into a per-line
        voltage variance (roadmap 1.2 phase 2), via the closed form
        ``var(V) = sum_j L_j^2 var(E_j)`` -- exact given the E-site
        variances (``L`` is the same static, real, known kernel used for
        the point estimate, and the site-to-site independence assumption
        already made when drawing each site's ``Z_var`` separately in
        phase 1/2). Requires a Dataset ``e`` carrying ``Ex_var``/``Ey_var``.
        This is the propagated **data** variance only -- the delaunay-vs-nn
        interpolation-*scheme* divergence is a separate, **structural**
        uncertainty not included here (see ``VALIDATION.md`` D7 for its
        measured magnitude); add it yourself if you need one combined
        number, since how large it is depends on your own site geometry.
    :return: DataArray ``(time, line)`` of voltages in volts (NaN for lines
        outside the site coverage); bare-array input returns an ndarray.
        With ``return_var=True``, a ``(V, V_var)`` tuple of DataArrays
        (units V and V^2).
    """
    is_dataset = hasattr(e, "data_vars")
    if return_var and not (is_dataset and "Ex_var" in e and "Ey_var" in e):
        raise ValueError("return_var=True requires a Dataset e= carrying "
                         "Ex_var/Ey_var (calc_geoelectric_field(..., "
                         "return_var=True)'s output)")
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
    v_da = xr.DataArray(V, dims=("time", "line"),
                        coords={"time": e["time"].values},
                        attrs={"units": "V"}, name="voltage")
    if not return_var:
        return v_da
    E_var = np.stack([e["Ex_var"].values, e["Ey_var"].values], axis=-1)
    V_var = apply_voltage_matrix(E_var, L**2, valid)
    v_var_da = xr.DataArray(V_var, dims=("time", "line"),
                            coords={"time": e["time"].values},
                            attrs={"units": "V^2"}, name="voltage_var")
    return v_da, v_var_da


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


# The two optional secondary reductions scan() can produce alongside E,
# keyed the same way throughout: reducer variable dimension, and the
# apply_*_matrix function that turns a chunk's E into that quantity's
# magnitude given its precomputed kernel.
_EXTRA_DIMS = {"V": "line", "I": "transformer"}


def _build_extra_kernels(lines, sites, how, network):
    """Precompute the optional V (line voltage) and I (transformer GIC)
    kernels, keyed like :data:`_EXTRA_DIMS`. Both are static across chunks
    (geometry and network topology never change), so :func:`_scan_chunk`
    only ever applies them.

    Sets each line's interpolation weights (if not already set) as a side
    effect -- ``gic_matrix`` needs them set the same way, so building the V
    kernel first (when both are requested) means it never redoes that work.
    """
    kernels = {}
    if lines is not None:
        site_xys = _site_latlons(sites)
        for line in lines:
            if how == "delaunay" and line.delaunay_vtx is None:
                line.set_delaunay_weights(site_xys)
            elif how == "nn" and line.nearest_sites is None:
                line.set_nearest_sites(site_xys)
        kernels["V"] = voltage_matrix(lines, len(sites), how=how)
    if network is not None:
        if lines is None:
            raise ValueError("network= requires lines= (GIC is driven by "
                             "line-integrated E, computed from the same lines)")
        kernels["I"] = gic_matrix(network, lines, len(sites), how=how)
    return kernels


def _scan_chunk(b_obs, sites, collection, window, field_kwargs, extra_kernels):
    """One chunk's reduced inputs: ``(times, |E|, {"V": |V|, "I": |I|, ...})``.

    The unit of work for both the sequential loop and each parallel worker
    in :func:`scan` -- raises ``ValueError`` ("no data in ...") for a chunk
    that falls entirely in a record gap, same as :func:`calc_geoelectric_field`.
    """
    e = calc_geoelectric_field(b_obs, sites, window=window,
                               collection=collection, **field_kwargs)
    times = pd.DatetimeIndex(e["time"].values).values
    magnitude = np.hypot(e["Ex"].values, e["Ey"].values)
    extras = {}
    if extra_kernels:
        E = np.stack([e["Ex"].values, e["Ey"].values], axis=-1)
        for name, (K, valid) in extra_kernels.items():
            apply_fn = apply_voltage_matrix if name == "V" else apply_gic_matrix
            extras[name] = np.abs(apply_fn(E, K, valid))
    return times, magnitude, extras


def _scan_chunk_or_skip(b_obs, sites, collection, window, field_kwargs, extra_kernels):
    """:func:`_scan_chunk`, but a record-gap chunk returns None instead of
    raising -- the shared skip policy for both the sequential loop and
    parallel workers."""
    try:
        return _scan_chunk(b_obs, sites, collection, window, field_kwargs, extra_kernels)
    except ValueError as exc:
        if "no data in" in str(exc):  # record gap spanning the chunk
            return None
        raise


# Per-process state for parallel scan workers, set once by _init_scan_worker
# (not per chunk) so the SiteCollection cache and the V/I kernels are built
# only once per worker rather than rebuilt on every task.
_scan_worker_state = None


def _init_scan_worker(b_obs, sites, field_kwargs, extra_kernels):
    global _scan_worker_state
    _scan_worker_state = (b_obs, sites, SiteCollection(sites), field_kwargs, extra_kernels)


def _scan_chunk_worker(window):
    b_obs, sites, collection, field_kwargs, extra_kernels = _scan_worker_state
    return _scan_chunk_or_skip(b_obs, sites, collection, window, field_kwargs, extra_kernels)


def _scan_ensemble(b_obs, sites, lines, network, chunk, reduce, threshold,
                   quantiles, out_path, how, despike_obs, workers,
                   n_realizations, seed, **field_kwargs):
    """``n_realizations`` independent :func:`scan` runs over impedance draws
    fixed for the whole record, stacked on a new ``realization`` dimension.

    See the ``n_realizations`` entry of :func:`scan`'s docstring for the
    fixed-per-scan-not-per-chunk rationale and current limitations.
    """
    if out_path is not None:
        raise ValueError("out_path (resumability) is not supported with "
                         "n_realizations yet")
    if "exceedances" in reduce:
        raise ValueError(
            "reduce='exceedances' is not supported with n_realizations "
            "(each realization collects a different number of exceedances, "
            "which cannot stack on a shared dimension); run per-realization "
            "scans separately if exceedance ensembles are needed")
    xr = _require_xarray()
    if hasattr(sites, "sites"):
        sites = sites.sites
    rng = np.random.default_rng(seed)
    results = [
        scan(b_obs, [site.perturbed(rng) for site in sites], lines=lines,
            network=network, chunk=chunk, reduce=reduce, threshold=threshold,
            quantiles=quantiles, how=how, despike_obs=despike_obs,
            workers=workers, **field_kwargs)
        for _ in range(n_realizations)
    ]
    stacked = xr.concat(results, dim="realization")
    stacked = stacked.assign_coords(realization=np.arange(n_realizations))
    for prefix in ("E", "V", "I"):
        maxvar = f"{prefix}_max"
        if maxvar in stacked:
            stacked[f"{prefix}_max_mean"] = stacked[maxvar].mean("realization")
            stacked[f"{prefix}_max_std"] = stacked[maxvar].std("realization")
    return stacked


def scan(b_obs, sites, lines=None, network=None, chunk="1Y", reduce=("max",),
         threshold=None, quantiles=(0.5, 0.9, 0.99), out_path=None,
         how="delaunay", despike_obs=False, workers=None,
         n_realizations=None, seed=None, **field_kwargs):
    """Continuous scan of the whole record with streaming reducers (D4).

    Processes ``b_obs`` chunk by chunk through
    :func:`calc_geoelectric_field` (and :func:`calc_voltages` when lines are
    given, and :func:`bezpy.tl.network.gic_matrix` when a network is given
    too), reducing |E| (and |V|, |I|) on the fly -- the full series are
    never held in memory. Each chunk is padded and trimmed by the underlying
    machinery, so the result is invariant to the chunk size.

    :param b_obs: ``(time, obs)`` Dataset covering the whole record
    :param sites: list of MT sites (``Site3d``/``Site1d``) or a
        ``SiteCollection``
    :param lines: optional TransmissionLines to reduce voltages for
    :param network: optional ``bezpy.tl.network.Network`` to reduce
        per-transformer GIC for (requires ``lines``, since GIC is driven by
        the same line-integrated E; each network branch is matched to one
        of ``lines`` by ``line.metadata["id"]``)
    :param chunk: chunk length as a pandas period frequency (default one
        year; use e.g. ``"3M"`` to bound memory for dense site sets).
        Peak memory per chunk is roughly ``50 * nsites * chunk_samples``
        bytes (E fields plus bounded FFT workspace) -- one year of 1-minute
        data is ~25 MB per site, so hundreds of sites want monthly chunks
        and gridded products (10^4 sites) want ~daily ones
    :param reduce: any of ``"max"``, ``"exceedances"``, ``"quantiles"``
    :param threshold: exceedance level for ``"exceedances"``: a number
        applied to every quantity, or ``{"E": mv_per_km, "V": volts, "I": amps}``
    :param quantiles: quantile levels reported for ``"quantiles"``
    :param out_path: write the (small) result Dataset to this netCDF path
        after every chunk, making long scans resumable: if the file already
        exists, completed chunks are skipped and accumulation continues.
        Checkpoints are still written strictly in chunk order with
        ``workers`` set, so resuming behaves identically either way.
    :param despike_obs: if True (or a dict of :func:`bezpy.mag.despike`
        keywords), despike ``b_obs`` once up front (not per chunk) --
        recommended for FDSN variometer inputs
    :param workers: if set, process chunks in parallel across this many
        subprocesses (``concurrent.futures.ProcessPoolExecutor``) instead of
        one at a time. Chunks are independent by construction (D4 -- see
        ``test_scan_chunk_invariance``), so results are bit-identical to
        ``workers=None``; only wall-clock time changes. ``b_obs`` and
        ``sites``/``lines`` are sent to each worker once (via the pool
        initializer) and each worker rebuilds its own ``SiteCollection``
        impedance cache, so they must be picklable -- true for anything that
        already round-trips through ``SiteCollection.to_netcdf`` /
        ``TransmissionLine``. Default ``None`` (sequential, the previous
        behavior).
    :param field_kwargs: forwarded to :func:`calc_geoelectric_field`
        (``interp``, ``extrapolate``, ``leave_one_out``, ...)
    :param n_realizations: uncertainty propagation (roadmap 1.2 phase 1).
        When set, each site's impedance is redrawn ``n_realizations`` times
        within its ``Z_var`` *once, up front* (the perturbed sites are fixed
        for the whole scan -- a site's true impedance does not fluctuate
        chunk to chunk, so each realization must stay internally consistent
        across the record) and the whole scan is rerun per draw via
        ``n_realizations`` recursive calls to :func:`scan`, adding a leading
        ``realization`` dimension to every reduced variable. Convenience
        ``*_max_mean``/``*_max_std`` variables summarize the spread.
        Not compatible with ``reduce="exceedances"`` (variable-length per
        realization) or ``out_path`` (resumability) in this phase; B-field
        uncertainty (``calc_geoelectric_field``'s ``perturb_b``) is not
        propagated here either -- use ``calc_geoelectric_field`` directly
        for storm-window ensembles that need it. Cost is linear in
        ``n_realizations`` (a full extra scan per draw).
    :param seed: RNG seed for reproducible realizations (default None)
    :return: Dataset with per-site (and per-line, per-transformer)
        reductions: ``E_max``, ``E_max_time`` [, ``E_qXX``, exceedance
        variables on their own dimension; likewise ``V_*``/``I_*``]. Feed
        exceedances to ``bezpy.storms.find_events`` for an event catalog and
        to ``bezpy.stats.return_level`` for hazard levels. With
        ``n_realizations`` set, every reduced variable gains a leading
        ``realization`` dimension (plus ``*_max_mean``/``*_max_std``).
    """
    if workers is not None and workers < 1:
        raise ValueError("workers must be a positive integer or None")
    if n_realizations is not None:
        return _scan_ensemble(b_obs, sites, lines=lines, network=network,
                              chunk=chunk, reduce=reduce, threshold=threshold,
                              quantiles=quantiles, out_path=out_path, how=how,
                              despike_obs=despike_obs, workers=workers,
                              n_realizations=n_realizations, seed=seed,
                              **field_kwargs)
    xr = _require_xarray()
    if hasattr(sites, "sites"):  # a SiteCollection was passed
        collection = sites
        sites = collection.sites
    else:
        collection = SiteCollection(sites)
    b_obs = _maybe_despike(b_obs, despike_obs)
    chunks = _chunk_edges(b_obs["time"].values, chunk)
    thresholds = threshold if isinstance(threshold, dict) \
        else dict.fromkeys(("E", "V", "I"), threshold)
    site_reducer = _Reducers(len(sites), reduce, thresholds.get("E"))

    # The V/I kernels are static across chunks: set interpolation weights,
    # solve the network structure, and contract the geometry once, then
    # apply per chunk (never rebuild).
    extra_kernels = _build_extra_kernels(lines, sites, how, network)
    extra_n = {"V": len(lines) if lines is not None else 0,
              "I": len(network.transformers) if network is not None else 0}
    extra_reducers = {name: _Reducers(extra_n[name], reduce, thresholds.get(name))
                      for name in extra_kernels}

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
            _restore_reducers(existing, site_reducer, extra_reducers)

    pending = [(c0, c1) for c0, c1 in chunks
              if done_until is None or c1 > done_until]

    def consume(result, c1):
        if result is None:
            return
        times, magnitude, extras = result
        site_reducer.update(times, magnitude)
        for name, reducer in extra_reducers.items():
            reducer.update(times, extras[name])
        if out_path is not None:
            checkpoint = _scan_result(xr, site_reducer, extra_reducers, sites,
                                      quantiles, threshold, done_until=c1)
            checkpoint.to_netcdf(out_path)

    if workers is None:
        for c0, c1 in pending:
            consume(_scan_chunk_or_skip(b_obs, sites, collection, (c0, c1),
                                        field_kwargs, extra_kernels), c1)
    else:
        # Submit every pending chunk up front (they run concurrently across
        # the pool), then consume results strictly in chunk order -- this is
        # what keeps output (and checkpoint) order identical to sequential.
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, initializer=_init_scan_worker,
                initargs=(b_obs, sites, field_kwargs, extra_kernels)) as pool:
            futures = [pool.submit(_scan_chunk_worker, w) for w in pending]
            for (c0, c1), future in zip(pending, futures):
                consume(future.result(), c1)

    return _scan_result(xr, site_reducer, extra_reducers, sites, quantiles,
                        threshold, done_until=chunks[-1][1])


def _scan_result(xr, site_reducer, extra_reducers, sites, quantiles, threshold,
                 done_until):
    """Assemble the (small) output Dataset from reducer state."""
    latlons = _site_latlons(sites)
    ds = xr.Dataset(coords={"site": [s.name for s in sites],
                            "latitude": ("site", latlons[:, 0]),
                            "longitude": ("site", latlons[:, 1])})
    ds.attrs["scan_done_until"] = str(done_until)
    ds.attrs["n_samples"] = site_reducer.n_samples
    thresholds = threshold if isinstance(threshold, dict) \
        else dict.fromkeys(("E", "V", "I"), threshold)
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
    for name, reducer in extra_reducers.items():
        dim = _EXTRA_DIMS[name]
        ds = ds.assign_coords(**{dim: np.arange(reducer.n)})
        add(reducer, dim, name)
    return ds


def _restore_reducers(existing, site_reducer, extra_reducers):
    """Load accumulated reducer state from a partially-completed scan file."""
    reducers = {"E": site_reducer, **extra_reducers}
    for prefix, reducer in reducers.items():
        if f"{prefix}_max" not in existing:
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
               despike_obs=False, **field_kwargs):
    """Per-storm maxima for a catalog of explicit windows (case-study path).

    :param catalog: DataFrame with ``start``/``end`` columns
        (``bezpy.storms.find_storms`` output) or a sequence of
        ``(start, end)`` pairs
    :param b_obs: ``(time, obs)`` Dataset covering the storms
    :param sites: list of MT sites (``Site3d``/``Site1d``) or a
        ``SiteCollection``
    :param lines: optional TransmissionLines
    :param despike_obs: if True (or a dict of :func:`bezpy.mag.despike`
        keywords), despike ``b_obs`` once before the storm loop -- recommended
        for FDSN variometer inputs
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
    b_obs = _maybe_despike(b_obs, despike_obs)
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


# ----------------------------------------------------------------------------
# Nowcast: causal, streaming B -> E (-> V) (roadmap 2.2)
# ----------------------------------------------------------------------------

def _site_periods(site, periods):
    """This site's impedance period band, or the ``periods=`` fallback.

    Raises for a site with neither -- the caller (``Nowcast``) needs a band
    to size the DTIR and to build it from an explicit ``Z``.
    """
    if getattr(site, "periods", None) is not None:
        return np.asarray(site.periods, dtype=float)
    if periods is not None:
        return np.asarray(periods, dtype=float)
    raise ValueError(
        f"site {site.name!r} has no .periods (not a Site3d); pass periods= "
        "for Site1d/period-less sites, or preconfigured dtirs=")


def _resolve_nmin_nmax(sites, dt, nmin, nmax, periods):
    """``(nmin, nmax)`` in samples, auto-derived from the sites' period band
    when requested, plus the undersampled-period warning (see the DTIR
    reconstruction-fidelity evidence in ``ROADMAP.md`` 2.2)."""
    bands = [_site_periods(site, periods) for site in sites]
    max_period = max(np.max(band) for band in bands)
    min_period = min(np.min(band) for band in bands)
    if nmax == "auto":
        nmax = max(1, int(np.ceil(2.0 * max_period / dt)))
    if nmin == "auto":
        nmin = -max(1, int(np.ceil(0.1 * min_period / dt)))
    if min_period < 10 * dt:
        warnings.warn(
            f"shortest site period ({min_period:.0f}s) is less than 10x dt "
            f"({dt}s); DTIR reconstruction of periods this close to the "
            "Nyquist limit is unreliable (measured 50-65% relative RMS "
            "error vs the batch FFT path at this ratio -- see Nowcast's "
            "Accuracy docstring section for the well-sampled-band numbers) "
            "-- restrict the periods passed to DTIR (dtirs=) or use finer "
            "B sampling", stacklevel=3)
    return int(nmin), int(nmax)


def _build_dtir(site, dt, nmin, nmax, decay_factor, Q_choice, periods):
    """A calculated DTIR for one site: from the site's own Z if it's a
    Site3d, else from ``periods=`` and ``site.calcZ`` (physical sites like
    Site1d are valid at any frequency, so any band works)."""
    dtir = DTIR(dt=dt, nmin=nmin, nmax=nmax, decay_factor=decay_factor, Q_choice=Q_choice)
    if isinstance(site, Site3d):
        dtir.calculate(site=site)
    else:
        p = _site_periods(site, periods)
        dtir.calculate(periods=p, Z=site.calcZ(1. / p))
    return dtir


class Nowcast:
    """Causal, streaming B -> E (-> V), for operational/real-time use.

    ``calc_geoelectric_field``/``scan`` are acausal: FFT convolution needs
    a pad of *future* samples on both sides of the analysis window, so
    neither can produce a result as new B arrives live. ``Nowcast`` composes
    the same site impedances differently -- through ``bezpy.mt.impulse.DTIR``
    (a regularized time-domain impulse response, causal FIR convolution)
    fed by ``SECSInterpolator.transfer_matrix`` (roadmap 2.1: a constant,
    per-sample-independent observatory-to-site B map) -- into a stateful
    engine: :meth:`update` appends new B to a bounded rolling buffer and
    returns E (and V) for whatever interval just became causally valid.

    **Latency.** Two independent sources, both fixed once built:

    * A start-up transient of :attr:`warmup_samples` (``2 * max(nmax,
      |nmin|) + 1``) samples before *any* output appears -- this is larger
      than ``nmax`` alone because of how ``DTIR.convolve``'s ``mode="same"``
      convolution is aligned, not a implementation looseness.
    * Thereafter, a steady-state delay of ``|nmin|`` samples: DTIR is only
      quasi-causal (Kelbert et al. 2017's design uses a small acausal lead),
      so an output sample only becomes final once ``|nmin|`` further samples
      have arrived.

    **Accuracy.** Two independent effects, both measured directly, not
    assumed:

    * Periods close to ``dt``'s Nyquist limit are an aliasing failure no
      amount of ``decay_factor``/``Q_choice`` tuning fixes (measured 50-65%
      relative RMS error for a realistic EMTF band reaching ~100s periods at
      ``dt=60``\\ s, vs ~a few percent for a band well above it, in practice
      :math:`\\gtrsim 10 \\times dt`). ``nmin``/``nmax="auto"`` warns when
      this applies; restrict the periods passed to DTIR (``dtirs=``) or use
      finer B sampling (``dt``) if it does.
    * Independently, even entirely within a well-sampled band, DTIR's
      regularized inversion reconstructs **broadband** B (many active
      frequencies at once, as in a real storm) much better than
      **narrowband** B (one or two dominant frequencies): measured ~5%
      relative RMS for a 15-tone broadband synthetic vs 10-50% for the same
      band driven by just 1-2 tones at the same periods, same
      ``decay_factor``. The regularized fit's per-frequency error doesn't
      average away when only a few frequencies carry the signal's energy.
      Real geomagnetic storm B is broadband, so this favors real operational
      use over narrowband synthetic tests -- but it means a narrowband bench
      test of this class understates error at unlucky frequencies and
      should not be read as "DTIR is accurate to a few percent" in general.

    ``despike_obs`` is not supported here: despike's centered rolling window
    needs its own future-sample lookahead on the raw observatory stream, a
    second instance of the buffering problem this class solves for DTIR.
    Run :func:`bezpy.mag.despike` on each chunk yourself before calling
    :meth:`update` if you need it.

    Parameters
    ----------
    sites : list of Site3d/Site1d, or a SiteCollection
        MT sites E/V are computed at.
    secs : SECSInterpolator
        An unfit-but-configured interpolator; ``Nowcast`` calls ``.fit()`` on
        it (mutating it in place), same convention as
        ``calc_geoelectric_field``'s own ``secs=``. Defaults to
        ``method="lstsq"`` engaging the fast constant-matrix path
        (:meth:`~bezpy.mag.SECSInterpolator.transfer_matrix`) once the first
        gap-free chunk arrives; ``method="kalman"`` (or any chunk with a
        gap) falls back to a per-call fit over the rolling buffer instead.
    dt : float
        Sample interval in seconds. Required -- unlike ``pad="auto"``, there
        is no data-derived point at which to infer this before the DTIRs
        are built.
    nmin, nmax : int or "auto"
        DTIR sample-offset window (see :class:`~bezpy.mt.impulse.DTIR`).
        ``"auto"`` (default) derives them from the sites' period band: see
        the accuracy note above.
    decay_factor, Q_choice
        Forwarded to every auto-built DTIR.
    periods : array, optional
        Fallback impedance periods for sites without ``.periods`` (Site1d or
        other period-less sites).
    dtirs : list of DTIR, optional
        Pre-``.calculate()``'d DTIRs, one per site, for full manual control.
        Must all share one ``(dt, nmin, nmax)`` -- the buffer/finalization
        bookkeeping needs a single shared horizon.
    lines : sequence of TransmissionLine, optional
        Also reduce voltages via the existing static Level-1 kernel
        (:func:`~bezpy.tl.kernels.voltage_matrix`) -- no new math, applied
        per call.
    how : str
        E interpolation scheme for ``lines`` (``"delaunay"`` default, ``"nn"``).
    force_slow_secs : bool
        Testing-only: disable the fast constant-matrix SECS path
        unconditionally, so it can be compared against the fallback on
        identical input.
    """

    def __init__(self, sites, secs, dt, nmin="auto", nmax="auto",
                decay_factor=1e-2, Q_choice=1, periods=None, dtirs=None,
                lines=None, how="delaunay", force_slow_secs=False):
        if hasattr(sites, "sites"):  # a SiteCollection was passed
            sites = sites.sites
        self.sites = list(sites)
        self.secs = secs
        self.dt = float(dt)
        self.force_slow_secs = force_slow_secs

        if dtirs is not None:
            if len(dtirs) != len(self.sites):
                raise ValueError("dtirs must have one entry per site")
            dts = {d.dt for d in dtirs}
            nmins = {d.nmin for d in dtirs}
            nmaxs = {d.nmax for d in dtirs}
            if len(dts) != 1 or len(nmins) != 1 or len(nmaxs) != 1:
                raise ValueError(
                    "all dtirs must share one (dt, nmin, nmax); Nowcast's "
                    "rolling buffer needs a single shared finalization horizon")
            self._dtirs = list(dtirs)
            self.dt = dts.pop()
            self.nmin, self.nmax = nmins.pop(), nmaxs.pop()
        else:
            self.nmin, self.nmax = _resolve_nmin_nmax(self.sites, self.dt, nmin, nmax, periods)
            self._dtirs = [_build_dtir(site, self.dt, self.nmin, self.nmax,
                                       decay_factor, Q_choice, periods)
                          for site in self.sites]

        self.warmup_samples = 2 * max(self.nmax, abs(self.nmin)) + 1

        self._lines = lines
        if lines is not None:
            site_xys = _site_latlons(self.sites)
            for line in lines:
                if how == "delaunay" and line.delaunay_vtx is None:
                    line.set_delaunay_weights(site_xys)
                elif how == "nn" and line.nearest_sites is None:
                    line.set_nearest_sites(site_xys)
            self._L, self._valid = voltage_matrix(lines, len(self.sites), how=how)
        else:
            self._L = self._valid = None

        n = len(self.sites)
        self._M = None
        self._M_flat = None
        self._obs_latlons = None
        self._obs_names = None
        self._buf_bx = np.empty((0, n))
        self._buf_by = np.empty((0, n))
        self._buf_b_obs = None
        self._buffer_start = 0
        self._last_finalized_abs = -1
        self._n_total = 0
        self._t0 = None
        self._is_xarray = None

    def update(self, new_b, obs_latlons=None):
        """Append new B samples and return E (and V) for the newly-valid interval.

        :param new_b: same forms as ``SECSInterpolator.fit``'s ``b_obs``: an
            xarray Dataset (``Bx``/``By``[/``Bz``] dimensioned ``(time, obs)``,
            latitude/longitude coordinates), or a bare ``(nnew, nobs, 2 or 3)``
            ndarray with ``obs_latlons=`` given. The observatory set (and, for
            xarray input, the time axis contiguity) must stay fixed across
            calls -- build a new ``Nowcast`` if the network changes.
        :param obs_latlons: required for bare-array input (see above).
        :return: an ``xr.Dataset`` (``Ex``/``Ey``[, ``V``], dims ``(time,
            site[, line])``) when ``new_b`` is/was ever xarray, else a
            ``(sample_index, Ex, Ey[, V])`` ndarray tuple. Zero newly-valid
            samples returns a zero-length result, never ``None``, never an
            exception -- this is the normal state during
            :attr:`warmup_samples`.
        """
        B, latlons, times, obs_names = _extract_b(new_b, obs_latlons)
        if self._is_xarray is None:
            self._is_xarray = times is not None
        if self._obs_latlons is None:
            self._obs_latlons, self._obs_names = latlons, obs_names
        elif not np.allclose(latlons, self._obs_latlons) or (
                obs_names is not None and self._obs_names is not None
                and list(obs_names) != list(self._obs_names)):
            raise ValueError(
                "Nowcast's observatory set must stay fixed across update() "
                "calls; build a new Nowcast if the network changes")
        if times is not None:
            if self._t0 is None:
                self._t0 = pd.Timestamp(times[0])
            else:
                expected = self._t0 + pd.to_timedelta(self._n_total * self.dt, unit="s")
                if pd.Timestamp(times[0]) != expected:
                    raise ValueError(
                        f"gap or overlap in the input stream: expected "
                        f"{expected}, got {pd.Timestamp(times[0])}")

        site_latlons = _site_latlons(self.sites)
        has_gap = np.isnan(B[:, :, :2]).any()

        if (self._M is None and self.secs.method == "lstsq"
                and not has_gap and not self.force_slow_secs):
            self.secs.fit(B, obs_latlons=self._obs_latlons, gaps="mask")
            self._M = self.secs.transfer_matrix(site_latlons)
            self._M_flat = self._M.reshape(len(self._obs_latlons) * 2, -1)

        if self._M is not None and not has_gap and not self.force_slow_secs:
            B_site_new = (B[:, :, :2].reshape(len(B), -1) @ self._M_flat
                         ).reshape(len(B), len(self.sites), 2)
        elif self.secs.method == "kalman":
            fit_input = (np.concatenate([self._buf_b_obs, B], axis=0)
                        if self._buf_b_obs is not None else B)
            # Kalman needs strictly-increasing times for its tau (seconds)
            # correlation time; a synthetic sample-cadence axis is correct
            # regardless of whether the caller's input was xarray or bare
            # (KalmanSECS only uses differences between samples).
            tmp = SECSInterpolator(**self.secs._config())
            tmp.fit(fit_input, obs_latlons=self._obs_latlons, gaps="mask",
                   times=np.arange(len(fit_input)) * self.dt)
            B_site_new = tmp.predict(site_latlons)[-len(B):, :, :2]
        else:
            # lstsq is memoryless in time -- only this chunk is needed
            tmp = SECSInterpolator(**self.secs._config())
            tmp.fit(B, obs_latlons=self._obs_latlons, gaps="mask")
            B_site_new = tmp.predict(site_latlons)[:, :, :2]

        if np.isnan(B_site_new).any():
            raise ValueError(
                "SECS-interpolated B has NaN at a site (a simultaneous "
                "all-observatory dropout) -- Nowcast is not gap-tolerant at "
                "that granularity; despike/mask upstream removes single-"
                "station artifacts but not a total network outage")

        self._buf_bx = np.concatenate([self._buf_bx, B_site_new[:, :, 0]], axis=0)
        self._buf_by = np.concatenate([self._buf_by, B_site_new[:, :, 1]], axis=0)
        if self.secs.method == "kalman":
            self._buf_b_obs = (B if self._buf_b_obs is None
                               else np.concatenate([self._buf_b_obs, B], axis=0))
        self._n_total += len(B)

        ex_new = ey_new = np.empty((0, len(self.sites)))
        abs_start_finalized = None
        buf_len = len(self._buf_bx)

        if buf_len >= self.warmup_samples:
            prev_local = self._last_finalized_abs - self._buffer_start + 1
            new_last_local = buf_len - 1 - abs(self.nmin)
            if new_last_local >= prev_local:
                ex_full = np.empty((buf_len, len(self.sites)))
                ey_full = np.empty((buf_len, len(self.sites)))
                for i, dtir in enumerate(self._dtirs):
                    ex_full[:, i], ey_full[:, i] = dtir.convolve(
                        self._buf_bx[:, i], self._buf_by[:, i])
                ex_new = ex_full[prev_local:new_last_local + 1]
                ey_new = ey_full[prev_local:new_last_local + 1]
                abs_start_finalized = self._buffer_start + prev_local
                self._last_finalized_abs = self._buffer_start + new_last_local

            # Two independent floors: never trim past what a not-yet-
            # finalized future index will need as left context, and never
            # let the buffer drop below warmup_samples (that changes
            # np.convolve(mode="same")'s output length -- see the
            # warmup_samples docstring note above).
            target_start_local = (self._last_finalized_abs + 1 - self.nmax) - self._buffer_start
            keep_from_local = max(0, min(target_start_local, buf_len - self.warmup_samples))
            if keep_from_local > 0:
                self._buf_bx = self._buf_bx[keep_from_local:]
                self._buf_by = self._buf_by[keep_from_local:]
                if self._buf_b_obs is not None:
                    self._buf_b_obs = self._buf_b_obs[keep_from_local:]
                self._buffer_start += keep_from_local

        V_new = None
        if self._lines is not None and len(ex_new):
            E_new = np.stack([ex_new, ey_new], axis=-1)
            V_new = apply_voltage_matrix(E_new, self._L, self._valid)

        return self._package(ex_new, ey_new, V_new, abs_start_finalized)

    def _package(self, ex_new, ey_new, V_new, abs_start_finalized):
        n = len(ex_new)
        if not self._is_xarray:
            idx = (np.arange(abs_start_finalized, abs_start_finalized + n)
                  if abs_start_finalized is not None else np.arange(0))
            if self._lines is not None:
                return idx, ex_new, ey_new, V_new
            return idx, ex_new, ey_new

        xr = _require_xarray()
        if abs_start_finalized is not None:
            time = self._t0 + pd.to_timedelta(
                (abs_start_finalized + np.arange(n)) * self.dt, unit="s")
        else:
            time = pd.DatetimeIndex([])
        site_latlons = _site_latlons(self.sites)
        coords = {"time": time, "site": [s.name for s in self.sites],
                 "latitude": ("site", site_latlons[:, 0]),
                 "longitude": ("site", site_latlons[:, 1])}
        data_vars = {"Ex": (("time", "site"), ex_new, {"units": "mV/km"}),
                    "Ey": (("time", "site"), ey_new, {"units": "mV/km"})}
        if self._lines is not None:
            v = V_new if V_new is not None else np.empty((n, len(self._lines)))
            data_vars["V"] = (("time", "line"), v, {"units": "V"})
            coords["line"] = np.arange(len(self._lines))
        return xr.Dataset(
            data_vars, coords=coords,
            attrs={"dt_seconds": self.dt, "nmin": self.nmin, "nmax": self.nmax,
                  "warmup_samples": self.warmup_samples, "n_total": self._n_total})
