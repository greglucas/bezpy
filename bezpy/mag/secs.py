"""SECS interpolation of magnetic observatory data.

Wraps `pysecs <https://github.com/greglucas/pysecs>`_ into a fit/predict
interpolator with the configuration used across the published geoelectric
hazard studies (divergence-free SEC shell at 110 km, epsilon = 0.05, Bz
excluded from the fit) as the baseline, and gap handling that keeps every
sample: missing data are masked with infinite variance instead of being
interpolated over or causing whole observatories to be discarded.

Requires pysecs >= 0.4, which provides the temporal Kalman filter/smoother
(``method="kalman"``), robust IRLS fitting (``robust=``), station-informed
grid generation, and prediction variances (``predict(return_var=True)``).
"""

__all__ = ["SECSInterpolator"]

import numpy as np

R_EARTH = 6371e3  # m


def _require_pysecs():
    try:
        import pysecs
    except ImportError:
        raise ImportError(
            "SECSInterpolator requires pysecs; install it with "
            "`pip install pysecs` (or the bezpy[secs] extra).") from None
    if not hasattr(pysecs, "KalmanSECS"):
        raise ImportError(
            "SECSInterpolator requires pysecs >= 0.4 "
            "(KalmanSECS, robust fitting, prediction variances); "
            "upgrade with `pip install -U pysecs`.")
    return pysecs


def _extract_latlons(locations):
    """(n, 2) [lat, lon] from an array, site-like objects, or a Dataset.

    Bare arrays are interpreted as (lat, lon) columns -- note this is the
    opposite order of ``TransmissionLine.pts`` (lon/lat).
    """
    # xarray objects with latitude/longitude coordinates
    if hasattr(locations, "coords") and "latitude" in locations.coords:
        return np.column_stack([np.atleast_1d(locations["latitude"].values),
                                np.atleast_1d(locations["longitude"].values)])
    # Objects with latitude/longitude attributes (e.g. Site3d)
    if isinstance(locations, (list, tuple)) and \
            all(hasattr(loc, "latitude") for loc in locations):
        return np.array([[loc.latitude, loc.longitude] for loc in locations])
    latlons = np.atleast_2d(np.asarray(locations, dtype=float))
    if latlons.shape[-1] != 2:
        raise ValueError("locations must be (n, 2) [latitude, longitude]")
    return latlons


def _extract_b(b_obs, obs_latlons):
    """Normalize B input to ``(B (ntimes, nobs, 3), latlons, times, obs_names)``.

    Accepts an xarray Dataset with ``Bx``/``By``[/``Bz``] (or ``X``/``Y``/``Z``)
    variables dimensioned (time, obs) and latitude/longitude coordinates on
    the obs dimension, or a bare ndarray (ntimes, nobs, 2 or 3) in nT with
    ``obs_latlons`` given explicitly.
    """
    if hasattr(b_obs, "data_vars"):  # xarray Dataset
        if "Bx" in b_obs:
            names = ("Bx", "By", "Bz")
        elif "X" in b_obs:
            names = ("X", "Y", "Z")
        else:
            raise ValueError("Dataset must contain Bx/By[/Bz] or X/Y[/Z] variables")
        present = [n for n in names if n in b_obs]
        arrays = [b_obs[n].transpose("time", "obs").values for n in present]
        B = np.stack(arrays, axis=-1)
        latlons = _extract_latlons(b_obs)
        times = b_obs["time"].values
        obs_names = b_obs["obs"].values
    else:
        B = np.asarray(b_obs, dtype=float)
        if B.ndim != 3 or B.shape[-1] not in (2, 3):
            raise ValueError("B must have shape (ntimes, nobs, 2 or 3)")
        if obs_latlons is None:
            raise ValueError("obs_latlons is required with bare-array input")
        latlons = _extract_latlons(obs_latlons)
        times = obs_names = None

    if B.shape[-1] == 2:  # no vertical component measured
        B = np.concatenate([B, np.full(B.shape[:-1] + (1,), np.nan)], axis=-1)
    return B, latlons, times, obs_names


def _distance_to_gap(missing):
    """Per-column distance (in samples) to the nearest missing sample.

    ``missing`` is boolean (ntimes, ...); the distance is computed along the
    time axis independently for each trailing-index column.
    """
    ntimes = missing.shape[0]
    flat = missing.reshape(ntimes, -1)
    dist = np.full(flat.shape, ntimes, dtype=float)
    from scipy.ndimage import distance_transform_edt
    for j in np.nonzero(flat.any(axis=0))[0]:
        dist[:, j] = distance_transform_edt(~flat[:, j])
    return dist.reshape(missing.shape)


class SECSInterpolator:
    """Interpolate observatory magnetic fields with Spherical Elementary
    Current Systems (Amm & Viljanen, 1999) via pysecs.

    The baseline configuration matches the published hazard workflows:
    a divergence-free SEC shell at 110 km altitude, epsilon = 0.05
    truncation of the SVD, and the vertical component excluded from the
    fit. Missing samples are masked with infinite variance (kept out of
    the fit without fabricating data or discarding stations); the legacy
    discard-the-observatory behavior is available as ``gaps="drop"``.

    Parameters
    ----------
    grid : "auto", (west, south, east, north), or (n, 2) array
        SEC pole locations. ``"auto"`` builds a regular grid covering the
        observatories with ``pysecs.make_grid`` (padding beyond the
        network, longitude spacing scaled by 1/cos(latitude) so cells
        stay roughly square); a 4-tuple gives the grid bounding box
        explicitly; an array gives (lat, lon) poles directly.
    spacing : float or None
        Grid spacing in degrees for constructed grids (default 2, the
        paper configuration). None with ``grid="auto"`` uses the
        station-informed pysecs default (median nearest-neighbor
        separation of the observatories).
    pad : float
        Padding in degrees beyond the observatory bounding box for
        ``grid="auto"`` (default 5).
    altitude_km : float
        Altitude of the SEC shell in km (default 110).
    epsilon : float
        SVD truncation level, relative to the largest singular value
        (default 0.05). See :meth:`loo_score` for tuning it on data.
    fit_bz : bool
        Include the vertical component in the fit (default False, as in
        the published workflows).
    method : {"lstsq", "kalman"}
        Fitting method. ``"lstsq"`` (default) fits every time step
        independently with a truncated-SVD least squares.
        ``"kalman"`` couples the time steps with pysecs' Kalman filter
        and Rauch-Tung-Striebel smoother: amplitudes evolve as an AR(1)
        process with correlation time ``tau``, which suppresses the
        step-to-step flicker of independent snapshot fits and rides
        through station dropouts. Note the smoother stores the full
        amplitude covariance, ``8 * ntimes * npoles**2`` bytes -- size
        the grid and analysis window accordingly.
    tau : float
        ``method="kalman"`` only: temporal correlation time of the SEC
        amplitudes in seconds (default 3600). Tune with
        :meth:`loo_score` rather than trusting the default.
    prior_std : float, optional
        ``method="kalman"`` only: stationary prior standard deviation of
        the amplitudes in amperes; the temporal analogue of ``epsilon``.
        Default None estimates it from a snapshot fit.
    gate_sigma : float, optional
        ``method="kalman"`` only: innovation gating threshold in
        standard deviations; observations inconsistent with the forecast
        (single-station spikes) are downweighted before they can steer
        the solution. None (default) disables gating.
    smoother : bool
        ``method="kalman"`` only: run the backward smoother (non-causal,
        recommended for post-processing; default True). False gives a
        purely causal filter for real-time use.
    robust : {None, "huber", "bisquare"}
        ``method="lstsq"`` only: iteratively reweighted least squares.
        Localized single-station artifacts (spikes, baseline jumps) are
        inconsistent with any smooth current system, so they show up as
        large residuals and are automatically downweighted ("huber") or
        fully rejected ("bisquare"). Costs roughly ``robust_maxiter``
        ordinary fits. For ``method="kalman"``, use ``gate_sigma``
        instead.

    Examples
    --------
    >>> interp = SECSInterpolator(grid="auto")
    >>> interp.fit(b_obs)                          # doctest: +SKIP
    >>> b_pred = interp.predict(latlons)           # doctest: +SKIP
    """

    def __init__(self, grid="auto", spacing=2.0, pad=5.0, altitude_km=110.,
                 epsilon=0.05, fit_bz=False, method="lstsq", tau=3600.,
                 prior_std=None, gate_sigma=None, smoother=True, robust=None):
        if method not in ("lstsq", "kalman"):
            raise ValueError("method must be one of: lstsq, kalman")
        if robust not in (None, "huber", "bisquare"):
            raise ValueError("robust must be one of: None, huber, bisquare")
        if robust is not None and method == "kalman":
            raise ValueError("robust= applies to method='lstsq'; for "
                             "method='kalman' use gate_sigma= instead")
        self.grid = grid
        self.spacing = spacing
        self.pad = pad
        self.altitude_km = altitude_km
        self.epsilon = epsilon
        self.fit_bz = fit_bz
        self.method = method
        self.tau = tau
        self.prior_std = prior_std
        self.gate_sigma = gate_sigma
        self.smoother = smoother
        self.robust = robust

        self._secs = None
        self._kalman = None
        self._times = None
        self._is_xarray_fit = False

    def __repr__(self):
        if isinstance(self.grid, str):
            grid = repr(self.grid)
        elif np.ndim(self.grid) == 1:
            grid = repr(tuple(self.grid))
        else:
            grid = f"<{len(self.grid)} poles>"
        parts = [f"grid={grid}", f"spacing={self.spacing}",
                 f"pad={self.pad}", f"altitude_km={self.altitude_km}",
                 f"epsilon={self.epsilon}", f"method={self.method!r}"]
        if self.fit_bz:
            parts.append("fit_bz=True")
        if self.method == "kalman":
            parts.append(f"tau={self.tau}")
            if self.gate_sigma is not None:
                parts.append(f"gate_sigma={self.gate_sigma}")
            if not self.smoother:
                parts.append("smoother=False")
        if self.robust is not None:
            parts.append(f"robust={self.robust!r}")
        state = (f"fitted: {len(self.obs_latlons_)} obs"
                 if self._secs is not None else "unfitted")
        return f"SECSInterpolator({', '.join(parts)}, {state})"

    def _config(self):
        """Constructor arguments reproducing this interpolator's setup."""
        return {name: getattr(self, name) for name in
                ("grid", "spacing", "pad", "altitude_km", "epsilon", "fit_bz",
                 "method", "tau", "prior_std", "gate_sigma", "smoother",
                 "robust")}

    def _grid_latlonr(self, obs_latlons):
        """SEC pole locations (npoles, 3 [lat, lon, r]) for the fit."""
        pysecs = _require_pysecs()
        r_shell = R_EARTH + self.altitude_km*1e3
        if isinstance(self.grid, str) and self.grid == "auto":
            obs_loc = np.column_stack(
                [obs_latlons, np.full(len(obs_latlons), R_EARTH)])
            return pysecs.make_grid(obs_loc, r_shell, spacing=self.spacing,
                                    padding=self.pad)
        if np.ndim(self.grid) == 1 and len(self.grid) == 4:
            if self.spacing is None:
                raise ValueError("spacing=None (station-informed) is only "
                                 "available with grid='auto'; give a numeric "
                                 "spacing with an explicit bounding box")
            west, south, east, north = self.grid
            south, north = max(south, -90.), min(north, 90.)
            lats = np.arange(south, north + self.spacing/2, self.spacing)
            lons = np.arange(west, east + self.spacing/2, self.spacing)
            lat, lon = map(np.ravel, np.meshgrid(lats, lons, indexing="ij"))
            return np.column_stack([lat, lon, np.full(lat.size, r_shell)])
        poles = np.atleast_2d(np.asarray(self.grid, dtype=float))
        return np.column_stack([poles, np.full(len(poles), r_shell)])

    def fit(self, b_obs, obs_latlons=None, obs_std=None, times=None,
            gaps="mask", weight_gaps=False, gap_inflation=10., gap_window=60):
        """Fit the SEC amplitudes to observed magnetic fields.

        :param b_obs: xarray Dataset with ``Bx``/``By``[/``Bz``] (or
            ``X``/``Y``/``Z``) variables dimensioned (time, obs) and
            latitude/longitude coordinates, or ndarray (ntimes, nobs, 2 or 3)
            in nT with ``obs_latlons`` given.
        :param obs_latlons: (nobs, 2) [latitude, longitude], required for
            bare-array input.
        :param obs_std: optional standard error, broadcastable to
            (ntimes, nobs, 3), used to weight the observations.
        :param times: observation times for ``method="kalman"`` with
            bare-array input -- datetime64 or float seconds, strictly
            increasing. Dataset input carries its own time axis.
        :param gaps: how to treat missing (NaN) samples.

            * ``"mask"`` (default): give them infinite variance -- they drop
              out of the fit sample-by-sample and every valid sample of every
              observatory is used. No data are fabricated.
            * ``"drop"``: discard any observatory containing NaNs entirely
              (the endpoint of the legacy interpolate-then-drop workflow;
              kept for reproduction).
        :param weight_gaps: additionally inflate the variance of valid
            samples near gap edges, where single-station spike/offset
            artifacts concentrate.
        :param gap_inflation: standard-error multiplier at a gap edge; decays
            linearly to 1 over ``gap_window`` samples.
        :param gap_window: number of samples over which the inflation decays.
        :return: self
        """
        pysecs = _require_pysecs()
        B, latlons, ds_times, obs_names = _extract_b(b_obs, obs_latlons)
        self._is_xarray_fit = ds_times is not None
        self._times = ds_times if ds_times is not None else times

        std = np.ones(B.shape) * (np.asarray(obs_std, dtype=float)
                                  if obs_std is not None else 1.)
        if not self.fit_bz:
            std[:, :, 2] = np.inf

        missing = np.isnan(B)
        if gaps == "drop":
            keep = ~missing[:, :, :2].any(axis=(0, 2)) if not self.fit_bz \
                else ~missing.any(axis=(0, 2))
            if not keep.any():
                raise ValueError("gaps='drop' removed every observatory")
            B, std, latlons = B[:, keep], std[:, keep], latlons[keep]
            missing = np.isnan(B)
            if obs_names is not None:
                obs_names = np.asarray(obs_names)[keep]
        elif gaps != "mask":
            raise ValueError("gaps must be one of: mask, drop")

        if weight_gaps and missing.any():
            dist = _distance_to_gap(missing)
            with np.errstate(invalid="ignore"):
                ramp = np.clip(1. - (dist - 1.)/gap_window, 0., 1.)
            std = std * (1. + (gap_inflation - 1.)*ramp)
        # Masked samples never enter the fit; zero-fill so B/std is 0, not NaN
        std[missing] = np.inf
        B = np.where(missing, 0., B)

        self.obs_latlons_ = latlons
        self.obs_names_ = obs_names
        obs_loc = np.column_stack([latlons, np.full(len(latlons), R_EARTH)])
        self._secs = pysecs.SECS(sec_df_loc=self._grid_latlonr(latlons))
        self._kalman = None

        if self.method == "kalman":
            if self._times is None:
                raise ValueError("method='kalman' needs the observation "
                                 "times: fit with a Dataset or pass times=")
            self._kalman = pysecs.KalmanSECS(
                self._secs, tau=self.tau, prior_std=self.prior_std,
                gate_sigma=self.gate_sigma, smoother=self.smoother,
                epsilon=self.epsilon)
            self._kalman.fit(obs_loc, B, np.asarray(self._times), obs_std=std)
        else:
            self._secs.fit(obs_loc=obs_loc, obs_B=B, obs_std=std,
                           epsilon=self.epsilon, robust=self.robust)
        return self

    def predict(self, locations, return_var=False):
        """Predict the magnetic field at new locations.

        :param locations: (n, 2) [latitude, longitude] array, a sequence of
            objects with ``.latitude``/``.longitude`` (e.g. MT sites), or an
            xarray object with latitude/longitude coordinates.
        :param return_var: also return the prediction variance, propagated
            from the posterior covariance of the fitted amplitudes (treating
            ``obs_std`` as 1-sigma errors of independent observations).
        :return: if the interpolator was fit with a Dataset, an xarray
            Dataset with ``Bx``/``By``/``Bz`` (time, loc), location
            coordinates, and ``Bx_var``/``By_var``/``Bz_var`` when
            ``return_var``; otherwise an ndarray (ntimes, n, 3 [Bx, By, Bz])
            (a ``(B, var)`` tuple when ``return_var``).
        """
        latlons = _extract_latlons(locations)
        B, var = self._predict_arrays(latlons, return_var)

        if not self._is_xarray_fit:
            return (B, var) if return_var else B
        import xarray as xr
        coords = {"time": self._times,
                  "latitude": ("loc", latlons[:, 0]),
                  "longitude": ("loc", latlons[:, 1])}
        out = xr.Dataset(
            {name: (("time", "loc"), B[:, :, i], {"units": "nT"})
             for i, name in enumerate(("Bx", "By", "Bz"))},
            coords=coords)
        if var is not None:
            for i, name in enumerate(("Bx", "By", "Bz")):
                out[f"{name}_var"] = (("time", "loc"), var[:, :, i],
                                      {"units": "nT2"})
        return out

    def _predict_arrays(self, latlons, return_var):
        """``(B (ntimes, n, 3), var or None)`` at (n, 2) [lat, lon] points."""
        if self._secs is None:
            raise ValueError("Interpolator has not been fit yet; call .fit(b_obs)")
        pred_loc = np.column_stack([latlons, np.full(len(latlons), R_EARTH)])
        model = self._kalman if self._kalman is not None else self._secs
        if return_var:
            B, var = model.predict(pred_loc, return_var=True)
        else:
            B, var = model.predict(pred_loc), None
        # pysecs squeezes single-location/-time predictions; keep (ntimes, n, 3)
        B = B.reshape(-1, len(latlons), 3)
        if var is not None:
            var = var.reshape(-1, len(latlons), 3)
        return B, var

    def predict_grid(self, lats, lons, return_var=False):
        """Predict the magnetic field on a regular latitude/longitude grid.

        Convenience wrapper for map and grid products (live maps, movies,
        gridded data files): builds the meshgrid from the axis values,
        predicts at every grid point, and reshapes the result back to
        (time, latitude, longitude). :meth:`predict` is the general
        point-set path; both produce identical numbers at the same points.

        :param lats: 1-D array of grid latitudes in degrees (the grid-axis
            values, not a meshgridded array).
        :param lons: 1-D array of grid longitudes in degrees.
        :param return_var: also return the prediction variance, propagated
            from the posterior covariance of the fitted amplitudes (treating
            ``obs_std`` as 1-sigma errors of independent observations).
        :return: if the interpolator was fit with a Dataset, an xarray
            Dataset with ``Bx``/``By``/``Bz`` (time, latitude, longitude),
            1-D latitude/longitude coordinates, and
            ``Bx_var``/``By_var``/``Bz_var`` when ``return_var``; otherwise
            an ndarray (ntimes, nlat, nlon, 3 [Bx, By, Bz]) (a ``(B, var)``
            tuple when ``return_var``).
        """
        lats = np.atleast_1d(np.asarray(lats, dtype=float))
        lons = np.atleast_1d(np.asarray(lons, dtype=float))
        if lats.ndim != 1 or lons.ndim != 1:
            raise ValueError("lats and lons must be 1-D grid-axis arrays; "
                             "use predict() for general point sets")
        glat, glon = np.meshgrid(lats, lons, indexing="ij")
        latlons = np.column_stack([glat.ravel(), glon.ravel()])
        B, var = self._predict_arrays(latlons, return_var)
        shape = (-1, len(lats), len(lons), 3)
        B = B.reshape(shape)
        if var is not None:
            var = var.reshape(shape)

        if not self._is_xarray_fit:
            return (B, var) if return_var else B
        import xarray as xr
        dims = ("time", "latitude", "longitude")
        coords = {"time": self._times, "latitude": lats, "longitude": lons}
        out = xr.Dataset(
            {name: (dims, B[..., i], {"units": "nT"})
             for i, name in enumerate(("Bx", "By", "Bz"))},
            coords=coords)
        if var is not None:
            for i, name in enumerate(("Bx", "By", "Bz")):
                out[f"{name}_var"] = (dims, var[..., i], {"units": "nT2"})
        return out

    def loo_score(self, b_obs, obs_latlons=None, metric="rmse", **overrides):
        """Leave-one-out observatory skill score, for tuning the setup.

        Each observatory in turn is removed from the fit (via infinite
        variance, so the SEC grid stays identical) and predicted from the
        others; the score compares prediction with observation over the
        horizontal components at samples where data exist.

        :param metric: ``"rmse"`` (nT) or ``"mae"`` (nT).
        :param overrides: any constructor argument (``epsilon``, ``tau``,
            ``robust``, ``method``, ...) to evaluate instead of this
            interpolator's own -- loop over candidate values and pick the
            minimizer to make the choice data-driven -- plus any
            :meth:`fit` keyword (``gaps``, ``times``, ...).
        :return: dict mapping observatory (name if known, else index) to score
        """
        B, latlons, times, obs_names = _extract_b(b_obs, obs_latlons)
        config = self._config()
        fit_kwargs = {key: overrides.pop(key) for key in list(overrides)
                      if key not in config}
        config.update(overrides)
        trial = SECSInterpolator(**config)
        if times is not None:
            fit_kwargs.setdefault("times", times)
        reducer = {"rmse": lambda err: np.sqrt(np.nanmean(err**2)),
                   "mae": lambda err: np.nanmean(np.abs(err))}[metric]

        scores = {}
        for i in range(len(latlons)):
            std = np.ones(B.shape)
            std[:, i, :] = np.inf
            trial.fit(B, obs_latlons=latlons, obs_std=std, **fit_kwargs)
            predicted = trial.predict(latlons[i:i+1])[:, 0, :2]
            err = predicted - B[:, i, :2]
            name = obs_names[i] if obs_names is not None else i
            scores[name] = reducer(err)
        return scores
