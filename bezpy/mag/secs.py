"""SECS interpolation of magnetic observatory data.

Wraps `pysecs <https://github.com/greglucas/pysecs>`_ into a fit/predict
interpolator with the configuration used across the published geoelectric
hazard studies (divergence-free SEC shell at 110 km, epsilon = 0.05, Bz
excluded from the fit) as the baseline, and gap handling that keeps every
sample: missing data are masked with infinite variance instead of being
interpolated over or causing whole observatories to be discarded.
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
        observatories padded by ``pad`` degrees; a 4-tuple gives the grid
        bounding box explicitly; an array gives (lat, lon) poles directly.
    spacing : float
        Grid spacing in degrees for constructed grids (default 2).
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
    method : {"lstsq"}
        Fitting method. ``"kalman"`` is reserved for pysecs' Kalman
        filter once released.

    Examples
    --------
    >>> interp = SECSInterpolator(grid="auto")
    >>> interp.fit(b_obs)                          # doctest: +SKIP
    >>> b_pred = interp.predict(latlons)           # doctest: +SKIP
    """

    def __init__(self, grid="auto", spacing=2.0, pad=5.0, altitude_km=110.,
                 epsilon=0.05, fit_bz=False, method="lstsq"):
        if method == "kalman":
            raise NotImplementedError(
                "method='kalman' requires a pysecs release with KalmanSECS; "
                "use method='lstsq'")
        if method != "lstsq":
            raise ValueError("method must be one of: lstsq, kalman")
        self.grid = grid
        self.spacing = spacing
        self.pad = pad
        self.altitude_km = altitude_km
        self.epsilon = epsilon
        self.fit_bz = fit_bz
        self.method = method

        self._secs = None
        self._times = None
        self._is_xarray_fit = False

    def _grid_latlonr(self, obs_latlons):
        """SEC pole locations (npoles, 3 [lat, lon, r]) for the fit."""
        if isinstance(self.grid, str) and self.grid == "auto":
            south, north = obs_latlons[:, 0].min() - self.pad, obs_latlons[:, 0].max() + self.pad
            west, east = obs_latlons[:, 1].min() - self.pad, obs_latlons[:, 1].max() + self.pad
        elif np.ndim(self.grid) == 1 and len(self.grid) == 4:
            west, south, east, north = self.grid
        else:
            poles = np.atleast_2d(np.asarray(self.grid, dtype=float))
            return np.column_stack([poles, np.full(len(poles),
                                                   R_EARTH + self.altitude_km*1e3)])
        south, north = max(south, -90.), min(north, 90.)
        lats = np.arange(south, north + self.spacing/2, self.spacing)
        lons = np.arange(west, east + self.spacing/2, self.spacing)
        lat, lon = map(np.ravel, np.meshgrid(lats, lons, indexing="ij"))
        return np.column_stack([lat, lon,
                                np.full(lat.size, R_EARTH + self.altitude_km*1e3)])

    def fit(self, b_obs, obs_latlons=None, obs_std=None, gaps="mask",
            weight_gaps=False, gap_inflation=10., gap_window=60):
        """Fit the SEC amplitudes to observed magnetic fields.

        :param b_obs: xarray Dataset with ``Bx``/``By``[/``Bz``] (or
            ``X``/``Y``/``Z``) variables dimensioned (time, obs) and
            latitude/longitude coordinates, or ndarray (ntimes, nobs, 2 or 3)
            in nT with ``obs_latlons`` given.
        :param obs_latlons: (nobs, 2) [latitude, longitude], required for
            bare-array input.
        :param obs_std: optional standard error, broadcastable to
            (ntimes, nobs, 3), used to weight the observations.
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
        B, latlons, self._times, obs_names = _extract_b(b_obs, obs_latlons)
        self._is_xarray_fit = self._times is not None

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

        # pysecs recomputes the (expensive) SVD whenever consecutive time
        # steps have different variance patterns. Gappy data alternates
        # patterns constantly, so fit in pattern-sorted time order (one SVD
        # per unique pattern) and un-permute the amplitudes afterwards.
        order = np.lexsort(std.reshape(len(B), -1).T)
        self._secs.fit(obs_loc=obs_loc, obs_B=B[order], obs_std=std[order],
                       epsilon=self.epsilon)
        inverse = np.argsort(order)
        self._secs.sec_amps = self._secs.sec_amps[inverse]
        self._secs.sec_amps_var = self._secs.sec_amps_var[inverse]
        return self

    def predict(self, locations):
        """Predict the magnetic field at new locations.

        :param locations: (n, 2) [latitude, longitude] array, a sequence of
            objects with ``.latitude``/``.longitude`` (e.g. MT sites), or an
            xarray object with latitude/longitude coordinates.
        :return: if the interpolator was fit with a Dataset, an xarray
            Dataset with ``Bx``/``By``/``Bz`` (time, loc) and location
            coordinates; otherwise an ndarray (ntimes, n, 3 [Bx, By, Bz]).
        """
        if self._secs is None:
            raise ValueError("Interpolator has not been fit yet; call .fit(b_obs)")
        latlons = _extract_latlons(locations)
        pred_loc = np.column_stack([latlons, np.full(len(latlons), R_EARTH)])
        B = self._secs.predict(pred_loc)
        # pysecs squeezes single-location predictions; keep (ntimes, n, 3)
        B = B.reshape(B.shape[0], len(latlons), 3)

        if not self._is_xarray_fit:
            return B
        import xarray as xr
        coords = {"time": self._times,
                  "latitude": ("loc", latlons[:, 0]),
                  "longitude": ("loc", latlons[:, 1])}
        return xr.Dataset(
            {name: (("time", "loc"), B[:, :, i], {"units": "nT"})
             for i, name in enumerate(("Bx", "By", "Bz"))},
            coords=coords)

    def loo_score(self, b_obs, obs_latlons=None, epsilon=None, metric="rmse",
                  **fit_kwargs):
        """Leave-one-out observatory skill score, for tuning ``epsilon``.

        Each observatory in turn is removed from the fit (via infinite
        variance, so the SEC grid stays identical) and predicted from the
        others; the score compares prediction with observation over the
        horizontal components at samples where data exist.

        :param epsilon: regularization to evaluate (default: the
            interpolator's own). Loop over candidate values and pick the
            minimizer to make the choice data-driven.
        :param metric: ``"rmse"`` (nT) or ``"mae"`` (nT).
        :return: dict mapping observatory (name if known, else index) to score
        """
        B, latlons, _, obs_names = _extract_b(b_obs, obs_latlons)
        trial = SECSInterpolator(grid=self.grid, spacing=self.spacing,
                                 pad=self.pad, altitude_km=self.altitude_km,
                                 epsilon=self.epsilon if epsilon is None else epsilon,
                                 fit_bz=self.fit_bz, method=self.method)
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
