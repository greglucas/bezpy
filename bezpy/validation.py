"""Validation: the apparatus that decides method choices with evidence.

Two tools:

* :func:`leave_one_out` -- withhold each observatory from the SECS fit and
  predict it from the others; the every-paper validation loop as a utility
  with defined skill scores (RMSE, MAE, correlation, and coherence per
  period band).
* :func:`synthetic_scenario` and its scoring helpers -- a known, smoothly
  varying conductivity structure under a spatially uniform source, where
  the true E at any point and the true V along any line are computable
  exactly. Every method option (``how=``, ``resample_km=``,
  ``extrapolate=``) is scored against that truth instead of tradition;
  ``VALIDATION.md`` records the current results.
"""

__all__ = ["leave_one_out", "synthetic_scenario",
           "score_interpolation_methods", "score_extrapolation",
           "score_fitting_methods", "score_ensemble_calibration",
           "score_analytic_calibration"]

import numpy as np
import pandas as pd
import scipy.fft

from .mag.secs import SECSInterpolator, _extract_b
from .mt.site import Site1d, Site3d

# ---------------------------------------------------------------------
# Leave-one-out observatory validation
# ---------------------------------------------------------------------

def _coherence_bands(observed, predicted, dt, bands):
    """Mean magnitude-squared coherence in each period band (seconds)."""
    from scipy.signal import coherence as msc
    # Welch coherence needs several averaged segments to be meaningful: with a
    # single segment it is identically 1 regardless of the signals. Cap the
    # segment length so even short windows average >= ~8 segments (with the
    # default 50% overlap), while keeping enough length to resolve the longest
    # requested band.
    nperseg = min(len(observed) // 4, 4096)
    nperseg = max(nperseg, 16)
    freqs, coh = msc(observed, predicted, fs=1./dt, nperseg=nperseg)
    out = []
    with np.errstate(divide="ignore"):
        periods = 1./freqs
    for t_lo, t_hi in bands:
        mask = (periods >= t_lo) & (periods <= t_hi)
        out.append(np.nanmean(coh[mask]) if mask.any() else np.nan)
    return out


def leave_one_out(b_obs, interp=None, metrics=("rmse", "mae", "correlation"),
                  bands=None, dt=None, **fit_kwargs):
    """Leave-one-out observatory skill for a SECS interpolation.

    Each observatory in turn is withheld from the fit (via infinite
    variance, so the SEC grid is identical across folds) and predicted from
    the others; predictions are scored against the withheld observations
    over the horizontal components.

    :param b_obs: ``(time, obs)`` Dataset (``Bx``/``By``[/``Bz``] or
        ``X``/``Y``/``Z``) or bare array with ``obs_latlons`` in
        ``fit_kwargs``
    :param interp: a configured :class:`bezpy.mag.SECSInterpolator`
        (default: baseline configuration)
    :param metrics: any of ``"rmse"``, ``"mae"``, ``"correlation"`` (all in
        nT except correlation)
    :param bands: optional list of ``(t_min, t_max)`` period bands in
        seconds; adds mean magnitude-squared coherence per band (needs
        ``dt`` for bare-array input)
    :param dt: sample interval in seconds (read from the time axis for
        Dataset input)
    :return: DataFrame indexed by observatory with one column per metric
        (coherence columns are named ``coherence_<t_min>-<t_max>s``)
    """
    interp = interp if interp is not None else SECSInterpolator()
    B, latlons, times, obs_names = _extract_b(b_obs,
                                              fit_kwargs.pop("obs_latlons", None))
    if bands is not None and dt is None:
        if times is None:
            raise ValueError("bands= needs dt= with bare-array input")
        dt = float(np.diff(times[:2]).astype("timedelta64[s]")[0]
                   / np.timedelta64(1, "s"))

    reducers = {
        "rmse": lambda o, p: np.sqrt(np.nanmean((p - o)**2)),
        "mae": lambda o, p: np.nanmean(np.abs(p - o)),
        "correlation": lambda o, p: np.corrcoef(
            o[np.isfinite(o)], p[np.isfinite(o)])[0, 1],
    }
    rows = []
    names = []
    trial = SECSInterpolator(**interp._config())
    if times is not None:
        fit_kwargs.setdefault("times", times)
    for i in range(len(latlons)):
        std = np.ones(B.shape)
        std[:, i, :] = np.inf
        trial.fit(B, obs_latlons=latlons, obs_std=std, **fit_kwargs)
        predicted = trial.predict(latlons[i:i+1])[:, 0, :2]
        observed = B[:, i, :2]

        row = {}
        # Score over both horizontal components together
        flat_o, flat_p = observed.ravel(), predicted.ravel()
        for metric in metrics:
            row[metric] = reducers[metric](flat_o, flat_p)
        if bands is not None:
            # Coherence per physical component, then averaged -- the two
            # components are separate time series and must not be mixed
            # into one. Missing samples are spliced out, which biases the
            # spectral estimate near gaps; acceptable for a skill score.
            good = np.isfinite(observed).all(axis=1)
            for k, band in enumerate(bands):
                coh = np.nanmean([
                    _coherence_bands(observed[good, c], predicted[good, c],
                                     dt, [band])[0]
                    for c in range(observed.shape[1])])
                row[f"coherence_{band[0]:g}-{band[1]:g}s"] = coh
        rows.append(row)
        names.append(obs_names[i] if obs_names is not None else i)
    return pd.DataFrame(rows, index=pd.Index(names, name="obs"))


# ---------------------------------------------------------------------
# Synthetic truth benchmark (D6 / D7 arbiter)
# ---------------------------------------------------------------------

class SyntheticScenario:
    """A conductivity structure + source where truth is computable exactly.

    A two-layer conductivity structure varies smoothly with position, the
    source is spatially uniform and band-limited, so the true E anywhere is
    an exact frequency-domain product and the true V along any line is a
    dense local quadrature of that field. Layering matters: it makes the
    impedance deviate from the sqrt(f) half-space asymptote, so band-edge
    extrapolation options are tested fairly.
    """

    def __init__(self, seed=0, n_lat=5, n_lon=5, ntimes=4096, dt=60.):
        self.dt = dt
        rng = np.random.default_rng(seed)
        # Smooth log-resistivity field over the domain
        self._phase = rng.uniform(0, 2*np.pi, 4)

        lats = np.linspace(36, 44, n_lat)
        lons = np.linspace(-108, -98, n_lon)
        self.sites = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                site = Site1d(f"S{i}{j}", *self.profile(lat, lon))
                site.latitude, site.longitude = lat, lon
                self.sites.append(site)

        # Band-limited random source: red-noise spectrum between 2 min and 8 h
        freqs = scipy.fft.rfftfreq(ntimes, d=dt)
        band = (freqs > 1/28800.) & (freqs < 1/120.)
        spectrum = np.zeros(len(freqs), dtype=complex)
        with np.errstate(divide="ignore"):
            spectrum[band] = ((freqs[band]/freqs[band][0])**-1.
                              * np.exp(2j*np.pi*rng.random(band.sum())))
        b = scipy.fft.irfft(spectrum, n=ntimes)
        scale = 200./np.max(np.abs(b))  # ~200 nT excursions
        self.mag_x = b*scale
        self.mag_y = np.roll(b, ntimes//3)*scale
        self.ntimes = ntimes

    def profile(self, lat, lon):
        """The known structure: ``(thicknesses, resistivities)`` at a point.

        A smoothly varying ~20 km surface layer over a basement with a
        smoothly varying resistivity contrast.
        """
        u = (lat - 40.)/8.
        v = (lon + 103.)/10.
        p = self._phase
        rho_surface = 10.**(2. + 0.7*np.sin(2*np.pi*u + p[0])
                            * np.cos(2*np.pi*v + p[1])
                            + 0.3*np.sin(4*np.pi*v + p[2]))
        contrast = 10.**(0.8*np.sin(2*np.pi*u + p[3]))
        thickness = 20e3*(1. + 0.5*np.cos(2*np.pi*v + p[0]))
        return [thickness, 1e5], [rho_surface, rho_surface*contrast]

    def true_e(self, lats, lons):
        """Exact E (mV/km) at points: the local layered Z in the full band.

        :return: array (ntimes, npoints, 2 [Ex, Ey])
        """
        lats, lons = np.atleast_1d(lats), np.atleast_1d(lons)
        N = 2**(int(np.log2(self.ntimes)) + 2)
        freqs = scipy.fft.rfftfreq(N, d=self.dt)
        bx = scipy.fft.rfft(self.mag_x, n=N)
        by = scipy.fft.rfft(self.mag_y, n=N)
        out = np.empty((self.ntimes, len(lats), 2))
        for k, (lat, lon) in enumerate(zip(lats, lons)):
            Z = Site1d("pt", *self.profile(lat, lon)).calcZ(freqs)
            out[:, k, 0] = scipy.fft.irfft(Z[1]*by)[:self.ntimes]
            out[:, k, 1] = scipy.fft.irfft(Z[2]*bx)[:self.ntimes]
        return out

    def true_voltages(self, line, quadrature_km=1.0):
        """Exact V(t) in volts: dense local quadrature of the true field."""
        dense = line.resampled(quadrature_km)
        e = self.true_e(dense.pts[:-1, 1], dense.pts[:-1, 0])
        return np.sum(e*dense.dl[np.newaxis, :, :], axis=(1, 2))/1000.

    def site_e(self, **kwargs):
        """E at the MT sites through the production path (SiteCollection)."""
        from .mt.site import SiteCollection
        ex, ey = SiteCollection(self.sites).convolve_fft(
            self.mag_x, self.mag_y, dt=self.dt, **kwargs)
        return np.stack([ex.T, ey.T], axis=-1)


def synthetic_scenario(**kwargs):
    """Create a :class:`SyntheticScenario` (see class docs)."""
    return SyntheticScenario(**kwargs)


def _relative_rms(predicted, truth):
    return float(np.sqrt(np.nanmean((predicted - truth)**2))
                 / np.sqrt(np.nanmean(truth**2)))


def score_interpolation_methods(scenario, lines, how=("nn", "delaunay"),
                                resample_km=(None, 1.0)):
    """Score line-voltage interpolation options against synthetic truth (D7).

    :return: DataFrame indexed by (how, resample_km) with the relative RMS
        voltage error against the exact quadrature, aggregated over lines
    """
    site_xys = np.array([[s.latitude, s.longitude] for s in scenario.sites])
    E = scenario.site_e()
    truth = np.column_stack([scenario.true_voltages(line) for line in lines])

    rows = []
    for method in how:
        for spacing in resample_km:
            used = [line if spacing is None else line.resampled(spacing)
                    for line in lines]
            for line in used:
                if method == "delaunay":
                    line.set_delaunay_weights(site_xys)
                else:
                    line.set_nearest_sites(site_xys)
            v = np.column_stack([line.calc_voltages(E, how=method)
                                 for line in used])
            rows.append({"how": method, "resample_km": spacing,
                         "rel_rms_error": _relative_rms(v, truth)})
    return pd.DataFrame(rows).set_index(["how", "resample_km"])


def score_fitting_methods(seed=0, ntimes=240, dt=60., noise=1.5,
                          tau=600., gate_sigma=4.):
    """Score SECS fitting methods against a synthetic SEC source (3.1).

    A smoothly time-varying SEC source generates the true field at a 6x6
    observatory network and at held-out locations. The observations are
    corrupted three ways -- white noise everywhere, a station dropout, and
    a single-station baseline jump -- and each fitting method is scored on
    the RMSE of its held-out predictions against the noise-free truth.

    :param noise: white-noise standard deviation in nT (the synthetic field
        scale is a few nT, so the default is a substantial ~40%)
    :param tau: Kalman temporal correlation time in seconds
    :param gate_sigma: innovation gate for the gated Kalman row
    :return: DataFrame indexed by method with ``rmse_nT``
    """
    from .mag.secs import R_EARTH, _require_pysecs
    pysecs = _require_pysecs()
    rng = np.random.default_rng(seed)

    obs_latlons = np.array([[lat, lon]
                            for lat in np.linspace(34, 46, 6)
                            for lon in np.linspace(-110, -96, 6)])
    test_latlons = np.array([[38.7, -104.1], [41.3, -101.5], [39.9, -106.2],
                             [43.5, -99.0], [36.2, -108.5]])
    poles = np.array([[lat, lon, R_EARTH + 110e3]
                      for lat in np.linspace(30, 50, 4)
                      for lon in np.linspace(-115, -90, 5)])
    source = pysecs.SECS(sec_df_loc=poles)
    t = np.linspace(0, 6*np.pi, ntimes)[:, None]
    source.sec_amps = (1e13 * np.sin(t + rng.uniform(0, 2*np.pi, len(poles)))
                       * rng.uniform(0.5, 1.5, len(poles)))

    def truth(latlons):
        return source.predict(
            np.column_stack([latlons, np.full(len(latlons), R_EARTH)]))

    b_obs = truth(obs_latlons) + rng.normal(0., noise, (ntimes, len(obs_latlons), 3))
    b_obs[ntimes//3:ntimes//3 + 20, 7, :] = np.nan          # station dropout
    b_obs[ntimes//2:, 20, :2] += 8.*noise                   # baseline jump
    b_true = truth(test_latlons)
    times = np.arange(ntimes)*dt

    configs = {
        "lstsq": {},
        "lstsq robust=huber": {"robust": "huber"},
        "lstsq robust=bisquare": {"robust": "bisquare"},
        "kalman": {"method": "kalman", "tau": tau},
        "kalman gated": {"method": "kalman", "tau": tau,
                         "gate_sigma": gate_sigma},
    }
    rows = []
    for name, kwargs in configs.items():
        interp = SECSInterpolator(**kwargs)
        interp.fit(b_obs, obs_latlons=obs_latlons, times=times)
        predicted = interp.predict(test_latlons)[:, :, :2]
        rows.append({"method": name,
                     "rmse_nT": float(np.sqrt(np.mean(
                         (predicted - b_true[:, :, :2])**2)))})
    return pd.DataFrame(rows).set_index("method")


def score_extrapolation(scenario, periods=None):
    """Score impedance band-edge options against synthetic truth (D6).

    Builds band-limited empirical sites (``Site3d``) by sampling the true
    impedances at ``periods``, runs the E-field path under each
    ``extrapolate=`` option, and scores against the full-band truth.

    :param periods: sampling periods in seconds (default 31 log-spaced over
        the EMTF band 7--30,000 s, truncating the scenario's source band)
    :return: DataFrame indexed by option with relative RMS error of E at
        the sites
    """
    from .mt.site import SiteCollection
    if periods is None:
        periods = np.logspace(np.log10(300.), np.log10(7200.), 31)
    truth = scenario.site_e()

    banded = []
    for site in scenario.sites:
        empirical = Site3d(site.name)
        empirical.latitude, empirical.longitude = site.latitude, site.longitude
        empirical.periods = np.asarray(periods, dtype=float)
        empirical.Z = site.calcZ(1./empirical.periods)
        empirical.Z_var = None
        banded.append(empirical)

    rows = []
    for option in ("zeros", "asymptotic"):
        ex, ey = SiteCollection(banded).convolve_fft(
            scenario.mag_x, scenario.mag_y, dt=scenario.dt, extrapolate=option)
        predicted = np.stack([ex.T, ey.T], axis=-1)
        rows.append({"extrapolate": option,
                     "rel_rms_error": _relative_rms(predicted, truth)})
    return pd.DataFrame(rows).set_index("extrapolate")


def _calibration_scenario(n_trials, ntimes, dt, relative_std, periods, rng):
    """Shared truth-vs-measured setup for :func:`score_ensemble_calibration`
    and :func:`score_analytic_calibration`: one true layered impedance
    (exact, from a known ``Site1d`` profile) corrupted with independent
    complex Gaussian noise -- the noise model either propagation method
    itself draws from -- into ``n_trials`` "measured" ``Site3d`` sites
    sharing one band-limited source field.

    :return: ``(b_obs, measured, truth)`` -- ``measured``/``truth`` are
        lists of ``n_trials`` ``Site3d`` objects (measured carries the
        noisy ``Z``/reported ``Z_var``; truth carries the exact ``Z``)
    """
    if periods is None:
        periods = np.logspace(np.log10(7.), np.log10(30000.), 31)

    true_site = Site1d("truth", [20e3, 1e5], [30., 300.])
    freqs = 1./periods
    Z_true = true_site.calcZ(freqs)
    # Off-diagonal-only truth (Site1d) has exact zeros on xx/yy; a tiny
    # variance floor keeps those components' spline fit well-posed (a zero
    # Z_var period is dropped entirely -- see Site3d.spline_interp) without
    # perturbing them meaningfully.
    Z_var = (relative_std * np.abs(Z_true))**2 + 1e-6

    noise = (rng.normal(size=(n_trials, 4, len(periods)))
            + 1j*rng.normal(size=(n_trials, 4, len(periods))))
    Z_measured = Z_true[np.newaxis] + noise * np.sqrt(Z_var/2.)[np.newaxis]

    measured, truth = [], []
    for i in range(n_trials):
        lat, lon = 40. + 0.01*i, -100.
        measured.append(Site3d(f"S{i}", latitude=lat, longitude=lon,
                               periods=periods, Z=Z_measured[i], Z_var=Z_var))
        truth.append(Site3d(f"S{i}", latitude=lat, longitude=lon,
                            periods=periods, Z=Z_true, Z_var=None))

    freqs_fft = scipy.fft.rfftfreq(ntimes, d=dt)
    band = (freqs_fft > 1/28800.) & (freqs_fft < 1/120.)
    spectrum = np.zeros(len(freqs_fft), dtype=complex)
    with np.errstate(divide="ignore"):
        spectrum[band] = ((freqs_fft[band]/freqs_fft[band][0])**-1.
                          * np.exp(2j*np.pi*rng.random(band.sum())))
    b = scipy.fft.irfft(spectrum, n=ntimes)
    scale = 200./np.max(np.abs(b))
    mag_x, mag_y = b*scale, np.roll(b, ntimes//3)*scale

    import xarray as xr
    times = pd.date_range("2024-01-01", periods=ntimes,
                          freq=pd.Timedelta(seconds=dt))
    b_obs = xr.Dataset(
        {"Bx": (("time", "obs"), mag_x[:, np.newaxis]),
         "By": (("time", "obs"), mag_y[:, np.newaxis])},
        coords={"time": times, "obs": ["OB0"],
                "latitude": ("obs", [40.]), "longitude": ("obs", [-100.])})
    return b_obs, measured, truth


def score_ensemble_calibration(n_trials=30, n_realizations=150, ntimes=2048,
                               dt=60., relative_std=0.05, periods=None, seed=0):
    """Coverage of ``calc_geoelectric_field``'s ``n_realizations`` ensemble
    against known truth (roadmap 1.2 phase 1 acceptance test).

    Simulates the actual estimation process the ensemble represents: one
    true layered impedance (exact, from a known ``Site1d`` profile) is
    corrupted with independent complex Gaussian noise to produce the
    "measured" ``Z``/``Z_var`` an MT survey would report for ``n_trials``
    synthetic sites -- the same noise model :meth:`Site3d.perturbed` itself
    draws from -- sharing one band-limited source field. A well-calibrated
    ensemble should cover the (otherwise unknown) truth within its +-1
    standard deviation about 68% of the time, pooled over every site/time
    sample.

    Note: ``Site3d.calcZ`` fits a weighted smoothing spline through the
    (noisy) measured impedance (``smooth_factor=None`` when ``Z_var`` is
    given -- see ``Site3d.spline_interp``), which damps propagated noise
    somewhat; observed coverage runs a bit under the nominal 68% (roughly
    0.6-0.65 across period counts and noise levels) as a result. This is a
    property of reusing the *production* impedance-interpolation path (the
    same one every real analysis goes through), not a test artifact -- see
    ``VALIDATION.md`` for the measured values. Exact analytic propagation
    (roadmap 1.2 phase 2) is the intended fix.

    :param n_trials: independent noisy-impedance draws (pooled samples)
    :param n_realizations: ensemble draws per trial
    :param ntimes: length of the shared band-limited source signal
    :param dt: sample interval in seconds
    :param relative_std: reported measurement std as a fraction of |Z_true|
        (sets ``Z_var``)
    :param periods: sampling periods in seconds (default 31 log-spaced over
        the EMTF band 7--30,000 s, as in :func:`score_extrapolation`)
    :param seed: RNG seed
    :return: dict with ``coverage_1sigma`` (fraction of true samples inside
        the ensemble's +-1 std band) and ``n_samples`` (site x time pairs
        pooled, across both E components)
    """
    from .pipeline import calc_geoelectric_field
    rng = np.random.default_rng(seed)
    b_obs, measured, truth = _calibration_scenario(
        n_trials, ntimes, dt, relative_std, periods, rng)

    e_ensemble = calc_geoelectric_field(b_obs, measured, interp="uniform",
                                        n_realizations=n_realizations,
                                        seed=seed + 1, pad="2h")
    e_truth = calc_geoelectric_field(b_obs, truth, interp="uniform", pad="2h")

    covered = []
    for comp in ("Ex", "Ey"):
        mean = e_ensemble[comp].values.mean(axis=0)
        std = e_ensemble[comp].values.std(axis=0)
        covered.append(np.abs(e_truth[comp].values - mean) <= std)
    covered = np.concatenate([c.ravel() for c in covered])
    return {"coverage_1sigma": float(covered.mean()), "n_samples": int(covered.size)}


def score_analytic_calibration(n_trials=15, ntimes=1024, dt=60.,
                               relative_std=0.05, periods=None, seed=0):
    """Coverage of ``calc_geoelectric_field(..., return_var=True)``'s
    analytic +-1 std band against known truth (roadmap 1.2 phase 2
    acceptance test) -- the deterministic counterpart of
    :func:`score_ensemble_calibration`, same scenario (see
    :func:`_calibration_scenario`), so the two numbers are directly
    comparable in ``VALIDATION.md``.

    Uses the point estimate's own ``Ex_var``/``Ey_var`` (no Monte Carlo
    draws) as the +-1 std band and checks how often the known truth falls
    inside it, pooled over every site/time sample.

    :param n_trials: independent noisy-impedance draws (pooled samples).
        Analytic propagation costs roughly one extra FFT convolution per
        valid (period, component) measurement per site (see
        ``Site3d.z_sensitivities``), so this defaults much smaller than
        ``score_ensemble_calibration``'s ``n_trials``.
    :param ntimes: length of the shared band-limited source signal
    :param dt: sample interval in seconds
    :param relative_std: reported measurement std as a fraction of |Z_true|
    :param periods: sampling periods in seconds (default 31 log-spaced over
        the EMTF band 7--30,000 s)
    :param seed: RNG seed
    :return: dict with ``coverage_1sigma`` and ``n_samples``, same shape as
        :func:`score_ensemble_calibration`'s return
    """
    from .pipeline import calc_geoelectric_field
    rng = np.random.default_rng(seed)
    b_obs, measured, truth = _calibration_scenario(
        n_trials, ntimes, dt, relative_std, periods, rng)

    e_analytic = calc_geoelectric_field(b_obs, measured, interp="uniform",
                                        return_var=True, pad="2h")
    e_truth = calc_geoelectric_field(b_obs, truth, interp="uniform", pad="2h")

    covered = []
    for comp in ("Ex", "Ey"):
        std = np.sqrt(e_analytic[f"{comp}_var"].values)
        covered.append(np.abs(e_truth[comp].values - e_analytic[comp].values) <= std)
    covered = np.concatenate([c.ravel() for c in covered])
    return {"coverage_1sigma": float(covered.mean()), "n_samples": int(covered.size)}
