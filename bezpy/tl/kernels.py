"""
Precomputed voltage functionals and power line telluric response functions.

This implements the runtime factorization described in ``ARCHITECTURE.md``:
every interpolation scheme in :meth:`TransmissionLine.calc_voltages` is a
*linear* map from per-site electric fields to a voltage, so the geometry can
be contracted out ahead of time.

* Level 1 -- "effective length vectors": for any of the linear schemes,

      V_l(t) = sum_s L_ls . E_s(t)

  where ``L`` is a static ``(nsites, 2)`` array per line (units km/1000, so
  V comes out in volts from E in mV/km). Build once, apply with one einsum.
  This reproduces ``calc_voltages`` exactly -- it is the same sum,
  reassociated -- but does no per-vertex work at runtime.

* Level 2 -- power line telluric response functions (Kelbert & Lucas, 2020,
  Eqs. 10-13): folding the site impedances into ``L`` gives per-line
  transfer functions

      V_l(omega) = tau^N_l(omega) Bx(omega) + tau^E_l(omega) By(omega)

      tau^N_l = sum_s (L_lsx Zxx_s + L_lsy Zyx_s)
      tau^E_l = sum_s (L_lsx Zxy_s + L_lsy Zyy_s)

  so line voltages come straight from the magnetic field spectra with one
  inverse FFT per line and no electric-field intermediate. ``tau`` has units
  of V/nT. Under the spatially uniform B assumption this is *identical* to
  computing E at the sites and interpolating it onto the line -- same linear
  operations, reassociated -- so the exact-mode kernel reproduces the current
  pipeline to rounding error.

Reference: Kelbert, A., & Lucas, G. M. (2020). Modified GIC estimation using
3-D Earth conductivity. Space Weather, 18, e2020SW002467.
https://doi.org/10.1029/2020SW002467
"""

__all__ = ["voltage_functional", "voltage_matrix", "apply_voltage_matrix",
           "TelluricResponse"]

import numpy as np
import scipy.fft
import scipy.interpolate
import scipy.sparse

# Cache tau on the FFT frequency grid between calc_voltages calls as long as
# it stays below this size; above it, recompute per call (chunked) instead.
_TAU_CACHE_MAX_BYTES = 256 * 2**20


def voltage_functional(line, n_inputs, how="delaunay"):
    """Effective length vector for one line.

    Returns ``(L, valid)`` where ``L`` has shape ``(n_inputs, 2)`` and
    ``V(t) = np.sum(E(t) * L)`` reproduces ``line.calc_voltages(E, how=how)``
    for ``E`` of shape ``(ntimes, n_inputs, 2)`` in mV/km, with ``V`` in volts.

    ``valid`` is False when the line uses points outside the Delaunay convex
    hull, in which case ``calc_voltages`` would return all-NaN voltages.

    :param line: a TransmissionLine with the weights for ``how`` already set
        (``set_nearest_sites``, ``set_1d_regions``, or ``set_delaunay_weights``)
    :param n_inputs: number of E-field input locations (sites or 1d regions)
    :param how: one of ``nn``, ``1d``, ``delaunay``
    """
    L = np.zeros((n_inputs, 2))
    valid = True

    if how == "nn":
        if line.nearest_sites is None:
            raise ValueError("Nearest sites not set; call .set_nearest_sites(site_xys)")
        np.add.at(L, line.nearest_sites[:-1], line.dl)

    elif how == "1d":
        if line.regions1d is None:
            raise ValueError("1d regions not set; call .set_1d_regions(region_polygons)")
        np.add.at(L, line.regions1d[:-1], line.dl)

    elif how == "delaunay":
        if line.delaunay_vtx is None or line.delaunay_wts is None:
            raise ValueError("Delaunay weights not set; call .set_delaunay_weights(site_xys)")
        # calc_voltages NaNs the whole line if any used point is out of the hull
        valid = not np.any(line.delaunay_wts[:-1] < 0)
        # V = sum_i sum_j w_ij (E_{vtx_ij} . dl_i)  ->  L[vtx_ij] += w_ij * dl_i
        contrib = line.delaunay_wts[:-1, :, np.newaxis] * line.dl[:, np.newaxis, :]
        np.add.at(L, line.delaunay_vtx[:-1].ravel(), contrib.reshape(-1, 2))

    else:
        raise ValueError("how must be a string of: nn, 1d, or delaunay")

    # Fold in the mV/km * km -> V conversion from calc_voltages
    return L / 1000., valid


def voltage_matrix(lines, n_inputs, how="delaunay"):
    """Stack :func:`voltage_functional` over lines.

    Returns ``(L, valid)`` with shapes ``(nlines, n_inputs, 2)`` and
    ``(nlines,)``. See :func:`apply_voltage_matrix` for usage.
    """
    L = np.empty((len(lines), n_inputs, 2))
    valid = np.empty(len(lines), dtype=bool)
    for i, line in enumerate(lines):
        L[i], valid[i] = voltage_functional(line, n_inputs, how=how)
    return L, valid


def apply_voltage_matrix(E, L, valid=None):
    """Voltages for all lines at once: ``V[t, l] = sum_se E[t, s, e] L[l, s, e]``.

    ``L`` is very sparse (each line touches only a handful of sites), so the
    contraction is done as a sparse matmul.

    :param E: electric fields, shape ``(ntimes, n_inputs, 2)``, mV/km
    :param L: from :func:`voltage_matrix`
    :param valid: optional boolean mask; invalid lines are set to NaN
    :return: voltages in volts, shape ``(ntimes, nlines)``
    """
    E = np.atleast_3d(E)
    ntimes = E.shape[0]
    L_sparse = scipy.sparse.csr_matrix(L.reshape(L.shape[0], -1))
    V = E.reshape(ntimes, -1) @ L_sparse.T
    if valid is not None:
        V[:, ~np.asarray(valid)] = np.nan
    return V


class TelluricResponse:
    """Power line telluric response functions tau^N, tau^E per line.

    Two modes, selected by ``periods``:

    * ``periods=None`` (exact mode): tau is evaluated on each run's FFT
      frequency grid directly from ``site.calcZ`` -- bitwise-equivalent
      physics to the current pipeline, useful for validation.
    * ``periods=array`` (master-grid mode): tau is built once at the given
      periods (e.g. the ~30 periods of the empirical MT transfer functions,
      as in Kelbert & Lucas, 2020) and spline-interpolated in log-period
      onto each run's frequency grid, exactly the way ``Site3d`` stores and
      interpolates Z today. This keeps memory bounded for long time series
      and makes the response functions a small, storable data product
      (see :meth:`to_netcdf`).

    Parameters
    ----------
    L : ndarray (nlines, nsites, 2)
        Effective length vectors from :func:`voltage_matrix`.
    sites : list of Site
        MT sites (must implement ``calcZ``), same order as the L site axis.
    valid : ndarray (nlines,) of bool, optional
        Lines flagged invalid produce NaN voltages.
    periods : ndarray, optional
        Master period grid in seconds; None for exact mode.
    extrapolate : {"zeros", "asymptotic"}
        Band-edge handling for frequencies outside the impedance (or master
        period) band. ``"zeros"`` is the historical hard band-stop;
        ``"asymptotic"`` continues tau with the half-space sqrt(f) asymptote
        anchored at the band edge (see ``Site3d.calcZ``), preserving the
        short-period content that a band-stop truncates.
    tau_master : ndarray (nlines, 2, nperiods), optional
        Precomputed master-grid tau (as restored by :meth:`from_netcdf`);
        when given, ``L``/``sites`` may be None and no construction happens.
    """

    def __init__(self, L, sites, valid=None, periods=None, extrapolate="zeros",
                 tau_master=None):
        if extrapolate not in ("zeros", "asymptotic"):
            raise ValueError("extrapolate must be one of: zeros, asymptotic")
        self.L = None if L is None else np.asarray(L)
        self.sites = sites
        self.valid = None if valid is None else np.asarray(valid, dtype=bool)
        self.periods = None if periods is None else np.sort(np.asarray(periods, float))
        self.extrapolate = extrapolate

        # tau on the most recent FFT frequency grid, keyed by (nfft, dt);
        # a historical scan over equal-length chunks reuses it thousands of
        # times, removing the per-chunk spline/calcZ cost.
        self._tau_grid_key = None
        self._tau_grid = None

        if tau_master is not None:
            # Restored product (from_netcdf): tau is already built.
            self.tau_master = np.asarray(tau_master)
        elif self.periods is not None:
            # Build tau once at the master periods (ascending period order)
            self.tau_master = self._build_tau(1. / self.periods)
        else:
            self.tau_master = None

    @classmethod
    def from_lines(cls, lines, sites, how="delaunay", periods=None,
                   extrapolate="zeros", resample_km=None):
        """Build from TransmissionLines and Sites, setting weights as needed.

        :param resample_km: if given, densify each line's geometry to at most
            this segment length (great-circle interpolation) before building
            the length vectors, refining the integration quadrature (Kelbert
            & Lucas, 2020, used 1 km). The input lines are not modified.
        """
        site_xys = np.array([[site.latitude, site.longitude] for site in sites])
        if resample_km is not None:
            lines = [line.resampled(resample_km) for line in lines]
        for line in lines:
            if how == "delaunay" and line.delaunay_vtx is None:
                line.set_delaunay_weights(site_xys)
            elif how == "nn" and line.nearest_sites is None:
                line.set_nearest_sites(site_xys)
        L, valid = voltage_matrix(lines, len(sites), how=how)
        return cls(L, sites, valid=valid, periods=periods, extrapolate=extrapolate)

    @property
    def nlines(self):
        """Number of lines."""
        return self.L.shape[0] if self.L is not None else self.tau_master.shape[0]

    def _calcZ(self, site, freqs):
        """site.calcZ with this response's band-edge handling when supported.

        Physical sites (e.g. ``Site1d``) compute Z at any frequency and take
        no ``extrapolate`` argument; band-limited empirical sites do.
        """
        try:
            return site.calcZ(freqs, extrapolate=self.extrapolate)
        except TypeError:
            return site.calcZ(freqs)

    def _build_tau(self, freqs, lines=slice(None)):
        """tau (nsel, 2, nfreq) accumulated site by site from calcZ."""
        L = self.L[lines]
        tau = np.zeros((L.shape[0], 2, len(freqs)), dtype=complex)
        for s, site in enumerate(self.sites):
            Lx, Ly = L[:, s, 0], L[:, s, 1]
            touched = np.nonzero((Lx != 0.) | (Ly != 0.))[0]
            if touched.size == 0:
                continue
            Z = self._calcZ(site, freqs)  # (xx, xy, yx, yy) x nfreq
            # V = Lx*Ex + Ly*Ey with Ex = Zxx Bx + Zxy By, Ey = Zyx Bx + Zyy By
            tau[touched, 0, :] += Lx[touched, None]*Z[0] + Ly[touched, None]*Z[2]
            tau[touched, 1, :] += Lx[touched, None]*Z[1] + Ly[touched, None]*Z[3]
        return tau

    def calc_tau(self, freqs, lines=slice(None)):
        """Evaluate tau^N, tau^E on the given frequencies.

        :return: complex array, shape ``(nselected_lines, 2, nfreqs)``, V/nT
        """
        if self.periods is None:
            return self._build_tau(freqs, lines)

        freqs = np.asarray(freqs, float)
        with np.errstate(divide="ignore"):
            x = np.log10(1. / freqs)
        # Interpolate in log-period, matching how Z itself is interpolated.
        # extrapolate=False -> NaN outside the master band (and at DC where
        # x is +inf).
        spline = scipy.interpolate.CubicSpline(np.log10(self.periods),
                                               self.tau_master[lines],
                                               axis=-1, extrapolate=False)
        tau = spline(x)

        if self.extrapolate == "asymptotic":
            # Continue tau past the master band with the same sqrt(f)
            # asymptote used for Z (tau is a linear combination of Z's).
            tau_edges = self.tau_master[lines]
            f_hi, f_lo = 1./self.periods[0], 1./self.periods[-1]
            high, low = freqs > f_hi, (freqs < f_lo) & (freqs > 0.)
            tau[..., high] = tau_edges[..., :1] * np.sqrt(freqs[high]/f_hi)
            tau[..., low] = tau_edges[..., -1:] * np.sqrt(freqs[low]/f_lo)

        # Zero whatever remains outside (DC always; the band edges for the
        # "zeros" hard band-stop, mirroring ext=1 for Z).
        return np.nan_to_num(tau, copy=False)

    def _tau_on_fft_grid(self, nfft, dt, freqs):
        """tau for all lines on the FFT grid, cached across equal-length chunks.

        Returns None when the full array would exceed the cache budget; the
        caller then evaluates tau chunkwise per call instead.
        """
        nbytes = self.nlines * 2 * len(freqs) * 16
        if nbytes > _TAU_CACHE_MAX_BYTES:
            return None
        if self._tau_grid_key != (nfft, dt):
            self._tau_grid = self.calc_tau(freqs)
            self._tau_grid_key = (nfft, dt)
        return self._tau_grid

    def calc_voltages(self, mag_x, mag_y, dt=60, line_chunk=256):
        """Line voltages directly from the magnetic field time series.

        Mirrors ``Site.convolve_fft`` (same padding and rfft conventions),
        but the frequency-domain multiply is by the per-line telluric
        response instead of the per-site impedance, so no electric field is
        ever formed. tau evaluated on the FFT grid is cached between calls
        with the same length and dt (up to a memory budget), so scanning a
        long record in equal-length chunks pays the tau construction once.

        :param mag_x: northward magnetic field, nT
        :param mag_y: eastward magnetic field, nT
        :param dt: sample interval in seconds
        :param line_chunk: number of lines processed per block, to bound the
            ``(chunk, nfreq)`` complex workspace for long time series
        :return: voltages in volts, shape ``(ntimes, nlines)``
        """
        N0 = len(mag_x)
        # Same padding rule as Site.convolve_fft (prevents circular convolution)
        N = 2**(int(np.log2(N0))+2)
        freqs = scipy.fft.rfftfreq(N, d=dt)

        mag_x_fft = scipy.fft.rfft(mag_x, n=N, workers=-1)
        mag_y_fft = scipy.fft.rfft(mag_y, n=N, workers=-1)

        tau_grid = self._tau_on_fft_grid(N, dt, freqs)
        voltages = np.empty((N0, self.nlines))
        for i0 in range(0, self.nlines, line_chunk):
            sel = slice(i0, min(i0 + line_chunk, self.nlines))
            tau = tau_grid[sel] if tau_grid is not None else self.calc_tau(freqs, lines=sel)
            V_fft = tau[:, 0, :]*mag_x_fft + tau[:, 1, :]*mag_y_fft
            voltages[:, sel] = scipy.fft.irfft(V_fft, axis=-1, workers=-1)[:, :N0].T

        if self.valid is not None:
            voltages[:, ~self.valid] = np.nan
        return voltages

    def to_netcdf(self, path):
        """Save master-grid tau as a small, self-contained netCDF product.

        Stores the response functions themselves (tau at the master periods,
        with validity flags and band-edge policy) -- everything needed to
        compute voltages from B, with no line geometry or MT sites required
        on the other end. Requires master-grid mode and xarray (install the
        ``bezpy[data]`` extra).
        """
        if self.tau_master is None:
            raise ValueError(
                "Only master-grid responses are storable (pass periods= when "
                "building); exact mode is tied to a run's FFT grid.")
        import xarray as xr
        valid = np.ones(self.nlines, dtype=bool) if self.valid is None else self.valid
        # Complex data split into real/imag so any netCDF engine can store it
        ds = xr.Dataset(
            {"tau_real": (("line", "component", "period"), self.tau_master.real),
             "tau_imag": (("line", "component", "period"), self.tau_master.imag),
             "valid": (("line",), valid)},
            coords={"period": ("period", self.periods, {"units": "s"}),
                    "component": ["N", "E"]},
            attrs={"title": "Power line telluric response functions",
                   "units": "V/nT",
                   "extrapolate": self.extrapolate,
                   "reference": "Kelbert & Lucas (2020), doi:10.1029/2020SW002467"})
        ds.to_netcdf(path)

    @classmethod
    def from_netcdf(cls, path):
        """Load a response saved by :meth:`to_netcdf`.

        The result computes tau and voltages directly from the stored master
        grid; it carries no line geometry or MT sites.
        """
        import xarray as xr
        with xr.open_dataset(path) as ds:
            ds.load()
        tau = ds["tau_real"].values + 1j*ds["tau_imag"].values
        return cls(L=None, sites=None, valid=ds["valid"].values,
                   periods=ds["period"].values, extrapolate=ds.attrs["extrapolate"],
                   tau_master=tau)
