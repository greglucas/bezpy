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
import scipy.interpolate
import scipy.sparse


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
      and makes the response functions a small, storable data product.
      Frequencies outside the master period band get tau = 0 (mirroring the
      ``ext=1`` band-stop of ``Site3d.calcZ``).

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
    """

    def __init__(self, L, sites, valid=None, periods=None):
        self.L = np.asarray(L)
        self.sites = sites
        self.valid = None if valid is None else np.asarray(valid, dtype=bool)
        self.periods = None if periods is None else np.sort(np.asarray(periods, float))

        self.tau_master = None
        if self.periods is not None:
            # Build tau once at the master periods (ascending period order)
            self.tau_master = self._build_tau(1. / self.periods)

    @classmethod
    def from_lines(cls, lines, sites, how="delaunay", periods=None):
        """Build from TransmissionLines and Sites, setting weights as needed."""
        site_xys = np.array([[site.latitude, site.longitude] for site in sites])
        for line in lines:
            if how == "delaunay" and line.delaunay_vtx is None:
                line.set_delaunay_weights(site_xys)
            elif how == "nn" and line.nearest_sites is None:
                line.set_nearest_sites(site_xys)
        L, valid = voltage_matrix(lines, len(sites), how=how)
        return cls(L, sites, valid=valid, periods=periods)

    @property
    def nlines(self):
        """Number of lines."""
        return self.L.shape[0]

    def _build_tau(self, freqs, lines=slice(None)):
        """tau (nsel, 2, nfreq) accumulated site by site from calcZ."""
        L = self.L[lines]
        tau = np.zeros((L.shape[0], 2, len(freqs)), dtype=complex)
        for s, site in enumerate(self.sites):
            Lx, Ly = L[:, s, 0], L[:, s, 1]
            touched = np.nonzero((Lx != 0.) | (Ly != 0.))[0]
            if touched.size == 0:
                continue
            Z = site.calcZ(freqs)  # (xx, xy, yx, yy) x nfreq
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
        # x is +inf); zero those, mirroring the ext=1 band-stop for Z.
        spline = scipy.interpolate.CubicSpline(np.log10(self.periods),
                                               self.tau_master[lines],
                                               axis=-1, extrapolate=False)
        return np.nan_to_num(spline(x), copy=False)

    def calc_voltages(self, mag_x, mag_y, dt=60, line_chunk=256):
        """Line voltages directly from the magnetic field time series.

        Mirrors ``Site.convolve_fft`` (same padding and rfft conventions),
        but the frequency-domain multiply is by the per-line telluric
        response instead of the per-site impedance, so no electric field is
        ever formed.

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
        freqs = np.fft.rfftfreq(N, d=dt)

        mag_x_fft = np.fft.rfft(mag_x, n=N)
        mag_y_fft = np.fft.rfft(mag_y, n=N)

        voltages = np.empty((N0, self.nlines))
        for i0 in range(0, self.nlines, line_chunk):
            sel = slice(i0, min(i0 + line_chunk, self.nlines))
            tau = self.calc_tau(freqs, lines=sel)
            V_fft = tau[:, 0, :]*mag_x_fft + tau[:, 1, :]*mag_y_fft
            voltages[:, sel] = np.fft.irfft(V_fft, axis=-1)[:, :N0].T

        if self.valid is not None:
            voltages[:, ~self.valid] = np.nan
        return voltages
